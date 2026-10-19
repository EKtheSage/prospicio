//! Generalized additive models: a GLM with penalized B-spline smooths
//! (P-splines), fitted by penalized IRLS with smoothing parameters chosen
//! by GCV, UBRE or REML.
//!
//! Each smooth replaces a numeric column of the design with a cubic
//! B-spline basis on equally spaced knots, with a second-order difference
//! penalty: mgcv's `s(x, bs = "ps")`. The basis is reparameterized so the
//! smooth sums to zero over the data (mgcv's centering constraint), which
//! keeps it identifiable next to the intercept. A cyclic smooth
//! ([`PSpline::cyclic`], mgcv's `bs = "cp"`) wraps the basis and the
//! penalty around a period, so its ends meet in value and slope. A
//! monotone smooth ([`PSpline::increasing`], [`PSpline::decreasing`],
//! scam's `bs = "mpi"` and `"mpd"`) is a sum of B-splines with
//! non-decreasing coefficients, written as cumulative sums of `exp(γ)` and
//! fitted by Gauss–Newton IRLS in `γ`.

use prospicio_core::{Error, Result};
use prospicio_math::linalg::{cholesky, cholesky_inverse, symmetric_eigenvalues};
use prospicio_math::optimize::{NelderMead, brent, minimize, nelder_mead};
use prospicio_math::special::ln_gamma;
use prospicio_math::spline::{
    bspline_basis, cyclic_bspline_basis, cyclic_difference_penalty, difference_penalty,
    pspline_knots,
};
use prospicio_models::{Design, Fitted, Model};
use prospicio_prob::{PredictiveDistribution, Provenance};

use crate::{Dispersion, Glm, Irls, dev, irls, simulate_responses, working_system};
use prospicio_models::Family;

/// What shape a [`PSpline`] takes.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SmoothKind {
    /// An ordinary P-spline, mgcv's `bs = "ps"`.
    #[default]
    Ordinary,
    /// A cyclic P-spline, mgcv's `bs = "cp"`: `n_basis` cyclic B-splines on
    /// equally spaced knots over the period, and a cyclic difference
    /// penalty, so the smooth and its derivatives match at the two ends
    /// (a month of the year, an hour of the day, an angle). The period is
    /// `period` when given (mgcv's `knots = list(x = c(lo, hi))`), else the
    /// data's range; values outside it wrap around.
    Cyclic {
        /// The period's ends `(lo, hi)`; `None` takes the data's range.
        period: Option<(f64, f64)>,
    },
    /// A monotone increasing P-spline, scam's `bs = "mpi"` (Pya and Wood
    /// 2015): cubic B-splines on equally spaced knots over the data's range
    /// whose coefficients rise, `β₁ + Σ_{i≤j} exp(γᵢ)`, with a first-order
    /// difference penalty on `γ` (which leaves a straight line
    /// unpenalized). The basis is centered by its column means over the
    /// training data, as scam does. Smoothing by GCV, UBRE or fixed
    /// parameters: scam has no REML, and neither do monotone smooths here.
    /// New data outside the training range takes the value at its end, so
    /// the smooth stays monotone everywhere.
    Increasing,
    /// A monotone decreasing P-spline, scam's `bs = "mpd"`: the
    /// [`Increasing`](Self::Increasing) basis negated.
    Decreasing,
}

impl SmoothKind {
    fn is_monotone(&self) -> bool {
        matches!(self, Self::Increasing | Self::Decreasing)
    }
}

/// A P-spline smooth of one numeric design column.
#[derive(Debug, Clone, PartialEq)]
pub struct PSpline {
    /// The design column it replaces.
    pub column: String,
    /// Number of B-spline basis functions before the centering constraint
    /// (mgcv's `k`); the smooth has `n_basis - 1` coefficients.
    pub n_basis: usize,
    /// B-spline degree (3: cubic).
    pub degree: usize,
    /// Order of the difference penalty (2 penalizes curvature, leaving a
    /// straight line unpenalized; cyclic, only a constant).
    pub order: usize,
    /// Ordinary or cyclic.
    pub kind: SmoothKind,
}

impl PSpline {
    /// A cubic P-spline with 10 basis functions and a second-order
    /// penalty, mgcv's `s(column, bs = "ps")` default.
    pub fn new(column: &str) -> Self {
        Self {
            column: column.into(),
            n_basis: 10,
            degree: 3,
            order: 2,
            kind: SmoothKind::Ordinary,
        }
    }

    /// A cyclic cubic P-spline with 10 basis functions and a second-order
    /// cyclic penalty over the data's range, mgcv's
    /// `s(column, bs = "cp")` default. Set the period with
    /// [`period`](Self::period).
    ///
    /// ```
    /// use prospicio_glm::{Glm, gam::{Gam, PSpline}};
    /// use prospicio_models::{Design, Family, Fitted, Link, Model};
    ///
    /// // A seasonal effect over the months 0 to 12.
    /// let t: Vec<f64> = (0..240).map(|i| (f64::from(i) + 0.5) / 20.0).collect();
    /// let y: Vec<f64> = t.iter().map(|v| (std::f64::consts::TAU * v / 12.0).sin()).collect();
    /// let design = Design::new(
    ///     vec!["(Intercept)".into(), "month".into()],
    ///     vec![vec![1.0; 240], t.clone()],
    /// )
    /// .unwrap();
    /// let gam = Gam::new(
    ///     Glm::new(Family::Gaussian, Link::Identity),
    ///     vec![PSpline::cyclic("month").period(0.0, 12.0)],
    /// );
    /// let fit = gam.fit(&design, &y).unwrap();
    /// // The ends of the year meet.
    /// let ends = Design::new(
    ///     vec!["(Intercept)".into(), "month".into()],
    ///     vec![vec![1.0; 2], vec![0.0, 12.0]],
    /// )
    /// .unwrap();
    /// let p = fit.predict(&ends).unwrap();
    /// assert!((p[0] - p[1]).abs() < 1e-12);
    /// ```
    pub fn cyclic(column: &str) -> Self {
        Self {
            kind: SmoothKind::Cyclic { period: None },
            ..Self::new(column)
        }
    }

    /// A monotone increasing cubic P-spline with 10 basis functions,
    /// scam's `s(column, bs = "mpi")` default ([`SmoothKind::Increasing`]).
    ///
    /// ```
    /// use prospicio_glm::{Glm, gam::{Gam, PSpline}};
    /// use prospicio_models::{Design, Family, Fitted, Link, Model};
    ///
    /// // A rising curve with a dip in the noise: the fit never falls.
    /// let x: Vec<f64> = (0..200).map(|i| f64::from(i) / 199.0).collect();
    /// let y: Vec<f64> = x
    ///     .iter()
    ///     .enumerate()
    ///     .map(|(i, v)| v * v + 0.05 * (f64::from(i as u32) * 1.7).sin())
    ///     .collect();
    /// let design = Design::new(
    ///     vec!["(Intercept)".into(), "x".into()],
    ///     vec![vec![1.0; 200], x],
    /// )
    /// .unwrap();
    /// let fit = Gam::new(Glm::new(Family::Gaussian, Link::Identity), vec![PSpline::increasing("x")])
    ///     .fit(&design, &y)
    ///     .unwrap();
    /// let f = fit.fitted();
    /// assert!(f.windows(2).all(|w| w[1] >= w[0] - 1e-12));
    /// ```
    pub fn increasing(column: &str) -> Self {
        Self {
            order: 1,
            kind: SmoothKind::Increasing,
            ..Self::new(column)
        }
    }

    /// A monotone decreasing cubic P-spline with 10 basis functions,
    /// scam's `s(column, bs = "mpd")` ([`SmoothKind::Decreasing`]).
    pub fn decreasing(column: &str) -> Self {
        Self {
            kind: SmoothKind::Decreasing,
            ..Self::increasing(column)
        }
    }

    /// The same cyclic smooth with the period `[lo, hi]`, which must cover
    /// the data when it is fitted. Makes an ordinary smooth cyclic.
    pub fn period(mut self, lo: f64, hi: f64) -> Self {
        self.kind = SmoothKind::Cyclic {
            period: Some((lo, hi)),
        };
        self
    }

    /// The same smooth with `k` basis functions.
    pub fn n_basis(mut self, k: usize) -> Self {
        self.n_basis = k;
        self
    }
}

/// A tensor product smooth of several numeric design columns: mgcv's
/// `te(x, z, bs = "ps", np = FALSE)`. Each margin is a cubic B-spline basis
/// with a difference penalty, the basis is their row-wise Kronecker
/// product, and margin `i` contributes the penalty `I ⊗ … ⊗ P_i ⊗ … ⊗ I`
/// with its own smoothing parameter, so the surface can be smoother in one
/// direction than another. The whole product is centered to sum to zero
/// over the data, as a [`PSpline`].
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSpline {
    /// The design columns it replaces, one per margin.
    pub columns: Vec<String>,
    /// Basis functions per margin (mgcv's `k`, 5 by default).
    pub n_basis: Vec<usize>,
    /// B-spline degree of every margin (3: cubic).
    pub degree: usize,
    /// Order of every margin's difference penalty.
    pub order: usize,
}

impl TensorSpline {
    /// A tensor smooth of `columns` with 5 cubic basis functions and a
    /// second-order penalty per margin, mgcv's `te()` default size.
    pub fn new(columns: &[&str]) -> Self {
        Self {
            columns: columns.iter().map(|c| (*c).to_string()).collect(),
            n_basis: vec![5; columns.len()],
            degree: 3,
            order: 2,
        }
    }

    /// The same smooth with `k[i]` basis functions in margin `i`.
    pub fn n_basis(mut self, k: Vec<usize>) -> Self {
        self.n_basis = k;
        self
    }

    /// The smooth's label, `te(x,z)`.
    pub fn label(&self) -> String {
        format!("te({})", self.columns.join(","))
    }
}

/// How smoothing parameters are chosen.
#[derive(Debug, Clone, PartialEq)]
pub enum Smoothing {
    /// UBRE (Mallows' Cp) when the dispersion is fixed, GCV when it is
    /// estimated: mgcv's `method = "GCV.Cp"`.
    Auto,
    /// Generalized cross-validation, `n D / (n - edf)²`.
    Gcv,
    /// Un-biased risk estimator, `D / n + 2 φ edf / n - φ`.
    Ubre,
    /// Restricted maximum likelihood, mgcv's `method = "REML"`: the Laplace
    /// approximation to the restricted likelihood (Wood 2011),
    /// `D_p/(2φ) − l_s(φ) + ½ ln|XᵀWX + S| − ½ ln|S|₊ − (M_p/2) ln(2πφ)`,
    /// minimized over the smoothing parameters, with `W` the observed
    /// (Newton) weights, `D_p` the penalized deviance, `l_s` the saturated
    /// log-likelihood and `M_p` the dimension of the penalty's null space.
    /// An unknown dispersion is estimated with them
    /// ([`GamFit::reml_scale`]). For the Gaussian, Poisson, gamma, inverse
    /// Gaussian and binomial families.
    Reml,
    /// Fixed smoothing parameters, one per smooth.
    Fixed(Vec<f64>),
}

/// A GAM specification: a [`Glm`] (family, link, dispersion) plus smooths.
///
/// ```
/// use prospicio_glm::{Glm, gam::{Gam, PSpline}};
/// use prospicio_models::{Design, Family, Fitted, Link, Model};
///
/// // A sine curve plus noise, smoothed.
/// let x: Vec<f64> = (0..200).map(|i| f64::from(i) / 199.0).collect();
/// let y: Vec<f64> = x
///     .iter()
///     .enumerate()
///     .map(|(i, v)| (6.0 * v).sin() + 0.1 * ((i * 7919 % 13) as f64 / 6.0 - 1.0))
///     .collect();
/// let design = Design::new(
///     vec!["(Intercept)".into(), "x".into()],
///     vec![vec![1.0; 200], x.clone()],
/// )
/// .unwrap();
/// let gam = Gam::new(Glm::new(Family::Gaussian, Link::Identity), vec![PSpline::new("x")]);
/// let fit = gam.fit(&design, &y).unwrap();
/// let pred = fit.predict(&design).unwrap();
/// assert!((pred[100] - (6.0 * x[100]).sin()).abs() < 0.05);
/// assert!(fit.edf() > 4.0 && fit.edf() < 10.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Gam {
    pub glm: Glm,
    pub smooths: Vec<PSpline>,
    /// Tensor product smooths, after the P-splines; each has one smoothing
    /// parameter per margin.
    pub tensors: Vec<TensorSpline>,
    pub smoothing: Smoothing,
}

impl Gam {
    /// A GAM with smoothing parameters chosen by [`Smoothing::Auto`].
    pub fn new(glm: Glm, smooths: Vec<PSpline>) -> Self {
        Self {
            glm,
            smooths,
            tensors: Vec::new(),
            smoothing: Smoothing::Auto,
        }
    }

    /// The same GAM with a tensor product smooth added. Smoothing
    /// parameters run over the P-splines first, then each tensor's margins.
    pub fn tensor(mut self, tensor: TensorSpline) -> Self {
        self.tensors.push(tensor);
        self
    }

    /// The same GAM with smoothing parameters chosen by `smoothing`.
    pub fn smoothing(mut self, smoothing: Smoothing) -> Self {
        self.smoothing = smoothing;
        self
    }
}

/// A smooth's basis as fitted: knots from the training data (a cyclic
/// smooth's: the period's two ends) and the constraint's null space `Z`
/// (`k × (k - 1)`, row-major); for a monotone smooth, `z` holds the
/// `k - 1` column means its centering subtracts.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Basis {
    pub(crate) spline: PSpline,
    pub(crate) knots: Vec<f64>,
    pub(crate) z: Vec<f64>,
}

impl Basis {
    /// Learns knots from `x` and the centering constraint from its basis.
    fn fit(spline: &PSpline, x: &[f64]) -> Result<Self> {
        let k = spline.n_basis;
        if k < spline.degree + 2 || spline.order >= k {
            return Err(Error::InvalidParameter {
                name: "n_basis",
                value: k as f64,
                reason: "is too small for the degree and penalty order",
            });
        }
        let (lo, hi) = x
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        if hi <= lo || hi.is_nan() || lo.is_nan() {
            return Err(Error::Data(format!(
                "smooth column {:?} has no spread",
                spline.column
            )));
        }
        let knots = match spline.kind {
            SmoothKind::Ordinary => pspline_knots(lo, hi, k, spline.degree),
            // scam's knots: the data's range in k - degree equal intervals,
            // not widened, extended degree intervals beyond each end.
            SmoothKind::Increasing | SmoothKind::Decreasing => {
                let h = (hi - lo) / (k - spline.degree) as f64;
                (0..=k + spline.degree)
                    .map(|i| lo + h * (i as f64 - spline.degree as f64))
                    .collect()
            }
            SmoothKind::Cyclic { period } => {
                // mgcv's "cp": the period's ends; k + 1 knots between them.
                let (a, b) = period.unwrap_or((lo, hi));
                if !(a.is_finite() && b.is_finite() && a < b) {
                    return Err(Error::InvalidParameter {
                        name: "period",
                        value: a,
                        reason: "must be finite with lo < hi",
                    });
                }
                if lo < a || hi > b {
                    return Err(Error::Data(format!(
                        "the period of smooth {:?} does not cover its data",
                        spline.column
                    )));
                }
                vec![a, b]
            }
        };
        let mut basis = Self {
            spline: spline.clone(),
            knots,
            z: Vec::new(),
        };
        if spline.kind.is_monotone() {
            // The column means of the cumulative basis, which centering
            // subtracts (scam's cmX).
            let mut means = vec![0.0; k - 1];
            for &v in x {
                for (m, c) in means.iter_mut().zip(basis.cumulative(v)) {
                    *m += c;
                }
            }
            basis.z = means.into_iter().map(|m| m / x.len() as f64).collect();
            return Ok(basis);
        }
        // c = column sums of the basis; Z spans c⊥ (Householder).
        let mut c = vec![0.0; k];
        for &v in x {
            for (cj, b) in c.iter_mut().zip(basis.raw(v)) {
                *cj += b;
            }
        }
        basis.z = centering(c);
        Ok(basis)
    }

    /// A monotone smooth's uncentered columns at `x`: `Σ_{i≥j} Bᵢ(x)` for
    /// `j = 1..k` (the first, a constant, dropped), negated if decreasing.
    /// `x` is clamped to the training range first: past it the B-splines
    /// no longer sum to 1 and the smooth could turn back, so it holds
    /// its value at the end instead (scam extrapolates the splines).
    fn cumulative(&self, x: f64) -> Vec<f64> {
        let (lo, hi) = (
            self.knots[self.spline.degree],
            self.knots[self.spline.n_basis],
        );
        let b = self.raw(x.clamp(lo, hi));
        let sign = if self.spline.kind == SmoothKind::Decreasing {
            -1.0
        } else {
            1.0
        };
        let mut out = vec![0.0; b.len() - 1];
        let mut tail = 0.0;
        for j in (1..b.len()).rev() {
            tail += b[j];
            out[j - 1] = sign * tail;
        }
        out
    }

    /// The unconstrained basis at `x`: `k` values.
    fn raw(&self, x: f64) -> Vec<f64> {
        let (k, degree) = (self.spline.n_basis, self.spline.degree);
        match self.spline.kind {
            SmoothKind::Ordinary | SmoothKind::Increasing | SmoothKind::Decreasing => {
                bspline_basis(x, &self.knots, degree)
            }
            SmoothKind::Cyclic { .. } => {
                cyclic_bspline_basis(x, self.knots[0], self.knots[1], k, degree)
            }
        }
    }

    fn width(&self) -> usize {
        self.spline.n_basis - 1
    }

    /// The constrained basis columns `B Z` at `x`.
    fn columns(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let (k, m) = (self.spline.n_basis, self.width());
        let mut cols = vec![vec![0.0; x.len()]; m];
        if self.spline.kind.is_monotone() {
            for (r, &v) in x.iter().enumerate() {
                for (j, c) in self.cumulative(v).into_iter().enumerate() {
                    cols[j][r] = c - self.z[j];
                }
            }
            return cols;
        }
        for (r, &v) in x.iter().enumerate() {
            let b = self.raw(v);
            for (j, col) in cols.iter_mut().enumerate() {
                col[r] = (0..k).map(|i| b[i] * self.z[i * m + j]).sum();
            }
        }
        cols
    }

    /// `Zᵀ P Z`, `(k - 1) × (k - 1)`.
    fn penalty(&self) -> Vec<f64> {
        let (k, m) = (self.spline.n_basis, self.width());
        let p = match self.spline.kind {
            SmoothKind::Ordinary => difference_penalty(k, self.spline.order),
            SmoothKind::Cyclic { .. } => cyclic_difference_penalty(k, self.spline.order),
            // On γ directly: no constraint to project through.
            SmoothKind::Increasing | SmoothKind::Decreasing => {
                return difference_penalty(m, self.spline.order);
            }
        };
        let mut out = vec![0.0; m * m];
        for a in 0..m {
            for b in 0..m {
                let mut s = 0.0;
                for i in 0..k {
                    for j in 0..k {
                        s += self.z[i * m + a] * p[i * k + j] * self.z[j * m + b];
                    }
                }
                out[a * m + b] = s;
            }
        }
        out
    }
}

/// `Z` (`k × (k − 1)`, row-major) spanning the complement of the column
/// sums `c`, by a Householder reflection: the sum-to-zero constraint.
fn centering(c: Vec<f64>) -> Vec<f64> {
    let k = c.len();
    let norm = c.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut v = c;
    v[0] += norm.copysign(v[0]);
    let vv: f64 = v.iter().map(|a| a * a).sum();
    let mut z = vec![0.0; k * (k - 1)];
    for i in 0..k {
        for j in 1..k {
            let h = f64::from(u8::from(i == j)) - 2.0 * v[i] * v[j] / vv;
            z[i * (k - 1) + j - 1] = h;
        }
    }
    z
}

/// `Zᵀ P Z` for a `k × k` penalty `P` and `Z` `k × m`.
fn project(p: &[f64], z: &[f64], k: usize, m: usize) -> Vec<f64> {
    // P Z first, then Zᵀ (P Z).
    let mut pz = vec![0.0; k * m];
    for i in 0..k {
        for j in 0..k {
            let pij = p[i * k + j];
            if pij != 0.0 {
                for b in 0..m {
                    pz[i * m + b] += pij * z[j * m + b];
                }
            }
        }
    }
    let mut out = vec![0.0; m * m];
    for i in 0..k {
        for a in 0..m {
            let zia = z[i * m + a];
            if zia != 0.0 {
                for b in 0..m {
                    out[a * m + b] += zia * pz[i * m + b];
                }
            }
        }
    }
    out
}

/// A tensor smooth's basis as fitted: knots per margin and the centering
/// constraint's `Z` (`K × (K − 1)` for `K = Π k_i`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TensorBasis {
    pub(crate) tensor: TensorSpline,
    pub(crate) knots: Vec<Vec<f64>>,
    pub(crate) z: Vec<f64>,
}

impl TensorBasis {
    fn fit(tensor: &TensorSpline, cols: &[&[f64]]) -> Result<Self> {
        if tensor.columns.len() < 2 || tensor.n_basis.len() != tensor.columns.len() {
            return Err(Error::InvalidParameter {
                name: "n_basis",
                value: tensor.n_basis.len() as f64,
                reason: "a tensor smooth needs two or more columns, one n_basis each",
            });
        }
        let mut knots = Vec::new();
        for ((col, &k), name) in cols.iter().zip(&tensor.n_basis).zip(&tensor.columns) {
            let spline = PSpline {
                column: name.clone(),
                n_basis: k,
                degree: tensor.degree,
                order: tensor.order,
                kind: SmoothKind::Ordinary,
            };
            // The margin's checks and knots, without its own constraint.
            knots.push(Basis::fit(&spline, col)?.knots);
        }
        let mut b = Self {
            tensor: tensor.clone(),
            knots,
            z: Vec::new(),
        };
        let raw = b.raw(cols);
        let c: Vec<f64> = raw.iter().map(|col| col.iter().sum()).collect();
        b.z = centering(c);
        Ok(b)
    }

    fn size(&self) -> usize {
        self.tensor.n_basis.iter().product()
    }

    fn width(&self) -> usize {
        self.size() - 1
    }

    /// The unconstrained product basis, one column per coefficient (the
    /// last margin's index fastest).
    fn raw(&self, cols: &[&[f64]]) -> Vec<Vec<f64>> {
        let n = cols[0].len();
        let size = self.size();
        let mut out = vec![vec![0.0; n]; size];
        for r in 0..n {
            let mut row = vec![1.0];
            for (m, col) in cols.iter().enumerate() {
                let b = bspline_basis(col[r], &self.knots[m], self.tensor.degree);
                row = row
                    .iter()
                    .flat_map(|&a| b.iter().map(move |&v| a * v))
                    .collect();
            }
            for (j, v) in row.into_iter().enumerate() {
                out[j][r] = v;
            }
        }
        out
    }

    /// The constrained basis columns.
    fn columns(&self, cols: &[&[f64]]) -> Vec<Vec<f64>> {
        let raw = self.raw(cols);
        let m = self.width();
        let n = cols[0].len();
        let mut out = vec![vec![0.0; n]; m];
        for (i, col) in raw.iter().enumerate() {
            for (j, o) in out.iter_mut().enumerate() {
                let zij = self.z[i * m + j];
                if zij != 0.0 {
                    for r in 0..n {
                        o[r] += col[r] * zij;
                    }
                }
            }
        }
        out
    }

    /// Margin `i`'s penalty `I ⊗ P_i ⊗ I`, projected by `Z`.
    fn penalties(&self) -> Vec<Vec<f64>> {
        let ks = &self.tensor.n_basis;
        let (size, m) = (self.size(), self.width());
        (0..ks.len())
            .map(|i| {
                let p = difference_penalty(ks[i], self.tensor.order);
                let before: usize = ks[..i].iter().product();
                let after: usize = ks[i + 1..].iter().product();
                let mut full = vec![0.0; size * size];
                for a in 0..before {
                    for c in 0..after {
                        for u in 0..ks[i] {
                            for v in 0..ks[i] {
                                let row = (a * ks[i] + u) * after + c;
                                let col = (a * ks[i] + v) * after + c;
                                full[row * size + col] = p[u * ks[i] + v];
                            }
                        }
                    }
                }
                project(&full, &self.z, size, m)
            })
            .collect()
    }
}

/// The design columns a smooth reads.
fn column_of<'a>(design: &'a Design, name: &str) -> Result<&'a [f64]> {
    let j = design
        .names()
        .iter()
        .position(|n| n == name)
        .ok_or_else(|| Error::Data(format!("no design column {name:?}")))?;
    Ok(design.column(j))
}

/// Replaces each smooth's raw column of `design` by its basis columns,
/// appended after the parametric columns: the P-splines, then the tensors.
fn expand(design: &Design, bases: &[Basis], tensors: &[TensorBasis]) -> Result<Design> {
    let mut raw: Vec<&str> = bases.iter().map(|b| b.spline.column.as_str()).collect();
    for t in tensors {
        raw.extend(t.tensor.columns.iter().map(String::as_str));
    }
    let mut names = Vec::new();
    let mut cols = Vec::new();
    for (j, name) in design.names().iter().enumerate() {
        if !raw.contains(&name.as_str()) {
            names.push(name.clone());
            cols.push(design.column(j).to_vec());
        }
    }
    for b in bases {
        let j = design
            .names()
            .iter()
            .position(|n| *n == b.spline.column)
            .ok_or_else(|| Error::Data(format!("no design column {:?}", b.spline.column)))?;
        for (i, col) in b.columns(design.column(j)).into_iter().enumerate() {
            names.push(format!("s({}).{}", b.spline.column, i + 1));
            cols.push(col);
        }
    }
    for t in tensors {
        let inputs = t
            .tensor
            .columns
            .iter()
            .map(|c| column_of(design, c))
            .collect::<Result<Vec<_>>>()?;
        let label = t.tensor.label();
        for (i, col) in t.columns(&inputs).into_iter().enumerate() {
            names.push(format!("{label}.{}", i + 1));
            cols.push(col);
        }
    }
    Design::new(names, cols)?
        .with_offset(design.offset().to_vec())?
        .with_weights(design.weights().to_vec())
}

/// A fitted GAM.
#[derive(Debug, Clone, PartialEq)]
pub struct GamFit {
    pub(crate) spec: Gam,
    pub(crate) bases: Vec<Basis>,
    pub(crate) tensors: Vec<TensorBasis>,
    pub(crate) names: Vec<String>,
    pub(crate) coefficients: Vec<f64>,
    pub(crate) lambdas: Vec<f64>,
    pub(crate) edf: f64,
    pub(crate) covariance: Vec<f64>,
    pub(crate) dispersion: f64,
    pub(crate) deviance: f64,
    pub(crate) log_likelihood: f64,
    pub(crate) score: f64,
    pub(crate) reml_scale: Option<f64>,
    pub(crate) fitted: Vec<f64>,
    pub(crate) input_hash: String,
}

/// The best of 64 equal cells of `(lo, hi)`, refined by Brent's method to
/// `1e-4`: a starting point for a joint search.
fn coarse_minimize(lo: f64, hi: f64, mut f: impl FnMut(f64) -> f64) -> f64 {
    const CELLS: usize = 64;
    let h = (hi - lo) / CELLS as f64;
    let x = |j: usize| lo + h * j as f64;
    let (mut best, mut f_best) = (1, f(x(1)));
    for j in 2..CELLS {
        let v = f(x(j));
        if v < f_best {
            (best, f_best) = (j, v);
        }
    }
    brent(x(best - 1), x(best + 1), 1e-4, f).0
}

/// The saturated log-likelihood `l_s(φ)` of mgcv's `fix.family.ls`, the
/// part of the log-likelihood that the deviance leaves out, for REML.
fn saturated_log_likelihood(family: Family, y: &[f64], w: &[f64], phi: f64) -> Result<f64> {
    let rows = y.iter().zip(w).filter(|(_, w)| **w > 0.0);
    Ok(match family {
        Family::Gaussian => rows
            .map(|(_, &w)| -0.5 * (2.0 * std::f64::consts::PI * phi).ln() + 0.5 * w.ln())
            .sum(),
        Family::InverseGaussian => rows
            .map(|(&y, &w)| {
                -0.5 * (2.0 * std::f64::consts::PI * phi * y.powi(3)).ln() + 0.5 * w.ln()
            })
            .sum(),
        Family::Poisson => rows
            .map(|(&y, &w)| {
                let ylny = if y > 0.0 { y * y.ln() } else { 0.0 };
                w * (ylny - y - ln_gamma(y + 1.0))
            })
            .sum(),
        Family::Gamma => rows
            .map(|(&y, &w)| {
                let s = phi / w;
                -ln_gamma(1.0 / s) - s.ln() / s - 1.0 / s - y.ln()
            })
            .sum(),
        Family::Binomial => rows
            .map(|(&y, &m)| {
                // ln dbinom(round(m y), round(m), y): m trials, y the share.
                let (k, n) = ((m * y).round(), m.round());
                let lc = ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0);
                let a = if k > 0.0 { k * y.ln() } else { 0.0 };
                let b = if n - k > 0.0 {
                    (n - k) * (-y).ln_1p()
                } else {
                    0.0
                };
                lc + a + b
            })
            .sum(),
        other => {
            return Err(Error::Data(format!(
                "REML smoothing is not available for the {} family",
                other.name()
            )));
        }
    })
}

/// `Xᵀ W X` with the observed (Newton) weights at the fitted means: the
/// expected weights less `(y − μ)` times the derivative of `μ'/V` in `η`.
/// They are the expected ones for a canonical link.
fn observed_information(glm: &Glm, x: &Design, y: &[f64], mu: &[f64]) -> Vec<f64> {
    let (family, link) = (glm.family, glm.link);
    let (n, p) = (x.n_rows(), x.n_cols());
    let w = x.weights();
    let mut info = vec![0.0; p * p];
    for i in 0..n {
        let eta = link.link(mu[i]);
        let (d1, d2) = (link.mu_eta(eta), link.mu_eta2(eta));
        let (v, dv) = (family.variance(mu[i]), family.variance_deriv(mu[i]));
        let r = y[i] - mu[i];
        let wi = w[i] * (d1 * d1 / v - r * (d2 / v - d1 * d1 * dv / (v * v)));
        for a in 0..p {
            let xa = x.column(a)[i] * wi;
            for b in 0..=a {
                info[a * p + b] += xa * x.column(b)[i];
            }
        }
    }
    for a in 0..p {
        for b in a + 1..p {
            info[a * p + b] = info[b * p + a];
        }
    }
    info
}

/// One penalized fit at fixed smoothing parameters.
struct Trial {
    /// `beta` holds `θ`: `γ` (log scale) in a monotone smooth's columns.
    irls: Irls,
    penalty: Vec<f64>,
    edf: f64,
    deviance: f64,
}

/// Penalized IRLS where the columns `exp_cols` enter as `exp(θ)`:
/// `η = X β(θ)` with `βⱼ = exp(θⱼ)` there and `βⱼ = θⱼ` elsewhere, the
/// deviance plus `θᵀ S θ` minimized over `θ` by Gauss–Newton (Fisher
/// scoring on the Jacobian `J = X diag(∂β/∂θ)`), with step halving, from
/// `theta0`. Its fixed point solves the penalized score equations, so it
/// reaches scam's estimate at the same smoothing parameters. Returns the
/// fit (with `beta` = `θ` and `information` = `JᵀWJ`) and the Hessian of
/// half the objective without the penalty, `JᵀWJ − E`, where `E` is
/// diagonal on `exp_cols`: `Σᵢ wᵢ (yᵢ − μᵢ) μ'ᵢ/Vᵢ Xᵢⱼ exp(θⱼ)`.
fn irls_exp(
    glm: &Glm,
    x: &Design,
    y: &[f64],
    theta0: &[f64],
    s: &[f64],
    exp_cols: &[usize],
) -> Result<(Irls, Vec<f64>)> {
    let (family, link) = (glm.family, glm.link);
    let (n, p) = (x.n_rows(), x.n_cols());
    let w = x.weights();
    let beta_of = |theta: &[f64]| -> Vec<f64> {
        let mut b = theta.to_vec();
        for &j in exp_cols {
            b[j] = theta[j].exp();
        }
        b
    };
    let jacobian = |theta: &[f64]| -> Result<Design> {
        let cols = (0..p)
            .map(|j| {
                let c = x.column(j);
                if exp_cols.contains(&j) {
                    let e = theta[j].exp();
                    c.iter().map(|v| v * e).collect()
                } else {
                    c.to_vec()
                }
            })
            .collect();
        Design::new(x.names().to_vec(), cols)?
            .with_offset(x.offset().to_vec())?
            .with_weights(w.to_vec())
    };
    let state = |theta: &[f64]| -> Option<(Vec<f64>, Vec<f64>, f64)> {
        let eta = x.linear_predictor(&beta_of(theta));
        if !eta.iter().all(|&e| link.valid_eta(e)) {
            return None;
        }
        let mu: Vec<f64> = eta.iter().map(|&e| link.inverse(e)).collect();
        if !mu.iter().all(|&m| family.valid_mu(m)) {
            return None;
        }
        let obj = dev(family, y, &mu, w) + crate::quadratic(s, theta);
        obj.is_finite().then_some((eta, mu, obj))
    };
    let mut theta = theta0.to_vec();
    let (mut eta, mut mu, mut current) = state(&theta)
        .ok_or_else(|| Error::Data("the starting values give invalid means".into()))?;
    for _ in 1..=glm.max_iterations {
        let j = jacobian(&theta)?;
        let (mut a, mut rhs) = working_system(glm, &j, y, &eta, &mu)?;
        // The working response of the linearized model, z + J (θ − 1) on
        // exp_cols: Jⱼ θⱼ replaces Xⱼ exp(θⱼ) = Jⱼ.
        let mut v = vec![0.0; p];
        for &c in exp_cols {
            v[c] = theta[c] - 1.0;
        }
        for r in 0..p {
            rhs[r] += (0..p).map(|c| a[r * p + c] * v[c]).sum::<f64>();
        }
        for (ai, si) in a.iter_mut().zip(s) {
            *ai += si;
        }
        let l = cholesky(&a, p).ok_or_else(|| {
            Error::Data("the design matrix is collinear: drop or merge columns".into())
        })?;
        let target = prospicio_math::linalg::cholesky_solve(&l, &rhs);
        let previous = theta.clone();
        // The linearized exp(γ) is poor far from γ: start from a step that
        // moves no γ by more than 2 (a factor e² in its increment).
        let reach = exp_cols
            .iter()
            .map(|&c| (target[c] - theta[c]).abs())
            .fold(0.0, f64::max);
        let mut step = if reach > 2.0 { 2.0 / reach } else { 1.0 };
        loop {
            let trial: Vec<f64> = previous
                .iter()
                .zip(&target)
                .map(|(b, t)| b + step * (t - b))
                .collect();
            if let Some((e, m, obj)) = state(&trial)
                && obj <= current * (1.0 + 1e-12)
            {
                theta = trial;
                (eta, mu) = (e, m);
                break;
            }
            step *= 0.5;
            if step < 1e-10 {
                // No step improves: converged as far as rounding allows.
                break;
            }
        }
        let new_obj = state(&theta).map_or(f64::INFINITY, |t| t.2);
        let settled = (current - new_obj).abs() / (new_obj.abs() + 0.1) < glm.tolerance;
        // Judged on β: an increment that wants to be 0 sends its γ towards
        // −∞ in steps that never shrink, while exp(γ) settles.
        let (now, before) = (beta_of(&theta), beta_of(&previous));
        let still = now
            .iter()
            .zip(&before)
            .all(|(b, q)| (b - q).abs() <= glm.tolerance.sqrt() * 1e-3 * (b.abs() + 1e-3));
        current = new_obj;
        if (settled && still) || step < 1e-10 {
            let j = jacobian(&theta)?;
            let (information, _) = working_system(glm, &j, y, &eta, &mu)?;
            let mut hessian = information.clone();
            for &c in exp_cols {
                let col = x.column(c);
                let e = theta[c].exp();
                let extra: f64 = (0..n)
                    .map(|i| {
                        let d = link.mu_eta(eta[i]);
                        w[i] * (y[i] - mu[i]) * d / family.variance(mu[i]) * col[i] * e
                    })
                    .sum();
                hessian[c * p + c] -= extra;
            }
            return Ok((
                Irls {
                    beta: theta,
                    information,
                    iterations: 0,
                    mu,
                },
                hessian,
            ));
        }
    }
    Err(Error::Data(format!(
        "IRLS did not converge in {} iterations",
        glm.max_iterations
    )))
}

impl Model for Gam {
    type Fitted = GamFit;

    fn fit(&self, design: &Design, y: &[f64]) -> Result<GamFit> {
        let glm = self.glm;
        glm.family.validate()?;
        if self.smooths.is_empty() && self.tensors.is_empty() {
            return Err(Error::InvalidParameter {
                name: "smooths",
                value: 0.0,
                reason: "must not be empty (fit a Glm instead)",
            });
        }
        let bases = self
            .smooths
            .iter()
            .map(|s| {
                let j = design
                    .names()
                    .iter()
                    .position(|n| *n == s.column)
                    .ok_or_else(|| Error::Data(format!("no design column {:?}", s.column)))?;
                Basis::fit(s, design.column(j))
            })
            .collect::<Result<Vec<_>>>()?;
        let tensors = self
            .tensors
            .iter()
            .map(|t| {
                let cols = t
                    .columns
                    .iter()
                    .map(|c| column_of(design, c))
                    .collect::<Result<Vec<_>>>()?;
                TensorBasis::fit(t, &cols)
            })
            .collect::<Result<Vec<_>>>()?;
        let x = expand(design, &bases, &tensors)?;
        let (n, p) = (x.n_rows(), x.n_cols());
        if y.len() != n {
            return Err(Error::Data(format!(
                "{} responses for {n} design rows",
                y.len()
            )));
        }
        if let Some(bad) = y.iter().find(|&&v| !glm.family.valid_y(v)) {
            return Err(Error::InvalidParameter {
                name: "y",
                value: *bad,
                reason: "is outside the family's range",
            });
        }
        // Each smooth's penalty block sits at the end, in order: its start,
        // width and penalties (one per smoothing parameter).
        let blocks: Vec<(usize, usize, Vec<Vec<f64>>)> = {
            let widths: usize = bases.iter().map(Basis::width).sum::<usize>()
                + tensors.iter().map(TensorBasis::width).sum::<usize>();
            let mut start = p - widths;
            let mut out = Vec::new();
            for b in &bases {
                out.push((start, b.width(), vec![b.penalty()]));
                start += b.width();
            }
            for t in &tensors {
                out.push((start, t.width(), t.penalties()));
                start += t.width();
            }
            out
        };
        let n_lambda: usize = blocks.iter().map(|b| b.2.len()).sum();
        // The columns of the monotone smooths, which enter as exp(γ).
        let exp_cols: Vec<usize> = bases
            .iter()
            .zip(&blocks)
            .filter(|(b, _)| b.spline.kind.is_monotone())
            .flat_map(|(_, (at, m, _))| *at..at + m)
            .collect();
        if !exp_cols.is_empty() && self.smoothing == Smoothing::Reml {
            return Err(Error::InvalidParameter {
                name: "smoothing",
                value: f64::NAN,
                reason: "REML is not available with monotone smooths: use GCV, UBRE or fixed \
                         smoothing parameters (as scam)",
            });
        }
        let w = x.weights();
        let y_mean = y.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / w.iter().sum::<f64>();
        // Each trial starts IRLS from the last trial's means (and, with
        // monotone smooths, its θ).
        let warm: std::cell::RefCell<Option<Vec<f64>>> = std::cell::RefCell::new(None);
        let warm_theta: std::cell::RefCell<Option<Vec<f64>>> = std::cell::RefCell::new(None);
        let trial = |lambdas: &[f64]| -> Result<Trial> {
            let mut s = vec![0.0; p * p];
            let mut lambda = lambdas.iter();
            for (at, m, penalties) in &blocks {
                for block in penalties {
                    let l = *lambda.next().expect("one lambda per penalty");
                    for a in 0..*m {
                        for b in 0..*m {
                            s[(at + a) * p + at + b] += l * block[a * m + b];
                        }
                    }
                }
            }
            if !exp_cols.is_empty() {
                // Start from the last θ, or from the link at the mean with
                // small equal increments.
                let theta0 = warm_theta.borrow().clone().unwrap_or_else(|| {
                    let mut t = vec![0.0; p];
                    if let Some(j) = x.names().iter().position(|n| n == "(Intercept)") {
                        t[j] = glm.link.link(y_mean);
                    }
                    for &c in &exp_cols {
                        t[c] = -2.0;
                    }
                    t
                });
                let (fit, hessian) = irls_exp(&glm, &x, y, &theta0, &s, &exp_cols)?;
                *warm_theta.borrow_mut() = Some(fit.beta.clone());
                // edf = tr((JᵀWJ − E + S)⁻¹ JᵀWJ), scam's influence with the
                // Newton Hessian.
                let mut a = hessian;
                for (v, si) in a.iter_mut().zip(&s) {
                    *v += si;
                }
                let l = cholesky(&a, p).ok_or_else(|| {
                    Error::Data("the penalized Hessian is not positive definite".into())
                })?;
                let inv = cholesky_inverse(&l, p);
                let edf: f64 = (0..p)
                    .map(|i| {
                        (0..p)
                            .map(|k| inv[i * p + k] * fit.information[k * p + i])
                            .sum::<f64>()
                    })
                    .sum();
                let deviance = dev(glm.family, y, &fit.mu, w);
                return Ok(Trial {
                    irls: fit,
                    penalty: s,
                    edf,
                    deviance,
                });
            }
            let start = warm.borrow().clone();
            let fit = irls(
                &glm,
                &x,
                y,
                |i| match &start {
                    Some(mu) => mu[i],
                    None => glm.family.initial_mu(y[i], w[i], y_mean),
                },
                Some(&s),
            )?;
            *warm.borrow_mut() = Some(fit.mu.clone());
            // edf = tr((XᵀWX + S)⁻¹ XᵀWX) = p - tr((XᵀWX + S)⁻¹ S).
            let mut a = fit.information.clone();
            for (v, si) in a.iter_mut().zip(&s) {
                *v += si;
            }
            let l = cholesky(&a, p)
                .ok_or_else(|| Error::Data("the penalized system is singular".into()))?;
            let inv = cholesky_inverse(&l, p);
            let trace: f64 = (0..p)
                .map(|i| (0..p).map(|k| inv[i * p + k] * s[k * p + i]).sum::<f64>())
                .sum();
            let deviance = dev(glm.family, y, &fit.mu, w);
            Ok(Trial {
                irls: fit,
                penalty: s,
                edf: p as f64 - trace,
                deviance,
            })
        };
        let fixed = match glm.dispersion {
            Dispersion::Fixed(phi) => Some(phi),
            _ if glm.family.unit_dispersion() && self.smoothing == Smoothing::Reml => Some(1.0),
            _ => None,
        };
        let use_ubre = match self.smoothing {
            Smoothing::Ubre => true,
            Smoothing::Gcv => false,
            _ => fixed.is_some(),
        };
        let use_reml = self.smoothing == Smoothing::Reml;
        // Each block's penalty rank (of the sum of its penalties) and, for a
        // single penalty, ln|P|₊: ln|λ P|₊ = r ln λ + ln|P|₊. A block with
        // several penalties (a tensor) takes ln|Σ λ_i P_i|₊ from the top r
        // eigenvalues at each trial.
        let ranks: Vec<(usize, f64)> = blocks
            .iter()
            .map(|(_, m, penalties)| {
                let mut sum = vec![0.0; m * m];
                for pen in penalties {
                    for (a, b) in sum.iter_mut().zip(pen) {
                        *a += b;
                    }
                }
                let e = symmetric_eigenvalues(&sum, *m);
                let top = e.iter().copied().fold(0.0, f64::max);
                let kept: Vec<f64> = e.into_iter().filter(|v| *v > 1e-10 * top).collect();
                (kept.len(), kept.iter().map(|v| v.ln()).sum())
            })
            .collect();
        let null_dim = p as f64 - ranks.iter().map(|r| r.0 as f64).sum::<f64>();
        let ln_pseudo_det = |lambdas: &[f64]| -> f64 {
            let mut lambda = lambdas.iter();
            blocks
                .iter()
                .zip(&ranks)
                .map(|((_, m, penalties), &(r, ln_p))| {
                    if penalties.len() == 1 {
                        let l = *lambda.next().expect("one lambda per penalty");
                        return r as f64 * l.ln() + ln_p;
                    }
                    let mut sum = vec![0.0; m * m];
                    for pen in penalties {
                        let l = *lambda.next().expect("one lambda per penalty");
                        for (a, b) in sum.iter_mut().zip(pen) {
                            *a += l * b;
                        }
                    }
                    let e = symmetric_eigenvalues(&sum, *m);
                    e[m - r..].iter().map(|v| v.ln()).sum()
                })
                .sum()
        };
        if use_reml {
            saturated_log_likelihood(glm.family, y, w, 1.0)?;
        }
        let nf = n as f64;
        let n_obs = w.iter().filter(|v| **v > 0.0).count() as f64;
        // The REML score and dispersion at a trial.
        let reml = |t: &Trial, lambdas: &[f64]| -> Result<(f64, f64)> {
            let dp = t.deviance + crate::quadratic(&t.penalty, &t.irls.beta);
            let mut a = observed_information(&glm, &x, y, &t.irls.mu);
            for (v, si) in a.iter_mut().zip(&t.penalty) {
                *v += si;
            }
            let l = cholesky(&a, p).ok_or_else(|| {
                Error::Data("the penalized observed information is not positive definite".into())
            })?;
            let ln_det: f64 = (0..p).map(|i| 2.0 * l[i * p + i].ln()).sum();
            let ln_s = ln_pseudo_det(lambdas);
            let v = |phi: f64| -> f64 {
                let ls = saturated_log_likelihood(glm.family, y, w, phi).unwrap_or(f64::NAN);
                dp / (2.0 * phi) - ls + 0.5 * ln_det
                    - 0.5 * ln_s
                    - 0.5 * null_dim * (2.0 * std::f64::consts::PI * phi).ln()
            };
            let phi = match (fixed, glm.family) {
                (Some(phi), _) => phi,
                (None, Family::Gaussian | Family::InverseGaussian) => dp / (n_obs - null_dim),
                (None, _) => {
                    let start = (dp / (n_obs - null_dim)).ln();
                    minimize(start - 8.0, start + 8.0, |lp| {
                        let s = v(lp.exp());
                        if s.is_finite() { s } else { f64::INFINITY }
                    })
                    .exp()
                }
            };
            Ok((v(phi), phi))
        };
        let score = |t: &Trial, lambdas: &[f64]| -> f64 {
            if use_reml {
                reml(t, lambdas).map_or(f64::INFINITY, |r| r.0)
            } else if use_ubre {
                let phi = fixed.unwrap_or(1.0);
                t.deviance / nf + 2.0 * phi * t.edf / nf - phi
            } else {
                nf * t.deviance / (nf - t.edf).powi(2)
            }
        };
        let lambdas = match &self.smoothing {
            Smoothing::Fixed(l) => {
                if l.len() != n_lambda || l.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
                    return Err(Error::InvalidParameter {
                        name: "smoothing",
                        value: l.len() as f64,
                        reason: "needs one finite, non-negative value per smoothing parameter \
                                 (one per P-spline, one per tensor margin)",
                    });
                }
                l.clone()
            }
            _ => {
                // One smoothing parameter: a scan of ln λ and golden section
                // to full precision. Several: one coarse coordinate pass (the
                // scan, then Brent to 1e-4), polished jointly by Nelder–Mead.
                let mut log_l = vec![0.0; n_lambda];
                for j in 0..n_lambda {
                    let objective = |v: f64| {
                        let mut l: Vec<f64> = log_l.iter().map(|a: &f64| a.exp()).collect();
                        l[j] = v.exp();
                        trial(&l).map_or(f64::INFINITY, |t| score(&t, &l))
                    };
                    log_l[j] = if n_lambda == 1 {
                        minimize(-15.0, 20.0, objective)
                    } else {
                        coarse_minimize(-15.0, 20.0, objective)
                    };
                }
                if n_lambda > 1 {
                    // Polish the coordinate search jointly.
                    let objective = |v: &[f64]| {
                        if v.iter().any(|a| !(-15.0..=20.0).contains(a)) {
                            return f64::INFINITY;
                        }
                        let l: Vec<f64> = v.iter().map(|a| a.exp()).collect();
                        trial(&l).map_or(f64::INFINITY, |t| score(&t, &l))
                    };
                    let options = NelderMead {
                        initial_step: 0.5,
                        ..NelderMead::default()
                    };
                    log_l = nelder_mead(objective, &log_l, &options).x;
                }
                log_l.iter().map(|v| v.exp()).collect()
            }
        };
        let best = trial(&lambdas)?;
        let df_resid = nf - best.edf;
        let dispersion = match glm.dispersion {
            Dispersion::Fixed(phi) => phi,
            Dispersion::Deviance => best.deviance / df_resid,
            Dispersion::Pearson => {
                (0..n)
                    .map(|i| {
                        let m = best.irls.mu[i];
                        w[i] * (y[i] - m).powi(2) / glm.family.variance(m)
                    })
                    .sum::<f64>()
                    / df_resid
            }
        };
        // Bayesian posterior covariance φ (XᵀWX + S)⁻¹ (mgcv's Vp).
        let mut a = best.irls.information.clone();
        for (v, si) in a.iter_mut().zip(&best.penalty) {
            *v += si;
        }
        let l = cholesky(&a, p)
            .ok_or_else(|| Error::Data("the penalized system is singular".into()))?;
        let mut covariance: Vec<f64> = cholesky_inverse(&l, p)
            .into_iter()
            .map(|v| v * dispersion)
            .collect();
        // Monotone columns: β = exp(θ), and its covariance by the delta
        // method, diag(∂β/∂θ) Cov(θ) diag(∂β/∂θ).
        let mut coefficients = best.irls.beta.clone();
        let mut scale = vec![1.0; p];
        for &c in &exp_cols {
            coefficients[c] = coefficients[c].exp();
            scale[c] = coefficients[c];
        }
        for a in 0..p {
            for b in 0..p {
                covariance[a * p + b] *= scale[a] * scale[b];
            }
        }
        let reml_scale = if use_reml {
            Some(reml(&best, &lambdas)?.1)
        } else {
            None
        };
        // As the GLM's: the Gaussian's at the maximum-likelihood variance.
        let ll_dispersion = match glm.family {
            Family::Gaussian => best.deviance / nf,
            _ => dispersion,
        };
        let log_likelihood = (0..n)
            .map(|i| {
                glm.family
                    .log_likelihood(y[i], best.irls.mu[i], w[i], ll_dispersion)
            })
            .sum();
        Ok(GamFit {
            spec: self.clone(),
            score: score(&best, &lambdas),
            reml_scale,
            bases,
            tensors,
            names: x.names().to_vec(),
            coefficients,
            lambdas,
            edf: best.edf,
            covariance,
            dispersion,
            deviance: best.deviance,
            log_likelihood,
            fitted: best.irls.mu,
            input_hash: crate::input_hash(design, y),
        })
    }
}

impl GamFit {
    /// Hash of the training data, as [`GlmFit::input_hash`](crate::GlmFit::input_hash).
    pub fn input_hash(&self) -> &str {
        &self.input_hash
    }

    /// Coefficient names: the parametric columns, then `s(x).1`, … for
    /// each smooth.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Coefficients.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Smoothing parameter of each smooth.
    pub fn lambdas(&self) -> &[f64] {
        &self.lambdas
    }

    /// Effective degrees of freedom of the whole model,
    /// `tr((XᵀWX + S)⁻¹ XᵀWX)`.
    pub fn edf(&self) -> f64 {
        self.edf
    }

    /// Bayesian posterior covariance of the coefficients,
    /// `φ (XᵀWX + S)⁻¹`, row-major.
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    /// Dispersion `φ` (Pearson's, on `n - edf` degrees of freedom, when
    /// estimated).
    pub fn dispersion(&self) -> f64 {
        self.dispersion
    }

    /// Residual deviance.
    pub fn deviance(&self) -> f64 {
        self.deviance
    }

    /// Log-likelihood at the fitted means and the dispersion, as
    /// [`GlmFit::log_likelihood`](crate::GlmFit::log_likelihood) (the
    /// Gaussian's at the maximum-likelihood variance `deviance / n`). NaN
    /// for a fit loaded from an artifact written before it was saved.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// `-2 ℓ + 2 edf`: the effective degrees of freedom as the penalty,
    /// not counting an estimated dispersion, as [`GlmFit::aic`](crate::GlmFit::aic)
    /// (mgcv's `AIC` adds 1 when the scale is estimated).
    pub fn aic(&self) -> f64 {
        -2.0 * self.log_likelihood + 2.0 * self.edf
    }

    /// The minimized GCV, UBRE or REML score (or the score at fixed
    /// smoothing parameters): mgcv's `gcv.ubre`.
    pub fn score(&self) -> f64 {
        self.score
    }

    /// The dispersion that REML estimates with the smoothing parameters
    /// (mgcv's `reml.scale`), or the fixed one; `None` for GCV or UBRE.
    /// [`dispersion`](Self::dispersion) stays the [`Dispersion`] estimate.
    pub fn reml_scale(&self) -> Option<f64> {
        self.reml_scale
    }

    /// Fitted means on the training data.
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }
}

impl Fitted for GamFit {
    /// Predictions for a design with the same columns as the training
    /// design (raw smooth columns included).
    fn predict(&self, design: &Design) -> Result<Vec<f64>> {
        let x = expand(design, &self.bases, &self.tensors)?;
        if x.names() != self.names.as_slice() {
            return Err(Error::Data(
                "design columns do not match the fitted GAM".into(),
            ));
        }
        Ok(x.linear_predictor(&self.coefficients)
            .into_iter()
            .map(|e| self.spec.glm.link.inverse(e))
            .collect())
    }

    /// Joint draws as for a GLM, with `β` drawn from its posterior
    /// `N(β̂, φ (XᵀWX + S)⁻¹)`.
    fn predict_distribution(
        &self,
        design: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        let x = expand(design, &self.bases, &self.tensors)?;
        let provenance = Provenance::new("gam")
            .version("prospicio-glm", env!("CARGO_PKG_VERSION"))
            .param("family", self.spec.glm.family.name())
            .param("link", format!("{:?}", self.spec.glm.link))
            .param("lambdas", format!("{:?}", self.lambdas));
        simulate_responses(
            &self.spec.glm,
            self.dispersion,
            &self.coefficients,
            Some(&self.covariance),
            None,
            &x,
            n_sims,
            seed,
            provenance,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_models::{Family, Link};

    #[test]
    fn constrained_basis_sums_to_zero_over_the_data() {
        let x: Vec<f64> = (0..50).map(|i| f64::from(i).sqrt()).collect();
        let b = Basis::fit(&PSpline::new("x"), &x).unwrap();
        for col in b.columns(&x) {
            assert!(col.iter().sum::<f64>().abs() < 1e-10);
        }
    }

    #[test]
    fn huge_smoothing_gives_a_straight_line() {
        // λ → ∞ leaves only the unpenalized linear part: the GLM on x.
        let x: Vec<f64> = (0..60).map(|i| f64::from(i) / 10.0).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|v| 1.0 + 0.5 * v + (3.0 * v).sin() * 0.3)
            .collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; 60], x.clone()],
        )
        .unwrap();
        let glm = Glm::new(Family::Gaussian, Link::Identity);
        let gam = Gam::new(glm, vec![PSpline::new("x")]).smoothing(Smoothing::Fixed(vec![1e8]));
        let fit = gam.fit(&d, &y).unwrap();
        let line = glm.fit(&d, &y).unwrap();
        for (a, b) in fit.fitted().iter().zip(line.fitted()) {
            assert!((a - b).abs() < 1e-6, "{a} {b}");
        }
        assert!((fit.edf() - 2.0).abs() < 1e-6);
        // At λ → ∞ the log-likelihood is the line's, and the AIC counts the
        // edf, here the line's two coefficients.
        assert!((fit.log_likelihood() - line.log_likelihood()).abs() < 1e-5);
        assert!((fit.aic() - line.aic()).abs() < 1e-5);
    }

    #[test]
    fn the_log_likelihood_is_the_familys_at_the_fit_and_round_trips() {
        let x: Vec<f64> = (0..80).map(|i| f64::from(i) / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|v| ((2.0 * v).sin() + 1.5).round()).collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; 80], x.clone()],
        )
        .unwrap();
        let glm = Glm::new(Family::Poisson, Link::Log);
        let fit = Gam::new(glm, vec![PSpline::new("x")]).fit(&d, &y).unwrap();
        let want: f64 = y
            .iter()
            .zip(fit.fitted())
            .map(|(&k, &m)| k * m.ln() - m - ln_gamma(k + 1.0))
            .sum();
        assert!((fit.log_likelihood() - want).abs() < 1e-9);
        assert!((fit.aic() - (-2.0 * want + 2.0 * fit.edf())).abs() < 1e-9);
        let back = GamFit::from_json(&fit.to_json()).unwrap();
        assert_eq!(back.log_likelihood(), fit.log_likelihood());
    }
}
