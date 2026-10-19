//! Fits of the body families to grouped losses, a count of losses in each
//! size band with no individual amounts (`docs/design/distributions.md`):
//! by maximum likelihood or minimum chi-square on the bands, and by the
//! minimum distance between the fitted and the empirical limited expected
//! values (Bahnemann, *Distributions for Actuaries*, CAS Monograph 2,
//! sections 2.4 and 2.5).
//!
//! The losses are conditioned on lying within the bands: a first bound
//! above 0 is a deductible's truncation and a finite last bound a
//! truncation from above. A sample with neither starts at 0 and ends with
//! an open band to `+∞`.

use prospicio_core::{Error, Result};
use prospicio_math::linalg::solve;
use prospicio_math::optimize::{NelderMead, nelder_mead};

use crate::LargeLosses;
use crate::dist::Dist;
use crate::distribution::Distribution;
use crate::severity_fit::{Family, fit_severity};

/// Losses counted by size band: `counts[k]` losses (or any weights) in
/// `(bounds[k], bounds[k + 1]]`.
///
/// ```
/// use prospicio_prob::grouped_fit::GroupedLosses;
///
/// let data = GroupedLosses::new(vec![0.0, 1000.0, f64::INFINITY], vec![30.0, 10.0]).unwrap();
/// assert_eq!(data.total(), 40.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GroupedLosses {
    bounds: Vec<f64>,
    counts: Vec<f64>,
}

impl GroupedLosses {
    /// `bounds` one longer than `counts`, non-negative and strictly
    /// increasing, finite but for a last bound of `+∞`; `counts`
    /// non-negative with a positive total.
    pub fn new(bounds: Vec<f64>, counts: Vec<f64>) -> Result<Self> {
        if counts.is_empty() || bounds.len() != counts.len() + 1 {
            return Err(Error::Data(format!(
                "need one more bound than counts and at least one band, not {} bounds and {} counts",
                bounds.len(),
                counts.len()
            )));
        }
        let n = bounds.len();
        for (k, w) in bounds.windows(2).enumerate() {
            if w[0].is_nan()
                || w[0] < 0.0
                || w[1].is_nan()
                || w[1] <= w[0]
                || (k + 2 < n && !w[1].is_finite())
            {
                return Err(Error::InvalidParameter {
                    name: "bounds",
                    value: w[1],
                    reason: "must be non-negative and strictly increasing, finite but for the last",
                });
            }
        }
        if counts.iter().any(|&c| !c.is_finite() || c < 0.0) {
            return Err(Error::Data("counts must be finite and non-negative".into()));
        }
        if counts.iter().sum::<f64>() <= 0.0 {
            return Err(Error::Data("counts must have a positive total".into()));
        }
        Ok(Self { bounds, counts })
    }

    /// The band bounds.
    pub fn bounds(&self) -> &[f64] {
        &self.bounds
    }

    /// The count in each band.
    pub fn counts(&self) -> &[f64] {
        &self.counts
    }

    /// The total count.
    pub fn total(&self) -> f64 {
        self.counts.iter().sum()
    }

    /// Each band's probability under `dist`, conditional on the range of
    /// the bands: `(S(b_k) − S(b_{k+1})) / (S(b_0) − S(b_n))`. `None` when
    /// `dist` gives the range no probability.
    pub fn probabilities<D: Distribution + ?Sized>(&self, dist: &D) -> Option<Vec<f64>> {
        let s: Vec<f64> = self
            .bounds
            .iter()
            .map(|&b| {
                if b <= 0.0 {
                    1.0
                } else if b == f64::INFINITY {
                    0.0
                } else {
                    dist.survival(b)
                }
            })
            .collect();
        let mass = s[0] - s[s.len() - 1];
        if mass.is_nan() || mass <= 0.0 {
            return None;
        }
        Some(s.windows(2).map(|w| (w[0] - w[1]) / mass).collect())
    }

    /// Pearson's chi-square statistic of `dist`, `Σ (O − E)² / E` with
    /// `E` the total times each band's probability. Infinite where a band
    /// with a count has no probability.
    pub fn chi_square<D: Distribution + ?Sized>(&self, dist: &D) -> f64 {
        let Some(p) = self.probabilities(dist) else {
            return f64::INFINITY;
        };
        let n = self.total();
        self.counts
            .iter()
            .zip(&p)
            .map(|(&o, &p)| {
                let e = n * p;
                if e > 0.0 {
                    (o - e).powi(2) / e
                } else if o > 0.0 {
                    f64::INFINITY
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// The multinomial log-likelihood of `dist`, `Σ O ln p`.
    pub fn log_likelihood<D: Distribution + ?Sized>(&self, dist: &D) -> f64 {
        let Some(p) = self.probabilities(dist) else {
            return f64::NEG_INFINITY;
        };
        self.counts
            .iter()
            .zip(&p)
            .map(|(&o, &p)| if o > 0.0 { o * p.ln() } else { 0.0 })
            .sum()
    }

    /// Losses at the bands' midpoints, the open top band censored at its
    /// lower bound and every loss reported above the first bound: a start
    /// for the grouped fits.
    fn pseudo_losses(&self) -> Result<LargeLosses> {
        let (lo, hi) = (self.bounds[0], self.bounds[self.bounds.len() - 1]);
        let (mut y, mut c, mut w) = (vec![], vec![], vec![]);
        for (k, &n) in self.counts.iter().enumerate() {
            if n <= 0.0 {
                continue;
            }
            let (a, b) = (self.bounds[k], self.bounds[k + 1]);
            if b.is_finite() {
                y.push(0.5 * (a + b));
                c.push(false);
            } else {
                y.push(a.max(f64::MIN_POSITIVE));
                c.push(true);
            }
            w.push(n);
        }
        let m = y.len();
        let mut data = LargeLosses::new(y)?.weights(w)?.censored(c)?;
        if lo > 0.0 {
            data = data.reporting_thresholds(vec![lo; m])?;
        }
        if hi.is_finite() {
            data = data.truncated_above(vec![hi; m])?;
        }
        Ok(data)
    }
}

/// The objective of a grouped fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupedMethod {
    /// Maximum likelihood: maximize `Σ O ln p`.
    MaximumLikelihood,
    /// Minimum chi-square: minimize Pearson's `Σ (O − E)² / E`.
    MinimumChiSquare,
}

impl GroupedMethod {
    /// The method's name, `"ml"` or `"chi_square"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::MaximumLikelihood => "ml",
            Self::MinimumChiSquare => "chi_square",
        }
    }

    /// The method from its name.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "ml" => Ok(Self::MaximumLikelihood),
            "chi_square" => Ok(Self::MinimumChiSquare),
            other => Err(Error::Data(format!(
                "method must be \"ml\" or \"chi_square\", not {other:?}"
            ))),
        }
    }
}

/// A severity fitted to grouped losses.
#[derive(Debug, Clone)]
pub struct GroupedFit {
    family: Family,
    method: GroupedMethod,
    dist: Dist,
    parameters: Vec<f64>,
    covariance: Vec<f64>,
    chi_square: f64,
    log_likelihood: f64,
    expected: Vec<f64>,
    converged: bool,
}

impl GroupedFit {
    /// The fitted family.
    pub fn family(&self) -> Family {
        self.family
    }

    /// The objective it was fitted by.
    pub fn method(&self) -> GroupedMethod {
        self.method
    }

    /// The fitted distribution.
    pub fn dist(&self) -> &Dist {
        &self.dist
    }

    /// The parameters, named by [`Family::parameter_names`].
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    /// The covariance of the parameters, row-major: the inverse of the
    /// grouped data's Fisher information `N Σ ∇p ∇pᵀ / p`, which both
    /// methods attain asymptotically; NaN where it is singular.
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    /// Each parameter's standard error.
    pub fn standard_errors(&self) -> Vec<f64> {
        let n = self.parameters.len();
        (0..n).map(|i| self.covariance[i * n + i].sqrt()).collect()
    }

    /// Pearson's chi-square at the fit.
    pub fn chi_square(&self) -> f64 {
        self.chi_square
    }

    /// The multinomial log-likelihood at the fit.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// The expected count in each band at the fit.
    pub fn expected(&self) -> &[f64] {
        &self.expected
    }

    /// The degrees of freedom of the chi-square test: bands with a positive
    /// expected count, less one, less the number of parameters.
    pub fn degrees_of_freedom(&self) -> usize {
        let bands = self.expected.iter().filter(|&&e| e > 0.0).count();
        bands.saturating_sub(1 + self.parameters.len())
    }

    /// The chi-square test's p-value, `P(χ²_df > chi_square)`; NaN with
    /// no degrees of freedom.
    pub fn p_value(&self) -> f64 {
        let df = self.degrees_of_freedom();
        match crate::Gamma::new(df as f64 / 2.0, 2.0) {
            Ok(g) if df > 0 => g.survival(self.chi_square),
            _ => f64::NAN,
        }
    }

    /// Whether the simplex met its tolerance.
    pub fn converged(&self) -> bool {
        self.converged
    }
}

fn options() -> NelderMead {
    NelderMead {
        max_iterations: 40_000,
        tolerance: 1e-12,
        ..NelderMead::default()
    }
}

/// The best of several Nelder–Mead runs over the unconstrained parameters.
fn minimize(
    family: Family,
    starts: &[Vec<f64>],
    objective: impl Fn(&Dist) -> f64,
) -> Result<(Vec<f64>, f64, bool)> {
    let f = |u: &[f64]| {
        if u.iter().any(|v| !v.is_finite() || v.abs() > 700.0) {
            return f64::NAN;
        }
        match family.dist(&family.natural(u)) {
            Ok(d) => objective(&d),
            Err(_) => f64::NAN,
        }
    };
    let best = starts
        .iter()
        .map(|s| {
            // A restart from the first minimum polishes a long valley.
            let first = nelder_mead(f, &family.unconstrained(s), &options());
            let again = nelder_mead(f, &first.x, &options());
            if again.value <= first.value {
                again
            } else {
                first
            }
        })
        .min_by(|a, b| a.value.total_cmp(&b.value))
        .ok_or_else(|| Error::Data("no start".into()))?;
    if !best.value.is_finite() {
        return Err(Error::Data(format!(
            "no {} fits these losses: the objective is not finite at any point tried",
            family.name()
        )));
    }
    Ok((family.natural(&best.x), best.value, best.converged))
}

/// Fits `family` to grouped losses by `method`, from the maximum likelihood
/// fit of the losses placed at the bands' midpoints (the open top band
/// censored) and, for the minimum chi-square, from the grouped maximum
/// likelihood fit too.
///
/// ```
/// use prospicio_prob::grouped_fit::{GroupedLosses, GroupedMethod, fit_grouped};
/// use prospicio_prob::severity_fit::Family;
///
/// // Bahnemann's P2.16: a gamma by minimum chi-square on six bands.
/// let inf = f64::INFINITY;
/// let data = GroupedLosses::new(
///     vec![0.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, inf],
///     vec![42.0, 61.0, 47.0, 26.0, 14.0, 10.0],
/// )
/// .unwrap();
/// let fit = fit_grouped(Family::Gamma, &data, GroupedMethod::MinimumChiSquare).unwrap();
/// assert!((fit.parameters()[0] - 4.7432).abs() < 5e-5);
/// assert!((fit.parameters()[1] - 337.31).abs() < 5e-3);
/// ```
pub fn fit_grouped(
    family: Family,
    data: &GroupedLosses,
    method: GroupedMethod,
) -> Result<GroupedFit> {
    let mut starts = vec![];
    if let Ok(f) = data.pseudo_losses().and_then(|p| fit_severity(family, &p)) {
        starts.push(f.parameters().to_vec());
    }
    if starts.is_empty() {
        return Err(Error::Data(
            "need at least two bands with counts below the open top band to start a grouped fit"
                .into(),
        ));
    }
    let ml = minimize(family, &starts, |d| -data.log_likelihood(d))?;
    let (parameters, converged) = match method {
        GroupedMethod::MaximumLikelihood => (ml.0, ml.2),
        GroupedMethod::MinimumChiSquare => {
            starts.push(ml.0);
            let cs = minimize(family, &starts, |d| data.chi_square(d))?;
            (cs.0, cs.2)
        }
    };
    let dist = family.dist(&parameters)?;
    let p = data.probabilities(&dist).unwrap_or_default();
    let n = data.total();
    Ok(GroupedFit {
        family,
        method,
        covariance: information_inverse(family, &parameters, data),
        chi_square: data.chi_square(&dist),
        log_likelihood: data.log_likelihood(&dist),
        expected: p.iter().map(|q| n * q).collect(),
        dist,
        parameters,
        converged,
    })
}

/// `(N Σ ∇p ∇pᵀ / p)⁻¹` by central differences of the band probabilities.
fn information_inverse(family: Family, p: &[f64], data: &GroupedLosses) -> Vec<f64> {
    let k = p.len();
    let nan = vec![f64::NAN; k * k];
    let probs = |q: &[f64]| family.dist(q).ok().and_then(|d| data.probabilities(&d));
    let Some(p0) = probs(p) else {
        return nan;
    };
    let mut grads = Vec::with_capacity(k);
    for i in 0..k {
        let h = 1e-5 * p[i].abs().max(1e-3);
        let (mut a, mut b) = (p.to_vec(), p.to_vec());
        a[i] += h;
        b[i] -= h;
        let (Some(pa), Some(pb)) = (probs(&a), probs(&b)) else {
            return nan;
        };
        grads.push(
            pa.iter()
                .zip(&pb)
                .map(|(x, y)| (x - y) / (2.0 * h))
                .collect::<Vec<_>>(),
        );
    }
    let n = data.total();
    let mut info = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            info[i * k + j] = n
                * (0..p0.len())
                    .filter(|&c| p0[c] > 0.0)
                    .map(|c| grads[i][c] * grads[j][c] / p0[c])
                    .sum::<f64>();
        }
    }
    let mut cov = nan.clone();
    for c in 0..k {
        let mut e = vec![0.0; k];
        e[c] = 1.0;
        match solve(info.clone(), e, k) {
            Some(col) if col.iter().all(|v| v.is_finite()) => {
                for r in 0..k {
                    cov[r * k + c] = col[r];
                }
            }
            _ => return nan,
        }
    }
    cov
}

/// A severity fitted by the minimum distance between its limited expected
/// values and empirical ones.
#[derive(Debug, Clone)]
pub struct LevDistanceFit {
    family: Family,
    dist: Dist,
    parameters: Vec<f64>,
    distance: f64,
    fitted: Vec<f64>,
    converged: bool,
}

impl LevDistanceFit {
    /// The fitted family.
    pub fn family(&self) -> Family {
        self.family
    }

    /// The fitted distribution.
    pub fn dist(&self) -> &Dist {
        &self.dist
    }

    /// The parameters, named by [`Family::parameter_names`].
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    /// The minimized `Σ w (E[X; x] − e)²`.
    pub fn distance(&self) -> f64 {
        self.distance
    }

    /// The fitted `E[X; x]` at each size.
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }

    /// Whether the simplex met its tolerance.
    pub fn converged(&self) -> bool {
        self.converged
    }
}

/// Fits `family` by minimizing `Σ w_j (E[X; x_j] − e_j)²`, the weighted
/// squared distance between its limited expected values and the empirical
/// ones `e_j` at the sizes `x_j` (Bahnemann's minimum distance, section
/// 2.5). The empirical limited means are exact from grouped data with each
/// band's total, and stay exact under a policy limit at or above the last
/// size, so the fit reads censored data as is.
///
/// Without a `start`, it starts from losses spread to match the empirical
/// limited means: the slope `(e_j − e_{j−1}) / (x_j − x_{j−1})` estimates
/// the survival on each interval.
///
/// ```
/// use prospicio_prob::Severity;
/// use prospicio_prob::grouped_fit::fit_lev_distance;
/// use prospicio_prob::severity_fit::Family;
/// use prospicio_prob::Lognormal;
///
/// let truth = Lognormal::new(7.0, 1.2).unwrap();
/// let x = [500.0, 1000.0, 2000.0, 5000.0, 10_000.0];
/// let e: Vec<f64> = x.iter().map(|&v| truth.lev(v)).collect();
/// let fit = fit_lev_distance(Family::Lognormal, &x, &e, None, None).unwrap();
/// assert!((fit.parameters()[0] - 7.0).abs() < 1e-5);
/// ```
pub fn fit_lev_distance(
    family: Family,
    sizes: &[f64],
    empirical: &[f64],
    weights: Option<&[f64]>,
    start: Option<&[f64]>,
) -> Result<LevDistanceFit> {
    let n = sizes.len();
    if n < family.parameter_names().len() || empirical.len() != n {
        return Err(Error::Data(format!(
            "need one empirical limited mean per size and at least as many sizes as parameters ({})",
            family.parameter_names().len()
        )));
    }
    if sizes.windows(2).any(|w| w[1] <= w[0]) || sizes.iter().any(|&x| !x.is_finite() || x <= 0.0) {
        return Err(Error::Data(
            "sizes must be finite, positive and strictly increasing".into(),
        ));
    }
    if empirical.iter().any(|&e| !e.is_finite() || e < 0.0) {
        return Err(Error::Data(
            "empirical limited means must be finite and non-negative".into(),
        ));
    }
    let w: Vec<f64> = match weights {
        Some(w) if w.len() == n && w.iter().all(|&v| v.is_finite() && v >= 0.0) => w.to_vec(),
        Some(_) => {
            return Err(Error::Data(
                "weights must be one finite non-negative value per size".into(),
            ));
        }
        None => vec![1.0; n],
    };
    let mut starts = vec![];
    if let Some(s) = start {
        if s.len() != family.parameter_names().len() {
            return Err(Error::Data(format!(
                "start must have {} parameters",
                family.parameter_names().len()
            )));
        }
        starts.push(s.to_vec());
    }
    if let Some(s) = lev_start(family, sizes, empirical) {
        starts.push(s);
    }
    if starts.is_empty() {
        return Err(Error::Data(
            "the empirical limited means give no start: pass one".into(),
        ));
    }
    let lev_of = |d: &Dist| -> Option<Vec<f64>> {
        let s = d.as_severity()?;
        Some(sizes.iter().map(|&x| s.lev(x)).collect())
    };
    let objective = |d: &Dist| match lev_of(d) {
        Some(l) => l
            .iter()
            .zip(empirical)
            .zip(&w)
            .map(|((a, b), w)| w * (a - b).powi(2))
            .sum(),
        None => f64::NAN,
    };
    let (parameters, distance, converged) = minimize(family, &starts, objective)?;
    let dist = family.dist(&parameters)?;
    let fitted = lev_of(&dist).unwrap_or_default();
    Ok(LevDistanceFit {
        family,
        dist,
        parameters,
        distance,
        fitted,
        converged,
    })
}

/// A start from the empirical limited means: the survival on each interval
/// is the slope of `E[X; x]`, so the interval's losses carry the drop in
/// it, and those past the last size are censored there.
fn lev_start(family: Family, sizes: &[f64], e: &[f64]) -> Option<Vec<f64>> {
    let mut s = Vec::with_capacity(sizes.len());
    let (mut x0, mut e0) = (0.0, 0.0);
    for (&x, &v) in sizes.iter().zip(e) {
        s.push(((v - e0) / (x - x0)).clamp(1e-9, 1.0));
        (x0, e0) = (x, v);
    }
    // Survival is non-increasing: the band between interval starts.
    for j in 1..s.len() {
        s[j] = s[j].min(s[j - 1]);
    }
    let (mut y, mut c, mut w) = (vec![], vec![], vec![]);
    let mut lo = 0.0;
    let mut before = 1.0;
    for (j, &x) in sizes.iter().enumerate() {
        let drop = before - s[j];
        if drop > 1e-12 {
            y.push(0.5 * (lo + x));
            c.push(false);
            w.push(drop);
        }
        before = s[j];
        lo = x;
    }
    y.push(sizes[sizes.len() - 1]);
    c.push(true);
    w.push(before.max(1e-9));
    let data = LargeLosses::new(y)
        .ok()?
        .weights(w)
        .ok()?
        .censored(c)
        .ok()?;
    fit_severity(family, &data)
        .ok()
        .map(|f| f.parameters().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Gamma, Lognormal};

    const INF: f64 = f64::INFINITY;

    fn ex_2_6() -> GroupedLosses {
        GroupedLosses::new(
            vec![0.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, INF],
            vec![42.0, 61.0, 47.0, 26.0, 14.0, 10.0],
        )
        .unwrap()
    }

    #[test]
    fn chi_square_of_bahnemann_ex_2_7() {
        // The printed gamma (3.907288, 397.931) gives 3.095 (the book rounds
        // the expected counts to 3.096).
        let g = Gamma::new(3.907288, 397.931).unwrap();
        assert!((ex_2_6().chi_square(&g) - 3.095).abs() < 5e-4);
        let p = ex_2_6().probabilities(&g).unwrap();
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn minimum_chi_square_is_a_minimum() {
        // Ex 2.8: the printed lognormal is not the minimum; the minimum is
        // 0.614 at (7.2857, 0.4570).
        let fit = fit_grouped(
            Family::Lognormal,
            &ex_2_6(),
            GroupedMethod::MinimumChiSquare,
        )
        .unwrap();
        let p = fit.parameters();
        assert!(
            (p[0] - 7.2857).abs() < 5e-5 && (p[1] - 0.4570).abs() < 5e-5,
            "{p:?}"
        );
        assert!((fit.chi_square() - 0.614).abs() < 5e-4);
        assert_eq!(fit.degrees_of_freedom(), 3);
        // P(χ²_3 > 0.614) = 0.8933.
        assert!((fit.p_value() - 0.8933).abs() < 5e-4, "{}", fit.p_value());
        let printed = Lognormal::new(7.274670, 0.442525).unwrap();
        assert!(ex_2_6().chi_square(&printed) > fit.chi_square());
        assert!((fit.expected().iter().sum::<f64>() - 200.0).abs() < 1e-9);
    }

    #[test]
    fn grouped_maximum_likelihood_solves_its_score() {
        let data = ex_2_6();
        let fit = fit_grouped(Family::Gamma, &data, GroupedMethod::MaximumLikelihood).unwrap();
        let p = fit.parameters().to_vec();
        let ll = |q: &[f64]| data.log_likelihood(&Gamma::new(q[0], q[1]).unwrap());
        for i in 0..2 {
            let h = 1e-6 * p[i];
            let (mut a, mut b) = (p.clone(), p.clone());
            a[i] += h;
            b[i] -= h;
            let g = (ll(&a) - ll(&b)) / (2.0 * h);
            assert!((g * fit.standard_errors()[i]).abs() < 1e-4, "{g}");
        }
        assert!((fit.log_likelihood() - ll(&p)).abs() < 1e-12);
        // Its chi-square is above the minimum chi-square fit's.
        let cs = fit_grouped(Family::Gamma, &data, GroupedMethod::MinimumChiSquare).unwrap();
        assert!(cs.chi_square() <= fit.chi_square());
        assert!(
            fit.standard_errors()
                .iter()
                .all(|s| s.is_finite() && *s > 0.0)
        );
    }

    #[test]
    fn truncated_bands_condition_the_probabilities() {
        // Ex 2.11: 770 claims above a 1,000 deductible, censored at 200,000.
        let data = GroupedLosses::new(
            vec![
                1000.0, 5000.0, 10000.0, 25000.0, 50000.0, 75000.0, 100000.0, 125000.0, 150000.0,
                175000.0, 200000.0, INF,
            ],
            vec![
                367.0, 112.0, 118.0, 65.0, 36.0, 13.0, 10.0, 8.0, 6.0, 4.0, 31.0,
            ],
        )
        .unwrap();
        let fit = fit_grouped(Family::Lognormal, &data, GroupedMethod::MinimumChiSquare).unwrap();
        let p = fit.parameters();
        assert!(
            (p[0] - 6.6916).abs() < 5e-5 && (p[1] - 2.6965).abs() < 5e-5,
            "{p:?}"
        );
        assert!((fit.chi_square() - 4.691).abs() < 5e-4);
        let want = [360, 122, 121, 63, 27, 16, 10, 7, 5, 4, 34];
        for (e, w) in fit.expected().iter().zip(want) {
            assert_eq!(e.round() as i64, w);
        }
        // P2.43: only claims up to 50,000 were kept.
        let data = GroupedLosses::new(
            (0..=10).map(|k| 5000.0 * k as f64).collect(),
            vec![104.0, 120.0, 81.0, 54.0, 45.0, 32.0, 26.0, 18.0, 12.0, 8.0],
        )
        .unwrap();
        let fit = fit_grouped(Family::Lognormal, &data, GroupedMethod::MinimumChiSquare).unwrap();
        let p = fit.parameters();
        assert!(
            (p[0] - 9.495111).abs() < 5e-5 && (p[1] - 1.084180).abs() < 5e-5,
            "{p:?}"
        );
    }

    #[test]
    fn lev_distance_reproduces_bahnemann() {
        // Ex 2.10: 1,500 claims under a 300,000 limit, 23 at the limit.
        let sizes = [
            1000., 5000., 10000., 25000., 50000., 100000., 200000., 300000.,
        ];
        let below = [729., 1096., 1208., 1326., 1391., 1440., 1468., 1500.];
        let total = [
            225_138.,
            1_102_272.,
            1_918_947.,
            3_752_091.,
            6_007_543.,
            9_234_739.,
            13_100_561.,
            22_343_455.,
        ];
        let e: Vec<f64> = (0..8)
            .map(|j| (total[j] + sizes[j] * (1500.0 - below[j])) / 1500.0)
            .collect();
        let fit = fit_lev_distance(Family::Lognormal, &sizes, &e, None, None).unwrap();
        let p = fit.parameters();
        assert!(
            (p[0] - 6.9852).abs() < 5e-5 && (p[1] - 2.5850).abs() < 5e-5,
            "{p:?}"
        );
        let want = [648, 2091, 3239, 5410, 7580, 10168, 13069, 14850];
        for (f, w) in fit.fitted().iter().zip(want) {
            assert_eq!(f.round() as i64, w);
        }
        // P2.39: 200 claims, 22 at a 100,000 limit, grouped at midpoints.
        let sizes = [
            10000., 20000., 30000., 40000., 50000., 60000., 80000., 100000.,
        ];
        let counts = [78., 27., 21., 15., 12., 10., 8., 7.];
        let mids = [
            5000., 15000., 25000., 35000., 45000., 55000., 70000., 90000.,
        ];
        let e: Vec<f64> = (0..8)
            .map(|j| {
                let below: f64 = counts[..=j].iter().sum();
                let sum: f64 = (0..=j).map(|i| counts[i] * mids[i]).sum();
                (sum + sizes[j] * (200.0 - below)) / 200.0
            })
            .collect();
        let fit = fit_lev_distance(Family::Lognormal, &sizes, &e, None, None).unwrap();
        let p = fit.parameters();
        assert!(
            (p[0] - 9.778102).abs() < 5e-5 && (p[1] - 1.444776).abs() < 5e-5,
            "{p:?}"
        );
    }

    #[test]
    fn every_family_recovers_its_parameters() {
        // Counts equal to their expectation under the truth (with a
        // deductible and an open top band) put both grouped fits at the
        // truth; exact limited means put the distance fit there too.
        let truths: [(Family, &[f64]); 5] = [
            (Family::Lognormal, &[7.0, 1.1]),
            (Family::Gamma, &[1.8, 900.0]),
            (Family::Weibull, &[0.8, 1500.0]),
            (Family::Burr, &[2.5, 1.4, 2000.0]),
            (Family::Lomax, &[2.2, 1800.0]),
        ];
        let bounds = vec![
            100.0, 250.0, 500.0, 1000.0, 2000.0, 3500.0, 6000.0, 10_000.0, 20_000.0, INF,
        ];
        let sizes = [250.0, 500.0, 1000.0, 2000.0, 5000.0, 10_000.0, 25_000.0];
        for (family, truth) in truths {
            let d = family.dist(truth).unwrap();
            let one = GroupedLosses::new(bounds.clone(), vec![1.0; bounds.len() - 1]).unwrap();
            let counts: Vec<f64> = one
                .probabilities(&d)
                .unwrap()
                .iter()
                .map(|p| 10_000.0 * p)
                .collect();
            let data = GroupedLosses::new(bounds.clone(), counts).unwrap();
            for method in [
                GroupedMethod::MaximumLikelihood,
                GroupedMethod::MinimumChiSquare,
            ] {
                let fit = fit_grouped(family, &data, method).unwrap();
                for (p, t) in fit.parameters().iter().zip(truth) {
                    assert!(
                        (p / t - 1.0).abs() < 1e-3,
                        "{family:?} {method:?}: {p} vs {t}"
                    );
                }
                assert!(fit.chi_square() < 1e-6, "{family:?} {method:?}");
                assert!((fit.expected().iter().sum::<f64>() - 10_000.0).abs() < 1e-6);
                assert!(
                    fit.standard_errors()
                        .iter()
                        .all(|s| s.is_finite() && *s > 0.0)
                );
            }
            let s = d.as_severity().unwrap();
            let e: Vec<f64> = sizes.iter().map(|&x| s.lev(x)).collect();
            let fit = fit_lev_distance(family, &sizes, &e, None, None).unwrap();
            for (p, t) in fit.parameters().iter().zip(truth) {
                assert!(
                    (p / t - 1.0).abs() < 1e-3,
                    "{family:?} distance: {p} vs {t}"
                );
            }
        }
    }

    #[test]
    fn rejects_bad_data() {
        assert!(GroupedLosses::new(vec![0.0, 1.0], vec![1.0, 2.0]).is_err());
        assert!(GroupedLosses::new(vec![0.0, 2.0, 1.0], vec![1.0, 2.0]).is_err());
        assert!(GroupedLosses::new(vec![0.0, INF, 5.0], vec![1.0, 2.0]).is_err());
        assert!(GroupedLosses::new(vec![0.0, 1.0, 2.0], vec![0.0, 0.0]).is_err());
        assert!(GroupedLosses::new(vec![0.0, 1.0, 2.0], vec![-1.0, 2.0]).is_err());
        assert!(GroupedMethod::from_name("mle").is_err());
        assert_eq!(
            GroupedMethod::from_name("chi_square").unwrap(),
            GroupedMethod::MinimumChiSquare
        );
        assert!(fit_lev_distance(Family::Lognormal, &[1.0], &[1.0], None, None).is_err());
        let (x, e) = ([1.0, 2.0, 3.0], [0.9, 1.6, 2.1]);
        let short = [1.0, 1.0];
        assert!(fit_lev_distance(Family::Lognormal, &x, &e, Some(&short), None).is_err());
        assert!(fit_lev_distance(Family::Lognormal, &x, &e, None, Some(&[0.0])).is_err());
        assert!(fit_lev_distance(Family::Lognormal, &[2.0, 1.0], &[1.0, 1.0], None, None).is_err());
    }
}
