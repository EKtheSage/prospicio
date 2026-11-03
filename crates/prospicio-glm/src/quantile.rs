//! Linear quantile and expectile regression (`docs/design/models.md`,
//! "Quantile and expectile regression").
//!
//! [`QuantileRegression`] minimises the weighted check loss
//! `Σ wᵢ ρ_τ(yᵢ − xᵢᵀβ)`, `ρ_τ(u) = u (τ − 1{u < 0})`, as R's
//! `quantreg::rq`. The linear program is solved by a primal-dual interior
//! point method on its dual (Koenker and Portnoy's Frisch-Newton, with
//! Mehrotra's predictor-corrector steps), and the solution is then moved to
//! the vertex through the `p` rows the interior point fits most closely,
//! the solution `rq(method = "br")` returns when it is unique. Standard
//! errors are Powell's kernel sandwich, `summary.rq(se = "ker")`, with
//! Hall and Sheather's bandwidth (or Bofinger's), or a paired bootstrap.
//! [`QuantileRegression::fit_many`] fits several `τ` and can sort each
//! row's predictions across them so they do not cross (Chernozhukov,
//! Fernández-Val and Galichon's rearrangement on the fitted grid).
//!
//! [`ExpectileRegression`] minimises `Σ wᵢ |τ − 1{uᵢ < 0}| uᵢ²`,
//! asymmetric least squares (Newey and Powell, 1987), by iteratively
//! reweighted least squares, with the sandwich covariance `A⁻¹ B A⁻¹`. It
//! takes an optional quadratic penalty `λ βᵀ P β`, so a GAM's P-spline
//! basis and difference penalty give an expectile smooth as `expectreg`'s
//! at a fixed `λ`. At `τ = 0.5` it is least squares.
//!
//! Both fits implement [`Fitted`]; a design's offset is subtracted from the
//! response and added back to the predictions.

use prospicio_core::{Error, Result};
use prospicio_math::linalg::{cholesky, cholesky_inverse, cholesky_solve, solve};
use prospicio_math::special::{norm_pdf, norm_quantile};
use prospicio_models::{Design, Fitted, Model};
use prospicio_prob::{ComponentKey, KeyValue, PredictiveDistribution, Provenance};

fn check_tau(tau: f64) -> Result<()> {
    if !(tau > 0.0 && tau < 1.0) {
        return Err(Error::InvalidParameter {
            name: "tau",
            value: tau,
            reason: "must lie strictly between 0 and 1",
        });
    }
    Ok(())
}

/// The rows of `design` with a positive weight: `(X, y − offset, w)` with
/// `X` row-major.
fn rows(design: &Design, y: &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let (n, p) = (design.n_rows(), design.n_cols());
    if y.len() != n {
        return Err(Error::Data(format!(
            "the response has {} rows and the design {n}",
            y.len()
        )));
    }
    if let Some(&bad) = y.iter().find(|v| !v.is_finite()) {
        return Err(Error::InvalidParameter {
            name: "y",
            value: bad,
            reason: "must be finite",
        });
    }
    let (w, off) = (design.weights(), design.offset());
    let keep: Vec<usize> = (0..n).filter(|&i| w[i] > 0.0).collect();
    if keep.len() <= p {
        return Err(Error::Data(format!(
            "{} rows with a positive weight for {p} coefficients",
            keep.len()
        )));
    }
    let mut x = Vec::with_capacity(keep.len() * p);
    for &i in &keep {
        for j in 0..p {
            x.push(design.column(j)[i]);
        }
    }
    let yy = keep.iter().map(|&i| y[i] - off[i]).collect();
    let ww = keep.iter().map(|&i| w[i]).collect();
    Ok((x, yy, ww))
}

/// `Σ w xᵢ xᵢᵀ` for row-major `x`.
fn cross(x: &[f64], w: &[f64], p: usize) -> Vec<f64> {
    let mut m = vec![0.0; p * p];
    for (i, wi) in w.iter().enumerate() {
        let r = &x[i * p..(i + 1) * p];
        for a in 0..p {
            let ra = wi * r[a];
            for b in a..p {
                m[a * p + b] += ra * r[b];
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            m[a * p + b] = m[b * p + a];
        }
    }
    m
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Type 7 quantile (R's default) of sorted values.
fn quantile7(sorted: &[f64], p: f64) -> f64 {
    let h = (sorted.len() - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// The bandwidth rule of the kernel standard errors, as
/// `quantreg::bandwidth.rq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Bandwidth {
    /// Hall and Sheather (1988), `hs = TRUE`, quantreg's default.
    #[default]
    HallSheather,
    /// Bofinger (1975), `hs = FALSE`.
    Bofinger,
}

impl Bandwidth {
    /// `bandwidth.rq(tau, n, hs)` at `alpha = 0.05`.
    pub fn width(self, tau: f64, n: usize) -> f64 {
        let x0 = norm_quantile(tau);
        let f0 = norm_pdf(x0);
        let n = n as f64;
        match self {
            Self::HallSheather => {
                n.powf(-1.0 / 3.0)
                    * norm_quantile(0.975).powf(2.0 / 3.0)
                    * (1.5 * f0 * f0 / (2.0 * x0 * x0 + 1.0)).powf(1.0 / 3.0)
            }
            Self::Bofinger => {
                n.powf(-0.2) * (4.5 * f0.powi(4) / (2.0 * x0 * x0 + 1.0).powi(2)).powf(0.2)
            }
        }
    }
}

/// Linear quantile regression at one `τ`.
///
/// ```
/// use prospicio_glm::quantile::QuantileRegression;
/// use prospicio_models::{Design, Fitted, Model};
///
/// // The median of 1..=9 with an intercept alone.
/// let y: Vec<f64> = (1..=9).map(f64::from).collect();
/// let d = Design::new(vec!["(Intercept)".into()], vec![vec![1.0; 9]]).unwrap();
/// let fit = QuantileRegression::new(0.5).fit(&d, &y).unwrap();
/// assert!((fit.coefficients()[0] - 5.0).abs() < 1e-12);
/// assert!((fit.predict(&d).unwrap()[0] - 5.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantileRegression {
    pub tau: f64,
    /// The kernel standard errors' bandwidth.
    pub bandwidth: Bandwidth,
    /// The interior point stops when the duality gap is below this times
    /// `1 + |objective|`.
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl QuantileRegression {
    /// At `tau`, Hall and Sheather's bandwidth, tolerance `1e-12`, at most
    /// 200 interior-point steps.
    pub fn new(tau: f64) -> Self {
        Self {
            tau,
            bandwidth: Bandwidth::HallSheather,
            tolerance: 1e-12,
            max_iterations: 200,
        }
    }

    /// The same model with the kernel standard errors' bandwidth set.
    pub fn bandwidth(mut self, bandwidth: Bandwidth) -> Self {
        self.bandwidth = bandwidth;
        self
    }

    /// One fit for each of `taus` on the same data.
    pub fn fit_many(&self, taus: &[f64], design: &Design, y: &[f64]) -> Result<QuantileFits> {
        if taus.is_empty() {
            return Err(Error::Data("no tau given".into()));
        }
        let fits = taus
            .iter()
            .map(|&tau| QuantileRegression { tau, ..*self }.fit(design, y))
            .collect::<Result<Vec<_>>>()?;
        Ok(QuantileFits { fits })
    }

    /// Paired bootstrap covariance of the coefficients: `n_boot` resamples
    /// of the rows (with their weights) with replacement, each refitted,
    /// resample `b` drawn from stream `b` of `seed`.
    pub fn bootstrap_covariance(
        &self,
        design: &Design,
        y: &[f64],
        n_boot: usize,
        seed: u64,
    ) -> Result<Vec<f64>> {
        check_tau(self.tau)?;
        if n_boot < 2 {
            return Err(Error::Data(
                "the bootstrap needs at least 2 resamples".into(),
            ));
        }
        let (x, yy, w) = rows(design, y)?;
        let p = design.n_cols();
        let n = yy.len();
        let mut draws = Vec::with_capacity(n_boot);
        for replicate in 0..n_boot {
            let mut rng = prospicio_core::rng::StreamRng::new(seed, replicate as u64);
            let pick: Vec<usize> = (0..n)
                .map(|_| ((rng.next_open01() * n as f64) as usize).min(n - 1))
                .collect();
            let xb: Vec<f64> = pick
                .iter()
                .flat_map(|&i| x[i * p..(i + 1) * p].iter().copied())
                .collect();
            let yb: Vec<f64> = pick.iter().map(|&i| yy[i]).collect();
            let wb: Vec<f64> = pick.iter().map(|&i| w[i]).collect();
            match self.solve(&xb, &yb, &wb, p) {
                Ok((beta, _)) => draws.push(beta),
                Err(e) => return Err(e),
            }
        }
        let mean: Vec<f64> = (0..p)
            .map(|j| draws.iter().map(|b| b[j]).sum::<f64>() / n_boot as f64)
            .collect();
        let mut cov = vec![0.0; p * p];
        for b in &draws {
            for a in 0..p {
                for c in 0..p {
                    cov[a * p + c] += (b[a] - mean[a]) * (b[c] - mean[c]);
                }
            }
        }
        for v in &mut cov {
            *v /= (n_boot - 1) as f64;
        }
        Ok(cov)
    }

    /// The interior point on the dual, then the vertex. Returns `β` and the
    /// interior-point steps.
    fn solve(&self, x: &[f64], y: &[f64], w: &[f64], p: usize) -> Result<(Vec<f64>, usize)> {
        let n = y.len();
        let tau = self.tau;
        // Rows scaled by their weights: ρ_τ(w r) = w ρ_τ(r).
        let a: Vec<f64> = (0..n)
            .flat_map(|i| x[i * p..(i + 1) * p].iter().map(move |v| v * w[i]))
            .collect();
        let c: Vec<f64> = (0..n).map(|i| -y[i] * w[i]).collect();
        // The dual: min cᵀd subject to Aᵀd = b, 0 ≤ d ≤ 1, with
        // b = (1 − τ) Aᵀ1, from d = 1 − τ.
        let mut b = vec![0.0; p];
        for i in 0..n {
            for j in 0..p {
                b[j] += (1.0 - tau) * a[i * p + j];
            }
        }
        let at_mul = |v: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; p];
            for i in 0..n {
                for j in 0..p {
                    out[j] += a[i * p + j] * v[i];
                }
            }
            out
        };
        let a_mul = |v: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|i| (0..p).map(|j| a[i * p + j] * v[j]).sum())
                .collect()
        };
        let mut d = vec![1.0 - tau; n];
        let mut s = vec![tau; n];
        // Dual start from least squares: dual variables of Aᵀd = b.
        let ones = vec![1.0; n];
        let aa = cross(&a, &ones, p);
        let l = cholesky(&aa, p)
            .ok_or_else(|| Error::Data("the design's columns are linearly dependent".into()))?;
        let mut v = cholesky_solve(&l, &at_mul(&c));
        let r: Vec<f64> = {
            let av = a_mul(&v);
            (0..n).map(|i| c[i] - av[i]).collect()
        };
        let scale = r.iter().map(|x| x.abs()).fold(0.0, f64::max).max(1e-300);
        let mut z: Vec<f64> = r.iter().map(|&ri| ri.max(0.0) + 1e-3 * scale).collect();
        let mut u: Vec<f64> = r.iter().map(|&ri| (-ri).max(0.0) + 1e-3 * scale).collect();
        let step_to = |val: &[f64], dir: &[f64]| -> f64 {
            val.iter()
                .zip(dir)
                .filter(|(_, dv)| **dv < 0.0)
                .map(|(&x, &dv)| -x / dv)
                .fold(f64::INFINITY, f64::min)
        };
        let mut iterations = 0;
        for it in 1..=self.max_iterations {
            iterations = it;
            // Residuals: primal Aᵀd − b and dual c − A v − z + u.
            let rp: Vec<f64> = {
                let ad = at_mul(&d);
                (0..p).map(|j| b[j] - ad[j]).collect()
            };
            let rd: Vec<f64> = {
                let av = a_mul(&v);
                (0..n).map(|i| c[i] - av[i] - z[i] + u[i]).collect()
            };
            let gap = dot(&d, &z) + dot(&s, &u);
            let objective = dot(&c, &d);
            let infeasible = rd.iter().map(|x| x.abs()).fold(0.0, f64::max)
                + rp.iter().map(|x| x.abs()).fold(0.0, f64::max);
            if gap <= self.tolerance * (1.0 + objective.abs()) && infeasible <= 1e-9 * (1.0 + scale)
            {
                break;
            }
            let q: Vec<f64> = (0..n).map(|i| z[i] / d[i] + u[i] / s[i]).collect();
            let qinv: Vec<f64> = q.iter().map(|v| 1.0 / v).collect();
            let m = cross(&a, &qinv, p);
            let lm = cholesky(&m, p).ok_or_else(|| {
                Error::Data("the interior point's normal matrix is singular".into())
            })?;
            // Solve for (dd, dv, dz, du) given complementarity targets.
            let direction = |rxz: &[f64], rsu: &[f64]| {
                // g = rd − rxz/d + rsu/s; dd = Q⁻¹(A dv − g);
                // Aᵀ Q⁻¹ A dv = rp + Aᵀ Q⁻¹ g.
                let g: Vec<f64> = (0..n)
                    .map(|i| rd[i] - rxz[i] / d[i] + rsu[i] / s[i])
                    .collect();
                let qg: Vec<f64> = (0..n).map(|i| qinv[i] * g[i]).collect();
                let rhs: Vec<f64> = {
                    let t = at_mul(&qg);
                    (0..p).map(|j| rp[j] + t[j]).collect()
                };
                let dv = cholesky_solve(&lm, &rhs);
                let adv = a_mul(&dv);
                let dd: Vec<f64> = (0..n).map(|i| qinv[i] * (adv[i] - g[i])).collect();
                let dz: Vec<f64> = (0..n).map(|i| (rxz[i] - z[i] * dd[i]) / d[i]).collect();
                let du: Vec<f64> = (0..n).map(|i| (rsu[i] + u[i] * dd[i]) / s[i]).collect();
                (dd, dv, dz, du)
            };
            let lengths = |dd: &[f64], dz: &[f64], du: &[f64]| {
                let ds: Vec<f64> = dd.iter().map(|x| -x).collect();
                let ap = step_to(&d, dd).min(step_to(&s, &ds)).min(1.0 / 0.99995);
                let ad = step_to(&z, dz).min(step_to(&u, du)).min(1.0 / 0.99995);
                (0.99995 * ap, 0.99995 * ad)
            };
            // Predictor.
            let rxz: Vec<f64> = (0..n).map(|i| -d[i] * z[i]).collect();
            let rsu: Vec<f64> = (0..n).map(|i| -s[i] * u[i]).collect();
            let (dd, _, dz, du) = direction(&rxz, &rsu);
            let (ap, ad) = lengths(&dd, &dz, &du);
            let gap_aff: f64 = (0..n)
                .map(|i| {
                    (d[i] + ap * dd[i]) * (z[i] + ad * dz[i])
                        + (s[i] - ap * dd[i]) * (u[i] + ad * du[i])
                })
                .sum();
            let mu = gap / (2 * n) as f64;
            let sigma = (gap_aff / gap).powi(3);
            // Corrector.
            let rxz: Vec<f64> = (0..n)
                .map(|i| sigma * mu - d[i] * z[i] - dd[i] * dz[i])
                .collect();
            let rsu: Vec<f64> = (0..n)
                .map(|i| sigma * mu - s[i] * u[i] + dd[i] * du[i])
                .collect();
            let (dd, dv, dz, du) = direction(&rxz, &rsu);
            let (ap, ad) = lengths(&dd, &dz, &du);
            for i in 0..n {
                d[i] += ap * dd[i];
                s[i] -= ap * dd[i];
                z[i] += ad * dz[i];
                u[i] += ad * du[i];
            }
            for j in 0..p {
                v[j] += ad * dv[j];
            }
            if it == self.max_iterations {
                return Err(Error::Data(format!(
                    "quantile regression did not converge in {} interior-point steps",
                    self.max_iterations
                )));
            }
        }
        // β = −v; then the vertex through the p rows closest to the fit.
        let beta: Vec<f64> = v.iter().map(|x| -x).collect();
        let loss = |b: &[f64]| -> f64 {
            (0..n)
                .map(|i| {
                    let r = y[i] - dot(&x[i * p..(i + 1) * p], b);
                    w[i] * r * (tau - f64::from(u8::from(r < 0.0)))
                })
                .sum()
        };
        let mut order: Vec<usize> = (0..n).collect();
        let resid: Vec<f64> = (0..n)
            .map(|i| (y[i] - dot(&x[i * p..(i + 1) * p], &beta)).abs())
            .collect();
        order.sort_by(|&i, &j| resid[i].total_cmp(&resid[j]));
        let h = &order[..p];
        let xh: Vec<f64> = h
            .iter()
            .flat_map(|&i| x[i * p..(i + 1) * p].iter().copied())
            .collect();
        let yh: Vec<f64> = h.iter().map(|&i| y[i]).collect();
        if let Some(vertex) = solve(xh, yh, p) {
            let (lv, lb) = (loss(&vertex), loss(&beta));
            if lv <= lb + 1e-9 * lb.abs().max(1.0) {
                return Ok((vertex, iterations));
            }
        }
        Ok((beta, iterations))
    }
}

impl Model for QuantileRegression {
    type Fitted = QuantileFit;

    /// Fits the `τ` quantile; rows with weight 0 are left out. Fails for
    /// `τ` outside (0, 1), linearly dependent columns or too few rows.
    fn fit(&self, design: &Design, y: &[f64]) -> Result<QuantileFit> {
        check_tau(self.tau)?;
        let (x, yy, w) = rows(design, y)?;
        let p = design.n_cols();
        let (beta, iterations) = self.solve(&x, &yy, &w, p)?;
        let n = yy.len();
        let resid: Vec<f64> = (0..n)
            .map(|i| yy[i] - dot(&x[i * p..(i + 1) * p], &beta))
            .collect();
        let objective: f64 = (0..n)
            .map(|i| w[i] * resid[i] * (self.tau - f64::from(u8::from(resid[i] < 0.0))))
            .sum();
        let (covariance, h) = kernel_covariance(self.tau, self.bandwidth, &x, &resid, &w, p)?;
        Ok(QuantileFit {
            tau: self.tau,
            names: design.names().to_vec(),
            coefficients: beta,
            covariance,
            bandwidth: h,
            objective,
            iterations,
        })
    }
}

/// `summary.rq(se = "ker")`: `τ(1 − τ) (XᵀFX)⁻¹ XᵀX (XᵀFX)⁻¹` with
/// `F = diag(φ(û/h)/h)` on the weighted rows. Returns the covariance and
/// `h` on the residuals' scale.
fn kernel_covariance(
    tau: f64,
    rule: Bandwidth,
    x: &[f64],
    resid: &[f64],
    w: &[f64],
    p: usize,
) -> Result<(Vec<f64>, f64)> {
    let n = resid.len();
    let mut bw = rule.width(tau, n);
    while tau - bw < 0.0 || tau + bw > 1.0 {
        bw /= 2.0;
    }
    // Weighted rows, as summary.rq scales x, y and the residuals by w.
    let u: Vec<f64> = (0..n).map(|i| resid[i] * w[i]).collect();
    let xw: Vec<f64> = (0..n)
        .flat_map(|i| x[i * p..(i + 1) * p].iter().map(move |v| v * w[i]))
        .collect();
    let mean = u.iter().sum::<f64>() / n as f64;
    let sd = (u.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
    let mut sorted = u.clone();
    sorted.sort_by(f64::total_cmp);
    let iqr = (quantile7(&sorted, 0.75) - quantile7(&sorted, 0.25)) / 1.34;
    let h = (norm_quantile(tau + bw) - norm_quantile(tau - bw)) * sd.min(iqr);
    let f: Vec<f64> = u.iter().map(|v| norm_pdf(v / h) / h).collect();
    let fxx = cross(&xw, &f, p);
    let lf = cholesky(&fxx, p).ok_or_else(|| {
        Error::Data(
            "the kernel-weighted information is singular; the bandwidth is too small".into(),
        )
    })?;
    let fxx_inv = cholesky_inverse(&lf, p);
    let xx = cross(&xw, &vec![1.0; n], p);
    let mut cov = vec![0.0; p * p];
    for a in 0..p {
        for b in 0..p {
            let mut sum = 0.0;
            for k in 0..p {
                for l in 0..p {
                    sum += fxx_inv[a * p + k] * xx[k * p + l] * fxx_inv[l * p + b];
                }
            }
            cov[a * p + b] = tau * (1.0 - tau) * sum;
        }
    }
    Ok((cov, h))
}

/// A fitted linear quantile regression, from [`QuantileRegression`].
#[derive(Debug, Clone, PartialEq)]
pub struct QuantileFit {
    tau: f64,
    names: Vec<String>,
    coefficients: Vec<f64>,
    covariance: Vec<f64>,
    bandwidth: f64,
    objective: f64,
    iterations: usize,
}

impl QuantileFit {
    pub fn tau(&self) -> f64 {
        self.tau
    }

    /// The design's column names.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// The kernel sandwich covariance, row-major (`summary.rq(se = "ker",
    /// covariance = TRUE)`).
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    /// The kernel standard errors.
    pub fn standard_errors(&self) -> Vec<f64> {
        let p = self.coefficients.len();
        (0..p).map(|j| self.covariance[j * p + j].sqrt()).collect()
    }

    /// The kernel's bandwidth on the residuals' scale.
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// The minimised weighted check loss `Σ w ρ_τ(y − xᵀβ)`; `rq`'s `rho`
    /// is the same sum without the weights.
    pub fn objective(&self) -> f64 {
        self.objective
    }

    /// Interior-point steps used.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    fn check(&self, design: &Design) -> Result<()> {
        if design.names() != self.names.as_slice() {
            return Err(Error::Data(format!(
                "the design's columns {:?} are not the fit's {:?}",
                design.names(),
                self.names
            )));
        }
        Ok(())
    }
}

impl Fitted for QuantileFit {
    /// The fitted `τ` quantile `xᵀβ + offset` of each row.
    fn predict(&self, design: &Design) -> Result<Vec<f64>> {
        self.check(design)?;
        Ok(design.linear_predictor(&self.coefficients))
    }

    /// Draws of each row's fitted quantile with the coefficients' normal
    /// approximation under the kernel covariance: the uncertainty of the
    /// quantile, not a draw of the response.
    fn predict_distribution(
        &self,
        design: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        self.check(design)?;
        coefficient_draws(
            &self.coefficients,
            &self.covariance,
            design,
            n_sims,
            seed,
            Provenance::new("quantile_regression")
                .version("prospicio-glm", env!("CARGO_PKG_VERSION"))
                .param("tau", self.tau),
        )
    }
}

fn coefficient_draws(
    beta: &[f64],
    cov: &[f64],
    design: &Design,
    n_sims: usize,
    seed: u64,
    provenance: Provenance,
) -> Result<PredictiveDistribution> {
    let p = beta.len();
    let factor = cholesky(cov, p)
        .ok_or_else(|| Error::Data("the coefficient covariance is singular".into()))?;
    let components: Vec<ComponentKey> = (0..design.n_rows())
        .map(|i| vec![KeyValue::from(i as i64)])
        .collect();
    PredictiveDistribution::simulate(
        vec!["row".into()],
        components,
        n_sims,
        seed,
        provenance,
        |rng, row| {
            let z: Vec<f64> = (0..p).map(|_| norm_quantile(rng.next_open01())).collect();
            let mut shift = vec![0.0; p];
            prospicio_math::linalg::lower_mul(&factor, &z, &mut shift);
            let b: Vec<f64> = beta.iter().zip(&shift).map(|(a, s)| a + s).collect();
            for (out, eta) in row.iter_mut().zip(design.linear_predictor(&b)) {
                *out = eta;
            }
        },
    )
}

/// Quantile regressions at several `τ`, from
/// [`QuantileRegression::fit_many`].
#[derive(Debug, Clone, PartialEq)]
pub struct QuantileFits {
    fits: Vec<QuantileFit>,
}

impl QuantileFits {
    /// The fits, in the order of the `τ` given.
    pub fn fits(&self) -> &[QuantileFit] {
        &self.fits
    }

    /// The `τ`, in the order given.
    pub fn taus(&self) -> Vec<f64> {
        self.fits.iter().map(|f| f.tau).collect()
    }

    /// Each row's predicted quantiles, one vector per row in the order of
    /// the `τ`. With `rearrange` the fits' `τ` are put in increasing order
    /// and each row's values sorted, so the quantiles do not cross
    /// (Chernozhukov, Fernández-Val and Galichon, 2010, on the fitted grid);
    /// the vector then follows the `τ` sorted.
    pub fn predict(&self, design: &Design, rearrange: bool) -> Result<Vec<Vec<f64>>> {
        let mut order: Vec<usize> = (0..self.fits.len()).collect();
        if rearrange {
            order.sort_by(|&a, &b| self.fits[a].tau.total_cmp(&self.fits[b].tau));
        }
        let columns = order
            .iter()
            .map(|&k| self.fits[k].predict(design))
            .collect::<Result<Vec<_>>>()?;
        Ok((0..design.n_rows())
            .map(|i| {
                let mut row: Vec<f64> = columns.iter().map(|c| c[i]).collect();
                if rearrange {
                    row.sort_by(f64::total_cmp);
                }
                row
            })
            .collect())
    }
}

/// Linear expectile regression (asymmetric least squares) at one `τ`,
/// with an optional quadratic penalty.
///
/// ```
/// use prospicio_glm::quantile::ExpectileRegression;
/// use prospicio_models::{Design, Model};
///
/// // At τ = 0.5 the expectile is the mean.
/// let y = [1.0, 2.0, 4.0, 9.0];
/// let d = Design::new(vec!["(Intercept)".into()], vec![vec![1.0; 4]]).unwrap();
/// let fit = ExpectileRegression::new(0.5).fit(&d, &y).unwrap();
/// assert!((fit.coefficients()[0] - 4.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ExpectileRegression {
    pub tau: f64,
    /// `λ P`, row-major `p × p`, added to the normal equations; `None` for
    /// none.
    pub penalty: Option<Vec<f64>>,
    pub max_iterations: usize,
}

impl ExpectileRegression {
    /// At `tau`, unpenalised, at most 100 reweightings.
    pub fn new(tau: f64) -> Self {
        Self {
            tau,
            penalty: None,
            max_iterations: 100,
        }
    }

    /// The same model with the penalty `λ βᵀ P β`, `P` row-major `p × p`
    /// (a P-spline's `Dᵀ D` with zeros for unpenalised columns).
    pub fn penalty(mut self, lambda: f64, matrix: Vec<f64>) -> Self {
        self.penalty = Some(matrix.into_iter().map(|v| lambda * v).collect());
        self
    }
}

impl Model for ExpectileRegression {
    type Fitted = ExpectileFit;

    /// Iterates weighted least squares with weights `w |τ − 1{u < 0}|`
    /// until the signs of the residuals stop changing (it ends in finitely
    /// many steps). Fails for `τ` outside (0, 1), a penalty of the wrong
    /// size or singular normal equations.
    fn fit(&self, design: &Design, y: &[f64]) -> Result<ExpectileFit> {
        check_tau(self.tau)?;
        let (x, yy, w) = rows(design, y)?;
        let p = design.n_cols();
        let n = yy.len();
        if let Some(pen) = &self.penalty {
            if pen.len() != p * p {
                return Err(Error::Data(format!(
                    "the penalty has {} entries for {p} coefficients",
                    pen.len()
                )));
            }
        }
        let tau = self.tau;
        let wls = |omega: &[f64]| -> Result<(Vec<f64>, Vec<f64>)> {
            let ww: Vec<f64> = (0..n).map(|i| w[i] * omega[i]).collect();
            let mut m = cross(&x, &ww, p);
            if let Some(pen) = &self.penalty {
                for (a, b) in m.iter_mut().zip(pen) {
                    *a += b;
                }
            }
            let mut rhs = vec![0.0; p];
            for i in 0..n {
                for j in 0..p {
                    rhs[j] += ww[i] * x[i * p + j] * yy[i];
                }
            }
            let l = cholesky(&m, p).ok_or_else(|| {
                Error::Data("the expectile's normal equations are singular".into())
            })?;
            Ok((cholesky_solve(&l, &rhs), l))
        };
        let mut omega = vec![0.5; n];
        let mut iterations = 0;
        let mut beta = Vec::new();
        let mut converged = false;
        for it in 1..=self.max_iterations {
            iterations = it;
            beta = wls(&omega)?.0;
            let next: Vec<f64> = (0..n)
                .map(|i| {
                    let r = yy[i] - dot(&x[i * p..(i + 1) * p], &beta);
                    if r < 0.0 { 1.0 - tau } else { tau }
                })
                .collect();
            if next == omega {
                converged = true;
                break;
            }
            omega = next;
        }
        if !converged {
            return Err(Error::Data(format!(
                "the expectile's weights did not settle in {} steps",
                self.max_iterations
            )));
        }
        // Sandwich: A = Σ w ω x xᵀ (+ penalty), B = Σ (w ω e)² x xᵀ.
        let resid: Vec<f64> = (0..n)
            .map(|i| yy[i] - dot(&x[i * p..(i + 1) * p], &beta))
            .collect();
        let ww: Vec<f64> = (0..n).map(|i| w[i] * omega[i]).collect();
        let mut a = cross(&x, &ww, p);
        if let Some(pen) = &self.penalty {
            for (u, v) in a.iter_mut().zip(pen) {
                *u += v;
            }
        }
        let la = cholesky(&a, p)
            .ok_or_else(|| Error::Data("the expectile's information is singular".into()))?;
        let a_inv = cholesky_inverse(&la, p);
        let bw: Vec<f64> = (0..n).map(|i| (ww[i] * resid[i]).powi(2)).collect();
        let b = cross(&x, &bw, p);
        let mut cov = vec![0.0; p * p];
        for r in 0..p {
            for c in 0..p {
                let mut sum = 0.0;
                for k in 0..p {
                    for l in 0..p {
                        sum += a_inv[r * p + k] * b[k * p + l] * a_inv[l * p + c];
                    }
                }
                cov[r * p + c] = sum;
            }
        }
        let objective = (0..n).map(|i| ww[i] * resid[i] * resid[i]).sum();
        Ok(ExpectileFit {
            tau,
            names: design.names().to_vec(),
            coefficients: beta,
            covariance: cov,
            objective,
            iterations,
        })
    }
}

/// A fitted expectile regression, from [`ExpectileRegression`].
#[derive(Debug, Clone, PartialEq)]
pub struct ExpectileFit {
    tau: f64,
    names: Vec<String>,
    coefficients: Vec<f64>,
    covariance: Vec<f64>,
    objective: f64,
    iterations: usize,
}

impl ExpectileFit {
    pub fn tau(&self) -> f64 {
        self.tau
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// The sandwich covariance `A⁻¹ B A⁻¹`, `A = Σ w ω x xᵀ` (with the
    /// penalty) and `B = Σ (w ω e)² x xᵀ`, row-major; at `τ = 0.5` without a
    /// penalty, least squares' HC0.
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    pub fn standard_errors(&self) -> Vec<f64> {
        let p = self.coefficients.len();
        (0..p).map(|j| self.covariance[j * p + j].sqrt()).collect()
    }

    /// The minimised `Σ w |τ − 1{e < 0}| e²` (without the penalty).
    pub fn objective(&self) -> f64 {
        self.objective
    }

    /// Weighted least-squares fits used.
    pub fn iterations(&self) -> usize {
        self.iterations
    }
}

impl Fitted for ExpectileFit {
    /// The fitted `τ` expectile `xᵀβ + offset` of each row.
    fn predict(&self, design: &Design) -> Result<Vec<f64>> {
        if design.names() != self.names.as_slice() {
            return Err(Error::Data(format!(
                "the design's columns {:?} are not the fit's {:?}",
                design.names(),
                self.names
            )));
        }
        Ok(design.linear_predictor(&self.coefficients))
    }

    /// Draws of each row's fitted expectile under the sandwich covariance.
    fn predict_distribution(
        &self,
        design: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        self.predict(design)?;
        coefficient_draws(
            &self.coefficients,
            &self.covariance,
            design,
            n_sims,
            seed,
            Provenance::new("expectile_regression")
                .version("prospicio-glm", env!("CARGO_PKG_VERSION"))
                .param("tau", self.tau),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Design, Vec<f64>) {
        let n = 120;
        let x: Vec<f64> = (0..n).map(|i| f64::from(i) / 10.0).collect();
        // A heteroscedastic response from a fixed sequence.
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let e = f64::from((i * 37 + 11) % 23) / 23.0 - 0.5;
                1.0 + 0.5 * x[i as usize] + (1.0 + 0.3 * x[i as usize]) * e
            })
            .collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; n as usize], x],
        )
        .unwrap();
        (d, y)
    }

    /// The fit is a vertex through p rows, and no single-coordinate move
    /// lowers the check loss: the subgradient condition.
    #[test]
    fn the_quantile_fit_is_an_optimal_vertex() {
        let (d, y) = data();
        for tau in [0.1, 0.5, 0.9] {
            let fit = QuantileRegression::new(tau).fit(&d, &y).unwrap();
            let pred = fit.predict(&d).unwrap();
            let on = y
                .iter()
                .zip(&pred)
                .filter(|(a, b)| (*a - *b).abs() < 1e-9)
                .count();
            assert!(on >= 2, "tau {tau}: {on} rows on the fit");
            let loss = |b: &[f64]| -> f64 {
                y.iter()
                    .zip(d.linear_predictor(b))
                    .map(|(yi, m)| {
                        let r = yi - m;
                        r * (tau - f64::from(u8::from(r < 0.0)))
                    })
                    .sum()
            };
            let base = loss(fit.coefficients());
            assert!((base - fit.objective()).abs() < 1e-9 * base);
            for j in 0..2 {
                for h in [1e-4, -1e-4] {
                    let mut b = fit.coefficients().to_vec();
                    b[j] += h;
                    assert!(loss(&b) >= base - 1e-12, "tau {tau}, coef {j}");
                }
            }
        }
    }

    /// Integer weights equal repeated rows.
    #[test]
    fn weights_repeat_rows() {
        let (d, y) = data();
        let w: Vec<f64> = (0..y.len())
            .map(|i| f64::from(1 + (i % 3) as u32))
            .collect();
        let fit = QuantileRegression::new(0.7)
            .fit(&d.clone().with_weights(w.clone()).unwrap(), &y)
            .unwrap();
        let mut rows = Vec::new();
        for (i, &wi) in w.iter().enumerate() {
            for _ in 0..wi as usize {
                rows.push(i);
            }
        }
        let rep = d.select(&rows);
        let yr: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
        let fit2 = QuantileRegression::new(0.7).fit(&rep, &yr).unwrap();
        for (a, b) in fit.coefficients().iter().zip(fit2.coefficients()) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn rearrangement_sorts_each_row() {
        let (d, y) = data();
        let fits = QuantileRegression::new(0.5)
            .fit_many(&[0.9, 0.1, 0.5], &d, &y)
            .unwrap();
        let rows = fits.predict(&d, true).unwrap();
        for r in &rows {
            assert!(r.windows(2).all(|w| w[0] <= w[1]));
        }
        let raw = fits.predict(&d, false).unwrap();
        assert!(raw[60][0] > raw[60][1]);
    }

    #[test]
    fn the_bootstrap_is_near_the_kernel() {
        let (d, y) = data();
        let spec = QuantileRegression::new(0.5);
        let fit = spec.fit(&d, &y).unwrap();
        let boot = spec.bootstrap_covariance(&d, &y, 300, 3).unwrap();
        let se = fit.standard_errors();
        for j in 0..2 {
            let r = boot[j * 2 + j].sqrt() / se[j];
            assert!(r > 0.5 && r < 2.0, "{j}: ratio {r}");
        }
    }

    /// At τ = 0.5 the expectile is least squares, its covariance HC0.
    #[test]
    fn the_half_expectile_is_least_squares() {
        let (d, y) = data();
        let fit = ExpectileRegression::new(0.5).fit(&d, &y).unwrap();
        let ls = crate::Glm::new(
            prospicio_models::Family::Gaussian,
            prospicio_models::Link::Identity,
        )
        .fit(&d, &y)
        .unwrap();
        for (a, b) in fit.coefficients().iter().zip(ls.coefficients()) {
            assert!((a - b).abs() < 1e-10);
        }
        assert_eq!(fit.iterations(), 1);
        let tau9 = ExpectileRegression::new(0.9).fit(&d, &y).unwrap();
        assert!(tau9.coefficients()[0] > fit.coefficients()[0]);
    }

    #[test]
    fn it_rejects_what_it_cannot_fit() {
        let (d, y) = data();
        assert!(QuantileRegression::new(1.0).fit(&d, &y).is_err());
        assert!(ExpectileRegression::new(0.0).fit(&d, &y).is_err());
        assert!(
            ExpectileRegression::new(0.5)
                .penalty(1.0, vec![1.0])
                .fit(&d, &y)
                .is_err()
        );
        let dup = Design::new(
            vec!["a".into(), "b".into()],
            vec![vec![1.0; y.len()], vec![2.0; y.len()]],
        )
        .unwrap();
        assert!(QuantileRegression::new(0.5).fit(&dup, &y).is_err());
    }
}
