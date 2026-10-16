//! Posterior predictive checks (Gelman, Meng and Stern, 1996; Gabry et
//! al., 2019): does data simulated from the fitted model look like the
//! data it was fitted to?
//!
//! Each check compares the observed outcomes `y` with replicated data
//! `y_rep`, one replicated dataset per posterior draw (`S × n`, row-major
//! as [`crate::elpd`]'s log-likelihood: draw `s`'s replicate of every
//! observation together):
//!
//! - [`check`]: a test statistic `T` (mean, standard deviation, extremes,
//!   a quantile, the share of zeros, or any function of a dataset) of the
//!   observed data against its distribution over the replicates, with the
//!   posterior predictive p-value `P(T(y_rep) ≥ T(y))`, ties counted half.
//!   A p-value near 0 or 1 says the model cannot reproduce that feature of
//!   the data. [`check_grouped`] does the same within groups.
//! - [`loo_pit`]: the leave-one-out probability integral transform, each
//!   observation's place in its predictive distribution with the
//!   observation itself left out, by Pareto-smoothed importance weights
//!   ([`crate::elpd::psis`]). Uniform on `(0, 1)` for a calibrated model;
//!   U-shaped when the predictive distributions are too narrow, humped when
//!   too wide. Unlike the PIT under the full posterior, it does not count
//!   an observation as evidence for itself. Ties (counts) are randomized.
//!
//! [`crate::glm::BayesGlmFit`] makes the replicates and log-likelihood for
//! its own data (`posterior_predictive_check`, `loo_pit`); any other model
//! passes its own.

use prospicio_core::{Error, Result, StreamRng};

use crate::elpd::psis;

/// A test statistic of a dataset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Statistic {
    Mean,
    /// The standard deviation, divisor `n − 1`.
    Sd,
    Min,
    Max,
    /// R's type 7 quantile at this probability.
    Quantile(f64),
    /// The share of values equal to 0.
    ProportionZero,
}

impl Statistic {
    /// The statistic's name: `mean`, `sd`, `min`, `max`, `q<p>` (`q0.9`)
    /// or `proportion_zero`.
    pub fn name(&self) -> String {
        match self {
            Statistic::Mean => "mean".into(),
            Statistic::Sd => "sd".into(),
            Statistic::Min => "min".into(),
            Statistic::Max => "max".into(),
            Statistic::Quantile(p) => format!("q{p}"),
            Statistic::ProportionZero => "proportion_zero".into(),
        }
    }

    /// The statistic of `x`.
    pub fn of(&self, x: &[f64]) -> f64 {
        let n = x.len() as f64;
        match self {
            Statistic::Mean => x.iter().sum::<f64>() / n,
            Statistic::Sd => {
                let m = x.iter().sum::<f64>() / n;
                (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
            }
            Statistic::Min => x.iter().copied().fold(f64::INFINITY, f64::min),
            Statistic::Max => x.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            Statistic::Quantile(p) => {
                let mut v = x.to_vec();
                v.sort_by(f64::total_cmp);
                let h = (v.len() - 1) as f64 * p;
                let lo = h.floor() as usize;
                let hi = (lo + 1).min(v.len() - 1);
                v[lo] + (h - lo as f64) * (v[hi] - v[lo])
            }
            Statistic::ProportionZero => x.iter().filter(|&&v| v == 0.0).count() as f64 / n,
        }
    }

    /// `Some` of the probability of a quantile outside `[0, 1]`.
    fn filter_quantile(&self) -> Option<Statistic> {
        match *self {
            Statistic::Quantile(p) if !(0.0..=1.0).contains(&p) => Some(*self),
            _ => None,
        }
    }

    fn validate(&self) -> Result<()> {
        if let Some(Statistic::Quantile(p)) = self.filter_quantile() {
            return Err(Error::InvalidParameter {
                name: "quantile",
                value: p,
                reason: "must be in [0, 1]",
            });
        }
        Ok(())
    }
}

/// One test statistic checked against the replicates.
#[derive(Debug, Clone, PartialEq)]
pub struct StatisticCheck {
    /// The statistic's name (with the group for [`check_grouped`]).
    pub name: String,
    /// `T(y)`.
    pub observed: f64,
    /// `T(y_rep)` for each replicate, in draw order.
    pub replicated: Vec<f64>,
    /// `P(T(y_rep) > T(y)) + P(T(y_rep) = T(y)) / 2`.
    pub p_value: f64,
}

/// The leave-one-out PIT of each observation.
#[derive(Debug, Clone, PartialEq)]
pub struct LooPit {
    /// One per observation, in `[0, 1]`.
    pub pit: Vec<f64>,
    /// The Pareto `k̂` of each observation's importance weights; above about
    /// 0.7 its PIT is unreliable, as its PSIS-LOO term is.
    pub pareto_k: Vec<f64>,
    /// The Kolmogorov–Smirnov distance of the PITs from the uniform; about
    /// `1.36 / √n` or less 95% of the time for a calibrated model.
    pub ks: f64,
}

impl LooPit {
    /// Counts of the PITs in `bins` equal bins of `[0, 1]`.
    pub fn histogram(&self, bins: usize) -> Result<Vec<usize>> {
        prospicio_models::metrics::pit_histogram(&self.pit, bins)
    }
}

/// Checks that `observed` (`n` values) and `replicated` (`S × n`,
/// row-major) agree in shape, and returns `S`.
fn shape(observed: &[f64], replicated: &[f64]) -> Result<usize> {
    let n = observed.len();
    if n == 0 || replicated.is_empty() || replicated.len() % n != 0 {
        return Err(Error::Data(format!(
            "{} replicated values are not a whole number of datasets of {n}",
            replicated.len()
        )));
    }
    Ok(replicated.len() / n)
}

/// `P(T_rep > T_obs) + P(T_rep = T_obs) / 2`.
fn p_value(observed: f64, replicated: &[f64]) -> f64 {
    let above = replicated.iter().filter(|&&t| t > observed).count() as f64;
    let ties = replicated.iter().filter(|&&t| t == observed).count() as f64;
    (above + 0.5 * ties) / replicated.len() as f64
}

/// Each statistic of `observed` against its distribution over the
/// replicated datasets (`S × n`, row-major).
///
/// ```
/// use prospicio_bayes::ppc::{Statistic, check};
///
/// // Three replicates of two observations.
/// let y = [1.0, 3.0];
/// let y_rep = [0.0, 2.0, 2.0, 4.0, 1.0, 3.0];
/// let c = check(&y, &y_rep, &[Statistic::Mean]).unwrap();
/// assert_eq!(c[0].observed, 2.0);
/// assert_eq!(c[0].replicated, [1.0, 3.0, 2.0]);
/// assert_eq!(c[0].p_value, 0.5); // one above, one tie of three
/// ```
pub fn check(
    observed: &[f64],
    replicated: &[f64],
    statistics: &[Statistic],
) -> Result<Vec<StatisticCheck>> {
    shape(observed, replicated)?;
    statistics
        .iter()
        .map(|t| {
            t.validate()?;
            check_with(observed, replicated, &t.name(), |x| t.of(x))
        })
        .collect()
}

/// Any test statistic `t` of a dataset, named `name`, against its
/// distribution over the replicated datasets.
pub fn check_with(
    observed: &[f64],
    replicated: &[f64],
    name: &str,
    t: impl Fn(&[f64]) -> f64,
) -> Result<StatisticCheck> {
    shape(observed, replicated)?;
    let n = observed.len();
    let obs = t(observed);
    let rep: Vec<f64> = replicated.chunks_exact(n).map(&t).collect();
    Ok(StatisticCheck {
        name: name.to_string(),
        observed: obs,
        p_value: p_value(obs, &rep),
        replicated: rep,
    })
}

/// [`check`] within groups: `groups[i]` is observation `i`'s group (from
/// 0), and each statistic is checked on each group's observations alone;
/// names are `<statistic>[<group>]`, statistic by statistic, then group by
/// group.
pub fn check_grouped(
    observed: &[f64],
    replicated: &[f64],
    groups: &[usize],
    statistics: &[Statistic],
) -> Result<Vec<StatisticCheck>> {
    let s = shape(observed, replicated)?;
    let n = observed.len();
    if groups.len() != n {
        return Err(Error::Data(format!(
            "{} groups for {n} observations",
            groups.len()
        )));
    }
    let count = groups.iter().max().map_or(0, |g| g + 1);
    let members: Vec<Vec<usize>> = (0..count)
        .map(|g| (0..n).filter(|&i| groups[i] == g).collect())
        .collect();
    let mut out = Vec::new();
    for t in statistics {
        t.validate()?;
        for (g, idx) in members.iter().enumerate() {
            if idx.is_empty() {
                continue;
            }
            let obs: Vec<f64> = idx.iter().map(|&i| observed[i]).collect();
            let rep: Vec<f64> = (0..s)
                .flat_map(|d| idx.iter().map(move |&i| replicated[d * n + i]))
                .collect();
            let mut c = check_with(&obs, &rep, &t.name(), |x| t.of(x))?;
            c.name = format!("{}[{g}]", t.name());
            out.push(c);
        }
    }
    Ok(out)
}

/// The leave-one-out PIT of each observation, `Σ_s w̃_is (1{y_rep,si <
/// y_i} + u_i 1{y_rep,si = y_i})`, with `w̃_is` the normalized
/// Pareto-smoothed importance weights for leaving `y_i` out (log ratios
/// `−log p(y_i | θ_s)`, as [`crate::elpd::loo`]) and `u_i` uniform from stream 0 of
/// `seed`, in observation order (it only matters for ties). `log_lik` is
/// `S × n`, row-major, as `replicated`; `r_eff` each observation's relative
/// efficiency (1, independent draws, if `None`).
///
/// ```
/// use prospicio_bayes::ppc::loo_pit;
///
/// // Equal log-likelihoods: equal weights, so the PIT is the share of
/// // replicates below.
/// let y = [2.5];
/// let y_rep = [1.0, 2.0, 3.0, 4.0];
/// let ll = [-1.0; 4];
/// let p = loo_pit(&y, &y_rep, &ll, None, 0).unwrap();
/// assert_eq!(p.pit, [0.5]);
/// ```
pub fn loo_pit(
    observed: &[f64],
    replicated: &[f64],
    log_lik: &[f64],
    r_eff: Option<&[f64]>,
    seed: u64,
) -> Result<LooPit> {
    let s = shape(observed, replicated)?;
    let n = observed.len();
    if log_lik.len() != s * n {
        return Err(Error::Data(format!(
            "log_lik has {} values for {s} draws of {n} observations",
            log_lik.len()
        )));
    }
    if let Some(r) = r_eff.filter(|r| r.len() != n) {
        return Err(Error::Data(format!(
            "{} r_eff for {n} observations",
            r.len()
        )));
    }
    let mut rng = StreamRng::new(seed, 0);
    let mut pit = Vec::with_capacity(n);
    let mut pareto_k = Vec::with_capacity(n);
    for i in 0..n {
        let log_ratios: Vec<f64> = (0..s).map(|d| -log_lik[d * n + i]).collect();
        let (log_w, k) = psis(&log_ratios, r_eff.map_or(1.0, |r| r[i]))?;
        let top = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = log_w.iter().map(|v| (v - top).exp()).collect();
        let total: f64 = w.iter().sum();
        let (mut below, mut ties) = (0.0, 0.0);
        for (d, wd) in w.iter().enumerate() {
            let x = replicated[d * n + i];
            if x < observed[i] {
                below += wd;
            } else if x == observed[i] {
                ties += wd;
            }
        }
        let u = rng.next_open01();
        pit.push(((below + u * ties) / total).clamp(0.0, 1.0));
        pareto_k.push(k);
    }
    let ks = prospicio_models::metrics::ks_uniform(&pit)?;
    Ok(LooPit { pit, pareto_k, ks })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_math::special::norm_quantile;

    #[test]
    fn statistics_by_hand() {
        let x = [3.0, 0.0, 1.0, 0.0, 6.0];
        assert_eq!(Statistic::Mean.of(&x), 2.0);
        assert!((Statistic::Sd.of(&x) - 6.5_f64.sqrt()).abs() < 1e-12);
        assert_eq!(Statistic::Min.of(&x), 0.0);
        assert_eq!(Statistic::Max.of(&x), 6.0);
        // Type 7: sorted 0 0 1 3 6, h = 4 × 0.9 = 3.6.
        assert!((Statistic::Quantile(0.9).of(&x) - 4.8).abs() < 1e-12);
        assert_eq!(Statistic::ProportionZero.of(&x), 0.4);
        assert_eq!(Statistic::Quantile(0.9).name(), "q0.9");
        assert!(check(&x, &x, &[Statistic::Quantile(1.5)]).is_err());
        assert!(check(&x, &x[..3], &[Statistic::Mean]).is_err());
    }

    #[test]
    fn grouped_checks_take_each_groups_observations() {
        let y = [1.0, 10.0, 2.0];
        let y_rep = [0.0, 11.0, 4.0, 3.0, 9.0, 1.0];
        let c = check_grouped(&y, &y_rep, &[0, 1, 0], &[Statistic::Mean]).unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].name, "mean[0]");
        assert_eq!(c[0].observed, 1.5);
        assert_eq!(c[0].replicated, [2.0, 2.0]);
        assert_eq!(c[1].replicated, [11.0, 9.0]);
        assert_eq!(c[1].p_value, 0.5);
    }

    #[test]
    fn a_calibrated_model_has_uniform_pits_and_central_p_values() {
        // y_i ~ N(0, 1); the "posterior" is the truth, so the replicates are
        // independent N(0, 1) and the likelihoods equal across draws.
        let (n, s) = (400, 1000);
        let mut rng = StreamRng::new(7, 0);
        let mut z = || norm_quantile(rng.next_open01());
        let y: Vec<f64> = (0..n).map(|_| z()).collect();
        let y_rep: Vec<f64> = (0..s * n).map(|_| z()).collect();
        let ll: Vec<f64> = (0..s)
            .flat_map(|_| y.iter().map(|v| -0.5 * v * v))
            .collect();
        let p = loo_pit(&y, &y_rep, &ll, None, 1).unwrap();
        assert!(p.ks < 1.63 / (n as f64).sqrt(), "{}", p.ks);
        let c = check(&y, &y_rep, &[Statistic::Mean, Statistic::Sd]).unwrap();
        assert!(c.iter().all(|c| c.p_value > 0.005 && c.p_value < 0.995));
        // Replicates twice as wide: PITs pile up in the middle, and the
        // observed sd is low against them.
        let wide: Vec<f64> = y_rep.iter().map(|v| 2.0 * v).collect();
        let p = loo_pit(&y, &wide, &ll, None, 1).unwrap();
        let h = p.histogram(4).unwrap();
        assert!(h[1] + h[2] > 3 * (h[0] + h[3]), "{h:?}");
        let c = check(&y, &wide, &[Statistic::Sd]).unwrap();
        assert!(c[0].p_value > 0.999);
    }

    #[test]
    fn counts_are_randomized_over_ties() {
        // Every replicate equals y: the PIT is u, uniform.
        let n = 2000;
        let y = vec![3.0; n];
        let y_rep = vec![3.0; 10 * n];
        let ll = vec![-1.0; 10 * n];
        let p = loo_pit(&y, &y_rep, &ll, None, 5).unwrap();
        assert!(p.ks < 1.63 / (n as f64).sqrt());
        let again = loo_pit(&y, &y_rep, &ll, None, 5).unwrap();
        assert_eq!(p, again);
    }
}
