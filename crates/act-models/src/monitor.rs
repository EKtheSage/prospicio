//! Monitoring a stored model on new periods: actual against expected.
//!
//! As a model's predictions meet actuals, period by period, the questions
//! are whether any period is off by more than noise and whether the ratio
//! is drifting. [`actual_vs_expected`] answers both from the model's own
//! variance function: each period's `A - E` is scaled by its standard
//! deviation under the model, and a weighted trend of `A / E - 1` across
//! periods tests for drift. [`lift`](crate::metrics::lift) does the same
//! split by predicted rate instead of by period.

use act_core::{Error, Result};

use crate::family::Family;

/// One period's (or the total's) actual against expected.
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodSummary<K> {
    /// The period (`None` for the total).
    pub period: Option<K>,
    /// Number of rows.
    pub n: usize,
    /// `Σ w`: exposure or prior weight.
    pub weight: f64,
    /// `Σ w y`.
    pub actual: f64,
    /// `Σ w μ`.
    pub expected: f64,
    /// Standard deviation of `A - E` under the model,
    /// `√(φ Σ w V(μ))`.
    pub std_dev: f64,
}

impl<K> PeriodSummary<K> {
    /// `actual / expected`.
    pub fn ratio(&self) -> f64 {
        self.actual / self.expected
    }

    /// `(A - E) / std_dev`: about standard normal when the model holds;
    /// beyond ±2 or ±3 is worth a look.
    pub fn z(&self) -> f64 {
        (self.actual - self.expected) / self.std_dev
    }
}

/// The result of [`actual_vs_expected`].
#[derive(Debug, Clone, PartialEq)]
pub struct Monitor<K> {
    /// One row per period, in sorted order.
    pub periods: Vec<PeriodSummary<K>>,
    /// All periods together.
    pub total: PeriodSummary<K>,
    /// Slope of `A / E - 1` per period step (periods numbered 0, 1, ... in
    /// sorted order), by least squares weighted by each period's
    /// precision; NaN with fewer than three periods.
    pub trend: f64,
    /// Standard error of [`trend`](Self::trend) under the model.
    pub trend_std_error: f64,
}

impl<K> Monitor<K> {
    /// `trend / trend_std_error`: drift beyond noise when large.
    pub fn trend_z(&self) -> f64 {
        self.trend / self.trend_std_error
    }
}

/// Actual against expected by period, with each period's z-score under the
/// model (`family`'s variance function and `dispersion`) and a test for
/// drift.
///
/// `y` and `mu` are the responses and the stored model's predicted means
/// on the new rows; `weights` are the prior weights the model was fitted
/// with (exposure for a rate), so `A = Σ w y` and `E = Σ w μ`. For a
/// Poisson claim count with exposure in the offset, leave `weights` out:
/// `A` and `E` are claim counts and `Var(A) = Σ μ`.
///
/// ```
/// use act_models::Family;
/// use act_models::monitor::actual_vs_expected;
///
/// let periods = [2023, 2023, 2024, 2024];
/// let y = [1.0, 3.0, 2.0, 6.0];
/// let mu = [2.0, 2.0, 2.0, 2.0];
/// let m = actual_vs_expected(&periods, &y, &mu, None, Family::Poisson, 1.0).unwrap();
/// assert_eq!(m.periods[1].ratio(), 2.0);
/// // 2024: A - E = 4 with variance 4 under the Poisson.
/// assert_eq!(m.periods[1].z(), 2.0);
/// ```
pub fn actual_vs_expected<K: Ord + Clone>(
    periods: &[K],
    y: &[f64],
    mu: &[f64],
    weights: Option<&[f64]>,
    family: Family,
    dispersion: f64,
) -> Result<Monitor<K>> {
    let n = y.len();
    if periods.len() != n || mu.len() != n || weights.is_some_and(|w| w.len() != n) {
        return Err(Error::Data(
            "periods, y, mu and weights need one entry per row".into(),
        ));
    }
    if n == 0 {
        return Err(Error::Data("no rows to monitor".into()));
    }
    if !(dispersion.is_finite() && dispersion > 0.0) {
        return Err(Error::InvalidParameter {
            name: "dispersion",
            value: dispersion,
            reason: "must be finite and positive",
        });
    }
    let mut keys: Vec<K> = periods.to_vec();
    keys.sort();
    keys.dedup();
    let empty = |period| PeriodSummary {
        period,
        n: 0,
        weight: 0.0,
        actual: 0.0,
        expected: 0.0,
        std_dev: 0.0,
    };
    let mut rows: Vec<PeriodSummary<K>> = keys.iter().cloned().map(|k| empty(Some(k))).collect();
    let mut var = vec![0.0; keys.len()];
    for i in 0..n {
        let w = weights.map_or(1.0, |w| w[i]);
        let t = keys
            .binary_search(&periods[i])
            .expect("every period is a key");
        let r = &mut rows[t];
        r.n += 1;
        r.weight += w;
        r.actual += w * y[i];
        r.expected += w * mu[i];
        var[t] += dispersion * w * family.variance(mu[i]);
    }
    let mut total = empty(None);
    for (r, v) in rows.iter_mut().zip(&var) {
        r.std_dev = v.sqrt();
        total.n += r.n;
        total.weight += r.weight;
        total.actual += r.actual;
        total.expected += r.expected;
    }
    total.std_dev = var.iter().sum::<f64>().sqrt();

    // Weighted least squares of r_t = A_t / E_t - 1 on t, with
    // Var(r_t) = (std_dev_t / E_t)².
    let (mut sw, mut st, mut sr, mut stt, mut str_) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (t, row) in rows.iter().enumerate() {
        let w = (row.expected / row.std_dev).powi(2);
        let (tf, r) = (t as f64, row.ratio() - 1.0);
        sw += w;
        st += w * tf;
        sr += w * r;
        stt += w * tf * tf;
        str_ += w * tf * r;
    }
    let sxx = stt - st * st / sw;
    let (trend, trend_std_error) = if rows.len() >= 3 && sxx > 0.0 {
        ((str_ - st * sr / sw) / sxx, (1.0 / sxx).sqrt())
    } else {
        (f64::NAN, f64::NAN)
    };
    Ok(Monitor {
        periods: rows,
        total,
        trend,
        trend_std_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn totals_weights_and_trend() {
        // Expected 10 each period; actuals drift up by 1 a period.
        let periods: Vec<&str> = ["2021", "2022", "2023", "2024"]
            .iter()
            .flat_map(|p| [*p, *p])
            .collect();
        let mu = vec![5.0; 8];
        let y = [4.0, 6.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0];
        let m = actual_vs_expected(&periods, &y, &mu, None, Family::Poisson, 1.0).unwrap();
        assert_eq!(m.periods.len(), 4);
        assert_eq!(m.periods[0].period, Some("2021"));
        assert_eq!(
            m.periods.iter().map(|p| p.actual).collect::<Vec<_>>(),
            [10.0, 11.0, 12.0, 13.0]
        );
        assert_eq!(m.total.actual, 46.0);
        assert!((m.total.std_dev - 40f64.sqrt()).abs() < 1e-12);
        // Equal precision: ordinary least squares of 0, .1, .2, .3 on t.
        assert!((m.trend - 0.1).abs() < 1e-12);
        // Var(r_t) = 10 / 100; Σ (t - 1.5)² = 5.
        assert!((m.trend_std_error - (0.1f64 / 5.0).sqrt()).abs() < 1e-12);

        // Prior weights scale both sums; the dispersion scales the variance.
        let w = vec![2.0; 8];
        let g = actual_vs_expected(&periods, &y, &mu, Some(&w), Family::Gamma, 0.5).unwrap();
        assert_eq!(g.periods[3].expected, 20.0);
        assert!((g.periods[3].std_dev - (0.5f64 * 2.0 * 2.0 * 25.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_input() {
        let p = [1, 2];
        assert!(actual_vs_expected(&p, &[1.0], &[1.0, 1.0], None, Family::Poisson, 1.0).is_err());
        assert!(
            actual_vs_expected(&p, &[1.0, 1.0], &[1.0, 1.0], None, Family::Poisson, 0.0).is_err()
        );
        let two =
            actual_vs_expected(&p, &[1.0, 1.0], &[1.0, 1.0], None, Family::Poisson, 1.0).unwrap();
        assert!(two.trend.is_nan());
    }
}
