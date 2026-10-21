//! A results table's statistics in one call: moments, percentiles, TVaRs
//! and mean ± k standard deviations under named conventions.
//!
//! Simulation pricing models report every cell of a results table with
//! their own quantile and TVaR conventions, often numpy's linear quantile
//! (R `type = 7`) and the mean of that quantile function over the tail.
//! [`StatisticsGrid`] states the convention once and returns each cell
//! under a stable name (`"Percentile 99.5"`, `"TVaR 99"`, `"Mean + 2 SD"`)
//! that a caller can join on; [`StatisticsGrid::components`] does the same
//! for every component of a [`PredictiveDistribution`].

use prospicio_core::{Error, Result};

use crate::distribution::Distribution;
use crate::predictive::{ComponentKey, PredictiveDistribution};
use crate::risk::{tvar_linear_sorted, tvar_sorted, var_linear_sorted, var_sorted};
use crate::sampled::{Empirical, Sampled};

/// How a quantile of draws is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantileMethod {
    /// The inverse of the empirical distribution function, the smallest
    /// draw `x` with `P(X <= x) >= p` (R `type = 1`, numpy
    /// `"inverted_cdf"`): [`var_sorted`]. `Sampled`'s own convention.
    #[default]
    InvertedCdf,
    /// Linear interpolation of the order statistics (R `type = 7`, numpy's
    /// default `"linear"`): [`var_linear_sorted`].
    Linear,
}

/// How a tail value at risk of draws is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TvarMethod {
    /// The exact TVaR of the empirical distribution, mass `1/n` on each
    /// draw: [`tvar_sorted`]. `Sampled`'s own convention.
    #[default]
    Exact,
    /// The mean of the linear quantile function over `[p, 1]`:
    /// [`tvar_linear_sorted`].
    Linear,
}

impl QuantileMethod {
    /// The quantile of ascending `sorted` at `p` under this method.
    pub fn quantile(self, sorted: &[f64], p: f64) -> Result<f64> {
        match self {
            QuantileMethod::InvertedCdf => var_sorted(sorted, p),
            QuantileMethod::Linear => var_linear_sorted(sorted, p),
        }
    }
}

impl TvarMethod {
    /// The TVaR of ascending `sorted` at `p` under this method.
    pub fn tvar(self, sorted: &[f64], p: f64) -> Result<f64> {
        match self {
            TvarMethod::Exact => tvar_sorted(sorted, p),
            TvarMethod::Linear => tvar_linear_sorted(sorted, p),
        }
    }
}

/// Named statistics of one set of draws, in the grid's order.
#[derive(Debug, Clone, PartialEq)]
pub struct Statistics {
    /// Stable names: `"Mean"`, `"SD"`, `"CoV"`, `"Min"`, `"Max"`,
    /// `"Variance"`, `"Percentile <q>"`, `"TVaR <q>"`, `"Mean + <k> SD"`
    /// and `"Mean - <k> SD"`, with `q` in percent as given.
    pub names: Vec<String>,
    /// The values, one per name.
    pub values: Vec<f64>,
}

impl Statistics {
    /// The value named `name`, if the grid has it.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.values[i])
    }
}

/// Which statistics a results table reports, and under which conventions.
///
/// Every grid reports the mean, the standard deviation (population, dividing
/// by `n`), the coefficient of variation (`sd / mean`; 0 when both are 0,
/// NaN when only the mean is), the minimum, the maximum and the variance,
/// then the percentiles, the TVaRs and mean ± k SD it was given. Levels are
/// in percent, so `99.5` is the 99.5th percentile and names it.
///
/// ```
/// use prospicio_prob::statistics::{QuantileMethod, StatisticsGrid, TvarMethod};
/// use prospicio_prob::Sampled;
///
/// let s = Sampled::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let grid = StatisticsGrid::new(vec![50.0, 99.0])
///     .unwrap()
///     .tvar_levels(vec![50.0])
///     .unwrap()
///     .quantile_method(QuantileMethod::Linear)
///     .tvar_method(TvarMethod::Linear);
/// let stats = grid.of(&s);
/// assert_eq!(stats.get("Percentile 50"), Some(2.5));
/// assert_eq!(stats.get("TVaR 50"), Some(3.25));
/// assert_eq!(stats.get("Mean + 1 SD"), Some(2.5 + 1.25f64.sqrt()));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct StatisticsGrid {
    percentiles: Vec<f64>,
    tvar_levels: Vec<f64>,
    sd_multiples: Vec<f64>,
    quantile: QuantileMethod,
    tvar: TvarMethod,
}

/// TVaR levels reported by default, in percent.
pub const DEFAULT_TVAR_LEVELS: [f64; 6] = [99.0, 99.5, 99.6, 99.8, 99.9, 99.99];

/// Multiples of the standard deviation reported by default.
pub const DEFAULT_SD_MULTIPLES: [f64; 3] = [1.0, 2.0, 3.0];

fn check_levels(levels: &[f64], name: &'static str) -> Result<()> {
    if let Some(&bad) = levels.iter().find(|&&q| !(0.0..=100.0).contains(&q)) {
        return Err(Error::InvalidParameter {
            name,
            value: bad,
            reason: "must be percentages in [0, 100]",
        });
    }
    Ok(())
}

impl StatisticsGrid {
    /// A grid with these percentiles (in percent), the default TVaR levels
    /// ([`DEFAULT_TVAR_LEVELS`]) and SD multiples ([`DEFAULT_SD_MULTIPLES`]),
    /// and `Sampled`'s conventions (inverted cdf, exact TVaR).
    ///
    /// Fails if a percentile is outside `[0, 100]`.
    pub fn new(percentiles: Vec<f64>) -> Result<Self> {
        check_levels(&percentiles, "percentiles")?;
        Ok(Self {
            percentiles,
            tvar_levels: DEFAULT_TVAR_LEVELS.to_vec(),
            sd_multiples: DEFAULT_SD_MULTIPLES.to_vec(),
            quantile: QuantileMethod::default(),
            tvar: TvarMethod::default(),
        })
    }

    /// The TVaR levels, in percent. Fails if one is outside `[0, 100]`.
    pub fn tvar_levels(mut self, levels: Vec<f64>) -> Result<Self> {
        check_levels(&levels, "tvar_levels")?;
        self.tvar_levels = levels;
        Ok(self)
    }

    /// The multiples `k` of mean ± k SD. Fails if one is not finite.
    pub fn sd_multiples(mut self, multiples: Vec<f64>) -> Result<Self> {
        if let Some(&bad) = multiples.iter().find(|k| !k.is_finite()) {
            return Err(Error::InvalidParameter {
                name: "sd_multiples",
                value: bad,
                reason: "must be finite",
            });
        }
        self.sd_multiples = multiples;
        Ok(self)
    }

    /// How percentiles are read.
    pub fn quantile_method(mut self, method: QuantileMethod) -> Self {
        self.quantile = method;
        self
    }

    /// How TVaRs are read.
    pub fn tvar_method(mut self, method: TvarMethod) -> Self {
        self.tvar = method;
        self
    }

    /// The names of the grid's cells, in order.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = ["Mean", "SD", "CoV", "Min", "Max", "Variance"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        names.extend(self.percentiles.iter().map(|q| format!("Percentile {q}")));
        names.extend(self.tvar_levels.iter().map(|q| format!("TVaR {q}")));
        for k in &self.sd_multiples {
            names.push(format!("Mean + {k} SD"));
            names.push(format!("Mean - {k} SD"));
        }
        names
    }

    /// The grid's statistics of `draws`.
    pub fn of(&self, draws: &Sampled) -> Statistics {
        let sorted = draws.sorted();
        let mean = draws.mean();
        let variance = draws.variance();
        let sd = variance.sqrt();
        let cov = if mean == 0.0 {
            if sd == 0.0 { 0.0 } else { f64::NAN }
        } else {
            sd / mean
        };
        let mut values = vec![mean, sd, cov, sorted[0], sorted[sorted.len() - 1], variance];
        // Levels were checked in [0, 100], so the estimators cannot fail.
        values.extend(self.percentiles.iter().map(|q| {
            self.quantile
                .quantile(sorted, q / 100.0)
                .unwrap_or(f64::NAN)
        }));
        values.extend(
            self.tvar_levels
                .iter()
                .map(|q| self.tvar.tvar(sorted, q / 100.0).unwrap_or(f64::NAN)),
        );
        for k in &self.sd_multiples {
            values.push(mean + k * sd);
            values.push(mean - k * sd);
        }
        Statistics {
            names: self.names(),
            values,
        }
    }

    /// The grid's statistics of each component of `pd`, in component order,
    /// with each component's key.
    pub fn components(&self, pd: &PredictiveDistribution) -> Vec<(ComponentKey, Statistics)> {
        pd.components()
            .iter()
            .map(|key| {
                // Every listed key has a marginal.
                let m = pd.marginal(key).expect("listed component");
                (key.clone(), self.of(&m))
            })
            .collect()
    }
}

/// Which simulation a driver's percentile picks for
/// [`StatisticsGrid::co_statistics`]: with `o` the driver's stable ascending
/// order over `n` simulations and `h = q (n - 1)`, the simulation `o[i]`
/// with `i` one of these roundings of `h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndexRule {
    /// The nearest index, halves to the even one, as Python's `round` and
    /// numpy's `round` do (`h = 2.5` picks 2, `3.5` picks 4); Rust's
    /// `f64::round` would take halves away from zero.
    #[default]
    RoundHalfEven,
    /// `floor(h)`.
    Floor,
    /// `ceil(h)`.
    Ceil,
}

impl IndexRule {
    fn index(self, h: f64) -> usize {
        (match self {
            IndexRule::RoundHalfEven => h.round_ties_even(),
            IndexRule::Floor => h.floor(),
            IndexRule::Ceil => h.ceil(),
        }) as usize
    }
}

impl StatisticsGrid {
    /// The names of [`co_statistics`](Self::co_statistics)' cells: `"Mean"`,
    /// then the grid's `"Percentile <q>"` and `"TVaR <q>"`. Moments and
    /// mean ± k SD have no reading at another variable's rank.
    pub fn co_names(&self) -> Vec<String> {
        let mut names = vec!["Mean".to_string()];
        names.extend(self.percentiles.iter().map(|q| format!("Percentile {q}")));
        names.extend(self.tvar_levels.iter().map(|q| format!("TVaR {q}")));
        names
    }

    /// Each item read at the driver's mean, percentiles and TVaRs: the
    /// item's value on the simulation the driver's statistic selects, as a
    /// results table reports a commission or a margin "at" each percentile
    /// of the loss ratio. With `o` the driver's stable ascending order (ties
    /// keep simulation order) and `h = q (n - 1)` for a level `q` in
    /// percent over 100:
    ///
    /// - `"Mean"`: the item on the first `o[i]` whose driver is at or above
    ///   the driver's mean;
    /// - `"Percentile <q>"`: the item on `o[i]`, `i` from `h` by `rule`;
    /// - `"TVaR <q>"`: the item's mean over `o[ceil(h)..]`, whatever `rule`.
    ///
    /// These are selections by rank, not the grid's quantile and TVaR
    /// methods, which do not apply to a variable read at another's rank.
    /// Items and driver must have the same length, at least 1, and be
    /// finite.
    ///
    /// ```
    /// use prospicio_prob::statistics::{IndexRule, StatisticsGrid};
    ///
    /// let ulr = [0.6, 0.9, 0.5, 0.7, 0.8];
    /// let margin = [0.30, 0.00, 0.40, 0.20, 0.10];
    /// let grid = StatisticsGrid::new(vec![62.5]).unwrap().tvar_levels(vec![62.5]).unwrap();
    /// // h = 0.625 * 4 = 2.5: halves to even pick the third smallest ULR, 0.7.
    /// let co = grid.co_statistics(&ulr, &[&margin], IndexRule::RoundHalfEven).unwrap();
    /// assert_eq!(co[0].get("Percentile 62.5"), Some(0.20));
    /// // The TVaR takes o[3..]: ULRs 0.8 and 0.9.
    /// assert!((co[0].get("TVaR 62.5").unwrap() - 0.05).abs() < 1e-12);
    /// // The mean ULR is 0.7, so the mean reads the margin at 0.7.
    /// assert_eq!(co[0].get("Mean"), Some(0.20));
    /// ```
    pub fn co_statistics(
        &self,
        driver: &[f64],
        items: &[&[f64]],
        rule: IndexRule,
    ) -> Result<Vec<Statistics>> {
        let n = driver.len();
        if n == 0 {
            return Err(Error::Data("the driver has no simulations".into()));
        }
        if let Some(&bad) = driver.iter().find(|x| !x.is_finite()) {
            return Err(Error::InvalidParameter {
                name: "driver",
                value: bad,
                reason: "must be finite",
            });
        }
        for item in items {
            if item.len() != n {
                return Err(Error::Data(format!(
                    "an item has {} simulations and the driver {n}",
                    item.len()
                )));
            }
            if let Some(&bad) = item.iter().find(|x| !x.is_finite()) {
                return Err(Error::InvalidParameter {
                    name: "items",
                    value: bad,
                    reason: "must be finite",
                });
            }
        }
        // Stable: ties keep simulation order.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| driver[a].total_cmp(&driver[b]));
        // The mean of equal values can round above them: capped at the
        // largest, it picks the first of those.
        let mean = (driver.iter().sum::<f64>() / n as f64).min(driver[order[n - 1]]);
        let at_mean = order
            .iter()
            .position(|&i| driver[i] >= mean)
            .expect("the mean is at most the largest value");
        let last = (n - 1) as f64;
        let picks: Vec<usize> = self
            .percentiles
            .iter()
            .map(|q| rule.index(q / 100.0 * last))
            .collect();
        let tails: Vec<usize> = self
            .tvar_levels
            .iter()
            .map(|q| (q / 100.0 * last).ceil() as usize)
            .collect();
        let names = self.co_names();
        Ok(items
            .iter()
            .map(|item| {
                let mut values = vec![item[order[at_mean]]];
                values.extend(picks.iter().map(|&i| item[order[i]]));
                values.extend(tails.iter().map(|&start| {
                    let tail = &order[start..];
                    tail.iter().map(|&i| item[i]).sum::<f64>() / tail.len() as f64
                }));
                Statistics {
                    names: names.clone(),
                    values,
                }
            })
            .collect())
    }

    /// [`co_statistics`](Self::co_statistics) of every component of `pd`
    /// (the driver among them) read at the ranks of the component `driver`,
    /// or of the total when `driver` is `None`, in component order with each
    /// component's key.
    pub fn co_components(
        &self,
        pd: &PredictiveDistribution,
        driver: Option<&ComponentKey>,
        rule: IndexRule,
    ) -> Result<Vec<(ComponentKey, Statistics)>> {
        let driver: Vec<f64> = match driver {
            None => pd.total().draws().to_vec(),
            Some(key) => pd
                .marginal(key)
                .ok_or_else(|| Error::Data(format!("no component {key:?}")))?
                .draws()
                .to_vec(),
        };
        let columns: Vec<Vec<f64>> = pd
            .components()
            .iter()
            .map(|key| pd.marginal(key).expect("listed component").draws().to_vec())
            .collect();
        let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
        let stats = self.co_statistics(&driver, &refs, rule)?;
        Ok(pd.components().iter().cloned().zip(stats).collect())
    }
}

/// The downside risk in a [`RiskRatios`] Sortino ratio's denominator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Sortino {
    /// The downside deviation about the threshold `t`,
    /// `sqrt(mean(min(x - t, 0)^2))` over all `n` simulations: the
    /// textbook ratio (Sortino and Price 1994).
    #[default]
    DownsideDeviation,
    /// The population standard deviation of the values below `t` only, as
    /// a reference simulation pricing model computes it: not a downside
    /// deviation, since it measures the spread of the losses about their
    /// own mean. NaN when no value is below `t`.
    BelowThresholdSd,
}

/// The [`RiskRatios`] Omega ratio's convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Omega {
    /// The ratio of partial moments about `t`,
    /// `mean(max(x - t, 0)) / mean(max(t - x, 0))` (Keating and Shadwick
    /// 2002).
    #[default]
    PartialMoments,
    /// The ratio of counts, `#(x > t) / #(x < t)`; values equal to `t`
    /// count in neither.
    Count,
}

/// Risk ratios of a per-simulation result such as a margin, where large
/// is good: Sharpe, Sortino, Omega and the TVaR ratio, about a threshold
/// `t` (0 by default) and under named conventions.
///
/// - `"Sharpe"`: `(mean - t) / sd`, the population standard deviation.
/// - `"Sortino"`: `(mean - t)` over the downside risk of [`Sortino`].
/// - `"Omega"`: [`Omega`]'s ratio.
/// - `"TVaR ratio <q>"`: `(mean - t) / TVaR_q(-x)`, the TVaR of the result
///   read as a loss at level `q` in percent, under the [`TvarMethod`] given
///   (STARR, the stable tail adjusted return ratio). A negative
///   denominator, the worst outcomes still above zero, gives a negative
///   ratio.
///
/// A zero denominator gives `±∞`, or NaN when the numerator is 0 too.
///
/// ```
/// use prospicio_prob::statistics::{Omega, RiskRatios, Sortino};
/// use prospicio_prob::Sampled;
///
/// let margin = Sampled::new(vec![-2.0, -1.0, 1.0, 2.0, 5.0]).unwrap();
/// let r = RiskRatios::new().of(&margin);
/// // Mean 1; partial moments 8/5 above 0 and 3/5 below.
/// assert!((r.get("Omega").unwrap() - 8.0 / 3.0).abs() < 1e-12);
/// // Downside deviation sqrt((4 + 1) / 5) = 1.
/// assert!((r.get("Sortino").unwrap() - 1.0).abs() < 1e-12);
/// let reference = RiskRatios::new()
///     .sortino(Sortino::BelowThresholdSd)
///     .omega(Omega::Count)
///     .of(&margin);
/// // The SD of -2 and -1 is 0.5; three values above 0 against two below.
/// assert!((reference.get("Sortino").unwrap() - 2.0).abs() < 1e-12);
/// assert_eq!(reference.get("Omega"), Some(1.5));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RiskRatios {
    threshold: f64,
    sortino: Sortino,
    omega: Omega,
    tvar_levels: Vec<f64>,
    tvar: TvarMethod,
}

impl Default for RiskRatios {
    fn default() -> Self {
        Self::new()
    }
}

impl RiskRatios {
    /// Threshold 0, the textbook Sortino and Omega, the TVaR ratio at 99%
    /// with the exact TVaR.
    pub fn new() -> Self {
        Self {
            threshold: 0.0,
            sortino: Sortino::default(),
            omega: Omega::default(),
            tvar_levels: vec![99.0],
            tvar: TvarMethod::default(),
        }
    }

    /// The threshold `t`. Fails if it is not finite.
    pub fn threshold(mut self, t: f64) -> Result<Self> {
        if !t.is_finite() {
            return Err(Error::InvalidParameter {
                name: "threshold",
                value: t,
                reason: "must be finite",
            });
        }
        self.threshold = t;
        Ok(self)
    }

    /// The Sortino ratio's downside risk.
    pub fn sortino(mut self, sortino: Sortino) -> Self {
        self.sortino = sortino;
        self
    }

    /// The Omega ratio's convention.
    pub fn omega(mut self, omega: Omega) -> Self {
        self.omega = omega;
        self
    }

    /// The TVaR ratio's levels, in percent. Fails if one is outside
    /// `[0, 100]`.
    pub fn tvar_levels(mut self, levels: Vec<f64>) -> Result<Self> {
        check_levels(&levels, "tvar_levels")?;
        self.tvar_levels = levels;
        Ok(self)
    }

    /// How the TVaR ratio's TVaR is read.
    pub fn tvar_method(mut self, method: TvarMethod) -> Self {
        self.tvar = method;
        self
    }

    /// The names of the ratios, in order.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = ["Sharpe", "Sortino", "Omega"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        names.extend(self.tvar_levels.iter().map(|q| format!("TVaR ratio {q}")));
        names
    }

    /// The ratios of `draws`.
    pub fn of(&self, draws: &Sampled) -> Statistics {
        let x = draws.draws();
        let n = x.len() as f64;
        let t = self.threshold;
        let excess = draws.mean() - t;
        let sharpe = excess / draws.variance().sqrt();
        let downside = match self.sortino {
            Sortino::DownsideDeviation => {
                (x.iter().map(|v| (v - t).min(0.0).powi(2)).sum::<f64>() / n).sqrt()
            }
            Sortino::BelowThresholdSd => {
                let below: Vec<f64> = x.iter().copied().filter(|&v| v < t).collect();
                if below.is_empty() {
                    f64::NAN
                } else {
                    let k = below.len() as f64;
                    let m = below.iter().sum::<f64>() / k;
                    (below.iter().map(|v| (v - m).powi(2)).sum::<f64>() / k).sqrt()
                }
            }
        };
        let omega = match self.omega {
            Omega::PartialMoments => {
                let up: f64 = x.iter().map(|v| (v - t).max(0.0)).sum();
                let down: f64 = x.iter().map(|v| (t - v).max(0.0)).sum();
                up / down
            }
            Omega::Count => {
                let up = x.iter().filter(|&&v| v > t).count() as f64;
                let down = x.iter().filter(|&&v| v < t).count() as f64;
                up / down
            }
        };
        let mut losses: Vec<f64> = x.iter().map(|v| -v).collect();
        losses.sort_by(f64::total_cmp);
        let mut values = vec![sharpe, excess / downside, omega];
        // Levels were checked in [0, 100], so the TVaR cannot fail.
        values.extend(
            self.tvar_levels
                .iter()
                .map(|q| excess / self.tvar.tvar(&losses, q / 100.0).unwrap_or(f64::NAN)),
        );
        Statistics {
            names: self.names(),
            values,
        }
    }

    /// The ratios of each component of `pd`, in component order, with each
    /// component's key.
    pub fn components(&self, pd: &PredictiveDistribution) -> Vec<(ComponentKey, Statistics)> {
        pd.components()
            .iter()
            .map(|key| {
                let m = pd.marginal(key).expect("listed component");
                (key.clone(), self.of(&m))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// numpy 2.5.3: `np.quantile(x, q, method="linear")` and the integral
    /// TVaR, on `x = [0, 0, 1, 3, 7, 7, 12, 20, 45, 100]`.
    #[test]
    fn matches_numpy_on_ties_and_a_long_tail() {
        let s = Sampled::new(vec![7.0, 0.0, 100.0, 3.0, 1.0, 0.0, 45.0, 7.0, 12.0, 20.0]).unwrap();
        let grid = StatisticsGrid::new(vec![0.0, 10.0, 25.0, 50.0, 90.0, 99.0, 100.0])
            .unwrap()
            .tvar_levels(vec![0.0, 50.0, 90.0, 99.0, 100.0])
            .unwrap()
            .quantile_method(QuantileMethod::Linear)
            .tvar_method(TvarMethod::Linear);
        let st = grid.of(&s);
        let want = [
            ("Percentile 0", 0.0),
            ("Percentile 10", 0.0),
            ("Percentile 25", 1.5),
            ("Percentile 50", 7.0),
            ("Percentile 90", 50.5),
            ("Percentile 99", 95.05),
            ("Percentile 100", 100.0),
        ];
        for (name, v) in want {
            let got = st.get(name).unwrap();
            assert!((got - v).abs() < 1e-12, "{name}: {got} against {v}");
        }
        // The linear quantile function's mean over [0, 1] is the trapezoid
        // rule on the order statistics.
        assert!((st.get("TVaR 0").unwrap() - 145.0 / 9.0).abs() < 1e-12);
        assert!((st.get("TVaR 50").unwrap() - 29.77777777777778).abs() < 1e-12);
        assert!((st.get("TVaR 90").unwrap() - 75.25).abs() < 1e-12);
        assert!((st.get("TVaR 99").unwrap() - 97.525).abs() < 1e-12);
        assert_eq!(st.get("TVaR 100"), Some(100.0));
        assert_eq!(st.get("Mean"), Some(19.5));
        assert_eq!(st.get("Min"), Some(0.0));
        assert_eq!(st.get("Max"), Some(100.0));
    }

    #[test]
    fn default_conventions_are_sampleds() {
        let s = Sampled::new((1..=20).map(f64::from).collect()).unwrap();
        let st = StatisticsGrid::new(vec![95.0]).unwrap().of(&s);
        assert_eq!(st.get("Percentile 95"), Some(s.var(0.95).unwrap()));
        assert_eq!(st.get("TVaR 99.5"), Some(s.tvar(0.995).unwrap()));
        assert_eq!(st.get("SD"), Some(s.std_dev()));
        assert_eq!(st.names.len(), 6 + 1 + 6 + 6);
        assert_eq!(st.names[st.names.len() - 1], "Mean - 3 SD");
    }

    #[test]
    fn coefficient_of_variation_at_a_zero_mean() {
        let grid = StatisticsGrid::new(vec![]).unwrap();
        let zero = Sampled::new(vec![0.0, 0.0]).unwrap();
        assert_eq!(grid.of(&zero).get("CoV"), Some(0.0));
        let centred = Sampled::new(vec![-1.0, 1.0]).unwrap();
        assert!(grid.of(&centred).get("CoV").unwrap().is_nan());
        let one = Sampled::new(vec![2.0, 4.0]).unwrap();
        assert_eq!(grid.of(&one).get("CoV"), Some(1.0 / 3.0));
    }

    #[test]
    fn levels_outside_a_percentage_are_refused() {
        assert!(StatisticsGrid::new(vec![101.0]).is_err());
        assert!(
            StatisticsGrid::new(vec![50.0])
                .unwrap()
                .tvar_levels(vec![-1.0])
                .is_err()
        );
        assert!(
            StatisticsGrid::new(vec![50.0])
                .unwrap()
                .sd_multiples(vec![f64::NAN])
                .is_err()
        );
    }

    #[test]
    fn a_single_draw_is_every_statistic() {
        let s = Sampled::new(vec![5.0]).unwrap();
        let grid = StatisticsGrid::new(vec![0.0, 50.0, 100.0])
            .unwrap()
            .quantile_method(QuantileMethod::Linear)
            .tvar_method(TvarMethod::Linear);
        let st = grid.of(&s);
        for name in [
            "Percentile 0",
            "Percentile 100",
            "TVaR 99",
            "TVaR 99.99",
            "Mean + 3 SD",
        ] {
            assert_eq!(st.get(name), Some(5.0), "{name}");
        }
    }

    #[test]
    fn co_statistics_pick_by_the_drivers_rank_under_each_rule() {
        // Driver 0..=10 reversed, so simulation i holds rank 10 - i.
        let driver: Vec<f64> = (0..11).map(|i| (10 - i) as f64).collect();
        let item: Vec<f64> = (0..11).map(|i| 100.0 + i as f64).collect();
        // h = 0.25 * 10 = 2.5 and 0.75 * 10 = 7.5: halves to even give 2
        // and 8, floor 2 and 7, ceil 3 and 8.
        let grid = StatisticsGrid::new(vec![25.0, 75.0])
            .unwrap()
            .tvar_levels(vec![50.0, 0.0, 100.0])
            .unwrap();
        let rank_item = |r: usize| 100.0 + (10 - r) as f64;
        for (rule, a, b) in [
            (IndexRule::RoundHalfEven, 2, 8),
            (IndexRule::Floor, 2, 7),
            (IndexRule::Ceil, 3, 8),
        ] {
            let co = &grid.co_statistics(&driver, &[&item], rule).unwrap()[0];
            assert_eq!(co.names, grid.co_names());
            assert_eq!(co.get("Percentile 25"), Some(rank_item(a)), "{rule:?}");
            assert_eq!(co.get("Percentile 75"), Some(rank_item(b)), "{rule:?}");
            // TVaR 50 averages ranks 5..=10 whatever the rule.
            let want = (5..11).map(rank_item).sum::<f64>() / 6.0;
            assert_eq!(co.get("TVaR 50"), Some(want));
            assert_eq!(co.get("TVaR 0"), Some(item.iter().sum::<f64>() / 11.0));
            assert_eq!(co.get("TVaR 100"), Some(rank_item(10)));
            // Mean 5: the first rank at or above it is 5.
            assert_eq!(co.get("Mean"), Some(rank_item(5)));
        }
        // Ties keep simulation order: three zeros, then 1.
        let tied = [0.0, 1.0, 0.0, 0.0];
        let ids = [10.0, 11.0, 12.0, 13.0];
        let g = StatisticsGrid::new(vec![0.0, 100.0 / 3.0, 200.0 / 3.0]).unwrap();
        let co = &g.co_statistics(&tied, &[&ids], IndexRule::Floor).unwrap()[0];
        assert_eq!(&co.values[1..4], &[10.0, 12.0, 13.0]);
        // A constant driver: the mean is the first simulation.
        let flat = [0.1; 3];
        let co = &g
            .co_statistics(&flat, &[&[1.0, 2.0, 3.0]], IndexRule::Floor)
            .unwrap()[0];
        assert_eq!(co.get("Mean"), Some(1.0));

        assert!(g.co_statistics(&[], &[], IndexRule::Floor).is_err());
        assert!(
            g.co_statistics(&[1.0, 2.0], &[&[1.0]], IndexRule::Floor)
                .is_err()
        );
        assert!(
            g.co_statistics(&[1.0, f64::NAN], &[], IndexRule::Floor)
                .is_err()
        );
    }

    #[test]
    fn co_components_read_at_the_total_approach_the_co_tvar() {
        use crate::{Distortion, KeyValue, Provenance};
        let n = 1001;
        let draws: Vec<f64> = (0..n)
            .flat_map(|i| [((i * 7919) % n) as f64, ((i * 104_729) % n) as f64 * 0.5])
            .collect();
        let pd = PredictiveDistribution::from_draws(
            vec!["lob".into()],
            vec![vec![KeyValue::from("a")], vec![KeyValue::from("b")]],
            draws,
            Provenance::new("test"),
        )
        .unwrap();
        let grid = StatisticsGrid::new(vec![50.0])
            .unwrap()
            .tvar_levels(vec![90.0])
            .unwrap();
        let co = grid
            .co_components(&pd, None, IndexRule::RoundHalfEven)
            .unwrap();
        let alloc = pd.allocate(&Distortion::tvar(0.9).unwrap());
        for ((_, st), want) in co.iter().zip(&alloc) {
            let got = st.get("TVaR 90").unwrap();
            assert!((got - want).abs() < 0.01 * want, "{got} against {want}");
        }
        // Read at component a's ranks, a's own percentile is its order
        // statistic.
        let a = vec![KeyValue::from("a")];
        let own = grid
            .co_components(&pd, Some(&a), IndexRule::RoundHalfEven)
            .unwrap();
        assert_eq!(own[0].1.get("Percentile 50"), Some(500.0));
        assert!(
            grid.co_components(&pd, Some(&vec![KeyValue::from("c")]), IndexRule::Floor)
                .is_err()
        );
    }

    #[test]
    fn risk_ratios_under_each_convention_and_at_the_edges() {
        let x = Sampled::new(vec![-3.0, -1.0, 0.0, 2.0, 4.0, 4.0]).unwrap();
        // Mean 1, population SD sqrt(46 / 6 - 1).
        let r = RiskRatios::new()
            .tvar_levels(vec![50.0, 100.0])
            .unwrap()
            .of(&x);
        assert_eq!(
            r.names,
            [
                "Sharpe",
                "Sortino",
                "Omega",
                "TVaR ratio 50",
                "TVaR ratio 100"
            ]
        );
        assert!((r.get("Sharpe").unwrap() - 1.0 / (46.0f64 / 6.0 - 1.0).sqrt()).abs() < 1e-12);
        assert!((r.get("Sortino").unwrap() - 1.0 / (10.0f64 / 6.0).sqrt()).abs() < 1e-12);
        assert!((r.get("Omega").unwrap() - 10.0 / 4.0).abs() < 1e-12);
        // The worst half as losses: 3, 1, 0, mean 4/3; the worst one: 3.
        assert!((r.get("TVaR ratio 50").unwrap() - 0.75).abs() < 1e-12);
        assert!((r.get("TVaR ratio 100").unwrap() - 1.0 / 3.0).abs() < 1e-12);

        // The reference's conventions; the 0 counts on neither side.
        let reference = RiskRatios::new()
            .sortino(Sortino::BelowThresholdSd)
            .omega(Omega::Count)
            .of(&x);
        assert!((reference.get("Sortino").unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(reference.get("Omega"), Some(1.5));

        // A threshold moves every numerator and the downside.
        let t = RiskRatios::new().threshold(2.0).unwrap().of(&x);
        let dd = ((25.0 + 9.0 + 4.0) / 6.0f64).sqrt();
        assert!((t.get("Sortino").unwrap() + 1.0 / dd).abs() < 1e-12);
        assert!((t.get("Omega").unwrap() - 4.0 / 10.0).abs() < 1e-12);

        // Nothing below the threshold.
        let up = Sampled::new(vec![1.0, 2.0, 3.0]).unwrap();
        let r = RiskRatios::new().of(&up);
        assert_eq!(r.get("Sortino"), Some(f64::INFINITY));
        assert_eq!(r.get("Omega"), Some(f64::INFINITY));
        assert!(r.get("TVaR ratio 99").unwrap() < 0.0);
        let r = RiskRatios::new().sortino(Sortino::BelowThresholdSd).of(&up);
        assert!(r.get("Sortino").unwrap().is_nan());
        // A constant result: no spread.
        let flat = RiskRatios::new().of(&Sampled::new(vec![0.0, 0.0]).unwrap());
        assert!(flat.get("Sharpe").unwrap().is_nan());
        assert!(flat.get("Omega").unwrap().is_nan());
        assert!(RiskRatios::new().threshold(f64::NAN).is_err());
        assert!(RiskRatios::new().tvar_levels(vec![101.0]).is_err());
    }
}
