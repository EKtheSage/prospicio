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
}
