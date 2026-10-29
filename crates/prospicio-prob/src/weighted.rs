//! Weighted draws: the estimators of an importance sample.
//!
//! An importance sample draws from a proposal `q` and weights each draw by
//! the likelihood ratio `w = p / q`, so that `E_p[h(X)] = E_q[w h(X)]`.
//! [`WeightedSample`] holds the draws and their weights and gives the
//! self-normalised estimators: the mean `Σ w x / Σ w`, and the VaR and TVaR
//! of the distribution that puts mass `w_i / Σ w` on draw `i`
//! ([`risk::var_weighted`] and [`risk::tvar_weighted`]). Standard errors
//! are by batch means: the draws split, in the order given, into `batches`
//! equal runs, each estimated alone, and the spread of those estimates
//! over `√batches`. The effective sample size `(Σ w)² / Σ w²` says how many
//! plain draws the weighted ones are worth for the mean.
//!
//! Equal weights give the plain empirical estimators of [`crate::Sampled`]
//! under the inverted-cdf convention.

use prospicio_core::{Error, Result};

use crate::risk;

/// Draws with importance weights.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedSample {
    values: Vec<f64>,
    weights: Vec<f64>,
    /// The values ascending and their normalised weights, for the
    /// estimators.
    sorted: Vec<f64>,
    probs: Vec<f64>,
}

impl WeightedSample {
    /// Draws `values` with weights `weights`, one each, finite and the
    /// weights non-negative with a positive sum.
    ///
    /// ```
    /// use prospicio_prob::weighted::WeightedSample;
    ///
    /// let s = WeightedSample::new(vec![1.0, 2.0, 3.0], vec![1.0, 1.0, 2.0]).unwrap();
    /// assert_eq!(s.mean(), 2.25);
    /// assert_eq!(s.var(0.5).unwrap(), 2.0);
    /// assert_eq!(s.tvar(0.5).unwrap(), 3.0);
    /// ```
    pub fn new(values: Vec<f64>, weights: Vec<f64>) -> Result<Self> {
        if values.is_empty() || values.len() != weights.len() {
            return Err(Error::Data(format!(
                "{} values and {} weights; give one weight a value",
                values.len(),
                weights.len()
            )));
        }
        if let Some(&x) = values.iter().find(|x| !x.is_finite()) {
            return Err(Error::InvalidParameter {
                name: "values",
                value: x,
                reason: "must be finite",
            });
        }
        if let Some(&w) = weights.iter().find(|w| !(w.is_finite() && **w >= 0.0)) {
            return Err(Error::InvalidParameter {
                name: "weights",
                value: w,
                reason: "must be finite and non-negative",
            });
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(Error::Data("the weights sum to zero".into()));
        }
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        let sorted = order.iter().map(|&i| values[i]).collect();
        let probs = order.iter().map(|&i| weights[i] / total).collect();
        Ok(WeightedSample {
            values,
            weights,
            sorted,
            probs,
        })
    }

    /// The draws, in the order given.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The weights, in the order given.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The mean of the weights; about 1 for likelihood ratios.
    pub fn mean_weight(&self) -> f64 {
        self.weights.iter().sum::<f64>() / self.weights.len() as f64
    }

    /// `(Σ w)² / Σ w²`.
    pub fn effective_sample_size(&self) -> f64 {
        let s: f64 = self.weights.iter().sum();
        let s2: f64 = self.weights.iter().map(|w| w * w).sum();
        s * s / s2
    }

    /// The self-normalised mean, `Σ w x / Σ w`.
    pub fn mean(&self) -> f64 {
        self.sorted
            .iter()
            .zip(&self.probs)
            .map(|(x, p)| x * p)
            .sum()
    }

    /// The VaR at `p`: the smallest draw whose normalised weight from the
    /// bottom reaches `p`.
    pub fn var(&self, p: f64) -> Result<f64> {
        risk::var_weighted(&self.sorted, &self.probs, p)
    }

    /// The TVaR at `p` of the weighted draws.
    pub fn tvar(&self, p: f64) -> Result<f64> {
        risk::tvar_weighted(&self.sorted, &self.probs, p)
    }

    /// The standard error of [`var`](Self::var) at `p` by batch means over
    /// `batches` runs of the draws in order (at least 2, each with at least
    /// one draw).
    pub fn standard_error_var(&self, p: f64, batches: usize) -> Result<f64> {
        self.batch_error(batches, |s| s.var(p))
    }

    /// The standard error of [`tvar`](Self::tvar) at `p` by batch means.
    pub fn standard_error_tvar(&self, p: f64, batches: usize) -> Result<f64> {
        self.batch_error(batches, |s| s.tvar(p))
    }

    /// The standard error of [`mean`](Self::mean) by batch means.
    pub fn standard_error_mean(&self, batches: usize) -> Result<f64> {
        self.batch_error(batches, |s| Ok(s.mean()))
    }

    fn batch_error(
        &self,
        batches: usize,
        estimate: impl Fn(&WeightedSample) -> Result<f64>,
    ) -> Result<f64> {
        let n = self.values.len();
        if batches < 2 || batches > n {
            return Err(Error::InvalidParameter {
                name: "batches",
                value: batches as f64,
                reason: "must be at least 2 and at most the number of draws",
            });
        }
        let mut estimates = Vec::with_capacity(batches);
        for b in 0..batches {
            let (lo, hi) = (b * n / batches, (b + 1) * n / batches);
            let w = self.weights[lo..hi].to_vec();
            if w.iter().sum::<f64>() == 0.0 {
                // A batch whose draws all weigh nothing says nothing.
                continue;
            }
            let batch = WeightedSample::new(self.values[lo..hi].to_vec(), w)?;
            estimates.push(estimate(&batch)?);
        }
        let k = estimates.len() as f64;
        if k < 2.0 {
            return Err(Error::Data("fewer than two batches carry weight".into()));
        }
        let m = estimates.iter().sum::<f64>() / k;
        let var = estimates.iter().map(|e| (e - m).powi(2)).sum::<f64>() / (k - 1.0);
        Ok((var / k).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Distribution, Empirical, Sampled};

    #[test]
    fn equal_weights_are_the_plain_estimators() {
        let x: Vec<f64> = (0..997).map(|i| ((i * 7919) % 997) as f64).collect();
        let s = WeightedSample::new(x.clone(), vec![2.0; 997]).unwrap();
        let plain = Sampled::new(x).unwrap();
        assert!((s.mean() - plain.mean()).abs() < 1e-9);
        for p in [0.5, 0.9, 0.99] {
            assert_eq!(s.var(p).unwrap(), plain.var(p).unwrap());
            let (a, b) = (s.tvar(p).unwrap(), plain.tvar(p).unwrap());
            assert!((a - b).abs() < 1e-11 * b, "{p}: {a} {b}");
        }
        assert!((s.effective_sample_size() - 997.0).abs() < 1e-9);
        assert_eq!(s.mean_weight(), 2.0);
    }

    #[test]
    fn inputs_and_batches_are_checked() {
        assert!(WeightedSample::new(vec![], vec![]).is_err());
        assert!(WeightedSample::new(vec![1.0], vec![1.0, 2.0]).is_err());
        assert!(WeightedSample::new(vec![1.0], vec![-1.0]).is_err());
        assert!(WeightedSample::new(vec![1.0, 2.0], vec![0.0, 0.0]).is_err());
        assert!(WeightedSample::new(vec![f64::NAN], vec![1.0]).is_err());
        let s = WeightedSample::new(vec![1.0, 2.0, 3.0, 4.0], vec![1.0; 4]).unwrap();
        assert!(s.standard_error_mean(1).is_err());
        assert!(s.standard_error_mean(5).is_err());
        // Two batches of means 1.5 and 3.5: sd √2, over √2.
        assert!((s.standard_error_mean(2).unwrap() - 1.0).abs() < 1e-12);
    }
}
