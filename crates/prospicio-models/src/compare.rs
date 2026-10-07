//! One table of out-of-sample scores across models of any kind.
//!
//! Every candidate is fitted and scored on the same splits with the same
//! metrics, so the table compares like with like. A [`Candidate`] wraps a
//! [`Model`] or any fit-and-predict function, which lets a GLM, a GAM, an
//! elastic net and a neural network (each of a different Rust type) sit
//! in one comparison.

use prospicio_core::{Error, Result};

use crate::design::Design;
use crate::model::{Fitted, Model};
use crate::resample::{Split, map_splits};

/// Fits on the training design and response, then predicts the test
/// design.
type FitPredict<'a> = dyn Fn(&Design, &[f64], &Design) -> Result<Vec<f64>> + Sync + 'a;

/// A loss on the test rows: `(y_test, predicted, test_design) -> score`,
/// lower is better.
pub type Metric<'a> = dyn Fn(&[f64], &[f64], &Design) -> f64 + Sync + 'a;

/// A named model to compare.
pub struct Candidate<'a> {
    name: String,
    fit_predict: Box<FitPredict<'a>>,
}

impl<'a> Candidate<'a> {
    /// A candidate that fits `model` to each training set.
    pub fn new<M: Model + Sync + 'a>(name: impl Into<String>, model: M) -> Self {
        Self::from_fn(name, move |train, y, test| {
            model.fit(train, y)?.predict(test)
        })
    }

    /// A candidate from a function that fits on `(train, y_train)` and
    /// returns predictions for `test`: for a model that builds its own
    /// features, or one that is not a [`Model`].
    pub fn from_fn(
        name: impl Into<String>,
        f: impl Fn(&Design, &[f64], &Design) -> Result<Vec<f64>> + Sync + 'a,
    ) -> Self {
        Self {
            name: name.into(),
            fit_predict: Box::new(f),
        }
    }

    /// The candidate's name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// The scores from [`compare`]: every candidate on every metric and split.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    models: Vec<String>,
    metrics: Vec<String>,
    n_splits: usize,
    /// Model-major, then metric, then split.
    scores: Vec<f64>,
}

impl Comparison {
    /// Candidate names, in the order given.
    pub fn models(&self) -> &[String] {
        &self.models
    }

    /// Metric names, in the order given.
    pub fn metrics(&self) -> &[String] {
        &self.metrics
    }

    /// Number of splits.
    pub fn n_splits(&self) -> usize {
        self.n_splits
    }

    /// Scores of `model` on `metric`, one per split (indices into
    /// [`models`](Self::models) and [`metrics`](Self::metrics)).
    pub fn split_scores(&self, model: usize, metric: usize) -> &[f64] {
        let k = self.n_splits;
        let start = (model * self.metrics.len() + metric) * k;
        &self.scores[start..start + k]
    }

    /// Mean score over the splits.
    pub fn mean(&self, model: usize, metric: usize) -> f64 {
        let s = self.split_scores(model, metric);
        s.iter().sum::<f64>() / s.len() as f64
    }

    /// Standard error of [`mean`](Self::mean): the splits' standard
    /// deviation over `√k` (NaN with one split), as cv.glmnet and tidymodels
    /// report it.
    pub fn std_error(&self, model: usize, metric: usize) -> f64 {
        let s = self.split_scores(model, metric);
        let k = s.len() as f64;
        let m = self.mean(model, metric);
        let var = s.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (k - 1.0);
        (var / k).sqrt()
    }

    /// The candidate with the lowest mean on `metric`.
    pub fn best(&self, metric: usize) -> usize {
        (0..self.models.len())
            .min_by(|&a, &b| self.mean(a, metric).total_cmp(&self.mean(b, metric)))
            .expect("at least one model")
    }

    /// Standard error of each candidate's per-split difference from the
    /// best on `metric` (0 for the best itself). Splits are shared, so the
    /// paired difference is far less noisy than either mean: a candidate
    /// within about two of these of the best is not clearly worse.
    pub fn difference_std_error(&self, model: usize, metric: usize) -> f64 {
        let best = self.best(metric);
        let (a, b) = (
            self.split_scores(model, metric),
            self.split_scores(best, metric),
        );
        let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
        let k = d.len() as f64;
        let m = d.iter().sum::<f64>() / k;
        let var = d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (k - 1.0);
        (var / k).sqrt()
    }
}

/// Fits every candidate on each split's training rows and scores its test
/// predictions on every metric. Splits run in parallel, as in
/// [`cross_validate`](crate::resample::cross_validate); within a split the
/// candidates run in order.
///
/// ```
/// use prospicio_models::compare::{Candidate, compare};
/// use prospicio_models::resample::k_fold;
/// # use prospicio_models::{Design, Fitted, Model};
/// # struct Mean;
/// # struct MeanFit(f64);
/// # impl Model for Mean {
/// #     type Fitted = MeanFit;
/// #     fn fit(&self, _: &Design, y: &[f64]) -> prospicio_core::Result<MeanFit> {
/// #         Ok(MeanFit(y.iter().sum::<f64>() / y.len() as f64))
/// #     }
/// # }
/// # impl Fitted for MeanFit {
/// #     fn predict(&self, d: &Design) -> prospicio_core::Result<Vec<f64>> { Ok(vec![self.0; d.n_rows()]) }
/// #     fn predict_distribution(&self, _: &Design, _: usize, _: u64)
/// #         -> prospicio_core::Result<prospicio_prob::PredictiveDistribution> { unimplemented!() }
/// # }
/// use prospicio_models::metrics::rmse;
///
/// let x = Design::new(vec!["x".into()], vec![(0..8).map(f64::from).collect()]).unwrap();
/// let y: Vec<f64> = (0..8).map(|i| 2.0 * i as f64).collect();
/// let candidates = [
///     Candidate::new("mean", Mean),
///     // y = 2x exactly.
///     Candidate::from_fn("double", |_, _, test| Ok(test.column(0).iter().map(|v| 2.0 * v).collect())),
/// ];
/// let rmse = |t: &[f64], p: &[f64], _: &Design| rmse(t, p).unwrap();
/// let table = compare(&candidates, &x, &y, &k_fold(8, 4, 1).unwrap(), &[("rmse", &rmse)]).unwrap();
/// assert_eq!(table.best(0), 1);
/// assert_eq!(table.mean(1, 0), 0.0);
/// ```
pub fn compare(
    candidates: &[Candidate<'_>],
    design: &Design,
    y: &[f64],
    splits: &[Split],
    metrics: &[(&str, &Metric<'_>)],
) -> Result<Comparison> {
    if candidates.is_empty() || metrics.is_empty() || splits.is_empty() {
        return Err(Error::Data(
            "compare needs at least one candidate, metric and split".into(),
        ));
    }
    if y.len() != design.n_rows() {
        return Err(Error::Data(format!(
            "{} responses for {} design rows",
            y.len(),
            design.n_rows()
        )));
    }
    // Per split: model-major, then metric.
    let per_split = map_splits(splits, |s| {
        let train = design.select(&s.train);
        let y_train: Vec<f64> = s.train.iter().map(|&i| y[i]).collect();
        let test = design.select(&s.test);
        let y_test: Vec<f64> = s.test.iter().map(|&i| y[i]).collect();
        let mut out = Vec::with_capacity(candidates.len() * metrics.len());
        for c in candidates {
            let pred = (c.fit_predict)(&train, &y_train, &test)
                .map_err(|e| Error::Data(format!("{}: {e}", c.name)))?;
            out.extend(metrics.iter().map(|(_, m)| m(&y_test, &pred, &test)));
        }
        Ok(out)
    })?;
    let (n_models, n_metrics, k) = (candidates.len(), metrics.len(), splits.len());
    let mut scores = vec![0.0; n_models * n_metrics * k];
    for (s, row) in per_split.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            scores[j * k + s] = *v;
        }
    }
    Ok(Comparison {
        models: candidates.iter().map(|c| c.name.clone()).collect(),
        metrics: metrics.iter().map(|(n, _)| (*n).to_string()).collect(),
        n_splits: k,
        scores,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resample::k_fold;

    fn constant(c: f64) -> Candidate<'static> {
        Candidate::from_fn(format!("c{c}"), move |_, _, test| {
            Ok(vec![c; test.n_rows()])
        })
    }

    #[test]
    fn scores_are_laid_out_by_model_metric_and_split() {
        let x = Design::new(vec!["x".into()], vec![vec![1.0; 6]]).unwrap();
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let splits = k_fold(6, 3, 2).unwrap();
        let mae = |t: &[f64], p: &[f64], _: &Design| {
            t.iter().zip(p).map(|(a, b)| (a - b).abs()).sum::<f64>() / t.len() as f64
        };
        let bias = |t: &[f64], p: &[f64], _: &Design| {
            (p.iter().sum::<f64>() - t.iter().sum::<f64>()) / t.len() as f64
        };
        let cands = [constant(0.0), constant(3.5)];
        let table = compare(&cands, &x, &y, &splits, &[("mae", &mae), ("bias", &bias)]).unwrap();
        assert_eq!(table.models(), ["c0", "c3.5"]);
        assert_eq!(table.metrics(), ["mae", "bias"]);
        for (s, split) in splits.iter().enumerate() {
            let test_mean = split.test.iter().map(|&i| y[i]).sum::<f64>() / 2.0;
            assert!((table.split_scores(0, 1)[s] + test_mean).abs() < 1e-12);
            assert!((table.split_scores(1, 1)[s] - (3.5 - test_mean)).abs() < 1e-12);
        }
        assert!((table.mean(0, 0) - 3.5).abs() < 1e-12);
        assert_eq!(table.best(0), 1);
        assert_eq!(table.difference_std_error(1, 0), 0.0);
        // Constant predictions: the difference is the same on every split.
        assert!(table.difference_std_error(0, 1).abs() < 1e-12);
        assert!(table.std_error(0, 1) > 0.0);
    }

    #[test]
    fn errors_name_the_candidate() {
        let x = Design::new(vec!["x".into()], vec![vec![1.0; 4]]).unwrap();
        let bad = Candidate::from_fn("broken", |_, _, _| Err(Error::Data("nope".into())));
        let m = |_: &[f64], _: &[f64], _: &Design| 0.0;
        let err = compare(
            &[bad],
            &x,
            &[1.0; 4],
            &k_fold(4, 2, 1).unwrap(),
            &[("m", &m)],
        )
        .unwrap_err();
        assert!(err.to_string().contains("broken"));
        assert!(compare(&[], &x, &[1.0; 4], &k_fold(4, 2, 1).unwrap(), &[("m", &m)]).is_err());
    }
}
