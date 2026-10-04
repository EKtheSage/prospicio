//! Resampling: train/test splits, cross-validation, and grid and random
//! search.
//!
//! Splits are row indices, so they work with any [`Design`]. Random splits
//! draw from [`StreamRng`] with a seed, so they replay exactly.

use act_core::{Error, Result, StreamRng};

use crate::design::Design;
use crate::model::{Fitted, Model};

/// One train/test split: row indices into the data.
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    pub train: Vec<usize>,
    pub test: Vec<usize>,
}

/// `k`-fold cross-validation: rows shuffled with `seed`, then cut into `k`
/// folds of near-equal size; fold `i` is the test set of split `i`.
///
/// ```
/// use act_models::resample::k_fold;
///
/// let splits = k_fold(10, 5, 1).unwrap();
/// assert_eq!(splits.len(), 5);
/// assert!(splits.iter().all(|s| s.test.len() == 2 && s.train.len() == 8));
/// ```
pub fn k_fold(n: usize, k: usize, seed: u64) -> Result<Vec<Split>> {
    check_k(n, k)?;
    let order = shuffled(n, seed);
    Ok(folds(&order, k)
        .into_iter()
        .map(|test| complement(n, test))
        .collect())
}

/// Grouped `k`-fold: every row of a group lands in the same fold, so a
/// policy or an insured seen in training never appears in its test set.
/// Groups are shuffled with `seed` and dealt into `k` folds.
pub fn group_k_fold(groups: &[String], k: usize, seed: u64) -> Result<Vec<Split>> {
    let mut names: Vec<&String> = groups.iter().collect();
    names.sort();
    names.dedup();
    check_k(names.len(), k)?;
    let order = shuffled(names.len(), seed);
    let fold_of_group: Vec<usize> = {
        let mut f = vec![0; names.len()];
        for (fold, members) in folds(&order, k).into_iter().enumerate() {
            for g in members {
                f[g] = fold;
            }
        }
        f
    };
    let mut tests = vec![Vec::new(); k];
    for (row, g) in groups.iter().enumerate() {
        let gi = names.binary_search(&g).expect("every group is listed");
        tests[fold_of_group[gi]].push(row);
    }
    Ok(tests
        .into_iter()
        .map(|test| complement(groups.len(), test))
        .collect())
}

/// Time-ordered splits for forecasting: for each of the last `n_test`
/// distinct periods `t`, train on rows with period before `t` and test on
/// rows with period `t`. For a reserving triangle, the period is the
/// calendar diagonal (origin + development), and this is the diagonal
/// backtest of `docs/design/models.md`.
///
/// ```
/// use act_models::resample::time_ordered;
///
/// // Diagonals of a 3 × 3 triangle's cells.
/// let diag = [0, 1, 2, 1, 2, 2];
/// let splits = time_ordered(&diag, 1).unwrap();
/// assert_eq!(splits[0].test, [2, 4, 5]);
/// assert_eq!(splits[0].train, [0, 1, 3]);
/// ```
pub fn time_ordered(periods: &[i64], n_test: usize) -> Result<Vec<Split>> {
    let mut distinct: Vec<i64> = periods.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    if n_test == 0 || n_test >= distinct.len() {
        return Err(Error::InvalidParameter {
            name: "n_test",
            value: n_test as f64,
            reason: "must be at least 1 and leave an earlier period to train on",
        });
    }
    Ok(distinct[distinct.len() - n_test..]
        .iter()
        .map(|&t| Split {
            train: (0..periods.len()).filter(|&i| periods[i] < t).collect(),
            test: (0..periods.len()).filter(|&i| periods[i] == t).collect(),
        })
        .collect())
}

/// Fits `model` on each split's training rows and scores its predictions
/// on the test rows with `score(y_test, predicted, test_design)`. Returns
/// one score per split.
///
/// ```
/// use act_models::resample::{cross_validate, k_fold};
/// # use act_models::{Design, Fitted, Model};
/// # struct MeanModel;
/// # struct MeanFit(f64);
/// # impl Model for MeanModel {
/// #     type Fitted = MeanFit;
/// #     fn fit(&self, _: &Design, y: &[f64]) -> act_core::Result<MeanFit> {
/// #         Ok(MeanFit(y.iter().sum::<f64>() / y.len() as f64))
/// #     }
/// # }
/// # impl Fitted for MeanFit {
/// #     fn predict(&self, d: &Design) -> act_core::Result<Vec<f64>> { Ok(vec![self.0; d.n_rows()]) }
/// #     fn predict_distribution(&self, _: &Design, _: usize, _: u64)
/// #         -> act_core::Result<act_prob::PredictiveDistribution> { unimplemented!() }
/// # }
/// let x = Design::new(vec!["x".into()], vec![vec![1.0; 6]]).unwrap();
/// let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let scores = cross_validate(&MeanModel, &x, &y, &k_fold(6, 3, 1).unwrap(), |t, p, _| {
///     act_models::metrics::rmse(t, p).unwrap()
/// })
/// .unwrap();
/// assert_eq!(scores.len(), 3);
/// ```
pub fn cross_validate<M, S>(
    model: &M,
    design: &Design,
    y: &[f64],
    splits: &[Split],
    score: S,
) -> Result<Vec<f64>>
where
    M: Model,
    S: Fn(&[f64], &[f64], &Design) -> f64,
{
    if y.len() != design.n_rows() {
        return Err(Error::Data(format!(
            "{} responses for {} design rows",
            y.len(),
            design.n_rows()
        )));
    }
    splits
        .iter()
        .map(|s| {
            let train = design.select(&s.train);
            let y_train: Vec<f64> = s.train.iter().map(|&i| y[i]).collect();
            let fitted = model.fit(&train, &y_train)?;
            let test = design.select(&s.test);
            let y_test: Vec<f64> = s.test.iter().map(|&i| y[i]).collect();
            let pred = fitted.predict(&test)?;
            Ok(score(&y_test, &pred, &test))
        })
        .collect()
}

/// The result of [`grid_search`]: each candidate's mean score, and the
/// best (lowest) one.
#[derive(Debug, Clone, PartialEq)]
pub struct GridSearch<P> {
    /// `(candidate, mean score over the splits)`, in the order given.
    pub scores: Vec<(P, f64)>,
    /// Index of the candidate with the lowest mean score.
    pub best: usize,
}

/// Scores every candidate in `candidates` by [`cross_validate`] on the same
/// splits, with `make(candidate)` building the model, and picks the lowest
/// mean score (scores are losses: deviance, RMSE, CRPS).
pub fn grid_search<P, M, B, S>(
    candidates: Vec<P>,
    make: B,
    design: &Design,
    y: &[f64],
    splits: &[Split],
    score: S,
) -> Result<GridSearch<P>>
where
    M: Model,
    B: Fn(&P) -> M,
    S: Fn(&[f64], &[f64], &Design) -> f64,
{
    if candidates.is_empty() {
        return Err(Error::InvalidParameter {
            name: "candidates",
            value: 0.0,
            reason: "must not be empty",
        });
    }
    let mut scores = Vec::with_capacity(candidates.len());
    for c in candidates {
        let s = cross_validate(&make(&c), design, y, splits, &score)?;
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        scores.push((c, mean));
    }
    let best = scores
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.1.total_cmp(&b.1.1))
        .map(|(i, _)| i)
        .expect("not empty");
    Ok(GridSearch { scores, best })
}

/// Random search: draws `n` candidates with `draw` from stream 0 of
/// `seed`, then scores them as [`grid_search`] does. With several
/// hyperparameters, random candidates cover each one's range far better
/// than a grid of the same size (Bergstra and Bengio, 2012), which is what
/// neural networks need. [`log_uniform`] draws scale parameters such as
/// learning rates.
///
/// ```
/// use act_models::resample::{k_fold, log_uniform, random_search};
/// # use act_models::{Design, Fitted, Model};
/// # struct Shrunk(f64);
/// # struct Fit(f64);
/// # impl Model for Shrunk {
/// #     type Fitted = Fit;
/// #     fn fit(&self, _: &Design, y: &[f64]) -> act_core::Result<Fit> {
/// #         Ok(Fit(y.iter().sum::<f64>() / (y.len() as f64 + self.0)))
/// #     }
/// # }
/// # impl Fitted for Fit {
/// #     fn predict(&self, d: &Design) -> act_core::Result<Vec<f64>> { Ok(vec![self.0; d.n_rows()]) }
/// #     fn predict_distribution(&self, _: &Design, _: usize, _: u64)
/// #         -> act_core::Result<act_prob::PredictiveDistribution> { unimplemented!() }
/// # }
/// let x = Design::new(vec!["x".into()], vec![vec![1.0; 8]]).unwrap();
/// let y = [3.0, 5.0, 4.0, 6.0, 5.0, 4.0, 5.0, 4.0];
/// let splits = k_fold(8, 4, 1).unwrap();
/// let found = random_search(
///     20,
///     7,
///     |rng| log_uniform(rng, 1e-3, 10.0),
///     |&shrink| Shrunk(shrink),
///     &x,
///     &y,
///     &splits,
///     |t, p, _| act_models::metrics::rmse(t, p).unwrap(),
/// )
/// .unwrap();
/// assert_eq!(found.scores.len(), 20);
/// let best = found.scores[found.best].1;
/// assert!(found.scores.iter().all(|(_, s)| *s >= best));
/// // Heavy shrinkage of the mean towards zero loses.
/// assert!(found.scores[found.best].0 < 1.0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn random_search<P, M, D, B, S>(
    n: usize,
    seed: u64,
    mut draw: D,
    make: B,
    design: &Design,
    y: &[f64],
    splits: &[Split],
    score: S,
) -> Result<GridSearch<P>>
where
    M: Model,
    D: FnMut(&mut StreamRng) -> P,
    B: Fn(&P) -> M,
    S: Fn(&[f64], &[f64], &Design) -> f64,
{
    let mut rng = StreamRng::new(seed, 0);
    let candidates = (0..n).map(|_| draw(&mut rng)).collect();
    grid_search(candidates, make, design, y, splits, score)
}

/// A draw log-uniform between `low` and `high` (both positive): uniform
/// in orders of magnitude, for learning rates and penalties.
pub fn log_uniform(rng: &mut StreamRng, low: f64, high: f64) -> f64 {
    (low.ln() + rng.next_open01() * (high.ln() - low.ln())).exp()
}

/// A draw uniform over `low..=high`, for layer widths and counts.
pub fn uniform_int(rng: &mut StreamRng, low: usize, high: usize) -> usize {
    let span = (high - low + 1) as f64;
    low + ((rng.next_open01() * span) as usize).min(high - low)
}

fn check_k(n: usize, k: usize) -> Result<()> {
    if k < 2 || k > n {
        return Err(Error::InvalidParameter {
            name: "k",
            value: k as f64,
            reason: "must be at least 2 and at most the number of rows (or groups)",
        });
    }
    Ok(())
}

/// `0..n` shuffled by Fisher–Yates on stream 0 of `seed`.
fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = StreamRng::new(seed, 0);
    for i in (1..n).rev() {
        let j = ((rng.next_open01() * (i + 1) as f64) as usize).min(i);
        order.swap(i, j);
    }
    order
}

/// `order` cut into `k` contiguous folds, the first `n mod k` one longer.
fn folds(order: &[usize], k: usize) -> Vec<Vec<usize>> {
    let n = order.len();
    let (base, extra) = (n / k, n % k);
    let mut out = Vec::with_capacity(k);
    let mut start = 0;
    for i in 0..k {
        let len = base + usize::from(i < extra);
        let mut fold = order[start..start + len].to_vec();
        fold.sort_unstable();
        out.push(fold);
        start += len;
    }
    out
}

fn complement(n: usize, test: Vec<usize>) -> Split {
    let mut in_test = vec![false; n];
    for &i in &test {
        in_test[i] = true;
    }
    Split {
        train: (0..n).filter(|&i| !in_test[i]).collect(),
        test,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_fold_partitions_and_replays() {
        let s = k_fold(23, 4, 9).unwrap();
        let mut all: Vec<usize> = s.iter().flat_map(|x| x.test.clone()).collect();
        all.sort_unstable();
        assert_eq!(all, (0..23).collect::<Vec<_>>());
        assert_eq!(s, k_fold(23, 4, 9).unwrap());
        assert_ne!(s, k_fold(23, 4, 10).unwrap());
        assert!(k_fold(3, 4, 1).is_err());
    }

    #[test]
    fn groups_stay_together() {
        let groups: Vec<String> = ["a", "b", "a", "c", "b", "d", "c"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        for s in group_k_fold(&groups, 2, 3).unwrap() {
            for &i in &s.test {
                assert!(s.train.iter().all(|&j| groups[j] != groups[i]));
            }
        }
    }
}
