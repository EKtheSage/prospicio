//! The Rearrangement Algorithm (Puccetti and Rüschendorf 2012; Embrechts,
//! Puccetti and Rüschendorf 2013): the worst VaR of a sum whose marginals
//! are known and whose dependence is not.
//!
//! The worst VaR at `p` puts the top `1 − p` of every marginal together and
//! arranges them so their sum is as flat as possible. The algorithm works on
//! a matrix whose column `j` holds `n` quantiles of marginal `j` from `p`
//! up, and repeatedly orders each column opposite to the sum of the others,
//! until the smallest row sum stops rising. That minimum is the estimate. It
//! is a heuristic: the answer depends on the starting arrangement, so the
//! start is part of the input and a result is reproducible by it.

use prospicio_core::{Error, Result, StreamRng};

use crate::distribution::Distribution;

/// How [`rearrange`] arranges the columns before it starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Start {
    /// As given.
    AsGiven,
    /// Column `j` rotated down by `shift × j` places (the guide's
    /// deterministic starts, the same in Python and R).
    Rotated(usize),
    /// Each column shuffled from stream `j` of the seed.
    Shuffled(u64),
}

/// The result of [`rearrange`] or [`worst_var`].
#[derive(Debug, Clone, PartialEq)]
pub struct Rearrangement {
    /// The smallest row sum of the final arrangement: the worst VaR
    /// estimate.
    pub value: f64,
    /// The final arrangement, column by column.
    pub columns: Vec<Vec<f64>>,
    /// Sweeps over the columns until the minimum stopped rising.
    pub iterations: usize,
}

/// Rearranges `columns` (each of the same length) from `start` until a
/// sweep over every column raises the smallest row sum by less than
/// `tolerance`, each column put in the order opposite to the sum of the
/// others (ties in that sum kept in row order).
///
/// ```
/// use prospicio_prob::rearrangement::{Start, rearrange};
///
/// // Two columns of 1..4: the flattest arrangement pairs 1 with 4.
/// let c = vec![vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0, 3.0, 4.0]];
/// let r = rearrange(&c, Start::AsGiven, 1e-9).unwrap();
/// assert_eq!(r.value, 5.0);
/// ```
///
/// Fails when the columns are empty, differ in length or hold a value that
/// is not finite.
pub fn rearrange(columns: &[Vec<f64>], start: Start, tolerance: f64) -> Result<Rearrangement> {
    let n = columns.first().map_or(0, Vec::len);
    if columns.is_empty() || n == 0 {
        return Err(invalid("columns", 0.0, "must not be empty"));
    }
    if columns.iter().any(|c| c.len() != n) {
        return Err(invalid(
            "columns",
            n as f64,
            "must all have the same length",
        ));
    }
    if let Some(&bad) = columns.iter().flatten().find(|v| !v.is_finite()) {
        return Err(invalid("columns", bad, "must be finite"));
    }
    let mut x: Vec<Vec<f64>> = columns
        .iter()
        .enumerate()
        .map(|(j, c)| match start {
            Start::AsGiven => c.clone(),
            Start::Rotated(shift) => {
                let s = (shift * j) % n;
                (0..n).map(|i| c[(i + n - s) % n]).collect()
            }
            Start::Shuffled(seed) => {
                let mut c = c.clone();
                let mut rng = StreamRng::new(seed, j as u64);
                for i in (1..n).rev() {
                    let k = (rng.next_open01() * (i + 1) as f64) as usize;
                    c.swap(i, k.min(i));
                }
                c
            }
        })
        .collect();
    let mut best = f64::NEG_INFINITY;
    let mut iterations = 0;
    let mut order: Vec<usize> = (0..n).collect();
    let mut sorted = vec![0.0; n];
    loop {
        iterations += 1;
        for j in 0..x.len() {
            let total: Vec<f64> = (0..n)
                .map(|i| x.iter().map(|c| c[i]).sum::<f64>())
                .collect();
            let rest: Vec<f64> = (0..n).map(|i| total[i] - x[j][i]).collect();
            order.iter_mut().enumerate().for_each(|(i, o)| *o = i);
            order.sort_by(|&a, &b| rest[a].total_cmp(&rest[b]));
            sorted.copy_from_slice(&x[j]);
            sorted.sort_by(|a, b| b.total_cmp(a));
            for (k, &row) in order.iter().enumerate() {
                x[j][row] = sorted[k];
            }
        }
        let low = (0..n)
            .map(|i| x.iter().map(|c| c[i]).sum::<f64>())
            .fold(f64::INFINITY, f64::min);
        if low - best < tolerance {
            return Ok(Rearrangement {
                value: low,
                columns: x,
                iterations,
            });
        }
        best = low;
    }
}

/// The worst VaR at `p` of the sum of `marginals`, by [`rearrange`] on
/// `n` quantiles of each, at `p + (1 − p) i / n` for `i = 0, …, n − 1`
/// (Embrechts, Puccetti and Rüschendorf's lower matrix, which converges to
/// the worst VaR from below as `n` grows; *Pricing Insurance Risk*,
/// Exercise 59).
///
/// Fails for `p` outside `(0, 1)`, `n` of 0, or a marginal whose quantile
/// is not finite there.
pub fn worst_var(
    marginals: &[&dyn Distribution],
    p: f64,
    n: usize,
    start: Start,
    tolerance: f64,
) -> Result<Rearrangement> {
    if !(p > 0.0 && p < 1.0) {
        return Err(invalid("p", p, "must be in (0, 1)"));
    }
    if n == 0 {
        return Err(invalid("n", 0.0, "must be positive"));
    }
    let columns = marginals
        .iter()
        .map(|m| {
            (0..n)
                .map(|i| m.quantile(p + (1.0 - p) * i as f64 / n as f64))
                .collect::<Result<Vec<f64>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    rearrange(&columns, start, tolerance)
}

fn invalid(name: &'static str, value: f64, reason: &'static str) -> Error {
    Error::InvalidParameter {
        name,
        value,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rearranging_flattens_the_row_sums() {
        let c = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![0.0, 0.0, 0.0, 0.0, 10.0],
        ];
        for start in [Start::AsGiven, Start::Rotated(2), Start::Shuffled(5)] {
            let r = rearrange(&c, start, 1e-12).unwrap();
            // Each column keeps its values.
            for (got, want) in r.columns.iter().zip(&c) {
                let mut g = got.clone();
                g.sort_by(f64::total_cmp);
                assert_eq!(&g, want);
            }
            // The total is 40 over 5 rows, so no row sum can exceed 8 at the
            // minimum, and the comonotonic minimum is 2.
            assert!(r.value > 2.0 && r.value <= 8.0, "{start:?}: {}", r.value);
        }
        assert!(rearrange(&[vec![1.0], vec![1.0, 2.0]], Start::AsGiven, 1e-9).is_err());
    }
}
