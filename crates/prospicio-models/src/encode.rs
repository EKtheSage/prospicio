//! Out-of-fold target encoding of a high-cardinality factor with
//! credibility (Holmes and Casotto, CAS Monograph 16, chapter 6).
//!
//! Each level's encoding is its actual-to-expected ratio shrunk to 1 by
//! the credibility `z = n / (n + K)`, with `n` the level's exposure:
//! `z A / E + (1 − z)`. The expected values come from the caller (the
//! overall mean times exposure, or a first model's predictions), so the
//! ratio measures what that model misses. On the training rows each row is
//! encoded from the other folds only, so a row's own response never enters
//! its encoding; new data is encoded from all the training rows.

use std::collections::BTreeMap;

use prospicio_core::{Error, Result};

use crate::resample::Split;

/// The encoding of each level learned from all the training rows, from
/// [`target_encode`]; [`encode`](Self::encode) applies it to new data.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetEncoding {
    /// The levels seen in training, sorted.
    pub levels: Vec<String>,
    /// Each level's credibility-weighted actual-to-expected ratio.
    pub ratios: Vec<f64>,
    /// The credibility constant `K`.
    pub k: f64,
}

impl TargetEncoding {
    /// The encoding of each value; a level not seen in training gets 1, no
    /// credibility.
    pub fn encode(&self, values: &[String]) -> Vec<f64> {
        values
            .iter()
            .map(|v| match self.levels.binary_search(v) {
                Ok(i) => self.ratios[i],
                Err(_) => 1.0,
            })
            .collect()
    }
}

/// The out-of-fold encoding of each training row, and the encoding from all
/// the rows for new data.
///
/// `values` are the factor's levels per row, `y` the actual responses,
/// `expected` the expected responses, `exposure` the credibility weights
/// (default 1 per row), and every row is in exactly one split's test set.
/// A row is encoded from its split's training rows; within those, a level
/// with no expected loss gets 1.
///
/// ```
/// use prospicio_models::encode::target_encode;
/// use prospicio_models::resample::Split;
///
/// let values: Vec<String> = ["a", "a", "b", "b"].iter().map(|s| s.to_string()).collect();
/// let splits = vec![
///     Split { train: vec![1, 3], test: vec![0, 2] },
///     Split { train: vec![0, 2], test: vec![1, 3] },
/// ];
/// let (rows, all) =
///     target_encode(&values, &[2.0, 2.0, 0.0, 1.0], &[1.0; 4], None, &splits, 1.0).unwrap();
/// // Row 0 sees only row 1 of level a: z = 1/2, ratio 2, so 1.5.
/// assert_eq!(rows[0], 1.5);
/// // From all rows, a has n = 2, z = 2/3 and ratio 2.
/// assert!((all.encode(&["a".to_string()])[0] - 5.0 / 3.0).abs() < 1e-15);
/// ```
///
/// Fails when the lengths differ, `k` is negative, or a row is in no test
/// set or several.
pub fn target_encode(
    values: &[String],
    y: &[f64],
    expected: &[f64],
    exposure: Option<&[f64]>,
    splits: &[Split],
    k: f64,
) -> Result<(Vec<f64>, TargetEncoding)> {
    let n = values.len();
    for (len, name) in [(y.len(), "y"), (expected.len(), "expected")] {
        if len != n {
            return Err(Error::InvalidParameter {
                name,
                value: len as f64,
                reason: "must have one value per row",
            });
        }
    }
    if let Some(e) = exposure {
        if e.len() != n {
            return Err(Error::InvalidParameter {
                name: "exposure",
                value: e.len() as f64,
                reason: "must have one value per row",
            });
        }
    }
    if !(k.is_finite() && k >= 0.0) {
        return Err(Error::InvalidParameter {
            name: "k",
            value: k,
            reason: "must be finite and non-negative",
        });
    }
    let mut seen = vec![false; n];
    for s in splits {
        for &i in &s.test {
            if i >= n || seen[i] {
                return Err(Error::InvalidParameter {
                    name: "splits",
                    value: i as f64,
                    reason: "every row must be in exactly one test set",
                });
            }
            seen[i] = true;
        }
    }
    if let Some(i) = seen.iter().position(|s| !s) {
        return Err(Error::InvalidParameter {
            name: "splits",
            value: i as f64,
            reason: "every row must be in exactly one test set",
        });
    }
    let e = |i: usize| exposure.map_or(1.0, |e| e[i]);
    let stats = |rows: &mut dyn Iterator<Item = usize>| {
        let mut by: BTreeMap<&str, (f64, f64, f64)> = BTreeMap::new();
        for i in rows {
            let s = by.entry(values[i].as_str()).or_default();
            s.0 += y[i];
            s.1 += expected[i];
            s.2 += e(i);
        }
        by
    };
    let ratio = |(actual, expected, exposure): (f64, f64, f64)| {
        let z = if exposure + k > 0.0 {
            exposure / (exposure + k)
        } else {
            0.0
        };
        if expected == 0.0 {
            1.0
        } else {
            z * actual / expected + (1.0 - z)
        }
    };
    let mut rows = vec![1.0; n];
    for s in splits {
        let by = stats(&mut s.train.iter().copied());
        for &i in &s.test {
            rows[i] = by.get(values[i].as_str()).copied().map_or(1.0, ratio);
        }
    }
    let by = stats(&mut (0..n));
    let encoding = TargetEncoding {
        levels: by.keys().map(|l| l.to_string()).collect(),
        ratios: by.values().copied().map(ratio).collect(),
        k,
    };
    Ok((rows, encoding))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resample::k_fold;

    #[test]
    fn a_row_never_sees_its_own_response() {
        let values: Vec<String> = (0..60).map(|i| format!("l{}", i % 7)).collect();
        let y: Vec<f64> = (0..60).map(|i| ((i * 13) % 5) as f64).collect();
        let expected = vec![2.0; 60];
        let splits = k_fold(60, 5, 3).unwrap();
        let (rows, all) = target_encode(&values, &y, &expected, None, &splits, 4.0).unwrap();
        // Change one row's response: its own encoding does not move.
        let mut y2 = y.clone();
        y2[10] += 100.0;
        let (rows2, _) = target_encode(&values, &y2, &expected, None, &splits, 4.0).unwrap();
        assert_eq!(rows[10], rows2[10]);
        assert!(rows2.iter().zip(&rows).any(|(a, b)| a != b));
        // K = 0 is the plain ratio, and an unseen level is 1.
        let (_, plain) = target_encode(&values, &y, &expected, None, &splits, 0.0).unwrap();
        let a: f64 = (0..60).filter(|i| i % 7 == 0).map(|i| y[i]).sum();
        let n = (0..60).filter(|i| i % 7 == 0).count() as f64;
        assert!((plain.encode(&["l0".into()])[0] - a / (2.0 * n)).abs() < 1e-12);
        assert_eq!(all.encode(&["new".into()]), [1.0]);
        assert!(target_encode(&values, &y, &expected, None, &splits[1..], 4.0).is_err());
    }
}
