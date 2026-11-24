//! Weighted Gram products `Xᵀ W X` and `Xᵀ W z`, blocked by rows (#969).
//!
//! The design is stored column by column. Walking it row by row and
//! touching every column per row strides across `p` separate arrays and
//! leaves the inner loop no contiguous run to vectorize. Here the rows are
//! cut into blocks of a fixed size, the blocks run in parallel, and within
//! a block a tile of rows is weighted once (`wᵢ xᵢₐ` for every column) and
//! then reduced to its `p (p + 1) / 2` dot products over contiguous slices.
//!
//! The partition depends only on the number of rows, never on the number
//! of threads, and the block partials are added in block order, so the
//! result is the same on one thread, on many, and on WASM (where rayon runs
//! on the calling thread).

use prospicio_core::Result;
use prospicio_models::Design;
use rayon::prelude::*;

/// Rows per parallel block.
const BLOCK: usize = 8192;
/// Rows per cache tile within a block: the weighted tile of `p` columns
/// (`8 p TILE` bytes) stays in L2.
const TILE: usize = 256;

/// `Xᵀ W X` (row-major `p × p`, symmetric) and, when `rhs` is set,
/// `Xᵀ W z`, for per-row weights and working responses `weight(i) =
/// (wᵢ, zᵢ)`. `weight` reports a row it cannot weigh as an error; the
/// error of the lowest such row is returned, as a row-by-row pass would.
pub(crate) fn weighted_gram<F>(x: &Design, rhs: bool, weight: F) -> Result<(Vec<f64>, Vec<f64>)>
where
    F: Fn(usize) -> Result<(f64, f64)> + Sync,
{
    let (n, p) = (x.n_rows(), x.n_cols());
    let columns: Vec<&[f64]> = (0..p).map(|j| x.column(j)).collect();
    let blocks: Vec<Result<Vec<f64>>> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|b| block(&columns, b * BLOCK, ((b + 1) * BLOCK).min(n), rhs, &weight))
        .collect();
    let mut acc = vec![0.0; p * p + p];
    for partial in blocks {
        for (a, v) in acc.iter_mut().zip(partial?) {
            *a += v;
        }
    }
    let xtwz = acc.split_off(p * p);
    let mut xtwx = acc;
    for a in 0..p {
        for b in a + 1..p {
            xtwx[a * p + b] = xtwx[b * p + a];
        }
    }
    Ok((xtwx, xtwz))
}

/// One block's lower triangle of `Xᵀ W X` (row-major, `p × p`) followed
/// by its `Xᵀ W z` (`p`).
fn block<F>(columns: &[&[f64]], start: usize, end: usize, rhs: bool, weight: &F) -> Result<Vec<f64>>
where
    F: Fn(usize) -> Result<(f64, f64)>,
{
    let p = columns.len();
    let mut out = vec![0.0; p * p + p];
    let mut w = [0.0; TILE];
    let mut z = [0.0; TILE];
    let mut xw = vec![0.0; p * TILE];
    let mut t = start;
    while t < end {
        let len = TILE.min(end - t);
        for k in 0..len {
            (w[k], z[k]) = weight(t + k)?;
        }
        for (a, col) in columns.iter().enumerate() {
            let dst = &mut xw[a * TILE..a * TILE + len];
            for ((d, x), wk) in dst.iter_mut().zip(&col[t..t + len]).zip(&w[..len]) {
                *d = x * wk;
            }
        }
        for a in 0..p {
            let xa = &xw[a * TILE..a * TILE + len];
            for (b, col) in columns.iter().enumerate().take(a + 1) {
                out[a * p + b] += dot(xa, &col[t..t + len]);
            }
            if rhs {
                out[p * p + a] += dot(xa, &z[..len]);
            }
        }
        t += len;
    }
    Ok(out)
}

/// `Σ aᵢ bᵢ` with eight running sums, so the loop vectorizes.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0; 8];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    for (x, y) in ca.iter().zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_core::Error;

    /// The row-by-row loop this module replaced, kept as the oracle.
    fn naive(x: &Design, w: &[f64], z: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let p = x.n_cols();
        let mut xtwx = vec![0.0; p * p];
        let mut xtwz = vec![0.0; p];
        for i in 0..x.n_rows() {
            for a in 0..p {
                let xa = x.column(a)[i] * w[i];
                xtwz[a] += xa * z[i];
                for b in 0..=a {
                    xtwx[a * p + b] += xa * x.column(b)[i];
                }
            }
        }
        for a in 0..p {
            for b in a + 1..p {
                xtwx[a * p + b] = xtwx[b * p + a];
            }
        }
        (xtwx, xtwz)
    }

    fn data(n: usize, p: usize) -> (Design, Vec<f64>, Vec<f64>) {
        let v = |i: usize, j: usize| ((i * 7919 + j * 104_729) % 1013) as f64 / 101.3 - 5.0;
        let columns = (0..p).map(|j| (0..n).map(|i| v(i, j)).collect()).collect();
        let names = (0..p).map(|j| format!("x{j}")).collect();
        let w = (0..n).map(|i| 0.5 + (i % 13) as f64 / 7.0).collect();
        let z = (0..n).map(|i| v(i, 99)).collect();
        (Design::new(names, columns).unwrap(), w, z)
    }

    #[test]
    fn matches_the_row_by_row_sums() {
        // Sizes on both sides of a tile and of a block, and a ragged end.
        for (n, p) in [(1, 1), (5, 3), (255, 4), (257, 7), (8191, 2), (20_000, 9)] {
            let (x, w, z) = data(n, p);
            let (xtwx, xtwz) = weighted_gram(&x, true, |i| Ok((w[i], z[i]))).unwrap();
            let (gx, gz) = naive(&x, &w, &z);
            let scale = gx.iter().chain(&gz).fold(1.0_f64, |m, v| m.max(v.abs()));
            for (a, b) in xtwx.iter().zip(&gx).chain(xtwz.iter().zip(&gz)) {
                assert!((a - b).abs() <= 1e-12 * scale, "n {n} p {p}: {a} vs {b}");
            }
            for a in 0..p {
                for b in 0..p {
                    assert_eq!(xtwx[a * p + b], xtwx[b * p + a]);
                }
            }
        }
    }

    #[test]
    fn is_the_same_on_one_thread_as_on_many() {
        let (x, w, z) = data(50_000, 6);
        let run = || weighted_gram(&x, true, |i| Ok((w[i], z[i]))).unwrap();
        let many = run();
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(run);
        assert_eq!(many, one);
    }

    #[test]
    fn reports_the_lowest_failing_row() {
        let (x, _, _) = data(30_000, 2);
        let err = weighted_gram(&x, true, |i| {
            if i == 9000 || i == 29_000 {
                Err(Error::Data(format!("row {i}")))
            } else {
                Ok((1.0, 0.0))
            }
        })
        .unwrap_err();
        assert_eq!(err.to_string(), Error::Data("row 9000".into()).to_string());
    }

    #[test]
    fn dot_handles_every_remainder() {
        for n in 0..20 {
            let a: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
            let b: Vec<f64> = (0..n).map(|i| 2.0 - i as f64).collect();
            let want: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert_eq!(dot(&a, &b), want);
        }
    }
}
