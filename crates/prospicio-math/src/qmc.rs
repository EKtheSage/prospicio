//! Sobol low-discrepancy points for quasi-Monte Carlo.
//!
//! [`Sobol`] gives the points of Joe and Kuo's Sobol sequence in up to
//! 1,111 dimensions (their `new-joe-kuo-6.21201` direction numbers, as SciPy
//! ships them), at 32 bits, in Gray-code order. Unscrambled, the points are
//! SciPy's `stats.qmc.Sobol(d, scramble=False, bits=32)` exactly, the first
//! one at the origin. Scrambled, each dimension's direction numbers are
//! multiplied by a random lower-triangular binary matrix with a unit
//! diagonal and the points are shifted by a random digit vector (Matoušek's
//! linear matrix scramble with a digital shift, SciPy's scramble), which
//! keeps the sequence's balance and makes each point uniform on `[0, 1)^d`.
//! The scramble is drawn from a seed, so a run is reproducible; it is not
//! SciPy's random stream, so scrambled points differ from SciPy's.
//!
//! The first `2^m` points are balanced: for `m` large enough, each of the
//! `2^m` intervals `[k / 2^m, (k + 1) / 2^m)` of a coordinate holds exactly
//! one point. Take point counts that are powers of two.

use crate::sobol_table::{INIT, INIT_START, MAX_DIM, POLY};

/// Bits per coordinate.
const BITS: usize = 32;
const SCALE: f64 = 1.0 / 4_294_967_296.0;

/// How the points are randomized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scramble {
    /// The plain sequence, SciPy's points with `scramble=False`.
    None,
    /// A linear matrix scramble with a digital shift, drawn from `seed`.
    Lms {
        /// Seed of the scramble.
        seed: u64,
    },
}

/// A Sobol sequence in `dim` dimensions.
///
/// ```
/// use prospicio_math::qmc::{Scramble, Sobol};
///
/// let mut s = Sobol::new(2, Scramble::None).unwrap();
/// let mut x = [0.0; 2];
/// s.next_point(&mut x);
/// assert_eq!(x, [0.0, 0.0]);
/// s.next_point(&mut x);
/// assert_eq!(x, [0.5, 0.5]);
/// s.next_point(&mut x);
/// assert_eq!(x, [0.75, 0.25]);
/// ```
#[derive(Clone, Debug)]
pub struct Sobol {
    dim: usize,
    /// Direction numbers, `BITS` per dimension, the most significant first.
    v: Vec<u32>,
    /// The current point's digits.
    x: Vec<u32>,
    /// Points given so far.
    index: u64,
}

impl Sobol {
    /// The sequence in `dim` dimensions, from 1 to 1,111.
    pub fn new(dim: usize, scramble: Scramble) -> Result<Self, String> {
        if dim == 0 || dim > MAX_DIM {
            return Err(format!(
                "a Sobol sequence needs 1 to {MAX_DIM} dimensions, not {dim}"
            ));
        }
        let mut v = vec![0u32; dim * BITS];
        for d in 0..dim {
            let row = &mut v[d * BITS..(d + 1) * BITS];
            // The odd integers m_j, then shifted into place.
            if d == 0 {
                row.fill(1);
            } else {
                let p = POLY[d];
                let s = (32 - p.leading_zeros() - 1) as usize;
                let init = &INIT[INIT_START[d] as usize..INIT_START[d + 1] as usize];
                row[..s].copy_from_slice(init);
                for j in s..BITS {
                    let mut m = row[j - s];
                    for k in 0..s {
                        if (p >> (s - 1 - k)) & 1 == 1 {
                            m ^= row[j - k - 1] << (k + 1);
                        }
                    }
                    row[j] = m;
                }
            }
            for (j, m) in row.iter_mut().enumerate() {
                *m <<= BITS - 1 - j;
            }
        }
        let mut x = vec![0u32; dim];
        if let Scramble::Lms { seed } = scramble {
            let mut rng = SplitMix(seed);
            for d in 0..dim {
                // Row p of the matrix: unit diagonal, random bits above it
                // (the more significant digits), in MSB-first positions.
                let rows: Vec<u32> = (0..BITS)
                    .map(|p| {
                        let above = if p == 0 {
                            0
                        } else {
                            (rng.next() as u32) & !(u32::MAX >> p)
                        };
                        above | (1u32 << (BITS - 1 - p))
                    })
                    .collect();
                for vj in &mut v[d * BITS..(d + 1) * BITS] {
                    let mut out = 0u32;
                    for (p, &r) in rows.iter().enumerate() {
                        out |= ((r & *vj).count_ones() & 1) << (BITS - 1 - p);
                    }
                    *vj = out;
                }
                x[d] = rng.next() as u32;
            }
        }
        Ok(Sobol {
            dim,
            v,
            x,
            index: 0,
        })
    }

    /// The number of dimensions.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Writes the next point into `out` (of length [`dim`](Self::dim)), each
    /// coordinate in `[0, 1)`. After `2^32` points the sequence starts again.
    ///
    /// # Panics
    ///
    /// When `out` is not `dim` long.
    pub fn next_point(&mut self, out: &mut [f64]) {
        assert_eq!(
            out.len(),
            self.dim,
            "the point has {} coordinates",
            self.dim
        );
        for (o, &x) in out.iter_mut().zip(&self.x) {
            *o = f64::from(x) * SCALE;
        }
        let c = (self.index as u32).trailing_ones() as usize;
        if c < BITS {
            for (d, x) in self.x.iter_mut().enumerate() {
                *x ^= self.v[d * BITS + c];
            }
        }
        self.index = self.index.wrapping_add(1);
    }

    /// The next `n` points, row by row (`n × dim`).
    pub fn points(&mut self, n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n * self.dim];
        for row in out.chunks_exact_mut(self.dim) {
            self.next_point(row);
        }
        out
    }
}

/// Vigna's SplitMix64, for the scramble.
pub(crate) struct SplitMix(pub(crate) u64);

impl SplitMix {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_points_are_balanced() {
        for scramble in [Scramble::None, Scramble::Lms { seed: 7 }] {
            let mut s = Sobol::new(5, scramble).unwrap();
            let p = s.points(1 << 8);
            for d in 0..5 {
                let mut seen = [false; 256];
                for row in p.chunks_exact(5) {
                    let k = (row[d] * 256.0) as usize;
                    assert!(!seen[k], "two points in one interval");
                    seen[k] = true;
                }
            }
        }
    }

    #[test]
    fn the_scramble_is_reproducible_and_changes_the_points() {
        let a = Sobol::new(3, Scramble::Lms { seed: 1 }).unwrap().points(64);
        let b = Sobol::new(3, Scramble::Lms { seed: 1 }).unwrap().points(64);
        let c = Sobol::new(3, Scramble::Lms { seed: 2 }).unwrap().points(64);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(Sobol::new(0, Scramble::None).is_err());
        assert!(Sobol::new(MAX_DIM + 1, Scramble::None).is_err());
        assert!(Sobol::new(MAX_DIM, Scramble::None).is_ok());
    }

    #[test]
    fn scrambled_means_are_a_half() {
        // The mean over 2^12 points of each coordinate.
        let mut s = Sobol::new(20, Scramble::Lms { seed: 3 }).unwrap();
        let p = s.points(1 << 12);
        for d in 0..20 {
            let m: f64 = p.chunks_exact(20).map(|r| r[d]).sum::<f64>() / 4096.0;
            assert!((m - 0.5).abs() < 1e-3, "{d} {m}");
        }
    }
}
