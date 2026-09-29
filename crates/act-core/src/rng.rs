//! Stream-indexed random numbers.
//!
//! Every simulation index draws from its own stream, identified by
//! `(seed, stream)`. Results are therefore bit-identical regardless of how
//! the work is split across threads, and any single simulation can be
//! replayed on its own. See `docs/design/rng.md`.

use rand_chacha::ChaCha20Rng;
use rand_chacha::rand_core::{Rng, SeedableRng};

/// A reproducible random-number stream: ChaCha20 (original 64-bit counter,
/// 64-bit nonce layout) keyed by `seed`, with the stream id as the nonce.
/// Each stream has 2^64 blocks, far more than any simulation needs.
///
/// The output for a given `(seed, stream)` is part of the crate's stability
/// contract; changing it requires a deliberate, documented version bump.
///
/// # Example
///
/// ```
/// use act_core::StreamRng;
///
/// let mut a = StreamRng::new(42, 7);
/// let mut b = StreamRng::new(42, 7);
/// assert_eq!(a.next_u64(), b.next_u64());
/// ```
#[derive(Debug, Clone)]
pub struct StreamRng {
    inner: ChaCha20Rng,
}

impl StreamRng {
    /// Stream `stream` of the generator keyed by `seed`.
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut inner = ChaCha20Rng::from_seed(expand_seed(seed));
        inner.set_stream(stream);
        Self { inner }
    }

    /// Next uniformly distributed 64-bit integer.
    pub fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    /// Next uniform draw strictly inside `(0, 1)`, on a grid of spacing
    /// `2^-53`, so it can be passed to any quantile function without
    /// producing an infinite value.
    pub fn next_open01(&mut self) -> f64 {
        let k = self.next_u64() >> 11;
        (k as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }
}

/// Expands a 64-bit seed to a 256-bit ChaCha key with SplitMix64, so that
/// nearby seeds give unrelated keys. Owned here rather than delegated to
/// `rand_core` so the mapping cannot change with a dependency update.
fn expand_seed(seed: u64) -> [u8; 32] {
    let mut state = seed;
    let mut key = [0u8; 32];
    for chunk in key.chunks_exact_mut(8) {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        chunk.copy_from_slice(&z.to_le_bytes());
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    fn first_draws(seed: u64, stream: u64, n: usize) -> Vec<u64> {
        let mut rng = StreamRng::new(seed, stream);
        (0..n).map(|_| rng.next_u64()).collect()
    }

    #[test]
    fn same_seed_and_stream_replays() {
        assert_eq!(first_draws(1, 2, 16), first_draws(1, 2, 16));
    }

    #[test]
    fn streams_and_seeds_differ() {
        assert_ne!(first_draws(1, 0, 4), first_draws(1, 1, 4));
        assert_ne!(first_draws(1, 0, 4), first_draws(2, 0, 4));
    }

    #[test]
    fn output_is_pinned() {
        // Golden values, reproduced independently with Python's
        // `cryptography` ChaCha20 (key from the same SplitMix64 expansion,
        // 16-byte nonce = zero counter || stream id, little-endian). If this
        // fails, the stream contract has changed.
        assert_eq!(
            first_draws(42, 0, 3),
            [
                693385945204756564,
                16436763086163553629,
                3187728548114239752
            ]
        );
        assert_eq!(
            first_draws(42, 5, 2),
            [2382546587937276428, 2377724087087505542]
        );
    }

    #[test]
    fn open01_is_strictly_inside() {
        let mut rng = StreamRng::new(0, 0);
        for _ in 0..10_000 {
            let u = rng.next_open01();
            assert!(u > 0.0 && u < 1.0);
        }
    }

    /// Sum of each simulation's draws, computed on a pool of `threads`.
    fn simulate(threads: usize) -> Vec<f64> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            (0..1_000u64)
                .into_par_iter()
                .map(|sim| {
                    let mut rng = StreamRng::new(7, sim);
                    (0..100).map(|_| rng.next_open01()).sum()
                })
                .collect()
        })
    }

    #[test]
    fn identical_across_thread_counts() {
        let one = simulate(1);
        assert_eq!(one, simulate(4));
        assert_eq!(one, simulate(16));
    }
}
