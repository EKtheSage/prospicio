//! Seed-pinned refactor guards for the Bayesian models (#841).
//!
//! Each model has one short seeded fit whose posterior draws and reserve
//! simulations are hashed bit for bit. CI's `bayes` job (Ubuntu x86_64,
//! release) asserts the hash captured there; on any other platform or in a
//! debug build the test fits twice with the same seed and checks that the
//! two runs agree, since another platform's `exp` and `ln` round
//! differently in the last bit and NUTS trajectories then diverge.
//!
//! When a pin's hash moves:
//!
//! - after a change to the model or the sampler path, update the hash in
//!   the same pull request and say why in its description;
//! - after a runner image update with no code change (a new glibc `libm`),
//!   re-pin in a pull request of its own.
//!
//! The failing assertion, and every run with `--nocapture`, prints
//! `PIN <model> actual 0x<16 hex digits>`, the value to paste.

use prospicio_bayes::nuts::PosteriorDraws;
use prospicio_prob::PredictiveDistribution;

/// The FNV-1a offset basis, the hash of no values.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// An FNV-1a hash of the bits of `values`, to pin a run exactly.
pub(crate) fn bits_hash(values: &[f64]) -> u64 {
    bits_hash_from(FNV_OFFSET, values)
}

/// Continues the FNV-1a hash `h` over the bits of `values`.
pub(crate) fn bits_hash_from(h: u64, values: &[f64]) -> u64 {
    values.iter().fold(h, |h, v| {
        (h ^ v.to_bits()).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The hash of a fit's posterior draws, then its reserve simulations.
pub(crate) fn fit_hash(posterior: &PosteriorDraws, reserves: &PredictiveDistribution) -> u64 {
    bits_hash_from(bits_hash(posterior.values()), reserves.draw_matrix())
}

/// Whether this build asserts the pinned hashes: CI's `bayes` job, a
/// release build on Linux x86_64.
const PINNED: bool =
    cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") && !cfg!(debug_assertions);

/// Runs `fit`, which fits a model with a fixed seed and hashes its draws,
/// and checks the hash against `expected` on CI's platform, or against a
/// second run with the same seed elsewhere.
pub(crate) fn check(model: &str, expected: u64, fit: impl Fn() -> u64) {
    let got = fit();
    eprintln!("PIN {model} actual {got:#018x}");
    if PINNED {
        assert_eq!(got, expected, "PIN {model} actual {got:#018x}");
    } else {
        let again = fit();
        assert_eq!(got, again, "PIN {model}: two runs with one seed differ");
    }
}
