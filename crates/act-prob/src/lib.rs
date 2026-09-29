//! Risk mathematics: the hub every domain reads and writes.
//!
//! Phase 0 holds the [`Distribution`] trait and one parametric distribution,
//! enough to prove the language bridge. The discretized and sampled
//! representations and `PredictiveDistribution` follow the design in
//! `docs/design/distributions.md` and `docs/design/predictive-distribution.md`.

pub mod distribution;
pub mod lognormal;

pub use distribution::Distribution;
pub use lognormal::Lognormal;
