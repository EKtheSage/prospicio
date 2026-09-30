//! Risk mathematics: the hub every domain reads and writes.
//!
//! Holds the [`Distribution`] trait, the parametric [`Lognormal`], the
//! sampled representation [`Sampled`] and the shared [`risk`] measures.
//! The discretized representation and `PredictiveDistribution` follow the
//! design in `docs/design/distributions.md` and
//! `docs/design/predictive-distribution.md`.

pub mod distribution;
pub mod lognormal;
pub mod risk;
pub mod sampled;

pub use distribution::Distribution;
pub use lognormal::Lognormal;
pub use sampled::{Empirical, Sampled};
