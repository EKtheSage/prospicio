//! Risk mathematics: the hub every domain reads and writes.
//!
//! Holds the [`Distribution`] trait, the parametric [`Lognormal`], the
//! sampled representation [`Sampled`], the shared [`risk`] measures and
//! [`PredictiveDistribution`], the joint result every model returns. The
//! discretized representation follows `docs/design/distributions.md`.

pub mod distribution;
pub mod lognormal;
pub mod predictive;
pub mod provenance;
pub mod risk;
pub mod sampled;

pub use distribution::Distribution;
pub use lognormal::Lognormal;
pub use predictive::{ComponentKey, KeyValue, PredictiveDistribution};
pub use provenance::Provenance;
pub use sampled::{Empirical, Sampled};
