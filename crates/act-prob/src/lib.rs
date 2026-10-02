//! Risk mathematics: the hub every domain reads and writes.
//!
//! Holds the [`Distribution`] and [`Severity`] traits, the parametric [`Lognormal`], the
//! sampled representation [`Sampled`], the shared [`risk`] measures and
//! [`PredictiveDistribution`], the joint result every model returns. The
//! discretized representation follows `docs/design/distributions.md`.
//!
//! With the `arrow` feature, [`PredictiveDistribution`] reads and writes
//! Arrow IPC files; the format is described in `ipc`.

pub mod counting;
pub mod distortion;
pub mod distribution;
pub mod grid;
#[cfg(feature = "arrow")]
pub mod ipc;
pub mod lognormal;
pub mod predictive;
pub mod provenance;
pub mod risk;
pub mod sampled;
pub mod severity;

pub use counting::{Counting, NegativeBinomial, Poisson};
pub use distortion::Distortion;
pub use distribution::Distribution;
pub use grid::{Discretization, DiscretizationReport, Grid};
pub use lognormal::Lognormal;
pub use predictive::{ComponentKey, KeyValue, PredictiveDistribution};
pub use provenance::{InputHasher, Provenance};
pub use sampled::{Empirical, Sampled};
pub use severity::Severity;
