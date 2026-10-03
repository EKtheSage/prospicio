//! Risk mathematics: the hub every domain reads and writes.
//!
//! Holds the [`Distribution`] and [`Severity`] traits, the parametric [`Lognormal`], the
//! sampled representation [`Sampled`], the shared [`risk`] measures,
//! [`Distortion`] risk measures and their allocation, [`copula`]s, extreme
//! value tails ([`evt`]) and [`PredictiveDistribution`], the joint result
//! every model returns. The
//! discretized representation follows `docs/design/distributions.md`.
//!
//! With the `arrow` feature, [`PredictiveDistribution`] reads and writes
//! Arrow IPC files; the format is described in `ipc`.

pub mod capital;
pub mod copula;
pub mod counting;
pub mod distortion;
pub mod distribution;
pub mod evt;
pub mod grid;
#[cfg(feature = "arrow")]
pub mod ipc;
pub mod large_losses;
pub mod local_pareto;
pub mod lognormal;
pub mod pareto;
pub mod piecewise_pareto;
pub mod predictive;
pub mod provenance;
pub mod risk;
pub mod sampled;
pub mod severity;

pub use copula::{Archimedean, ArchimedeanCopula, Copula, GaussianCopula, StudentTCopula};
pub use counting::{Binomial, Counting, NegativeBinomial, PanjerClass, Poisson};
pub use distortion::Distortion;
pub use distribution::Distribution;
pub use grid::{Discretization, DiscretizationReport, Grid};
pub use large_losses::LargeLosses;
pub use local_pareto::{
    LocalParetoApproximation, LocalParetoConversion, LogAffinePareto, local_pareto_to_piecewise,
};
pub use lognormal::Lognormal;
pub use pareto::Pareto;
pub use piecewise_pareto::{PiecewisePareto, Truncation};
pub use predictive::{ComponentKey, KeyValue, PredictiveDistribution};
pub use provenance::{InputHasher, Provenance};
pub use sampled::{Empirical, Sampled};
pub use severity::Severity;
