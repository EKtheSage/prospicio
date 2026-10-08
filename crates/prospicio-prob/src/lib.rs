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

pub mod beta;
pub mod burr;
pub mod capital;
pub mod copula;
pub mod count_families;
pub mod counting;
pub mod custom;
pub mod dist;
pub mod distortion;
pub mod distribution;
pub mod evt;
pub mod gamma;
pub mod grid;
pub mod inverse_gamma;
pub mod inverse_gaussian;
#[cfg(feature = "arrow")]
pub mod ipc;
pub mod large_losses;
pub mod local_pareto;
pub mod loglogistic;
pub mod lognormal;
pub mod mixture;
pub mod pareto;
pub mod piecewise_pareto;
pub mod portfolio;
pub mod predictive;
pub mod provenance;
pub mod risk;
pub mod sampled;
pub mod serial;
pub mod severity;
pub mod truncated;
pub mod tweedie;
pub mod weibull;

pub use beta::Beta;
pub use burr::Burr;
pub use copula::{Archimedean, ArchimedeanCopula, Copula, GaussianCopula, StudentTCopula};
pub use counting::{Binomial, Counting, NegativeBinomial, PanjerClass, Poisson};
pub use custom::Custom;
pub use dist::{Dist, SeverityDist};
pub use distortion::{Distortion, Family};
pub use distribution::Distribution;
pub use gamma::Gamma;
pub use grid::{Discretization, DiscretizationReport, Grid};
pub use inverse_gamma::InverseGamma;
pub use inverse_gaussian::InverseGaussian;
pub use large_losses::LargeLosses;
pub use local_pareto::{
    LocalParetoApproximation, LocalParetoConversion, LogAffinePareto, local_pareto_to_piecewise,
};
pub use loglogistic::Loglogistic;
pub use lognormal::Lognormal;
pub use mixture::Mixture;
pub use pareto::Pareto;
pub use piecewise_pareto::{PiecewisePareto, Truncation};
pub use predictive::{ComponentKey, KeyValue, PredictiveDistribution};
pub use provenance::{InputHasher, Provenance};
pub use sampled::{Empirical, Sampled};
pub use severity::Severity;
pub use truncated::Truncated;
pub use tweedie::Tweedie;
pub use weibull::Weibull;
