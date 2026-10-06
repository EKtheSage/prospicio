//! Aggregate loss and reinsurance.
//!
//! The collective (frequency-severity) model with closed-form layer
//! moments, compound distributions by Panjer's recursion, FFT and Monte
//! Carlo, and reinsurance layers and towers applied to simulated events
//! or, exactly, to the aggregate grid. See
//! `docs/design/aggregate.md`.
//!
//! This crate holds no numerics of its own beyond the aggregation
//! algorithms: distributions, grids and risk measures come from `act-prob`.

pub mod collective;
pub mod compound;
pub mod fft;
pub mod grid_reinsurance;
pub mod monte_carlo;
pub mod panjer;
pub mod reinsurance;
pub mod serial;

pub use collective::CollectiveModel;
pub use compound::{CompoundMethod, CompoundReport};
pub use fft::fft;
pub use grid_reinsurance::TowerGrids;
pub use monte_carlo::{EventSet, simulate_events};
pub use panjer::panjer;
pub use reinsurance::{Basis, Layer, Tower};
