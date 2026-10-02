//! Aggregate loss and reinsurance.
//!
//! Compound (frequency-severity) distributions by Panjer's recursion, FFT
//! and Monte Carlo, with reinsurance layers and towers to follow. See
//! `docs/design/aggregate.md`.
//!
//! This crate holds no numerics of its own beyond the aggregation
//! algorithms: distributions, grids and risk measures come from `act-prob`.

pub mod compound;
pub mod fft;
pub mod monte_carlo;
pub mod panjer;

pub use compound::{CompoundMethod, CompoundReport};
pub use fft::fft;
pub use monte_carlo::{EventSet, simulate_events};
pub use panjer::panjer;
