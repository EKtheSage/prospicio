//! Aggregate loss and reinsurance.
//!
//! Compound (frequency-severity) distributions by Panjer's recursion and
//! FFT, with Monte Carlo and reinsurance layers and towers to follow. See
//! `docs/design/aggregate.md`.
//!
//! This crate holds no numerics of its own beyond the aggregation
//! algorithms: distributions, grids and risk measures come from `act-prob`.

pub mod compound;
pub mod fft;
pub mod panjer;

pub use compound::{CompoundMethod, CompoundReport};
pub use fft::fft;
pub use panjer::panjer;
