//! Aggregate loss and reinsurance.
//!
//! Compound (frequency-severity) distributions by Panjer's recursion, with
//! FFT and Monte Carlo to follow, then reinsurance layers and towers. See
//! `docs/design/aggregate.md`.
//!
//! This crate holds no numerics of its own beyond the aggregation
//! algorithms: distributions, grids and risk measures come from `act-prob`.

pub mod compound;
pub mod panjer;

pub use compound::{CompoundMethod, CompoundReport};
pub use panjer::panjer;
