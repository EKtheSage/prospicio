//! Pricing on top of `act-prob` severities and `act-aggregate` models.
//!
//! - [`layer`]: rating limits and layers from a severity, shared by
//!   primary pricing (increased limit factors, deductibles) and
//!   reinsurance pricing (layer rating).
//! - `tower` (planned): matching a tower of reinsurance layer prices with
//!   one collective model (Riegel 2018).
//!
//! See `docs/design/pareto.md`.

pub mod layer;
