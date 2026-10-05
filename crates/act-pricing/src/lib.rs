//! Pricing on top of `act-prob` severities and `act-aggregate` models.
//!
//! - [`layer`]: rating limits and layers, shared by primary pricing
//!   (increased limit factors, deductibles) and reinsurance pricing:
//!   Pareto extrapolation between layers and the alphas implied by two
//!   layers, a frequency and a layer, or two frequencies.
//! - [`risk_load`]: risk-loaded prices from simulated losses (a distortion
//!   or a cost of capital on distortion-measured assets), for one cover
//!   or allocated across a portfolio's components.
//! - [`tower`]: matching a tower of reinsurance layer prices with
//!   one collective model (Riegel 2018).
//!
//! See `docs/design/pareto.md`.

pub mod layer;
pub mod risk_load;
pub mod tower;
