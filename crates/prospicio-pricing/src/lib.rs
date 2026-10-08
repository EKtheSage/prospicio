//! Pricing on top of `prospicio-prob` severities and `prospicio-aggregate` models.
//!
//! - [`classical`]: the classical premium principles (expected value,
//!   variance, standard deviation, semi-variance, exponential, Esscher,
//!   Dutch, Fischer, VaR), each calibrated to a premium.
//! - [`natural`]: pricing with limited liability and the natural
//!   allocation of Mildenhall and Major, with Bodoff's allocation, EPD,
//!   the premium pentagon and pricing bounds.
//! - [`exposure`]: exposure curves for property per-risk rating, the
//!   MBBEFD class (Bernegger 1997) with the Swiss Re curves, and the curve
//!   of any severity capped at a maximum possible loss.
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

pub mod classical;
pub mod exposure;
pub mod layer;
pub mod natural;
pub mod profile;
pub mod risk_load;
pub mod tower;
