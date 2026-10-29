//! Pricing on top of `prospicio-prob` severities and `prospicio-aggregate` models.
//!
//! - [`bonus_malus`]: bonus-malus and no-claim-discount scales as
//!   Markov chains: stationary levels, Norberg's optimal relativities
//!   over a mixed Poisson portfolio and Loimaranta's efficiency.
//! - [`bounds`]: pricing bounds, the range of another risk's price over
//!   the distortions that price a reference risk at a premium, and the
//!   envelope of those distortions (Mildenhall 2022).
//! - [`casualty`]: casualty exposure rating over a table of policies, each
//!   with its attachment, limit, share, premium, ELR and curve (a severity,
//!   an ILF table or an ELF table), in the ILF and frequency forms with an
//!   explicit ALAE treatment (Clark's casualty exposure rating).
//! - [`classical`]: the classical premium principles (expected value,
//!   variance, standard deviation, semi-variance, exponential, Esscher,
//!   Dutch, Fischer, VaR), each calibrated to a premium.
//! - [`credibility`]: classical credibility, the limited fluctuation
//!   standards and the Bühlmann–Straub model.
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
//! - [`returns`]: return metrics on simulated cash flows: NPV, the IRR
//!   with a flag for whether a root was found, MOIC, leverage and a fee
//!   tiered on the margin.
//! - [`risk_load`]: risk-loaded prices from simulated losses (a distortion
//!   or a cost of capital on distortion-measured assets), for one cover
//!   or allocated across a portfolio's components.
//! - [`ratemaking`]: premium at the current rate level by the
//!   parallelogram method, and trend: fits, factors and the average written
//!   and earned dates that set trend lengths (Werner and Modlin).
//! - [`tower`]: matching a tower of reinsurance layer prices with
//!   one collective model (Riegel 2018).
//!
//! See `docs/design/pareto.md`.

pub mod bonus_malus;
pub mod book;
pub mod bounds;
pub mod cashflow;
pub mod casualty;
pub mod classical;
pub mod credibility;
pub mod experience;
pub mod exposure;
pub mod frontier;
pub mod ils;
pub mod layer;
pub mod natural;
pub mod payout;
pub mod profile;
pub mod quote;
pub mod ratemaking;
pub mod returns;
pub mod risk_load;
pub mod tower;
