//! The model interface and its life cycle, shared by every engine (GLM,
//! GAM, neural networks, boosting adapters, Bayesian models). See
//! `docs/design/models.md`.
//!
//! - [`Family`] and [`Link`]: exponential dispersion families and link
//!   functions, the likelihoods and losses engines share;
//! - [`Terms`], [`Coding`] and [`Design`]: model terms, factor coding
//!   learned on training data, and the design matrix with offset and
//!   weights;
//! - [`Model`] and [`Fitted`]: fit, predict, and joint predictive
//!   distributions;
//! - [`metrics`]: deviance, Gini, lift, CRPS, coverage;
//! - [`resample`]: k-fold, grouped and time-ordered splits,
//!   cross-validation and grid search.
//!
//! This crate has no heavy dependencies; engines live in their own crates.

pub mod compare;
pub mod design;
pub mod family;
pub mod link;
pub mod metrics;
pub mod model;
pub mod monitor;
pub mod resample;
pub mod stack;

pub use design::{Coding, Column, Design, Frame, Term, Terms};
pub use family::Family;
pub use link::Link;
pub use model::{Fitted, Model};
