//! Gesmann and Morris' hierarchical compartmental reserving model with one
//! outstanding compartment (`docs/design/reserving-compartmental.md`):
//! paid and outstanding fitted jointly from one set of interpretable
//! parameters, sampled with NUTS on a density written once over
//! [`Scalar`] and differentiated with the chunked dual type (decision 4),
//! or with per-cell duals and a hand-written chain rule when the priors put
//! effects on each cell (model 2).

mod backtest;
mod data;
mod fit;
mod model;
mod priors;
mod results;
mod settings;
#[cfg(test)]
mod subannual_study;
#[cfg(test)]
mod tests;
mod year;

pub use backtest::CompartmentalCandidate;
pub use fit::{CUMULATIVE_SUB_ANNUAL_WARNING, Compartmental};
pub use priors::{
    CompartmentalPriors, HalfPrior, MAX_INCURRED_ONLY_KP_SD, PositivePrior, RealPrior,
};
pub use results::{
    CompartmentalFit, CompartmentalFits, CompartmentalLoo, CompartmentalOneYearFits,
    CompartmentalPaths, CompartmentalSegment,
};
pub use settings::{
    CompartmentalLikelihood, CompartmentalStart, DevelopmentLevels, Fitted, Outstanding,
    PaidResponse, RateOrder, Scale,
};

use super::scalar::{Scalar, exprel_neg};
#[cfg(test)]
use data::Response;
use data::{Data, Future};
use fit::cumulative_sub_annual_warning;
use model::{ATANH_RHO, B_KP, B_RLR, B_RRF, KER, KP, LOG_TAU, Model, Obs, RLR, RRF};
use priors::lognormal;

/// The one-compartment closed form (decision 2) per unit of premium:
/// `OS(t) / (Π RLR) = k_er g(t)` and
/// `PD(t) / (Π RLR RRF) = (1 - e^(-k_p t)) - k_p g(t)`, with
/// `g(t) = t e^(-k_p t) (1 - e^(-Δ t)) / (Δ t)` and `Δ = k_er - k_p`,
/// finite and smooth through `k_er = k_p` by [`exprel_neg`].
pub(crate) fn one_os<S: Scalar>(k_er: S, k_p: S, t: f64) -> (S, S) {
    let decay = k_p * (-t);
    let g = decay.exp() * t * exprel_neg((k_er - k_p) * t);
    (k_er * g, -decay.exp_m1() - k_p * g)
}
