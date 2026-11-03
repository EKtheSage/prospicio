//! The posterior predictive loss ratio of an underwriting year that is not
//! yet in the triangle (`docs/design/reserving-bayes.md`, decision 27,
//! issue #598): [`BayesBfNormalisedFit::new_year`](super::BayesBfNormalisedFit::new_year)
//! and [`MeyersFit::new_year`](super::MeyersFit::new_year).
//!
//! A deal is priced on a year that has not been written. Each simulation
//! takes one posterior draw of the fit and draws the new origin's ultimate
//! from it, parameter and process uncertainty both, so the result carries
//! what the triangle says about the level of the coming year and no more.
//! What it says depends on the model: see each method.
//!
//! The development pattern of the posterior draw each simulation took comes
//! with it, row `i` simulation `i` of the loss ratio, so
//! [`DevelopmentPatterns::monthly`] gives the new year's emergence by month
//! for the cash-flow engine, paired with its loss ratio.

use prospicio_core::{Lag, Period, StreamRng};
use prospicio_prob::{Distribution, Gamma, KeyValue, PredictiveDistribution, Provenance};

use crate::error::{Error, Result};
use crate::pattern_draws::{DevelopmentPatterns, origin_grain, predictive_keyed_with_shares};

/// The posterior predictive of one new underwriting year, the origin after
/// the triangle's newest.
#[derive(Debug, Clone)]
pub struct NewYear {
    /// The new origin period: the one after the triangle's newest, of the
    /// same grain.
    pub origin: Period,
    /// The new year's premium (its exposure for
    /// [`BayesBfNormalised`](super::BayesBfNormalised)).
    pub premium: f64,
    /// The ultimate loss ratio of the new year: dimension `origin`, one
    /// component, the new origin. Parameter and process uncertainty.
    pub loss_ratio: PredictiveDistribution,
    /// The ultimate loss of the new year, `loss_ratio` times `premium`,
    /// simulation by simulation, with the same layout.
    pub ultimate: PredictiveDistribution,
    /// The development pattern of the posterior draw each simulation took,
    /// row `i` simulation `i` of `loss_ratio` (the fit's own pattern draws'
    /// definition, `reserving-v02.md` decision 28): the cumulative share of
    /// the ultimate at each of the triangle's ages. Its
    /// [`monthly`](DevelopmentPatterns::monthly) is the emergence by month.
    pub patterns: DevelopmentPatterns,
}

/// The origin period after the newest of `origins`.
pub(crate) fn next_origin(origins: &[Period]) -> Result<Period> {
    let last = origins
        .last()
        .copied()
        .ok_or_else(|| Error::Bayes("the triangle has no origin".into()))?;
    Ok(Period::containing(last.end().add_months(1), last.grain()))
}

/// Checks the arguments common to every model's `new_year`.
pub(crate) fn validate(premium: f64, n_sims: usize) -> Result<()> {
    if !(premium.is_finite() && premium > 0.0) {
        return Err(Error::InvalidSetting {
            name: "premium",
            value: premium,
            expected: "a finite, positive premium for the new year",
        });
    }
    if n_sims == 0 {
        return Err(Error::InvalidSetting {
            name: "n_sims",
            value: 0.0,
            expected: "a positive number of simulations",
        });
    }
    Ok(())
}

/// The new year from `simulate(theta, rng, shares) -> ultimate`, which
/// draws the new origin's ultimate loss from the posterior draw `theta` and
/// writes the draw's cumulative shares at the triangle's `ages` into
/// `shares` without drawing; `origins` are the triangle's.
#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate<F>(
    posterior: &prospicio_bayes::nuts::PosteriorDraws,
    origins: &[Period],
    ages: &[Lag],
    premium: f64,
    n_sims: usize,
    seed: u64,
    provenance: Provenance,
    simulate: F,
) -> Result<NewYear>
where
    F: Fn(&[f64], &mut StreamRng, &mut [f64]) -> f64 + Sync,
{
    let origin = next_origin(origins)?;
    let provenance = provenance
        .param("premium", premium)
        .param("new_year_sims", n_sims)
        .param("new_year_seed", seed);
    let (ultimate, patterns) = predictive_keyed_with_shares(
        posterior,
        "origin",
        vec![vec![KeyValue::from(origin)]],
        origin_grain(origins),
        ages.to_vec(),
        n_sims,
        super::derive_seed(seed, super::Stream::NewYear),
        provenance.param("output", "ultimate"),
        |theta, rng, values, shares| values[0] = simulate(theta, rng, shares),
    )?;
    let mut provenance = ultimate.provenance().clone();
    for (key, value) in &mut provenance.parameters {
        if key == "output" {
            *value = "loss_ratio".into();
        }
    }
    let loss_ratio = PredictiveDistribution::from_draws(
        ultimate.dims().to_vec(),
        ultimate.components().to_vec(),
        ultimate.draw_matrix().iter().map(|u| u / premium).collect(),
        provenance,
    )?;
    Ok(NewYear {
        origin,
        premium,
        loss_ratio,
        ultimate,
        patterns,
    })
}

/// One draw of a gamma with `shape` and `scale`, both finite and positive.
pub(crate) fn gamma_draw(shape: f64, scale: f64, rng: &mut StreamRng) -> f64 {
    Gamma::new(shape, scale)
        .expect("shape and scale are finite and positive")
        .sample(rng, 1)[0]
}
