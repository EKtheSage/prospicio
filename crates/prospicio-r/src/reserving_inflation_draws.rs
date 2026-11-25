//! `index_paths()` and `reinflate_draws()` (issue #979,
//! `docs/design/reserving-inflation.md`, PR 8, decision 11): stochastic
//! re-inflation of simulated future cells, over
//! `prospicio_reserving::inflation_draws`.
//!
//! The R side holds the paths as a first period and a matrix in an S7
//! object; every function here takes them and rebuilds the Rust values.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_core::Period;
use prospicio_prob::{ComponentKey, KeyValue, PredictiveDistribution as PdInner};
use prospicio_reserving::IndexPaths;

use crate::distributions::PredictiveDistribution;
use crate::reserving::to_r;
use crate::reserving_inflation::{indexation, parse_period, period};

/// Paths from their first period, the number of simulations and the
/// values, simulation-major.
fn paths(first: &str, n_sims: i32, values: Vec<f64>) -> Result<IndexPaths> {
    let n_sims = usize::try_from(n_sims)
        .map_err(|_| Error::Other(format!("n_sims must not be negative, got {n_sims}")))?;
    IndexPaths::new(period(first, "first")?, n_sims, values).map_err(to_r)
}

/// The paths' periods, checking the paths.
#[extendr]
fn index_paths_periods_rust(first: &str, n_sims: i32, values: Vec<f64>) -> Result<Vec<String>> {
    Ok(paths(first, n_sims, values)?
        .periods()
        .iter()
        .map(ToString::to_string)
        .collect())
}

/// The components with each `origin` key a period: a label (`"2020"`,
/// `"2020Q3"`) or a whole year, as R key columns hold them. Other keys
/// are kept; a key that is no period is left for Rust to reject.
fn origin_periods(draws: &PdInner) -> Option<Vec<ComponentKey>> {
    let at = draws.dims().iter().position(|d| d == "origin")?;
    let year = |y: i64| i32::try_from(y).ok().map(Period::year);
    Some(
        draws
            .components()
            .iter()
            .map(|key| {
                let mut key = key.clone();
                let converted = match &key[at] {
                    KeyValue::Text(s) => parse_period(s),
                    KeyValue::Int(y) => year(*y),
                    KeyValue::Period(_) => None,
                };
                if let Some(p) = converted {
                    key[at] = KeyValue::Period(p);
                }
                key
            })
            .collect(),
    )
}

/// The draws re-inflated, with the index's future rate (`paths_first`
/// `NULL`) or with one index path per simulation.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn reinflate_draws_rust(
    draws: &PredictiveDistribution,
    first: &str,
    values: Vec<f64>,
    rate: Nullable<f64>,
    base: &str,
    paths_first: Nullable<String>,
    n_sims: i32,
    path_values: Vec<f64>,
) -> Result<PredictiveDistribution> {
    let ix = indexation(first, values, rate, base)?;
    let paths = match paths_first {
        Nullable::NotNull(p) => Some(paths(&p, n_sims, path_values)?),
        Nullable::Null => None,
    };
    let input = &draws.inner;
    let keyed;
    let draws = match origin_periods(input) {
        Some(components) if components.as_slice() != input.components() => {
            keyed = PdInner::from_draws(
                input.dims().to_vec(),
                components,
                input.draw_matrix().to_vec(),
                input.provenance().clone(),
            )
            .map_err(crate::to_r)?;
            &keyed
        }
        _ => input,
    };
    let out = ix.reinflate_draws(draws, paths.as_ref()).map_err(to_r)?;
    // The caller's keys back, as they went in.
    let inner = PdInner::from_draws(
        out.dims().to_vec(),
        input.components().to_vec(),
        out.draw_matrix().to_vec(),
        out.provenance().clone(),
    )
    .map_err(crate::to_r)?;
    Ok(PredictiveDistribution { inner })
}

extendr_module! {
    mod reserving_inflation_draws;
    fn index_paths_periods_rust;
    fn reinflate_draws_rust;
}
