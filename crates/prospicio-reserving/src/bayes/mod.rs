//! Bayesian reserving, sampled with NUTS from `prospicio-bayes`
//! (`docs/design/reserving-bayes.md`). Behind the `bayes` feature, so the
//! default build (the WASM scope) does not compile the sampler.
//!
//! [`BayesOdp`] is the Bayesian chain ladder of England and Verrall: the
//! over-dispersed Poisson model of the incremental values with log-linear
//! origin and development effects, normal priors, and the chain ladder's
//! Pearson scale plugged in. Its reserves are a [`PredictiveDistribution`]
//! with the layout of [`OdpBootstrapFit::reserves`], so totals, quantiles,
//! capital allocation and blending work on it unchanged.
//!
//! [`Meyers`] holds Meyers' (2019, CAS Monograph 8) log-normal models of
//! cumulative values with premium, the correlated chain ladder (CRC), the
//! changing settlement rate model (CSR) and the correlated accident year
//! model (CAY), with his priors. Its `ultimates` and `reserves` are a
//! [`PredictiveDistribution`] by origin too. [`BayesOdp`] and [`Meyers`]
//! both fit over segments and take part in the diagonal backtest, and
//! [`Compartmental`] fits over segments too.
//!
//! [`Compartmental`] is Gesmann and Morris' (2020) hierarchical
//! compartmental model with one outstanding compartment
//! (`docs/design/reserving-compartmental.md`): paid and outstanding fitted
//! jointly from a reporting rate, a payment rate and each origin's reported
//! loss ratio and reserve robustness factor. Its density is written once
//! over a number type and differentiated with forward-mode dual numbers;
//! its `reserves` are a [`PredictiveDistribution`] by origin.
//!
//! [`BayesOdp::one_year`], [`BayesBf::one_year`], [`Meyers::one_year`] and
//! [`Compartmental::one_year`] give the one-year view, the claims
//! development result over the coming year, by reweighting the posterior
//! draws with Pareto smoothed importance sampling, moment-matching the
//! simulations whose Pareto `k` is too high and refitting those matching
//! cannot rescue ([`BayesOneYear`], decisions 16, 18 and 20). The view is
//! written once over one internal interface that every model's one-year
//! view implements, and a conformance suite runs on each (decision 17).
//!
//! # Seeds
//!
//! One `seed` drives a fit, as in the bootstraps. The chains and the
//! predictive simulations each take a seed derived from it,
//! `StreamRng::new(seed, u64::MAX - tag).next_u64()`, with tag 0 for the
//! chains and tag 1 for the simulations (decision 5). Chain `c` then uses
//! stream `c` of its seed and simulation `i` stream `i` of its seed, so no
//! simulation shares a stream with a chain, and a fit replays exactly for
//! a seed whatever the number of threads. In
//! [`BayesOdp::fit_segments`], [`Meyers::fit_segments`] and
//! [`Compartmental::fit_segments`], segment `k`
//! (in index order) takes tags `2 + 2k` (its chains) and `3 + 2k` (its
//! simulations) instead, so every segment samples and simulates from its
//! own streams and independent segments are independent (decision 11).
//! The one-year view's simulations take the first value of stream
//! `u64::MAX` of the simulations' derived seed (the fit's, or the
//! segment's), and simulation `i` stream `i` of it; a refitted simulation
//! samples from a seed drawn on its own stream. [`Compartmental`]'s view
//! adds standard normal effects for the origins and ages with no observed
//! cell to each posterior draw: draw `d` from stream `d` of the first
//! value of stream `u64::MAX` of the view's seed, which no simulation
//! reaches (decision 20).
//! The pilot run of [`Meyers`] and [`Compartmental`], which sets the
//! sampling scale, takes stream
//! `u64::MAX` of its chains' seed (the segment's, over segments), which no
//! chain reaches.
//!
//! [`PredictiveDistribution`]: prospicio_prob::PredictiveDistribution
//! [`OdpBootstrapFit::reserves`]: crate::OdpBootstrapFit::reserves

#[cfg(test)]
mod autodiff;
mod bf;
mod bf_normalised;
mod compartmental;
mod dual;
mod laplace;
mod meyers;
mod moment_match;
mod odp;
mod one_year;
mod scalar;

pub use crate::diagnostics::Diagnostics;
pub use bf::{BayesBf, BayesBfFit};
pub use bf_normalised::{BayesBfNormalised, BayesBfNormalisedFit};
pub use compartmental::{
    Compartmental, CompartmentalCandidate, CompartmentalFit, CompartmentalFits,
    CompartmentalLikelihood, CompartmentalLoo, CompartmentalOneYearFits, CompartmentalPaths,
    CompartmentalPriors, CompartmentalSegment, CompartmentalStart, Fitted, HalfPrior, Outstanding,
    PaidResponse, PositivePrior, RateOrder, RealPrior, Scale,
};
pub use meyers::{Meyers, MeyersFit, MeyersFits, MeyersModel, MeyersSegment};
pub use odp::{BayesOdp, BayesOdpFit, BayesOdpFits, BayesOdpSegment};
pub use one_year::{
    BayesOneYear, BayesOneYearFit, BayesOneYearFits, BayesOneYearSegment, OneYearRefit,
};

use prospicio_bayes::nuts::ParamSummary;
use prospicio_core::StreamRng;

/// Largest acceptable R-hat.
pub(crate) const MAX_RHAT: f64 = 1.01;
/// Smallest acceptable bulk or tail ESS, per chain.
pub(crate) const MIN_ESS_PER_CHAIN: f64 = 100.0;

impl Diagnostics {
    /// The diagnostics of a posterior summary from `chains` chains with
    /// `divergences` divergent transitions in all.
    pub(crate) fn new(summary: &[ParamSummary], chains: usize, divergences: usize) -> Self {
        let min_ess = MIN_ESS_PER_CHAIN * chains as f64;
        let mut warnings = Vec::new();
        for s in summary {
            // NaN (a constant draw) fails too.
            if s.rhat.is_nan() || s.rhat > MAX_RHAT {
                warnings.push(format!(
                    "{}: R-hat {:.4} is above {MAX_RHAT}",
                    s.name, s.rhat
                ));
            }
            if s.ess_bulk.is_nan() || s.ess_bulk < min_ess {
                warnings.push(format!(
                    "{}: bulk ESS {:.0} is below {min_ess:.0} (100 per chain)",
                    s.name, s.ess_bulk
                ));
            }
            if s.ess_tail.is_nan() || s.ess_tail < min_ess {
                warnings.push(format!(
                    "{}: tail ESS {:.0} is below {min_ess:.0} (100 per chain)",
                    s.name, s.ess_tail
                ));
            }
        }
        if divergences > 0 {
            warnings.push(format!(
                "{divergences} divergent transitions among the kept draws"
            ));
        }
        // A NaN R-hat or ESS (a constant draw) propagates, so the summary
        // fields never look healthier than the warnings.
        let fold = |f: fn(&ParamSummary) -> f64, init: f64, pick: fn(f64, f64) -> f64| {
            summary.iter().map(f).fold(init, |acc, v| {
                if acc.is_nan() || v.is_nan() {
                    f64::NAN
                } else {
                    pick(acc, v)
                }
            })
        };
        Self {
            max_rhat: fold(|s| s.rhat, f64::NEG_INFINITY, f64::max),
            min_ess_bulk: fold(|s| s.ess_bulk, f64::INFINITY, f64::min),
            min_ess_tail: fold(|s| s.ess_tail, f64::INFINITY, f64::min),
            divergences,
            warnings,
        }
    }
}

/// What a derived seed drives (module docs, "Seeds").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stream {
    /// The NUTS chains.
    Chains,
    /// The posterior predictive simulations.
    Predictive,
    /// The NUTS chains of segment `k` of a fit of several segments.
    SegmentChains(usize),
    /// The simulations of segment `k` of a fit of several segments.
    SegmentPredictive(usize),
}

impl Stream {
    fn tag(self) -> u64 {
        match self {
            Self::Chains => 0,
            Self::Predictive => 1,
            Self::SegmentChains(k) => 2 + 2 * k as u64,
            Self::SegmentPredictive(k) => 3 + 2 * k as u64,
        }
    }
}

/// The seed that `stream` of a fit with `seed` uses: the first value of
/// stream `u64::MAX - tag` of `seed`, a stream that neither the chains
/// (`0..chains`) nor the simulations (`0..n_sims`) of any derived seed
/// reach.
pub(crate) fn derive_seed(seed: u64, stream: Stream) -> u64 {
    StreamRng::new(seed, u64::MAX - stream.tag()).next_u64()
}

/// The seed of a pilot run that sets a model's sampling scale (for
/// [`Meyers`]): the first value of stream `u64::MAX` of the chains' derived
/// seed `chains` (a fit's, or a segment's over segments), a stream its
/// chains (`0..chains`) do not reach, so it takes no tag of [`Stream`].
pub(crate) fn pilot_seed(chains: u64) -> u64 {
    StreamRng::new(chains, u64::MAX).next_u64()
}

/// The seed of the one-year view's simulations (decision 16): the first value
/// of stream `u64::MAX` of the predictive simulations' derived seed
/// `predictive` (a fit's, or a segment's over segments), a stream its
/// simulations (`0..n_sims`) do not reach. One-year simulation `i` then uses
/// stream `i` of it, and a refit the chains seed that simulation draws.
pub(crate) fn one_year_seed(predictive: u64) -> u64 {
    StreamRng::new(predictive, u64::MAX).next_u64()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(name: &str, rhat: f64, ess_bulk: f64, ess_tail: f64) -> ParamSummary {
        ParamSummary {
            name: name.into(),
            mean: 0.0,
            sd: 1.0,
            q05: -1.6,
            q50: 0.0,
            q95: 1.6,
            rhat,
            ess_bulk,
            ess_tail,
        }
    }

    #[test]
    fn warnings_name_the_parameter_and_the_threshold() {
        let ok = Diagnostics::new(&[summary("c", 1.001, 3000.0, 2500.0)], 4, 0);
        assert!(ok.warnings.is_empty());
        assert_eq!(ok.max_rhat, 1.001);
        let bad = Diagnostics::new(
            &[
                summary("c", 1.001, 3000.0, 2500.0),
                summary("a[2]", 1.05, 300.0, 399.0),
            ],
            4,
            3,
        );
        assert_eq!(bad.warnings.len(), 4, "{:?}", bad.warnings);
        assert!(bad.warnings[0].starts_with("a[2]: R-hat 1.0500 is above 1.01"));
        assert!(bad.warnings[1].contains("bulk ESS 300 is below 400"));
        assert!(bad.warnings[2].contains("tail ESS 399 is below 400"));
        assert!(bad.warnings[3].starts_with("3 divergent"));
        assert_eq!((bad.min_ess_bulk, bad.min_ess_tail), (300.0, 399.0));
        assert_eq!(bad.divergences, 3);
    }

    #[test]
    fn a_nan_diagnostic_propagates_to_the_summary_fields() {
        let d = Diagnostics::new(
            &[
                summary("c", 1.001, 3000.0, 2500.0),
                summary("a[2]", f64::NAN, f64::NAN, f64::NAN),
                summary("b[2]", 1.002, 2000.0, 1500.0),
            ],
            4,
            0,
        );
        assert!(d.max_rhat.is_nan() && d.min_ess_bulk.is_nan() && d.min_ess_tail.is_nan());
        assert_eq!(d.warnings.len(), 3, "{:?}", d.warnings);
    }

    #[test]
    fn derived_seeds_differ_by_stream_and_seed() {
        let a = derive_seed(0, Stream::Chains);
        assert_eq!(a, derive_seed(0, Stream::Chains));
        assert_ne!(a, derive_seed(0, Stream::Predictive));
        assert_ne!(a, derive_seed(1, Stream::Chains));
        let tags = [
            Stream::Chains,
            Stream::Predictive,
            Stream::SegmentChains(0),
            Stream::SegmentPredictive(0),
            Stream::SegmentChains(1),
            Stream::SegmentPredictive(1),
        ]
        .map(Stream::tag);
        assert_eq!(tags, [0, 1, 2, 3, 4, 5]);
    }
}
