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
//! cumulative values with premium, the correlated chain ladder (CRC) and the
//! changing settlement rate model (CSR), with his priors. Its `ultimates`
//! and `reserves` are a [`PredictiveDistribution`] by origin too.
//!
//! # Seeds
//!
//! One `seed` drives a fit, as in the bootstraps. The chains and the
//! predictive simulations each take a seed derived from it,
//! `StreamRng::new(seed, u64::MAX - tag).next_u64()`, with tag 0 for the
//! chains and tag 1 for the simulations (decision 5). Chain `c` then uses
//! stream `c` of its seed and simulation `i` stream `i` of its seed, so no
//! simulation shares a stream with a chain, and a fit replays exactly for
//! a seed whatever the number of threads. [`Meyers`]' pilot run, which sets
//! its sampling scale, takes stream `u64::MAX` of the chains' seed, which
//! no chain reaches.
//!
//! [`PredictiveDistribution`]: prospicio_prob::PredictiveDistribution
//! [`OdpBootstrapFit::reserves`]: crate::OdpBootstrapFit::reserves

mod bf;
mod laplace;
mod meyers;
mod odp;

pub use bf::{BayesBf, BayesBfFit};
pub use meyers::{Meyers, MeyersFit, MeyersModel};
pub use odp::{BayesOdp, BayesOdpFit};

use prospicio_bayes::nuts::ParamSummary;
use prospicio_core::StreamRng;

/// MCMC diagnostics of a fit, with a warning for each threshold of
/// Vehtari et al. (2021) that fails: R-hat above 1.01, bulk or tail ESS
/// below 100 per chain, any divergent transition. A fit with warnings is
/// still returned; check them before relying on it.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics {
    /// The largest R-hat over the parameters; NaN when any parameter's is
    /// NaN (a constant draw).
    pub max_rhat: f64,
    /// The smallest bulk ESS over the parameters; NaN when any is NaN.
    pub min_ess_bulk: f64,
    /// The smallest tail ESS over the parameters; NaN when any is NaN.
    pub min_ess_tail: f64,
    /// Divergent transitions among the kept draws, over all chains.
    pub divergences: usize,
    /// One message per failed threshold, naming the parameter, its value
    /// and the threshold; empty when every threshold holds.
    pub warnings: Vec<String>,
}

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
}

impl Stream {
    fn tag(self) -> u64 {
        match self {
            Self::Chains => 0,
            Self::Predictive => 1,
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
/// seed, a stream its chains (`0..chains`) do not reach, so it takes no tag
/// of [`Stream`].
pub(crate) fn pilot_seed(seed: u64) -> u64 {
    StreamRng::new(derive_seed(seed, Stream::Chains), u64::MAX).next_u64()
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
    }
}
