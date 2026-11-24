//! Reserving lane: wrappers over `prospicio_reserving` (the Triangle, chain
//! ladder, Mack with its one-year view, tails, the expected-loss methods,
//! Clark's growth curves and a growth curve fitted to a benchmark pattern, and the ODP and Mack bootstraps with their simulated one-year
//! view, `docs/design/triangle.md`, `docs/design/reserving-v02.md`) for the
//! R `reserving.R` API.
//!
//! Arrays cross the boundary flattened row-major over the Rust axes
//! (index, column, origin, development) with NaN where a cell is not
//! observed; R reshapes them. Months come in as (year, month) integer pairs.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_reserving::{
    AnnualReading, Apriori, Average, BenchmarkCurve, BenchmarkCurveFit as BenchmarkCurveInner,
    Benktander, BornhuetterFerguson, CapeCod, CapeCodFit as CapeCodInner, ChainLadder,
    ChainLadderFit as ChainLadderInner, ClaimsDevelopmentResult as ClaimsDevelopmentInner,
    ClarkCapeCod, ClarkFit as ClarkInner, ClarkLdf, CurveShape, Development, DevelopmentColumn,
    DiagonalAge, ExpectedLoss, ExpectedLossFit as ExpectedLossInner, FileLayout, FitTable, Grain,
    GrowthCurve, Label, Lag, LatestDiagonals as LatestDiagonalsInner, Long, LongDiagonal, Mack,
    MackBootstrap, MackBootstrapFits, MackBootstrapSegment, MackFit as MackInner, MackProcess,
    Month, MunichChainLadder, MunichFit as MunichInner, OdpBootstrap, OdpBootstrapFits,
    OneYearFits, OneYearMethod, OneYearPattern, PastPatternEnd, PatternKind, PatternWeight, Period,
    ProcessDistribution, ReserveFit, SegmentDependence, SegmentFits, SegmentMap, SelectedPattern,
    SigmaInterpolation, Tail, TailBondy, TailConstant, TailCurve, Triangle as TriangleInner,
    UnderwritingBasis,
};
use prospicio_reserving::{
    CalendarYearTest as CalendarYearInner,
    DevelopmentCorrelationTest as DevelopmentCorrelationInner, DevelopmentCurve,
    DevelopmentCurveFit as DevelopmentCurveInner, DevelopmentExhibit as DevelopmentExhibitInner,
    MethodWeights, SelectedUltimates as SelectedUltimatesInner,
    UltimateExhibit as UltimateExhibitInner, UltimateSelection, calendar_year_effect,
    development_correlation, friedland_averages,
};
use prospicio_reserving::{
    PaidIncurredChain, PicConvention, PicFit, PicPriors, VarianceExtrapolation,
};

use prospicio_bayes::nuts::{ParamSummary, Sampler as BayesSampler};
use prospicio_reserving::bayes::{
    BayesOdpFits as BayesOdpFitsInner, BayesOdpSegment, BayesOneYear,
    BayesOneYearFit as BayesOneYearInner, BayesOneYearFits as BayesOneYearFitsInner,
    BayesOneYearSegment, Compartmental, CompartmentalCandidate,
    CompartmentalFit as CompartmentalInner, CompartmentalFits as CompartmentalFitsInner,
    CompartmentalLikelihood, CompartmentalOneYearFits as CompartmentalOneYearFitsInner,
    CompartmentalPriors, CompartmentalSegment, CompartmentalStart, DevelopmentLevels, Fitted,
    HalfPrior, LightRefit, MeyersFits as MeyersFitsInner, MeyersSegment, NewOriginLevel, NewYear,
    OneYearRefit, Outstanding, PaidResponse, PositivePrior, RateOrder, RealPrior,
    Scale as CompartmentalScale,
};
use prospicio_reserving::{
    BayesBf, BayesBfFit as BayesBfInner, BayesBfNormalised,
    BayesBfNormalisedFit as BayesBfNormalisedInner, BayesOdp, BayesOdpFit as BayesOdpInner,
    Diagnostics, HierarchicalBf, HierarchicalBfFit as HierarchicalBfInner, HierarchicalBfProgram,
    Meyers, MeyersFit as MeyersInner, MeyersModel,
};

use prospicio_glm::Glm;
use prospicio_models::{Column, Terms};
use prospicio_reserving::{
    Backtest, GlmCandidate, NamedModel, TriangleFrame, TriangleModel, diagonal_backtest,
};

use crate::distributions::{PredictiveDistribution, key_from_list};
use crate::whole;
use prospicio_prob::ComponentKey;
use prospicio_reserving::{DevelopmentPatterns, Monotone, MonthlyBasis, UltimateDraws};
use prospicio_reserving::{IndustryPrior, IndustryPriorSettings, PatternPrior};

mod basis;
mod chain_ladder;
mod common;
mod curves;
mod exhibits;
mod expected_loss;
mod latest_diagonal;
mod mack;
mod munich;
mod paid_incurred;
mod tails;
mod triangle;

pub(crate) use basis::*;
pub(crate) use chain_ladder::*;
pub(crate) use common::*;
pub(crate) use curves::*;
pub(crate) use exhibits::*;
pub(crate) use expected_loss::*;
pub(crate) use mack::*;
pub(crate) use munich::*;
pub(crate) use paid_incurred::*;
pub(crate) use tails::*;
pub(crate) use triangle::*;

/// An ODP bootstrap of every segment of a triangle column. `fitted` and
/// `residuals` are row-major over origin x development, NaN where not
/// observed.
#[extendr]
pub(crate) struct OdpBootstrapFit {
    pub(crate) inner: OdpBootstrapFits,
}

#[extendr]
impl OdpBootstrapFit {
    /// The chain ladder the bootstrap is centred on, with the bootstrap's
    /// tail.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.segments.map(|s| s.chain_ladder.clone()),
        }
    }

    fn fitted(&self) -> Result<Vec<f64>> {
        let s = single(&self.inner.segments, "fitted", "")?;
        Ok(s.fitted.clone())
    }

    fn residuals(&self) -> Result<Vec<f64>> {
        let s = single(&self.inner.segments, "residuals", "")?;
        Ok(s.residuals.clone())
    }

    fn scale(&self) -> Result<f64> {
        Ok(single(&self.inner.segments, "scale", "totals_frame()")?.scale)
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// The joint ultimates: the reserves with each origin's latest value
    /// added in every simulation (`UltimateDraws::ultimates`).
    fn ultimates(&self) -> Result<PredictiveDistribution> {
        ultimates_of(&self.inner)
    }

    /// One origin's ultimate in each simulation, or the total's with
    /// `origin` NULL, in simulation order.
    fn ultimate_draws(&self, origin: Nullable<List>) -> Result<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// [`ultimate_draws`](Self::ultimate_draws) over `premium`.
    fn loss_ratio_draws(&self, origin: Nullable<List>, premium: f64) -> Result<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// The development ages of a segment's pattern draws, in months
    /// (`DevelopmentPatterns::segment`); `segment` is empty for a fit of
    /// one segment.
    fn pattern_ages(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_ages_of(&self.inner.patterns, &segment)
    }

    /// The pattern draws of a segment, row-major: simulation `i`'s
    /// cumulative share at each age in row `i`.
    fn pattern_draws(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_draws_of(&self.inner.patterns, &segment)
    }

    /// The monthly patterns for the cash-flow engine, row-major: simulation
    /// `i`'s share at the end of each of `months` months in row `i`.
    fn monthly_pattern_draws(
        &self,
        segment: Vec<String>,
        months: f64,
        basis: Robj,
        monotone: &str,
    ) -> Result<Vec<f64>> {
        monthly_pattern_draws_of(&self.inner.patterns, &segment, months, &basis, monotone)
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn development_table(&self) -> List {
        fit_table(self.inner.development_table())
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: self.inner.segment(&choice(&keys, &values)?).map_err(to_r)?,
        })
    }
}

/// Mack's bootstrap, the lifetime view, of every segment of a triangle
/// column. `residuals` is row-major over origin x development, NaN where
/// there is no residual.
#[extendr]
pub(crate) struct MackBootstrapFit {
    pub(crate) inner: MackBootstrapFits,
}

#[extendr]
impl MackBootstrapFit {
    /// The chain ladder of Mack's model.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.segments.map(|s| s.mack.chain_ladder.clone()),
        }
    }

    /// Mack's model on the observed triangle, with the bootstrap's tail.
    fn mack(&self) -> MackFit {
        MackFit {
            inner: self.inner.segments.map(|s| s.mack.clone()),
        }
    }

    fn residuals(&self) -> Result<Vec<f64>> {
        let s = single(&self.inner.segments, "residuals", "")?;
        Ok(s.residuals.clone())
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// The joint ultimates: the reserves with each origin's latest value
    /// added in every simulation (`UltimateDraws::ultimates`).
    fn ultimates(&self) -> Result<PredictiveDistribution> {
        ultimates_of(&self.inner)
    }

    /// One origin's ultimate in each simulation, or the total's with
    /// `origin` NULL, in simulation order.
    fn ultimate_draws(&self, origin: Nullable<List>) -> Result<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// [`ultimate_draws`](Self::ultimate_draws) over `premium`.
    fn loss_ratio_draws(&self, origin: Nullable<List>, premium: f64) -> Result<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// The development ages of a segment's pattern draws, in months
    /// (`DevelopmentPatterns::segment`); `segment` is empty for a fit of
    /// one segment.
    fn pattern_ages(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_ages_of(&self.inner.patterns, &segment)
    }

    /// The pattern draws of a segment, row-major: simulation `i`'s
    /// cumulative share at each age in row `i`.
    fn pattern_draws(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_draws_of(&self.inner.patterns, &segment)
    }

    /// The monthly patterns for the cash-flow engine, row-major: simulation
    /// `i`'s share at the end of each of `months` months in row `i`.
    fn monthly_pattern_draws(
        &self,
        segment: Vec<String>,
        months: f64,
        basis: Robj,
        monotone: &str,
    ) -> Result<Vec<f64>> {
        monthly_pattern_draws_of(&self.inner.patterns, &segment, months, &basis, monotone)
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn development_table(&self) -> List {
        fit_table(self.inner.development_table())
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: self.inner.segment(&choice(&keys, &values)?).map_err(to_r)?,
        })
    }
}

/// The simulated one-year view of every segment of a triangle column:
/// each segment's bootstrap and opening ultimate, and the joint claims
/// development result, from the ODP bootstrap or Mack's.
#[extendr]
pub(crate) struct OneYearFit {
    inner: OneYear,
}

/// A one-year view from either bootstrap.
enum OneYear {
    Odp(OneYearFits),
    Mack(OneYearFits<MackBootstrapSegment>),
}

/// `$body` on whichever one-year view `$fit` holds, bound to `$f`.
macro_rules! each_one_year {
    ($fit:expr, $f:ident => $body:expr) => {
        match $fit {
            OneYear::Odp($f) => $body,
            OneYear::Mack($f) => $body,
        }
    };
}

#[extendr]
impl OneYearFit {
    /// "odp" or "mack".
    fn model(&self) -> &'static str {
        match self.inner {
            OneYear::Odp(_) => "odp",
            OneYear::Mack(_) => "mack",
        }
    }

    /// The bootstrap's chain ladder: volume-weighted for the ODP, Mack's
    /// averaging for Mack's.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: each_one_year!(&self.inner, f => f
                .segments
                .map(|s| s.bootstrap.chain_ladder().clone())),
        }
    }

    /// Mack's model behind `mack_one_year()`.
    fn mack(&self) -> Result<MackFit> {
        match &self.inner {
            OneYear::Mack(f) => Ok(MackFit {
                inner: f.segments.map(|s| s.bootstrap.mack.clone()),
            }),
            OneYear::Odp(_) => Err(Error::Other(
                "mack is Mack's bootstrap's model; this one-year view is the ODP's".into(),
            )),
        }
    }

    fn opening_ultimate(&self) -> Vec<f64> {
        each_one_year!(&self.inner, f => by_origin(&f.segments, |s| s.opening_ultimate.clone()))
    }

    fn opening_reserve(&self) -> Vec<f64> {
        each_one_year!(&self.inner, f => by_origin(&f.segments, |s| s.opening_reserve.clone()))
    }

    fn scale(&self) -> Result<f64> {
        match &self.inner {
            OneYear::Odp(f) => Ok(single(&f.segments, "scale", "totals_frame()")?
                .bootstrap
                .scale),
            OneYear::Mack(_) => Err(Error::Other(
                "scale is the ODP bootstrap's; this one-year view is Mack's".into(),
            )),
        }
    }

    fn cdr(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: each_one_year!(&self.inner, f => f.cdr.clone()),
        }
    }

    fn long_table(&self) -> List {
        fit_table(each_one_year!(&self.inner, f => f.to_long()))
    }

    fn totals_table(&self) -> List {
        fit_table(each_one_year!(&self.inner, f => f.totals()))
    }

    fn development_table(&self) -> List {
        fit_table(each_one_year!(&self.inner, f => f.segments.development_table()))
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let keys = choice(&keys, &values)?;
        let inner = match &self.inner {
            OneYear::Odp(f) => OneYear::Odp(f.segment(&keys).map_err(to_r)?),
            OneYear::Mack(f) => OneYear::Mack(f.segment(&keys).map_err(to_r)?),
        };
        Ok(Self { inner })
    }
}

/// A Bayesian ODP chain ladder of a single-segment triangle column.
#[extendr]
pub(crate) struct BayesOdpFit {
    pub(crate) inner: BayesOdpInner,
}

#[extendr]
impl BayesOdpFit {
    /// The volume-weighted chain ladder the model takes its latest values
    /// and scale from.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.inner.chain_ladder.clone()],
            },
        }
    }

    fn scale(&self) -> f64 {
        self.inner.scale
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// The joint ultimates: the reserves with each origin's latest value
    /// added in every simulation (`UltimateDraws::ultimates`).
    fn ultimates(&self) -> Result<PredictiveDistribution> {
        ultimates_of(&self.inner)
    }

    /// One origin's ultimate in each simulation, or the total's with
    /// `origin` NULL, in simulation order.
    fn ultimate_draws(&self, origin: Nullable<List>) -> Result<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// [`ultimate_draws`](Self::ultimate_draws) over `premium`.
    fn loss_ratio_draws(&self, origin: Nullable<List>, premium: f64) -> Result<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// The development ages of a segment's pattern draws, in months
    /// (`DevelopmentPatterns::segment`); `segment` is empty for a fit of
    /// one segment.
    fn pattern_ages(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_ages_of(&self.inner.patterns, &segment)
    }

    /// The pattern draws of a segment, row-major: simulation `i`'s
    /// cumulative share at each age in row `i`.
    fn pattern_draws(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_draws_of(&self.inner.patterns, &segment)
    }

    /// The monthly patterns for the cash-flow engine, row-major: simulation
    /// `i`'s share at the end of each of `months` months in row `i`.
    fn monthly_pattern_draws(
        &self,
        segment: Vec<String>,
        months: f64,
        basis: Robj,
        monotone: &str,
    ) -> Result<Vec<f64>> {
        monthly_pattern_draws_of(&self.inner.patterns, &segment, months, &basis, monotone)
    }

    /// `list(max_rhat, min_ess_bulk, min_ess_tail, divergences, warnings)`.
    fn diagnostics(&self) -> List {
        let d = &self.inner.diagnostics;
        list!(
            max_rhat = d.max_rhat,
            min_ess_bulk = d.min_ess_bulk,
            min_ess_tail = d.min_ess_tail,
            divergences = d.divergences as f64,
            warnings = d.warnings.clone()
        )
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, row-major `n_draws x n_parameters`, chain by chain.
    fn draws(&self) -> Vec<f64> {
        self.inner.posterior.values().to_vec()
    }

    fn chains(&self) -> f64 {
        self.inner.posterior.chains() as f64
    }

    /// The posterior summary, columns as `bayes_glm_model@summary`.
    fn summary(&self) -> Result<List> {
        let s = self.inner.summary().map_err(to_r)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        Ok(list!(
            parameter = s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            mean = col(|p| p.mean),
            sd = col(|p| p.sd),
            q05 = col(|p| p.q05),
            q50 = col(|p| p.q50),
            q95 = col(|p| p.q95),
            rhat = col(|p| p.rhat),
            ess_bulk = col(|p| p.ess_bulk),
            ess_tail = col(|p| p.ess_tail)
        ))
    }

    /// PSIS-LOO of the observed increments: `list(estimates, pointwise,
    /// pareto_k, k_threshold)`, as `BayesGlmFit$loo()`.
    fn loo(&self) -> Result<List> {
        let l = self.inner.loo().map_err(to_r)?;
        let e = &l.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.pareto_k,
            k_threshold = l.k_threshold
        ))
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn development_table(&self) -> List {
        fit_table(self.inner.development_table())
    }

    /// The fit itself, as for any single-segment fit without keys: a
    /// condition on a key is an error.
    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let one = SegmentFits {
            key_names: Vec::new(),
            labels: vec![Label::new(Vec::<String>::new())],
            fits: vec![()],
        };
        pick(&one, keys, values)?;
        Ok(Self {
            inner: self.inner.clone(),
        })
    }
}

/// The `BayesOdp` of the one-year views: `priors` are `intercept_sd`,
/// `origin_sd` and `development_sd`; `run` is `chains`, `tune`, `draws`,
/// `n_sims`, `seed`, `target_accept` and `max_depth`.
fn bayes_odp_model(
    priors: &[f64],
    process: &str,
    run: &[f64],
    dependence: SegmentDependence,
) -> Result<BayesOdp> {
    let (&[intercept_sd, origin_sd, development_sd], &[chains, tune, draws, n_sims, seed, ta, md]) =
        (priors, run)
    else {
        return Err(Error::Other(
            "bayes_odp_model takes 3 priors and 7 run settings".into(),
        ));
    };
    let boot = bootstrap(n_sims, seed, process, SegmentDependence::Independent)?;
    Ok(BayesOdp {
        intercept_sd,
        origin_sd,
        development_sd,
        process: boot.process,
        sampler: BayesSampler {
            chains: whole(chains, "chains")? as usize,
            tune: whole(tune, "tune")? as usize,
            draws: whole(draws, "draws")? as usize,
            seed: 0,
            target_accept: ta,
            max_depth: whole(md, "max_depth")?,
        },
        n_sims: boot.n_sims,
        seed: boot.seed,
        dependence,
    })
}

/// The `BayesBf` of `bayes_bf` and `bayes_bf_one_year`: `priors` are
/// `apriori`, `cv` and `development_sd`, `run` is `chains`, `tune`,
/// `draws`, `n_sims`, `seed`, `target_accept` and `max_depth`.
fn bayes_bf_model(priors: &[f64], process: &str, run: &[f64]) -> Result<BayesBf> {
    let (&[apriori, cv, development_sd], &[chains, tune, draws, n_sims, seed, ta, md]) =
        (priors, run)
    else {
        return Err(Error::Other(
            "bayes_bf_model takes 3 priors and 7 run settings".into(),
        ));
    };
    let boot = bootstrap(n_sims, seed, process, SegmentDependence::Independent)?;
    Ok(BayesBf {
        apriori,
        cv,
        development_sd,
        process: boot.process,
        sampler: BayesSampler {
            chains: whole(chains, "chains")? as usize,
            tune: whole(tune, "tune")? as usize,
            draws: whole(draws, "draws")? as usize,
            seed: 0,
            target_accept: ta,
            max_depth: whole(md, "max_depth")?,
        },
        n_sims: boot.n_sims,
        seed: boot.seed,
    })
}

/// The `BayesBfNormalised` of `bayes_bf_normalised` and
/// `bayes_bf_normalised_one_year`: `priors` are
/// `apriori`, `cv`, `development_sd` and `common_cv` (0 for no multiplier
/// common to all origins), `run` is `chains`, `tune`,
/// `draws`, `n_sims`, `seed`, `target_accept` and `max_depth`.
fn bayes_bf_normalised_model(
    priors: &[f64],
    process: &str,
    run: &[f64],
) -> Result<BayesBfNormalised> {
    let (&[apriori, cv, development_sd, common_cv], &[chains, tune, draws, n_sims, seed, ta, md]) =
        (priors, run)
    else {
        return Err(Error::Other(
            "bayes_bf_normalised_model takes 4 priors and 7 run settings".into(),
        ));
    };
    let boot = bootstrap(n_sims, seed, process, SegmentDependence::Independent)?;
    Ok(BayesBfNormalised {
        apriori,
        cv,
        development_sd,
        process: boot.process,
        sampler: BayesSampler {
            chains: whole(chains, "chains")? as usize,
            tune: whole(tune, "tune")? as usize,
            draws: whole(draws, "draws")? as usize,
            seed: 0,
            target_accept: ta,
            max_depth: whole(md, "max_depth")?,
        },
        n_sims: boot.n_sims,
        seed: boot.seed,
        common_cv: (common_cv != 0.0).then_some(common_cv),
    })
}

/// The `HierarchicalBf` of `hierarchical_bf`: `priors` are `apriori`, `cv`,
/// `development_sd`, `tau_scale` and `omega_scale`, `run` is `chains`,
/// `tune`, `draws`, `n_sims`, `seed`, `target_accept` and `max_depth`.
fn hierarchical_bf_model(
    priors: &[f64],
    level_cv: Nullable<f64>,
    process: &str,
    run: &[f64],
) -> Result<HierarchicalBf> {
    let (
        &[apriori, cv, development_sd, tau_scale, omega_scale],
        &[chains, tune, draws, n_sims, seed, ta, md],
    ) = (priors, run)
    else {
        return Err(Error::Other(
            "hierarchical_bf_model takes 5 priors and 7 run settings".into(),
        ));
    };
    let boot = bootstrap(n_sims, seed, process, SegmentDependence::Independent)?;
    Ok(HierarchicalBf {
        apriori,
        level_cv: match level_cv {
            Nullable::NotNull(c) => Some(c),
            Nullable::Null => None,
        },
        cv,
        development_sd,
        tau_scale,
        omega_scale,
        process: boot.process,
        sampler: BayesSampler {
            chains: whole(chains, "chains")? as usize,
            tune: whole(tune, "tune")? as usize,
            draws: whole(draws, "draws")? as usize,
            seed: 0,
            target_accept: ta,
            max_depth: whole(md, "max_depth")?,
        },
        n_sims: boot.n_sims,
        seed: boot.seed,
    })
}

/// The settings of a Bayesian one-year view: `k_threshold`, the `refit`
/// policy by name (`None` for the model's one-year default), the most refits
/// (`NULL` for no limit) and the number
/// of simulations (`NULL` for the model's one-year default).
fn one_year_settings(
    k_threshold: f64,
    refit: Option<&str>,
    max_refits: Nullable<f64>,
    one_year_sims: Nullable<f64>,
) -> Result<BayesOneYear> {
    let refit = match refit {
        None => None,
        Some("refit_flagged") => Some(OneYearRefit::RefitFlagged),
        Some("reweight_only") => Some(OneYearRefit::ReweightOnly),
        Some("refit_all") => Some(OneYearRefit::RefitAll),
        Some("moment_match_then_refit") => Some(OneYearRefit::MomentMatchThenRefit),
        Some(refit) => {
            return Err(Error::Other(format!(
                "refit must be \"moment_match_then_refit\", \"refit_flagged\", \
                 \"reweight_only\" or \"refit_all\", got \"{refit}\""
            )));
        }
    };
    let max_refits = match max_refits {
        Nullable::NotNull(m) => Some(whole(m, "max_refits")? as usize),
        Nullable::Null => None,
    };
    let n_sims = match one_year_sims {
        Nullable::NotNull(n) => Some(whole(n, "one_year_sims")? as usize),
        Nullable::Null => None,
    };
    Ok(BayesOneYear {
        k_threshold,
        refit,
        max_refits,
        n_sims,
        light_refit: None,
        no_light_refit: false,
    })
}

/// A compartmental or hierarchical one-year view's `light_refit`: `NULL`
/// for the model's default (model 2's light refit), the vector `0` to turn
/// it off (the R wrapper sends it for `FALSE`), an empty vector for
/// [`LightRefit::default`] (4 chains of 150 warm-up and 250 kept draws), or
/// `c(chains, tune, draws)`. Returns the sampler and whether the default is
/// turned off.
fn light_refit_arg(light_refit: Nullable<Vec<f64>>) -> Result<(Option<LightRefit>, bool)> {
    match light_refit {
        Nullable::Null => Ok((None, false)),
        Nullable::NotNull(v) if v.is_empty() => Ok((Some(LightRefit::default()), false)),
        Nullable::NotNull(v) if v.len() == 1 && v[0] == 0.0 => Ok((None, true)),
        Nullable::NotNull(v) if v.len() == 3 => Ok((
            Some(LightRefit {
                chains: whole(v[0], "light_refit chains")? as usize,
                tune: whole(v[1], "light_refit tune")? as usize,
                draws: whole(v[2], "light_refit draws")? as usize,
            }),
            false,
        )),
        Nullable::NotNull(_) => Err(Error::Other(
            "light_refit must be NULL, TRUE, FALSE or c(chains, tune, draws)".into(),
        )),
    }
}

/// The one-year view of a Bayesian model on a single-segment triangle
/// column: the lifetime fit, the opening reserve, the claims development
/// result and each simulation's Pareto k.
#[extendr]
pub(crate) struct BayesOneYearFit {
    inner: BayesOneYearOf,
}

/// The one-year view of each Bayesian model with one, by its lifetime fit.
#[derive(Clone)]
enum BayesOneYearOf {
    Odp(BayesOneYearInner<BayesOdpInner>),
    Bf(BayesOneYearInner<BayesBfInner>),
    BfNormalised(BayesOneYearInner<BayesBfNormalisedInner>),
    Meyers(Box<BayesOneYearInner<MeyersInner>>),
}

/// Evaluates `$body` with `$f` bound to the view, whatever its model.
macro_rules! each_bayes_one_year {
    ($view:expr, $f:ident => $body:expr) => {
        match $view {
            BayesOneYearOf::Odp($f) => $body,
            BayesOneYearOf::Bf($f) => $body,
            BayesOneYearOf::BfNormalised($f) => $body,
            BayesOneYearOf::Meyers($f) => $body,
        }
    };
}

#[extendr]
impl BayesOneYearFit {
    /// The model behind the view: "bayes_odp", "bayes_bf",
    /// "bayes_bf_normalised" or "meyers".
    fn model(&self) -> &'static str {
        match &self.inner {
            BayesOneYearOf::Odp(_) => "bayes_odp",
            BayesOneYearOf::Bf(_) => "bayes_bf",
            BayesOneYearOf::BfNormalised(_) => "bayes_bf_normalised",
            BayesOneYearOf::Meyers(_) => "meyers",
        }
    }

    /// The lifetime fit, as `bayes_odp` gives it; an error for another
    /// model.
    fn fit(&self) -> Result<BayesOdpFit> {
        match &self.inner {
            BayesOneYearOf::Odp(f) => Ok(BayesOdpFit {
                inner: f.fit.clone(),
            }),
            _ => Err(Error::Other(format!(
                "the one-year view is {}'s, not bayes_odp's",
                self.model()
            ))),
        }
    }

    /// The lifetime fit, as `bayes_bf` gives it; an error for another
    /// model.
    fn bf_fit(&self) -> Result<BayesBfFit> {
        match &self.inner {
            BayesOneYearOf::Bf(f) => Ok(BayesBfFit {
                inner: f.fit.clone(),
            }),
            _ => Err(Error::Other(format!(
                "the one-year view is {}'s, not bayes_bf's",
                self.model()
            ))),
        }
    }

    /// The lifetime fit, as `bayes_bf_normalised` gives it; an error for
    /// another model.
    fn bf_normalised_fit(&self) -> Result<BayesBfNormalisedFit> {
        match &self.inner {
            BayesOneYearOf::BfNormalised(f) => Ok(BayesBfNormalisedFit {
                inner: f.fit.clone(),
            }),
            _ => Err(Error::Other(format!(
                "the one-year view is {}'s, not bayes_bf_normalised's",
                self.model()
            ))),
        }
    }

    /// The lifetime fit, as `meyers` gives it; an error for another model.
    fn meyers_fit(&self) -> Result<MeyersFit> {
        match &self.inner {
            BayesOneYearOf::Meyers(f) => Ok(MeyersFit {
                inner: f.fit.clone(),
            }),
            _ => Err(Error::Other(format!(
                "the one-year view is {}'s, not meyers'",
                self.model()
            ))),
        }
    }

    fn opening_ultimate(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.opening_ultimate.clone())
    }

    fn opening_reserve(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.opening_reserve.clone())
    }

    fn cdr(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: each_bayes_one_year!(&self.inner, f => f.cdr.clone()),
        }
    }

    fn pareto_k(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.pareto_k.clone())
    }

    /// The simulations refitted with NUTS, 1-based.
    fn refitted(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f
            .refitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect())
    }

    /// The simulations moment matching rescued, 1-based.
    fn moment_matched(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f
            .moment_matched
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect())
    }

    /// The simulations due for a refit beyond `max_refits`, reweighted
    /// instead, 1-based.
    fn unrefitted(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f
            .unrefitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect())
    }

    fn refit_warnings(&self) -> f64 {
        each_bayes_one_year!(&self.inner, f => f.refit_warnings as f64)
    }

    fn k_threshold(&self) -> f64 {
        each_bayes_one_year!(&self.inner, f => f.k_threshold)
    }

    fn flagged_share(&self) -> f64 {
        each_bayes_one_year!(&self.inner, f => f.flagged_share())
    }

    fn long_table(&self) -> List {
        fit_table(each_bayes_one_year!(&self.inner, f => f.to_long()))
    }

    fn totals_table(&self) -> List {
        fit_table(each_bayes_one_year!(&self.inner, f => f.totals()))
    }

    fn development_table(&self) -> List {
        fit_table(each_bayes_one_year!(&self.inner, f => f.fit.development_table()))
    }

    /// The fit itself, as for any single-segment fit without keys: a
    /// condition on a key is an error.
    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let one = SegmentFits {
            key_names: Vec::new(),
            labels: vec![Label::new(Vec::<String>::new())],
            fits: vec![()],
        };
        pick(&one, keys, values)?;
        Ok(Self {
            inner: self.inner.clone(),
        })
    }
}

/// The one-year view of a Bayesian model on every segment of a triangle
/// column: each segment's posterior, opening reserve and Pareto k, and the
/// joint claims development result.
#[extendr]
pub(crate) struct BayesOneYearFits {
    inner: BayesOneYearFitsOf,
}

/// The one-year view over segments of each Bayesian model with one, by
/// its segment fit.
enum BayesOneYearFitsOf {
    Odp(BayesOneYearFitsInner<BayesOdpSegment>),
    Meyers(BayesOneYearFitsInner<MeyersSegment>),
}

/// Evaluates `$body` with `$f` bound to the views over segments, whatever
/// the model.
macro_rules! each_bayes_one_years {
    ($views:expr, $f:ident => $body:expr) => {
        match $views {
            BayesOneYearFitsOf::Odp($f) => $body,
            BayesOneYearFitsOf::Meyers($f) => $body,
        }
    };
}

#[extendr]
impl BayesOneYearFits {
    /// The model behind the view: "bayes_odp" or "meyers".
    fn model(&self) -> &'static str {
        match &self.inner {
            BayesOneYearFitsOf::Odp(_) => "bayes_odp",
            BayesOneYearFitsOf::Meyers(_) => "meyers",
        }
    }

    /// The volume-weighted chain ladder of every segment.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: each_bayes_one_years!(&self.inner, f => f
                .segments
                .map(|s| s.fit.chain_ladder.clone())),
        }
    }

    fn opening_ultimate(&self) -> Vec<f64> {
        each_bayes_one_years!(&self.inner, f => by_origin(&f.segments, |s| s
            .opening_ultimate
            .clone()))
    }

    fn opening_reserve(&self) -> Vec<f64> {
        each_bayes_one_years!(&self.inner, f => by_origin(&f.segments, |s| s
            .opening_reserve
            .clone()))
    }

    fn cdr(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: each_bayes_one_years!(&self.inner, f => f.cdr.clone()),
        }
    }

    /// The plugged-in scale of a single-segment `bayes_odp` view.
    fn scale(&self) -> Result<f64> {
        match &self.inner {
            BayesOneYearFitsOf::Odp(f) => {
                Ok(single(&f.segments, "scale", "totals_frame()")?.fit.scale)
            }
            BayesOneYearFitsOf::Meyers(_) => Err(Error::Other(
                "scale is the ODP's plugged-in dispersion; Meyers' models have none".into(),
            )),
        }
    }

    /// The premium of each origin of a single-segment `meyers` view.
    fn premium(&self) -> Result<Vec<f64>> {
        match &self.inner {
            BayesOneYearFitsOf::Meyers(f) => Ok(single(&f.segments, "premium", "totals_frame()")?
                .fit
                .premium
                .clone()),
            BayesOneYearFitsOf::Odp(_) => Err(Error::Other(
                "premium is Meyers' input; bayes_odp has none".into(),
            )),
        }
    }

    fn pareto_k(&self) -> Result<Vec<f64>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "pareto_k", "")?
            .pareto_k
            .clone()))
    }

    /// The simulations of a single-segment fit refitted with NUTS, 1-based.
    fn refitted(&self) -> Result<Vec<f64>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "refitted", "")?
            .refitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()))
    }

    /// The simulations of a single-segment fit that moment matching
    /// rescued, 1-based.
    fn moment_matched(&self) -> Result<Vec<f64>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "moment_matched", "")?
            .moment_matched
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()))
    }

    fn unrefitted(&self) -> Result<Vec<f64>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "unrefitted", "")?
            .unrefitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()))
    }

    fn refit_warnings(&self) -> Result<f64> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "refit_warnings", "")?
            .refit_warnings as f64))
    }

    /// Each segment's number of refits and of refits with a diagnostic
    /// warning, for the warning of the one-year views over segments.
    fn refit_counts(&self) -> List {
        each_bayes_one_years!(&self.inner, f => {
            let fits = &f.segments.fits;
            list!(
                refits = fits
                    .iter()
                    .map(|s| s.refitted.len() as f64)
                    .collect::<Vec<_>>(),
                warned = fits
                    .iter()
                    .map(|s| s.refit_warnings as f64)
                    .collect::<Vec<_>>(),
                unrefitted = fits
                    .iter()
                    .map(|s| s.unrefitted.len() as f64)
                    .collect::<Vec<_>>()
            )
        })
    }

    fn k_threshold(&self) -> f64 {
        each_bayes_one_years!(&self.inner, f => f.segments.fits[0].k_threshold)
    }

    fn flagged_share(&self) -> Result<f64> {
        each_bayes_one_years!(&self.inner, f => Ok(single(
            &f.segments,
            "flagged_share",
            "totals_frame()"
        )?
        .flagged_share()))
    }

    /// One row per segment: `list(table, warnings)`, as
    /// `segment_diagnostics`.
    fn diagnostics_table(&self) -> List {
        match &self.inner {
            BayesOneYearFitsOf::Odp(f) => {
                segment_diagnostics(&f.segments, f.totals(), |s| &s.fit.diagnostics)
            }
            BayesOneYearFitsOf::Meyers(f) => {
                segment_diagnostics(&f.segments, f.totals(), |s| &s.fit.diagnostics)
            }
        }
    }

    fn long_table(&self) -> List {
        fit_table(each_bayes_one_years!(&self.inner, f => f.to_long()))
    }

    fn totals_table(&self) -> List {
        fit_table(each_bayes_one_years!(&self.inner, f => f.totals()))
    }

    fn development_table(&self) -> List {
        fit_table(each_bayes_one_years!(&self.inner, f => f.segments.development_table()))
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let keys = choice(&keys, &values)?;
        let inner = match &self.inner {
            BayesOneYearFitsOf::Odp(f) => BayesOneYearFitsOf::Odp(f.segment(&keys).map_err(to_r)?),
            BayesOneYearFitsOf::Meyers(f) => {
                BayesOneYearFitsOf::Meyers(f.segment(&keys).map_err(to_r)?)
            }
        };
        Ok(Self { inner })
    }
}

/// The `Meyers` of `meyers` and `meyers_segments`: `priors` are
/// `logelr_mean`, `logelr_sd`, `origin_sd`, `development_sd`, `gamma_sd` and
/// `rho_shape`; `run` is `chains`, `tune`, `draws`, `n_sims`, `seed`,
/// `target_accept` and `max_depth`.
fn meyers_settings(
    model: &str,
    priors: [f64; 6],
    run: [f64; 7],
    dependence: SegmentDependence,
) -> Result<Meyers> {
    let model = MeyersModel::from_name(model)
        .ok_or_else(|| crate::unknown_name("model", model, MeyersModel::NAMES))?;
    let [
        logelr_mean,
        logelr_sd,
        origin_sd,
        development_sd,
        gamma_sd,
        rho_shape,
    ] = priors;
    let [chains, tune, draws, n_sims, seed, target_accept, max_depth] = run;
    let n_sims = whole(n_sims, "n_sims")? as usize;
    if n_sims == 0 {
        return Err(Error::Other("n_sims must be positive".into()));
    }
    Ok(Meyers {
        model,
        logelr_mean,
        logelr_sd,
        origin_sd,
        development_sd,
        gamma_sd,
        rho_shape,
        sampler: BayesSampler {
            chains: whole(chains, "chains")? as usize,
            tune: whole(tune, "tune")? as usize,
            draws: whole(draws, "draws")? as usize,
            seed: 0,
            target_accept,
            max_depth: whole(max_depth, "max_depth")?,
        },
        n_sims,
        seed: whole(seed, "seed")?,
        dependence,
    })
}

/// One row per segment: `list(table, warnings)`, `table` a fit table with
/// the key columns of `totals` and `max_rhat`, `min_ess_bulk`,
/// `min_ess_tail` and `divergences`, `warnings` each segment's messages
/// joined by "; ".
fn segment_diagnostics<T>(
    segments: &SegmentFits<T>,
    totals: FitTable,
    diag: fn(&T) -> &Diagnostics,
) -> List {
    let fits = &segments.fits;
    let col = |g: fn(&Diagnostics) -> f64| fits.iter().map(|f| g(diag(f))).collect::<Vec<f64>>();
    let table = FitTable {
        keys: totals.keys,
        origin: None,
        age: None,
        values: vec![
            ("max_rhat".into(), col(|d| d.max_rhat)),
            ("min_ess_bulk".into(), col(|d| d.min_ess_bulk)),
            ("min_ess_tail".into(), col(|d| d.min_ess_tail)),
            ("divergences".into(), col(|d| d.divergences as f64)),
        ],
    };
    let warnings: Vec<String> = fits.iter().map(|f| diag(f).warnings.join("; ")).collect();
    list!(table = fit_table(table), warnings = warnings)
}

/// A posterior summary as a list of columns, as `bayes_glm_model@summary`.
fn summary_list(s: &[ParamSummary]) -> List {
    let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
    list!(
        parameter = s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
        mean = col(|p| p.mean),
        sd = col(|p| p.sd),
        q05 = col(|p| p.q05),
        q50 = col(|p| p.q50),
        q95 = col(|p| p.q95),
        rhat = col(|p| p.rhat),
        ess_bulk = col(|p| p.ess_bulk),
        ess_tail = col(|p| p.ess_tail)
    )
}

/// The Bayesian ODP chain ladder of every segment of a triangle column:
/// each segment's posterior and the joint reserves. The posterior of one
/// segment needs a single-segment fit.
#[extendr]
pub(crate) struct BayesOdpFits {
    pub(crate) inner: BayesOdpFitsInner,
}

impl BayesOdpFits {
    fn one(&self, field: &str, instead: &str) -> Result<&BayesOdpSegment> {
        single(&self.inner.segments, field, instead)
    }
}

#[extendr]
impl BayesOdpFits {
    /// The volume-weighted chain ladder of every segment.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.segments.map(|s| s.chain_ladder.clone()),
        }
    }

    fn scale(&self) -> Result<f64> {
        Ok(self.one("scale", "totals_frame()")?.scale)
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// The joint ultimates: the reserves with each origin's latest value
    /// added in every simulation (`UltimateDraws::ultimates`).
    fn ultimates(&self) -> Result<PredictiveDistribution> {
        ultimates_of(&self.inner)
    }

    /// One origin's ultimate in each simulation, or the total's with
    /// `origin` NULL, in simulation order.
    fn ultimate_draws(&self, origin: Nullable<List>) -> Result<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// [`ultimate_draws`](Self::ultimate_draws) over `premium`.
    fn loss_ratio_draws(&self, origin: Nullable<List>, premium: f64) -> Result<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// The development ages of a segment's pattern draws, in months
    /// (`DevelopmentPatterns::segment`); `segment` is empty for a fit of
    /// one segment.
    fn pattern_ages(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_ages_of(&self.inner.patterns, &segment)
    }

    /// The pattern draws of a segment, row-major: simulation `i`'s
    /// cumulative share at each age in row `i`.
    fn pattern_draws(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_draws_of(&self.inner.patterns, &segment)
    }

    /// The monthly patterns for the cash-flow engine, row-major: simulation
    /// `i`'s share at the end of each of `months` months in row `i`.
    fn monthly_pattern_draws(
        &self,
        segment: Vec<String>,
        months: f64,
        basis: Robj,
        monotone: &str,
    ) -> Result<Vec<f64>> {
        monthly_pattern_draws_of(&self.inner.patterns, &segment, months, &basis, monotone)
    }

    /// One row per segment: `list(table, warnings)`, as
    /// `segment_diagnostics`.
    fn diagnostics_table(&self) -> List {
        segment_diagnostics(&self.inner.segments, self.inner.totals(), |s| {
            &s.diagnostics
        })
    }

    fn parameter_names(&self) -> Result<Vec<String>> {
        Ok(self.one("draws", "")?.posterior.names().to_vec())
    }

    /// Posterior draws of a single-segment fit, row-major `n_draws x
    /// n_parameters`, chain by chain.
    fn draws(&self) -> Result<Vec<f64>> {
        Ok(self.one("draws", "")?.posterior.values().to_vec())
    }

    fn chains(&self) -> Result<f64> {
        Ok(self.one("draws", "")?.posterior.chains() as f64)
    }

    fn summary(&self) -> Result<List> {
        let s = self.one("summary", "")?.summary().map_err(to_r)?;
        Ok(summary_list(&s))
    }

    /// PSIS-LOO of a single-segment fit's observed increments, as
    /// `BayesOdpFit$loo()`.
    fn loo(&self) -> Result<List> {
        let l = self.one("bayes_loo()", "")?.loo().map_err(to_r)?;
        let e = &l.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.pareto_k,
            k_threshold = l.k_threshold
        ))
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn development_table(&self) -> List {
        fit_table(self.inner.development_table())
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: self.inner.segment(&choice(&keys, &values)?).map_err(to_r)?,
        })
    }
}

/// Meyers' CRC, CSR or CAY model of a single-segment triangle column.
#[extendr]
pub(crate) struct MeyersFit {
    pub(crate) inner: MeyersInner,
}

#[extendr]
impl MeyersFit {
    /// The posterior predictive of a new underwriting year with `premium`,
    /// the level `"exchangeable"` or `"prior"` (decision 28).
    fn new_year(&self, premium: f64, n_sims: f64, seed: f64, level: &str) -> Result<BayesNewYear> {
        let level = match level {
            "exchangeable" => NewOriginLevel::Exchangeable,
            "prior" => NewOriginLevel::Prior,
            other => {
                return Err(Error::Other(format!(
                    "level must be \"exchangeable\" or \"prior\", not {other:?}"
                )));
            }
        };
        let n_sims = whole(n_sims, "n_sims")? as usize;
        let inner = self
            .inner
            .new_year_with_level(premium, level, n_sims, whole(seed, "seed")?)
            .map_err(to_r)?;
        Ok(BayesNewYear { inner })
    }

    /// "crc", "csr" or "cay".
    fn model(&self) -> &'static str {
        self.inner.model().name()
    }

    /// The volume-weighted chain ladder the reserves are measured from.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.inner.chain_ladder.clone()],
            },
        }
    }

    fn premium(&self) -> Vec<f64> {
        self.inner.premium.clone()
    }

    fn ultimates(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.ultimates.clone(),
        }
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// One origin's ultimate in each simulation, or the total's with
    /// `origin` NULL, in simulation order.
    fn ultimate_draws(&self, origin: Nullable<List>) -> Result<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// [`ultimate_draws`](Self::ultimate_draws) over `premium`.
    fn loss_ratio_draws(&self, origin: Nullable<List>, premium: f64) -> Result<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// The development ages of a segment's pattern draws, in months
    /// (`DevelopmentPatterns::segment`); `segment` is empty for a fit of
    /// one segment.
    fn pattern_ages(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_ages_of(&self.inner.patterns, &segment)
    }

    /// The pattern draws of a segment, row-major: simulation `i`'s
    /// cumulative share at each age in row `i`.
    fn pattern_draws(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_draws_of(&self.inner.patterns, &segment)
    }

    /// The monthly patterns for the cash-flow engine, row-major: simulation
    /// `i`'s share at the end of each of `months` months in row `i`.
    fn monthly_pattern_draws(
        &self,
        segment: Vec<String>,
        months: f64,
        basis: Robj,
        monotone: &str,
    ) -> Result<Vec<f64>> {
        monthly_pattern_draws_of(&self.inner.patterns, &segment, months, &basis, monotone)
    }

    /// `list(max_rhat, min_ess_bulk, min_ess_tail, divergences, warnings)`.
    fn diagnostics(&self) -> List {
        let d = &self.inner.diagnostics;
        list!(
            max_rhat = d.max_rhat,
            min_ess_bulk = d.min_ess_bulk,
            min_ess_tail = d.min_ess_tail,
            divergences = d.divergences as f64,
            warnings = d.warnings.clone()
        )
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, row-major `n_draws x n_parameters`, chain by chain.
    fn draws(&self) -> Vec<f64> {
        self.inner.posterior.values().to_vec()
    }

    fn chains(&self) -> f64 {
        self.inner.posterior.chains() as f64
    }

    /// The posterior summary, columns as `bayes_glm_model@summary`.
    fn summary(&self) -> Result<List> {
        let s = self.inner.summary().map_err(to_r)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        Ok(list!(
            parameter = s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            mean = col(|p| p.mean),
            sd = col(|p| p.sd),
            q05 = col(|p| p.q05),
            q50 = col(|p| p.q50),
            q95 = col(|p| p.q95),
            rhat = col(|p| p.rhat),
            ess_bulk = col(|p| p.ess_bulk),
            ess_tail = col(|p| p.ess_tail)
        ))
    }

    /// PSIS-LOO of the observed cells on the log scale: `list(estimates,
    /// pointwise, pareto_k, k_threshold)`, as `BayesGlmFit$loo()`.
    fn loo(&self) -> Result<List> {
        let l = self.inner.loo().map_err(to_r)?;
        let e = &l.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.pareto_k,
            k_threshold = l.k_threshold
        ))
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn development_table(&self) -> List {
        fit_table(self.inner.development_table())
    }

    /// The fit itself, as for any single-segment fit without keys: a
    /// condition on a key is an error.
    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let one = SegmentFits {
            key_names: Vec::new(),
            labels: vec![Label::new(Vec::<String>::new())],
            fits: vec![()],
        };
        pick(&one, keys, values)?;
        Ok(Self {
            inner: self.inner.clone(),
        })
    }
}

/// A pattern prior from R's `pattern_prior()` arguments: `ages` (`NULL`
/// for none), then the means and sds of the log share ratios after the
/// first age.
fn pattern_arg(
    ages: Nullable<Vec<f64>>,
    mean: Vec<f64>,
    sd: Vec<f64>,
) -> Result<Option<PatternPrior>> {
    match ages {
        Nullable::Null => Ok(None),
        Nullable::NotNull(ages) => {
            let ages = lags(&ages)?;
            Ok(Some(PatternPrior::new(&ages, &mean, &sd).map_err(to_r)?))
        }
    }
}

/// Whole months from R numbers.
fn lags(ages: &[f64]) -> Result<Vec<Lag>> {
    ages.iter()
        .map(|&a| {
            let a = whole(a, "ages")?;
            Lag::try_from(a).map_err(|_| Error::Other(format!("age {a} is too large")))
        })
        .collect()
}

/// The prior's means and sds from `ages` and either `mean` or, with
/// `from_shares`, incremental shares, checked as `PatternPrior` checks
/// them: a list of `ages`, `mean` and `sd`.
#[extendr]
fn pattern_prior_rust(
    ages: Vec<f64>,
    values: Vec<f64>,
    sd: Vec<f64>,
    from_shares: bool,
) -> Result<List> {
    let lags = lags(&ages)?;
    let prior = if from_shares {
        PatternPrior::from_shares(&lags, &values, &sd)
    } else {
        PatternPrior::new(&lags, &values, &sd)
    }
    .map_err(to_r)?;
    Ok(list!(
        ages = prior
            .ages
            .iter()
            .map(|&a| f64::from(a))
            .collect::<Vec<f64>>(),
        mean = prior.mean.clone(),
        sd = prior.sd.clone(),
        shares = prior.shares(),
        percent_developed = prior.percent_developed()
    ))
}

/// `settings` is `min_developed`, `min_companies` and `subtract_process`
/// (0 or 1).
fn industry_settings(settings: &[f64]) -> Result<IndustryPriorSettings> {
    let &[min_developed, min_companies, subtract] = settings else {
        return Err(Error::Other(
            "industry prior settings are min_developed, min_companies and subtract_process".into(),
        ));
    };
    Ok(IndustryPriorSettings {
        min_developed,
        min_companies: whole(min_companies, "min_companies")? as usize,
        subtract_process: subtract != 0.0,
    })
}

/// The industry prior of `triangles` (a list of triangle pointers, one per
/// company) of loss `column`, with `premiums` (a list of numeric vectors,
/// one per company).
#[extendr]
fn industry_prior_rust(
    triangles: List,
    column: &str,
    premiums: List,
    settings: Vec<f64>,
) -> Result<IndustryPriorFit> {
    let mut tris = Vec::with_capacity(triangles.len());
    for robj in triangles.values() {
        let t: &Triangle = (&robj)
            .try_into()
            .map_err(|_| Error::Other("each of triangles must be a triangle".into()))?;
        tris.push(t.inner.clone());
    }
    let mut prem = Vec::with_capacity(premiums.len());
    for robj in premiums.values() {
        let p: Vec<f64> = robj
            .as_real_vector()
            .ok_or_else(|| Error::Other("each of premiums must be a numeric vector".into()))?;
        prem.push(p);
    }
    let inner = industry_settings(&settings)?
        .from_triangles(&tris, column, &prem)
        .map_err(to_r)?;
    Ok(IndustryPriorFit { inner })
}

/// The industry prior of a triangle with one segment per company, premium
/// from its `exposure` column.
#[extendr]
fn industry_prior_segments_rust(
    triangle: &Triangle,
    column: &str,
    exposure: &str,
    settings: Vec<f64>,
) -> Result<IndustryPriorFit> {
    let inner = industry_settings(&settings)?
        .from_segments(&triangle.inner, column, exposure)
        .map_err(to_r)?;
    Ok(IndustryPriorFit { inner })
}

/// An industry prior for the Bayesian BF (decision 29).
#[extendr]
pub(crate) struct IndustryPriorFit {
    inner: IndustryPrior,
}

#[extendr]
impl IndustryPriorFit {
    /// `apriori`, `apriori_se`, `between_cv`, `within_cv`, `process_cv`,
    /// `cv` and `tail_factor`.
    fn numbers(&self) -> List {
        let p = &self.inner;
        list!(
            apriori = p.apriori,
            apriori_se = p.apriori_se,
            between_cv = p.between_cv,
            within_cv = p.within_cv,
            process_cv = p.process_cv,
            cv = p.cv,
            tail_factor = p.tail_factor
        )
    }

    /// The pattern prior: `ages`, `mean`, `sd`, `shares` and
    /// `percent_developed`.
    fn pattern(&self) -> List {
        let prior = &self.inner.pattern;
        list!(
            ages = prior
                .ages
                .iter()
                .map(|&a| f64::from(a))
                .collect::<Vec<f64>>(),
            mean = prior.mean.clone(),
            sd = prior.sd.clone(),
            shares = prior.shares(),
            percent_developed = prior.percent_developed()
        )
    }

    fn percent_developed(&self) -> Vec<f64> {
        self.inner.percent_developed.clone()
    }

    fn percent_developed_sd(&self) -> Vec<f64> {
        self.inner.percent_developed_sd.clone()
    }

    fn company_loss_ratios(&self) -> Vec<f64> {
        self.inner.company_loss_ratios.clone()
    }

    fn report(&self) -> List {
        let r = &self.inner.report;
        let ints = |v: &[usize]| v.iter().map(|&n| n as i32).collect::<Vec<i32>>();
        list!(
            n_companies = r.n_companies as i32,
            n_companies_used = r.n_companies_used as i32,
            n_origins_used = r.n_origins_used as i32,
            origins_used = ints(&r.origins_used),
            companies_by_age = ints(&r.companies_by_age),
            excluded = r.excluded.clone(),
            warnings = r.warnings.clone()
        )
    }

    /// The prior for a triangle whose last age is `age`.
    fn at_age(&self, age: f64) -> Result<Self> {
        let age = Lag::try_from(whole(age, "age")?)
            .map_err(|_| Error::Other(format!("age {age} is too large")))?;
        Ok(Self {
            inner: self.inner.at_age(age).map_err(to_r)?,
        })
    }
}

/// The posterior predictive of a new underwriting year (decision 28).
#[extendr]
pub(crate) struct BayesNewYear {
    inner: NewYear,
}

#[extendr]
impl BayesNewYear {
    /// The new origin period's label.
    fn origin(&self) -> String {
        self.inner.origin.to_string()
    }

    fn premium(&self) -> f64 {
        self.inner.premium
    }

    fn loss_ratio(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.loss_ratio.clone(),
        }
    }

    fn ultimate(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.ultimate.clone(),
        }
    }

    fn pattern_ages(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_ages_of(&self.inner.patterns, &segment)
    }

    fn pattern_draws(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_draws_of(&self.inner.patterns, &segment)
    }

    fn monthly_pattern_draws(
        &self,
        segment: Vec<String>,
        months: f64,
        basis: Robj,
        monotone: &str,
    ) -> Result<Vec<f64>> {
        monthly_pattern_draws_of(&self.inner.patterns, &segment, months, &basis, monotone)
    }
}

/// A Bayesian Bornhuetter-Ferguson with a normalised pattern of a
/// single-segment triangle column.
#[extendr]
pub(crate) struct BayesBfNormalisedFit {
    pub(crate) inner: BayesBfNormalisedInner,
}

#[extendr]
impl BayesBfNormalisedFit {
    /// The posterior predictive of a new underwriting year with `premium`
    /// (decision 28).
    fn new_year(&self, premium: f64, n_sims: f64, seed: f64) -> Result<BayesNewYear> {
        let n_sims = whole(n_sims, "n_sims")? as usize;
        let inner = self
            .inner
            .new_year(premium, n_sims, whole(seed, "seed")?)
            .map_err(to_r)?;
        Ok(BayesNewYear { inner })
    }

    /// The volume-weighted chain ladder the model takes its latest values
    /// and scale from.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.inner.chain_ladder.clone()],
            },
        }
    }

    fn scale(&self) -> f64 {
        self.inner.scale
    }

    fn exposure(&self) -> Vec<f64> {
        self.inner.exposure.clone()
    }

    fn prior_ultimate(&self) -> Vec<f64> {
        self.inner.prior_ultimate.clone()
    }

    /// The posterior mean of each age's share of the ultimate.
    fn pattern(&self) -> Vec<f64> {
        self.inner.pattern()
    }

    /// The posterior mean of each origin's expected ultimate.
    fn expected_ultimate(&self) -> Vec<f64> {
        self.inner.expected_ultimate()
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// The joint ultimates: the reserves with each origin's latest value
    /// added in every simulation (`UltimateDraws::ultimates`).
    fn ultimates(&self) -> Result<PredictiveDistribution> {
        ultimates_of(&self.inner)
    }

    /// One origin's ultimate in each simulation, or the total's with
    /// `origin` NULL, in simulation order.
    fn ultimate_draws(&self, origin: Nullable<List>) -> Result<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// [`ultimate_draws`](Self::ultimate_draws) over `premium`.
    fn loss_ratio_draws(&self, origin: Nullable<List>, premium: f64) -> Result<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// The development ages of a segment's pattern draws, in months
    /// (`DevelopmentPatterns::segment`); `segment` is empty for a fit of
    /// one segment.
    fn pattern_ages(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_ages_of(&self.inner.patterns, &segment)
    }

    /// The pattern draws of a segment, row-major: simulation `i`'s
    /// cumulative share at each age in row `i`.
    fn pattern_draws(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_draws_of(&self.inner.patterns, &segment)
    }

    /// The monthly patterns for the cash-flow engine, row-major: simulation
    /// `i`'s share at the end of each of `months` months in row `i`.
    fn monthly_pattern_draws(
        &self,
        segment: Vec<String>,
        months: f64,
        basis: Robj,
        monotone: &str,
    ) -> Result<Vec<f64>> {
        monthly_pattern_draws_of(&self.inner.patterns, &segment, months, &basis, monotone)
    }

    /// `list(max_rhat, min_ess_bulk, min_ess_tail, divergences, warnings)`.
    fn diagnostics(&self) -> List {
        let d = &self.inner.diagnostics;
        list!(
            max_rhat = d.max_rhat,
            min_ess_bulk = d.min_ess_bulk,
            min_ess_tail = d.min_ess_tail,
            divergences = d.divergences as f64,
            warnings = d.warnings.clone()
        )
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, row-major `n_draws x n_parameters`, chain by chain.
    fn draws(&self) -> Vec<f64> {
        self.inner.posterior.values().to_vec()
    }

    fn chains(&self) -> f64 {
        self.inner.posterior.chains() as f64
    }

    /// The posterior summary, columns as `bayes_glm_model@summary`.
    fn summary(&self) -> Result<List> {
        let s = self.inner.summary().map_err(to_r)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        Ok(list!(
            parameter = s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            mean = col(|p| p.mean),
            sd = col(|p| p.sd),
            q05 = col(|p| p.q05),
            q50 = col(|p| p.q50),
            q95 = col(|p| p.q95),
            rhat = col(|p| p.rhat),
            ess_bulk = col(|p| p.ess_bulk),
            ess_tail = col(|p| p.ess_tail)
        ))
    }

    /// PSIS-LOO of the observed increments: `list(estimates, pointwise,
    /// pareto_k, k_threshold)`, as `BayesGlmFit$loo()`.
    fn loo(&self) -> Result<List> {
        let l = self.inner.loo().map_err(to_r)?;
        let e = &l.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.pareto_k,
            k_threshold = l.k_threshold
        ))
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn development_table(&self) -> List {
        fit_table(self.inner.development_table())
    }

    /// The fit itself, as for any single-segment fit without keys: a
    /// condition on a key is an error.
    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let one = SegmentFits {
            key_names: Vec::new(),
            labels: vec![Label::new(Vec::<String>::new())],
            fits: vec![()],
        };
        pick(&one, keys, values)?;
        Ok(Self {
            inner: self.inner.clone(),
        })
    }
}

/// A hierarchical Bayesian Bornhuetter-Ferguson over the programs of a
/// triangle column.
#[extendr]
pub(crate) struct HierarchicalBfFit {
    pub(crate) inner: HierarchicalBfInner,
}

#[extendr]
impl HierarchicalBfFit {
    /// One row per program, the core's table (#838).
    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    /// Each program's segment label, joined by " / ", in the triangle's
    /// order.
    fn program_labels(&self) -> Vec<String> {
        self.inner
            .programs
            .iter()
            .map(|p| p.label.to_string())
            .collect()
    }

    /// One program as a list: `label`, `key` (its key values), `origins`,
    /// `exposure`, `latest`, `prior_ultimate`, `scale`, `expected_ultimate`,
    /// `pattern`, `loss_ratio`, `own_loss_ratio` and `shrinkage`. `index` is
    /// 1-based.
    fn program(&self, index: f64) -> Result<List> {
        let p = self.program_at(index)?;
        Ok(list!(
            label = p.label.to_string(),
            key = p.label.parts().to_vec(),
            origins = p
                .origins
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            exposure = p.exposure.clone(),
            latest = p.latest.clone(),
            prior_ultimate = p.prior_ultimate.clone(),
            scale = p.scale,
            expected_ultimate = p.expected_ultimate.clone(),
            pattern = p.pattern.clone(),
            loss_ratio = p.loss_ratio,
            own_loss_ratio = p.own_loss_ratio,
            shrinkage = p.shrinkage
        ))
    }

    /// The program's part of the joint reserves (`index` is 1-based).
    fn program_reserves(&self, index: f64) -> Result<PredictiveDistribution> {
        Ok(PredictiveDistribution {
            inner: self.program_at(index)?.reserves.clone(),
        })
    }

    /// The development ages of every program's pattern, in months.
    fn ages(&self) -> Vec<f64> {
        self.inner.ages.iter().map(|&a| f64::from(a)).collect()
    }

    /// The pooled scale per unit of premium.
    fn scale_per_premium(&self) -> f64 {
        self.inner.scale_per_premium
    }

    /// The joint reserves by program and origin.
    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// `list(max_rhat, min_ess_bulk, min_ess_tail, divergences, warnings)`.
    fn diagnostics(&self) -> List {
        let d = &self.inner.diagnostics;
        list!(
            max_rhat = d.max_rhat,
            min_ess_bulk = d.min_ess_bulk,
            min_ess_tail = d.min_ess_tail,
            divergences = d.divergences as f64,
            warnings = d.warnings.clone()
        )
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, row-major `n_draws x n_parameters`, chain by chain.
    fn draws(&self) -> Vec<f64> {
        self.inner.posterior.values().to_vec()
    }

    fn chains(&self) -> f64 {
        self.inner.posterior.chains() as f64
    }

    /// The posterior summary of every sampled parameter.
    fn summary(&self) -> Result<List> {
        Ok(summary_list(&self.inner.summary().map_err(to_r)?))
    }

    /// The posterior summary of the hyperparameters `level`, `tau` and
    /// `omega` on their natural scale.
    fn hyperparameters(&self) -> Result<List> {
        Ok(summary_list(&self.inner.hyperparameters().map_err(to_r)?))
    }
}

impl HierarchicalBfFit {
    fn program_at(&self, index: f64) -> Result<&HierarchicalBfProgram> {
        let n = self.inner.programs.len();
        let i = whole(index, "index")? as usize;
        if i == 0 || i > n {
            return Err(Error::Other(format!(
                "program index {i} is outside 1 to {n}"
            )));
        }
        Ok(&self.inner.programs[i - 1])
    }
}

/// The one-year view of a hierarchical Bayesian Bornhuetter-Ferguson over
/// every program: the lifetime fit, the opening reserve, the claims
/// development result by program and origin and each simulation's Pareto k.
#[extendr]
pub(crate) struct HierarchicalBfOneYearFit {
    inner: BayesOneYearInner<HierarchicalBfInner>,
}

#[extendr]
impl HierarchicalBfOneYearFit {
    /// One row per program and origin, the core's table (#838).
    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    /// One row per program, the core's table (#838).
    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    /// The lifetime fit, as `bayes_hierarchical_bf` gives it.
    fn fit(&self) -> HierarchicalBfFit {
        HierarchicalBfFit {
            inner: self.inner.fit.clone(),
        }
    }

    /// The latest value plus the opening reserve, per program and origin.
    fn opening_ultimate(&self) -> Vec<f64> {
        self.inner.opening_ultimate.clone()
    }

    /// Each origin's posterior mean reserve on the observed triangle, per
    /// program and origin.
    fn opening_reserve(&self) -> Vec<f64> {
        self.inner.opening_reserve.clone()
    }

    /// The joint claims development result by program and origin.
    fn cdr(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.cdr.clone(),
        }
    }

    fn pareto_k(&self) -> Vec<f64> {
        self.inner.pareto_k.clone()
    }

    /// The simulations refitted with NUTS, 1-based.
    fn refitted(&self) -> Vec<f64> {
        self.inner
            .refitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()
    }

    /// The simulations moment matching rescued, 1-based.
    fn moment_matched(&self) -> Vec<f64> {
        self.inner
            .moment_matched
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()
    }

    /// The simulations due for a refit beyond `max_refits`, reweighted
    /// instead, 1-based.
    fn unrefitted(&self) -> Vec<f64> {
        self.inner
            .unrefitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()
    }

    fn refit_warnings(&self) -> f64 {
        self.inner.refit_warnings as f64
    }

    fn k_threshold(&self) -> f64 {
        self.inner.k_threshold
    }

    fn flagged_share(&self) -> f64 {
        self.inner.flagged_share()
    }

    /// What else the view warns of, one message each: the programs whose
    /// next diagonal falls at development ages they have never observed.
    fn warnings(&self) -> Vec<String> {
        self.inner.warnings.clone()
    }
}

/// A Bayesian Bornhuetter-Ferguson of a single-segment triangle column.
#[extendr]
pub(crate) struct BayesBfFit {
    pub(crate) inner: BayesBfInner,
}

#[extendr]
impl BayesBfFit {
    /// The volume-weighted chain ladder the model takes its latest values
    /// and scale from.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.inner.chain_ladder.clone()],
            },
        }
    }

    fn scale(&self) -> f64 {
        self.inner.scale
    }

    fn exposure(&self) -> Vec<f64> {
        self.inner.exposure.clone()
    }

    fn prior_ultimate(&self) -> Vec<f64> {
        self.inner.prior_ultimate.clone()
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// The joint ultimates: the reserves with each origin's latest value
    /// added in every simulation (`UltimateDraws::ultimates`).
    fn ultimates(&self) -> Result<PredictiveDistribution> {
        ultimates_of(&self.inner)
    }

    /// One origin's ultimate in each simulation, or the total's with
    /// `origin` NULL, in simulation order.
    fn ultimate_draws(&self, origin: Nullable<List>) -> Result<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// [`ultimate_draws`](Self::ultimate_draws) over `premium`.
    fn loss_ratio_draws(&self, origin: Nullable<List>, premium: f64) -> Result<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// The development ages of a segment's pattern draws, in months
    /// (`DevelopmentPatterns::segment`); `segment` is empty for a fit of
    /// one segment.
    fn pattern_ages(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_ages_of(&self.inner.patterns, &segment)
    }

    /// The pattern draws of a segment, row-major: simulation `i`'s
    /// cumulative share at each age in row `i`.
    fn pattern_draws(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_draws_of(&self.inner.patterns, &segment)
    }

    /// The monthly patterns for the cash-flow engine, row-major: simulation
    /// `i`'s share at the end of each of `months` months in row `i`.
    fn monthly_pattern_draws(
        &self,
        segment: Vec<String>,
        months: f64,
        basis: Robj,
        monotone: &str,
    ) -> Result<Vec<f64>> {
        monthly_pattern_draws_of(&self.inner.patterns, &segment, months, &basis, monotone)
    }

    /// `list(max_rhat, min_ess_bulk, min_ess_tail, divergences, warnings)`.
    fn diagnostics(&self) -> List {
        let d = &self.inner.diagnostics;
        list!(
            max_rhat = d.max_rhat,
            min_ess_bulk = d.min_ess_bulk,
            min_ess_tail = d.min_ess_tail,
            divergences = d.divergences as f64,
            warnings = d.warnings.clone()
        )
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, row-major `n_draws x n_parameters`, chain by chain.
    fn draws(&self) -> Vec<f64> {
        self.inner.posterior.values().to_vec()
    }

    fn chains(&self) -> f64 {
        self.inner.posterior.chains() as f64
    }

    /// The posterior summary, columns as `bayes_glm_model@summary`.
    fn summary(&self) -> Result<List> {
        let s = self.inner.summary().map_err(to_r)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        Ok(list!(
            parameter = s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            mean = col(|p| p.mean),
            sd = col(|p| p.sd),
            q05 = col(|p| p.q05),
            q50 = col(|p| p.q50),
            q95 = col(|p| p.q95),
            rhat = col(|p| p.rhat),
            ess_bulk = col(|p| p.ess_bulk),
            ess_tail = col(|p| p.ess_tail)
        ))
    }

    /// PSIS-LOO of the observed increments: `list(estimates, pointwise,
    /// pareto_k, k_threshold)`, as `BayesGlmFit$loo()`.
    fn loo(&self) -> Result<List> {
        let l = self.inner.loo().map_err(to_r)?;
        let e = &l.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.pareto_k,
            k_threshold = l.k_threshold
        ))
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn development_table(&self) -> List {
        fit_table(self.inner.development_table())
    }

    /// The fit itself, as for any single-segment fit without keys: a
    /// condition on a key is an error.
    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let one = SegmentFits {
            key_names: Vec::new(),
            labels: vec![Label::new(Vec::<String>::new())],
            fits: vec![()],
        };
        pick(&one, keys, values)?;
        Ok(Self {
            inner: self.inner.clone(),
        })
    }
}

/// Meyers' model of every segment of a triangle column: each segment's
/// posterior and the joint ultimates and reserves. The posterior of one
/// segment needs a single-segment fit.
#[extendr]
pub(crate) struct MeyersFits {
    pub(crate) inner: MeyersFitsInner,
}

impl MeyersFits {
    fn one(&self, field: &str, instead: &str) -> Result<&MeyersSegment> {
        single(&self.inner.segments, field, instead)
    }
}

#[extendr]
impl MeyersFits {
    /// "crc", "csr" or "cay".
    fn model(&self) -> &'static str {
        self.inner.segments.fits[0].model().name()
    }

    /// The volume-weighted chain ladder of every segment.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.segments.map(|s| s.chain_ladder.clone()),
        }
    }

    /// Each segment's premium by origin, in the order of the origins.
    fn premium(&self) -> Vec<f64> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.premium.iter().copied())
            .collect()
    }

    fn ultimates(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.ultimates.clone(),
        }
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// One origin's ultimate in each simulation, or the total's with
    /// `origin` NULL, in simulation order.
    fn ultimate_draws(&self, origin: Nullable<List>) -> Result<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// [`ultimate_draws`](Self::ultimate_draws) over `premium`.
    fn loss_ratio_draws(&self, origin: Nullable<List>, premium: f64) -> Result<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// The development ages of a segment's pattern draws, in months
    /// (`DevelopmentPatterns::segment`); `segment` is empty for a fit of
    /// one segment.
    fn pattern_ages(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_ages_of(&self.inner.patterns, &segment)
    }

    /// The pattern draws of a segment, row-major: simulation `i`'s
    /// cumulative share at each age in row `i`.
    fn pattern_draws(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_draws_of(&self.inner.patterns, &segment)
    }

    /// The monthly patterns for the cash-flow engine, row-major: simulation
    /// `i`'s share at the end of each of `months` months in row `i`.
    fn monthly_pattern_draws(
        &self,
        segment: Vec<String>,
        months: f64,
        basis: Robj,
        monotone: &str,
    ) -> Result<Vec<f64>> {
        monthly_pattern_draws_of(&self.inner.patterns, &segment, months, &basis, monotone)
    }

    /// One row per segment: `list(table, warnings)`, as
    /// `segment_diagnostics`.
    fn diagnostics_table(&self) -> List {
        segment_diagnostics(&self.inner.segments, self.inner.totals(), |s| {
            &s.diagnostics
        })
    }

    fn parameter_names(&self) -> Result<Vec<String>> {
        Ok(self.one("draws", "")?.posterior.names().to_vec())
    }

    /// Posterior draws of a single-segment fit, row-major `n_draws x
    /// n_parameters`, chain by chain.
    fn draws(&self) -> Result<Vec<f64>> {
        Ok(self.one("draws", "")?.posterior.values().to_vec())
    }

    fn chains(&self) -> Result<f64> {
        Ok(self.one("draws", "")?.posterior.chains() as f64)
    }

    fn summary(&self) -> Result<List> {
        let s = self.one("summary", "")?.summary().map_err(to_r)?;
        Ok(summary_list(&s))
    }

    /// PSIS-LOO of a single-segment fit's observed cells on the log scale,
    /// as `MeyersFit$loo()`.
    fn loo(&self) -> Result<List> {
        let l = self.one("bayes_loo()", "")?.loo().map_err(to_r)?;
        let e = &l.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.pareto_k,
            k_threshold = l.k_threshold
        ))
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn development_table(&self) -> List {
        fit_table(self.inner.development_table())
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: self.inner.segment(&choice(&keys, &values)?).map_err(to_r)?,
        })
    }
}

/// The `CompartmentalPriors` of `priors` (31 numbers) and `families` (10
/// names): `k_er`, `k_p`, `rlr` and `rrf` as (median, sd); the half priors
/// on the effect sds of `RLR` and `RRF` as (nu, scale) for "student_t" or
/// (scale, NaN) for "normal"; `lkj_eta`; the priors on `log sigma_os` and
/// `log sigma_pd` as (nu, location, scale) for "student_t" or (mean, sd,
/// NaN) for "normal"; then the optional half priors on the origin effect
/// sds of `k_er` and `k_p` and on the development effect sds of `k_er`,
/// `k_p`, `rlr` and `rrf`, each two numbers with family "none" (NaN, NaN)
/// for no such effect.
fn compartmental_priors(priors: &[f64], families: &[String]) -> Result<CompartmentalPriors> {
    if priors.len() != 31 || families.len() != 10 {
        return Err(Error::Other(
            "priors must hold 31 numbers and 10 family names".into(),
        ));
    }
    let positive = |i: usize| PositivePrior::LogNormal {
        median: priors[i],
        sd: priors[i + 1],
    };
    let half = |family: &str, i: usize, arg: &str| -> Result<HalfPrior> {
        match family {
            "student_t" => Ok(HalfPrior::StudentT {
                nu: priors[i],
                scale: priors[i + 1],
            }),
            "normal" => Ok(HalfPrior::Normal { scale: priors[i] }),
            _ => Err(Error::Other(format!(
                "{arg} family must be \"student_t\" or \"normal\", got \"{family}\""
            ))),
        }
    };
    let real = |family: &str, i: usize, arg: &str| -> Result<RealPrior> {
        match family {
            "student_t" => Ok(RealPrior::StudentT {
                nu: priors[i],
                location: priors[i + 1],
                scale: priors[i + 2],
            }),
            "normal" => Ok(RealPrior::Normal {
                mean: priors[i],
                sd: priors[i + 1],
            }),
            _ => Err(Error::Other(format!(
                "{arg} family must be \"student_t\" or \"normal\", got \"{family}\""
            ))),
        }
    };
    let optional = |j: usize, arg: &str| -> Result<Option<HalfPrior>> {
        let family = &families[4 + j];
        if family == "none" {
            Ok(None)
        } else {
            half(family, 19 + 2 * j, arg).map(Some)
        }
    };
    Ok(CompartmentalPriors {
        k_er: positive(0),
        k_p: positive(2),
        rlr: positive(4),
        rrf: positive(6),
        rlr_effect_sd: half(&families[0], 8, "rlr_effect_sd")?,
        rrf_effect_sd: half(&families[1], 10, "rrf_effect_sd")?,
        lkj_eta: priors[12],
        log_sigma: [
            real(&families[2], 13, "log_sigma[[1]]")?,
            real(&families[3], 16, "log_sigma[[2]]")?,
        ],
        ker_effect_sd: optional(0, "ker_effect_sd")?,
        kp_effect_sd: optional(1, "kp_effect_sd")?,
        development_effect_sd: [
            optional(2, "development_effect_sd$k_er")?,
            optional(3, "development_effect_sd$k_p")?,
            optional(4, "development_effect_sd$rlr")?,
            optional(5, "development_effect_sd$rrf")?,
        ],
    })
}

/// The `Compartmental` of `compartmental` and
/// `compartmental_prior_predictive`.
fn compartmental_settings(
    priors: &[f64],
    families: &[String],
    settings: &[String],
    run: &[f64],
    dependence: SegmentDependence,
) -> Result<Compartmental> {
    let [
        likelihood,
        scale,
        paid_response,
        fitted,
        rate_order,
        development_levels,
        start,
    ] = settings
    else {
        return Err(Error::Other("settings must hold 7 names".into()));
    };
    let &[chains, tune, draws, n_sims, seed, target_accept, max_depth] = run else {
        return Err(Error::Other("run must hold 7 numbers".into()));
    };
    let bad = |arg: &str, value: &str, expected: &str| {
        Err(Error::Other(format!(
            "{arg} must be {expected}, got \"{value}\""
        )))
    };
    let likelihood = match likelihood.as_str() {
        "normal" => CompartmentalLikelihood::Normal,
        "lognormal" => CompartmentalLikelihood::LogNormal,
        v => return bad("likelihood", v, "\"normal\" or \"lognormal\""),
    };
    let scale = match scale.as_str() {
        "amount" => CompartmentalScale::Amount,
        "loss_ratio" => CompartmentalScale::LossRatio,
        v => return bad("scale", v, "\"amount\" or \"loss_ratio\""),
    };
    let paid_response = match paid_response.as_str() {
        "cumulative" => PaidResponse::Cumulative,
        "incremental" => PaidResponse::Incremental,
        v => return bad("paid_response", v, "\"cumulative\" or \"incremental\""),
    };
    let fitted = match fitted.as_str() {
        "paid_and_outstanding" => Fitted::PaidAndOutstanding,
        "paid_and_incurred" => Fitted::PaidAndIncurred,
        "paid_only" => Fitted::PaidOnly,
        "incurred_only" => Fitted::IncurredOnly,
        v => {
            return bad(
                "fitted",
                v,
                "\"paid_and_outstanding\", \"paid_and_incurred\", \"paid_only\" or \"incurred_only\"",
            );
        }
    };
    let rate_order = match rate_order.as_str() {
        "free" => RateOrder::Free,
        "ordered" => RateOrder::Ordered,
        v => return bad("rate_order", v, "\"free\" or \"ordered\""),
    };
    let development_levels = match development_levels.as_str() {
        "per_age" => DevelopmentLevels::PerAge,
        "by_year" => DevelopmentLevels::ByYear,
        v => return bad("development_levels", v, "\"per_age\" or \"by_year\""),
    };
    let start = match start.as_str() {
        "data" => CompartmentalStart::Data,
        "prior_centre" => CompartmentalStart::PriorCentre,
        v => return bad("start", v, "\"data\" or \"prior_centre\""),
    };
    let n_sims = whole(n_sims, "n_sims")? as usize;
    if n_sims == 0 {
        return Err(Error::Other("n_sims must be positive".into()));
    }
    let model = Compartmental {
        priors: compartmental_priors(priors, families)?,
        likelihood,
        scale,
        paid_response,
        fitted,
        rate_order,
        development_levels,
        start,
        sampler: BayesSampler {
            chains: whole(chains, "chains")? as usize,
            tune: whole(tune, "tune")? as usize,
            draws: whole(draws, "draws")? as usize,
            seed: 0,
            target_accept,
            max_depth: whole(max_depth, "max_depth")?,
        },
        n_sims,
        seed: whole(seed, "seed")?,
        dependence,
    };
    // Refuse a combination of settings the model does not support when the
    // settings are read, before any data or sampling (#580).
    model.check_supported().map_err(to_r)?;
    Ok(model)
}

/// The flat pieces of a `Backtest` that R builds its tables from.
fn backtest_list(cells: &TriangleFrame, bt: &Backtest) -> Result<List> {
    let (nm, ns) = (bt.models().len(), bt.n_splits());
    // The calendar period of each split, as the frame labels it (the
    // index when the frame has no calendar labels).
    let labels = match cells.frame().column("calendar") {
        Some(Column::Categorical(c)) => Some(c),
        _ => None,
    };
    let diagonals: Vec<String> = bt
        .scored_rows()
        .iter()
        .zip(bt.diagonals())
        .map(|(rows, &index)| match (labels, rows.first()) {
            (Some(c), Some(&r)) => c[r].clone(),
            _ => index.to_string(),
        })
        .collect();
    let mut scores = Vec::with_capacity(nm * bt.metrics().len() * ns);
    for m in 0..nm {
        for k in 0..bt.metrics().len() {
            scores.extend_from_slice(bt.split_scores(m, k));
        }
    }
    // One row per model and split, model-major; NaN where the model
    // has no sampler.
    let (mut rhat, mut bulk, mut tail, mut div, mut nwarn) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for m in 0..nm {
        for d in bt.diagnostics(m) {
            rhat.push(d.as_ref().map_or(f64::NAN, |d| d.max_rhat));
            bulk.push(d.as_ref().map_or(f64::NAN, |d| d.min_ess_bulk));
            tail.push(d.as_ref().map_or(f64::NAN, |d| d.min_ess_tail));
            div.push(d.as_ref().map_or(f64::NAN, |d| d.divergences as f64));
            nwarn.push(d.as_ref().map_or(f64::NAN, |d| d.warnings.len() as f64));
        }
    }
    let warnings = bt.warnings();
    let log_densities: Vec<Robj> = bt
        .log_densities()
        .iter()
        .map(|l| Nullable::from(l.clone()).into())
        .collect();
    Ok(list!(
        models = bt.models().to_vec(),
        metrics = bt
            .metrics()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        n_splits = ns as f64,
        diagonals = diagonals,
        scores = scores,
        excluded = bt.excluded().iter().map(|&e| e as f64).collect::<Vec<_>>(),
        scored_rows = List::from_values(
            bt.scored_rows()
                .iter()
                .map(|r| r.iter().map(|&x| x as f64).collect::<Vec<_>>())
        ),
        log_densities = List::from_values(log_densities),
        max_rhat = rhat,
        min_ess_bulk = bulk,
        min_ess_tail = tail,
        divergences = div,
        n_warnings = nwarn,
        warning_model = warnings.iter().map(|w| w.0 as f64).collect::<Vec<_>>(),
        warning_split = warnings.iter().map(|w| w.1 as f64).collect::<Vec<_>>(),
        warning_message = warnings.iter().map(|w| w.2.clone()).collect::<Vec<_>>()
    ))
}

/// Where a compartmental fit's outstanding comes from: the `outstanding`
/// column itself ("column") or incurred less paid ("incurred").
fn compartmental_outstanding<'a>(
    outstanding: &'a str,
    outstanding_from: &str,
) -> Result<Outstanding<'a>> {
    match outstanding_from {
        "column" => Ok(Outstanding::Column(outstanding)),
        "incurred" => Ok(Outstanding::FromIncurred(outstanding)),
        _ => Err(Error::Other(format!(
            "outstanding_from must be \"column\" or \"incurred\", got \"{outstanding_from}\""
        ))),
    }
}

/// A compartmental model as a candidate of the diagonal backtest: its
/// settings with the triangle and columns it reads. Put in the `models` of
/// `diagonal_backtest` (the backtest takes a clone that shares the fits the
/// handle keeps), its `outstanding_backtest` scores the held-out
/// outstanding on those fits.
#[extendr]
pub(crate) struct CompartmentalBacktestModel {
    inner: CompartmentalCandidate,
}

#[extendr]
impl CompartmentalBacktestModel {
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    fn paid(&self) -> String {
        self.inner.paid().to_string()
    }

    fn premium(&self) -> String {
        self.inner.premium().to_string()
    }

    /// The flat pieces of the outstanding `Backtest` (as
    /// `Triangle$diagonal_backtest` returns the paid one).
    fn outstanding_backtest(
        &self,
        n_diagonals: f64,
        n_sims: f64,
        seed: f64,
        interval: f64,
    ) -> Result<List> {
        let c = &self.inner;
        let cells = TriangleFrame::new(c.triangle(), c.paid(), None).map_err(to_r)?;
        let bt = c
            .outstanding_backtest(
                &cells,
                whole(n_diagonals, "n_diagonals")? as usize,
                whole(n_sims, "n_sims")? as usize,
                whole(seed, "seed")?,
                interval,
            )
            .map_err(to_r)?;
        backtest_list(&cells, &bt)
    }
}

/// The one-year view of a compartmental model: the lifetime fit, the
/// opening reserve, the claims development result and each simulation's
/// Pareto k, by the premium column's origins.
#[extendr]
pub(crate) struct CompartmentalOneYearFit {
    inner: BayesOneYearInner<CompartmentalInner>,
}

#[extendr]
impl CompartmentalOneYearFit {
    fn fit(&self) -> CompartmentalFit {
        CompartmentalFit {
            inner: self.inner.fit.clone(),
        }
    }

    fn origins(&self) -> Vec<String> {
        self.inner
            .fit
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    fn latest_paid(&self) -> Vec<f64> {
        self.inner.fit.latest_paid.clone()
    }

    fn opening_ultimate(&self) -> Vec<f64> {
        self.inner.opening_ultimate.clone()
    }

    fn opening_reserve(&self) -> Vec<f64> {
        self.inner.opening_reserve.clone()
    }

    fn cdr(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.cdr.clone(),
        }
    }

    fn pareto_k(&self) -> Vec<f64> {
        self.inner.pareto_k.clone()
    }

    /// The simulations refitted with NUTS, 1-based.
    fn refitted(&self) -> Vec<f64> {
        self.inner
            .refitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()
    }

    /// The simulations moment matching rescued, 1-based.
    fn moment_matched(&self) -> Vec<f64> {
        self.inner
            .moment_matched
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()
    }

    /// The simulations due for a refit beyond `max_refits`, reweighted
    /// instead, 1-based.
    fn unrefitted(&self) -> Vec<f64> {
        self.inner
            .unrefitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()
    }

    fn refit_warnings(&self) -> f64 {
        self.inner.refit_warnings as f64
    }

    /// Why the view is approximate (one string), or none: set when `refit`
    /// was left to a model with development effects whose default
    /// light refit was turned off (`light_refit = FALSE`), so that it
    /// reweights the flagged simulations only.
    fn approximate(&self) -> Vec<String> {
        self.inner.approximate.iter().cloned().collect()
    }

    fn k_threshold(&self) -> f64 {
        self.inner.k_threshold
    }

    fn flagged_share(&self) -> f64 {
        self.inner.flagged_share()
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    /// The view itself, as for any single-segment fit without keys: a
    /// condition on a key is an error.
    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let one = SegmentFits {
            key_names: Vec::new(),
            labels: vec![Label::new(Vec::<String>::new())],
            fits: vec![()],
        };
        pick(&one, keys, values)?;
        Ok(Self {
            inner: self.inner.clone(),
        })
    }
}

/// The one-year view of a compartmental model on every segment of a
/// triangle: each segment's posterior, opening reserve and Pareto k, and
/// the joint claims development result. The posterior of one segment needs
/// a single-segment fit.
#[extendr]
pub(crate) struct CompartmentalOneYearFits {
    inner: CompartmentalOneYearFitsInner,
}

impl CompartmentalOneYearFits {
    fn one(
        &self,
        field: &str,
        instead: &str,
    ) -> Result<&BayesOneYearSegment<CompartmentalSegment>> {
        single(&self.inner.segments, field, instead)
    }
}

#[extendr]
impl CompartmentalOneYearFits {
    fn n_segments(&self) -> i32 {
        self.inner.segments.len() as i32
    }

    /// Origins of each segment's premium column, in turn.
    fn origins(&self) -> Vec<String> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.fit.origins.iter().map(ToString::to_string))
            .collect()
    }

    fn premium(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.fit.fit.premium.clone())
    }

    fn latest_paid(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.fit.fit.latest_paid.clone())
    }

    fn opening_ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.opening_ultimate.clone())
    }

    fn opening_reserve(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.opening_reserve.clone())
    }

    fn cdr(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.cdr.clone(),
        }
    }

    /// Why the views are approximate (one string), or none: set when
    /// `refit` was left to a model with development effects whose default
    /// light refit was turned off (`light_refit = FALSE`), so that it
    /// reweights the flagged simulations only.
    fn approximate(&self) -> Vec<String> {
        self.inner.approximate.iter().cloned().collect()
    }

    fn pareto_k(&self) -> Result<Vec<f64>> {
        Ok(self.one("pareto_k", "")?.pareto_k.clone())
    }

    /// The simulations of a single-segment fit refitted with NUTS, 1-based.
    fn refitted(&self) -> Result<Vec<f64>> {
        Ok(self
            .one("refitted", "")?
            .refitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect())
    }

    /// The simulations of a single-segment fit that moment matching
    /// rescued, 1-based.
    fn moment_matched(&self) -> Result<Vec<f64>> {
        Ok(self
            .one("moment_matched", "")?
            .moment_matched
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect())
    }

    fn unrefitted(&self) -> Result<Vec<f64>> {
        Ok(self
            .one("unrefitted", "")?
            .unrefitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect())
    }

    fn refit_warnings(&self) -> Result<f64> {
        Ok(self.one("refit_warnings", "")?.refit_warnings as f64)
    }

    /// Each segment's number of refits and of refits with a diagnostic
    /// warning, for the warning of the one-year views over segments.
    fn refit_counts(&self) -> List {
        let fits = &self.inner.segments.fits;
        list!(
            refits = fits
                .iter()
                .map(|s| s.refitted.len() as f64)
                .collect::<Vec<_>>(),
            warned = fits
                .iter()
                .map(|s| s.refit_warnings as f64)
                .collect::<Vec<_>>(),
            unrefitted = fits
                .iter()
                .map(|s| s.unrefitted.len() as f64)
                .collect::<Vec<_>>()
        )
    }

    fn k_threshold(&self) -> f64 {
        self.inner.segments.fits[0].k_threshold
    }

    fn flagged_share(&self) -> Result<f64> {
        Ok(self.one("flagged_share", "totals_frame()")?.flagged_share())
    }

    /// One row per segment: `list(table, warnings)`, as
    /// `segment_diagnostics`.
    fn diagnostics_table(&self) -> List {
        segment_diagnostics(&self.inner.segments, self.inner.totals(), |s| {
            &s.fit.fit.diagnostics
        })
    }

    fn parameter_names(&self) -> Result<Vec<String>> {
        Ok(self
            .one("parameter_names", "")?
            .fit
            .fit
            .posterior
            .names()
            .to_vec())
    }

    /// Posterior draws of a single-segment fit, row-major `n_draws x
    /// n_parameters`, chain by chain.
    fn draws(&self) -> Result<Vec<f64>> {
        Ok(self.one("draws", "")?.fit.fit.posterior.values().to_vec())
    }

    fn chains(&self) -> Result<f64> {
        Ok(self.one("chains", "")?.fit.fit.posterior.chains() as f64)
    }

    fn summary(&self) -> Result<List> {
        let s = self.one("summary", "")?.fit.fit.summary().map_err(to_r)?;
        Ok(summary_list(&s))
    }

    /// PSIS-LOO of a single-segment fit's observed cells, as
    /// `CompartmentalFit$loo()`.
    fn loo(&self) -> Result<List> {
        let l = self.one("bayes_loo()", "")?.fit.fit.loo().map_err(to_r)?;
        let e = &l.loo.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.loo.pareto_k,
            k_threshold = l.loo.k_threshold,
            n_outstanding = l.n_outstanding as f64,
            elpd_outstanding = l.elpd_outstanding,
            elpd_paid = l.elpd_paid
        ))
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: self.inner.segment(&choice(&keys, &values)?).map_err(to_r)?,
        })
    }
}

/// A fitted Gesmann and Morris compartmental model of every segment of a
/// triangle: each segment's posterior and the joint reserves and future
/// paid and outstanding cells. The posterior of one segment needs a
/// single-segment fit.
#[extendr]
pub(crate) struct CompartmentalFits {
    pub(crate) inner: CompartmentalFitsInner,
}

impl CompartmentalFits {
    fn one(&self, field: &str, instead: &str) -> Result<&CompartmentalInner> {
        single(&self.inner.segments, field, instead).map(|s| &s.fit)
    }
}

#[extendr]
impl CompartmentalFits {
    fn n_segments(&self) -> i32 {
        self.inner.segments.len() as i32
    }

    /// Origins of each segment's premium column, in turn.
    fn origins(&self) -> Vec<String> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.origins.iter().map(ToString::to_string))
            .collect()
    }

    fn premium(&self) -> Vec<f64> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.premium.iter().copied())
            .collect()
    }

    fn latest_paid(&self) -> Vec<f64> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.latest_paid.iter().copied())
            .collect()
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// The joint ultimates: the reserves with each origin's latest value
    /// added in every simulation (`UltimateDraws::ultimates`).
    fn ultimates(&self) -> Result<PredictiveDistribution> {
        ultimates_of(&self.inner)
    }

    /// One origin's ultimate in each simulation, or the total's with
    /// `origin` NULL, in simulation order.
    fn ultimate_draws(&self, origin: Nullable<List>) -> Result<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// [`ultimate_draws`](Self::ultimate_draws) over `premium`.
    fn loss_ratio_draws(&self, origin: Nullable<List>, premium: f64) -> Result<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// The development ages of a segment's pattern draws, in months
    /// (`DevelopmentPatterns::segment`); `segment` is empty for a fit of
    /// one segment.
    fn pattern_ages(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_ages_of(&self.inner.patterns, &segment)
    }

    /// The pattern draws of a segment, row-major: simulation `i`'s
    /// cumulative share at each age in row `i`.
    fn pattern_draws(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_draws_of(&self.inner.patterns, &segment)
    }

    /// The monthly patterns for the cash-flow engine, row-major: simulation
    /// `i`'s share at the end of each of `months` months in row `i`.
    fn monthly_pattern_draws(
        &self,
        segment: Vec<String>,
        months: f64,
        basis: Robj,
        monotone: &str,
    ) -> Result<Vec<f64>> {
        monthly_pattern_draws_of(&self.inner.patterns, &segment, months, &basis, monotone)
    }

    fn paid(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.paid.clone(),
        }
    }

    /// The future outstanding (or incurred) cells, or `NULL` for a fit of
    /// paid alone.
    fn outstanding(&self) -> Robj {
        match &self.inner.outstanding {
            Some(inner) => PredictiveDistribution {
                inner: inner.clone(),
            }
            .into(),
            None => ().into(),
        }
    }

    /// Posterior draws with `k_er <= k_p` in each segment, in index order.
    fn rate_swaps(&self) -> Vec<f64> {
        self.inner
            .segments
            .fits
            .iter()
            .map(|s| s.fit.rate_swaps as f64)
            .collect()
    }

    /// One row per segment: `list(table, warnings)`, as
    /// `segment_diagnostics`.
    fn diagnostics_table(&self) -> List {
        segment_diagnostics(&self.inner.segments, self.inner.totals(), |s| {
            &s.fit.diagnostics
        })
    }

    fn parameter_names(&self) -> Result<Vec<String>> {
        Ok(self.one("draws", "")?.posterior.names().to_vec())
    }

    /// Posterior draws of a single-segment fit, row-major `n_draws x
    /// n_parameters`, chain by chain.
    fn draws(&self) -> Result<Vec<f64>> {
        Ok(self.one("draws", "")?.posterior.values().to_vec())
    }

    fn chains(&self) -> Result<f64> {
        Ok(self.one("draws", "")?.posterior.chains() as f64)
    }

    fn summary(&self) -> Result<List> {
        let s = self.one("summary", "")?.summary().map_err(to_r)?;
        Ok(summary_list(&s))
    }

    /// PSIS-LOO of a single-segment fit's observed cells, as
    /// `CompartmentalFit$loo()`.
    fn loo(&self) -> Result<List> {
        let l = self.one("bayes_loo()", "")?.loo().map_err(to_r)?;
        let e = &l.loo.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.loo.pareto_k,
            k_threshold = l.loo.k_threshold,
            n_outstanding = l.n_outstanding as f64,
            elpd_outstanding = l.elpd_outstanding,
            elpd_paid = l.elpd_paid
        ))
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn paid_table(&self) -> Result<List> {
        Ok(fit_table(self.inner.paid_table().map_err(to_r)?))
    }

    fn outstanding_table(&self) -> Result<List> {
        Ok(fit_table(self.inner.outstanding_table().map_err(to_r)?))
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: self.inner.segment(&choice(&keys, &values)?).map_err(to_r)?,
        })
    }
}

/// A fitted Gesmann and Morris compartmental model of a single-segment triangle.
#[extendr]
pub(crate) struct CompartmentalFit {
    pub(crate) inner: CompartmentalInner,
}

#[extendr]
impl CompartmentalFit {
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    fn premium(&self) -> Vec<f64> {
        self.inner.premium.clone()
    }

    fn latest_paid(&self) -> Vec<f64> {
        self.inner.latest_paid.clone()
    }

    fn population_origins(&self) -> Vec<String> {
        self.inner
            .population_origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// The joint ultimates: the reserves with each origin's latest value
    /// added in every simulation (`UltimateDraws::ultimates`).
    fn ultimates(&self) -> Result<PredictiveDistribution> {
        ultimates_of(&self.inner)
    }

    /// One origin's ultimate in each simulation, or the total's with
    /// `origin` NULL, in simulation order.
    fn ultimate_draws(&self, origin: Nullable<List>) -> Result<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// [`ultimate_draws`](Self::ultimate_draws) over `premium`.
    fn loss_ratio_draws(&self, origin: Nullable<List>, premium: f64) -> Result<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// The development ages of a segment's pattern draws, in months
    /// (`DevelopmentPatterns::segment`); `segment` is empty for a fit of
    /// one segment.
    fn pattern_ages(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_ages_of(&self.inner.patterns, &segment)
    }

    /// The pattern draws of a segment, row-major: simulation `i`'s
    /// cumulative share at each age in row `i`.
    fn pattern_draws(&self, segment: Vec<String>) -> Result<Vec<f64>> {
        pattern_draws_of(&self.inner.patterns, &segment)
    }

    /// The monthly patterns for the cash-flow engine, row-major: simulation
    /// `i`'s share at the end of each of `months` months in row `i`.
    fn monthly_pattern_draws(
        &self,
        segment: Vec<String>,
        months: f64,
        basis: Robj,
        monotone: &str,
    ) -> Result<Vec<f64>> {
        monthly_pattern_draws_of(&self.inner.patterns, &segment, months, &basis, monotone)
    }

    fn paid(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.paid.clone(),
        }
    }

    /// The future outstanding (or incurred) cells, or `NULL` for a fit of
    /// paid alone.
    fn outstanding(&self) -> Robj {
        match &self.inner.outstanding {
            Some(inner) => PredictiveDistribution {
                inner: inner.clone(),
            }
            .into(),
            None => ().into(),
        }
    }

    fn ultimate_loss_ratio(&self) -> Vec<f64> {
        self.inner.ultimate_loss_ratio.clone()
    }

    fn rate_swaps(&self) -> f64 {
        self.inner.rate_swaps as f64
    }

    /// The share of draws and observed cells with `k_er <= k_p` when the
    /// rates vary by cell; NA when they are shared.
    fn cell_rate_swaps(&self) -> Rfloat {
        self.inner
            .cell_rate_swaps
            .map_or(Rfloat::na(), Rfloat::from)
    }

    /// `list(max_rhat, min_ess_bulk, min_ess_tail, divergences, warnings)`.
    fn diagnostics(&self) -> List {
        let d = &self.inner.diagnostics;
        list!(
            max_rhat = d.max_rhat,
            min_ess_bulk = d.min_ess_bulk,
            min_ess_tail = d.min_ess_tail,
            divergences = d.divergences as f64,
            warnings = d.warnings.clone()
        )
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, row-major `n_draws x n_parameters`, chain by chain.
    fn draws(&self) -> Vec<f64> {
        self.inner.posterior.values().to_vec()
    }

    fn chains(&self) -> f64 {
        self.inner.posterior.chains() as f64
    }

    /// The posterior summary, columns as `bayes_glm_model@summary`.
    fn summary(&self) -> Result<List> {
        let s = self.inner.summary().map_err(to_r)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        Ok(list!(
            parameter = s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            mean = col(|p| p.mean),
            sd = col(|p| p.sd),
            q05 = col(|p| p.q05),
            q50 = col(|p| p.q50),
            q95 = col(|p| p.q95),
            rhat = col(|p| p.rhat),
            ess_bulk = col(|p| p.ess_bulk),
            ess_tail = col(|p| p.ess_tail)
        ))
    }

    /// PSIS-LOO of the observed cells, outstanding first: `list(estimates,
    /// pointwise, pareto_k, k_threshold, n_outstanding, elpd_outstanding,
    /// elpd_paid)`.
    fn loo(&self) -> Result<List> {
        let l = self.inner.loo().map_err(to_r)?;
        let e = &l.loo.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.loo.pareto_k,
            k_threshold = l.loo.k_threshold,
            n_outstanding = l.n_outstanding as f64,
            elpd_outstanding = l.elpd_outstanding,
            elpd_paid = l.elpd_paid
        ))
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn paid_table(&self) -> Result<List> {
        Ok(fit_table(self.inner.paid_table().map_err(to_r)?))
    }

    fn outstanding_table(&self) -> Result<List> {
        Ok(fit_table(self.inner.outstanding_table().map_err(to_r)?))
    }

    /// The fit itself, as for any single-segment fit without keys: a
    /// condition on a key is an error.
    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let one = SegmentFits {
            key_names: Vec::new(),
            labels: vec![Label::new(Vec::<String>::new())],
            fits: vec![()],
        };
        pick(&one, keys, values)?;
        Ok(Self {
            inner: self.inner.clone(),
        })
    }
}

extendr_module! {
    mod reserving;
    fn pattern_prior_rust;
    fn industry_prior_rust;
    fn industry_prior_segments_rust;
    impl OdpBootstrapFit;
    impl BayesOdpFit;
    impl BayesOdpFits;
    impl BayesOneYearFit;
    impl BayesOneYearFits;
    impl MeyersFit;
    impl MeyersFits;
    impl CompartmentalFit;
    impl CompartmentalBacktestModel;
    impl CompartmentalFits;
    impl CompartmentalOneYearFit;
    impl CompartmentalOneYearFits;
    impl BayesBfFit;
    impl BayesBfNormalisedFit;
    impl BayesNewYear;
    impl HierarchicalBfFit;
    impl HierarchicalBfOneYearFit;
    impl IndustryPriorFit;
    impl MackBootstrapFit;
    impl OneYearFit;
    use basis;
    use chain_ladder;
    use curves;
    use exhibits;
    use expected_loss;
    use latest_diagonal;
    use mack;
    use munich;
    use paid_incurred;
    use tails;
    use triangle;
}
