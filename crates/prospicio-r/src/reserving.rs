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
mod bayes_bf;
mod bayes_models;
mod bayes_odp;
mod bayes_one_year;
mod chain_ladder;
mod common;
mod compartmental;
mod curves;
mod exhibits;
mod expected_loss;
mod hierarchical_bf;
mod latest_diagonal;
mod mack;
mod mack_bootstrap;
mod meyers;
mod munich;
mod odp_bootstrap;
mod one_year;
mod paid_incurred;
mod priors;
mod tails;
mod triangle;

pub(crate) use basis::*;
pub(crate) use bayes_bf::*;
pub(crate) use bayes_models::*;
pub(crate) use bayes_odp::*;
pub(crate) use bayes_one_year::*;
pub(crate) use chain_ladder::*;
pub(crate) use common::*;
pub(crate) use compartmental::*;
pub(crate) use curves::*;
pub(crate) use exhibits::*;
pub(crate) use expected_loss::*;
pub(crate) use hierarchical_bf::*;
pub(crate) use mack::*;
pub(crate) use mack_bootstrap::*;
pub(crate) use meyers::*;
pub(crate) use munich::*;
pub(crate) use odp_bootstrap::*;
pub(crate) use one_year::*;
pub(crate) use paid_incurred::*;
pub(crate) use priors::*;
pub(crate) use tails::*;
pub(crate) use triangle::*;

extendr_module! {
    mod reserving;
    use basis;
    use bayes_bf;
    use bayes_odp;
    use bayes_one_year;
    use chain_ladder;
    use compartmental;
    use curves;
    use exhibits;
    use expected_loss;
    use hierarchical_bf;
    use latest_diagonal;
    use mack;
    use mack_bootstrap;
    use meyers;
    use munich;
    use odp_bootstrap;
    use one_year;
    use paid_incurred;
    use priors;
    use tails;
    use triangle;
}
