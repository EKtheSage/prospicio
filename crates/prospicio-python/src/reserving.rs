//! `prospicio.reserving` (Reserving lane): the loss triangle, the chain
//! ladder, Mack's model, the expected-loss methods, Clark's growth curves
//! (and a growth curve fitted to a benchmark pattern)
//! and the ODP bootstrap over `prospicio_reserving` (`docs/design/triangle.md`,
//! `docs/design/reserving-v02.md`).
//!
//! Long tables come in as array-likes (lists, numpy arrays, pandas or
//! Polars columns) and go out as dicts of lists; numpy and pandas are used
//! when the caller passes them but are not required.

use prospicio_core::{Grain, Lag, Month, Period};
use prospicio_reserving::{
    AnnualReading, Apriori, Average, BenchmarkCurve, BenchmarkCurveFit, Benktander,
    BornhuetterFerguson, CapeCod, CapeCodFit, ChainLadder, ChainLadderFit, ClaimsDevelopmentResult,
    ClarkCapeCod, ClarkFit, ClarkLdf, CurveShape, Development, DevelopmentColumn, DevelopmentCurve,
    DevelopmentCurveFit, DevelopmentPatterns, DiagonalAge, ExpectedLoss, ExpectedLossFit,
    FileLayout, FitTable, GrowthCurve, Label, LatestDiagonals, Long, LongDiagonal, Mack,
    MackBootstrap, MackBootstrapFits, MackBootstrapSegment, MackFit, MackProcess, Monotone,
    MonthlyBasis, MunichChainLadder, MunichFit, OdpBootstrap, OdpBootstrapFits,
    OdpBootstrapSegment, OneYearFits, OneYearMethod, OneYearPattern, PastPatternEnd, PatternKind,
    PatternWeight, ProcessDistribution, ReserveFit, SegmentDependence, SegmentFits, SegmentMap,
    SelectedPattern, SigmaInterpolation, Tail, TailBondy, TailConstant, TailCurve, Triangle,
    UnderwritingBasis, view,
};
use prospicio_reserving::{
    CALENDAR_YEAR_CI, CalendarYearTest, DEVELOPMENT_CORRELATION_CI, DevelopmentCorrelationTest,
    DevelopmentExhibit, MethodWeights, SelectedUltimates, UltimateExhibit, UltimateSelection,
    calendar_year_effect as run_calendar_year_effect,
    development_correlation as run_development_correlation, friedland_averages,
};
use prospicio_reserving::{IndustryPrior, IndustryPriorSettings, PatternPrior};
use prospicio_reserving::{
    PaidIncurredChain, PicConvention, PicFit, PicPriors, VarianceExtrapolation,
};
use pyo3::exceptions::{PyImportError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyDate, PyDelta, PyDict, PyFloat, PyInt, PyList, PyModule, PyString, PyTuple,
};

use prospicio_bayes::nuts::{ParamSummary, Sampler as BayesSampler};
use prospicio_reserving::bayes::{
    BayesBf, BayesBfFit, BayesBfNormalised, BayesBfNormalisedFit, BayesOdpFits, BayesOdpSegment,
    BayesOneYear, BayesOneYearFit, BayesOneYearFits, BayesOneYearSegment,
    CUMULATIVE_SUB_ANNUAL_WARNING, Compartmental, CompartmentalCandidate, CompartmentalFit,
    CompartmentalFits, CompartmentalLikelihood, CompartmentalOneYearFits, CompartmentalPriors,
    CompartmentalSegment, CompartmentalStart, DevelopmentLevels, Fitted, HalfPrior, HierarchicalBf,
    HierarchicalBfFit, HierarchicalBfProgram, LayerSharePrior, LightRefit, MeyersFits,
    MeyersSegment, NewOriginLevel, NewYear, OneYearRefit, Outstanding, PaidResponse, PositivePrior,
    RateOrder, RealPrior, Scale as CompartmentalScale, ShareDependence,
};
use prospicio_reserving::{
    Backtest, BayesOdp, BayesOdpFit, Diagnostics, GlmCandidate, Meyers, MeyersFit, MeyersModel,
    NamedModel, TriangleFrame, TriangleModel, diagonal_backtest as run_backtest,
};

use prospicio_glm::Glm;
use prospicio_models::{Column, Terms};

use crate::distributions::{KeyArg, PyPredictiveDistribution, key_from_py};
use crate::models::{PyElpd, PyGlm, PyTerms};
use crate::reserving_discount::{DiscountArgs, PyDiscountCurve, PyDiscountedReserves, discounted};
use crate::to_py;
use prospicio_prob::ComponentKey;
use prospicio_reserving::UltimateDraws;

mod backtest;
mod basis;
mod bayes_bf;
mod bayes_bf_normalised;
mod bayes_diagnostics;
mod bayes_odp;
mod bayes_one_year;
mod chain_ladder;
mod common;
mod compartmental;
mod compartmental_fits;
mod compartmental_priors;
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
mod patterns;
mod priors;
mod tails;
mod triangle;

pub(crate) use backtest::*;
pub(crate) use basis::*;
pub(crate) use bayes_bf::*;
pub(crate) use bayes_bf_normalised::*;
pub(crate) use bayes_diagnostics::*;
pub(crate) use bayes_odp::*;
pub(crate) use bayes_one_year::*;
pub(crate) use chain_ladder::*;
pub(crate) use common::*;
pub(crate) use compartmental::*;
pub(crate) use compartmental_fits::*;
pub(crate) use compartmental_priors::*;
pub(crate) use curves::*;
pub(crate) use exhibits::*;
pub(crate) use expected_loss::*;
pub(crate) use hierarchical_bf::*;
pub(crate) use latest_diagonal::*;
pub(crate) use mack::*;
pub(crate) use mack_bootstrap::*;
pub(crate) use meyers::*;
pub(crate) use munich::*;
pub(crate) use odp_bootstrap::*;
pub(crate) use one_year::*;
pub(crate) use paid_incurred::*;
pub(crate) use patterns::*;
pub(crate) use priors::*;
pub(crate) use tails::*;
pub(crate) use triangle::*;
