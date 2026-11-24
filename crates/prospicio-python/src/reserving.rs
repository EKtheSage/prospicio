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
mod chain_ladder;
mod common;
mod curves;
mod exhibits;
mod expected_loss;
mod latest_diagonal;
mod mack;
mod mack_bootstrap;
mod munich;
mod odp_bootstrap;
mod one_year;
mod paid_incurred;
mod patterns;
mod tails;
mod triangle;

pub(crate) use backtest::*;
pub(crate) use basis::*;
pub(crate) use chain_ladder::*;
pub(crate) use common::*;
pub(crate) use curves::*;
pub(crate) use exhibits::*;
pub(crate) use expected_loss::*;
pub(crate) use latest_diagonal::*;
pub(crate) use mack::*;
pub(crate) use mack_bootstrap::*;
pub(crate) use munich::*;
pub(crate) use odp_bootstrap::*;
pub(crate) use one_year::*;
pub(crate) use paid_incurred::*;
pub(crate) use patterns::*;
pub(crate) use tails::*;
pub(crate) use triangle::*;

/// The Bayesian chain ladder of England and Verrall (2002, 2006): the
/// over-dispersed Poisson model of a cumulative triangle's incremental
/// values ``X`` with mean ``exp(c + a[w] + b[d])`` (``a[1] = b[1] = 0``),
/// normal priors on the log scale, and the chain ladder's Pearson scale
/// ``phi`` (``OdpBootstrapFit.scale``) plugged in. The posterior is sampled
/// with NUTS, and each draw projects the future cells up to the oldest
/// observed age (no tail) with the process error of ``process``, as
/// ``OdpBootstrap`` does. Every origin's and every development column's
/// observed increments must sum to more than zero; negative cells are
/// allowed.
///
/// One ``seed`` replays a fit exactly, whatever the number of threads: the
/// chains and the simulations each take a seed derived from it.
///
/// Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS below 100
/// per chain, any divergent transition) do not fail the fit: ``fit``
/// raises one ``UserWarning`` listing them, and they stay in
/// ``BayesOdpFit.diagnostics``.
///
/// Parameters
/// ----------
/// intercept_sd : float, default 10.0
///     Standard deviation of the normal prior on the intercept ``c``,
///     centred on the log of the mean observed increment.
/// origin_sd : float, default 10.0
///     Standard deviation of the normal prior on each origin effect
///     ``a[w]``, ``w >= 2``, centred on 0.
/// development_sd : float, default 10.0
///     Standard deviation of the normal prior on each development effect
///     ``b[d]``, ``d >= 2``, centred on 0.
/// process : {"gamma", "none"}, default "gamma"
///     Process error on each simulated future increment: Gamma with mean
///     ``mu`` and variance ``phi * mu``, or none for parameter error only.
/// chains, tune, draws : int, default 4, 1000, 1000
///     NUTS chains, warm-up draws per chain and kept draws per chain.
/// n_sims : int, default 10000
///     Number of predictive simulations, independent of the number of
///     posterior draws; positive.
/// seed : int, default 0
///     Seed of the chains and of the simulations, from 0 to ``2**64 - 1``.
/// target_accept : float, default 0.95
///     Target mean acceptance rate of the step-size adaptation; above
///     NUTS's usual 0.8, which on triangles such as RAA usually leaves a
///     few divergent transitions.
/// max_depth : int, default 10
///     Largest NUTS tree depth.
/// dependence : {"independent", "rank_correlation", "synchronized"}, default "independent"
///     How the segments of a multi-segment triangle depend on each other
///     in ``fit_segments`` (``fit`` has one segment and does not use it):
///     each segment's own posterior and simulations, with seeds derived
///     from ``seed`` and the segment's position so that the segments share
///     no random numbers; or the segments' simulations reordered as whole
///     rows (Iman–Conover on the segment totals) to the Spearman matrix
///     ``spearman`` and then put in a random order, as
///     ``OdpBootstrap``'s. ``"synchronized"`` is accepted here, as the
///     bootstraps take it, but ``fit_segments`` raises: it resamples the
///     residuals of the same cells in every segment, and a posterior has no
///     residuals to share.
/// spearman : list of list of float, optional
///     With ``dependence="rank_correlation"`` only: Spearman's rho between
///     the segments' totals, as ``OdpBootstrap``'s.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero, ``process`` or ``dependence`` is unknown, or
///     ``spearman`` is missing for ``"rank_correlation"``, given for another
///     dependence or not square.
/// OverflowError
///     If an integer setting is negative or too large.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import BayesOdp, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> fit = BayesOdp(n_sims=2000, seed=1).fit(tri, "values")
/// >>> fit.reserves.components()
/// [('2020',), ('2021',), ('2022',), ('2023',)]
/// >>> fit.reserves.mean() > 0
/// True
/// >>> fit.parameter_names[:2]
/// ['c', 'a[2]']
#[pyclass(name = "BayesOdp", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesOdp {
    inner: BayesOdp,
}

#[pymethods]
impl PyBayesOdp {
    #[new]
    #[pyo3(signature = (
        intercept_sd = 10.0,
        origin_sd = 10.0,
        development_sd = 10.0,
        process = "gamma",
        chains = 4,
        tune = 1000,
        draws = 1000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.95,
        max_depth = 10,
        dependence = "independent",
        spearman = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        intercept_sd: f64,
        origin_sd: f64,
        development_sd: f64,
        process: &str,
        chains: usize,
        tune: usize,
        draws: usize,
        n_sims: usize,
        seed: u64,
        target_accept: f64,
        max_depth: u64,
        dependence: &str,
        spearman: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            inner: BayesOdp {
                intercept_sd,
                origin_sd,
                development_sd,
                process: process_distribution(process)?,
                sampler: BayesSampler {
                    chains,
                    tune,
                    draws,
                    seed: 0,
                    target_accept,
                    max_depth,
                },
                n_sims,
                seed,
                dependence: segment_dependence(dependence, spearman)?,
            },
        })
    }

    /// Standard deviation of the intercept's prior.
    #[getter]
    fn intercept_sd(&self) -> f64 {
        self.inner.intercept_sd
    }

    /// Standard deviation of each origin effect's prior.
    #[getter]
    fn origin_sd(&self) -> f64 {
        self.inner.origin_sd
    }

    /// Standard deviation of each development effect's prior.
    #[getter]
    fn development_sd(&self) -> f64 {
        self.inner.development_sd
    }

    /// Process error: ``"gamma"`` or ``"none"``.
    #[getter]
    fn process(&self) -> &'static str {
        process_name(self.inner.process)
    }

    /// Number of NUTS chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.sampler.chains
    }

    /// Warm-up draws per chain.
    #[getter]
    fn tune(&self) -> usize {
        self.inner.sampler.tune
    }

    /// Kept draws per chain.
    #[getter]
    fn draws(&self) -> usize {
        self.inner.sampler.draws
    }

    /// Number of predictive simulations.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims
    }

    /// Seed of the chains and of the simulations.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Target mean acceptance rate.
    #[getter]
    fn target_accept(&self) -> f64 {
        self.inner.sampler.target_accept
    }

    /// Largest NUTS tree depth.
    #[getter]
    fn max_depth(&self) -> u64 {
        self.inner.sampler.max_depth
    }

    /// How segments depend on each other in ``fit_segments``:
    /// ``"independent"``, ``"rank_correlation"`` or ``"synchronized"``.
    #[getter]
    fn dependence(&self) -> &'static str {
        dependence_name(&self.inner.dependence)
    }

    /// The Spearman matrix of ``dependence="rank_correlation"``; ``None``
    /// otherwise.
    #[getter]
    fn spearman(&self) -> Option<Vec<Vec<f64>>> {
        dependence_spearman(&self.inner.dependence)
    }

    /// Samples the posterior of one measure column of a single-segment
    /// cumulative triangle and simulates its reserves. Every origin must be
    /// observed from the first age up to its latest, there must be more
    /// observed cells than parameters, and every origin's and every
    /// development column's increments must sum to more than zero. Raises
    /// one ``UserWarning`` listing the failed diagnostics, if any.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///
    /// Returns
    /// -------
    /// BayesOdpFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``OdpBootstrap.fit``; if the triangle has several segments; if
    ///     a prior standard deviation is not positive and finite or a
    ///     sampler setting is invalid; or if an origin's or a development
    ///     column's increments do not sum to more than zero.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyBayesOdpFit> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py.detach(|| model.fit(tri, column)).map_err(err)?;
        warn_diagnostics(py, "BayesOdp", &inner.diagnostics.warnings)?;
        Ok(PyBayesOdpFit { inner })
    }

    /// Samples the posterior of one measure column in every segment of a
    /// cumulative triangle, each segment with its own posterior (the same
    /// model and settings, sampled independently), into one joint
    /// distribution of the reserves, the segments depending on each other
    /// as ``dependence`` says. Segment ``k`` (in index order) takes seeds
    /// derived from ``seed`` and ``k`` for its chains and its simulations,
    /// so independent segments share no random numbers, and a segment's
    /// draws differ from ``fit`` of that segment alone. Raises one
    /// ``UserWarning`` listing the failed diagnostics of every segment,
    /// each message prefixed with its segment's label.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// column : str
    ///
    /// Returns
    /// -------
    /// BayesOdpFits
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit`` for any segment (the message names the segment); if
    ///     ``dependence`` is ``"synchronized"`` (before any sampling); or if
    ///     ``spearman`` does not fit the segments.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import BayesOdp, Triangle
    /// >>> paid = [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0]
    /// >>> tri = Triangle.from_long(
    /// ...     ([2020] * 4 + [2021] * 3 + [2022] * 2 + [2023]) * 2,
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12] * 2,
    /// ...     {"paid": paid + [2 * v for v in paid]},
    /// ...     keys={"lob": ["Auto"] * 10 + ["Home"] * 10},
    /// ... )
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     fits = BayesOdp(tune=300, draws=300, n_sims=1000).fit_segments(tri, "paid")
    /// >>> fits.index, fits.reserves.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    /// >>> fits.segment(lob="Home").scale > 0
    /// True
    fn fit_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyBayesOdpFits> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py.detach(|| model.fit_segments(tri, column)).map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.diagnostics);
        warn_diagnostics(py, "BayesOdp", &warnings)?;
        Ok(PyBayesOdpFits { inner })
    }

    /// The one-year view of one measure column of a single-segment
    /// cumulative triangle: the claims development result over the coming
    /// year, by Pareto smoothed importance sampling of the posterior
    /// (decision 16 of ``docs/design/reserving-bayes.md``). The posterior
    /// and the lifetime reserves are ``fit``'s for the same settings. Each
    /// simulation picks a posterior draw, draws the year's increments from
    /// it with ``process`` error (the cells valued in the next twelve
    /// months, as in ``OdpBootstrap.one_year``), reweights every posterior
    /// draw by the year's quasi likelihood with ``scale`` and the priors
    /// unchanged, and smooths the weights (one Pareto ``k`` per
    /// simulation). Origin ``w``'s claims development result is
    ///
    /// ``CDR[w] = E[R[w] | D0] - (X[w] + E[R'[w] | D0, X])``
    ///
    /// with ``R`` the expected run-off at the opening, ``X`` the year's
    /// payments and ``R'`` the expected run-off after the year, each
    /// expectation the (weighted) mean over the posterior draws; its mean
    /// is about zero. A simulation whose ``k`` is above ``k_threshold`` is
    /// flagged. With ``refit="moment_match_then_refit"`` (the default) its
    /// posterior draws are first moved by affine maps to the moments the
    /// weights estimate (moment matching, as R's
    /// ``loo::loo_moment_match``), and only a simulation whose ``k`` stays
    /// above ``k_threshold`` is refitted with NUTS on the triangle extended
    /// by its year, with this model's sampler. 14% to 26% of the
    /// simulations are flagged on RAA and GenIns; matching rescues nearly
    /// all of them, within 0.04 to 0.06 sd of their refits, at about 1 ms
    /// each against about 26 ms for a refit, so a default run of 10,000
    /// simulations takes a few seconds in a release build.
    /// ``"refit_flagged"`` refits every flagged simulation (one to two
    /// minutes there), and ``"reweight_only"`` takes about a second and
    /// leaves the flagged simulations' reweighting as it is. Every cell up to the end of the
    /// year is observed and conditioned on, also for an origin that lags
    /// the valuation. Raises one ``UserWarning`` listing the failed
    /// diagnostics of the fit, and another if any refit failed its
    /// diagnostics.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    /// k_threshold : float, default 0.7
    ///     The Pareto ``k`` above which a simulation's importance weights
    ///     are not trusted (the threshold of Vehtari et al. 2024 for a few
    ///     thousand draws).
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, default "moment_match_then_refit"
    ///     What happens to a flagged simulation: moment-matched, and
    ///     refitted only if its ``k`` stays above ``k_threshold``
    ///     (``moment_matched`` lists the rescued); refitted with NUTS
    ///     (``"refit_flagged"``); or reweighted anyway (fast, biased where
    ///     ``k`` is high); ``"refit_all"`` refits every simulation, to
    ///     check the reweighting.
    /// max_refits : int, optional
    ///     The most simulations refitted with NUTS: the first this many due
    ///     for a refit, in simulation order; the others are reweighted as
    ///     they are (as unreliable as their ``k`` says), listed in
    ///     ``unrefitted`` and warned about. ``None`` (the default) refits
    ///     every one, ``0`` none.
    /// n_sims : int, optional
    ///     The number of one-year simulations; by default this model's
    ///     ``n_sims``. A smaller count gives the first simulations of a
    ///     larger one.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit``; if ``k_threshold`` is NaN or ``refit`` is unknown.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import BayesOdp, Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    /// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    /// ... )
    /// >>> view = BayesOdp(n_sims=1000, seed=1).one_year(tri, "values", refit="reweight_only")
    /// >>> view.cdr.components()
    /// [('2020',), ('2021',), ('2022',), ('2023',)]
    /// >>> len(view.pareto_k), view.refitted
    /// (1000, [])
    /// >>> view.cdr.variance() < view.fit.reserves.variance()
    /// True
    #[pyo3(signature = (triangle, column, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFit> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year(tri, column, &settings))
            .map_err(err)?;
        warn_diagnostics(py, "BayesOdp", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "BayesOdp",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        Ok(PyBayesOneYearFit {
            inner: BayesOneYearOf::Odp(inner),
        })
    }

    /// The one-year view of one measure column in every segment of a
    /// cumulative triangle: each segment's posterior and simulations as in
    /// ``fit_segments`` (segment ``k``'s own seeds), reweighted and
    /// refitted as in ``one_year``, into one joint claims development
    /// result, the segments depending on each other as ``dependence`` says:
    /// ``"independent"``, or ``"rank_correlation"`` by Iman–Conover on the
    /// segments' total claims development results, as the bootstraps'
    /// one-year view does. Raises one ``UserWarning`` listing the failed
    /// diagnostics of every segment's fit, and another if any refit failed
    /// its diagnostics, each message prefixed with its segment's label.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// column : str
    /// k_threshold : float, default 0.7
    ///     As in ``one_year``.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, default "moment_match_then_refit"
    ///     As in ``one_year``.
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``, for each segment: ``k`` segments
    ///     refit at most ``k`` times this many.
    /// n_sims : int, optional
    ///     As in ``one_year``.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFits
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``one_year`` and ``fit_segments``, including
    ///     ``dependence="synchronized"``.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import BayesOdp, Triangle
    /// >>> paid = [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0]
    /// >>> tri = Triangle.from_long(
    /// ...     ([2020] * 4 + [2021] * 3 + [2022] * 2 + [2023]) * 2,
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12] * 2,
    /// ...     {"paid": paid + [2 * v for v in paid]},
    /// ...     keys={"lob": ["Auto"] * 10 + ["Home"] * 10},
    /// ... )
    /// >>> model = BayesOdp(tune=300, draws=300, n_sims=500)
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     views = model.one_year_segments(tri, "paid", refit="reweight_only")
    /// >>> views.index, views.cdr.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    /// >>> len(views.segment(lob="Home").pareto_k)
    /// 500
    #[pyo3(signature = (triangle, column, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFits> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year_segments(tri, column, &settings))
            .map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.fit.diagnostics);
        warn_diagnostics(py, "BayesOdp", &warnings)?;
        warn_refits(py, "BayesOdp", &segment_refits(&inner.segments))?;
        Ok(PyBayesOneYearFits {
            inner: BayesOneYearFitsOf::Odp(inner),
        })
    }

    fn __repr__(&self) -> String {
        let s = &self.inner.sampler;
        format!(
            "BayesOdp(intercept_sd={:?}, origin_sd={:?}, development_sd={:?}, process={:?}, \
             chains={}, tune={}, draws={}, n_sims={}, seed={}{})",
            self.inner.intercept_sd,
            self.inner.origin_sd,
            self.inner.development_sd,
            self.process(),
            s.chains,
            s.tune,
            s.draws,
            self.inner.n_sims,
            self.inner.seed,
            dependence_repr(&self.inner.dependence),
        )
    }
}

/// A fitted ``BayesOdp``: the posterior of ``c``, ``a[2..n]`` and
/// ``b[2..m]``, its MCMC diagnostics, and the posterior predictive
/// distribution of the reserves, laid out as ``OdpBootstrapFit.reserves``.
/// The frames' ``latest``, ``ultimate`` and ``reserve`` are the
/// volume-weighted chain ladder's (``chain_ladder``); ``mean`` and
/// ``std_dev`` are the posterior predictive reserve's.
#[pyclass(name = "BayesOdpFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesOdpFit {
    inner: BayesOdpFit,
}

#[pymethods]
impl PyBayesOdpFit {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// Each simulation's reserve is spread over the pattern of its own
    /// posterior draw.
    ///
    /// Parameters
    /// ----------
    /// basis : {"paid", "incurred"}
    ///     What the fit's column is, stated by the caller (no default), so
    ///     that a fit on incurred claims is not discounted by its reporting
    ///     pattern by mistake. ``"paid"`` pays each simulation's reserve
    ///     along that simulation's own pattern. ``"incurred"`` pays each
    ///     reserve (the IBNR) plus the origin's case reserve (its latest
    ///     incurred less ``latest_paid``) along ``paid_pattern``.
    /// rate : float, optional
    ///     A flat annual effective rate. Give ``rate`` or ``curve``.
    /// curve : DiscountCurve, optional
    ///     Spot rates by maturity.
    /// timing : {"mid", "end"}, default "mid"
    ///     Payments at the middle or the end of each development period.
    /// paid_pattern : list of float, optional
    ///     For ``basis="incurred"``: the cumulative paid share at each of
    ///     the fit's pattern ages (with the tail's age when the fit has a
    ///     tail), ending at 1.
    /// latest_paid : list of float, optional
    ///     For ``basis="incurred"``: each origin's latest paid, one per
    ///     component of ``reserves`` in their order.
    ///
    /// Returns
    /// -------
    /// DiscountedReserves
    #[pyo3(signature = (
        *,
        basis,
        rate = None,
        curve = None,
        timing = "mid",
        paid_pattern = None,
        latest_paid = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn discounted(
        &self,
        py: Python<'_>,
        basis: &str,
        rate: Option<f64>,
        curve: Option<PyRef<'_, PyDiscountCurve>>,
        timing: &str,
        paid_pattern: Option<Vec<f64>>,
        latest_paid: Option<Vec<f64>>,
    ) -> PyResult<PyDiscountedReserves> {
        let args = DiscountArgs {
            basis,
            rate,
            curve,
            timing,
            paid_pattern,
            latest_paid,
        };
        discounted(py, &self.inner, args)
    }

    /// Each simulation's development pattern, to pair with the
    /// simulation's loss ratio in the cash-flow engine.
    ///
    /// Row ``i`` of every pattern is simulation ``i`` of ``reserves``,
    /// ``ultimates`` and ``ultimate_draws``.
    ///
    /// Returns
    /// -------
    /// DevelopmentPatterns
    #[getter]
    fn patterns(&self) -> PyDevelopmentPatterns {
        PyDevelopmentPatterns {
            inner: self.inner.patterns.clone(),
        }
    }
    /// The volume-weighted chain ladder of the triangle, whose latest
    /// values and Pearson scale the model uses.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.inner.chain_ladder.clone()],
            },
        }
    }

    /// Origin period of each reserve component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .chain_ladder
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.chain_ladder.development.development.clone()
    }

    /// The plugged-in scale ``phi``, the chain ladder's Pearson scale (the
    /// ODP bootstrap's).
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    /// Joint posterior predictive distribution of the reserve (the sum of
    /// future incremental values) by origin: dimension ``"origin"``, one
    /// component per origin, one row per simulation.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// Joint distribution of the ultimate by origin: ``reserves`` with
    /// each origin's latest value added in every simulation, the same
    /// dimensions, components and rows, so ``join``, ``select`` and
    /// ``reorder_groups`` keep each simulation's origins together.
    #[getter]
    fn ultimates(&self) -> PyResult<PyPredictiveDistribution> {
        ultimates_of(&self.inner)
    }

    /// Each simulation's ultimate, for a tower, a deal book or the cash-flow
    /// engine: index ``i`` is simulation ``i`` for every origin and every
    /// call, the same as row ``i`` of ``reserves`` and ``ultimates``, and
    /// the same again when the fit is run with the same seed.
    ///
    /// The ultimate is the origin's latest value plus its reserve in that
    /// simulation.
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     The component key (``(2021,)``) or the origin alone; an origin period by its label
    ///     (``"2021"``, ``2021``, ``"2021Q3"``). ``None`` (the default) gives
    ///     the total over every component, summed simulation by simulation.
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (origin = None))]
    fn ultimate_draws(&self, origin: Option<OriginArg>) -> PyResult<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// Each simulation's ultimate over ``premium``: the loss ratio a
    /// contract on the loss ratio reads, in the order of
    /// ``ultimate_draws``. Multiply by the premium for
    /// ``Tower.apply_annual``.
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     As in ``ultimate_draws``; ``None`` for the total.
    /// premium : float
    ///     The subject premium of the origin (of every origin for the
    ///     total), finite and positive.
    ///
    /// Returns
    /// -------
    /// list of float
    fn loss_ratio_draws(&self, origin: Option<OriginArg>, premium: f64) -> PyResult<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// MCMC diagnostics: a dict with ``max_rhat``, ``min_ess_bulk``,
    /// ``min_ess_tail``, ``divergences`` and ``warnings`` (one message per
    /// failed threshold; empty when every threshold holds).
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = &self.inner.diagnostics;
        let out = PyDict::new(py);
        out.set_item("max_rhat", d.max_rhat)?;
        out.set_item("min_ess_bulk", d.min_ess_bulk)?;
        out.set_item("min_ess_tail", d.min_ess_tail)?;
        out.set_item("divergences", d.divergences)?;
        out.set_item("warnings", d.warnings.clone())?;
        Ok(out)
    }

    /// Parameter names: ``c``, ``a[2]`` to ``a[n]``, ``b[2]`` to ``b[m]``.
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, one row per draw (chain by chain), one column per
    /// parameter in the order of ``parameter_names``.
    #[getter]
    fn draws(&self) -> Vec<Vec<f64>> {
        let p = self.inner.posterior.dim().max(1);
        self.inner
            .posterior
            .values()
            .chunks(p)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// Number of chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.posterior.chains()
    }

    /// Posterior summary, one row per parameter: ``parameter``, ``mean``,
    /// ``sd``, ``q05``, ``q50``, ``q95``, ``rhat``, ``ess_bulk`` and
    /// ``ess_tail``, the fields of ``BayesGlmFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fit = &self.inner;
        let s = py.detach(|| fit.summary()).map_err(err)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        let out = PyDict::new(py);
        out.set_item(
            "parameter",
            s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
        )?;
        out.set_item("mean", col(|p| p.mean))?;
        out.set_item("sd", col(|p| p.sd))?;
        out.set_item("q05", col(|p| p.q05))?;
        out.set_item("q50", col(|p| p.q50))?;
        out.set_item("q95", col(|p| p.q95))?;
        out.set_item("rhat", col(|p| p.rhat))?;
        out.set_item("ess_bulk", col(|p| p.ess_bulk))?;
        out.set_item("ess_tail", col(|p| p.ess_tail))?;
        py.import("pandas")?.call_method1("DataFrame", (out,))
    }

    /// PSIS-LOO of the observed increments, with each cell's relative
    /// efficiency from the chains, as ``BayesGlmFit.loo``. The pointwise
    /// term is the normalised over-dispersed Poisson density at the cell's
    /// mean and ``scale``, so compare the result only with models of the
    /// same incremental values.
    ///
    /// Returns
    /// -------
    /// Elpd
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an observed increment is negative (the density does not
    ///     exist there).
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let fit = &self.inner;
        let l = py.detach(|| fit.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l))
    }

    /// One row per origin: ``origin``, the chain ladder's ``latest``,
    /// ``ultimate`` and ``reserve``, and the ``mean`` and ``std_dev`` of
    /// the posterior predictive reserve. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the chain ladder's totals, the ``scale``, and the ``mean``
    /// and ``std_dev`` of the posterior predictive total reserve. Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The chain ladder's development factors, one row per age, as
    /// ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesOdpFit(origins={}, draws={}, n_sims={}, scale={:?}, warnings={})",
            self.inner.chain_ladder.origins.len(),
            self.inner.posterior.n_draws(),
            self.inner.reserves.n_sims(),
            self.inner.scale,
            self.inner.diagnostics.warnings.len(),
        )
    }
}

/// One ``UserWarning`` from `model` listing failed diagnostics, if any.
fn warn_diagnostics(py: Python<'_>, model: &str, warnings: &[String]) -> PyResult<()> {
    if warnings.is_empty() {
        return Ok(());
    }
    let message = format!(
        "{model}: the MCMC diagnostics failed; check them before relying on the fit: {}",
        warnings.join("; ")
    );
    let message =
        std::ffi::CString::new(message).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
    PyErr::warn(py, category.as_any(), &message, 1)
}

/// `Compartmental`'s warnings: the warning of cumulative paid on cells finer
/// than a year fires on healthy fits too, so it comes as a ``UserWarning`` of
/// its own and not as a failed diagnostic.
fn warn_compartmental(py: Python<'_>, warnings: &[String]) -> PyResult<()> {
    let (notes, failed): (Vec<String>, Vec<String>) = warnings
        .iter()
        .cloned()
        .partition(|w| w.contains(CUMULATIVE_SUB_ANNUAL_WARNING));
    warn_diagnostics(py, "Compartmental", &failed)?;
    if notes.is_empty() {
        return Ok(());
    }
    let message = format!("Compartmental: {}", notes.join("; "));
    let message =
        std::ffi::CString::new(message).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
    PyErr::warn(py, category.as_any(), &message, 1)
}

/// MCMC diagnostics as a dict, as ``BayesOdpFit.diagnostics``.
fn diagnostics_dict<'py>(py: Python<'py>, d: &Diagnostics) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("max_rhat", d.max_rhat)?;
    out.set_item("min_ess_bulk", d.min_ess_bulk)?;
    out.set_item("min_ess_tail", d.min_ess_tail)?;
    out.set_item("divergences", d.divergences)?;
    out.set_item("warnings", d.warnings.clone())?;
    Ok(out)
}

/// Every segment's failed diagnostics, each message prefixed with its
/// segment's label when the triangle has keys.
fn segment_warnings<T>(segments: &SegmentFits<T>, diag: fn(&T) -> &Diagnostics) -> Vec<String> {
    segments
        .labels
        .iter()
        .zip(&segments.fits)
        .flat_map(|(label, fit)| {
            let prefix = if segments.key_names.is_empty() {
                String::new()
            } else {
                format!("{label}: ")
            };
            diag(fit)
                .warnings
                .iter()
                .map(move |w| format!("{prefix}{w}"))
        })
        .collect()
}

/// One row per segment: the key columns, then ``max_rhat``,
/// ``min_ess_bulk``, ``min_ess_tail``, ``divergences`` and ``warnings``
/// (the segment's messages joined by ``"; "``).
fn segment_diagnostics_frame<'py, T>(
    py: Python<'py>,
    segments: &SegmentFits<T>,
    diag: fn(&T) -> &Diagnostics,
) -> PyResult<Bound<'py, PyAny>> {
    let out = PyDict::new(py);
    for (k, name) in segments.key_names.iter().enumerate() {
        let values: Vec<String> = segments
            .labels
            .iter()
            .map(|l| l.parts().get(k).cloned().unwrap_or_default())
            .collect();
        out.set_item(name, values)?;
    }
    let col = |g: fn(&Diagnostics) -> f64| {
        segments
            .fits
            .iter()
            .map(|f| g(diag(f)))
            .collect::<Vec<f64>>()
    };
    out.set_item("max_rhat", col(|d| d.max_rhat))?;
    out.set_item("min_ess_bulk", col(|d| d.min_ess_bulk))?;
    out.set_item("min_ess_tail", col(|d| d.min_ess_tail))?;
    out.set_item(
        "divergences",
        segments
            .fits
            .iter()
            .map(|f| diag(f).divergences)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "warnings",
        segments
            .fits
            .iter()
            .map(|f| diag(f).warnings.join("; "))
            .collect::<Vec<_>>(),
    )?;
    py.import("pandas")?.call_method1("DataFrame", (out,))
}

/// A posterior summary as a pandas DataFrame, as ``BayesOdpFit.summary``.
fn summary_frame<'py>(py: Python<'py>, s: &[ParamSummary]) -> PyResult<Bound<'py, PyAny>> {
    let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
    let out = PyDict::new(py);
    out.set_item(
        "parameter",
        s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
    )?;
    out.set_item("mean", col(|p| p.mean))?;
    out.set_item("sd", col(|p| p.sd))?;
    out.set_item("q05", col(|p| p.q05))?;
    out.set_item("q50", col(|p| p.q50))?;
    out.set_item("q95", col(|p| p.q95))?;
    out.set_item("rhat", col(|p| p.rhat))?;
    out.set_item("ess_bulk", col(|p| p.ess_bulk))?;
    out.set_item("ess_tail", col(|p| p.ess_tail))?;
    py.import("pandas")?.call_method1("DataFrame", (out,))
}

/// ``BayesOdp.fit_segments``: each segment's own posterior and the joint
/// posterior predictive distribution of the reserves, laid out as
/// ``OdpBootstrapFit``.
///
/// ``reserves`` has the triangle's keys and ``"origin"`` as dimensions, so
/// ``reserves.aggregate(["lob"])`` keeps the dependence between segments.
/// Per-origin lists run over the origins of each segment in turn, like the
/// rows of ``to_frame()`` and the components of ``reserves``. The
/// posterior of one segment (``scale``, ``diagnostics``,
/// ``parameter_names``, ``draws``, ``chains``, ``summary()`` and
/// ``loo()``) needs a single-segment fit: for several segments use
/// ``segment(...)``, ``totals_frame()`` (the scales) or
/// ``diagnostics_frame()``.
#[pyclass(name = "BayesOdpFits", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesOdpFits {
    inner: BayesOdpFits,
}

impl PyBayesOdpFits {
    fn one(&self, field: &str, instead: &str) -> PyResult<&BayesOdpSegment> {
        single(&self.inner.segments, field, instead)
    }
}

#[pymethods]
impl PyBayesOdpFits {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// Each segment is spread over its own posterior patterns, keeping the
    /// simulations together across segments.
    ///
    /// Parameters
    /// ----------
    /// basis : {"paid", "incurred"}
    ///     What the fit's column is, stated by the caller (no default), so
    ///     that a fit on incurred claims is not discounted by its reporting
    ///     pattern by mistake. ``"paid"`` pays each simulation's reserve
    ///     along that simulation's own pattern. ``"incurred"`` pays each
    ///     reserve (the IBNR) plus the origin's case reserve (its latest
    ///     incurred less ``latest_paid``) along ``paid_pattern``.
    /// rate : float, optional
    ///     A flat annual effective rate. Give ``rate`` or ``curve``.
    /// curve : DiscountCurve, optional
    ///     Spot rates by maturity.
    /// timing : {"mid", "end"}, default "mid"
    ///     Payments at the middle or the end of each development period.
    /// paid_pattern : list of float, optional
    ///     For ``basis="incurred"``: the cumulative paid share at each of
    ///     the fit's pattern ages (with the tail's age when the fit has a
    ///     tail), ending at 1.
    /// latest_paid : list of float, optional
    ///     For ``basis="incurred"``: each origin's latest paid, one per
    ///     component of ``reserves`` in their order.
    ///
    /// Returns
    /// -------
    /// DiscountedReserves
    #[pyo3(signature = (
        *,
        basis,
        rate = None,
        curve = None,
        timing = "mid",
        paid_pattern = None,
        latest_paid = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn discounted(
        &self,
        py: Python<'_>,
        basis: &str,
        rate: Option<f64>,
        curve: Option<PyRef<'_, PyDiscountCurve>>,
        timing: &str,
        paid_pattern: Option<Vec<f64>>,
        latest_paid: Option<Vec<f64>>,
    ) -> PyResult<PyDiscountedReserves> {
        let args = DiscountArgs {
            basis,
            rate,
            curve,
            timing,
            paid_pattern,
            latest_paid,
        };
        discounted(py, &self.inner, args)
    }

    /// Each simulation's development pattern, to pair with the
    /// simulation's loss ratio in the cash-flow engine.
    ///
    /// Row ``i`` of every pattern is simulation ``i`` of ``reserves``,
    /// ``ultimates`` and ``ultimate_draws``.
    ///
    /// Returns
    /// -------
    /// DevelopmentPatterns
    #[getter]
    fn patterns(&self) -> PyDevelopmentPatterns {
        PyDevelopmentPatterns {
            inner: self.inner.patterns.clone(),
        }
    }
    /// The volume-weighted chain ladder of every segment, whose latest
    /// values and Pearson scales the model uses.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.segments.map(|s| s.chain_ladder.clone()),
        }
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.segments.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner.segments)
    }

    /// Origin period of each per-origin value and reserve component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        origin_labels(&self.inner.segments)
    }

    /// Development ages in months of the first segment.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.segments.fits[0]
            .chain_ladder
            .development
            .development
            .clone()
    }

    /// The plugged-in scale ``phi`` of a single-segment fit.
    #[getter]
    fn scale(&self) -> PyResult<f64> {
        Ok(self.one("scale", "totals_frame()")?.scale)
    }

    /// Joint posterior predictive distribution of the reserve by segment
    /// and origin: the triangle's keys and ``"origin"`` are its dimensions,
    /// one component per segment and origin, one row per simulation.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// Joint distribution of the ultimate by segment and origin: ``reserves`` with
    /// each origin's latest value added in every simulation, the same
    /// dimensions, components and rows, so ``join``, ``select`` and
    /// ``reorder_groups`` keep each simulation's origins together.
    #[getter]
    fn ultimates(&self) -> PyResult<PyPredictiveDistribution> {
        ultimates_of(&self.inner)
    }

    /// Each simulation's ultimate, for a tower, a deal book or the cash-flow
    /// engine: index ``i`` is simulation ``i`` for every origin and every
    /// call, the same as row ``i`` of ``reserves`` and ``ultimates``, and
    /// the same again when the fit is run with the same seed.
    ///
    /// The ultimate is the origin's latest value plus its reserve in that
    /// simulation.
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     The component key, the segment's key values then the origin (``("Auto", 2021)``); an origin period by its label
    ///     (``"2021"``, ``2021``, ``"2021Q3"``). ``None`` (the default) gives
    ///     the total over every component, summed simulation by simulation.
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (origin = None))]
    fn ultimate_draws(&self, origin: Option<OriginArg>) -> PyResult<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// Each simulation's ultimate over ``premium``: the loss ratio a
    /// contract on the loss ratio reads, in the order of
    /// ``ultimate_draws``. Multiply by the premium for
    /// ``Tower.apply_annual``.
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     As in ``ultimate_draws``; ``None`` for the total.
    /// premium : float
    ///     The subject premium of the origin (of every origin for the
    ///     total), finite and positive.
    ///
    /// Returns
    /// -------
    /// list of float
    fn loss_ratio_draws(&self, origin: Option<OriginArg>, premium: f64) -> PyResult<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// MCMC diagnostics of a single-segment fit, as
    /// ``BayesOdpFit.diagnostics``.
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        diagnostics_dict(
            py,
            &self.one("diagnostics", "diagnostics_frame()")?.diagnostics,
        )
    }

    /// Parameter names of a single-segment fit, as
    /// ``BayesOdpFit.parameter_names``.
    #[getter]
    fn parameter_names(&self) -> PyResult<Vec<String>> {
        Ok(self.one("parameter_names", "")?.posterior.names().to_vec())
    }

    /// Posterior draws of a single-segment fit, as ``BayesOdpFit.draws``.
    #[getter]
    fn draws(&self) -> PyResult<Vec<Vec<f64>>> {
        let posterior = &self.one("draws", "")?.posterior;
        let p = posterior.dim().max(1);
        Ok(posterior.values().chunks(p).map(<[f64]>::to_vec).collect())
    }

    /// Number of chains of a single-segment fit.
    #[getter]
    fn chains(&self) -> PyResult<usize> {
        Ok(self.one("chains", "")?.posterior.chains())
    }

    /// Posterior summary of a single-segment fit, as
    /// ``BayesOdpFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let one = self.one("summary()", "")?;
        let s = py.detach(|| one.summary()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// PSIS-LOO of a single-segment fit's observed increments, as
    /// ``BayesOdpFit.loo``.
    ///
    /// Returns
    /// -------
    /// Elpd
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the fit has several segments, or an observed increment is
    ///     negative.
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let one = self.one("loo()", "")?;
        let l = py.detach(|| one.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l))
    }

    /// One row per segment: the key columns, then ``max_rhat``,
    /// ``min_ess_bulk``, ``min_ess_tail``, ``divergences`` and
    /// ``warnings`` (the segment's messages joined by ``"; "``, empty when
    /// every threshold holds). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn diagnostics_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        segment_diagnostics_frame(py, &self.inner.segments, |s| &s.diagnostics)
    }

    /// One row per segment and origin: the key columns, ``origin``, the
    /// chain ladder's ``latest``, ``ultimate`` and ``reserve``, and the
    /// ``mean`` and ``std_dev`` of the posterior predictive reserve. Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the chain ladder's totals,
    /// the ``scale``, and the ``mean`` and ``std_dev`` of the segment's
    /// posterior predictive total reserve. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The chain ladders' development factors, one row per segment and
    /// age, as ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint reserves
    /// (same dimensions).
    ///
    /// Returns
    /// -------
    /// BayesOdpFits
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        Ok(Self {
            inner: self.inner.segment(&keys).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        let segments = &self.inner.segments;
        let scale = match segments.fits.as_slice() {
            [one] => format!(", scale={:?}", one.scale),
            _ => String::new(),
        };
        let warnings: usize = segments
            .fits
            .iter()
            .map(|f| f.diagnostics.warnings.len())
            .sum();
        format!(
            "BayesOdpFits({}origins={}, n_sims={}{scale}, warnings={warnings})",
            segments_prefix(segments),
            self.origins().len(),
            self.inner.reserves.n_sims(),
        )
    }
}

/// The settings of a Bayesian one-year view: `k_threshold`, the `refit`
/// policy by name (`None` for the model's one-year default), the refit
/// budget and the number of simulations (`None` for the model's
/// one-year default).
fn one_year_settings(
    k_threshold: f64,
    refit: Option<&str>,
    max_refits: Option<usize>,
    n_sims: Option<usize>,
) -> PyResult<BayesOneYear> {
    let refit = match refit {
        None => None,
        Some("refit_flagged") => Some(OneYearRefit::RefitFlagged),
        Some("reweight_only") => Some(OneYearRefit::ReweightOnly),
        Some("refit_all") => Some(OneYearRefit::RefitAll),
        Some("moment_match_then_refit") => Some(OneYearRefit::MomentMatchThenRefit),
        Some(refit) => {
            return Err(PyValueError::new_err(format!(
                "refit must be \"moment_match_then_refit\", \"refit_flagged\", \
                 \"reweight_only\" or \"refit_all\", got {refit:?}"
            )));
        }
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

/// A compartmental or hierarchical one-year view's ``light_refit``:
/// ``None`` for the model's default (model 2's light refit), ``False`` to
/// turn it off, ``True`` for [`LightRefit::default`] (4 chains of 150
/// warm-up and 250 kept draws), or ``(chains, tune, draws)``. Returns the
/// sampler and whether the default is turned off.
fn light_refit_arg(light_refit: Option<&Bound<'_, PyAny>>) -> PyResult<(Option<LightRefit>, bool)> {
    let Some(arg) = light_refit else {
        return Ok((None, false));
    };
    if arg.is_none() {
        return Ok((None, false));
    }
    if let Ok(flag) = arg.cast::<pyo3::types::PyBool>() {
        return Ok(if flag.is_true() {
            (Some(LightRefit::default()), false)
        } else {
            (None, true)
        });
    }
    let message = "light_refit must be None, a bool or a tuple (chains, tune, draws) of                    positive integers";
    let (chains, tune, draws): (usize, usize, usize) =
        arg.extract().map_err(|_| PyValueError::new_err(message))?;
    Ok((
        Some(LightRefit {
            chains,
            tune,
            draws,
        }),
        false,
    ))
}

/// One ``UserWarning`` from `model` if any NUTS refit of a one-year view
/// failed its diagnostics, and another if any simulation due for a refit
/// was left unrefitted by ``max_refits``: `refits` holds each segment's
/// label prefix, its number of refits, how many of them warned and how
/// many simulations were left unrefitted.
fn warn_refits(
    py: Python<'_>,
    model: &str,
    refits: &[(String, usize, usize, usize)],
) -> PyResult<()> {
    let warn = |message: String| -> PyResult<()> {
        let message =
            std::ffi::CString::new(message).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
        PyErr::warn(py, category.as_any(), &message, 1)
    };
    let failed: Vec<String> = refits
        .iter()
        .filter(|(_, _, warned, _)| *warned > 0)
        .map(|(prefix, n, warned, _)| format!("{prefix}{warned} of {n} refits"))
        .collect();
    if !failed.is_empty() {
        warn(format!(
            "{model}: the MCMC diagnostics of NUTS refits in the one-year view failed (R-hat, \
             ESS or a divergence): {}",
            failed.join("; ")
        ))?;
    }
    let left: Vec<String> = refits
        .iter()
        .filter(|(_, _, _, left)| *left > 0)
        .map(|(prefix, _, _, left)| format!("{prefix}{left}"))
        .collect();
    if !left.is_empty() {
        warn(format!(
            "{model}: simulations due for a refit beyond max_refits were reweighted instead, \
             their reweighting as unreliable as their Pareto k (listed in unrefitted): {}",
            left.join("; ")
        ))?;
    }
    Ok(())
}

/// The one-year view of a Bayesian model on a single-segment triangle,
/// from ``BayesOdp.one_year``, ``BayesBf.one_year``,
/// ``BayesBfNormalised.one_year`` or ``Meyers.one_year``: the lifetime fit,
/// the opening reserve, the claims development result over the coming
/// year and how reliable each simulation's importance reweighting was.
///
/// ``cdr`` is the joint distribution of the claims development result by
/// origin (dimension ``"origin"``), the opening reserve less the year's
/// payment and the posterior mean reserve after the year, so a negative
/// value is an adverse development and ``cdr.quantile(0.005)`` is minus
/// the one-year value at risk at 99.5%. ``pareto_k`` has one Pareto ``k``
/// per simulation; ``flagged_share`` is the share above ``k_threshold``,
/// whether or not they were refitted, ``refitted`` lists the simulations
/// refitted with NUTS and ``moment_matched`` those moment matching
/// rescued (``refit="moment_match_then_refit"``, the default). ``fit`` is
/// the model's own lifetime fit, a ``BayesOdpFit``, a ``BayesBfFit``, a
/// ``BayesBfNormalisedFit`` or a ``MeyersFit``.
#[pyclass(name = "BayesOneYearFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesOneYearFit {
    inner: BayesOneYearOf,
}

/// The one-year view of each Bayesian model with one, by its lifetime fit.
enum BayesOneYearOf {
    Odp(BayesOneYearFit<BayesOdpFit>),
    Bf(BayesOneYearFit<BayesBfFit>),
    BfNormalised(BayesOneYearFit<BayesBfNormalisedFit>),
    Meyers(Box<BayesOneYearFit<MeyersFit>>),
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

impl PyBayesOneYearFit {
    /// The lifetime fit's chain ladder.
    fn chain_ladder_fit(&self) -> &ChainLadderFit {
        each_bayes_one_year!(&self.inner, f => f.fit.chain_ladder())
    }
}

#[pymethods]
impl PyBayesOneYearFit {
    /// The model fitted to the observed triangle, with its posterior and
    /// lifetime reserves, as the model's ``fit`` gives it: a
    /// ``BayesOdpFit`` from ``BayesOdp.one_year``, a ``BayesBfFit`` from
    /// ``BayesBf.one_year``, a ``BayesBfNormalisedFit`` from
    /// ``BayesBfNormalised.one_year``, a ``MeyersFit`` from
    /// ``Meyers.one_year``.
    ///
    /// Returns
    /// -------
    /// BayesOdpFit, BayesBfFit, BayesBfNormalisedFit or MeyersFit
    #[getter]
    fn fit(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match &self.inner {
            BayesOneYearOf::Odp(f) => Py::new(
                py,
                PyBayesOdpFit {
                    inner: f.fit.clone(),
                },
            )?
            .into_any(),
            BayesOneYearOf::Bf(f) => Py::new(
                py,
                PyBayesBfFit {
                    inner: f.fit.clone(),
                },
            )?
            .into_any(),
            BayesOneYearOf::BfNormalised(f) => Py::new(
                py,
                PyBayesBfNormalisedFit {
                    inner: f.fit.clone(),
                },
            )?
            .into_any(),
            BayesOneYearOf::Meyers(f) => Py::new(
                py,
                PyMeyersFit {
                    inner: f.fit.clone(),
                },
            )?
            .into_any(),
        })
    }

    /// The model's volume-weighted chain ladder.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.chain_ladder_fit().clone()],
            },
        }
    }

    /// Origin period of each per-origin value and CDR component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.chain_ladder_fit()
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.chain_ladder_fit().development.development.clone()
    }

    /// Latest observed value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        self.chain_ladder_fit().latest.clone()
    }

    /// The latest value plus the opening reserve, per origin.
    #[getter]
    fn opening_ultimate(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.opening_ultimate.clone())
    }

    /// Each origin's posterior mean reserve on the observed triangle: the
    /// mean over the posterior draws of the expected future increments.
    #[getter]
    fn opening_reserve(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.opening_reserve.clone())
    }

    /// Joint distribution of the claims development result by origin:
    /// dimension ``"origin"``, one component per origin, one row per
    /// simulation.
    #[getter]
    fn cdr(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: each_bayes_one_year!(&self.inner, f => f.cdr.clone()),
        }
    }

    /// Each simulation's Pareto ``k``, in the order of the simulations
    /// (``inf`` when too few weights differ to fit the tail).
    #[getter]
    fn pareto_k(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.pareto_k.clone())
    }

    /// The simulations refitted with NUTS, in order (0-based).
    #[getter]
    fn refitted(&self) -> Vec<usize> {
        each_bayes_one_year!(&self.inner, f => f.refitted.clone())
    }

    /// The simulations moment matching rescued (their ``k`` brought to at
    /// most ``k_threshold``), in order (0-based); empty unless ``refit``
    /// was ``"moment_match_then_refit"``, when each flagged simulation is
    /// in this, in ``refitted`` or in ``unrefitted``.
    #[getter]
    fn moment_matched(&self) -> Vec<usize> {
        each_bayes_one_year!(&self.inner, f => f.moment_matched.clone())
    }

    /// The simulations due for a refit beyond ``max_refits``, reweighted
    /// instead (as unreliable as their ``k`` says), in order (0-based);
    /// empty without ``max_refits``.
    #[getter]
    fn unrefitted(&self) -> Vec<usize> {
        each_bayes_one_year!(&self.inner, f => f.unrefitted.clone())
    }

    /// How many of the refits had a diagnostic warning (R-hat, ESS or a
    /// divergence).
    #[getter]
    fn refit_warnings(&self) -> usize {
        each_bayes_one_year!(&self.inner, f => f.refit_warnings)
    }

    /// The Pareto ``k`` above which a simulation is flagged.
    #[getter]
    fn k_threshold(&self) -> f64 {
        each_bayes_one_year!(&self.inner, f => f.k_threshold)
    }

    /// The share of simulations whose ``k`` is above ``k_threshold`` (or
    /// NaN), whether or not they were refitted.
    #[getter]
    fn flagged_share(&self) -> f64 {
        each_bayes_one_year!(&self.inner, f => f.flagged_share())
    }

    /// One row per origin: ``origin``, ``latest``, ``opening_ultimate``,
    /// ``opening_reserve``, the model's own columns (``BayesBf``'s
    /// ``exposure`` and ``prior_ultimate`` and its ``opening_ulr``, the
    /// opening ultimate over the exposure), and the ``cdr_mean`` and
    /// ``cdr_std_dev`` of the claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, each_bayes_one_year!(&self.inner, f => f.to_long()))
    }

    /// One row: the totals of ``to_frame()``'s columns, the ``scale``, the
    /// ``flagged_share``, and the ``cdr_mean`` and ``cdr_std_dev`` of the
    /// total claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, each_bayes_one_year!(&self.inner, f => f.totals()))
    }

    /// The chain ladder's development factors, one row per age, as
    /// ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(
            py,
            each_bayes_one_year!(&self.inner, f => f.fit.development_table()),
        )
    }

    fn __repr__(&self) -> String {
        each_bayes_one_year!(&self.inner, f => format!(
            "BayesOneYearFit(origins={}, n_sims={}, flagged_share={:?}, refitted={})",
            f.opening_reserve.len(),
            f.cdr.n_sims(),
            f.flagged_share(),
            f.refitted.len(),
        ))
    }
}

/// The one-year view of a Bayesian model in every segment of a triangle,
/// from ``BayesOdp.one_year_segments`` or ``Meyers.one_year_segments``,
/// laid out as ``OneYearFit``.
///
/// ``cdr`` is one joint distribution of the claims development result with
/// the triangle's keys and ``"origin"`` as dimensions, so
/// ``cdr.aggregate(["lob"])`` keeps the dependence between segments.
/// Per-origin lists run over the origins of each segment in turn, like the
/// rows of ``to_frame()`` and the components of ``cdr``. ``scale`` (the
/// ODP's only), ``premium`` (Meyers' only), ``diagnostics``, ``pareto_k``,
/// ``refitted``, ``refit_warnings`` and ``flagged_share`` are one
/// segment's and need a single-segment fit: for several segments use
/// ``segment(...)``, ``totals_frame()`` (the scales or premiums and the
/// flagged shares) or ``diagnostics_frame()``.
#[pyclass(name = "BayesOneYearFits", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesOneYearFits {
    inner: BayesOneYearFitsOf,
}

/// The one-year view over segments of each Bayesian model with one, by
/// its segment fit.
enum BayesOneYearFitsOf {
    Odp(BayesOneYearFits<BayesOdpSegment>),
    Meyers(BayesOneYearFits<MeyersSegment>),
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

/// A segment's label prefix, its number of refits and how many of them
/// warned, for ``warn_refits``.
fn segment_refits<S>(
    segments: &SegmentFits<BayesOneYearSegment<S>>,
) -> Vec<(String, usize, usize, usize)> {
    segments
        .labels
        .iter()
        .zip(&segments.fits)
        .map(|(label, s)| {
            let prefix = if segments.key_names.is_empty() {
                String::new()
            } else {
                format!("{label}: ")
            };
            (
                prefix,
                s.refitted.len(),
                s.refit_warnings,
                s.unrefitted.len(),
            )
        })
        .collect()
}

#[pymethods]
impl PyBayesOneYearFits {
    /// The volume-weighted chain ladder of every segment.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: each_bayes_one_years!(&self.inner, f => f
                .segments
                .map(|s| s.fit.chain_ladder.clone())),
        }
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        each_bayes_one_years!(&self.inner, f => f.segments.key_names.clone())
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        each_bayes_one_years!(&self.inner, f => segment_labels(py, &f.segments))
    }

    /// Origin period of each per-origin value and CDR component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        each_bayes_one_years!(&self.inner, f => origin_labels(&f.segments))
    }

    /// Development ages in months of the first segment.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        each_bayes_one_years!(&self.inner, f => f.segments.fits[0]
            .fit
            .chain_ladder
            .development
            .development
            .clone())
    }

    /// Latest observed value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        each_bayes_one_years!(&self.inner, f => by_origin(&f.segments, |s| s
            .fit
            .chain_ladder
            .latest
            .clone()))
    }

    /// The latest value plus the opening reserve, per origin.
    #[getter]
    fn opening_ultimate(&self) -> Vec<f64> {
        each_bayes_one_years!(&self.inner, f => by_origin(&f.segments, |s| s
            .opening_ultimate
            .clone()))
    }

    /// Each origin's posterior mean reserve on the observed triangle.
    #[getter]
    fn opening_reserve(&self) -> Vec<f64> {
        each_bayes_one_years!(&self.inner, f => by_origin(&f.segments, |s| s
            .opening_reserve
            .clone()))
    }

    /// Joint distribution of the claims development result by segment and
    /// origin: the triangle's keys and ``"origin"`` are its dimensions, one
    /// component per segment and origin, one row per simulation.
    #[getter]
    fn cdr(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: each_bayes_one_years!(&self.inner, f => f.cdr.clone()),
        }
    }

    /// The plugged-in scale ``phi`` of a single-segment ``BayesOdp`` view.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For several segments, or for a model without a scale
    ///     (``Meyers``).
    #[getter]
    fn scale(&self) -> PyResult<f64> {
        match &self.inner {
            BayesOneYearFitsOf::Odp(f) => {
                Ok(single(&f.segments, "scale", "totals_frame()")?.fit.scale)
            }
            BayesOneYearFitsOf::Meyers(_) => Err(PyValueError::new_err(
                "scale is the ODP's plugged-in dispersion; Meyers' models have none",
            )),
        }
    }

    /// The premium of each origin of a single-segment ``Meyers`` view.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For several segments, or for a model without premium
    ///     (``BayesOdp``).
    #[getter]
    fn premium(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            BayesOneYearFitsOf::Meyers(f) => Ok(single(&f.segments, "premium", "totals_frame()")?
                .fit
                .premium
                .clone()),
            BayesOneYearFitsOf::Odp(_) => Err(PyValueError::new_err(
                "premium is Meyers' input; BayesOdp has none",
            )),
        }
    }

    /// MCMC diagnostics of a single-segment fit's posterior, as
    /// ``BayesOdpFit.diagnostics``.
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        each_bayes_one_years!(&self.inner, f => diagnostics_dict(
            py,
            &single(&f.segments, "diagnostics", "diagnostics_frame()")?
                .fit
                .diagnostics,
        ))
    }

    /// Each simulation's Pareto ``k`` in a single-segment fit.
    #[getter]
    fn pareto_k(&self) -> PyResult<Vec<f64>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "pareto_k", "")?
            .pareto_k
            .clone()))
    }

    /// The simulations of a single-segment fit refitted with NUTS.
    #[getter]
    fn refitted(&self) -> PyResult<Vec<usize>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "refitted", "")?
            .refitted
            .clone()))
    }

    /// The simulations of a single-segment fit that moment matching
    /// rescued.
    #[getter]
    fn moment_matched(&self) -> PyResult<Vec<usize>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "moment_matched", "")?
            .moment_matched
            .clone()))
    }

    /// The simulations of a single-segment fit due for a refit beyond
    /// ``max_refits``, reweighted instead.
    #[getter]
    fn unrefitted(&self) -> PyResult<Vec<usize>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "unrefitted", "")?
            .unrefitted
            .clone()))
    }

    /// How many of a single-segment fit's refits had a diagnostic warning.
    #[getter]
    fn refit_warnings(&self) -> PyResult<usize> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "refit_warnings", "")?
            .refit_warnings))
    }

    /// The Pareto ``k`` above which a simulation is flagged.
    #[getter]
    fn k_threshold(&self) -> f64 {
        each_bayes_one_years!(&self.inner, f => f.segments.fits[0].k_threshold)
    }

    /// The share of a single-segment fit's simulations whose ``k`` is
    /// above ``k_threshold`` (or NaN).
    #[getter]
    fn flagged_share(&self) -> PyResult<f64> {
        each_bayes_one_years!(&self.inner, f => Ok(single(
            &f.segments,
            "flagged_share",
            "totals_frame()"
        )?
        .flagged_share()))
    }

    /// One row per segment: the key columns, then ``max_rhat``,
    /// ``min_ess_bulk``, ``min_ess_tail``, ``divergences`` and
    /// ``warnings`` of the segment's posterior, as
    /// ``BayesOdpFits.diagnostics_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn diagnostics_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            BayesOneYearFitsOf::Odp(f) => {
                segment_diagnostics_frame(py, &f.segments, |s| &s.fit.diagnostics)
            }
            BayesOneYearFitsOf::Meyers(f) => {
                segment_diagnostics_frame(py, &f.segments, |s| &s.fit.diagnostics)
            }
        }
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``latest``, ``opening_ultimate``, ``opening_reserve``, the model's
    /// own columns (Meyers' ``premium`` and ``opening_ulr``, the opening
    /// ultimate over the premium), and the ``cdr_mean`` and
    /// ``cdr_std_dev`` of the claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, each_bayes_one_years!(&self.inner, f => f.to_long()))
    }

    /// One row per segment: the key columns, the totals of ``to_frame()``'s
    /// columns, the model's own totals (the ODP's ``scale``, Meyers'
    /// ``premium``), the ``flagged_share``, and the ``cdr_mean`` and
    /// ``cdr_std_dev`` of the segment's total claims development result.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, each_bayes_one_years!(&self.inner, f => f.totals()))
    }

    /// The chain ladders' development factors, one row per segment and
    /// age, as ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(
            py,
            each_bayes_one_years!(&self.inner, f => f.segments.development_table()),
        )
    }

    /// The one-year view of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint claims
    /// development result (same dimensions).
    ///
    /// Returns
    /// -------
    /// BayesOneYearFits
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let inner = match &self.inner {
            BayesOneYearFitsOf::Odp(f) => BayesOneYearFitsOf::Odp(f.segment(&keys).map_err(err)?),
            BayesOneYearFitsOf::Meyers(f) => {
                BayesOneYearFitsOf::Meyers(f.segment(&keys).map_err(err)?)
            }
        };
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        each_bayes_one_years!(&self.inner, f => {
            let segments = &f.segments;
            let detail = match segments.fits.as_slice() {
                [one] => format!(
                    ", flagged_share={:?}, refitted={}",
                    one.flagged_share(),
                    one.refitted.len()
                ),
                _ => String::new(),
            };
            format!(
                "BayesOneYearFits({}origins={}, n_sims={}{detail})",
                segments_prefix(segments),
                origin_labels(segments).len(),
                f.cdr.n_sims(),
            )
        })
    }
}

/// The Bayesian Bornhuetter–Ferguson of England, Verrall and Wüthrich
/// (2012, Model 3.1):the over-dispersed Poisson model of a cumulative
/// triangle's incremental values ``X`` with mean ``mu[w] * gamma[d]``, an
/// informative gamma prior on each origin's row parameter ``mu[w]`` with
/// mean ``apriori * exposure[w]`` and coefficient of variation ``cv``, a
/// vague normal prior on each ``log gamma[d]`` (the pattern is not
/// normalised), and the
/// chain ladder's Pearson scale ``phi`` (``OdpBootstrapFit.scale``) plugged
/// in. The posterior is sampled with NUTS, and each draw projects the
/// future cells up to the oldest observed age (no tail) with the process
/// error of ``process``, as ``BayesOdp`` does.
///
/// The prior's strength moves the reserve between two limits. With
/// ``cv=float("inf")`` (a flat prior on ``log mu[w]``) the model is the
/// Bayesian chain ladder, and the reserves agree with ``BayesOdp``'s within
/// Monte Carlo error. As ``cv`` falls to 0, ``mu[w]`` is fixed at the prior
/// ultimate ``m[w]`` and each origin's mean reserve tends to ``m[w]`` times
/// the sum of the future ``S[d] / M[d]`` (``S[d]`` the observed increments
/// of column ``d``, ``M[d]`` the prior ultimates of the origins observed
/// there): EVW's strong-prior limit, Mack's (2008) Bornhuetter–Ferguson
/// with the raw pattern. That equals the classical
/// ``BornhuetterFerguson`` reserve ``(1 - 1 / cdf) * m[w]`` only when the
/// prior ultimates are the chain ladder's.
///
/// ``mu[w]`` is EVW's row parameter, not the origin's expected ultimate:
/// that is ``mu[w]`` times the sum of the ``gamma[d]``, which under a tight
/// prior is ``m[w]`` times the sum of ``S[d] / M[d]`` over all columns. On
/// EVW's triangle that sum is 0.885, so the implied ultimate loss ratio is
/// about 11.5% below ``apriori``.
///
/// Every development column's observed increments must sum to more than
/// zero, and every origin's to more than ``-phi / cv**2``; negative cells
/// are allowed. One ``seed`` replays a fit exactly, whatever the number of
/// threads. Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS
/// below 100 per chain, any divergent transition) do not fail the fit:
/// ``fit`` raises one ``UserWarning`` listing them, and they stay in
/// ``BayesBfFit.diagnostics``.
///
/// Parameters
/// ----------
/// apriori : float, default 1.0
///     Prior mean of each origin's ``mu[w]`` per unit of exposure;
///     positive and finite. It is the expected loss ratio only when the
///     pattern ``gamma`` sums to 1 (see above).
/// cv : float, default 0.1
///     Coefficient of variation of the gamma prior on each ``mu[w]`` (shape
///     ``1 / cv**2``, EVW's ``a``); positive, ``float("inf")`` for a flat
///     prior.
/// development_sd : float, default 10.0
///     Standard deviation of the normal prior on each ``log gamma[d]``,
///     centred on 0. With a flat ``cv``, a much vaguer prior (1,000, say)
///     leaves divergent transitions.
/// process : {"gamma", "none"}, default "gamma"
///     Process error on each simulated future increment: Gamma with mean
///     ``mu`` and variance ``phi * mu``, or none for parameter error only.
/// chains, tune, draws : int, default 4, 1000, 1000
///     NUTS chains, warm-up draws per chain and kept draws per chain.
/// n_sims : int, default 10000
///     Number of predictive simulations, independent of the number of
///     posterior draws; positive.
/// seed : int, default 0
///     Seed of the chains and of the simulations, from 0 to ``2**64 - 1``.
/// target_accept : float, default 0.95
///     Target mean acceptance rate of the step-size adaptation.
/// max_depth : int, default 10
///     Largest NUTS tree depth.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero or ``process`` is unknown.
/// OverflowError
///     If an integer setting is negative or too large.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import BayesBf, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     {
/// ...         "paid": [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ...         "premium": [200.0] * 4 + [220.0] * 3 + [240.0] * 2 + [260.0],
/// ...     },
/// ... )
/// >>> fit = BayesBf(apriori=0.85, cv=0.1, n_sims=2000, seed=1).fit(tri, "paid", "premium")
/// >>> fit.reserves.components()
/// [('2020',), ('2021',), ('2022',), ('2023',)]
/// >>> fit.prior_ultimate[3] == 0.85 * 260.0
/// True
/// >>> fit.parameter_names[:2]
/// ['alpha[1]', 'alpha[2]']
#[pyclass(name = "BayesBf", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesBf {
    inner: BayesBf,
}

#[pymethods]
impl PyBayesBf {
    #[new]
    #[pyo3(signature = (
        apriori = 1.0,
        cv = 0.1,
        development_sd = 10.0,
        process = "gamma",
        chains = 4,
        tune = 1000,
        draws = 1000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.95,
        max_depth = 10,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        apriori: f64,
        cv: f64,
        development_sd: f64,
        process: &str,
        chains: usize,
        tune: usize,
        draws: usize,
        n_sims: usize,
        seed: u64,
        target_accept: f64,
        max_depth: u64,
    ) -> PyResult<Self> {
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            inner: BayesBf {
                apriori,
                cv,
                development_sd,
                process: process_distribution(process)?,
                sampler: BayesSampler {
                    chains,
                    tune,
                    draws,
                    seed: 0,
                    target_accept,
                    max_depth,
                },
                n_sims,
                seed,
            },
        })
    }

    /// Prior mean of each origin's ``mu[w]`` per unit of exposure.
    #[getter]
    fn apriori(&self) -> f64 {
        self.inner.apriori
    }

    /// Coefficient of variation of each origin's prior; ``inf`` is flat.
    #[getter]
    fn cv(&self) -> f64 {
        self.inner.cv
    }

    /// Standard deviation of each development effect's prior.
    #[getter]
    fn development_sd(&self) -> f64 {
        self.inner.development_sd
    }

    /// Process error: ``"gamma"`` or ``"none"``.
    #[getter]
    fn process(&self) -> &'static str {
        process_name(self.inner.process)
    }

    /// Number of NUTS chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.sampler.chains
    }

    /// Warm-up draws per chain.
    #[getter]
    fn tune(&self) -> usize {
        self.inner.sampler.tune
    }

    /// Kept draws per chain.
    #[getter]
    fn draws(&self) -> usize {
        self.inner.sampler.draws
    }

    /// Number of predictive simulations.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims
    }

    /// Seed of the chains and of the simulations.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Target mean acceptance rate.
    #[getter]
    fn target_accept(&self) -> f64 {
        self.inner.sampler.target_accept
    }

    /// Largest NUTS tree depth.
    #[getter]
    fn max_depth(&self) -> u64 {
        self.inner.sampler.max_depth
    }

    /// Samples the posterior of one loss column of a single-segment
    /// cumulative triangle, with each origin's exposure from another
    /// column, and simulates its reserves. Every origin must be observed
    /// from the first age up to its latest and there must be more observed
    /// cells than chain-ladder parameters. Raises one ``UserWarning``
    /// listing the failed diagnostics, if any.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure, as for ``BornhuetterFerguson``.
    ///
    /// Returns
    /// -------
    /// BayesBfFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``OdpBootstrap.fit``; if the triangle has several segments; if
    ///     an origin has no observed, finite, positive exposure; if
    ///     ``apriori``, ``cv`` or ``development_sd`` is invalid or a sampler
    ///     setting is; if a development column's increments do not sum to
    ///     more than zero; or if an origin's sum to ``-phi / cv**2`` or
    ///     less.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyBayesBfFit> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.fit(tri, column, exposure))
            .map_err(err)?;
        let warnings = &inner.diagnostics.warnings;
        if !warnings.is_empty() {
            let message = format!(
                "BayesBf: the MCMC diagnostics failed; check them before relying on the fit: {}",
                warnings.join("; ")
            );
            let message = std::ffi::CString::new(message)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
            PyErr::warn(py, category.as_any(), &message, 1)?;
        }
        Ok(PyBayesBfFit { inner })
    }

    /// The one-year view of one loss column of a single-segment cumulative
    /// triangle, with each origin's exposure from another column: the
    /// claims development result over the coming year, by Pareto smoothed
    /// importance sampling of ``fit``'s posterior (decisions 16 and 17 of
    /// ``docs/design/reserving-bayes.md``), exactly as
    /// ``BayesOdp.one_year`` with this model's means
    /// ``mu[w] * gamma[d]``. Each simulation draws the year's increments
    /// from a posterior draw with ``process`` error, reweights every
    /// posterior draw by the year's quasi likelihood (``scale``, the
    /// priors and the prior ultimates unchanged), and takes each origin's
    /// claims development result as its posterior mean reserve at the
    /// opening less its payment in the year and its weighted posterior mean
    /// reserve after it. A simulation whose Pareto ``k`` is above
    /// ``k_threshold`` is handled as ``refit`` says: by default
    /// moment-matched, and refitted with NUTS on the extended triangle
    /// only if matching cannot bring its ``k`` to the threshold. A flat prior (``cv=inf``) gives about
    /// ``BayesOdp.one_year``'s view; a tighter prior a narrower one, in
    /// which the development pattern learns from the year and the origins'
    /// levels hardly do. Raises one ``UserWarning`` listing the failed
    /// diagnostics of the fit, and another if any refit failed its
    /// diagnostics.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column, as in ``fit``.
    /// k_threshold : float, default 0.7
    ///     The Pareto ``k`` above which a simulation's importance weights
    ///     are not trusted.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, default "moment_match_then_refit"
    ///     What happens to a flagged simulation, as in ``BayesOdp.one_year``.
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``.
    /// n_sims : int, optional
    ///     The number of one-year simulations; by default this model's
    ///     ``n_sims``. A smaller count gives the first simulations of a
    ///     larger one.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFit
    ///     With ``fit`` a ``BayesBfFit``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit``; if ``k_threshold`` is NaN or ``refit`` is unknown.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import BayesBf, Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    /// ...     {
    /// ...         "paid": [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    /// ...         "premium": [200.0] * 4 + [220.0] * 3 + [240.0] * 2 + [260.0],
    /// ...     },
    /// ... )
    /// >>> model = BayesBf(apriori=0.85, cv=0.1, n_sims=1000, seed=1)
    /// >>> view = model.one_year(tri, "paid", "premium", refit="reweight_only")
    /// >>> view.cdr.components()
    /// [('2020',), ('2021',), ('2022',), ('2023',)]
    /// >>> type(view.fit).__name__, len(view.pareto_k)
    /// ('BayesBfFit', 1000)
    /// >>> view.cdr.variance() < view.fit.reserves.variance()
    /// True
    #[pyo3(signature = (triangle, column, exposure, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFit> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year(tri, column, exposure, &settings))
            .map_err(err)?;
        warn_diagnostics(py, "BayesBf", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "BayesBf",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        Ok(PyBayesOneYearFit {
            inner: BayesOneYearOf::Bf(inner),
        })
    }

    fn __repr__(&self) -> String {
        let s = &self.inner.sampler;
        format!(
            "BayesBf(apriori={:?}, cv={:?}, development_sd={:?}, process={:?}, chains={}, \
             tune={}, draws={}, n_sims={}, seed={})",
            self.inner.apriori,
            self.inner.cv,
            self.inner.development_sd,
            self.process(),
            s.chains,
            s.tune,
            s.draws,
            self.inner.n_sims,
            self.inner.seed,
        )
    }
}

/// A fitted ``BayesBf``: the posterior of ``alpha[1..n]`` (``log mu[w]``)
/// and ``beta[1..m]`` (``log gamma[d]``), its MCMC diagnostics, and the
/// posterior predictive distribution of the reserves, laid out as
/// ``BayesOdpFit.reserves``. The frames' ``latest``, ``ultimate`` and
/// ``reserve`` are the volume-weighted chain ladder's (``chain_ladder``),
/// with each origin's ``exposure`` and ``prior_ultimate``; ``mean`` and
/// ``std_dev`` are the posterior predictive reserve's.
#[pyclass(name = "BayesBfFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesBfFit {
    inner: BayesBfFit,
}

#[pymethods]
impl PyBayesBfFit {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// Each simulation's reserve is spread over the pattern of its own
    /// posterior draw.
    ///
    /// Parameters
    /// ----------
    /// basis : {"paid", "incurred"}
    ///     What the fit's column is, stated by the caller (no default), so
    ///     that a fit on incurred claims is not discounted by its reporting
    ///     pattern by mistake. ``"paid"`` pays each simulation's reserve
    ///     along that simulation's own pattern. ``"incurred"`` pays each
    ///     reserve (the IBNR) plus the origin's case reserve (its latest
    ///     incurred less ``latest_paid``) along ``paid_pattern``.
    /// rate : float, optional
    ///     A flat annual effective rate. Give ``rate`` or ``curve``.
    /// curve : DiscountCurve, optional
    ///     Spot rates by maturity.
    /// timing : {"mid", "end"}, default "mid"
    ///     Payments at the middle or the end of each development period.
    /// paid_pattern : list of float, optional
    ///     For ``basis="incurred"``: the cumulative paid share at each of
    ///     the fit's pattern ages (with the tail's age when the fit has a
    ///     tail), ending at 1.
    /// latest_paid : list of float, optional
    ///     For ``basis="incurred"``: each origin's latest paid, one per
    ///     component of ``reserves`` in their order.
    ///
    /// Returns
    /// -------
    /// DiscountedReserves
    #[pyo3(signature = (
        *,
        basis,
        rate = None,
        curve = None,
        timing = "mid",
        paid_pattern = None,
        latest_paid = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn discounted(
        &self,
        py: Python<'_>,
        basis: &str,
        rate: Option<f64>,
        curve: Option<PyRef<'_, PyDiscountCurve>>,
        timing: &str,
        paid_pattern: Option<Vec<f64>>,
        latest_paid: Option<Vec<f64>>,
    ) -> PyResult<PyDiscountedReserves> {
        let args = DiscountArgs {
            basis,
            rate,
            curve,
            timing,
            paid_pattern,
            latest_paid,
        };
        discounted(py, &self.inner, args)
    }

    /// Each simulation's development pattern, to pair with the
    /// simulation's loss ratio in the cash-flow engine.
    ///
    /// Row ``i`` of every pattern is simulation ``i`` of ``reserves``,
    /// ``ultimates`` and ``ultimate_draws``.
    ///
    /// Returns
    /// -------
    /// DevelopmentPatterns
    #[getter]
    fn patterns(&self) -> PyDevelopmentPatterns {
        PyDevelopmentPatterns {
            inner: self.inner.patterns.clone(),
        }
    }
    /// The volume-weighted chain ladder of the triangle, whose latest
    /// values and Pearson scale the model uses.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.inner.chain_ladder.clone()],
            },
        }
    }

    /// Origin period of each reserve component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .chain_ladder
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.chain_ladder.development.development.clone()
    }

    /// The plugged-in scale ``phi``, the chain ladder's Pearson scale.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    /// Exposure per origin: the exposure column's latest observed value.
    #[getter]
    fn exposure(&self) -> Vec<f64> {
        self.inner.exposure.clone()
    }

    /// The prior mean of each origin's row parameter ``mu[w]``,
    /// ``apriori * exposure`` (EVW's prior ultimate). The model's expected
    /// ultimate is ``mu[w]`` times the sum of the ``gamma[d]``, not
    /// ``mu[w]``.
    #[getter]
    fn prior_ultimate(&self) -> Vec<f64> {
        self.inner.prior_ultimate.clone()
    }

    /// Joint posterior predictive distribution of the reserve (the sum of
    /// future incremental values) by origin: dimension ``"origin"``, one
    /// component per origin, one row per simulation.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// Joint distribution of the ultimate by origin: ``reserves`` with
    /// each origin's latest value added in every simulation, the same
    /// dimensions, components and rows, so ``join``, ``select`` and
    /// ``reorder_groups`` keep each simulation's origins together.
    #[getter]
    fn ultimates(&self) -> PyResult<PyPredictiveDistribution> {
        ultimates_of(&self.inner)
    }

    /// Each simulation's ultimate, for a tower, a deal book or the cash-flow
    /// engine: index ``i`` is simulation ``i`` for every origin and every
    /// call, the same as row ``i`` of ``reserves`` and ``ultimates``, and
    /// the same again when the fit is run with the same seed.
    ///
    /// The ultimate is the origin's latest value plus its reserve in that
    /// simulation.
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     The component key (``(2021,)``) or the origin alone; an origin period by its label
    ///     (``"2021"``, ``2021``, ``"2021Q3"``). ``None`` (the default) gives
    ///     the total over every component, summed simulation by simulation.
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (origin = None))]
    fn ultimate_draws(&self, origin: Option<OriginArg>) -> PyResult<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// Each simulation's ultimate over ``premium``: the loss ratio a
    /// contract on the loss ratio reads, in the order of
    /// ``ultimate_draws``. Multiply by the premium for
    /// ``Tower.apply_annual``.
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     As in ``ultimate_draws``; ``None`` for the total.
    /// premium : float
    ///     The subject premium of the origin (of every origin for the
    ///     total), finite and positive.
    ///
    /// Returns
    /// -------
    /// list of float
    fn loss_ratio_draws(&self, origin: Option<OriginArg>, premium: f64) -> PyResult<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// MCMC diagnostics: a dict with ``max_rhat``, ``min_ess_bulk``,
    /// ``min_ess_tail``, ``divergences`` and ``warnings`` (one message per
    /// failed threshold; empty when every threshold holds).
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = &self.inner.diagnostics;
        let out = PyDict::new(py);
        out.set_item("max_rhat", d.max_rhat)?;
        out.set_item("min_ess_bulk", d.min_ess_bulk)?;
        out.set_item("min_ess_tail", d.min_ess_tail)?;
        out.set_item("divergences", d.divergences)?;
        out.set_item("warnings", d.warnings.clone())?;
        Ok(out)
    }

    /// Parameter names: ``alpha[1]`` to ``alpha[n]``, ``beta[1]`` to
    /// ``beta[m]``.
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, one row per draw (chain by chain), one column per
    /// parameter in the order of ``parameter_names``.
    #[getter]
    fn draws(&self) -> Vec<Vec<f64>> {
        let p = self.inner.posterior.dim().max(1);
        self.inner
            .posterior
            .values()
            .chunks(p)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// Number of chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.posterior.chains()
    }

    /// Posterior summary, one row per parameter: ``parameter``, ``mean``,
    /// ``sd``, ``q05``, ``q50``, ``q95``, ``rhat``, ``ess_bulk`` and
    /// ``ess_tail``, as ``BayesOdpFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fit = &self.inner;
        let s = py.detach(|| fit.summary()).map_err(err)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        let out = PyDict::new(py);
        out.set_item(
            "parameter",
            s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
        )?;
        out.set_item("mean", col(|p| p.mean))?;
        out.set_item("sd", col(|p| p.sd))?;
        out.set_item("q05", col(|p| p.q05))?;
        out.set_item("q50", col(|p| p.q50))?;
        out.set_item("q95", col(|p| p.q95))?;
        out.set_item("rhat", col(|p| p.rhat))?;
        out.set_item("ess_bulk", col(|p| p.ess_bulk))?;
        out.set_item("ess_tail", col(|p| p.ess_tail))?;
        py.import("pandas")?.call_method1("DataFrame", (out,))
    }

    /// PSIS-LOO of the observed increments, as ``BayesOdpFit.loo``: the
    /// pointwise term is the normalised over-dispersed Poisson density at
    /// the cell's mean and ``scale``, so the result compares with
    /// ``BayesOdp``'s on the same column.
    ///
    /// Returns
    /// -------
    /// Elpd
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an observed increment is negative (the density does not
    ///     exist there).
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let fit = &self.inner;
        let l = py.detach(|| fit.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l))
    }

    /// One row per origin: ``origin``, the chain ladder's ``latest``,
    /// ``ultimate`` and ``reserve``, the ``exposure``, ``prior_ultimate``
    /// and ``ulr`` (the chain ladder's ultimate over the exposure, a loss
    /// ratio only when the exposure is premium), the ``mean`` and
    /// ``std_dev`` of the posterior predictive reserve, ``ultimate_mean``
    /// (``latest`` plus ``mean``, the posterior mean ultimate) and
    /// ``ulr_mean`` (``ultimate_mean`` over the exposure). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the chain ladder's totals, the total ``exposure`` and
    /// ``prior_ultimate``, the ``scale``, the total ``ulr``, the ``mean``
    /// and ``std_dev`` of the posterior predictive total reserve, and the
    /// total ``ultimate_mean`` and ``ulr_mean`` (over the total exposure).
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The chain ladder's development factors, one row per age, as
    /// ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesBfFit(origins={}, draws={}, n_sims={}, scale={:?}, warnings={})",
            self.inner.chain_ladder.origins.len(),
            self.inner.posterior.n_draws(),
            self.inner.reserves.n_sims(),
            self.inner.scale,
            self.inner.diagnostics.warnings.len(),
        )
    }
}

/// The Bayesian Bornhuetter–Ferguson whose prior sits on the ultimate: the
/// over-dispersed Poisson model of a cumulative triangle's incremental
/// values ``X`` with mean ``U[w] * p[d]``, the origin's expected ultimate
/// ``U[w]`` times the share ``p[d]`` of it that falls at age ``d``, the
/// shares summing to 1 (``p[d] = exp(beta[d]) / sum(exp(beta))`` with
/// ``beta[1] = 0``). The informative gamma prior is on ``U[w]`` itself,
/// with mean ``apriori * exposure[w]`` and coefficient of variation
/// ``cv``, so ``apriori`` is the expected ultimate loss ratio and ``cv``
/// says how firmly it is held, as in the classical
/// ``BornhuetterFerguson``. The shares have a vague normal prior on their
/// log ratios to the first age's, and the chain ladder's Pearson scale
/// ``phi`` (``OdpBootstrapFit.scale``) is plugged in. The posterior is
/// sampled with NUTS, and each draw projects the future cells up to the
/// oldest observed age (no tail) with the process error of ``process``, as
/// ``BayesOdp`` does.
///
/// ``BayesBf`` (England, Verrall and Wüthrich's Model 3.1) instead
/// leaves the pattern unnormalised and puts its prior on ``mu[w]``, which
/// is not the ultimate: there the level of ``apriori`` is cancelled by the
/// pattern and barely moves the reserve. Use this model when the
/// a-priori loss ratio is a judgment about the ultimate; use ``BayesBf``
/// to match England, Verrall and Wüthrich's published results.
///
/// The prior's strength moves the reserve between two limits. With
/// ``cv=float("inf")`` (a flat prior on ``log U[w]``) the model is the
/// Bayesian chain ladder, and the reserves agree with ``BayesOdp``'s within
/// Monte Carlo error. As ``cv`` falls to 0, ``U[w]`` is fixed at the prior
/// ultimate ``m[w]`` and the shares are ``S[d] / (M[d] + lambda)`` (``S[d]``
/// the observed increments of column ``d``, ``M[d]`` the prior ultimates of
/// the origins observed there, ``lambda`` such that they sum to 1). When
/// the prior ultimates are the chain ladder's, ``lambda`` is 0 and the
/// reserve is exactly the classical ``BornhuetterFerguson`` reserve
/// ``(1 - 1 / cdf) * m[w]``; otherwise the shares move with the ultimates,
/// where the classical method keeps the chain ladder's pattern.
///
/// Every development column's observed increments must sum to more than
/// zero, and every origin's to more than ``-phi / cv**2``; negative cells
/// are allowed. The error names the age or origin, its number of observed
/// cells, their sum and the remedies. A column that sums to zero or less
/// has no log-linear mean, since every mean is an ultimate times a
/// positive share; the chain ladder accepts the same triangle, and
/// ``BayesOdp`` and ``BayesBf`` refuse it too. To drop a sparse last age,
/// keep the ages before it with ``Triangle.max_age`` (``tri.max_age(108)``
/// for a last age of 120 months) and carry the movement left out as a tail
/// outside the model. A falling first age is dropped by rebuilding the
/// triangle without its rows (``Triangle.from_frame`` on the rows of
/// ``tri.to_frame()`` at the other ages); a falling middle age has no
/// remedy in this model. For an origin, a finite ``cv`` below the bound
/// the error prints makes its posterior proper. One ``seed`` replays a fit
/// exactly, whatever the number of threads. Failed MCMC diagnostics (R-hat
/// above 1.01, bulk or tail ESS below 100 per chain, any divergent
/// transition) do not fail the fit: ``fit`` raises one ``UserWarning``
/// listing them, and they stay in ``BayesBfNormalisedFit.diagnostics``.
///
/// Parameters
/// ----------
/// apriori : float, default 1.0
///     Prior mean of each origin's expected ultimate ``U[w]`` per unit of
///     exposure, the expected ultimate loss ratio; positive and finite.
/// cv : float, default 0.1
///     Coefficient of variation of the gamma prior on each ``U[w]`` (shape
///     ``1 / cv**2``); positive, ``float("inf")`` for a flat prior.
/// development_sd : float, default 10.0
///     Standard deviation of the normal prior on each ``beta[d]`` (the log
///     ratio of the share at age ``d`` to the first age's), centred on 0.
/// process : {"gamma", "none"}, default "gamma"
///     Process error on each simulated future increment: Gamma with mean
///     ``mu`` and variance ``phi * mu``, or none for parameter error only.
/// chains, tune, draws : int, default 4, 1000, 1000
///     NUTS chains, warm-up draws per chain and kept draws per chain.
/// n_sims : int, default 10000
///     Number of predictive simulations, independent of the number of
///     posterior draws; positive.
/// seed : int, default 0
///     Seed of the chains and of the simulations, from 0 to ``2**64 - 1``.
/// target_accept : float, default 0.95
///     Target mean acceptance rate of the step-size adaptation.
/// max_depth : int, default 10
///     Largest NUTS tree depth.
/// common_cv : float, optional
///     The coefficient of variation of one gamma multiplier (mean 1) common
///     to all origins' prior ultimates; ``None`` has none.
/// pattern_prior : PatternPrior, optional
///     An informative prior on the pattern: each ``beta[d]`` is normal with
///     the prior's mean and sd in place of ``N(0, development_sd**2)``, for
///     example ``IndustryPrior.pattern``. Its ages must be the fitted
///     triangle's (``PatternPrior.truncated`` or ``IndustryPrior.at_age``
///     cut a longer one). ``None`` (the default) keeps the vague prior.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero or ``process`` is unknown.
/// OverflowError
///     If an integer setting is negative or too large.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import BayesBfNormalised, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     {
/// ...         "paid": [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ...         "premium": [200.0] * 4 + [220.0] * 3 + [240.0] * 2 + [260.0],
/// ...     },
/// ... )
/// >>> fit = BayesBfNormalised(apriori=0.85, cv=0.1, n_sims=2000, seed=1).fit(tri, "paid", "premium")
/// >>> fit.reserves.components()
/// [('2020',), ('2021',), ('2022',), ('2023',)]
/// >>> fit.prior_ultimate[3] == 0.85 * 260.0
/// True
/// >>> fit.parameter_names[:2]
/// ['alpha[1]', 'alpha[2]']
#[pyclass(name = "BayesBfNormalised", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesBfNormalised {
    pub(crate) inner: BayesBfNormalised,
    /// The pattern prior its fits take, if any (not a setting of the Rust
    /// model, which takes it at the fit: decision 29).
    pub(crate) pattern_prior: Option<PatternPrior>,
}

#[pymethods]
impl PyBayesBfNormalised {
    #[new]
    #[pyo3(signature = (
        apriori = 1.0,
        cv = 0.1,
        development_sd = 10.0,
        process = "gamma",
        chains = 4,
        tune = 1000,
        draws = 1000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.95,
        max_depth = 10,
        common_cv = None,
        pattern_prior = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        apriori: f64,
        cv: f64,
        development_sd: f64,
        process: &str,
        chains: usize,
        tune: usize,
        draws: usize,
        n_sims: usize,
        seed: u64,
        target_accept: f64,
        max_depth: u64,
        common_cv: Option<f64>,
        pattern_prior: Option<PyRef<'_, PyPatternPrior>>,
    ) -> PyResult<Self> {
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            inner: BayesBfNormalised {
                apriori,
                cv,
                development_sd,
                process: process_distribution(process)?,
                sampler: BayesSampler {
                    chains,
                    tune,
                    draws,
                    seed: 0,
                    target_accept,
                    max_depth,
                },
                n_sims,
                seed,
                common_cv,
            },
            pattern_prior: pattern_prior.map(|p| p.inner.clone()),
        })
    }

    /// The coefficient of variation of the multiplier common to all
    /// origins' prior ultimates, or ``None`` if there is none.
    #[getter]
    fn common_cv(&self) -> Option<f64> {
        self.inner.common_cv
    }

    /// The prior on the pattern, or ``None`` for the vague
    /// ``N(0, development_sd**2)``.
    #[getter]
    fn pattern_prior(&self) -> Option<PyPatternPrior> {
        self.pattern_prior
            .clone()
            .map(|inner| PyPatternPrior { inner })
    }

    /// Prior mean of each origin's expected ultimate per unit of exposure.
    #[getter]
    fn apriori(&self) -> f64 {
        self.inner.apriori
    }

    /// Coefficient of variation of each origin's prior; ``inf`` is flat.
    #[getter]
    fn cv(&self) -> f64 {
        self.inner.cv
    }

    /// Standard deviation of each development effect's prior.
    #[getter]
    fn development_sd(&self) -> f64 {
        self.inner.development_sd
    }

    /// Process error: ``"gamma"`` or ``"none"``.
    #[getter]
    fn process(&self) -> &'static str {
        process_name(self.inner.process)
    }

    /// Number of NUTS chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.sampler.chains
    }

    /// Warm-up draws per chain.
    #[getter]
    fn tune(&self) -> usize {
        self.inner.sampler.tune
    }

    /// Kept draws per chain.
    #[getter]
    fn draws(&self) -> usize {
        self.inner.sampler.draws
    }

    /// Number of predictive simulations.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims
    }

    /// Seed of the chains and of the simulations.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Target mean acceptance rate.
    #[getter]
    fn target_accept(&self) -> f64 {
        self.inner.sampler.target_accept
    }

    /// Largest NUTS tree depth.
    #[getter]
    fn max_depth(&self) -> u64 {
        self.inner.sampler.max_depth
    }

    /// Samples the posterior of one loss column of a single-segment
    /// cumulative triangle, with each origin's exposure from another
    /// column, and simulates its reserves. Every origin must be observed
    /// from the first age up to its latest and there must be more observed
    /// cells than chain-ladder parameters. Raises one ``UserWarning``
    /// listing the failed diagnostics, if any.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure, as for ``BornhuetterFerguson``.
    ///
    /// Returns
    /// -------
    /// BayesBfNormalisedFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``OdpBootstrap.fit``; if the triangle has several segments; if
    ///     an origin has no observed, finite, positive exposure; if
    ///     ``apriori``, ``cv`` or ``development_sd`` is invalid or a sampler
    ///     setting is; if a development column's increments do not sum to
    ///     more than zero; or if an origin's sum to ``-phi / cv**2`` or
    ///     less.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyBayesBfNormalisedFit> {
        let (model, tri, prior) = (&self.inner, &triangle.inner, &self.pattern_prior);
        let inner = py
            .detach(|| match prior {
                Some(p) => model.fit_with_pattern_prior(tri, column, exposure, p),
                None => model.fit(tri, column, exposure),
            })
            .map_err(err)?;
        let warnings = &inner.diagnostics.warnings;
        if !warnings.is_empty() {
            let message = format!(
                "BayesBfNormalised: the MCMC diagnostics failed; check them before relying on the fit: {}",
                warnings.join("; ")
            );
            let message = std::ffi::CString::new(message)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
            PyErr::warn(py, category.as_any(), &message, 1)?;
        }
        Ok(PyBayesBfNormalisedFit { inner })
    }

    /// The one-year view of one loss column of a single-segment cumulative
    /// triangle, with each origin's exposure from another column: the
    /// claims development result over the coming year, by Pareto smoothed
    /// importance sampling of ``fit``'s posterior (decisions 16 and 17 of
    /// ``docs/design/reserving-bayes.md``), exactly as
    /// ``BayesOdp.one_year`` with this model's means
    /// ``exp(alpha[w]) * p[d]``. Each simulation draws the year's increments
    /// from a posterior draw with ``process`` error, reweights every
    /// posterior draw by the year's quasi likelihood (``scale``, the
    /// priors and the prior ultimates unchanged), and takes each origin's
    /// claims development result as its posterior mean reserve at the
    /// opening less its payment in the year and its weighted posterior mean
    /// reserve after it. A simulation whose Pareto ``k`` is above
    /// ``k_threshold`` is handled as ``refit`` says: by default
    /// moment-matched, and refitted with NUTS on the extended triangle
    /// only if matching cannot bring its ``k`` to the threshold. A flat
    /// prior (``cv=inf``) gives about ``BayesOdp.one_year``'s view; a
    /// tighter prior holds each origin's ultimate more firmly and gives a
    /// narrower one. Raises one ``UserWarning`` listing the failed
    /// diagnostics of the fit, and another if any refit failed its
    /// diagnostics.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column, as in ``fit``.
    /// k_threshold : float, default 0.7
    ///     The Pareto ``k`` above which a simulation's importance weights
    ///     are not trusted.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, default "moment_match_then_refit"
    ///     What happens to a flagged simulation, as in ``BayesOdp.one_year``.
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``.
    /// n_sims : int, optional
    ///     The number of one-year simulations; by default this model's
    ///     ``n_sims``. A smaller count gives the first simulations of a
    ///     larger one.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFit
    ///     With ``fit`` a ``BayesBfNormalisedFit``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit``; if ``k_threshold`` is NaN or ``refit`` is unknown.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import BayesBfNormalised, Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    /// ...     {
    /// ...         "paid": [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    /// ...         "premium": [200.0] * 4 + [220.0] * 3 + [240.0] * 2 + [260.0],
    /// ...     },
    /// ... )
    /// >>> model = BayesBfNormalised(apriori=0.85, cv=0.1, n_sims=1000, seed=1)
    /// >>> view = model.one_year(tri, "paid", "premium", refit="reweight_only")
    /// >>> view.cdr.components()
    /// [('2020',), ('2021',), ('2022',), ('2023',)]
    /// >>> type(view.fit).__name__, len(view.pareto_k)
    /// ('BayesBfNormalisedFit', 1000)
    /// >>> view.cdr.variance() < view.fit.reserves.variance()
    /// True
    #[pyo3(signature = (triangle, column, exposure, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFit> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let (model, tri, prior) = (&self.inner, &triangle.inner, &self.pattern_prior);
        let inner = py
            .detach(|| match prior {
                Some(p) => model.one_year_with_pattern_prior(tri, column, exposure, &settings, p),
                None => model.one_year(tri, column, exposure, &settings),
            })
            .map_err(err)?;
        warn_diagnostics(py, "BayesBfNormalised", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "BayesBfNormalised",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        Ok(PyBayesOneYearFit {
            inner: BayesOneYearOf::BfNormalised(inner),
        })
    }

    fn __repr__(&self) -> String {
        let s = &self.inner.sampler;
        format!(
            "BayesBfNormalised(apriori={:?}, cv={:?}, development_sd={:?}, process={:?}, chains={}, \
             tune={}, draws={}, n_sims={}, seed={}{}{})",
            self.inner.apriori,
            self.inner.cv,
            self.inner.development_sd,
            self.process(),
            s.chains,
            s.tune,
            s.draws,
            self.inner.n_sims,
            self.inner.seed,
            self.inner
                .common_cv
                .map_or_else(String::new, |c| format!(", common_cv={c:?}")),
            if self.pattern_prior.is_some() {
                ", pattern_prior=PatternPrior(...)"
            } else {
                ""
            },
        )
    }
}

/// A fitted ``BayesBfNormalised``: the posterior of ``alpha[1..n]`` (the log
/// expected ultimates) and ``beta[2..m]`` (the log ratios of the shares to
/// the first age's), its MCMC diagnostics, and the
/// posterior predictive distribution of the reserves, laid out as
/// ``BayesOdpFit.reserves``. The frames' ``latest``, ``ultimate`` and
/// ``reserve`` are the volume-weighted chain ladder's (``chain_ladder``),
/// with each origin's ``exposure`` and ``prior_ultimate``; ``mean`` and
/// ``std_dev`` are the posterior predictive reserve's.
#[pyclass(name = "BayesBfNormalisedFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesBfNormalisedFit {
    inner: BayesBfNormalisedFit,
}

#[pymethods]
impl PyBayesBfNormalisedFit {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// Each simulation's reserve is spread over the pattern of its own
    /// posterior draw.
    ///
    /// Parameters
    /// ----------
    /// basis : {"paid", "incurred"}
    ///     What the fit's column is, stated by the caller (no default), so
    ///     that a fit on incurred claims is not discounted by its reporting
    ///     pattern by mistake. ``"paid"`` pays each simulation's reserve
    ///     along that simulation's own pattern. ``"incurred"`` pays each
    ///     reserve (the IBNR) plus the origin's case reserve (its latest
    ///     incurred less ``latest_paid``) along ``paid_pattern``.
    /// rate : float, optional
    ///     A flat annual effective rate. Give ``rate`` or ``curve``.
    /// curve : DiscountCurve, optional
    ///     Spot rates by maturity.
    /// timing : {"mid", "end"}, default "mid"
    ///     Payments at the middle or the end of each development period.
    /// paid_pattern : list of float, optional
    ///     For ``basis="incurred"``: the cumulative paid share at each of
    ///     the fit's pattern ages (with the tail's age when the fit has a
    ///     tail), ending at 1.
    /// latest_paid : list of float, optional
    ///     For ``basis="incurred"``: each origin's latest paid, one per
    ///     component of ``reserves`` in their order.
    ///
    /// Returns
    /// -------
    /// DiscountedReserves
    #[pyo3(signature = (
        *,
        basis,
        rate = None,
        curve = None,
        timing = "mid",
        paid_pattern = None,
        latest_paid = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn discounted(
        &self,
        py: Python<'_>,
        basis: &str,
        rate: Option<f64>,
        curve: Option<PyRef<'_, PyDiscountCurve>>,
        timing: &str,
        paid_pattern: Option<Vec<f64>>,
        latest_paid: Option<Vec<f64>>,
    ) -> PyResult<PyDiscountedReserves> {
        let args = DiscountArgs {
            basis,
            rate,
            curve,
            timing,
            paid_pattern,
            latest_paid,
        };
        discounted(py, &self.inner, args)
    }

    /// The posterior predictive of a new underwriting year, the origin
    /// after the triangle's newest, for pricing a year not yet written.
    ///
    /// Each simulation takes a posterior draw and from it the new origin's
    /// expected ultimate, a gamma with cv ``cv`` and mean ``s * apriori *
    /// premium`` (``s`` the draw's common multiplier, 1 without
    /// ``common_cv``), then its ultimate with the fit's process error. The
    /// data reach the new year only through ``s``: without ``common_cv``
    /// the new year is the prior, ``Gamma(1 / cv**2, apriori)`` in loss
    /// ratio, whatever the triangle says; with a vague ``common_cv`` (say
    /// 10) the triangle sets its level.
    ///
    /// Parameters
    /// ----------
    /// premium : float
    ///     The new year's premium (its exposure), finite and positive.
    /// n_sims : int
    ///     Number of simulations.
    /// seed : int
    ///     Seed of the simulations.
    ///
    /// Returns
    /// -------
    /// NewYear
    ///     ``loss_ratio`` and ``ultimate`` by simulation, and ``patterns``,
    ///     the development pattern of each simulation's posterior draw, whose
    ///     ``monthly`` is the new year's emergence by month.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``premium`` is not positive, ``n_sims`` is 0, or the fit's
    ///     ``cv`` is infinite (a flat prior gives a new origin no proper
    ///     predictive).
    #[pyo3(signature = (premium, n_sims = 10_000, seed = 0))]
    fn new_year(
        &self,
        py: Python<'_>,
        premium: f64,
        n_sims: usize,
        seed: u64,
    ) -> PyResult<PyNewYear> {
        let fit = &self.inner;
        let inner = py
            .detach(|| fit.new_year(premium, n_sims, seed))
            .map_err(err)?;
        Ok(PyNewYear { inner })
    }

    /// Each simulation's development pattern, to pair with the
    /// simulation's loss ratio in the cash-flow engine.
    ///
    /// Row ``i`` of every pattern is simulation ``i`` of ``reserves``,
    /// ``ultimates`` and ``ultimate_draws``.
    ///
    /// Returns
    /// -------
    /// DevelopmentPatterns
    #[getter]
    fn patterns(&self) -> PyDevelopmentPatterns {
        PyDevelopmentPatterns {
            inner: self.inner.patterns.clone(),
        }
    }
    /// The volume-weighted chain ladder of the triangle, whose latest
    /// values and Pearson scale the model uses.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.inner.chain_ladder.clone()],
            },
        }
    }

    /// Origin period of each reserve component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .chain_ladder
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.chain_ladder.development.development.clone()
    }

    /// The plugged-in scale ``phi``, the chain ladder's Pearson scale.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    /// Exposure per origin: the exposure column's latest observed value.
    #[getter]
    fn exposure(&self) -> Vec<f64> {
        self.inner.exposure.clone()
    }

    /// The prior mean of each origin's expected ultimate,
    /// ``apriori * exposure``.
    #[getter]
    fn prior_ultimate(&self) -> Vec<f64> {
        self.inner.prior_ultimate.clone()
    }

    /// The posterior mean of each age's share ``p[d]`` of the ultimate, in
    /// development order; the shares sum to 1.
    #[getter]
    fn pattern(&self) -> Vec<f64> {
        self.inner.pattern()
    }

    /// The posterior mean of each origin's expected ultimate
    /// ``exp(alpha[w])``, the quantity that ``apriori`` and ``cv`` put the
    /// prior on (the reserve's process error and the cells already paid are
    /// not in it).
    #[getter]
    fn expected_ultimate(&self) -> Vec<f64> {
        self.inner.expected_ultimate()
    }

    /// Joint posterior predictive distribution of the reserve (the sum of
    /// future incremental values) by origin: dimension ``"origin"``, one
    /// component per origin, one row per simulation.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// Joint distribution of the ultimate by origin: ``reserves`` with
    /// each origin's latest value added in every simulation, the same
    /// dimensions, components and rows, so ``join``, ``select`` and
    /// ``reorder_groups`` keep each simulation's origins together.
    #[getter]
    fn ultimates(&self) -> PyResult<PyPredictiveDistribution> {
        ultimates_of(&self.inner)
    }

    /// Each simulation's ultimate, for a tower, a deal book or the cash-flow
    /// engine: index ``i`` is simulation ``i`` for every origin and every
    /// call, the same as row ``i`` of ``reserves`` and ``ultimates``, and
    /// the same again when the fit is run with the same seed.
    ///
    /// The ultimate is the origin's latest value plus its reserve in that
    /// simulation.
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     The component key (``(2021,)``) or the origin alone; an origin period by its label
    ///     (``"2021"``, ``2021``, ``"2021Q3"``). ``None`` (the default) gives
    ///     the total over every component, summed simulation by simulation.
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (origin = None))]
    fn ultimate_draws(&self, origin: Option<OriginArg>) -> PyResult<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// Each simulation's ultimate over ``premium``: the loss ratio a
    /// contract on the loss ratio reads, in the order of
    /// ``ultimate_draws``. Multiply by the premium for
    /// ``Tower.apply_annual``.
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     As in ``ultimate_draws``; ``None`` for the total.
    /// premium : float
    ///     The subject premium of the origin (of every origin for the
    ///     total), finite and positive.
    ///
    /// Returns
    /// -------
    /// list of float
    fn loss_ratio_draws(&self, origin: Option<OriginArg>, premium: f64) -> PyResult<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// MCMC diagnostics: a dict with ``max_rhat``, ``min_ess_bulk``,
    /// ``min_ess_tail``, ``divergences`` and ``warnings`` (one message per
    /// failed threshold; empty when every threshold holds).
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = &self.inner.diagnostics;
        let out = PyDict::new(py);
        out.set_item("max_rhat", d.max_rhat)?;
        out.set_item("min_ess_bulk", d.min_ess_bulk)?;
        out.set_item("min_ess_tail", d.min_ess_tail)?;
        out.set_item("divergences", d.divergences)?;
        out.set_item("warnings", d.warnings.clone())?;
        Ok(out)
    }

    /// Parameter names: ``alpha[1]`` to ``alpha[n]``, ``beta[2]`` to
    /// ``beta[m]`` (the first age is the corner, ``beta[1] = 0``).
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, one row per draw (chain by chain), one column per
    /// parameter in the order of ``parameter_names``.
    #[getter]
    fn draws(&self) -> Vec<Vec<f64>> {
        let p = self.inner.posterior.dim().max(1);
        self.inner
            .posterior
            .values()
            .chunks(p)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// Number of chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.posterior.chains()
    }

    /// Posterior summary, one row per parameter: ``parameter``, ``mean``,
    /// ``sd``, ``q05``, ``q50``, ``q95``, ``rhat``, ``ess_bulk`` and
    /// ``ess_tail``, as ``BayesOdpFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fit = &self.inner;
        let s = py.detach(|| fit.summary()).map_err(err)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        let out = PyDict::new(py);
        out.set_item(
            "parameter",
            s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
        )?;
        out.set_item("mean", col(|p| p.mean))?;
        out.set_item("sd", col(|p| p.sd))?;
        out.set_item("q05", col(|p| p.q05))?;
        out.set_item("q50", col(|p| p.q50))?;
        out.set_item("q95", col(|p| p.q95))?;
        out.set_item("rhat", col(|p| p.rhat))?;
        out.set_item("ess_bulk", col(|p| p.ess_bulk))?;
        out.set_item("ess_tail", col(|p| p.ess_tail))?;
        py.import("pandas")?.call_method1("DataFrame", (out,))
    }

    /// PSIS-LOO of the observed increments, as ``BayesOdpFit.loo``: the
    /// pointwise term is the normalised over-dispersed Poisson density at
    /// the cell's mean and ``scale``, so the result compares with
    /// ``BayesOdp``'s on the same column.
    ///
    /// Returns
    /// -------
    /// Elpd
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an observed increment is negative (the density does not
    ///     exist there).
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let fit = &self.inner;
        let l = py.detach(|| fit.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l))
    }

    /// One row per origin: ``origin``, the chain ladder's ``latest``,
    /// ``ultimate`` and ``reserve``, the ``exposure``, ``prior_ultimate``
    /// and ``ulr`` (the chain ladder's ultimate over the exposure, a loss
    /// ratio only when the exposure is premium), the ``mean`` and
    /// ``std_dev`` of the posterior predictive reserve, ``ultimate_mean``
    /// (``latest`` plus ``mean``, the posterior mean ultimate) and
    /// ``ulr_mean`` (``ultimate_mean`` over the exposure). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the chain ladder's totals, the total ``exposure`` and
    /// ``prior_ultimate``, the ``scale``, the total ``ulr``, the ``mean``
    /// and ``std_dev`` of the posterior predictive total reserve, and the
    /// total ``ultimate_mean`` and ``ulr_mean`` (over the total exposure).
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The chain ladder's development factors, one row per age, as
    /// ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    fn __repr__(&self) -> String {
        format!(
            "BayesBfNormalisedFit(origins={}, draws={}, n_sims={}, scale={:?}, warnings={})",
            self.inner.chain_ladder.origins.len(),
            self.inner.posterior.n_draws(),
            self.inner.reserves.n_sims(),
            self.inner.scale,
            self.inner.diagnostics.warnings.len(),
        )
    }
}

/// The programs of a dict of name to one-segment `Triangle`, stacked into
/// one triangle with a segment per program under the key `program`, through
/// each triangle's long table (`HierarchicalBf.fit`).
fn stack_programs(programs: &Bound<'_, PyDict>, columns: &[&str]) -> PyResult<Triangle> {
    if programs.is_empty() {
        return Err(PyValueError::new_err("no programs: the dict is empty"));
    }
    let mut named: Vec<(String, PyRef<'_, PyTriangle>)> = Vec::with_capacity(programs.len());
    for (key, value) in programs.iter() {
        let name: String = key
            .extract()
            .map_err(|_| PyTypeError::new_err("program names must be strings"))?;
        let t = value
            .extract::<PyRef<'_, PyTriangle>>()
            .map_err(|_| PyTypeError::new_err(format!("program {name:?} is not a Triangle")))?;
        named.push((name, t));
    }
    // The core's rules (#838), the same in R.
    let refs: Vec<(&str, &Triangle)> = named.iter().map(|(n, t)| (n.as_str(), &t.inner)).collect();
    Triangle::stack(&refs, "program", columns).map_err(err)
}

/// The triangle of `HierarchicalBf.fit` and `.one_year`: a `Triangle` with
/// a segment per program, or a dict of program name to one-segment
/// `Triangle`, stacked.
fn programs_triangle(
    triangle: &Bound<'_, PyAny>,
    column: &str,
    exposure: &str,
) -> PyResult<Triangle> {
    if let Ok(programs) = triangle.cast::<PyDict>() {
        stack_programs(programs, &[column, exposure])
    } else if let Ok(t) = triangle.extract::<PyRef<'_, PyTriangle>>() {
        Ok(t.inner.clone())
    } else {
        Err(PyTypeError::new_err(
            "triangle must be a Triangle or a dict of program name to Triangle",
        ))
    }
}

/// The hierarchical Bayesian Bornhuetter-Ferguson: several programs, one
/// segment of a triangle each, pooled under an industry hyperprior on the
/// loss ratio, with the development pattern partially pooled
/// (``docs/design/reserving-bayes.md``, decision 30).
///
/// Each origin's cumulative-value increments are the over-dispersed Poisson
/// of ``BayesBfNormalised`` with mean ``U[c, w] * p[c, d]``, the origin's
/// expected ultimate times the share of it that falls at age ``d``. The
/// industry loss ratio ``L`` has a lognormal prior with mean ``apriori`` and
/// coefficient of variation ``level_cv`` (flat on ``log L`` when
/// ``level_cv`` is ``None``, so the programs alone set the level). Program
/// ``c`` has loss ratio ``L * s[c]`` with ``log s[c]`` normal with sd
/// ``tau`` and ``E[s[c]] = 1``, ``tau`` half-normal with scale
/// ``tau_scale``; its origins' expected ultimates are gamma with mean
/// ``L * s[c] * exposure`` and coefficient of variation ``cv``. The shared
/// log pattern ``beta[d]`` is normal with sd ``development_sd`` (or the
/// ``pattern_prior``), and each program's own pattern departs from it by
/// ``omega`` times a standard normal per age, ``omega`` half-normal with
/// scale ``omega_scale``. A thin program's young ages therefore move with
/// its own data, its later ages follow the shared pattern, and its loss
/// ratio is shrunk towards ``L`` by as much as the spread between the
/// programs supports.
///
/// The scale is plugged in per unit of premium, pooled over the programs
/// (``scale_per_premium``), so a program of two origins, which has no
/// degree of freedom of its own, still has a scale. The panel together must
/// have a degree of freedom. Failed MCMC diagnostics (R-hat above 1.01,
/// bulk or tail ESS below 100 per chain, any divergent transition) do not
/// fail the fit: ``fit`` raises one ``UserWarning`` listing them, and they
/// stay in ``HierarchicalBfFit.diagnostics``. One ``seed`` replays a fit
/// exactly. The one-year view is not available yet.
///
/// Parameters
/// ----------
/// apriori : float, default 1.0
///     Prior mean of the industry loss ratio ``L``; positive and finite.
/// level_cv : float, optional
///     Coefficient of variation of the lognormal prior on ``L``; ``None``
///     (the default) is flat on ``log L``.
/// cv : float, default 0.1
///     Coefficient of variation of each origin's expected ultimate about
///     its program's level; fixed, finite and positive.
/// development_sd : float, default 10.0
///     Standard deviation of the normal prior on each shared ``beta[d]``;
///     unused with a ``pattern_prior``.
/// tau_scale : float, default 0.3
///     Scale of the half-normal prior on ``tau``, the spread of the
///     programs' log loss ratios (a median spread of about 20%).
/// omega_scale : float, default 0.5
///     Scale of the half-normal prior on ``omega``, the spread of the
///     programs' log share ratios.
/// process : {"gamma", "none"}, default "gamma"
///     Process error on each simulated future increment.
/// chains, tune, draws : int, default 4, 1000, 1000
///     NUTS chains, warm-up draws per chain and kept draws per chain.
/// n_sims : int, default 10000
///     Number of predictive simulations; positive.
/// seed : int, default 0
///     Seed of the chains and of the simulations.
/// target_accept : float, default 0.99
///     Target mean acceptance rate of the step-size adaptation.
/// max_depth : int, default 10
///     Largest NUTS tree depth.
/// pattern_prior : PatternPrior, optional
///     An informative prior on the shared pattern, for example
///     ``IndustryPrior.pattern``; its ages must be the programs' ages.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero or ``process`` is unknown. ``fit`` raises it
///     when falling columns reach ``1 / omega_scale`` in the programs'
///     scales, jointly over the programs, or when a program's falling
///     columns exceed its first age: such a panel has no posterior.
/// OverflowError
///     If an integer setting is negative or too large.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import HierarchicalBf, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2021, 2021, 2021, 2022, 2022, 2023, 2022, 2022, 2023],
/// ...     [12, 24, 36, 12, 24, 12, 12, 24, 12],
/// ...     {
/// ...         "paid": [100.0, 150.0, 170.0, 110.0, 172.0, 130.0, 55.0, 80.0, 60.0],
/// ...         "premium": [250.0] * 3 + [270.0] * 2 + [290.0] + [120.0] * 2 + [130.0],
/// ...     },
/// ...     keys={"program": ["a"] * 6 + ["b"] * 3},
/// ... )
/// >>> import warnings
/// >>> warnings.simplefilter("ignore")  # two tiny programs: the sampler is short
/// >>> model = HierarchicalBf(apriori=0.7, level_cv=0.2, chains=2, tune=200, draws=200, n_sims=500, seed=1)
/// >>> fit = model.fit(tri, "paid", "premium")
/// >>> [p.label for p in fit.programs]
/// ['a', 'b']
/// >>> fit.reserves.dims
/// ['program', 'origin']
#[pyclass(name = "HierarchicalBf", module = "prospicio.reserving", frozen)]
pub(crate) struct PyHierarchicalBf {
    inner: HierarchicalBf,
    pattern_prior: Option<PatternPrior>,
}

#[pymethods]
impl PyHierarchicalBf {
    #[new]
    #[pyo3(signature = (
        apriori = 1.0,
        level_cv = None,
        cv = 0.1,
        development_sd = 10.0,
        tau_scale = 0.3,
        omega_scale = 0.5,
        process = "gamma",
        chains = 4,
        tune = 1000,
        draws = 1000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.99,
        max_depth = 10,
        pattern_prior = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        apriori: f64,
        level_cv: Option<f64>,
        cv: f64,
        development_sd: f64,
        tau_scale: f64,
        omega_scale: f64,
        process: &str,
        chains: usize,
        tune: usize,
        draws: usize,
        n_sims: usize,
        seed: u64,
        target_accept: f64,
        max_depth: u64,
        pattern_prior: Option<PyRef<'_, PyPatternPrior>>,
    ) -> PyResult<Self> {
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            inner: HierarchicalBf {
                apriori,
                level_cv,
                cv,
                development_sd,
                tau_scale,
                omega_scale,
                process: process_distribution(process)?,
                sampler: BayesSampler {
                    chains,
                    tune,
                    draws,
                    seed: 0,
                    target_accept,
                    max_depth,
                },
                n_sims,
                seed,
            },
            pattern_prior: pattern_prior.map(|p| p.inner.clone()),
        })
    }

    /// Prior mean of the industry loss ratio.
    #[getter]
    fn apriori(&self) -> f64 {
        self.inner.apriori
    }

    /// Coefficient of variation of the industry loss ratio's prior, or
    /// ``None`` for a flat prior on its log.
    #[getter]
    fn level_cv(&self) -> Option<f64> {
        self.inner.level_cv
    }

    /// Coefficient of variation of each origin's expected ultimate.
    #[getter]
    fn cv(&self) -> f64 {
        self.inner.cv
    }

    /// Standard deviation of each shared development effect's prior.
    #[getter]
    fn development_sd(&self) -> f64 {
        self.inner.development_sd
    }

    /// Scale of the half-normal prior on ``tau``.
    #[getter]
    fn tau_scale(&self) -> f64 {
        self.inner.tau_scale
    }

    /// Scale of the half-normal prior on ``omega``.
    #[getter]
    fn omega_scale(&self) -> f64 {
        self.inner.omega_scale
    }

    /// Process error: ``"gamma"`` or ``"none"``.
    #[getter]
    fn process(&self) -> &'static str {
        process_name(self.inner.process)
    }

    /// Number of NUTS chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.sampler.chains
    }

    /// Warm-up draws per chain.
    #[getter]
    fn tune(&self) -> usize {
        self.inner.sampler.tune
    }

    /// Kept draws per chain.
    #[getter]
    fn draws(&self) -> usize {
        self.inner.sampler.draws
    }

    /// Number of predictive simulations.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims
    }

    /// Seed of the chains and of the simulations.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Target mean acceptance rate.
    #[getter]
    fn target_accept(&self) -> f64 {
        self.inner.sampler.target_accept
    }

    /// Largest NUTS tree depth.
    #[getter]
    fn max_depth(&self) -> u64 {
        self.inner.sampler.max_depth
    }

    /// The prior on the shared pattern, or ``None`` for the vague
    /// ``N(0, development_sd**2)``.
    #[getter]
    fn pattern_prior(&self) -> Option<PyPatternPrior> {
        self.pattern_prior
            .clone()
            .map(|inner| PyPatternPrior { inner })
    }

    /// Samples the posterior of one loss column of a cumulative triangle
    /// with a segment per program, with each origin's exposure from
    /// another column, and simulates every program's reserves jointly.
    /// Every origin must be observed from the first age up to its latest.
    /// A triangle with one segment is one program. Raises one
    /// ``UserWarning`` listing the failed diagnostics, if any.
    ///
    /// Instead of one triangle, ``triangle`` may be a dict of program name
    /// to a ``Triangle`` with one segment. The dict is stacked into a
    /// triangle with a segment per program under the key ``"program"``
    /// (any keys the triangles carry are dropped). The names label the
    /// programs, which run in the sorted order of their names, as the
    /// segments of a triangle do, not in the dict's order. The triangles
    /// must all have the two columns, the same grains and the same
    /// cumulative state. The fit is then the same, draw for draw, as the
    /// fit of the same panel given as one triangle with that key.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle or dict of str to Triangle
    ///     Cumulative, with a segment per program, all on the same ages; or
    ///     the programs by name, each a triangle with one segment.
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure.
    ///
    /// Returns
    /// -------
    /// HierarchicalBfFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the triangle is not cumulative or has an origin with a gap or
    ///     without exposure, if a setting is invalid, if the programs
    ///     together have no degree of freedom, or if a development column
    ///     or origin sums to too little (the error names it). For a dict:
    ///     if it is empty, a name is empty, a triangle has several
    ///     segments or lacks a column, or the grains or cumulative state
    ///     differ.
    /// TypeError
    ///     If a dict's key is not a string or its value not a ``Triangle``.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: &Bound<'_, PyAny>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyHierarchicalBfFit> {
        let tri = programs_triangle(triangle, column, exposure)?;
        let (model, prior) = (&self.inner, &self.pattern_prior);
        let inner = py
            .detach(|| match prior {
                Some(p) => model.fit_with_pattern_prior(&tri, column, exposure, p),
                None => model.fit(&tri, column, exposure),
            })
            .map_err(err)?;
        warn_diagnostics(py, "HierarchicalBf", &inner.diagnostics.warnings)?;
        Ok(PyHierarchicalBfFit { inner })
    }

    /// The one-year view of the fit over every program at once: the
    /// claims development result of each program and origin over the
    /// coming year, by importance reweighting of the posterior by the
    /// year's cells, as ``BayesBfNormalised.one_year`` does for one
    /// triangle (``docs/design/reserving-bayes.md``, decision 30). It is
    /// one view over the stacked programs and not one per program, because
    /// a program's next diagonal moves the industry level, the spreads and
    /// the shared pattern, and through them the other programs' closing
    /// reserves. A refit of a flagged simulation samples the whole panel.
    /// Raises ``UserWarning`` for the fit's failed diagnostics, for refits
    /// whose diagnostics failed, for simulations left unrefitted by
    /// ``max_refits``, and for each of the view's ``warnings``: programs
    /// whose next diagonal falls at development ages they have never
    /// observed, which make the view expensive (nearly every simulation is
    /// flagged and refitted) or, with ``refit="reweight_only"``,
    /// approximate.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle or dict of str to Triangle
    ///     As in ``fit``.
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column.
    /// k_threshold : float, default 0.7
    ///     The Pareto ``k`` above which a simulation's importance weights
    ///     are not trusted.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, optional
    ///     What happens to a flagged simulation, as in ``BayesOdp.one_year``.
    ///     ``None`` (the default) is ``"moment_match_then_refit"`` with the
    ///     model's default sampler of the refits (``light_refit``).
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``.
    /// n_sims : int, optional
    ///     The number of one-year simulations; by default this model's
    ///     ``n_sims``, at most 2,000. A smaller count gives the first
    ///     simulations of a larger one.
    /// light_refit : bool or tuple, optional
    ///     The sampler of the refits: a shorter, warm-started NUTS run
    ///     instead of the fit's own. ``None`` (the default) is the model's
    ///     own when ``refit`` is ``None``, the light refit of ``True`` below,
    ///     and the fit's sampler, with its pilot run, when ``refit`` is given.
    ///     ``True`` takes 4 chains of 150 warm-up and 250 kept draws, a
    ///     tuple ``(chains, tune, draws)`` its own, and ``False`` turns a default
    ///     light refit off. Each refit scales its sampler by the opening
    ///     posterior and starts at its mean, with no pilot run, and warns when
    ///     its closing reserve has R-hat above 1.05, fewer than 100 effective
    ///     draws or a divergence.
    ///
    /// Returns
    /// -------
    /// HierarchicalBfOneYearFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit``; if ``k_threshold`` is NaN or ``refit`` is unknown.
    /// TypeError
    ///     As ``fit``.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import HierarchicalBf, Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2021, 2021, 2021, 2022, 2022, 2023, 2022, 2022, 2023],
    /// ...     [12, 24, 36, 12, 24, 12, 12, 24, 12],
    /// ...     {
    /// ...         "paid": [100.0, 150.0, 170.0, 110.0, 172.0, 130.0, 55.0, 80.0, 60.0],
    /// ...         "premium": [250.0] * 3 + [270.0] * 2 + [290.0] + [120.0] * 2 + [130.0],
    /// ...     },
    /// ...     keys={"program": ["a"] * 6 + ["b"] * 3},
    /// ... )
    /// >>> import warnings
    /// >>> warnings.simplefilter("ignore")  # two tiny programs: the sampler is short
    /// >>> model = HierarchicalBf(apriori=0.7, level_cv=0.2, chains=2, tune=200, draws=200, n_sims=500, seed=1)
    /// >>> view = model.one_year(tri, "paid", "premium", refit="reweight_only")
    /// >>> view.program
    /// ['a', 'a', 'a', 'b', 'b']
    /// >>> view.cdr.dims
    /// ['program', 'origin']
    /// >>> len(view.pareto_k)
    /// 500
    #[pyo3(signature = (triangle, column, exposure, k_threshold = 0.7, refit = None, max_refits = None, n_sims = None, light_refit = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: &Bound<'_, PyAny>,
        column: &str,
        exposure: &str,
        k_threshold: f64,
        refit: Option<&str>,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
        light_refit: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyHierarchicalBfOneYearFit> {
        let mut settings = one_year_settings(k_threshold, refit, max_refits, n_sims)?;
        (settings.light_refit, settings.no_light_refit) = light_refit_arg(light_refit)?;
        let tri = programs_triangle(triangle, column, exposure)?;
        let (model, prior) = (&self.inner, &self.pattern_prior);
        let inner = py
            .detach(|| match prior {
                Some(p) => model.one_year_with_pattern_prior(&tri, column, exposure, &settings, p),
                None => model.one_year(&tri, column, exposure, &settings),
            })
            .map_err(err)?;
        warn_diagnostics(py, "HierarchicalBf", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "HierarchicalBf",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        for message in &inner.warnings {
            let message = std::ffi::CString::new(format!("HierarchicalBf: {message}"))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
            PyErr::warn(py, category.as_any(), &message, 1)?;
        }
        Ok(PyHierarchicalBfOneYearFit { inner })
    }

    fn __repr__(&self) -> String {
        let s = &self.inner.sampler;
        format!(
            "HierarchicalBf(apriori={:?}, level_cv={}, cv={:?}, development_sd={:?}, \
             tau_scale={:?}, omega_scale={:?}, process={:?}, chains={}, tune={}, draws={}, \
             n_sims={}, seed={}{})",
            self.inner.apriori,
            self.inner
                .level_cv
                .map_or_else(|| "None".to_string(), |c| format!("{c:?}")),
            self.inner.cv,
            self.inner.development_sd,
            self.inner.tau_scale,
            self.inner.omega_scale,
            self.process(),
            s.chains,
            s.tune,
            s.draws,
            self.inner.n_sims,
            self.inner.seed,
            if self.pattern_prior.is_some() {
                ", pattern_prior=PatternPrior(...)"
            } else {
                ""
            },
        )
    }
}

/// A fitted ``HierarchicalBf``: the posterior of the industry loss ratio
/// ``level``, the spreads ``tau`` and ``omega``, and every program's
/// effects, its MCMC diagnostics, and the joint posterior predictive
/// distribution of every program's reserves by origin (``reserves``, with
/// the triangle's keys and ``"origin"`` as dimensions, paired by posterior
/// draw because the programs share parameters). ``programs`` holds what
/// the fit says about each program.
#[pyclass(name = "HierarchicalBfFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyHierarchicalBfFit {
    inner: HierarchicalBfFit,
}

#[pymethods]
impl PyHierarchicalBfFit {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// The hierarchical BF has no pattern per simulation: this raises
    /// ``NotSupportedError``.
    ///
    /// Parameters
    /// ----------
    /// basis : {"paid", "incurred"}
    ///     What the fit's column is, stated by the caller (no default), so
    ///     that a fit on incurred claims is not discounted by its reporting
    ///     pattern by mistake. ``"paid"`` pays each simulation's reserve
    ///     along that simulation's own pattern. ``"incurred"`` pays each
    ///     reserve (the IBNR) plus the origin's case reserve (its latest
    ///     incurred less ``latest_paid``) along ``paid_pattern``.
    /// rate : float, optional
    ///     A flat annual effective rate. Give ``rate`` or ``curve``.
    /// curve : DiscountCurve, optional
    ///     Spot rates by maturity.
    /// timing : {"mid", "end"}, default "mid"
    ///     Payments at the middle or the end of each development period.
    /// paid_pattern : list of float, optional
    ///     For ``basis="incurred"``: the cumulative paid share at each of
    ///     the fit's pattern ages (with the tail's age when the fit has a
    ///     tail), ending at 1.
    /// latest_paid : list of float, optional
    ///     For ``basis="incurred"``: each origin's latest paid, one per
    ///     component of ``reserves`` in their order.
    ///
    /// Returns
    /// -------
    /// DiscountedReserves
    #[pyo3(signature = (
        *,
        basis,
        rate = None,
        curve = None,
        timing = "mid",
        paid_pattern = None,
        latest_paid = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn discounted(
        &self,
        py: Python<'_>,
        basis: &str,
        rate: Option<f64>,
        curve: Option<PyRef<'_, PyDiscountCurve>>,
        timing: &str,
        paid_pattern: Option<Vec<f64>>,
        latest_paid: Option<Vec<f64>>,
    ) -> PyResult<PyDiscountedReserves> {
        let args = DiscountArgs {
            basis,
            rate,
            curve,
            timing,
            paid_pattern,
            latest_paid,
        };
        discounted(py, &self.inner, args)
    }

    /// The programs, in the triangle's segment order.
    ///
    /// Returns
    /// -------
    /// list of HierarchicalBfProgram
    #[getter]
    fn programs(&self) -> Vec<PyHierarchicalBfProgram> {
        self.inner
            .programs
            .iter()
            .map(|p| PyHierarchicalBfProgram { inner: p.clone() })
            .collect()
    }

    /// The development ages of every program's pattern, in months.
    #[getter]
    fn ages(&self) -> Vec<Lag> {
        self.inner.ages.clone()
    }

    /// The pooled scale per unit of premium, ``kappa``: a program's scale
    /// is ``kappa`` times its mean premium per origin.
    #[getter]
    fn scale_per_premium(&self) -> f64 {
        self.inner.scale_per_premium
    }

    /// Joint posterior predictive distribution of the reserve (the sum of
    /// future incremental values) by program and origin: one component per
    /// program and origin, one row per simulation. Each program's part is
    /// its ``reserves``.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// MCMC diagnostics: a dict with ``max_rhat``, ``min_ess_bulk``,
    /// ``min_ess_tail``, ``divergences`` and ``warnings`` (one message per
    /// failed threshold; empty when every threshold holds).
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = &self.inner.diagnostics;
        let out = PyDict::new(py);
        out.set_item("max_rhat", d.max_rhat)?;
        out.set_item("min_ess_bulk", d.min_ess_bulk)?;
        out.set_item("min_ess_tail", d.min_ess_tail)?;
        out.set_item("divergences", d.divergences)?;
        out.set_item("warnings", d.warnings.clone())?;
        Ok(out)
    }

    /// Parameter names, in the order of the columns of ``draws``:
    /// ``log_level``, ``log_tau``, ``z[c]``, ``alpha[c,w]``, ``beta[d]``,
    /// ``log_omega`` and ``delta[c,d]``.
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, one row per draw (chain by chain), one column per
    /// parameter in the order of ``parameter_names``.
    #[getter]
    fn draws(&self) -> Vec<Vec<f64>> {
        let p = self.inner.posterior.dim().max(1);
        self.inner
            .posterior
            .values()
            .chunks(p)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// Number of chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.posterior.chains()
    }

    /// Posterior summary, one row per sampled parameter: ``parameter``,
    /// ``mean``, ``sd``, ``q05``, ``q50``, ``q95``, ``rhat``, ``ess_bulk``
    /// and ``ess_tail``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fit = &self.inner;
        let s = py.detach(|| fit.summary()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// The posterior of the hyperparameters on their natural scale, in the
    /// columns of ``summary``: ``level`` (the industry loss ratio), ``tau``
    /// (the spread of the programs' loss ratios) and ``omega`` (the spread
    /// of their patterns). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn hyperparameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fit = &self.inner;
        let s = py.detach(|| fit.hyperparameters()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// One row per program: ``program``, ``origins`` (their number),
    /// ``exposure``, ``latest``, ``prior_ultimate`` and
    /// ``expected_ultimate`` (summed over the origins), ``scale``,
    /// ``loss_ratio``, ``own_loss_ratio``, ``shrinkage`` and the ``mean``
    /// and ``std_dev`` of the program's posterior predictive reserve.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // The core's table (#838), the same in R; origins is a count.
        let frame = table_frame(py, self.inner.totals())?;
        let origins = frame
            .get_item("origins")?
            .call_method1("astype", ("int64",))?;
        frame.set_item("origins", origins)?;
        Ok(frame)
    }

    fn __repr__(&self) -> String {
        format!(
            "HierarchicalBfFit(programs={}, ages={}, draws={}, n_sims={}, warnings={})",
            self.inner.programs.len(),
            self.inner.ages.len(),
            self.inner.posterior.n_draws(),
            self.inner.reserves.n_sims(),
            self.inner.diagnostics.warnings.len(),
        )
    }
}

/// One program of a ``HierarchicalBfFit``. Per-origin lists run over the
/// program's origins in order (``origins``).
#[pyclass(name = "HierarchicalBfProgram", module = "prospicio.reserving", frozen)]
pub(crate) struct PyHierarchicalBfProgram {
    inner: HierarchicalBfProgram,
}

#[pymethods]
impl PyHierarchicalBfProgram {
    /// The program's segment label: the key values joined by ``" / "``.
    #[getter]
    fn label(&self) -> String {
        self.inner.label.to_string()
    }

    /// The program's key values, one per key of the triangle.
    #[getter]
    fn key(&self) -> Vec<String> {
        self.inner.label.parts().to_vec()
    }

    /// Origin period of each origin.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    /// Exposure per origin: the exposure column's latest observed value.
    #[getter]
    fn exposure(&self) -> Vec<f64> {
        self.inner.exposure.clone()
    }

    /// The latest observed cumulative value of each origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        self.inner.latest.clone()
    }

    /// The hyperprior's mean of each origin's expected ultimate,
    /// ``apriori * exposure``.
    #[getter]
    fn prior_ultimate(&self) -> Vec<f64> {
        self.inner.prior_ultimate.clone()
    }

    /// The plugged-in scale, ``scale_per_premium`` times the program's mean
    /// premium per origin.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    /// The posterior mean of each origin's expected ultimate.
    #[getter]
    fn expected_ultimate(&self) -> Vec<f64> {
        self.inner.expected_ultimate.clone()
    }

    /// The posterior mean of the program's share at each of the fit's
    /// ``ages``; the shares sum to 1.
    #[getter]
    fn pattern(&self) -> Vec<f64> {
        self.inner.pattern.clone()
    }

    /// The posterior mean of the program's loss ratio ``L * s``.
    #[getter]
    fn loss_ratio(&self) -> f64 {
        self.inner.loss_ratio
    }

    /// The loss ratio of the program's own chain ladder (its ultimates over
    /// its premium, with the factors pooled over every program past the
    /// ages it observes).
    #[getter]
    fn own_loss_ratio(&self) -> f64 {
        self.inner.own_loss_ratio
    }

    /// How far the program stays from the collective on the log scale: 1
    /// keeps its own level and 0 takes the collective's; ``nan`` when its
    /// own loss ratio is not positive.
    #[getter]
    fn shrinkage(&self) -> f64 {
        self.inner.shrinkage
    }

    /// The program's part of the fit's ``reserves``, by origin, paired
    /// with it simulation by simulation.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HierarchicalBfProgram(label='{}', origins={}, loss_ratio={:?}, own_loss_ratio={:?}, \
             shrinkage={:?})",
            self.inner.label,
            self.inner.origins.len(),
            self.inner.loss_ratio,
            self.inner.own_loss_ratio,
            self.inner.shrinkage,
        )
    }
}

/// The one-year view of a ``HierarchicalBf`` over every program at once,
/// from ``HierarchicalBf.one_year``: the lifetime fit, the opening reserve,
/// the claims development result over the coming year and how reliable each
/// simulation's importance reweighting was.
///
/// It is one view over the stacked programs, not one per program, because
/// a program's next diagonal moves the industry level, the spreads and the
/// shared pattern, and through them the other programs' closing reserves.
/// ``cdr`` is the joint distribution of the claims development result by
/// program and origin (the triangle's keys and ``"origin"`` as dimensions),
/// the opening reserve less the year's payment and the posterior mean
/// reserve after the year, so a negative value is an adverse development
/// and ``cdr.aggregate(["program"])`` is each program's result with the
/// dependence between programs kept. Per-origin lists run over every
/// program's origins in turn, program by program, like the components of
/// ``cdr``. ``pareto_k`` has one Pareto ``k`` per simulation;
/// ``flagged_share`` is the share above ``k_threshold``, whether or not
/// they were refitted, ``refitted`` lists the simulations refitted with
/// NUTS and ``moment_matched`` those moment matching rescued.
#[pyclass(
    name = "HierarchicalBfOneYearFit",
    module = "prospicio.reserving",
    frozen
)]
pub(crate) struct PyHierarchicalBfOneYearFit {
    inner: BayesOneYearFit<HierarchicalBfFit>,
}

#[pymethods]
impl PyHierarchicalBfOneYearFit {
    /// The model fitted to the observed triangle, with its posterior and
    /// lifetime reserves, as ``HierarchicalBf.fit`` gives it.
    ///
    /// Returns
    /// -------
    /// HierarchicalBfFit
    #[getter]
    fn fit(&self) -> PyHierarchicalBfFit {
        PyHierarchicalBfFit {
            inner: self.inner.fit.clone(),
        }
    }

    /// The program of each per-origin value and CDR component: the
    /// segment label, repeated over its origins.
    #[getter]
    fn program(&self) -> Vec<String> {
        self.inner
            .fit
            .programs
            .iter()
            .flat_map(|p| std::iter::repeat_n(p.label.to_string(), p.origins.len()))
            .collect()
    }

    /// Origin period of each per-origin value and CDR component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .fit
            .programs
            .iter()
            .flat_map(|p| p.origins.iter().map(ToString::to_string))
            .collect()
    }

    /// Latest observed value per program and origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        self.inner
            .fit
            .programs
            .iter()
            .flat_map(|p| p.latest.iter().copied())
            .collect()
    }

    /// The latest value plus the opening reserve, per program and origin.
    #[getter]
    fn opening_ultimate(&self) -> Vec<f64> {
        self.inner.opening_ultimate.clone()
    }

    /// Each origin's posterior mean reserve on the observed triangle: the
    /// mean over the posterior draws of the expected future increments.
    #[getter]
    fn opening_reserve(&self) -> Vec<f64> {
        self.inner.opening_reserve.clone()
    }

    /// Joint distribution of the claims development result by program and
    /// origin: one component per program and origin, one row per
    /// simulation.
    #[getter]
    fn cdr(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.cdr.clone(),
        }
    }

    /// Each simulation's Pareto ``k``, in the order of the simulations
    /// (``inf`` when too few weights differ to fit the tail).
    #[getter]
    fn pareto_k(&self) -> Vec<f64> {
        self.inner.pareto_k.clone()
    }

    /// The simulations refitted with NUTS, in order (0-based).
    #[getter]
    fn refitted(&self) -> Vec<usize> {
        self.inner.refitted.clone()
    }

    /// The simulations moment matching rescued (their ``k`` brought to at
    /// most ``k_threshold``), in order (0-based); empty unless ``refit``
    /// was ``"moment_match_then_refit"``, when each flagged simulation is
    /// in this, in ``refitted`` or in ``unrefitted``.
    #[getter]
    fn moment_matched(&self) -> Vec<usize> {
        self.inner.moment_matched.clone()
    }

    /// The simulations due for a refit beyond ``max_refits``, reweighted
    /// instead, in order (0-based); empty without ``max_refits``.
    #[getter]
    fn unrefitted(&self) -> Vec<usize> {
        self.inner.unrefitted.clone()
    }

    /// How many of the refits had a diagnostic warning (R-hat, ESS or a
    /// divergence).
    #[getter]
    fn refit_warnings(&self) -> usize {
        self.inner.refit_warnings
    }

    /// The Pareto ``k`` above which a simulation is flagged.
    #[getter]
    fn k_threshold(&self) -> f64 {
        self.inner.k_threshold
    }

    /// The share of simulations whose ``k`` is above ``k_threshold`` (or
    /// NaN), whether or not they were refitted.
    #[getter]
    fn flagged_share(&self) -> f64 {
        self.inner.flagged_share()
    }

    /// What else the view warns of, one message each (``one_year`` raises
    /// each as a ``UserWarning``): the programs whose next diagonal falls at
    /// development ages they have never observed, which make the view
    /// expensive (or, reweighting only, approximate). Empty when the
    /// programs share their ages.
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.inner.warnings.clone()
    }

    /// One row per program and origin: ``program``, ``origin``, ``latest``,
    /// ``opening_ultimate``, ``opening_reserve`` and the ``cdr_mean`` and
    /// ``cdr_std_dev`` of the claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per program: the ``latest`` value, ``opening_ultimate`` and
    /// ``opening_reserve`` summed over its origins, and the ``cdr_mean``
    /// and ``cdr_std_dev`` of the program's claims development result (its
    /// origins' sum), with the ``flagged_share`` of the whole view. Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    fn __repr__(&self) -> String {
        format!(
            "HierarchicalBfOneYearFit(programs={}, origins={}, n_sims={}, flagged_share={:?}, \
             refitted={})",
            self.inner.fit.programs.len(),
            self.inner.opening_reserve.len(),
            self.inner.cdr.n_sims(),
            self.inner.flagged_share(),
            self.inner.refitted.len(),
        )
    }
}

/// The prior of a layer's expected ultimate for ``BayesBfNormalised``, from
/// the ground-up expected loss ratio and the layer share, each with an
/// uncertainty (``docs/design/reserving-excess.md``, PR 4).
///
/// The layer's a-priori ultimate for origin ``i`` is the ground-up exposure
/// times ``apriori`` times the share. The normalised BF puts a gamma prior
/// on the layer's expected ultimate, centred there, whose coefficient of
/// variation is that of a product of two independent variables,
/// ``cv = sqrt((1 + apriori_cv**2) * (1 + share_cv**2) - 1)``: the moments
/// are exact. One share error is common to all origins (the default,
/// ``share_dependence="common"``): the model carries one more parameter, a
/// gamma multiplier ``s`` with mean 1 and cv ``share_cv``, sampled with the
/// others, and each origin's prior ultimate is a gamma with mean
/// ``exposure * apriori * share * s`` and cv ``apriori_cv``, so every origin
/// keeps the marginal cv above, the origins' priors are correlated and the
/// data update the share for every origin at once (``log_share`` in the
/// fit's parameters). ``share_dependence="per_origin"`` is the first
/// version: an independent share error at each origin, which understates
/// the total's interval. The layer's development pattern is estimated from
/// the layer's own triangle under the model's vague prior on the pattern;
/// an excess pattern from ``ExcessPattern`` does not enter (open decision
/// 22 of the design note).
///
/// Parameters
/// ----------
/// apriori : float
///     The ground-up expected ultimate loss ratio.
/// shares : float or list of float
///     The layer share (``layer_share`` of a severity, or an exposure
///     curve's ``layer_share``), one for all origins or one per origin,
///     each in ``(0, 1]``.
/// apriori_cv : float, default 0.1
///     The loss ratio's coefficient of variation (``inf`` is flat).
/// share_cv : float, default 0.0
///     The share's coefficient of variation, the same at every origin.
/// share_dependence : {"common", "per_origin"}, default "common"
///     Whether one share error is common to all origins or each origin has
///     its own. A known share (``share_cv`` 0) or a flat prior has nothing
///     to share and is the per-origin model; a common share error needs a
///     positive ``apriori_cv``.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import LayerSharePrior
/// >>> prior = LayerSharePrior(0.65, 0.2, apriori_cv=0.1, share_cv=0.3)
/// >>> round(prior.layer_apriori(0), 12), round(prior.cv ** 2, 12)
/// (0.13, 0.1009)
/// >>> prior.share_dependence
/// 'common'
#[pyclass(name = "LayerSharePrior", module = "prospicio.reserving", frozen)]
pub(crate) struct PyLayerSharePrior {
    inner: LayerSharePrior,
}

fn share_dependence_arg(name: &str) -> PyResult<ShareDependence> {
    match name {
        "common" => Ok(ShareDependence::Common),
        "per_origin" => Ok(ShareDependence::PerOrigin),
        _ => Err(PyValueError::new_err(format!(
            "unknown share_dependence {name:?}; expected \"common\" or \"per_origin\""
        ))),
    }
}

fn shares_arg(shares: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(one) = shares.extract::<f64>() {
        return Ok(vec![one]);
    }
    shares.extract().map_err(|_| {
        PyTypeError::new_err(format!(
            "shares must be a number or a list of numbers, got {}",
            py_repr(shares)
        ))
    })
}

#[pymethods]
impl PyLayerSharePrior {
    #[new]
    #[pyo3(signature = (apriori, shares, apriori_cv = 0.1, share_cv = 0.0, share_dependence = "common"))]
    fn new(
        apriori: f64,
        shares: &Bound<'_, PyAny>,
        apriori_cv: f64,
        share_cv: f64,
        share_dependence: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: LayerSharePrior::new(apriori, apriori_cv, &shares_arg(shares)?, share_cv)
                .map_err(err)?
                .with_dependence(share_dependence_arg(share_dependence)?),
        })
    }

    /// A prior from one share and its standard error: the share's
    /// coefficient of variation is ``share_se / share``.
    ///
    /// Parameters
    /// ----------
    /// apriori : float
    /// share : float
    /// share_se : float
    /// apriori_cv : float, default 0.1
    /// share_dependence : {"common", "per_origin"}, default "common"
    ///
    /// Returns
    /// -------
    /// LayerSharePrior
    #[staticmethod]
    #[pyo3(signature = (apriori, share, share_se, apriori_cv = 0.1, share_dependence = "common"))]
    fn from_share_se(
        apriori: f64,
        share: f64,
        share_se: f64,
        apriori_cv: f64,
        share_dependence: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: LayerSharePrior::from_share_se(apriori, apriori_cv, share, share_se)
                .map_err(err)?
                .with_dependence(share_dependence_arg(share_dependence)?),
        })
    }

    /// A prior with the share of a severity in a layer (``layer_share``,
    /// ultimate basis) and its coefficient of variation.
    ///
    /// Parameters
    /// ----------
    /// apriori : float
    /// severity : severity distribution
    /// layer : ExcessLayer
    /// apriori_cv : float, default 0.1
    /// share_cv : float, default 0.0
    /// share_dependence : {"common", "per_origin"}, default "common"
    ///
    /// Returns
    /// -------
    /// LayerSharePrior
    #[staticmethod]
    #[pyo3(signature = (apriori, severity, layer, apriori_cv = 0.1, share_cv = 0.0, share_dependence = "common"))]
    fn from_severity(
        apriori: f64,
        severity: &Bound<'_, PyAny>,
        layer: &Bound<'_, PyAny>,
        apriori_cv: f64,
        share_cv: f64,
        share_dependence: &str,
    ) -> PyResult<Self> {
        let sev = crate::distributions::extract_severity(severity)?;
        let layer = crate::excess::layer_of(layer, "layer")?;
        Ok(Self {
            inner: LayerSharePrior::from_severity(apriori, apriori_cv, &sev, layer, share_cv)
                .map_err(err)?
                .with_dependence(share_dependence_arg(share_dependence)?),
        })
    }

    /// The ground-up expected ultimate loss ratio.
    #[getter]
    fn apriori(&self) -> f64 {
        self.inner.apriori
    }

    /// The loss ratio's coefficient of variation.
    #[getter]
    fn apriori_cv(&self) -> f64 {
        self.inner.apriori_cv
    }

    /// The layer shares, one or one per origin.
    #[getter]
    fn shares(&self) -> Vec<f64> {
        self.inner.shares.clone()
    }

    /// The share's coefficient of variation.
    #[getter]
    fn share_cv(&self) -> f64 {
        self.inner.share_cv
    }

    /// ``"common"`` or ``"per_origin"``: whether one share error is common
    /// to all origins.
    #[getter]
    fn share_dependence(&self) -> &'static str {
        match self.inner.dependence {
            ShareDependence::Common => "common",
            ShareDependence::PerOrigin => "per_origin",
        }
    }

    /// The marginal coefficient of variation of the prior on each origin's
    /// expected ultimate: ``sqrt((1 + apriori_cv**2) * (1 + share_cv**2) -
    /// 1)``, ``inf`` if either is. It is the gamma's cv under
    /// ``"per_origin"``; under ``"common"`` the model's own cv is
    /// ``apriori_cv`` and the multiplier's ``share_cv``.
    #[getter]
    fn cv(&self) -> f64 {
        self.inner.cv()
    }

    /// Whether the model carries a multiplier common to all origins: the
    /// share error is ``"common"`` and both cvs are positive and finite.
    #[getter]
    fn has_common_factor(&self) -> bool {
        self.inner.has_common_factor()
    }

    /// The mean and standard deviation of the prior of the layer's total
    /// expected ultimate, in closed form: ``c_e**2 * (1 + c_s**2) * sum(m**2)
    /// + c_s**2 * sum(m)**2`` for a common share error and ``((1 + c_e**2) *
    /// (1 + c_s**2) - 1) * sum(m**2)`` for independent ones, with ``m`` each
    /// origin's prior ultimate ``exposure * apriori * share``.
    ///
    /// Parameters
    /// ----------
    /// exposure : list of float
    ///     The ground-up exposure of each origin.
    ///
    /// Returns
    /// -------
    /// tuple of float
    ///     ``(mean, sd)``; ``sd`` is ``inf`` for a flat prior.
    fn prior_total(&self, exposure: Vec<f64>) -> PyResult<(f64, f64)> {
        self.inner.prior_total(&exposure).map_err(err)
    }

    /// The layer's a-priori loss ratio at origin ``i`` (0-based): the
    /// ground-up loss ratio times that origin's share.
    ///
    /// Parameters
    /// ----------
    /// i : int
    ///
    /// Returns
    /// -------
    /// float
    fn layer_apriori(&self, i: usize) -> PyResult<f64> {
        if self.inner.shares.len() > 1 && i >= self.inner.shares.len() {
            return Err(PyValueError::new_err(format!(
                "origin {i} is out of range for {} shares",
                self.inner.shares.len()
            )));
        }
        Ok(self.inner.layer_apriori(i))
    }

    /// ``model`` (default settings if omitted) with this prior's ground-up
    /// ``apriori`` and its ``cv``, for the triangle of ``triangle``.
    ///
    /// Parameters
    /// ----------
    /// model : BayesBfNormalised, optional
    ///     Supplies the sampler, process error, simulations, seed and
    ///     ``pattern_prior``; its own ``apriori``, ``cv`` and ``common_cv``
    ///     are replaced.
    ///
    /// Returns
    /// -------
    /// BayesBfNormalised
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A common share error with ``apriori_cv`` 0.
    #[pyo3(signature = (model = None))]
    fn model(
        &self,
        model: Option<PyRef<'_, PyBayesBfNormalised>>,
    ) -> PyResult<PyBayesBfNormalised> {
        let pattern_prior = model.as_ref().and_then(|m| m.pattern_prior.clone());
        let base = model.map_or_else(BayesBfNormalised::default, |m| m.inner.clone());
        Ok(PyBayesBfNormalised {
            inner: self.inner.model(&base).map_err(err)?,
            pattern_prior,
        })
    }

    /// ``triangle`` with a ``"layer_exposure"`` column: the ground-up
    /// exposure column times each origin's share.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// exposure : str
    ///     The ground-up exposure column.
    ///
    /// Returns
    /// -------
    /// Triangle
    fn triangle(&self, triangle: PyRef<'_, PyTriangle>, exposure: &str) -> PyResult<PyTriangle> {
        Ok(PyTriangle {
            inner: self
                .inner
                .triangle(&triangle.inner, exposure)
                .map_err(err)?,
        })
    }

    /// Fits the layer losses ``column`` of a single-segment cumulative
    /// triangle that also holds the ground-up ``exposure`` column, with
    /// this prior: ``BayesBfNormalised.fit`` on the layer exposure. Raises
    /// one ``UserWarning`` listing the failed diagnostics, if any.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///     The layer losses.
    /// exposure : str
    ///     The ground-up exposure column.
    /// model : BayesBfNormalised, optional
    ///     The sampler and other settings; its ``apriori`` and ``cv`` are
    ///     replaced by the prior's.
    ///
    /// Returns
    /// -------
    /// BayesBfNormalisedFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``BayesBfNormalised.fit``; if the prior's ``cv`` is 0.
    #[pyo3(signature = (triangle, column, exposure, model = None))]
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
        model: Option<PyRef<'_, PyBayesBfNormalised>>,
    ) -> PyResult<PyBayesBfNormalisedFit> {
        let pattern = model.as_ref().and_then(|m| m.pattern_prior.clone());
        let base = model.map_or_else(BayesBfNormalised::default, |m| m.inner.clone());
        let (prior, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| match &pattern {
                Some(p) => prior.model(&base)?.fit_with_pattern_prior(
                    &prior.triangle(tri, exposure)?,
                    column,
                    prospicio_reserving::bayes::LAYER_EXPOSURE_COLUMN,
                    p,
                ),
                None => prior.fit(&base, tri, column, exposure),
            })
            .map_err(err)?;
        warn_diagnostics(py, "LayerSharePrior", &inner.diagnostics.warnings)?;
        Ok(PyBayesBfNormalisedFit { inner })
    }

    /// The one-year view of the same fit (``BayesBfNormalised.one_year``).
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    /// exposure : str
    /// model : BayesBfNormalised, optional
    /// k_threshold : float, default 0.7
    /// refit : str, default "moment_match_then_refit"
    /// max_refits : int, optional
    /// n_sims : int, optional
    ///     As in ``BayesBfNormalised.one_year``.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFit
    #[pyo3(signature = (triangle, column, exposure, model = None, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
        model: Option<PyRef<'_, PyBayesBfNormalised>>,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFit> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let pattern = model.as_ref().and_then(|m| m.pattern_prior.clone());
        let base = model.map_or_else(BayesBfNormalised::default, |m| m.inner.clone());
        let (prior, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| match &pattern {
                Some(p) => prior.model(&base)?.one_year_with_pattern_prior(
                    &prior.triangle(tri, exposure)?,
                    column,
                    prospicio_reserving::bayes::LAYER_EXPOSURE_COLUMN,
                    &settings,
                    p,
                ),
                None => prior.one_year(&base, tri, column, exposure, &settings),
            })
            .map_err(err)?;
        warn_diagnostics(py, "LayerSharePrior", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "LayerSharePrior",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        Ok(PyBayesOneYearFit {
            inner: BayesOneYearOf::BfNormalised(inner),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "LayerSharePrior(apriori={:?}, shares={:?}, apriori_cv={:?}, share_cv={:?}, share_dependence={:?})",
            self.inner.apriori,
            self.inner.shares,
            self.inner.apriori_cv,
            self.inner.share_cv,
            self.share_dependence(),
        )
    }
}

/// A normal prior on the development pattern of ``BayesBfNormalised``: at
/// each age after the first, ``beta[d] = log(p[d] / p[1])`` (the log ratio
/// of the share of the ultimate at age ``d`` to the first age's) is normal
/// with mean ``mean[d]`` and sd ``sd[d]``, in place of the model's
/// ``N(0, development_sd**2)`` (``docs/design/reserving-bayes.md``,
/// decision 29). ``IndustryPrior.pattern`` builds one from other companies'
/// triangles.
///
/// Parameters
/// ----------
/// ages : list of int
///     The development ages in months, increasing; at least two.
/// mean : list of float
///     The prior mean of each ``beta[d]``, one fewer than the ages.
/// sd : list of float
///     The prior sd of each ``beta[d]``, finite and positive.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import PatternPrior
/// >>> prior = PatternPrior.from_shares([12, 24, 36], [0.5, 0.3, 0.2], [0.1, 0.1])
/// >>> [round(p, 12) for p in prior.shares()]
/// [0.5, 0.3, 0.2]
/// >>> prior.truncated(24).ages
/// [12, 24]
#[pyclass(name = "PatternPrior", module = "prospicio.reserving", frozen)]
pub(crate) struct PyPatternPrior {
    pub(crate) inner: PatternPrior,
}

#[pymethods]
impl PyPatternPrior {
    #[new]
    fn new(ages: Vec<Lag>, mean: Vec<f64>, sd: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: PatternPrior::new(&ages, &mean, &sd).map_err(err)?,
        })
    }

    /// A prior centred on incremental ``shares`` of the ultimate (positive;
    /// they need not sum to 1), with ``sd`` on each log ratio after the
    /// first age.
    ///
    /// Parameters
    /// ----------
    /// ages : list of int
    /// shares : list of float
    ///     One per age.
    /// sd : list of float
    ///     One per age after the first.
    ///
    /// Returns
    /// -------
    /// PatternPrior
    #[staticmethod]
    fn from_shares(ages: Vec<Lag>, shares: Vec<f64>, sd: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: PatternPrior::from_shares(&ages, &shares, &sd).map_err(err)?,
        })
    }

    /// The development ages in months.
    #[getter]
    fn ages(&self) -> Vec<Lag> {
        self.inner.ages.clone()
    }

    /// The prior mean of each ``beta[d]``, ages after the first.
    #[getter]
    fn mean(&self) -> Vec<f64> {
        self.inner.mean.clone()
    }

    /// The prior sd of each ``beta[d]``, ages after the first.
    #[getter]
    fn sd(&self) -> Vec<f64> {
        self.inner.sd.clone()
    }

    /// The incremental shares at the prior mean, summing to 1.
    ///
    /// Returns
    /// -------
    /// list of float
    fn shares(&self) -> Vec<f64> {
        self.inner.shares()
    }

    /// The percent developed at each age at the prior mean, ending at 1.
    ///
    /// Returns
    /// -------
    /// list of float
    fn percent_developed(&self) -> Vec<f64> {
        self.inner.percent_developed()
    }

    /// The prior on the ages up to ``age`` (one of its ages, not the
    /// first): the log ratios to the first age keep their means and sds.
    ///
    /// Parameters
    /// ----------
    /// age : int
    ///
    /// Returns
    /// -------
    /// PatternPrior
    fn truncated(&self, age: Lag) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.truncated(age).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PatternPrior(ages={:?}, mean={:?}, sd={:?})",
            self.inner.ages, self.inner.mean, self.inner.sd
        )
    }
}

/// An industry prior for ``BayesBfNormalised`` from other companies'
/// triangles (``docs/design/reserving-bayes.md``, decision 29): the mean
/// ultimate loss ratio over the companies (``apriori``), the cv of a
/// company's level about it (``between_cv``, the model's ``common_cv``),
/// the cv of an origin about its company's level (``cv``, the model's
/// ``cv``) and a ``PatternPrior`` with the mean and sd over the companies
/// of each age's log share ratio. Each company's loss ratios and pattern
/// come from its own volume-weighted chain ladder (no tail, so the
/// triangles' last age is the ultimate); only origins developed at least
/// ``min_developed`` by the industry pattern count. The estimators are
/// moments on the natural scale (a one-way random-effects decomposition),
/// written out in the Rust docs of ``IndustryPrior``. No data ship with
/// the library: the triangles are the user's, for example Schedule P
/// triangles read with the ``cas-schedule-p`` package.
///
/// ``within_cv`` is the realised origin-to-origin spread and ``cv`` equals
/// it by default; ``subtract_process=True`` takes out the ODP process error
/// that each company's Pearson scale implies (``process_cv``), which the
/// model adds back through its own scale. On Schedule P that scale often
/// explains more than the whole spread (incurred above all), so the default
/// keeps the realised spread and errs wide.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import IndustryPrior, Triangle
/// >>> def company(scale):
/// ...     return Triangle.from_long(
/// ...         [2020, 2020, 2020, 2021, 2021, 2022],
/// ...         [12, 24, 36, 12, 24, 12],
/// ...         {"paid": [v * scale for v in [50.0, 80.0, 100.0, 55.0, 88.0, 60.0]]},
/// ...     )
/// >>> prior = IndustryPrior.from_triangles(
/// ...     [company(1.0), company(1.2), company(0.9)], "paid", [[160.0] * 3] * 3
/// ... )
/// >>> prior.report["n_companies_used"]
/// 3
/// >>> prior.pattern.ages
/// [12, 24, 36]
#[pyclass(name = "IndustryPrior", module = "prospicio.reserving", frozen)]
pub(crate) struct PyIndustryPrior {
    inner: IndustryPrior,
}

fn industry_settings(
    min_developed: f64,
    min_companies: usize,
    subtract_process: bool,
) -> IndustryPriorSettings {
    IndustryPriorSettings {
        min_developed,
        min_companies,
        subtract_process,
    }
}

#[pymethods]
impl PyIndustryPrior {
    /// The prior from single-segment triangles, one per company, of loss
    /// ``column`` (cumulative or incremental), with each company's premium
    /// per origin. Every triangle must have the same development ages.
    ///
    /// Parameters
    /// ----------
    /// triangles : list of Triangle
    ///     One per company; at least two.
    /// column : str
    ///     The losses.
    /// premiums : list of list of float
    ///     Each company's premium per origin, in origin order. An origin
    ///     whose premium is not positive is left out.
    /// min_developed : float, default 0.5
    ///     The smallest share of the ultimate developed (by the industry
    ///     pattern) at an origin's latest age for its loss ratio to count.
    /// min_companies : int, default 10
    ///     Fewer companies than this add a warning to ``report``.
    /// subtract_process : bool, default False
    ///     Whether ``cv`` is ``within_cv`` less the ODP process error.
    ///
    /// Returns
    /// -------
    /// IndustryPrior
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If fewer than two companies have a loss ratio, the triangles'
    ///     ages differ, a company's premiums do not match its origins, no
    ///     company has a pattern at some age, or ``min_developed`` is
    ///     outside 0 to 1.
    #[staticmethod]
    #[pyo3(signature = (triangles, column, premiums, min_developed = 0.5, min_companies = 10, subtract_process = false))]
    fn from_triangles(
        py: Python<'_>,
        triangles: Vec<PyRef<'_, PyTriangle>>,
        column: &str,
        premiums: Vec<Vec<f64>>,
        min_developed: f64,
        min_companies: usize,
        subtract_process: bool,
    ) -> PyResult<Self> {
        let triangles: Vec<Triangle> = triangles.iter().map(|t| t.inner.clone()).collect();
        let settings = industry_settings(min_developed, min_companies, subtract_process);
        let inner = py
            .detach(|| settings.from_triangles(&triangles, column, &premiums))
            .map_err(err)?;
        Ok(Self { inner })
    }

    /// The prior from one triangle with a segment per company (a key such
    /// as the group code), loss ``column``, and each origin's premium from
    /// the ``exposure`` column (its latest observed value).
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    /// exposure : str
    /// min_developed, min_companies, subtract_process
    ///     As in ``from_triangles``.
    ///
    /// Returns
    /// -------
    /// IndustryPrior
    #[staticmethod]
    #[pyo3(signature = (triangle, column, exposure, min_developed = 0.5, min_companies = 10, subtract_process = false))]
    fn from_segments(
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
        min_developed: f64,
        min_companies: usize,
        subtract_process: bool,
    ) -> PyResult<Self> {
        let settings = industry_settings(min_developed, min_companies, subtract_process);
        let tri = &triangle.inner;
        let inner = py
            .detach(|| settings.from_segments(tri, column, exposure))
            .map_err(err)?;
        Ok(Self { inner })
    }

    /// The mean over the companies of their mean ultimate loss ratio.
    #[getter]
    fn apriori(&self) -> f64 {
        self.inner.apriori
    }

    /// The standard error of ``apriori``.
    #[getter]
    fn apriori_se(&self) -> f64 {
        self.inner.apriori_se
    }

    /// The cv of a company's level about ``apriori``.
    #[getter]
    fn between_cv(&self) -> f64 {
        self.inner.between_cv
    }

    /// The cv of the realised loss ratios of a company's origins about its
    /// level.
    #[getter]
    fn within_cv(&self) -> f64 {
        self.inner.within_cv
    }

    /// The part of ``within_cv`` that the companies' ODP process error
    /// explains.
    #[getter]
    fn process_cv(&self) -> f64 {
        self.inner.process_cv
    }

    /// The model's ``cv``: ``within_cv``, or less the process error with
    /// ``subtract_process``; at least 0.01.
    #[getter]
    fn cv(&self) -> f64 {
        self.inner.cv
    }

    /// The model's ``common_cv``: ``between_cv``, or ``None`` when it is 0.
    #[getter]
    fn common_cv(&self) -> Option<f64> {
        self.inner.common_cv()
    }

    /// The prior on the pattern.
    #[getter]
    fn pattern(&self) -> PyPatternPrior {
        PyPatternPrior {
            inner: self.inner.pattern.clone(),
        }
    }

    /// The percent developed at each age at the pattern prior's mean.
    #[getter]
    fn percent_developed(&self) -> Vec<f64> {
        self.inner.percent_developed.clone()
    }

    /// The sd over the companies of their percent developed at each age.
    #[getter]
    fn percent_developed_sd(&self) -> Vec<f64> {
        self.inner.percent_developed_sd.clone()
    }

    /// Each company's mean ultimate loss ratio, companies left out omitted.
    #[getter]
    fn company_loss_ratios(&self) -> Vec<f64> {
        self.inner.company_loss_ratios.clone()
    }

    /// The factor from the pattern's last age to the ultimate: 1, or more
    /// for a prior cut by ``at_age``.
    #[getter]
    fn tail_factor(&self) -> f64 {
        self.inner.tail_factor
    }

    /// The report: ``n_companies``, ``n_companies_used``,
    /// ``n_origins_used``, ``origins_used`` (per company),
    /// ``companies_by_age`` (with a pattern at each age after the first),
    /// ``excluded`` and ``warnings`` (lists of str).
    #[getter]
    fn report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let r = &self.inner.report;
        let d = PyDict::new(py);
        d.set_item("n_companies", r.n_companies)?;
        d.set_item("n_companies_used", r.n_companies_used)?;
        d.set_item("n_origins_used", r.n_origins_used)?;
        d.set_item("origins_used", r.origins_used.clone())?;
        d.set_item("companies_by_age", r.companies_by_age.clone())?;
        d.set_item("excluded", r.excluded.clone())?;
        d.set_item("warnings", r.warnings.clone())?;
        Ok(d)
    }

    /// The prior for a triangle whose last age is ``age``: ``apriori``
    /// becomes the loss ratio developed to it, the pattern keeps the ages up
    /// to it and ``tail_factor`` is the factor from it to ultimate, which
    /// the user applies outside the model.
    ///
    /// Parameters
    /// ----------
    /// age : int
    ///
    /// Returns
    /// -------
    /// IndustryPrior
    fn at_age(&self, age: Lag) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.at_age(age).map_err(err)?,
        })
    }

    /// ``model`` (default settings if omitted) with this prior's
    /// ``apriori``, ``cv``, ``common_cv`` and ``pattern`` as its
    /// ``pattern_prior``; its other settings are kept.
    ///
    /// Parameters
    /// ----------
    /// model : BayesBfNormalised, optional
    ///
    /// Returns
    /// -------
    /// BayesBfNormalised
    #[pyo3(signature = (model = None))]
    fn model(&self, model: Option<PyRef<'_, PyBayesBfNormalised>>) -> PyBayesBfNormalised {
        let base = model.map_or_else(BayesBfNormalised::default, |m| m.inner.clone());
        PyBayesBfNormalised {
            inner: self.inner.model(&base),
            pattern_prior: Some(self.inner.pattern.clone()),
        }
    }

    /// ``model()`` on the default settings.
    ///
    /// Returns
    /// -------
    /// BayesBfNormalised
    fn to_bf_normalised_settings(&self) -> PyBayesBfNormalised {
        self.model(None)
    }

    /// ``self.model(model).fit(triangle, column, exposure)``: the pattern
    /// prior's ages must be the programs' ages (``at_age`` cuts it). Raises
    /// one ``UserWarning`` listing the failed diagnostics, if any.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    /// exposure : str
    /// model : BayesBfNormalised, optional
    ///
    /// Returns
    /// -------
    /// BayesBfNormalisedFit
    #[pyo3(signature = (triangle, column, exposure, model = None))]
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
        model: Option<PyRef<'_, PyBayesBfNormalised>>,
    ) -> PyResult<PyBayesBfNormalisedFit> {
        let base = model.map_or_else(BayesBfNormalised::default, |m| m.inner.clone());
        let (prior, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| prior.fit(&base, tri, column, exposure))
            .map_err(err)?;
        warn_diagnostics(py, "IndustryPrior", &inner.diagnostics.warnings)?;
        Ok(PyBayesBfNormalisedFit { inner })
    }

    /// The one-year view of the same fit (``BayesBfNormalised.one_year``).
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    /// exposure : str
    /// model : BayesBfNormalised, optional
    /// k_threshold : float, default 0.7
    /// refit : str, default "moment_match_then_refit"
    /// max_refits : int, optional
    /// n_sims : int, optional
    ///     As in ``BayesBfNormalised.one_year``.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFit
    #[pyo3(signature = (triangle, column, exposure, model = None, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
        model: Option<PyRef<'_, PyBayesBfNormalised>>,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFit> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let base = model.map_or_else(BayesBfNormalised::default, |m| m.inner.clone());
        let (prior, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| prior.one_year(&base, tri, column, exposure, &settings))
            .map_err(err)?;
        warn_diagnostics(py, "IndustryPrior", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "IndustryPrior",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        Ok(PyBayesOneYearFit {
            inner: BayesOneYearOf::BfNormalised(inner),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "IndustryPrior(apriori={:.4}, between_cv={:.4}, cv={:.4}, ages={:?}, companies={})",
            self.inner.apriori,
            self.inner.between_cv,
            self.inner.cv,
            self.inner.pattern.ages,
            self.inner.report.n_companies_used,
        )
    }
}

fn meyers_model(name: &str) -> PyResult<MeyersModel> {
    MeyersModel::from_name(name)
        .ok_or_else(|| crate::unknown_name("model", name, MeyersModel::NAMES))
}

/// Meyers' (2019, CAS Monograph 8) log-normal models of a cumulative
/// triangle ``C`` with premium ``P``, for origins ``w = 1..n`` and ages
/// ``d = 1..m``: ``log C[w, d]`` is normal with mean
/// ``log P[w] + logelr + alpha[w] + beta[d] g[w]`` (``alpha[1] = 0``,
/// ``beta[m] = 0``) and sd ``sigma[d]``, decreasing with age. ``g[w]`` is 1
/// for the correlated chain ladder without its correlation between origins
/// (``"crc"``) and ``(1 - gamma)^(w - 1)`` for the changing settlement rate
/// (``"csr"``). The correlated accident year model (``"cay"``) keeps
/// ``g[w] = 1`` and adds ``rho`` times the previous origin's residual at
/// the same age, ``log C[w - 1, d]`` less its mean, to the mean of every
/// origin after the first, so neighbouring origins are correlated; Meyers
/// fits it to incurred data. The priors are those of Meyers' Stan code:
/// ``logelr`` normal with mean ``logelr_mean`` and sd ``logelr_sd``,
/// ``alpha`` and ``beta`` normal around 0, ``gamma`` normal around 0 with
/// sd ``gamma_sd``, ``(1 - rho) / 2`` beta with both shapes ``rho_shape``,
/// and each step of ``sigma[d]^2`` uniform on ``(1 - exp(-1e-5), 1)``.
///
/// The ultimate is the value at the oldest age (no tail): each posterior
/// draw gives each open origin a log-normal ultimate; an origin observed at
/// the oldest age keeps its value; under ``"cay"`` the ultimates are drawn
/// origin by origin, each from the previous origin's residual at the oldest
/// age, observed for a closed origin and simulated otherwise.
/// The reserve is the ultimate less the latest value of the fitted column
/// (an outstanding reserve on paid data, IBNR on incurred data) and can be
/// negative. Every observed cell must be positive, and under ``"cay"``
/// every observed cell after the first origin needs the previous origin's
/// cell at the same age.
///
/// One ``seed`` replays a fit exactly, whatever the number of threads: the
/// chains and the simulations each take a seed derived from it.
///
/// Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS below 100
/// per chain, any divergent transition) do not fail the fit: ``fit``
/// raises one ``UserWarning`` listing them, and they stay in
/// ``MeyersFit.diagnostics``. With the defaults, all 120 fits of a seed
/// sweep on Meyers' group 353 data (paid and incurred, CRC and CSR, 30
/// seeds) passed every threshold, and CAY on incurred passed at 199 of 200
/// seeds (one divergent transition in 8,000 draws at the other).
///
/// Parameters
/// ----------
/// model : {"crc", "csr", "cay"}, default "crc"
///     The correlated chain ladder without its correlation between origins,
///     the changing settlement rate model, or the correlated accident year
///     model.
/// logelr_mean : float, default -0.4
///     Mean of the normal prior on ``logelr``, the log expected loss ratio.
/// logelr_sd : float, default 3.162
///     Standard deviation of the normal prior on ``logelr`` (3.162 is
///     Meyers' rounding of ``sqrt(10)``).
/// origin_sd : float, default 3.162
///     Standard deviation of the normal prior on each origin effect
///     ``alpha[w]``, ``w >= 2``, centred on 0.
/// development_sd : float, default 3.162
///     Standard deviation of the normal prior on each development effect
///     ``beta[d]``, ``d < m``, centred on 0.
/// gamma_sd : float, default 0.05
///     Standard deviation of the normal prior on ``gamma``, centred on 0
///     (``"csr"`` only).
/// rho_shape : float, default 2.0
///     Both shapes of the beta prior on ``(1 - rho) / 2`` (``"cay"``
///     only); Meyers' 2 makes the prior density of ``rho`` proportional to
///     ``1 - rho^2``.
/// chains, tune, draws : int, default 4, 1000, 2000
///     NUTS chains, warm-up draws per chain and kept draws per chain; with
///     1,000 kept draws the late ``sigma[d]`` sometimes miss the R-hat
///     threshold.
/// n_sims : int, default 10000
///     Number of predictive simulations, independent of the number of
///     posterior draws; positive.
/// seed : int, default 0
///     Seed of the chains and of the simulations, from 0 to ``2**64 - 1``.
/// target_accept : float, default 0.95
///     Target mean acceptance rate of the step-size adaptation; at NUTS's
///     usual 0.8 the paid fits of group 353 leave divergent transitions.
/// max_depth : int, default 10
///     Largest NUTS tree depth.
/// dependence : {"independent", "rank_correlation", "synchronized"}, default "independent"
///     How the segments of a multi-segment triangle depend on each other
///     in ``fit_segments`` (``fit`` has one segment and does not use it),
///     as ``BayesOdp``'s: each segment's own posterior and simulations, with
///     seeds derived from ``seed`` and the segment's position; or the
///     segments' simulations reordered as whole rows (Iman–Conover on the
///     segments' total reserves, each segment's ultimates moving with its
///     reserves) to the Spearman matrix ``spearman`` and then put in a
///     random order. ``"synchronized"`` is accepted here, as the bootstraps
///     take it, but ``fit_segments`` raises: a posterior has no residuals to
///     share.
/// spearman : list of list of float, optional
///     With ``dependence="rank_correlation"`` only: Spearman's rho between
///     the segments' total reserves, as ``OdpBootstrap``'s.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero, ``model`` or ``dependence`` is unknown, or
///     ``spearman`` is missing for ``"rank_correlation"``, given for another
///     dependence or not square.
/// OverflowError
///     If an integer setting is negative or too large.
///
/// Examples
/// --------
/// >>> import warnings
/// >>> from prospicio.reserving import Meyers, Triangle
/// >>> paid = [[952, 1529, 2813, 3647, 3724, 3832], [849, 1564, 2202, 2432, 2468],
/// ...         [983, 2211, 2830, 3832], [1657, 2685, 3169], [932, 1940], [1162]]
/// >>> premium = [5812, 4908, 5454, 5165, 5214, 5230]
/// >>> rows = [(1988 + w, 12 * (d + 1), float(v), float(premium[w]))
/// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
/// >>> origin, age, values, prem = map(list, zip(*rows))
/// >>> tri = Triangle.from_long(origin, age, {"paid": values, "premium": prem})
/// >>> with warnings.catch_warnings():
/// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
/// ...     fit = Meyers("csr", tune=300, draws=300, n_sims=2000, seed=1).fit(
/// ...         tri, "paid", "premium")
/// >>> fit.reserves.components()
/// [('1988',), ('1989',), ('1990',), ('1991',), ('1992',), ('1993',)]
/// >>> fit.parameter_names[0], fit.parameter_names[-1]
/// ('logelr', 'sigma[6]')
/// >>> fit.premium[:2]
/// [5812.0, 4908.0]
#[pyclass(name = "Meyers", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMeyers {
    inner: Meyers,
}

#[pymethods]
impl PyMeyers {
    #[new]
    #[pyo3(signature = (
        model = "crc",
        logelr_mean = -0.4,
        logelr_sd = 3.162,
        origin_sd = 3.162,
        development_sd = 3.162,
        gamma_sd = 0.05,
        rho_shape = 2.0,
        chains = 4,
        tune = 1000,
        draws = 2000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.95,
        max_depth = 10,
        dependence = "independent",
        spearman = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &str,
        logelr_mean: f64,
        logelr_sd: f64,
        origin_sd: f64,
        development_sd: f64,
        gamma_sd: f64,
        rho_shape: f64,
        chains: usize,
        tune: usize,
        draws: usize,
        n_sims: usize,
        seed: u64,
        target_accept: f64,
        max_depth: u64,
        dependence: &str,
        spearman: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            inner: Meyers {
                model: meyers_model(model)?,
                logelr_mean,
                logelr_sd,
                origin_sd,
                development_sd,
                gamma_sd,
                rho_shape,
                sampler: BayesSampler {
                    chains,
                    tune,
                    draws,
                    seed: 0,
                    target_accept,
                    max_depth,
                },
                n_sims,
                seed,
                dependence: segment_dependence(dependence, spearman)?,
            },
        })
    }

    /// The model: ``"crc"``, ``"csr"`` or ``"cay"``.
    #[getter]
    fn model(&self) -> &'static str {
        self.inner.model.name()
    }

    /// Mean of the prior on ``logelr``.
    #[getter]
    fn logelr_mean(&self) -> f64 {
        self.inner.logelr_mean
    }

    /// Standard deviation of the prior on ``logelr``.
    #[getter]
    fn logelr_sd(&self) -> f64 {
        self.inner.logelr_sd
    }

    /// Standard deviation of each origin effect's prior.
    #[getter]
    fn origin_sd(&self) -> f64 {
        self.inner.origin_sd
    }

    /// Standard deviation of each development effect's prior.
    #[getter]
    fn development_sd(&self) -> f64 {
        self.inner.development_sd
    }

    /// Standard deviation of the prior on ``gamma`` (``"csr"`` only).
    #[getter]
    fn gamma_sd(&self) -> f64 {
        self.inner.gamma_sd
    }

    /// Both shapes of the beta prior on ``(1 - rho) / 2`` (``"cay"`` only).
    #[getter]
    fn rho_shape(&self) -> f64 {
        self.inner.rho_shape
    }

    /// Number of NUTS chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.sampler.chains
    }

    /// Warm-up draws per chain.
    #[getter]
    fn tune(&self) -> usize {
        self.inner.sampler.tune
    }

    /// Kept draws per chain.
    #[getter]
    fn draws(&self) -> usize {
        self.inner.sampler.draws
    }

    /// Number of predictive simulations.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims
    }

    /// Seed of the chains and of the simulations.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Target mean acceptance rate.
    #[getter]
    fn target_accept(&self) -> f64 {
        self.inner.sampler.target_accept
    }

    /// Largest NUTS tree depth.
    #[getter]
    fn max_depth(&self) -> u64 {
        self.inner.sampler.max_depth
    }

    /// How segments depend on each other in ``fit_segments``:
    /// ``"independent"``, ``"rank_correlation"`` or ``"synchronized"``.
    #[getter]
    fn dependence(&self) -> &'static str {
        dependence_name(&self.inner.dependence)
    }

    /// The Spearman matrix of ``dependence="rank_correlation"``; ``None``
    /// otherwise.
    #[getter]
    fn spearman(&self) -> Option<Vec<Vec<f64>>> {
        dependence_spearman(&self.inner.dependence)
    }

    /// Samples the posterior of one measure column of a single-segment
    /// cumulative triangle, with each origin's premium from another column
    /// (its latest value, as ``CapeCod`` reads exposure), and simulates the
    /// ultimates and reserves. Every origin must be observed from the first
    /// age up to its latest, every observed cell must be positive, and
    /// there must be more observed cells than parameters. Raises one
    /// ``UserWarning`` listing the failed diagnostics, if any.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///     The losses: paid, or case incurred.
    /// premium : str
    ///     The premium of each origin.
    ///
    /// Returns
    /// -------
    /// MeyersFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a column is missing, the triangle has several segments or an
    ///     origin has a gap; if a premium is missing or not positive; if a
    ///     prior standard deviation or ``rho_shape`` is not positive and
    ///     finite, ``logelr_mean`` is not finite or a sampler setting is
    ///     invalid; if an observed cell is not positive; if there are no more
    ///     observed cells than parameters; or, under ``"cay"``, if a cell
    ///     lacks the previous origin's cell at the same age.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        premium: &str,
    ) -> PyResult<PyMeyersFit> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py.detach(|| model.fit(tri, column, premium)).map_err(err)?;
        warn_diagnostics(py, "Meyers", &inner.diagnostics.warnings)?;
        Ok(PyMeyersFit { inner })
    }

    /// Samples the posterior of one measure column in every segment of a
    /// cumulative triangle, each segment with its own posterior (the same
    /// model and settings, sampled independently) and its premium from its
    /// own ``premium`` column as ``fit`` reads it, into one joint
    /// distribution of the ultimates and one of the reserves, the segments
    /// depending on each other as ``dependence`` says. Segment ``k`` (in
    /// index order) takes seeds derived from ``seed`` and ``k`` for its
    /// chains and its simulations, so independent segments share no random
    /// numbers, and a segment's draws differ from ``fit`` of that segment
    /// alone. Raises one ``UserWarning`` listing the failed diagnostics of
    /// every segment, each message prefixed with its segment's label.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// column : str
    ///     The losses: paid, or case incurred.
    /// premium : str
    ///     The premium of each origin.
    ///
    /// Returns
    /// -------
    /// MeyersFits
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit`` for any segment (the message names the segment); if
    ///     ``dependence`` is ``"synchronized"`` (before any sampling); or if
    ///     ``spearman`` does not fit the segments.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import Meyers, Triangle
    /// >>> paid = [[952, 1529, 2813, 3647, 3724], [849, 1564, 2202, 2432],
    /// ...         [983, 2211, 2830], [1657, 2685], [932]]
    /// >>> premium = [5812, 4908, 5454, 5165, 5214]
    /// >>> rows = [(lob, 1988 + w, 12 * (d + 1), k * v, k * premium[w])
    /// ...         for lob, k in [("Auto", 1.0), ("Home", 2.0)]
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> lob, origin, age, values, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(
    /// ...     origin, age, {"paid": values, "premium": prem}, keys={"lob": lob})
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     fits = Meyers(tune=200, draws=200, n_sims=1000).fit_segments(
    /// ...         tri, "paid", "premium")
    /// >>> fits.index, fits.reserves.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    /// >>> fits.segment(lob="Home").premium[:2]
    /// [11624.0, 9816.0]
    fn fit_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        premium: &str,
    ) -> PyResult<PyMeyersFits> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.fit_segments(tri, column, premium))
            .map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.diagnostics);
        warn_diagnostics(py, "Meyers", &warnings)?;
        Ok(PyMeyersFits { inner })
    }

    /// The one-year view of one measure column of a single-segment
    /// cumulative triangle with premium: the claims development result over
    /// the coming year, by Pareto smoothed importance sampling of the
    /// posterior (decision 19 of ``docs/design/reserving-bayes.md``). The
    /// posterior and the lifetime ultimates and reserves are ``fit``'s for
    /// the same settings. Each simulation picks a posterior draw, draws the
    /// cumulative values of the year's cells from it (the cells valued in
    /// the next twelve months, as in ``OdpBootstrap.one_year``), reweights
    /// every posterior draw by the likelihood of those values, and smooths
    /// the weights (one Pareto ``k`` per simulation). Origin ``w``'s claims
    /// development result is
    ///
    /// ``CDR[w] = E[U[w] | D0] - E[U[w] | D0, C]``
    ///
    /// with ``U`` the ultimate (the value at the oldest age) and each
    /// expectation the (weighted) posterior mean of the ultimate's expected
    /// value given a draw; an origin the year takes to the oldest age has
    /// its value there instead of the second term. The mean is about zero.
    /// A simulation whose ``k`` is above ``k_threshold`` is flagged and
    /// handled as ``refit`` says: by default moment-matched, and refitted
    /// with NUTS only if its ``k`` stays above the threshold. On Meyers'
    /// group 353 nearly every simulation is flagged and matching rescues
    /// about nine in ten, so 10,000 simulations take 4 to 7 minutes on 16
    /// threads in a release build, and ``"refit_flagged"`` about 6.7 times
    /// as long. The view therefore runs at most 2,000 simulations by
    /// default (about a fifth of that time), whatever the model's
    /// ``n_sims`` is above that; ``n_sims`` here sets the count. Raises one ``UserWarning``
    /// listing the failed diagnostics of the fit, and another if any refit
    /// failed its diagnostics.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///     The losses: paid, or case incurred.
    /// premium : str
    ///     The premium of each origin, as in ``fit``.
    /// k_threshold : float, default 0.7
    ///     The Pareto ``k`` above which a simulation's importance weights
    ///     are not trusted.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, default "moment_match_then_refit"
    ///     What happens to a flagged simulation, as in ``BayesOdp.one_year``.
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``.
    /// n_sims : int, optional
    ///     The number of one-year simulations. By default this model's
    ///     ``n_sims``, at most 2,000: nearly every simulation of these
    ///     models is flagged and costs a moment match or a refit. The
    ///     lifetime view in ``fit`` keeps ``n_sims``; a smaller count
    ///     gives the first simulations of a larger one.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFit
    ///     With ``fit`` a ``MeyersFit``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit``; if ``k_threshold`` is NaN or ``refit`` is unknown.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import Meyers, Triangle
    /// >>> paid = [[952, 1529, 2813, 3647, 3724, 3832], [849, 1564, 2202, 2432, 2468],
    /// ...         [983, 2211, 2830, 3832], [1657, 2685, 3169], [932, 1940], [1162]]
    /// >>> premium = [5812, 4908, 5454, 5165, 5214, 5230]
    /// >>> rows = [(1988 + w, 12 * (d + 1), v, premium[w])
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> origin, age, values, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(origin, age, {"paid": values, "premium": prem})
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     view = Meyers(tune=300, draws=300, n_sims=1000, seed=1).one_year(
    /// ...         tri, "paid", "premium", refit="reweight_only")
    /// >>> type(view.fit).__name__, view.cdr.components()[:2]
    /// ('MeyersFit', [('1988',), ('1989',)])
    /// >>> view.cdr.variance() < view.fit.reserves.variance()
    /// True
    #[pyo3(signature = (triangle, column, premium, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        premium: &str,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFit> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year(tri, column, premium, &settings))
            .map_err(err)?;
        warn_diagnostics(py, "Meyers", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "Meyers",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        Ok(PyBayesOneYearFit {
            inner: BayesOneYearOf::Meyers(Box::new(inner)),
        })
    }

    /// The one-year view of one measure column in every segment of a
    /// cumulative triangle with premium: each segment's posterior and
    /// simulations as in ``fit_segments`` (segment ``k``'s own seeds), its
    /// view as in ``one_year``, into one joint claims development result,
    /// the segments depending on each other as ``dependence`` says:
    /// ``"independent"``, or ``"rank_correlation"`` by Iman–Conover on the
    /// segments' total claims development results. Raises one
    /// ``UserWarning`` listing the failed diagnostics of every segment's
    /// fit, and another if any refit failed its diagnostics, each message
    /// prefixed with its segment's label.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// column : str
    /// premium : str
    /// k_threshold : float, default 0.7
    ///     As in ``one_year``.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, default "moment_match_then_refit"
    ///     As in ``one_year``.
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``, for each segment: ``k`` segments
    ///     refit at most ``k`` times this many.
    /// n_sims : int, optional
    ///     As in ``one_year``.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFits
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``one_year`` and ``fit_segments``, including
    ///     ``dependence="synchronized"``.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import Meyers, Triangle
    /// >>> paid = [[952, 1529, 2813, 3647, 3724], [849, 1564, 2202, 2432],
    /// ...         [983, 2211, 2830], [1657, 2685], [932]]
    /// >>> premium = [5812, 4908, 5454, 5165, 5214]
    /// >>> rows = [(lob, 1988 + w, 12 * (d + 1), k * v, k * premium[w])
    /// ...         for lob, k in [("Auto", 1.0), ("Home", 2.0)]
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> lob, origin, age, values, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(
    /// ...     origin, age, {"paid": values, "premium": prem}, keys={"lob": lob})
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     views = Meyers(tune=200, draws=200, n_sims=500).one_year_segments(
    /// ...         tri, "paid", "premium", refit="reweight_only")
    /// >>> views.index, views.cdr.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    /// >>> views.segment(lob="Home").premium[:2]
    /// [11624.0, 9816.0]
    #[pyo3(signature = (triangle, column, premium, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        premium: &str,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFits> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year_segments(tri, column, premium, &settings))
            .map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.fit.diagnostics);
        warn_diagnostics(py, "Meyers", &warnings)?;
        warn_refits(py, "Meyers", &segment_refits(&inner.segments))?;
        Ok(PyBayesOneYearFits {
            inner: BayesOneYearFitsOf::Meyers(inner),
        })
    }

    fn __repr__(&self) -> String {
        let (m, s) = (&self.inner, &self.inner.sampler);
        format!(
            "Meyers(model={:?}, logelr_mean={:?}, logelr_sd={:?}, origin_sd={:?}, \
             development_sd={:?}, gamma_sd={:?}, rho_shape={:?}, chains={}, tune={}, \
             draws={}, n_sims={}, seed={}{})",
            m.model.name(),
            m.logelr_mean,
            m.logelr_sd,
            m.origin_sd,
            m.development_sd,
            m.gamma_sd,
            m.rho_shape,
            s.chains,
            s.tune,
            s.draws,
            m.n_sims,
            m.seed,
            dependence_repr(&m.dependence),
        )
    }
}

/// The posterior predictive of a new underwriting year, from
/// ``BayesBfNormalisedFit.new_year`` or ``MeyersFit.new_year``.
///
/// Row ``i`` of ``loss_ratio``, ``ultimate`` and every pattern of
/// ``patterns`` is simulation ``i``.
#[pyclass(name = "NewYear", module = "prospicio.reserving", frozen)]
pub(crate) struct PyNewYear {
    inner: NewYear,
}

#[pymethods]
impl PyNewYear {
    /// The new origin period, the one after the triangle's newest.
    #[getter]
    fn origin(&self) -> String {
        self.inner.origin.to_string()
    }

    /// The new year's premium.
    #[getter]
    fn premium(&self) -> f64 {
        self.inner.premium
    }

    /// The ultimate loss ratio of the new year: dimension ``"origin"``,
    /// one component, one row per simulation.
    #[getter]
    fn loss_ratio(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.loss_ratio.clone(),
        }
    }

    /// The ultimate loss of the new year, ``loss_ratio`` times
    /// ``premium``, with the same layout.
    #[getter]
    fn ultimate(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.ultimate.clone(),
        }
    }

    /// The development pattern of each simulation's posterior draw; its
    /// ``monthly`` is the new year's emergence by month.
    ///
    /// Returns
    /// -------
    /// DevelopmentPatterns
    #[getter]
    fn patterns(&self) -> PyDevelopmentPatterns {
        PyDevelopmentPatterns {
            inner: self.inner.patterns.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NewYear(origin={}, premium={}, n_sims={})",
            self.inner.origin,
            self.inner.premium,
            self.inner.patterns.n_sims()
        )
    }
}

/// A fitted ``Meyers``: the posterior of ``logelr``, ``alpha[2..n]``,
/// ``beta[1..m-1]``, ``gamma`` (``"csr"`` only) or ``rho`` (``"cay"``
/// only), and ``sigma[1..m]``, its
/// MCMC diagnostics, and the posterior predictive distributions of the
/// ultimates and reserves by origin. The frames' ``latest``, ``ultimate``
/// and ``reserve`` are the volume-weighted chain ladder's
/// (``chain_ladder``); ``mean`` and ``std_dev`` are the posterior
/// predictive reserve's, and ``ultimate_mean`` and ``ultimate_std_dev``
/// the posterior predictive ultimate's.
#[pyclass(name = "MeyersFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMeyersFit {
    inner: MeyersFit,
}

#[pymethods]
impl PyMeyersFit {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// Meyers' models simulate no path, so each reserve is spread over its
    /// simulation's pattern.
    ///
    /// Parameters
    /// ----------
    /// basis : {"paid", "incurred"}
    ///     What the fit's column is, stated by the caller (no default), so
    ///     that a fit on incurred claims is not discounted by its reporting
    ///     pattern by mistake. ``"paid"`` pays each simulation's reserve
    ///     along that simulation's own pattern. ``"incurred"`` pays each
    ///     reserve (the IBNR) plus the origin's case reserve (its latest
    ///     incurred less ``latest_paid``) along ``paid_pattern``.
    /// rate : float, optional
    ///     A flat annual effective rate. Give ``rate`` or ``curve``.
    /// curve : DiscountCurve, optional
    ///     Spot rates by maturity.
    /// timing : {"mid", "end"}, default "mid"
    ///     Payments at the middle or the end of each development period.
    /// paid_pattern : list of float, optional
    ///     For ``basis="incurred"``: the cumulative paid share at each of
    ///     the fit's pattern ages (with the tail's age when the fit has a
    ///     tail), ending at 1.
    /// latest_paid : list of float, optional
    ///     For ``basis="incurred"``: each origin's latest paid, one per
    ///     component of ``reserves`` in their order.
    ///
    /// Returns
    /// -------
    /// DiscountedReserves
    #[pyo3(signature = (
        *,
        basis,
        rate = None,
        curve = None,
        timing = "mid",
        paid_pattern = None,
        latest_paid = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn discounted(
        &self,
        py: Python<'_>,
        basis: &str,
        rate: Option<f64>,
        curve: Option<PyRef<'_, PyDiscountCurve>>,
        timing: &str,
        paid_pattern: Option<Vec<f64>>,
        latest_paid: Option<Vec<f64>>,
    ) -> PyResult<PyDiscountedReserves> {
        let args = DiscountArgs {
            basis,
            rate,
            curve,
            timing,
            paid_pattern,
            latest_paid,
        };
        discounted(py, &self.inner, args)
    }

    /// The posterior predictive of a new underwriting year, the origin
    /// after the triangle's newest, for pricing a year not yet written.
    ///
    /// Each simulation takes a posterior draw and from it the new origin's
    /// level ``logelr + alpha``, then its ultimate, log-normal with sd
    /// ``sigma[m]`` (in CAY, plus ``rho`` times the newest origin's
    /// residual). Meyers' models have no hyperparameter for the origin
    /// effects, so ``level`` says how the new one is drawn.
    ///
    /// Parameters
    /// ----------
    /// premium : float
    ///     The new year's premium, finite and positive. The loss ratio does
    ///     not depend on it.
    /// n_sims : int
    ///     Number of simulations.
    /// seed : int
    ///     Seed of the simulations.
    /// level : str
    ///     ``"exchangeable"`` (the default): the new origin is like the
    ///     triangle's, its level a Student t draw from a normal fitted to the
    ///     draw's origin levels (needs three origins). ``"prior"``: Meyers'
    ///     own prior ``alpha ~ N(0, origin_sd**2)``, too wide to price on at
    ///     his ``origin_sd`` of 3.162.
    ///
    /// Returns
    /// -------
    /// NewYear
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``premium`` is not positive, ``n_sims`` is 0, ``level`` is
    ///     unknown, or the triangle has fewer than three origins under
    ///     ``"exchangeable"``.
    #[pyo3(signature = (premium, n_sims = 10_000, seed = 0, level = "exchangeable"))]
    fn new_year(
        &self,
        py: Python<'_>,
        premium: f64,
        n_sims: usize,
        seed: u64,
        level: &str,
    ) -> PyResult<PyNewYear> {
        let level = match level {
            "exchangeable" => NewOriginLevel::Exchangeable,
            "prior" => NewOriginLevel::Prior,
            other => {
                return Err(PyValueError::new_err(format!(
                    "level must be \"exchangeable\" or \"prior\", not {other:?}"
                )));
            }
        };
        let fit = &self.inner;
        let inner = py
            .detach(|| fit.new_year_with_level(premium, level, n_sims, seed))
            .map_err(err)?;
        Ok(PyNewYear { inner })
    }

    /// The model: ``"crc"``, ``"csr"`` or ``"cay"``.
    #[getter]
    fn model(&self) -> &'static str {
        self.inner.model().name()
    }

    /// The volume-weighted chain ladder of the fitted column, whose latest
    /// values the reserves are measured from.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.inner.chain_ladder.clone()],
            },
        }
    }

    /// Origin period of each component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .chain_ladder
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.chain_ladder.development.development.clone()
    }

    /// The premium of each origin.
    #[getter]
    fn premium(&self) -> Vec<f64> {
        self.inner.premium.clone()
    }

    /// Each simulation's development pattern, to pair with the
    /// simulation's loss ratio in the cash-flow engine: the shares
    /// ``exp(beta_d g)`` of the posterior draw the simulation took, the
    /// median value at each age over the median at the oldest age of an
    /// origin after the newest (``g`` its speed-up ``(1 - gamma)^n`` in CSR,
    /// 1 otherwise). No tail: each pattern ends at 1 at the oldest age.
    /// Nothing keeps ``beta_d`` rising or below 0, so a share can fall or
    /// pass 1 (on Meyers' group 353 data it falls in nearly every
    /// simulation), which ``monthly`` refuses unless
    /// ``monotone="running_max"``.
    ///
    /// Row ``i`` of every pattern is simulation ``i`` of ``reserves``,
    /// ``ultimates`` and ``ultimate_draws``.
    ///
    /// Returns
    /// -------
    /// DevelopmentPatterns
    #[getter]
    fn patterns(&self) -> PyDevelopmentPatterns {
        PyDevelopmentPatterns {
            inner: self.inner.patterns.clone(),
        }
    }

    /// Joint posterior predictive distribution of the ultimate (the value
    /// at the oldest age) by origin: dimension ``"origin"``, one component
    /// per origin, one row per simulation. An origin observed at the
    /// oldest age keeps its value.
    #[getter]
    fn ultimates(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.ultimates.clone(),
        }
    }

    /// The ultimates less the latest values, simulation by simulation,
    /// laid out as ``ultimates`` and as ``OdpBootstrapFit.reserves``.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// Each simulation's ultimate, for a tower, a deal book or the cash-flow
    /// engine: index ``i`` is simulation ``i`` for every origin and every
    /// call, the same as row ``i`` of ``reserves`` and ``ultimates``, and
    /// the same again when the fit is run with the same seed.
    ///
    /// The ultimate is the origin's latest value plus its reserve in that
    /// simulation, which is the simulated value at the oldest age itself (``ultimates``).
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     The component key (``(2021,)``) or the origin alone; an origin period by its label
    ///     (``"2021"``, ``2021``, ``"2021Q3"``). ``None`` (the default) gives
    ///     the total over every component, summed simulation by simulation.
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (origin = None))]
    fn ultimate_draws(&self, origin: Option<OriginArg>) -> PyResult<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// Each simulation's ultimate over ``premium``: the loss ratio a
    /// contract on the loss ratio reads, in the order of
    /// ``ultimate_draws``. Multiply by the premium for
    /// ``Tower.apply_annual``.
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     As in ``ultimate_draws``; ``None`` for the total.
    /// premium : float
    ///     The subject premium of the origin (of every origin for the
    ///     total), finite and positive.
    ///
    /// Returns
    /// -------
    /// list of float
    fn loss_ratio_draws(&self, origin: Option<OriginArg>, premium: f64) -> PyResult<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// MCMC diagnostics: a dict with ``max_rhat``, ``min_ess_bulk``,
    /// ``min_ess_tail``, ``divergences`` and ``warnings`` (one message per
    /// failed threshold; empty when every threshold holds).
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = &self.inner.diagnostics;
        let out = PyDict::new(py);
        out.set_item("max_rhat", d.max_rhat)?;
        out.set_item("min_ess_bulk", d.min_ess_bulk)?;
        out.set_item("min_ess_tail", d.min_ess_tail)?;
        out.set_item("divergences", d.divergences)?;
        out.set_item("warnings", d.warnings.clone())?;
        Ok(out)
    }

    /// Parameter names: ``logelr``, ``alpha[2]`` to ``alpha[n]``,
    /// ``beta[1]`` to ``beta[m-1]``, ``gamma`` (``"csr"`` only) or ``rho``
    /// (``"cay"`` only), and ``sigma[1]`` to ``sigma[m]``.
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, one row per draw (chain by chain), one column per
    /// parameter in the order of ``parameter_names``.
    #[getter]
    fn draws(&self) -> Vec<Vec<f64>> {
        let p = self.inner.posterior.dim().max(1);
        self.inner
            .posterior
            .values()
            .chunks(p)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// Number of chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.posterior.chains()
    }

    /// Posterior summary, one row per parameter: ``parameter``, ``mean``,
    /// ``sd``, ``q05``, ``q50``, ``q95``, ``rhat``, ``ess_bulk`` and
    /// ``ess_tail``, the fields of ``BayesGlmFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fit = &self.inner;
        let s = py.detach(|| fit.summary()).map_err(err)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        let out = PyDict::new(py);
        out.set_item(
            "parameter",
            s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
        )?;
        out.set_item("mean", col(|p| p.mean))?;
        out.set_item("sd", col(|p| p.sd))?;
        out.set_item("q05", col(|p| p.q05))?;
        out.set_item("q50", col(|p| p.q50))?;
        out.set_item("q95", col(|p| p.q95))?;
        out.set_item("rhat", col(|p| p.rhat))?;
        out.set_item("ess_bulk", col(|p| p.ess_bulk))?;
        out.set_item("ess_tail", col(|p| p.ess_tail))?;
        py.import("pandas")?.call_method1("DataFrame", (out,))
    }

    /// PSIS-LOO of the observed cells, with each cell's relative
    /// efficiency from the chains, as ``BayesGlmFit.loo``. The pointwise
    /// term is Meyers', the normal density of ``log C[w, d]``, so the
    /// result is on the log scale and comparable only between ``"crc"``,
    /// ``"csr"`` and ``"cay"`` fits of the same column (Meyers 2019,
    /// Tables 6.4, 7.3 and 8.3). Under ``"cay"`` each cell's density is
    /// conditional on the previous origin's cell, as in Meyers' code.
    ///
    /// Returns
    /// -------
    /// Elpd
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let fit = &self.inner;
        let l = py.detach(|| fit.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l))
    }

    /// One row per origin: ``origin``, the chain ladder's ``latest``,
    /// ``ultimate`` and ``reserve``, the ``premium``, ``ulr`` (the chain
    /// ladder's ultimate over the premium), the ``mean`` and
    /// ``std_dev`` of the posterior predictive reserve, the
    /// ``ultimate_mean`` and ``ultimate_std_dev`` of the posterior
    /// predictive ultimate, and ``ulr_mean``, the ``ultimate_mean`` over
    /// the premium. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the chain ladder's totals, the total ``premium`` and
    /// ``ulr``, the ``mean`` and ``std_dev`` of the posterior predictive
    /// total reserve, the ``ultimate_mean`` and ``ultimate_std_dev`` of the
    /// total ultimate, and ``ulr_mean``, the total ``ultimate_mean`` over
    /// the total premium. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The chain ladder's development factors, one row per age, as
    /// ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    fn __repr__(&self) -> String {
        format!(
            "MeyersFit(model={:?}, origins={}, draws={}, n_sims={}, warnings={})",
            self.inner.model().name(),
            self.inner.chain_ladder.origins.len(),
            self.inner.posterior.n_draws(),
            self.inner.reserves.n_sims(),
            self.inner.diagnostics.warnings.len(),
        )
    }
}

/// ``Meyers.fit_segments``: each segment's own posterior and the joint
/// posterior predictive distributions of the ultimates and the reserves,
/// laid out as ``BayesOdpFits``.
///
/// ``ultimates`` and ``reserves`` have the triangle's keys and ``"origin"``
/// as dimensions, so ``reserves.aggregate(["lob"])`` keeps the dependence
/// between segments. Per-origin lists (``origins``, ``premium``) run over
/// the origins of each segment in turn, like the rows of ``to_frame()``
/// and the components of ``reserves``. The posterior of one segment
/// (``diagnostics``, ``parameter_names``, ``draws``, ``chains``,
/// ``summary()`` and ``loo()``) needs a single-segment fit: for several
/// segments use ``segment(...)`` or ``diagnostics_frame()``.
#[pyclass(name = "MeyersFits", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMeyersFits {
    inner: MeyersFits,
}

impl PyMeyersFits {
    fn one(&self, field: &str, instead: &str) -> PyResult<&MeyersSegment> {
        single(&self.inner.segments, field, instead)
    }
}

#[pymethods]
impl PyMeyersFits {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// Meyers' models simulate no path, so each reserve is spread over its
    /// simulation's pattern, each segment by its own.
    ///
    /// Parameters
    /// ----------
    /// basis : {"paid", "incurred"}
    ///     What the fit's column is, stated by the caller (no default), so
    ///     that a fit on incurred claims is not discounted by its reporting
    ///     pattern by mistake. ``"paid"`` pays each simulation's reserve
    ///     along that simulation's own pattern. ``"incurred"`` pays each
    ///     reserve (the IBNR) plus the origin's case reserve (its latest
    ///     incurred less ``latest_paid``) along ``paid_pattern``.
    /// rate : float, optional
    ///     A flat annual effective rate. Give ``rate`` or ``curve``.
    /// curve : DiscountCurve, optional
    ///     Spot rates by maturity.
    /// timing : {"mid", "end"}, default "mid"
    ///     Payments at the middle or the end of each development period.
    /// paid_pattern : list of float, optional
    ///     For ``basis="incurred"``: the cumulative paid share at each of
    ///     the fit's pattern ages (with the tail's age when the fit has a
    ///     tail), ending at 1.
    /// latest_paid : list of float, optional
    ///     For ``basis="incurred"``: each origin's latest paid, one per
    ///     component of ``reserves`` in their order.
    ///
    /// Returns
    /// -------
    /// DiscountedReserves
    #[pyo3(signature = (
        *,
        basis,
        rate = None,
        curve = None,
        timing = "mid",
        paid_pattern = None,
        latest_paid = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn discounted(
        &self,
        py: Python<'_>,
        basis: &str,
        rate: Option<f64>,
        curve: Option<PyRef<'_, PyDiscountCurve>>,
        timing: &str,
        paid_pattern: Option<Vec<f64>>,
        latest_paid: Option<Vec<f64>>,
    ) -> PyResult<PyDiscountedReserves> {
        let args = DiscountArgs {
            basis,
            rate,
            curve,
            timing,
            paid_pattern,
            latest_paid,
        };
        discounted(py, &self.inner, args)
    }

    /// The model: ``"crc"``, ``"csr"`` or ``"cay"``.
    #[getter]
    fn model(&self) -> &'static str {
        self.inner.segments.fits[0].model().name()
    }

    /// The volume-weighted chain ladder of every segment's fitted column,
    /// whose latest values the reserves are measured from.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.segments.map(|s| s.chain_ladder.clone()),
        }
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.segments.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner.segments)
    }

    /// Origin period of each per-origin value and component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        origin_labels(&self.inner.segments)
    }

    /// Development ages in months of the first segment.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.segments.fits[0]
            .chain_ladder
            .development
            .development
            .clone()
    }

    /// The premium of each segment's origins, in the order of ``origins``.
    #[getter]
    fn premium(&self) -> Vec<f64> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.premium.iter().copied())
            .collect()
    }

    /// Each simulation's development pattern, to pair with the
    /// simulation's loss ratio in the cash-flow engine: the shares
    /// ``exp(beta_d g)`` of the posterior draw the simulation took, the
    /// median value at each age over the median at the oldest age of an
    /// origin after the newest (``g`` its speed-up ``(1 - gamma)^n`` in CSR,
    /// 1 otherwise). No tail: each pattern ends at 1 at the oldest age.
    /// Nothing keeps ``beta_d`` rising or below 0, so a share can fall or
    /// pass 1 (on Meyers' group 353 data it falls in nearly every
    /// simulation), which ``monthly`` refuses unless
    /// ``monotone="running_max"``.
    ///
    /// Row ``i`` of every pattern is simulation ``i`` of ``reserves``,
    /// ``ultimates`` and ``ultimate_draws``.
    ///
    /// Returns
    /// -------
    /// DevelopmentPatterns
    #[getter]
    fn patterns(&self) -> PyDevelopmentPatterns {
        PyDevelopmentPatterns {
            inner: self.inner.patterns.clone(),
        }
    }

    /// Joint posterior predictive distribution of the ultimate (the value
    /// at the oldest age) by segment and origin: the triangle's keys and
    /// ``"origin"`` are its dimensions, one component per segment and
    /// origin, one row per simulation.
    #[getter]
    fn ultimates(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.ultimates.clone(),
        }
    }

    /// The ultimates less the latest values, simulation by simulation,
    /// laid out as ``ultimates``.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// Each simulation's ultimate, for a tower, a deal book or the cash-flow
    /// engine: index ``i`` is simulation ``i`` for every origin and every
    /// call, the same as row ``i`` of ``reserves`` and ``ultimates``, and
    /// the same again when the fit is run with the same seed.
    ///
    /// The ultimate is the origin's latest value plus its reserve in that
    /// simulation, which is the simulated value at the oldest age itself (``ultimates``).
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     The component key, the segment's key values then the origin (``("Auto", 2021)``); an origin period by its label
    ///     (``"2021"``, ``2021``, ``"2021Q3"``). ``None`` (the default) gives
    ///     the total over every component, summed simulation by simulation.
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (origin = None))]
    fn ultimate_draws(&self, origin: Option<OriginArg>) -> PyResult<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// Each simulation's ultimate over ``premium``: the loss ratio a
    /// contract on the loss ratio reads, in the order of
    /// ``ultimate_draws``. Multiply by the premium for
    /// ``Tower.apply_annual``.
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     As in ``ultimate_draws``; ``None`` for the total.
    /// premium : float
    ///     The subject premium of the origin (of every origin for the
    ///     total), finite and positive.
    ///
    /// Returns
    /// -------
    /// list of float
    fn loss_ratio_draws(&self, origin: Option<OriginArg>, premium: f64) -> PyResult<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// MCMC diagnostics of a single-segment fit, as
    /// ``MeyersFit.diagnostics``.
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        diagnostics_dict(
            py,
            &self.one("diagnostics", "diagnostics_frame()")?.diagnostics,
        )
    }

    /// Parameter names of a single-segment fit, as
    /// ``MeyersFit.parameter_names``.
    #[getter]
    fn parameter_names(&self) -> PyResult<Vec<String>> {
        Ok(self.one("parameter_names", "")?.posterior.names().to_vec())
    }

    /// Posterior draws of a single-segment fit, as ``MeyersFit.draws``.
    #[getter]
    fn draws(&self) -> PyResult<Vec<Vec<f64>>> {
        let posterior = &self.one("draws", "")?.posterior;
        let p = posterior.dim().max(1);
        Ok(posterior.values().chunks(p).map(<[f64]>::to_vec).collect())
    }

    /// Number of chains of a single-segment fit.
    #[getter]
    fn chains(&self) -> PyResult<usize> {
        Ok(self.one("chains", "")?.posterior.chains())
    }

    /// Posterior summary of a single-segment fit, as
    /// ``MeyersFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let one = self.one("summary()", "")?;
        let s = py.detach(|| one.summary()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// PSIS-LOO of a single-segment fit's observed cells on Meyers' log
    /// scale, as ``MeyersFit.loo``.
    ///
    /// Returns
    /// -------
    /// Elpd
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the fit has several segments.
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let one = self.one("loo()", "")?;
        let l = py.detach(|| one.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l))
    }

    /// One row per segment: the key columns, then ``max_rhat``,
    /// ``min_ess_bulk``, ``min_ess_tail``, ``divergences`` and
    /// ``warnings`` (the segment's messages joined by ``"; "``, empty when
    /// every threshold holds). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn diagnostics_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        segment_diagnostics_frame(py, &self.inner.segments, |s| &s.diagnostics)
    }

    /// One row per segment and origin: the key columns, ``origin``, the
    /// chain ladder's ``latest``, ``ultimate`` and ``reserve``, the
    /// ``premium``, ``ulr`` (the chain ladder's ultimate over the premium),
    /// the ``mean`` and ``std_dev`` of the posterior predictive reserve,
    /// the ``ultimate_mean`` and ``ultimate_std_dev`` of the posterior
    /// predictive ultimate, and ``ulr_mean``, the ``ultimate_mean`` over the
    /// premium. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the chain ladder's totals,
    /// the total ``premium`` and ``ulr``, the ``mean`` and ``std_dev`` of
    /// the segment's posterior predictive total reserve, and the
    /// ``ultimate_mean`` and ``ultimate_std_dev`` of its total ultimate,
    /// and ``ulr_mean``, the total ``ultimate_mean`` over the total premium.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The chain ladders' development factors, one row per segment and
    /// age, as ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint ultimates and
    /// reserves (same dimensions).
    ///
    /// Returns
    /// -------
    /// MeyersFits
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        Ok(Self {
            inner: self.inner.segment(&keys).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        let segments = &self.inner.segments;
        let warnings: usize = segments
            .fits
            .iter()
            .map(|f| f.diagnostics.warnings.len())
            .sum();
        format!(
            "MeyersFits(model={:?}, {}origins={}, n_sims={}, warnings={warnings})",
            self.model(),
            segments_prefix(segments),
            self.origins().len(),
            self.inner.reserves.n_sims(),
        )
    }
}

// ---------------------------------------------------------------------------
// Gesmann and Morris' compartmental model (docs/design/reserving-compartmental.md)
// ---------------------------------------------------------------------------

/// `(median, sd)` of a log-normal prior on a positive parameter.
fn positive_prior(name: &str, value: (f64, f64)) -> PyResult<PositivePrior> {
    let (median, sd) = value;
    if !(median.is_finite() && median > 0.0 && sd.is_finite() && sd > 0.0) {
        return Err(PyValueError::new_err(format!(
            "{name} must be (median, sd) with both finite and positive, got ({median}, {sd})"
        )));
    }
    Ok(PositivePrior::LogNormal { median, sd })
}

fn positive_prior_tuple(prior: PositivePrior) -> (f64, f64) {
    let PositivePrior::LogNormal { median, sd } = prior;
    (median, sd)
}

/// A half prior from ``("student_t", nu, scale)`` or ``("normal", scale)``.
fn half_prior(name: &str, value: &Bound<'_, PyTuple>) -> PyResult<HalfPrior> {
    let bad = || {
        PyValueError::new_err(format!(
            "{name} must be (\"student_t\", nu, scale) or (\"normal\", scale)"
        ))
    };
    let family: String = value.get_item(0).map_err(|_| bad())?.extract()?;
    let num = |i: usize| -> PyResult<f64> { value.get_item(i).map_err(|_| bad())?.extract() };
    match (family.to_ascii_lowercase().as_str(), value.len()) {
        ("student_t", 3) => Ok(HalfPrior::StudentT {
            nu: num(1)?,
            scale: num(2)?,
        }),
        ("normal", 2) => Ok(HalfPrior::Normal { scale: num(1)? }),
        _ => Err(bad()),
    }
}

fn half_prior_tuple<'py>(py: Python<'py>, prior: HalfPrior) -> PyResult<Bound<'py, PyTuple>> {
    match prior {
        HalfPrior::StudentT { nu, scale } => ("student_t", nu, scale).into_pyobject(py),
        HalfPrior::Normal { scale } => ("normal", scale).into_pyobject(py),
    }
}

/// A real prior from ``("student_t", nu, location, scale)`` or
/// ``("normal", mean, sd)``.
fn real_prior(name: &str, value: &Bound<'_, PyTuple>) -> PyResult<RealPrior> {
    let bad = || {
        PyValueError::new_err(format!(
            "{name} must be (\"student_t\", nu, location, scale) or (\"normal\", mean, sd)"
        ))
    };
    let family: String = value.get_item(0).map_err(|_| bad())?.extract()?;
    let num = |i: usize| -> PyResult<f64> { value.get_item(i).map_err(|_| bad())?.extract() };
    match (family.to_ascii_lowercase().as_str(), value.len()) {
        ("student_t", 4) => Ok(RealPrior::StudentT {
            nu: num(1)?,
            location: num(2)?,
            scale: num(3)?,
        }),
        ("normal", 3) => Ok(RealPrior::Normal {
            mean: num(1)?,
            sd: num(2)?,
        }),
        _ => Err(bad()),
    }
}

fn real_prior_tuple<'py>(py: Python<'py>, prior: RealPrior) -> PyResult<Bound<'py, PyTuple>> {
    match prior {
        RealPrior::StudentT {
            nu,
            location,
            scale,
        } => ("student_t", nu, location, scale).into_pyobject(py),
        RealPrior::Normal { mean, sd } => ("normal", mean, sd).into_pyobject(py),
    }
}

fn log_sigma_priors(value: &[Bound<'_, PyTuple>]) -> PyResult<[RealPrior; 2]> {
    if value.len() != 2 {
        return Err(PyValueError::new_err(
            "log_sigma must hold two priors: log sigma_os, then log sigma_pd",
        ));
    }
    Ok([
        real_prior("log_sigma[0]", &value[0])?,
        real_prior("log_sigma[1]", &value[1])?,
    ])
}

/// An optional half prior: ``None`` (no such effect) or a half prior's
/// tuple.
fn optional_half_prior(
    name: &str,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<HalfPrior>> {
    match value {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => {
            let t = v.cast::<PyTuple>().map_err(|_| {
                PyValueError::new_err(format!(
                    "{name} must be None, (\"student_t\", nu, scale) or (\"normal\", scale)"
                ))
            })?;
            Ok(Some(half_prior(name, t)?))
        }
    }
}

fn optional_half_prior_tuple<'py>(
    py: Python<'py>,
    prior: Option<HalfPrior>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    prior.map(|p| half_prior_tuple(py, p)).transpose()
}

/// The development-year effect sds on ``k_er``, ``k_p``, ``rlr`` and
/// ``rrf``: ``None`` (none) or four items, each ``None`` or a half prior.
fn development_effect_sds(value: Option<&Bound<'_, PyAny>>) -> PyResult<[Option<HalfPrior>; 4]> {
    let Some(v) = value.filter(|v| !v.is_none()) else {
        return Ok([None; 4]);
    };
    let bad = || {
        PyValueError::new_err(
            "development_effect_sd must be None or four priors, on k_er, k_p, rlr and rrf in \
             that order, each None or a half prior",
        )
    };
    let items: Vec<Bound<'_, PyAny>> = v.try_iter().map_err(|_| bad())?.collect::<PyResult<_>>()?;
    if items.len() != 4 {
        return Err(bad());
    }
    let mut out = [None; 4];
    for (i, (slot, item)) in out.iter_mut().zip(&items).enumerate() {
        *slot = optional_half_prior(&format!("development_effect_sd[{i}]"), Some(item))?;
    }
    Ok(out)
}

/// ``replace``'s value for an optional effect: ``None`` keeps it,
/// ``False`` removes it, a half prior's tuple sets it.
fn replace_effect(
    name: &str,
    value: Option<&Bound<'_, PyAny>>,
    current: Option<HalfPrior>,
) -> PyResult<Option<HalfPrior>> {
    match value {
        None => Ok(current),
        Some(v) if v.is_none() => Ok(current),
        Some(v) if v.is_instance_of::<PyBool>() => {
            if v.extract::<bool>()? {
                Err(PyValueError::new_err(format!(
                    "{name} must be False (no effect) or a half prior, not True"
                )))
            } else {
                Ok(None)
            }
        }
        Some(v) => optional_half_prior(name, Some(v)),
    }
}

/// The priors of ``Compartmental``, a required argument: the rates and
/// ratios are specific to a line and a book, so the library does not choose
/// them silently. ``CompartmentalPriors.gesmann_morris()`` is the paper's
/// model 1 and ``CompartmentalPriors.gesmann_morris_2()`` its model 2;
/// check any set with ``Compartmental.prior_predictive`` before fitting.
/// ``CompartmentalPriors.diffuse()`` is the starting point for a fit to paid alone
/// (``fitted="paid_only"``): it leaves the rates to the data. A fit to incurred alone
/// (``fitted="incurred_only"``) refuses a ``k_p`` prior with a log-scale sd above 0.5,
/// so give it a prior you believe, such as the paper's. Use a narrower prior than the
/// preset also when you hold information about the rates that is not in the data.
///
/// Each positive parameter ``θ`` (``k_er``, ``k_p``, ``rlr``, ``rrf``) is
/// ``median * exp(sd * o)`` with ``o ~ N(0, 1)``: a log-normal prior with
/// that median and log-scale sd, given as ``(median, sd)``. The effects
/// live on that ``o`` scale.
///
/// The hierarchy follows from the priors. ``RLR`` and ``RRF`` always vary
/// by origin, with correlated effects. ``ker_effect_sd`` and
/// ``kp_effect_sd`` add origin effects on the rates, uncorrelated with the
/// others, and ``development_effect_sd`` adds development-year effects on
/// any of the four parameters; ``None`` (model 1) means no such effect.
/// A development age with no observed cell takes a new effect from the
/// population in each simulation, as an origin with no cell does.
///
/// Parameters
/// ----------
/// k_er, k_p : tuple of (float, float)
///     ``(median, sd)`` of the reporting rate and the payment rate, per year.
/// rlr, rrf : tuple of (float, float)
///     ``(median, sd)`` of the population reported loss ratio and reserve
///     robustness factor (paid over reported).
/// rlr_effect_sd, rrf_effect_sd : tuple
///     Priors on the sds of the origin effects on the ``o`` scale:
///     ``("student_t", nu, scale)`` for a half Student-t or
///     ``("normal", scale)`` for a half normal.
/// lkj_eta : float
///     LKJ shape of the correlation of the two effects (1: uniform).
/// log_sigma : sequence of two tuples
///     Priors on ``log sigma_os`` and ``log sigma_pd``, in that order, each
///     ``("student_t", nu, location, scale)`` or ``("normal", mean, sd)``.
/// ker_effect_sd, kp_effect_sd : tuple, optional
///     Priors on the sds of origin effects on ``k_er`` and ``k_p``, in the
///     form of ``rlr_effect_sd``; ``None``: no such effect.
/// development_effect_sd : sequence of four, optional
///     Priors on the sds of development-year effects on ``k_er``, ``k_p``,
///     ``rlr`` and ``rrf``, in that order, each ``None`` (no effect) or a
///     half prior; ``None``: no development effect.
///
/// Raises
/// ------
/// ValueError
///     If a prior has the wrong form or a log-normal median or sd is not
///     finite and positive. The other settings are checked by the fit.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import CompartmentalPriors
/// >>> p = CompartmentalPriors.gesmann_morris()
/// >>> p.k_er, p.rlr_effect_sd, p.log_sigma[0]
/// ((3.0, 0.1), ('student_t', 10.0, 0.2), ('student_t', 1.0, 0.0, 1000.0))
/// >>> p.replace(k_er=(2.0, 0.3)).k_er
/// (2.0, 0.3)
/// >>> q = CompartmentalPriors.gesmann_morris_2()
/// >>> q.kp_effect_sd, q.development_effect_sd[2]
/// (('student_t', 10.0, 0.3), ('student_t', 10.0, 0.7))
/// >>> q.replace(kp_effect_sd=False).kp_effect_sd is None
/// True
#[pyclass(name = "CompartmentalPriors", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCompartmentalPriors {
    inner: CompartmentalPriors,
}

#[pymethods]
impl PyCompartmentalPriors {
    #[new]
    #[pyo3(signature = (
        *,
        k_er,
        k_p,
        rlr,
        rrf,
        rlr_effect_sd,
        rrf_effect_sd,
        lkj_eta,
        log_sigma,
        ker_effect_sd = None,
        kp_effect_sd = None,
        development_effect_sd = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        k_er: (f64, f64),
        k_p: (f64, f64),
        rlr: (f64, f64),
        rrf: (f64, f64),
        rlr_effect_sd: Bound<'_, PyTuple>,
        rrf_effect_sd: Bound<'_, PyTuple>,
        lkj_eta: f64,
        log_sigma: Vec<Bound<'_, PyTuple>>,
        ker_effect_sd: Option<Bound<'_, PyAny>>,
        kp_effect_sd: Option<Bound<'_, PyAny>>,
        development_effect_sd: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CompartmentalPriors {
                k_er: positive_prior("k_er", k_er)?,
                k_p: positive_prior("k_p", k_p)?,
                rlr: positive_prior("rlr", rlr)?,
                rrf: positive_prior("rrf", rrf)?,
                rlr_effect_sd: half_prior("rlr_effect_sd", &rlr_effect_sd)?,
                rrf_effect_sd: half_prior("rrf_effect_sd", &rrf_effect_sd)?,
                lkj_eta,
                log_sigma: log_sigma_priors(&log_sigma)?,
                ker_effect_sd: optional_half_prior("ker_effect_sd", ker_effect_sd.as_ref())?,
                kp_effect_sd: optional_half_prior("kp_effect_sd", kp_effect_sd.as_ref())?,
                development_effect_sd: development_effect_sds(development_effect_sd.as_ref())?,
            },
        })
    }

    /// Gesmann and Morris' (2020) case study, model 1: ``k_er`` (3, 0.1),
    /// ``k_p`` (1, 0.1), ``rlr`` (0.7, 0.2), ``rrf`` (0.8, 0.1), half
    /// Student-t(10) effect sds with scales 0.2 and 0.1, LKJ(1), and
    /// Student-t(1, 0, 1000) on each ``log sigma`` (effectively flat), with
    /// no effect on the rates and no development effect. On company 337 the
    /// data pull the rates 5 to 9 prior sds from these medians: the
    /// posterior is a compromise with the preset.
    ///
    /// Returns
    /// -------
    /// CompartmentalPriors
    #[staticmethod]
    fn gesmann_morris() -> Self {
        Self {
            inner: CompartmentalPriors::gesmann_morris(),
        }
    }

    /// A diffuse set for model 1: the paper's medians (``k_er`` 3, ``k_p`` 1,
    /// ``rlr`` 0.7, ``rrf`` 0.8) with a log-scale sd of 1 on each of the four,
    /// 5 to 10 times the paper's, and its effect sds and ``log sigma`` priors.
    /// Use it as the starting point for a fit to paid alone
    /// (``fitted="paid_only"``), where the paid cells pin the rates. It is
    /// refused for a fit to incurred alone (``fitted="incurred_only"``): no
    /// paid cell is fitted there, so the payment rate ``k_p`` and with it the
    /// paid run-off and the reserve follow the prior on ``k_p`` at every width
    /// (on company 337 -51,000 with an sd of about 150,000 at a log-scale sd of
    /// 1, 203,000 at 0.3), and a ``k_p`` prior with a log-scale sd above 0.5 is
    /// an error. For incurred alone replace ``k_p`` with a prior you believe
    /// (``replace``), for example one from the book's paid history or from a
    /// paid-and-incurred fit of a similar book; the paper's narrow prior put
    /// the reserve on company 337 26% above the fit to paid and outstanding.
    ///
    /// Returns
    /// -------
    /// CompartmentalPriors
    #[staticmethod]
    fn diffuse() -> Self {
        Self {
            inner: CompartmentalPriors::diffuse(),
        }
    }

    /// Gesmann and Morris' (2020) case study, model 2 (their code, sec.
    /// 7.2.3, where it and the text disagree): model 1's rate and ratio
    /// priors; origin effects on all four parameters (``RLR`` and ``RRF``
    /// correlated, LKJ(1)) and development-year effects on all four, every
    /// effect sd half Student-t(10) on the ``o`` scale with one scale per
    /// parameter, shared by its two groups: 0.3 for ``k_er`` and ``k_p``,
    /// 0.7 for ``rlr`` and 0.5 for ``rrf``; and ``("normal", log 0.2,
    /// 0.2)`` on each ``log sigma`` (the text gives a median of 0.1). Use
    /// it with ``Compartmental.model_2``.
    ///
    /// Returns
    /// -------
    /// CompartmentalPriors
    #[staticmethod]
    fn gesmann_morris_2() -> Self {
        Self {
            inner: CompartmentalPriors::gesmann_morris_2(),
        }
    }

    /// A copy with the given priors replaced, each in the constructor's
    /// form. For ``ker_effect_sd`` and ``kp_effect_sd``, ``None`` keeps the
    /// current prior and ``False`` removes the effect; a
    /// ``development_effect_sd`` replaces all four, its ``None`` items
    /// removing those effects.
    ///
    /// Returns
    /// -------
    /// CompartmentalPriors
    #[pyo3(signature = (
        *,
        k_er = None,
        k_p = None,
        rlr = None,
        rrf = None,
        rlr_effect_sd = None,
        rrf_effect_sd = None,
        lkj_eta = None,
        log_sigma = None,
        ker_effect_sd = None,
        kp_effect_sd = None,
        development_effect_sd = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn replace(
        &self,
        k_er: Option<(f64, f64)>,
        k_p: Option<(f64, f64)>,
        rlr: Option<(f64, f64)>,
        rrf: Option<(f64, f64)>,
        rlr_effect_sd: Option<Bound<'_, PyTuple>>,
        rrf_effect_sd: Option<Bound<'_, PyTuple>>,
        lkj_eta: Option<f64>,
        log_sigma: Option<Vec<Bound<'_, PyTuple>>>,
        ker_effect_sd: Option<Bound<'_, PyAny>>,
        kp_effect_sd: Option<Bound<'_, PyAny>>,
        development_effect_sd: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut p = self.inner;
        if let Some(v) = k_er {
            p.k_er = positive_prior("k_er", v)?;
        }
        if let Some(v) = k_p {
            p.k_p = positive_prior("k_p", v)?;
        }
        if let Some(v) = rlr {
            p.rlr = positive_prior("rlr", v)?;
        }
        if let Some(v) = rrf {
            p.rrf = positive_prior("rrf", v)?;
        }
        if let Some(v) = rlr_effect_sd {
            p.rlr_effect_sd = half_prior("rlr_effect_sd", &v)?;
        }
        if let Some(v) = rrf_effect_sd {
            p.rrf_effect_sd = half_prior("rrf_effect_sd", &v)?;
        }
        if let Some(v) = lkj_eta {
            p.lkj_eta = v;
        }
        if let Some(v) = log_sigma {
            p.log_sigma = log_sigma_priors(&v)?;
        }
        p.ker_effect_sd = replace_effect("ker_effect_sd", ker_effect_sd.as_ref(), p.ker_effect_sd)?;
        p.kp_effect_sd = replace_effect("kp_effect_sd", kp_effect_sd.as_ref(), p.kp_effect_sd)?;
        if let Some(v) = development_effect_sd.filter(|v| !v.is_none()) {
            p.development_effect_sd = development_effect_sds(Some(&v))?;
        }
        Ok(Self { inner: p })
    }

    /// ``(median, sd)`` of the reporting rate.
    #[getter]
    fn k_er(&self) -> (f64, f64) {
        positive_prior_tuple(self.inner.k_er)
    }

    /// ``(median, sd)`` of the payment rate.
    #[getter]
    fn k_p(&self) -> (f64, f64) {
        positive_prior_tuple(self.inner.k_p)
    }

    /// ``(median, sd)`` of the population reported loss ratio.
    #[getter]
    fn rlr(&self) -> (f64, f64) {
        positive_prior_tuple(self.inner.rlr)
    }

    /// ``(median, sd)`` of the population reserve robustness factor.
    #[getter]
    fn rrf(&self) -> (f64, f64) {
        positive_prior_tuple(self.inner.rrf)
    }

    /// Prior on the sd of the origin effects on ``RLR``.
    #[getter]
    fn rlr_effect_sd<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        half_prior_tuple(py, self.inner.rlr_effect_sd)
    }

    /// Prior on the sd of the origin effects on ``RRF``.
    #[getter]
    fn rrf_effect_sd<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        half_prior_tuple(py, self.inner.rrf_effect_sd)
    }

    /// LKJ shape of the correlation of the two effects.
    #[getter]
    fn lkj_eta(&self) -> f64 {
        self.inner.lkj_eta
    }

    /// Priors on ``log sigma_os`` and ``log sigma_pd``.
    #[getter]
    fn log_sigma<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        self.inner
            .log_sigma
            .iter()
            .map(|&p| real_prior_tuple(py, p))
            .collect()
    }

    /// Prior on the sd of the origin effects on ``k_er``; ``None``: no such
    /// effect.
    #[getter]
    fn ker_effect_sd<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        optional_half_prior_tuple(py, self.inner.ker_effect_sd)
    }

    /// Prior on the sd of the origin effects on ``k_p``; ``None``: no such
    /// effect.
    #[getter]
    fn kp_effect_sd<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        optional_half_prior_tuple(py, self.inner.kp_effect_sd)
    }

    /// Priors on the sds of the development-year effects on ``k_er``,
    /// ``k_p``, ``rlr`` and ``rrf``, each ``None`` when there is no such
    /// effect.
    #[getter]
    fn development_effect_sd<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<Option<Bound<'py, PyTuple>>>> {
        self.inner
            .development_effect_sd
            .iter()
            .map(|&p| optional_half_prior_tuple(py, p))
            .collect()
    }

    fn __eq__(&self, other: PyRef<'_, Self>) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        let p = &self.inner;
        let half = |h: HalfPrior| match h {
            HalfPrior::StudentT { nu, scale } => format!("(\"student_t\", {nu:?}, {scale:?})"),
            HalfPrior::Normal { scale } => format!("(\"normal\", {scale:?})"),
        };
        let maybe = |h: Option<HalfPrior>| h.map_or_else(|| "None".to_string(), half);
        let real = |r: RealPrior| match r {
            RealPrior::StudentT {
                nu,
                location,
                scale,
            } => format!("(\"student_t\", {nu:?}, {location:?}, {scale:?})"),
            RealPrior::Normal { mean, sd } => format!("(\"normal\", {mean:?}, {sd:?})"),
        };
        let pos = |v: PositivePrior| format!("{:?}", positive_prior_tuple(v));
        let dev = if p.development_effect_sd.iter().all(Option::is_none) {
            "None".to_string()
        } else {
            let items: Vec<String> = p.development_effect_sd.iter().map(|&h| maybe(h)).collect();
            format!("[{}]", items.join(", "))
        };
        format!(
            "CompartmentalPriors(k_er={}, k_p={}, rlr={}, rrf={}, rlr_effect_sd={}, \
             rrf_effect_sd={}, lkj_eta={:?}, log_sigma=[{}, {}], ker_effect_sd={}, \
             kp_effect_sd={}, development_effect_sd={})",
            pos(p.k_er),
            pos(p.k_p),
            pos(p.rlr),
            pos(p.rrf),
            half(p.rlr_effect_sd),
            half(p.rrf_effect_sd),
            p.lkj_eta,
            real(p.log_sigma[0]),
            real(p.log_sigma[1]),
            maybe(p.ker_effect_sd),
            maybe(p.kp_effect_sd),
            dev,
        )
    }
}

/// A setting given as a string, with its accepted names in order.
fn named<T: Copy>(setting: &str, name: &str, values: &[(&str, T)]) -> PyResult<T> {
    let lower = name.to_ascii_lowercase();
    values
        .iter()
        .find(|(n, _)| *n == lower)
        .map(|(_, v)| *v)
        .ok_or_else(|| {
            let expected: Vec<String> = values.iter().map(|(n, _)| format!("{n:?}")).collect();
            PyValueError::new_err(format!(
                "unknown {setting} {name:?}; expected one of {}",
                expected.join(", ")
            ))
        })
}

/// The name of a setting's value.
fn name_of<T: PartialEq>(value: T, values: &[(&'static str, T)]) -> &'static str {
    values
        .iter()
        .find(|(_, v)| *v == value)
        .map_or("?", |(n, _)| n)
}

const LIKELIHOODS: [(&str, CompartmentalLikelihood); 2] = [
    ("normal", CompartmentalLikelihood::Normal),
    ("lognormal", CompartmentalLikelihood::LogNormal),
];
const SCALES: [(&str, CompartmentalScale); 2] = [
    ("amount", CompartmentalScale::Amount),
    ("loss_ratio", CompartmentalScale::LossRatio),
];
const PAID_RESPONSES: [(&str, PaidResponse); 2] = [
    ("cumulative", PaidResponse::Cumulative),
    ("incremental", PaidResponse::Incremental),
];
const FITTED: [(&str, Fitted); 4] = [
    ("paid_and_outstanding", Fitted::PaidAndOutstanding),
    ("paid_and_incurred", Fitted::PaidAndIncurred),
    ("paid_only", Fitted::PaidOnly),
    ("incurred_only", Fitted::IncurredOnly),
];
const RATE_ORDERS: [(&str, RateOrder); 2] =
    [("free", RateOrder::Free), ("ordered", RateOrder::Ordered)];
const DEVELOPMENT_LEVELS: [(&str, DevelopmentLevels); 2] = [
    ("per_age", DevelopmentLevels::PerAge),
    ("by_year", DevelopmentLevels::ByYear),
];
const STARTS: [(&str, CompartmentalStart); 2] = [
    ("data", CompartmentalStart::Data),
    ("prior_centre", CompartmentalStart::PriorCentre),
];

/// Gesmann and Morris' (2020) hierarchical compartmental reserving model
/// with one outstanding compartment, their models 1 and 2
/// (``docs/design/reserving-compartmental.md``). Claims flow from exposure
/// into outstanding as they are reported, at rate ``k_er``, and from
/// outstanding into paid as they settle, at rate ``k_p``. For origin ``i``
/// with premium ``P_i`` and development time ``t`` in years, with
/// ``g(t) = (exp(-k_p t) - exp(-k_er t)) / (k_er - k_p)``::
///
///     OS_i(t) = P_i RLR_i k_er g(t)
///     PD_i(t) = P_i RLR_i RRF_i [(1 - exp(-k_p t)) - k_p g(t)]
///
/// so ``RLR_i`` is the reported loss ratio, ``RRF_i`` the reserve
/// robustness factor (paid over reported) and ``RLR_i RRF_i`` the ultimate
/// loss ratio. ``RLR`` and ``RRF`` vary by origin with correlated effects
/// (LKJ prior on the correlation) on the priors' ``o`` scale; the priors
/// can add origin effects on the rates and development-year effects on any
/// of the four (``CompartmentalPriors``), each cell then taking its own
/// parameters. NUTS samples the posterior on a pilot run's dense scale,
/// with the gradient from dual numbers.
///
/// Model 1 (the defaults) has normal outstanding and cumulative paid
/// amounts about these curves, each type with its own sigma. Model 2
/// (``Compartmental.model_2``, with
/// ``CompartmentalPriors.gesmann_morris_2()``) has log-normal outstanding
/// and incremental paid loss ratios to premium; every combination of
/// ``likelihood``, ``scale`` and ``paid_response`` is built. Under
/// ``likelihood="lognormal"`` every response must be positive, and under
/// ``paid_response="incremental"`` each paid cell needs the paid at the
/// age before; ``fit`` raises ``ValueError`` naming the cell otherwise.
/// ``fitted="paid_and_incurred"`` fits paid and incurred, with incurred in place of
/// outstanding as the second response: the mean of an incurred cell is ``RLR_i (OS(t) +
/// RRF_i PD(t))`` times premium (or as a loss ratio), with the cumulative ``PD`` even
/// under ``paid_response="incremental"``, and ``log_sigma_os`` is the incurred sigma.
/// Give the incurred column as ``incurred=``, or an outstanding column as
/// ``outstanding=`` and the incurred is outstanding plus paid where both are observed.
/// The fit's ``outstanding`` then holds the future incurred cells.
///
/// ``fitted="paid_only"`` and ``fitted="incurred_only"`` fit one response alone and need
/// ``rate_order="ordered"`` (``ValueError`` under ``"free"``): from one response the means
/// do not change under the rate swap below, and from paid alone only ``RLR x RRF`` is
/// identified, so the priors carry the split of ``RLR`` and ``RRF``. The model then has one
/// sigma, ``sigma_pd`` or ``sigma_os``. Under ``"paid_only"`` the second column is not read
/// (``outstanding=`` and ``incurred=`` may both be left out) and the fit has no
/// ``outstanding`` (``None``). Under ``"incurred_only"`` the incurred is given as for
/// ``"paid_and_incurred"`` and the paid column supplies each origin's latest paid only: the
/// reserve is the expected paid at the last age less the latest paid, and the future paid
/// cells are expected values, without noise, since no paid cell is fitted.
///
/// ``CompartmentalPriors.diffuse()`` is the starting point for ``"paid_only"``. Under
/// ``"incurred_only"`` no paid cell is fitted, so the incurred does not identify the payment
/// rate ``k_p`` and its prior sets the paid run-off and the reserve at every width (on company
/// 337 the paper's narrow prior put the reserve 26% above the fit to paid and outstanding):
/// the fit refuses a ``k_p`` prior with a log-scale sd above 0.5, so the diffuse preset is an
/// error there, and you give a ``k_p`` prior you believe, for example from the book's paid
/// history or from a paid-and-incurred fit of a similar book.
///
/// An origin with a premium and no observed cell (the papers' training
/// triangle has one) draws its effects from the population in each
/// simulation, and so does a development age with no observed cell. The
/// reserve is the paid at the square's last age less the latest paid; the
/// square runs for as many origin periods as the premium column has
/// origins, in development periods (40 quarterly ages for ten annual
/// origins). The development grain may be finer than the origin grain
/// (the triangle's own grains), and each cell's development time is its
/// age in years on every grain, so the rates are per year and an
/// incremental paid cell is the increment over one development period.
/// The coming year of ``one_year`` is the next twelve months of calendar
/// time, up to four quarterly cells per origin. One ``seed`` replays a fit
/// exactly: the chains and the simulations each take a seed derived from
/// it.
///
/// The means are exactly invariant under swapping the two rates (with
/// ``RLR`` and ``RRF`` rescaled), so only the priors tell the two modes
/// apart. Under ``rate_order="free"`` (the paper's priors) the fit warns
/// when any posterior draw has population rates with ``k_er <= k_p``;
/// ``rate_order="ordered"`` samples ``k_er = k_p + exp(delta)``, the same
/// priors restricted to ``k_er > k_p``. With effects on the rates,
/// ``CompartmentalFit.cell_rate_swaps`` reports the share of draws and
/// cells with ``k_er <= k_p``.
///
/// Fitting sub-annual data
/// -----------------------
/// Quarterly or monthly development cells are allowed. What follows is from
/// a study on simulated books (design note decision 14,
/// ``knowledge/findings/compartmental-subannual.md``); no real sub-annual
/// triangle was available.
///
/// * Rate priors: the rates are per year on every grain, and the paper's
///   presets are kept as they are. Finer cells pull the preset towards the
///   truth; a line whose first year runs much faster or slower than the
///   preset's ``k_er`` of about 3 needs priors of its own.
/// * Development levels (model 2): ``development_levels="per_age"`` or
///   ``"by_year"``, a choice to make with ``loo``. Where effects move by
///   year, by-year levels agree with per-age ones on the reserve (0.10 sd
///   apart on average in the study), win elpd in 30 of 30 quarterly books and
///   on 10 monthly books under a short sampler took 89 s against 126 s (rough
///   times) with a worst R-hat of 1.07 against 1.62, though a median bulk ESS
///   of 101 against 114; where effects move within a year they lose 361 of
///   elpd in 30 of 30 books and widen the reserve's sd by 59%.
/// * Seasonality: there is no seasonal term. Model 1 reads a season as noise
///   (the interval widens, by up to 54% in the sd for a 30% season, and the
///   mean can sit nearly 1 sd off the outcome), and model 2's per-age levels
///   do not absorb a season unless every origin starts in the same calendar
///   quarter. A calendar-quarter factor is not built.
/// * Cumulative paid: the interval narrows as the cells get finer, right where
///   the noise of the cells is independent and too narrow where it persists.
///   In the study's world with persistent payment noise, at one persistence
///   chosen by hand and fitted to no book (``PERSIST_PHI`` 25, ``PERSIST_RHO``
///   0.95; a smaller fall is expected with less, untested), the 90% interval
///   covered 88%, 70% and 56% at annual, quarterly and monthly cells and the
///   RMSE rose from 1,623 to 2,033. A fit with cumulative paid on cells finer
///   than a year therefore warns, asking you to check the residuals; it also
///   warns when
///   the fit is calibrated, since the data do not say which world they are
///   in. ``paid_response="incremental"`` keeps the interval but, in that
///   world, ran 1.3 to 2 times too wide, with a worse RMSE at quarterly and
///   monthly grain and a failing Meyers' D at annual grain. Neither is chosen
///   for you.
/// * Earned exposure: all of an origin's exposure starts at age 0, so
///   sub-annual development of annual origins fits an exposure the model does
///   not have; use quarterly origins where the data allow. On simulated books
///   with earned premium annual origins failed Meyers' D at annual, quarterly
///   and monthly development, and quarterly origins of the same books passed
///   narrowly (17.8 against a critical 19.2, on a short sampler, with 3 books
///   at the default sampler); that comparison also changes the number of
///   origins and the effects' variance. A setting that spreads the exposure
///   over the origin's period is not built.
///
/// Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS below 100
/// per chain, any divergent transition) and rate swaps do not fail the
/// fit: ``fit`` raises one ``UserWarning`` listing them, and they stay in
/// ``CompartmentalFit.diagnostics``. With the defaults, 194 of 200 seeds on
/// company 337 passed every threshold for model 1, and with
/// ``model_2``'s sampler 195 of 200 for model 2; each of the others just
/// missed one.
///
/// Parameters
/// ----------
/// priors : CompartmentalPriors
///     The priors; required.
/// likelihood : str
///     ``"normal"`` (model 1) or ``"lognormal"`` (model 2).
/// scale : str
///     The response scale: ``"amount"`` (model 1) or ``"loss_ratio"``,
///     amounts over the origin's premium (model 2).
/// paid_response : str
///     ``"cumulative"`` (model 1) or ``"incremental"`` (model 2).
/// fitted : str
///     ``"paid_and_outstanding"``, ``"paid_and_incurred"`` (incurred replaces
///     outstanding as the second response, as above), ``"paid_only"`` or
///     ``"incurred_only"`` (one response alone, with ``rate_order="ordered"``).
/// rate_order : str
///     ``"free"`` or ``"ordered"``, as above.
/// development_levels : str
///     The width of a development level, for priors with development
///     effects (model 2): ``"per_age"`` (default), one level for each age, or
///     ``"by_year"``, one for each development year (counted from the
///     origin's start: ages 1 to 12 months, 13 to 24 and so on). Choose with
///     ``loo``; see "Fitting sub-annual data" above. The same fit at annual
///     grain. ``ValueError`` for ``"by_year"`` with priors that have no
///     development effects (model 1).
/// start : str
///     Where the pilot run starts: ``"data"``, a least-squares fit of the
///     population curve (the prior centre if that fit fails, and under one
///     response if its log posterior is below the prior centre's), or
///     ``"prior_centre"``.
/// chains, tune, draws : int
///     NUTS chains, warm-up draws per chain and kept draws per chain.
/// n_sims : int
///     Number of predictive simulations; positive.
/// seed : int
///     Seed of the chains and of the simulations.
/// target_accept : float
///     Target mean acceptance rate (the paper's ``adapt_delta``, 0.99).
/// max_depth : int
///     Largest NUTS tree depth.
///
/// Examples
/// --------
/// >>> import warnings
/// >>> from prospicio.reserving import Compartmental, CompartmentalPriors, Triangle
/// >>> paid = [[9.6, 22.8, 33.3, 40.3, 45.1], [7.9, 19.5, 29.6, 36.8],
/// ...         [8.7, 24.3, 35.4], [13.3, 33.0], [11.4]]
/// >>> incurred = [[62.7, 64.0, 65.6, 65.3, 62.3], [62.1, 56.7, 56.6, 56.8],
/// ...             [64.0, 61.5, 63.2], [73.9, 75.1], [77.1]]
/// >>> premium = [104.4, 88.9, 86.0, 99.3, 104.9]
/// >>> rows = [(1988 + w, 12 * (d + 1), paid[w][d], incurred[w][d], premium[w])
/// ...         for w in range(5) for d in range(len(paid[w]))]
/// >>> origin, age, pd, inc, prem = map(list, zip(*rows))
/// >>> tri = Triangle.from_long(
/// ...     origin, age, {"paid": pd, "incurred": inc, "premium": prem})
/// >>> model = Compartmental(CompartmentalPriors.gesmann_morris(), chains=2,
/// ...                       tune=150, draws=150, target_accept=0.9, n_sims=1000)
/// >>> # Check the priors first: the mean paid of 1988 at age 60 over its premium.
/// >>> paths = model.prior_predictive(tri, "premium", n_sims=1000)
/// >>> ratio = paths["paid"].marginal(("1988", 60))
/// >>> 0.2 < ratio.quantile(0.05) / 104.4 and ratio.quantile(0.95) / 104.4 < 1.2
/// True
/// >>> with warnings.catch_warnings():
/// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
/// ...     fit = model.fit(tri, "paid", "premium", incurred="incurred")
/// >>> fit.reserves.dims, fit.parameter_names[0]
/// (['origin'], 'k_er')
#[pyclass(name = "Compartmental", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCompartmental {
    inner: Compartmental,
}

#[pymethods]
impl PyCompartmental {
    #[new]
    #[pyo3(signature = (
        priors,
        *,
        likelihood = "normal",
        scale = "amount",
        paid_response = "cumulative",
        fitted = "paid_and_outstanding",
        rate_order = "free",
        development_levels = "per_age",
        start = "data",
        chains = 4,
        tune = 1000,
        draws = 2000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.99,
        max_depth = 10,
        dependence = "independent",
        spearman = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        priors: PyRef<'_, PyCompartmentalPriors>,
        likelihood: &str,
        scale: &str,
        paid_response: &str,
        fitted: &str,
        rate_order: &str,
        development_levels: &str,
        start: &str,
        chains: usize,
        tune: usize,
        draws: usize,
        n_sims: usize,
        seed: u64,
        target_accept: f64,
        max_depth: u64,
        dependence: &str,
        spearman: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        let inner = Compartmental {
            priors: priors.inner,
            likelihood: named("likelihood", likelihood, &LIKELIHOODS)?,
            scale: named("scale", scale, &SCALES)?,
            paid_response: named("paid_response", paid_response, &PAID_RESPONSES)?,
            fitted: named("fitted", fitted, &FITTED)?,
            rate_order: named("rate_order", rate_order, &RATE_ORDERS)?,
            development_levels: named(
                "development_levels",
                development_levels,
                &DEVELOPMENT_LEVELS,
            )?,
            start: named("start", start, &STARTS)?,
            sampler: BayesSampler {
                chains,
                tune,
                draws,
                seed: 0,
                target_accept,
                max_depth,
            },
            n_sims,
            seed,
            dependence: segment_dependence(dependence, spearman)?,
        };
        // Refuse a combination of settings the model does not support when
        // it is built, not when it is fitted (#580).
        inner.check_supported().map_err(err)?;
        Ok(Self { inner })
    }

    /// Gesmann and Morris' model 2 with ``priors`` (the paper's are
    /// ``CompartmentalPriors.gesmann_morris_2()``): the log-normal
    /// likelihood on loss ratios with incremental paid, and a longer, finer
    /// sampler than the constructor's, four chains of 2,000 warm-up and
    /// 8,000 kept draws with target acceptance 0.999. With 87 parameters,
    /// effect sds whose posteriors pile up near 0 and a slowly mixing
    /// ``k_p``, model 1's sampler left divergent transitions or an R-hat
    /// above 1.01 on most seeds; this one passes every threshold on 195 of
    /// 200 seeds on company 337. A fit takes about a minute. The other
    /// arguments are the constructor's.
    ///
    /// Returns
    /// -------
    /// Compartmental
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import Compartmental, CompartmentalPriors
    /// >>> m = Compartmental.model_2(CompartmentalPriors.gesmann_morris_2())
    /// >>> m.likelihood, m.scale, m.paid_response, m.tune, m.draws, m.target_accept
    /// ('lognormal', 'loss_ratio', 'incremental', 2000, 8000, 0.999)
    #[staticmethod]
    #[pyo3(signature = (
        priors,
        *,
        rate_order = "free",
        development_levels = "per_age",
        start = "data",
        chains = 4,
        tune = 2000,
        draws = 8000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.999,
        max_depth = 10,
        dependence = "independent",
        spearman = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn model_2(
        priors: PyRef<'_, PyCompartmentalPriors>,
        rate_order: &str,
        development_levels: &str,
        start: &str,
        chains: usize,
        tune: usize,
        draws: usize,
        n_sims: usize,
        seed: u64,
        target_accept: f64,
        max_depth: u64,
        dependence: &str,
        spearman: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        Self::new(
            priors,
            "lognormal",
            "loss_ratio",
            "incremental",
            "paid_and_outstanding",
            rate_order,
            development_levels,
            start,
            chains,
            tune,
            draws,
            n_sims,
            seed,
            target_accept,
            max_depth,
            dependence,
            spearman,
        )
    }

    /// The priors.
    #[getter]
    fn priors(&self) -> PyCompartmentalPriors {
        PyCompartmentalPriors {
            inner: self.inner.priors,
        }
    }

    /// The likelihood: ``"normal"`` or ``"lognormal"``.
    #[getter]
    fn likelihood(&self) -> &'static str {
        name_of(self.inner.likelihood, &LIKELIHOODS)
    }

    /// The response scale: ``"amount"`` or ``"loss_ratio"``.
    #[getter]
    fn scale(&self) -> &'static str {
        name_of(self.inner.scale, &SCALES)
    }

    /// The paid response: ``"cumulative"`` or ``"incremental"``.
    #[getter]
    fn paid_response(&self) -> &'static str {
        name_of(self.inner.paid_response, &PAID_RESPONSES)
    }

    /// The responses fitted.
    #[getter]
    fn fitted(&self) -> &'static str {
        name_of(self.inner.fitted, &FITTED)
    }

    /// How the rates are kept apart: ``"free"`` or ``"ordered"``.
    #[getter]
    fn rate_order(&self) -> &'static str {
        name_of(self.inner.rate_order, &RATE_ORDERS)
    }

    /// The width of a development level: ``"per_age"`` or ``"by_year"``.
    #[getter]
    fn development_levels(&self) -> &'static str {
        name_of(self.inner.development_levels, &DEVELOPMENT_LEVELS)
    }

    /// Where the pilot run starts: ``"data"`` or ``"prior_centre"``.
    #[getter]
    fn start(&self) -> &'static str {
        name_of(self.inner.start, &STARTS)
    }

    /// Number of NUTS chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.sampler.chains
    }

    /// Warm-up draws per chain.
    #[getter]
    fn tune(&self) -> usize {
        self.inner.sampler.tune
    }

    /// Kept draws per chain.
    #[getter]
    fn draws(&self) -> usize {
        self.inner.sampler.draws
    }

    /// Number of predictive simulations.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims
    }

    /// Seed of the chains and of the simulations.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Target mean acceptance rate.
    #[getter]
    fn target_accept(&self) -> f64 {
        self.inner.sampler.target_accept
    }

    /// Largest NUTS tree depth.
    #[getter]
    fn max_depth(&self) -> u64 {
        self.inner.sampler.max_depth
    }

    /// How ``fit_segments`` joins the segments: ``"independent"`` or
    /// ``"rank_correlation"`` (``"synchronized"`` is refused by
    /// ``fit_segments``). ``fit`` ignores it.
    #[getter]
    fn dependence(&self) -> &'static str {
        dependence_name(&self.inner.dependence)
    }

    /// The Spearman matrix of ``dependence="rank_correlation"``; ``None``
    /// otherwise.
    #[getter]
    fn spearman(&self) -> Option<Vec<Vec<f64>>> {
        dependence_spearman(&self.inner.dependence)
    }

    /// Samples the posterior of paid and outstanding of a single-segment
    /// cumulative triangle, with each origin's premium from another column
    /// (its latest value), and simulates the future cells of the square and
    /// the reserves. Outstanding comes from ``outstanding``, a column of
    /// outstanding amounts, or from ``incurred``, a column of incurred
    /// amounts less paid cell by cell (the papers' outstanding on company
    /// 337 is incurred with bulk and IBNR less paid); give exactly one, or
    /// neither under ``fitted="paid_only"``, which does not read it.
    /// Raises one ``UserWarning`` listing the failed diagnostics and any
    /// rate swap.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// paid : str
    ///     Cumulative paid amounts.
    /// premium : str
    ///     The premium of each origin; the fit's origins are this column's.
    /// outstanding : str, optional
    ///     A column of outstanding amounts.
    /// incurred : str, optional
    ///     A column of incurred amounts.
    ///
    /// Returns
    /// -------
    /// CompartmentalFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``fitted`` is ``"paid_only"`` or ``"incurred_only"`` and
    ///     ``rate_order`` is not ``"ordered"``; if not exactly one of
    ///     ``outstanding`` and ``incurred`` is given (at most one under
    ///     ``"paid_only"``); if a column is
    ///     missing, the triangle has several segments, or a premium is
    ///     missing or not positive; if a prior or sampler setting is
    ///     invalid; or if there are no more observed cells than parameters.
    #[pyo3(signature = (triangle, paid, premium, *, outstanding = None, incurred = None))]
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        paid: &str,
        premium: &str,
        outstanding: Option<&str>,
        incurred: Option<&str>,
    ) -> PyResult<PyCompartmentalFit> {
        let os = compartmental_outstanding(self.inner.fitted, outstanding, incurred)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.fit(tri, paid, os, premium))
            .map_err(err)?;
        warn_compartmental(py, &inner.diagnostics.warnings)?;
        Ok(PyCompartmentalFit { inner })
    }

    /// Samples the posterior of paid and outstanding in every segment of a
    /// cumulative triangle, each segment with its own posterior (the same
    /// model and settings, sampled independently) and its premium from its
    /// own ``premium`` column as ``fit`` reads it, into joint distributions
    /// of the reserves and of the future paid and outstanding cells, the
    /// segments depending on each other as ``dependence`` says. Under
    /// ``"rank_correlation"`` each segment's reserves, future paid cells and
    /// future outstanding cells are reordered together, ranked on the
    /// reserves, so every output of a segment stays consistent. Segment
    /// ``k`` (in index order) takes seeds derived from ``seed`` and ``k``
    /// for its chains and its simulations, so independent segments share no
    /// random numbers, and a segment's draws differ from ``fit`` of that
    /// segment alone. Raises one ``UserWarning`` listing the failed
    /// diagnostics and any rate swap of every segment, each message
    /// prefixed with its segment's label.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// paid : str
    ///     Cumulative paid amounts.
    /// premium : str
    ///     The premium of each origin.
    /// outstanding : str, optional
    ///     A column of outstanding amounts.
    /// incurred : str, optional
    ///     A column of incurred amounts.
    ///
    /// Returns
    /// -------
    /// CompartmentalFits
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit`` for any segment (the message names the segment); if
    ///     ``dependence`` is ``"synchronized"`` (before any sampling); or if
    ///     ``spearman`` does not fit the segments.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import Compartmental, CompartmentalPriors, Triangle
    /// >>> paid = [[952, 1529, 2813, 3647, 3724], [849, 1564, 2202, 2432],
    /// ...         [983, 2211, 2830], [1657, 2685], [932]]
    /// >>> case = [[1500, 1400, 900, 400, 100], [1300, 1200, 800, 300],
    /// ...         [1400, 1100, 700], [1200, 900], [1100]]
    /// >>> premium = [5812, 4908, 5454, 5165, 5214]
    /// >>> rows = [(lob, 1988 + w, 12 * (d + 1), k * v, k * case[w][d], k * premium[w])
    /// ...         for lob, k in [("Auto", 1.0), ("Home", 2.0)]
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> lob, origin, age, values, os, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(
    /// ...     origin, age,
    /// ...     {"paid": values, "case": os, "premium": prem}, keys={"lob": lob})
    /// >>> model = Compartmental(
    /// ...     CompartmentalPriors.gesmann_morris(), tune=100, draws=100, n_sims=200,
    /// ...     chains=2)
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     fits = model.fit_segments(tri, "paid", "premium", outstanding="case")
    /// >>> fits.index, fits.reserves.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    #[pyo3(signature = (triangle, paid, premium, *, outstanding = None, incurred = None))]
    fn fit_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        paid: &str,
        premium: &str,
        outstanding: Option<&str>,
        incurred: Option<&str>,
    ) -> PyResult<PyCompartmentalFits> {
        let os = compartmental_outstanding(self.inner.fitted, outstanding, incurred)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.fit_segments(tri, paid, os, premium))
            .map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.fit.diagnostics);
        warn_compartmental(py, &warnings)?;
        Ok(PyCompartmentalFits { inner })
    }

    /// The one-year view: the claims development result over the coming
    /// year, by Pareto smoothed importance sampling of the posterior
    /// (decision 20 of ``docs/design/reserving-bayes.md``). The posterior
    /// and the lifetime reserves are ``fit``'s for the same settings. Each
    /// simulation picks a posterior draw, draws the paid and outstanding of
    /// the cells valued in the next twelve months from it (every origin's
    /// next diagonal), reweights every posterior draw by the likelihood of
    /// those cells, and smooths the weights (one Pareto ``k`` per
    /// simulation). Origin ``w``'s claims development result is the
    /// expected paid to the square's last age given the observed cells,
    /// less the year's paid and the same expectation given the year too;
    /// its mean is about zero. An origin with no observed cell (1997 on
    /// the papers' triangle) and, with development effects (model 2), an
    /// age with none take their effects as part of each posterior draw, so
    /// the year's cells update them. A simulation whose ``k`` is above
    /// ``k_threshold`` is flagged and handled as ``refit`` says. On company
    /// 337 about four in five of model 1's simulations are flagged and all
    /// of model 2's; for model 1 moment matching rescues about nineteen in
    /// twenty of them, for model 2 none, and a refit takes as long as a fit
    /// (for model 2 about 31 s of wall time with 8 running side by side).
    /// So by default (``refit=None``, the user's decisions of 2026-10-16 and
    /// 2026-10-21) model 1 moment-matches its flagged simulations and refits
    /// with NUTS only those whose ``k`` stays above the threshold, and a
    /// model with development effects (model 2) refits them with the light
    /// refit (``light_refit``): a warm-started NUTS run of 4 chains, 150
    /// warm-up and 250 kept draws, within about 0.03 sd of the total claims
    /// development result of a full refit, at about 20 times the run time
    /// of reweighting only (about 30 minutes for the default 2,000
    /// simulations on company 337). ``light_refit=False`` turns it off:
    /// model 2 then reweights its flagged simulations only, which is
    /// approximate (about 0.4 sd off a refit per simulation) and warns so
    /// (``approximate``). An explicit ``refit`` is used as it is, with the
    /// fit's own sampler unless ``light_refit`` is given, with
    /// ``max_refits`` as a budget. The view runs at most 2,000 simulations
    /// by default (``n_sims``).
    ///
    /// Raises one ``UserWarning`` listing the failed diagnostics of the
    /// fit, another if any refit failed its diagnostics, another if
    /// ``max_refits`` left simulations unrefitted, and another if the
    /// view is approximate (model 2 with ``light_refit=False``).
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// paid : str
    /// premium : str
    /// outstanding : str, optional
    /// incurred : str, optional
    ///     As in ``fit``.
    /// k_threshold : float, default 0.7
    ///     The Pareto ``k`` above which a simulation's importance weights
    ///     are not trusted.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, optional
    ///     What happens to a flagged simulation, as in ``BayesOdp.one_year``.
    ///     ``None`` (the default) takes the model's own:
    ///     ``"moment_match_then_refit"`` without development effects (model
    ///     1), ``"refit_flagged"`` with the light refit with them (model 2;
    ///     ``light_refit=False`` makes it ``"reweight_only"``, with a
    ///     warning).
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``.
    /// n_sims : int, optional
    ///     The number of one-year simulations. By default this model's
    ///     ``n_sims``, at most 2,000: nearly every simulation of these
    ///     models is flagged. The lifetime view in ``fit`` keeps
    ///     ``n_sims``; a smaller count gives the first simulations of a
    ///     larger one.
    /// light_refit : bool or tuple, optional
    ///     The sampler of the refits: a shorter, warm-started NUTS run
    ///     instead of the fit's own. ``None`` (the default) is the model's
    ///     own: the light refit for model 2 when ``refit`` is ``None``, the
    ///     fit's own sampler otherwise. ``True`` takes 4 chains of 150
    ///     warm-up and 250 kept draws, a tuple ``(chains, tune, draws)`` its
    ///     own. ``False`` turns model 2's default off: with ``refit=None``
    ///     the view reweights only and is approximate, with a warning. Each
    ///     refit scales its sampler by the opening posterior and starts
    ///     there, with no pilot run. On company 337 it comes within about
    ///     0.03 sd of the total claims development result of a full refit
    ///     of model 2, at about a thirtieth of the cost. A refit warns when
    ///     its closing reserve has R-hat above 1.05, fewer than 100
    ///     effective draws or a divergence.
    ///
    /// Returns
    /// -------
    /// CompartmentalOneYearFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit``; if ``k_threshold`` is NaN or ``refit`` is unknown.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import Compartmental, CompartmentalPriors, Triangle
    /// >>> paid = [[9.6, 22.8, 33.3, 40.3, 45.1], [7.9, 19.5, 29.6, 36.8],
    /// ...         [8.7, 24.3, 35.4], [13.3, 33.0], [11.4]]
    /// >>> incurred = [[62.7, 64.0, 65.6, 65.3, 62.3], [62.1, 56.7, 56.6, 56.8],
    /// ...             [64.0, 61.5, 63.2], [73.9, 75.1], [77.1]]
    /// >>> premium = [104.4, 88.9, 86.0, 99.3, 104.9]
    /// >>> rows = [(1988 + w, 12 * (d + 1), v, incurred[w][d], premium[w])
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> origin, age, pd, inc, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(origin, age, {"paid": pd, "incurred": inc, "premium": prem})
    /// >>> model = Compartmental(CompartmentalPriors.gesmann_morris(), chains=2, tune=150,
    /// ...                       draws=150, target_accept=0.9, n_sims=500)
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     view = model.one_year(tri, "paid", "premium", incurred="incurred",
    /// ...                           refit="reweight_only")
    /// >>> view.cdr.components()[:2]
    /// [('1988',), ('1989',)]
    /// >>> view.cdr.variance() < view.fit.reserves.variance()
    /// True
    #[pyo3(signature = (
        triangle,
        paid,
        premium,
        *,
        outstanding = None,
        incurred = None,
        k_threshold = 0.7,
        refit = None,
        max_refits = None,
        n_sims = None,
        light_refit = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        paid: &str,
        premium: &str,
        outstanding: Option<&str>,
        incurred: Option<&str>,
        k_threshold: f64,
        refit: Option<&str>,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
        light_refit: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyCompartmentalOneYearFit> {
        let os = compartmental_outstanding(self.inner.fitted, outstanding, incurred)?;
        let mut settings = one_year_settings(k_threshold, refit, max_refits, n_sims)?;
        (settings.light_refit, settings.no_light_refit) = light_refit_arg(light_refit)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year(tri, paid, os, premium, &settings))
            .map_err(err)?;
        warn_compartmental(py, &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "Compartmental",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        if let Some(message) = &inner.approximate {
            let message = std::ffi::CString::new(format!("Compartmental: {message}"))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
            PyErr::warn(py, category.as_any(), &message, 1)?;
        }
        Ok(PyCompartmentalOneYearFit { inner })
    }

    /// The one-year view of every segment of a triangle: each segment's
    /// view as ``one_year`` gives it, on the segment's own seeds as in
    /// ``fit_segments`` (a segment's simulations differ from
    /// ``one_year`` of the segment alone), and one joint claims development
    /// result across segments, which depend on each other as ``dependence``
    /// says: ``"independent"`` as drawn, ``"rank_correlation"`` by
    /// Iman-Conover on the segments' total claims development results.
    /// Everything else is as ``one_year``: the refit default of the model
    /// (moment matching for model 1, the light refit for model 2, or
    /// reweighting only with a warning with ``light_refit=False``), at most 2,000 simulations by default, and ``max_refits``
    /// as a budget of each segment, so ``k`` segments refit at most ``k``
    /// times this many. Raises one ``UserWarning`` listing the failed
    /// diagnostics of each segment's fit (with the segment's label), another
    /// if any refit failed its diagnostics, another if ``max_refits`` left
    /// simulations unrefitted, and another if the view is approximate.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// paid : str
    /// premium : str
    /// outstanding : str, optional
    /// incurred : str, optional
    ///     As in ``fit_segments``.
    /// k_threshold : float, default 0.7
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, optional
    /// max_refits : int, optional
    /// n_sims : int, optional
    /// light_refit : bool or tuple, optional
    ///     As in ``one_year``.
    ///
    /// Returns
    /// -------
    /// CompartmentalOneYearFits
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``one_year`` and ``fit_segments``, including
    ///     ``dependence="synchronized"``.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import Compartmental, CompartmentalPriors, Triangle
    /// >>> paid = [[952, 1529, 2813, 3647, 3724], [849, 1564, 2202, 2432],
    /// ...         [983, 2211, 2830], [1657, 2685], [932]]
    /// >>> case = [[1500, 1400, 900, 400, 100], [1300, 1200, 800, 300],
    /// ...         [1400, 1100, 700], [1200, 900], [1100]]
    /// >>> premium = [5812, 4908, 5454, 5165, 5214]
    /// >>> rows = [(lob, 1988 + w, 12 * (d + 1), k * v, k * case[w][d], k * premium[w])
    /// ...         for lob, k in [("Auto", 1.0), ("Home", 2.0)]
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> lob, origin, age, values, os, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(
    /// ...     origin, age,
    /// ...     {"paid": values, "case": os, "premium": prem}, keys={"lob": lob})
    /// >>> model = Compartmental(
    /// ...     CompartmentalPriors.gesmann_morris(), tune=100, draws=100, n_sims=200,
    /// ...     chains=2)
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     views = model.one_year_segments(
    /// ...         tri, "paid", "premium", outstanding="case", refit="reweight_only")
    /// >>> views.index, views.cdr.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    /// >>> views.segment(lob="Home").premium[:2]
    /// [11624.0, 9816.0]
    #[pyo3(signature = (
        triangle,
        paid,
        premium,
        *,
        outstanding = None,
        incurred = None,
        k_threshold = 0.7,
        refit = None,
        max_refits = None,
        n_sims = None,
        light_refit = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn one_year_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        paid: &str,
        premium: &str,
        outstanding: Option<&str>,
        incurred: Option<&str>,
        k_threshold: f64,
        refit: Option<&str>,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
        light_refit: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyCompartmentalOneYearFits> {
        let os = compartmental_outstanding(self.inner.fitted, outstanding, incurred)?;
        let mut settings = one_year_settings(k_threshold, refit, max_refits, n_sims)?;
        (settings.light_refit, settings.no_light_refit) = light_refit_arg(light_refit)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year_segments(tri, paid, os, premium, &settings))
            .map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.fit.fit.diagnostics);
        warn_compartmental(py, &warnings)?;
        warn_refits(py, "Compartmental", &segment_refits(&inner.segments))?;
        if let Some(message) = &inner.approximate {
            let message = std::ffi::CString::new(format!("Compartmental: {message}"))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
            PyErr::warn(py, category.as_any(), &message, 1)?;
        }
        Ok(PyCompartmentalOneYearFits { inner })
    }

    /// Paths of the mean outstanding and cumulative paid of each origin of
    /// the premium column at each age of the square, drawn from the priors
    /// alone (Gesmann and Morris, sec. 3.4): a check of a prior set before
    /// fitting. The paths are means, without the likelihood's noise, which
    /// under the preset's effectively flat prior on ``log sigma`` would
    /// swamp them. The seed is derived from ``seed``.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// premium : str
    ///     The premium of each origin.
    /// n_sims : int
    ///     Number of paths; positive.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``{"outstanding": PredictiveDistribution, "paid":
    ///     PredictiveDistribution}``, each with dimensions ``origin`` and
    ///     ``age`` (months).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a prior setting is invalid, ``n_sims`` is 0, or the premium
    ///     column is missing or not positive.
    #[pyo3(signature = (triangle, premium, n_sims = 1000))]
    fn prior_predictive<'py>(
        &self,
        py: Python<'py>,
        triangle: PyRef<'_, PyTriangle>,
        premium: &str,
        n_sims: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let paths = py
            .detach(|| model.prior_predictive(tri, premium, n_sims))
            .map_err(err)?;
        let out = PyDict::new(py);
        out.set_item(
            "outstanding",
            PyPredictiveDistribution {
                inner: paths.outstanding,
            },
        )?;
        out.set_item("paid", PyPredictiveDistribution { inner: paths.paid })?;
        Ok(out)
    }

    /// This model as a candidate of ``diagonal_backtest``, with the
    /// columns it reads: the cumulative ``paid`` column (the backtest's
    /// ``column``), the outstanding (from ``outstanding`` or from
    /// ``incurred``, as ``fit``) and each origin's ``premium``. The
    /// backtest fits the model to the cells valued before each held-out
    /// diagonal and forecasts the paid increments of the diagonal; the
    /// candidate's ``outstanding_backtest`` scores the held-out outstanding
    /// of the same diagonals on the **same fits**. The settings are
    /// ``fit``'s, except that the backtest's ``n_sims`` and ``seed``
    /// replace ``n_sims`` and ``seed``, and the sampler runs once per
    /// split (the whole ``chains`` x ``draws`` posterior of ``fit``, so
    /// size them for the number of diagonals).
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     The cumulative triangle with one segment. It must be the one
    ///     given to ``diagonal_backtest``, with ``paid`` as its ``column``.
    /// paid : str
    ///     Cumulative paid amounts.
    /// premium : str
    ///     The premium of each origin.
    /// outstanding : str, optional
    ///     A column of outstanding amounts.
    /// incurred : str, optional
    ///     A column of incurred amounts.
    ///
    /// Returns
    /// -------
    /// CompartmentalCandidate
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If not exactly one of ``outstanding`` and ``incurred`` is given
    ///     (at most one under ``fitted="paid_only"``).
    #[pyo3(signature = (triangle, paid, premium, *, outstanding = None, incurred = None))]
    fn candidate(
        &self,
        triangle: PyRef<'_, PyTriangle>,
        paid: &str,
        premium: &str,
        outstanding: Option<&str>,
        incurred: Option<&str>,
    ) -> PyResult<PyCompartmentalCandidate> {
        let os = compartmental_outstanding(self.inner.fitted, outstanding, incurred)?;
        Ok(PyCompartmentalCandidate {
            inner: CompartmentalCandidate::new(
                self.inner.clone(),
                &triangle.inner,
                paid,
                os,
                premium,
            ),
        })
    }

    fn __repr__(&self) -> String {
        let (m, s) = (&self.inner, &self.inner.sampler);
        format!(
            "Compartmental(likelihood={:?}, scale={:?}, paid_response={:?}, fitted={:?}, \
             rate_order={:?}, development_levels={:?}, start={:?}, chains={}, tune={}, \
             draws={}, n_sims={}, seed={})",
            name_of(m.likelihood, &LIKELIHOODS),
            name_of(m.scale, &SCALES),
            name_of(m.paid_response, &PAID_RESPONSES),
            name_of(m.fitted, &FITTED),
            name_of(m.rate_order, &RATE_ORDERS),
            name_of(m.development_levels, &DEVELOPMENT_LEVELS),
            name_of(m.start, &STARTS),
            s.chains,
            s.tune,
            s.draws,
            m.n_sims,
            m.seed,
        )
    }
}

/// A ``Compartmental`` for ``diagonal_backtest``, from
/// ``Compartmental.candidate``: the model's settings with the triangle and
/// the paid, outstanding (or incurred) and premium columns it reads.
///
/// Put it in the ``models`` of ``diagonal_backtest`` (a list, or a dict
/// that names it) to score the paid increments of the held-out diagonals
/// beside the other models, and call ``outstanding_backtest`` for the
/// second table, the held-out outstanding of the same diagonals. Both
/// tables use the same fits (one run of the sampler per split, kept by the
/// candidate), whichever is asked for first.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Compartmental, CompartmentalPriors, Triangle
/// >>> paid = [[952, 1529, 2813, 3647, 3724], [849, 1564, 2202, 2432],
/// ...         [983, 2211, 2830], [1657, 2685], [932]]
/// >>> incurred = [[1500, 1400, 900, 400, 100], [1300, 1200, 800, 300],
/// ...             [1400, 1100, 700], [1200, 900], [1100]]
/// >>> premium = [5812, 4908, 5454, 5165, 5214]
/// >>> rows = [(1988 + w, 12 * (d + 1), v, incurred[w][d] + v, premium[w])
/// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
/// >>> origin, age, values, inc, prem = map(list, zip(*rows))
/// >>> tri = Triangle.from_long(
/// ...     origin, age, {"paid": values, "incurred": inc, "premium": prem})
/// >>> model = Compartmental(CompartmentalPriors.gesmann_morris())
/// >>> cand = model.candidate(tri, "paid", "premium", incurred="incurred")
/// >>> cand.name, cand.paid, cand.premium, cand.outstanding_from
/// ('compartmental_gm1', 'paid', 'premium', 'incurred')
#[pyclass(
    name = "CompartmentalCandidate",
    module = "prospicio.reserving",
    frozen
)]
pub(crate) struct PyCompartmentalCandidate {
    inner: CompartmentalCandidate,
}

#[pymethods]
impl PyCompartmentalCandidate {
    /// The model's name in a backtest: ``"compartmental_gm1"`` for model 1,
    /// ``"compartmental_gm2"`` for model 2, ``"compartmental"`` otherwise.
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// The cumulative paid column, the backtest's ``column``.
    #[getter]
    fn paid(&self) -> String {
        self.inner.paid().to_string()
    }

    /// The premium column.
    #[getter]
    fn premium(&self) -> String {
        self.inner.premium().to_string()
    }

    /// The column the outstanding comes from.
    #[getter]
    fn outstanding(&self) -> String {
        match self.inner.outstanding() {
            Outstanding::Column(c) | Outstanding::FromIncurred(c) => c.to_string(),
        }
    }

    /// ``"column"`` when ``outstanding`` is a column of outstanding
    /// amounts, ``"incurred"`` when it is a column of incurred amounts less
    /// paid.
    #[getter]
    fn outstanding_from(&self) -> &'static str {
        match self.inner.outstanding() {
            Outstanding::Column(_) => "column",
            Outstanding::FromIncurred(_) => "incurred",
        }
    }

    /// Scores the held-out **outstanding** of the latest ``n_diagonals``
    /// calendar diagonals, a second table beside ``diagonal_backtest``'s
    /// paid one: for each, the model is fitted to the cells valued before
    /// it (the outstanding of the held-out diagonal is always withheld) and
    /// forecasts the outstanding of the diagonal's cells, scored by the
    /// backtest's metrics. The splits, ``n_sims``, ``seed`` and
    /// ``interval`` are ``diagonal_backtest``'s, so a backtest with the
    /// same arguments and this candidate (or a copy of it in its
    /// ``models``) has fitted the model already and the fits are reused.
    /// The one model of the table is named after the candidate with
    /// ``_outstanding``. A held-out cell without an outstanding in the
    /// triangle, or without a fitted origin or development period, is
    /// counted in ``excluded``.
    ///
    /// Parameters
    /// ----------
    /// n_diagonals : int
    ///     The number of latest calendar diagonals to hold out.
    /// n_sims : int, default 1000
    ///     Simulations per forecast.
    /// seed : int, default 0
    ///     Seed of every fit and simulation.
    /// interval : float, default 0.9
    ///     The central interval whose coverage is scored.
    ///
    /// Returns
    /// -------
    /// Backtest
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a diagonal cannot be held out, the model cannot be fitted or
    ///     a setting is invalid.
    #[pyo3(signature = (n_diagonals, *, n_sims = 1000, seed = 0, interval = 0.9))]
    fn outstanding_backtest(
        &self,
        py: Python<'_>,
        n_diagonals: usize,
        n_sims: usize,
        seed: u64,
        interval: f64,
    ) -> PyResult<PyBacktest> {
        let candidate = &self.inner;
        let (inner, diagonals) = py
            .detach(|| {
                let cells = TriangleFrame::new(candidate.triangle(), candidate.paid(), None)?;
                let bt =
                    candidate.outstanding_backtest(&cells, n_diagonals, n_sims, seed, interval)?;
                let diagonals = diagonal_labels(&cells, &bt);
                Ok((bt, diagonals))
            })
            .map_err(err)?;
        Ok(PyBacktest { inner, diagonals })
    }

    fn __repr__(&self) -> String {
        format!(
            "CompartmentalCandidate(name={}, paid={}, premium={})",
            quoted(self.inner.name()),
            quoted(self.inner.paid()),
            quoted(self.inner.premium())
        )
    }
}

/// Where a compartmental fit's outstanding comes from: exactly one of a
/// column of outstanding amounts and a column of incurred amounts, or
/// under ``fitted="paid_only"``, which does not read it, at most one.
fn compartmental_outstanding<'a>(
    fitted: Fitted,
    outstanding: Option<&'a str>,
    incurred: Option<&'a str>,
) -> PyResult<Outstanding<'a>> {
    match (outstanding, incurred) {
        (Some(c), None) => Ok(Outstanding::Column(c)),
        (None, Some(c)) => Ok(Outstanding::FromIncurred(c)),
        // Not read under paid alone.
        (None, None) if fitted == Fitted::PaidOnly => Ok(Outstanding::Column("")),
        _ => Err(PyValueError::new_err(
            "give exactly one of outstanding (a column of outstanding amounts) and \
             incurred (a column of incurred amounts); under fitted=\"paid_only\" neither \
             is needed",
        )),
    }
}

/// The one-year view of a ``Compartmental`` model, from
/// ``Compartmental.one_year``: the lifetime fit, the opening reserve, the
/// claims development result over the coming year and how reliable each
/// simulation's importance reweighting was, by the premium column's
/// origins.
///
/// ``cdr`` is the joint distribution of the claims development result by
/// origin (dimension ``"origin"``): the opening reserve less the year's
/// paid and the posterior mean reserve after the year; a negative value
/// is an adverse development. ``pareto_k`` holds one Pareto ``k`` per
/// simulation, ``flagged_share`` the share above ``k_threshold``;
/// ``moment_matched``, ``refitted`` and ``unrefitted`` list what happened
/// to the flagged simulations; ``approximate`` says why the view is
/// approximate when it reweighted them only by its model's default.
#[pyclass(
    name = "CompartmentalOneYearFit",
    module = "prospicio.reserving",
    frozen
)]
pub(crate) struct PyCompartmentalOneYearFit {
    inner: BayesOneYearFit<CompartmentalFit>,
}

#[pymethods]
impl PyCompartmentalOneYearFit {
    /// The model fitted to the observed triangle, as ``Compartmental.fit``
    /// gives it.
    #[getter]
    fn fit(&self) -> PyCompartmentalFit {
        PyCompartmentalFit {
            inner: self.inner.fit.clone(),
        }
    }

    /// Origin period of each component, those of the premium column.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .fit
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Each origin's latest observed paid (0 with none).
    #[getter]
    fn latest_paid(&self) -> Vec<f64> {
        self.inner.fit.latest_paid.clone()
    }

    /// The latest paid plus the opening reserve, per origin.
    #[getter]
    fn opening_ultimate(&self) -> Vec<f64> {
        self.inner.opening_ultimate.clone()
    }

    /// Each origin's posterior mean reserve on the observed triangle: the
    /// expected paid to the square's last age less the latest paid.
    #[getter]
    fn opening_reserve(&self) -> Vec<f64> {
        self.inner.opening_reserve.clone()
    }

    /// Joint distribution of the claims development result by origin.
    #[getter]
    fn cdr(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.cdr.clone(),
        }
    }

    /// Each simulation's Pareto ``k``, in order.
    #[getter]
    fn pareto_k(&self) -> Vec<f64> {
        self.inner.pareto_k.clone()
    }

    /// The simulations refitted with NUTS, in order (0-based).
    #[getter]
    fn refitted(&self) -> Vec<usize> {
        self.inner.refitted.clone()
    }

    /// The simulations moment matching rescued, in order (0-based).
    #[getter]
    fn moment_matched(&self) -> Vec<usize> {
        self.inner.moment_matched.clone()
    }

    /// The simulations due for a refit beyond ``max_refits``, reweighted
    /// instead, in order (0-based).
    #[getter]
    fn unrefitted(&self) -> Vec<usize> {
        self.inner.unrefitted.clone()
    }

    /// Why the view is approximate, or ``None``: set when ``refit`` was
    /// left to a model with development effects (model 2) whose default
    /// light refit was turned off (``light_refit=False``), so that it
    /// reweights the flagged simulations only.
    #[getter]
    fn approximate(&self) -> Option<String> {
        self.inner.approximate.clone()
    }

    /// How many of the refits had a diagnostic warning.
    #[getter]
    fn refit_warnings(&self) -> usize {
        self.inner.refit_warnings
    }

    /// The Pareto ``k`` above which a simulation is flagged.
    #[getter]
    fn k_threshold(&self) -> f64 {
        self.inner.k_threshold
    }

    /// The share of simulations whose ``k`` is above ``k_threshold`` (or
    /// NaN), whether or not they were refitted.
    #[getter]
    fn flagged_share(&self) -> f64 {
        self.inner.flagged_share()
    }

    /// One row per origin: ``origin``, ``premium``, ``latest_paid``,
    /// ``opening_ultimate``, ``opening_reserve``, and the ``cdr_mean`` and
    /// ``cdr_std_dev`` of the claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the totals of ``to_frame``'s columns, the
    /// ``flagged_share``, and the ``cdr_mean`` and ``cdr_std_dev`` of the
    /// total claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    fn __repr__(&self) -> String {
        format!(
            "CompartmentalOneYearFit(origins={}, n_sims={}, flagged_share={:.3}, \
             moment_matched={}, refitted={}, unrefitted={})",
            self.inner.fit.origins.len(),
            self.inner.pareto_k.len(),
            self.inner.flagged_share(),
            self.inner.moment_matched.len(),
            self.inner.refitted.len(),
            self.inner.unrefitted.len(),
        )
    }
}

/// A fitted ``Compartmental``: the posterior, its MCMC diagnostics, and the
/// posterior predictive distributions of the reserves and of the future
/// paid and outstanding cells, all from the same simulations.
#[pyclass(name = "CompartmentalFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCompartmentalFit {
    inner: CompartmentalFit,
}

#[pymethods]
impl PyCompartmentalFit {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// Paid by the simulated paid cells, each increment at its own age
    /// (process timing included), not spread; ``basis`` must be ``"paid"``,
    /// and a fit on incurred claims only is refused.
    ///
    /// Parameters
    /// ----------
    /// basis : {"paid", "incurred"}
    ///     What the fit's column is, stated by the caller (no default), so
    ///     that a fit on incurred claims is not discounted by its reporting
    ///     pattern by mistake. ``"paid"`` pays each simulation's reserve
    ///     along that simulation's own pattern. ``"incurred"`` pays each
    ///     reserve (the IBNR) plus the origin's case reserve (its latest
    ///     incurred less ``latest_paid``) along ``paid_pattern``.
    /// rate : float, optional
    ///     A flat annual effective rate. Give ``rate`` or ``curve``.
    /// curve : DiscountCurve, optional
    ///     Spot rates by maturity.
    /// timing : {"mid", "end"}, default "mid"
    ///     Payments at the middle or the end of each development period.
    /// paid_pattern : list of float, optional
    ///     For ``basis="incurred"``: the cumulative paid share at each of
    ///     the fit's pattern ages (with the tail's age when the fit has a
    ///     tail), ending at 1.
    /// latest_paid : list of float, optional
    ///     For ``basis="incurred"``: each origin's latest paid, one per
    ///     component of ``reserves`` in their order.
    ///
    /// Returns
    /// -------
    /// DiscountedReserves
    #[pyo3(signature = (
        *,
        basis,
        rate = None,
        curve = None,
        timing = "mid",
        paid_pattern = None,
        latest_paid = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn discounted(
        &self,
        py: Python<'_>,
        basis: &str,
        rate: Option<f64>,
        curve: Option<PyRef<'_, PyDiscountCurve>>,
        timing: &str,
        paid_pattern: Option<Vec<f64>>,
        latest_paid: Option<Vec<f64>>,
    ) -> PyResult<PyDiscountedReserves> {
        let args = DiscountArgs {
            basis,
            rate,
            curve,
            timing,
            paid_pattern,
            latest_paid,
        };
        discounted(py, &self.inner, args)
    }

    /// Origin period of each component, those of the premium column.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    /// Each origin's premium.
    #[getter]
    fn premium(&self) -> Vec<f64> {
        self.inner.premium.clone()
    }

    /// Each origin's latest observed paid (0 with none).
    #[getter]
    fn latest_paid(&self) -> Vec<f64> {
        self.inner.latest_paid.clone()
    }

    /// The origins with no observed cell, whose effects each simulation
    /// draws from the population.
    #[getter]
    fn population_origins(&self) -> Vec<String> {
        self.inner
            .population_origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Joint posterior predictive distribution of each origin's reserve, the
    /// paid at the square's last age less the latest paid: dimension
    /// ``"origin"``, laid out as ``OdpBootstrapFit.reserves``.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// Joint distribution of the ultimate by origin: ``reserves`` with
    /// each origin's latest paid added in every simulation, the same
    /// dimensions, components and rows, so ``join``, ``select`` and
    /// ``reorder_groups`` keep each simulation's origins together.
    #[getter]
    fn ultimates(&self) -> PyResult<PyPredictiveDistribution> {
        ultimates_of(&self.inner)
    }

    /// Each simulation's ultimate, for a tower, a deal book or the cash-flow
    /// engine: index ``i`` is simulation ``i`` for every origin and every
    /// call, the same as row ``i`` of ``reserves`` and ``ultimates``, and
    /// the same again when the fit is run with the same seed.
    ///
    /// The ultimate is the origin's latest paid plus its reserve in that
    /// simulation: the paid at the square's last age, on the paid basis.
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     The component key (``(2021,)``) or the origin alone; an origin period by its label
    ///     (``"2021"``, ``2021``, ``"2021Q3"``). ``None`` (the default) gives
    ///     the total over every component, summed simulation by simulation.
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (origin = None))]
    fn ultimate_draws(&self, origin: Option<OriginArg>) -> PyResult<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// Each simulation's ultimate over ``premium``: the loss ratio a
    /// contract on the loss ratio reads, in the order of
    /// ``ultimate_draws``. Multiply by the premium for
    /// ``Tower.apply_annual``.
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     As in ``ultimate_draws``; ``None`` for the total.
    /// premium : float
    ///     The subject premium of the origin (of every origin for the
    ///     total), finite and positive.
    ///
    /// Returns
    /// -------
    /// list of float
    fn loss_ratio_draws(&self, origin: Option<OriginArg>, premium: f64) -> PyResult<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// Posterior predictive cumulative paid of each future cell of the
    /// square: dimensions ``"origin"`` and ``"age"`` (months).
    #[getter]
    fn paid(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.paid.clone(),
        }
    }

    /// Posterior predictive outstanding (or incurred, the second response)
    /// of each future cell of the square, laid out as ``paid``; ``None``
    /// under ``fitted="paid_only"``.
    #[getter]
    fn outstanding(&self) -> Option<PyPredictiveDistribution> {
        self.inner
            .outstanding
            .clone()
            .map(|inner| PyPredictiveDistribution { inner })
    }

    /// Each simulation's paid development pattern, to pair with the
    /// simulation's loss ratio in the cash-flow engine: at each age of the
    /// square (to the last age, where the reserves end), the expected
    /// cumulative paid of the ODE for the posterior draw the simulation
    /// took, for an origin with nothing observed (the population's rates,
    /// with each observed age's development effects and none at an age with
    /// no observed cell), over that at the last age. No tail: each pattern
    /// ends at 1 at the square's last age. The paid curve rises, and so does
    /// a running sum of positive incremental means, so with cumulative paid
    /// and no development effects (model 1) or incremental paid (model 2) a
    /// share never falls.
    ///
    /// Row ``i`` of every pattern is simulation ``i`` of ``reserves``,
    /// ``paid``, ``outstanding`` and ``ultimate_draws``.
    ///
    /// Returns
    /// -------
    /// DevelopmentPatterns
    #[getter]
    fn patterns(&self) -> PyDevelopmentPatterns {
        PyDevelopmentPatterns {
            inner: self.inner.patterns.clone(),
        }
    }

    /// Each origin's posterior mean ultimate loss ratio ``RLR_i RRF_i``,
    /// without process error past the square.
    #[getter]
    fn ultimate_loss_ratio(&self) -> Vec<f64> {
        self.inner.ultimate_loss_ratio.clone()
    }

    /// The number of posterior draws whose population rates have
    /// ``k_er <= k_p`` (0 under ``rate_order="ordered"``).
    #[getter]
    fn rate_swaps(&self) -> usize {
        self.inner.rate_swaps
    }

    /// With effects on the rates (each cell its own ``k_er`` and ``k_p``),
    /// the share of posterior draws and observed cells with
    /// ``k_er <= k_p``: a diagnostic, not a warning, since each cell's
    /// means are invariant under the swap on their own. ``None`` when the
    /// rates are shared.
    #[getter]
    fn cell_rate_swaps(&self) -> Option<f64> {
        self.inner.cell_rate_swaps
    }

    /// MCMC diagnostics: a dict with ``max_rhat``, ``min_ess_bulk``,
    /// ``min_ess_tail``, ``divergences`` and ``warnings`` (one message per
    /// failed threshold and one for rate swaps; empty when all is well).
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        diagnostics_dict(py, &self.inner.diagnostics)
    }

    /// Parameter names: on the natural scale ``k_er``, ``k_p``, ``RLR``,
    /// ``RRF`` (the population values, at a zero effect), ``sigma_os``,
    /// ``sigma_pd``, ``RLR[o]`` and ``RRF[o]`` for each origin ``o`` with
    /// an observed cell (at a zero development effect), with origin effects
    /// on the rates ``k_er[o]`` and ``k_p[o]`` too, and with development
    /// effects ``dev_ker[a]``, ``dev_kp[a]``, ``dev_RLR[a]`` and
    /// ``dev_RRF[a]``, each development age ``a`` (months) with an observed
    /// cell's effect on the ``o`` scale; then on the scale the paper prints
    /// ``b_ker``, ``b_kp``, ``b_RLR``, ``b_RRF``, ``tau_RLR``,
    /// ``tau_RRF`` (and ``tau_ker``, ``tau_kp``), ``rho`` (and
    /// ``tau_dev_ker``, ``tau_dev_kp``, ``tau_dev_RLR``, ``tau_dev_RRF``),
    /// ``log_sigma_os`` and ``log_sigma_pd``. Each name in parentheses is
    /// there only when the priors have that effect.
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, one row per draw (chain by chain), one column per
    /// parameter in the order of ``parameter_names``.
    #[getter]
    fn draws(&self) -> Vec<Vec<f64>> {
        let p = self.inner.posterior.dim().max(1);
        self.inner
            .posterior
            .values()
            .chunks(p)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// Number of chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.posterior.chains()
    }

    /// Posterior summary, one row per parameter, as ``MeyersFit.summary``.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fit = &self.inner;
        let s = py.detach(|| fit.summary()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// PSIS-LOO of the observed cells, outstanding cells first and then
    /// paid cells, with each cell's relative efficiency from the chains.
    /// The pointwise term is the full log density of each response, normal
    /// or log-normal (with its ``-log y`` term), as brms' ``log_lik``, on
    /// the response's scale: amounts or loss ratios, cumulative or
    /// incremental paid. Compare it only with fits of the same responses.
    /// ``loo_by_type`` splits the elpd.
    ///
    /// Returns
    /// -------
    /// Elpd
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let fit = &self.inner;
        let l = py.detach(|| fit.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l.loo))
    }

    /// The elpd of ``loo`` split by type: a dict with ``n_outstanding``
    /// (the number of outstanding cells, which come first in the pointwise
    /// values), ``elpd_outstanding`` and ``elpd_paid``.
    ///
    /// Returns
    /// -------
    /// dict
    fn loo_by_type<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let fit = &self.inner;
        let l = py.detach(|| fit.loo()).map_err(err)?;
        let out = PyDict::new(py);
        out.set_item("n_outstanding", l.n_outstanding)?;
        out.set_item("elpd_outstanding", l.elpd_outstanding)?;
        out.set_item("elpd_paid", l.elpd_paid)?;
        Ok(out)
    }

    /// One row per origin: ``origin``, ``premium``, ``latest_paid``, the
    /// ``mean`` and ``std_dev`` of the posterior predictive reserve, and the
    /// posterior mean ``ultimate_loss_ratio``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the total ``premium`` and ``latest_paid``, and the ``mean``
    /// and ``std_dev`` of the posterior predictive total reserve. Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// One row per future paid cell: ``origin``, ``development`` (age in
    /// months), and the ``mean``, ``std_dev``, ``q05``, ``q50`` and ``q95``
    /// of the posterior predictive cumulative paid. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn paid_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.paid_table().map_err(err)?)
    }

    /// The same for the future outstanding cells. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Under ``fitted="paid_only"``, which has no outstanding cells.
    fn outstanding_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.outstanding_table().map_err(err)?)
    }

    fn __repr__(&self) -> String {
        format!(
            "CompartmentalFit(origins={}, draws={}, n_sims={}, rate_swaps={}, warnings={})",
            self.inner.origins.len(),
            self.inner.posterior.n_draws(),
            self.inner.reserves.n_sims(),
            self.inner.rate_swaps,
            self.inner.diagnostics.warnings.len(),
        )
    }
}

/// ``Compartmental.fit_segments``: each segment's own posterior and the
/// joint posterior predictive distributions of the reserves and of the
/// future paid and outstanding cells.
///
/// ``reserves``, ``paid`` and ``outstanding`` have the triangle's keys as
/// their first dimensions, so ``reserves.aggregate(["lob"])`` keeps the
/// dependence between segments. Per-origin lists (``origins``,
/// ``premium``, ``latest_paid``) run over the origins of each segment in
/// turn, like the rows of ``to_frame()`` and the components of
/// ``reserves``. The posterior of one segment (``diagnostics``,
/// ``parameter_names``, ``draws``, ``chains``, ``summary()``, ``loo()``)
/// needs a single-segment fit: for several segments use ``segment(...)`` or
/// ``diagnostics_frame()``.
#[pyclass(name = "CompartmentalFits", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCompartmentalFits {
    inner: CompartmentalFits,
}

impl PyCompartmentalFits {
    fn one(&self, field: &str, instead: &str) -> PyResult<&CompartmentalSegment> {
        single(&self.inner.segments, field, instead)
    }
}

#[pymethods]
impl PyCompartmentalFits {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// Paid by each segment's simulated paid cells, not spread; ``basis``
    /// must be ``"paid"``.
    ///
    /// Parameters
    /// ----------
    /// basis : {"paid", "incurred"}
    ///     What the fit's column is, stated by the caller (no default), so
    ///     that a fit on incurred claims is not discounted by its reporting
    ///     pattern by mistake. ``"paid"`` pays each simulation's reserve
    ///     along that simulation's own pattern. ``"incurred"`` pays each
    ///     reserve (the IBNR) plus the origin's case reserve (its latest
    ///     incurred less ``latest_paid``) along ``paid_pattern``.
    /// rate : float, optional
    ///     A flat annual effective rate. Give ``rate`` or ``curve``.
    /// curve : DiscountCurve, optional
    ///     Spot rates by maturity.
    /// timing : {"mid", "end"}, default "mid"
    ///     Payments at the middle or the end of each development period.
    /// paid_pattern : list of float, optional
    ///     For ``basis="incurred"``: the cumulative paid share at each of
    ///     the fit's pattern ages (with the tail's age when the fit has a
    ///     tail), ending at 1.
    /// latest_paid : list of float, optional
    ///     For ``basis="incurred"``: each origin's latest paid, one per
    ///     component of ``reserves`` in their order.
    ///
    /// Returns
    /// -------
    /// DiscountedReserves
    #[pyo3(signature = (
        *,
        basis,
        rate = None,
        curve = None,
        timing = "mid",
        paid_pattern = None,
        latest_paid = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn discounted(
        &self,
        py: Python<'_>,
        basis: &str,
        rate: Option<f64>,
        curve: Option<PyRef<'_, PyDiscountCurve>>,
        timing: &str,
        paid_pattern: Option<Vec<f64>>,
        latest_paid: Option<Vec<f64>>,
    ) -> PyResult<PyDiscountedReserves> {
        let args = DiscountArgs {
            basis,
            rate,
            curve,
            timing,
            paid_pattern,
            latest_paid,
        };
        discounted(py, &self.inner, args)
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.segments.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner.segments)
    }

    /// Origin period of each per-origin value and component, those of each
    /// segment's premium column.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.origins.iter().map(ToString::to_string))
            .collect()
    }

    /// The premium of each segment's origins, in the order of ``origins``.
    #[getter]
    fn premium(&self) -> Vec<f64> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.premium.iter().copied())
            .collect()
    }

    /// Each origin's latest observed paid, in the order of ``origins``.
    #[getter]
    fn latest_paid(&self) -> Vec<f64> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.latest_paid.iter().copied())
            .collect()
    }

    /// Joint posterior predictive distribution of the reserve by segment
    /// and origin: the triangle's keys and ``"origin"`` are its dimensions,
    /// one component per segment and origin, one row per simulation.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// Joint distribution of the ultimate by segment and origin: ``reserves`` with
    /// each origin's latest paid added in every simulation, the same
    /// dimensions, components and rows, so ``join``, ``select`` and
    /// ``reorder_groups`` keep each simulation's origins together.
    #[getter]
    fn ultimates(&self) -> PyResult<PyPredictiveDistribution> {
        ultimates_of(&self.inner)
    }

    /// Each simulation's ultimate, for a tower, a deal book or the cash-flow
    /// engine: index ``i`` is simulation ``i`` for every origin and every
    /// call, the same as row ``i`` of ``reserves`` and ``ultimates``, and
    /// the same again when the fit is run with the same seed.
    ///
    /// The ultimate is the origin's latest paid plus its reserve in that
    /// simulation: the paid at the square's last age, on the paid basis.
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     The component key, the segment's key values then the origin (``("Auto", 2021)``); an origin period by its label
    ///     (``"2021"``, ``2021``, ``"2021Q3"``). ``None`` (the default) gives
    ///     the total over every component, summed simulation by simulation.
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (origin = None))]
    fn ultimate_draws(&self, origin: Option<OriginArg>) -> PyResult<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// Each simulation's ultimate over ``premium``: the loss ratio a
    /// contract on the loss ratio reads, in the order of
    /// ``ultimate_draws``. Multiply by the premium for
    /// ``Tower.apply_annual``.
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     As in ``ultimate_draws``; ``None`` for the total.
    /// premium : float
    ///     The subject premium of the origin (of every origin for the
    ///     total), finite and positive.
    ///
    /// Returns
    /// -------
    /// list of float
    fn loss_ratio_draws(&self, origin: Option<OriginArg>, premium: f64) -> PyResult<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// Posterior predictive cumulative paid of each future cell of every
    /// segment: dimensions the keys, ``"origin"`` and ``"age"`` (months),
    /// from the same simulations as ``reserves``.
    #[getter]
    fn paid(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.paid.clone(),
        }
    }

    /// Posterior predictive outstanding (or incurred) of each future cell of
    /// every segment, laid out as ``paid``; ``None`` under
    /// ``fitted="paid_only"``.
    #[getter]
    fn outstanding(&self) -> Option<PyPredictiveDistribution> {
        self.inner
            .outstanding
            .clone()
            .map(|inner| PyPredictiveDistribution { inner })
    }

    /// Each simulation's paid development pattern, to pair with the
    /// simulation's loss ratio in the cash-flow engine: at each age of the
    /// square (to the last age, where the reserves end), the expected
    /// cumulative paid of the ODE for the posterior draw the simulation
    /// took, for an origin with nothing observed (the population's rates,
    /// with each observed age's development effects and none at an age with
    /// no observed cell), over that at the last age. No tail: each pattern
    /// ends at 1 at the square's last age. The paid curve rises, and so does
    /// a running sum of positive incremental means, so with cumulative paid
    /// and no development effects (model 1) or incremental paid (model 2) a
    /// share never falls.
    ///
    /// Row ``i`` of every pattern is simulation ``i`` of ``reserves``,
    /// ``paid``, ``outstanding`` and ``ultimate_draws``.
    ///
    /// Returns
    /// -------
    /// DevelopmentPatterns
    #[getter]
    fn patterns(&self) -> PyDevelopmentPatterns {
        PyDevelopmentPatterns {
            inner: self.inner.patterns.clone(),
        }
    }

    /// The number of posterior draws with ``k_er <= k_p`` in each segment,
    /// in index order.
    #[getter]
    fn rate_swaps(&self) -> Vec<usize> {
        self.inner
            .segments
            .fits
            .iter()
            .map(|s| s.fit.rate_swaps)
            .collect()
    }

    /// MCMC diagnostics of a single-segment fit, as
    /// ``CompartmentalFit.diagnostics``.
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        diagnostics_dict(
            py,
            &self
                .one("diagnostics", "diagnostics_frame()")?
                .fit
                .diagnostics,
        )
    }

    /// Parameter names of a single-segment fit, as
    /// ``CompartmentalFit.parameter_names``.
    #[getter]
    fn parameter_names(&self) -> PyResult<Vec<String>> {
        Ok(self
            .one("parameter_names", "")?
            .fit
            .posterior
            .names()
            .to_vec())
    }

    /// Posterior draws of a single-segment fit, as
    /// ``CompartmentalFit.draws``.
    #[getter]
    fn draws(&self) -> PyResult<Vec<Vec<f64>>> {
        let posterior = &self.one("draws", "")?.fit.posterior;
        let p = posterior.dim().max(1);
        Ok(posterior.values().chunks(p).map(<[f64]>::to_vec).collect())
    }

    /// Number of chains of a single-segment fit.
    #[getter]
    fn chains(&self) -> PyResult<usize> {
        Ok(self.one("chains", "")?.fit.posterior.chains())
    }

    /// Posterior summary of a single-segment fit, as
    /// ``CompartmentalFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let one = &self.one("summary()", "")?.fit;
        let s = py.detach(|| one.summary()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// PSIS-LOO of a single-segment fit's observed cells, as
    /// ``CompartmentalFit.loo``.
    ///
    /// Returns
    /// -------
    /// Elpd
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the fit has several segments.
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let one = &self.one("loo()", "")?.fit;
        let l = py.detach(|| one.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l.loo))
    }

    /// One row per segment: the key columns, then ``max_rhat``,
    /// ``min_ess_bulk``, ``min_ess_tail``, ``divergences`` and
    /// ``warnings`` (the segment's messages joined by ``"; "``, empty when
    /// every threshold holds). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn diagnostics_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        segment_diagnostics_frame(py, &self.inner.segments, |s| &s.fit.diagnostics)
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``premium``, ``latest_paid``, the ``mean`` and ``std_dev`` of the
    /// posterior predictive reserve, and the posterior mean
    /// ``ultimate_loss_ratio``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the total ``premium`` and
    /// ``latest_paid``, and the ``mean`` and ``std_dev`` of the segment's
    /// posterior predictive total reserve. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// One row per segment and future paid cell, as
    /// ``CompartmentalFit.paid_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn paid_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.paid_table().map_err(err)?)
    }

    /// The same for the future outstanding cells. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Under ``fitted="paid_only"``, which has no outstanding cells.
    fn outstanding_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.outstanding_table().map_err(err)?)
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint
    /// distributions (same dimensions).
    ///
    /// Returns
    /// -------
    /// CompartmentalFits
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        Ok(Self {
            inner: self.inner.segment(&keys).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        let segments = &self.inner.segments;
        let warnings: usize = segments
            .fits
            .iter()
            .map(|s| s.fit.diagnostics.warnings.len())
            .sum();
        format!(
            "CompartmentalFits({}origins={}, n_sims={}, warnings={warnings})",
            segments_prefix(segments),
            self.origins().len(),
            self.inner.reserves.n_sims(),
        )
    }
}

/// The one-year view of a ``Compartmental`` in every segment of a
/// triangle, from ``Compartmental.one_year_segments``: each segment's
/// posterior, opening reserve and Pareto ``k``, and the joint claims
/// development result across segments.
///
/// ``cdr`` is one joint distribution of the claims development result with
/// the triangle's keys and ``"origin"`` as dimensions, so
/// ``cdr.aggregate(["lob"])`` keeps the dependence between segments.
/// Per-origin lists (``origins``, ``premium``, ``latest_paid``,
/// ``opening_ultimate``, ``opening_reserve``) run over the origins of each
/// segment in turn, like the rows of ``to_frame()`` and the components of
/// ``cdr``. ``diagnostics``, ``parameter_names``, ``draws``, ``chains``,
/// ``summary()``, ``loo()``, ``pareto_k``, ``moment_matched``, ``refitted``,
/// ``unrefitted``, ``refit_warnings`` and ``flagged_share`` are one
/// segment's and need a single-segment fit: for several segments use
/// ``segment(...)``, ``totals_frame()`` (which has the flagged shares) or
/// ``diagnostics_frame()``. ``approximate`` says why the views are
/// approximate when they reweighted the flagged simulations only by their
/// model's default (``light_refit=False``).
#[pyclass(
    name = "CompartmentalOneYearFits",
    module = "prospicio.reserving",
    frozen
)]
pub(crate) struct PyCompartmentalOneYearFits {
    inner: CompartmentalOneYearFits,
}

impl PyCompartmentalOneYearFits {
    fn one(
        &self,
        field: &str,
        instead: &str,
    ) -> PyResult<&BayesOneYearSegment<CompartmentalSegment>> {
        single(&self.inner.segments, field, instead)
    }
}

#[pymethods]
impl PyCompartmentalOneYearFits {
    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.segments.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner.segments)
    }

    /// Origin period of each per-origin value and component, those of each
    /// segment's premium column.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.fit.origins.iter().map(ToString::to_string))
            .collect()
    }

    /// The premium of each segment's origins, in the order of ``origins``.
    #[getter]
    fn premium(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.fit.fit.premium.clone())
    }

    /// Each origin's latest observed paid, in the order of ``origins``.
    #[getter]
    fn latest_paid(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.fit.fit.latest_paid.clone())
    }

    /// The latest paid plus the opening reserve, per origin.
    #[getter]
    fn opening_ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.opening_ultimate.clone())
    }

    /// Each origin's posterior mean reserve on the observed triangle.
    #[getter]
    fn opening_reserve(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.opening_reserve.clone())
    }

    /// Joint distribution of the claims development result by segment and
    /// origin: the triangle's keys and ``"origin"`` are its dimensions, one
    /// component per segment and origin, one row per simulation.
    #[getter]
    fn cdr(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.cdr.clone(),
        }
    }

    /// Why the views are approximate, or ``None``: set when ``refit`` was
    /// left to a model with development effects (model 2) whose default
    /// light refit was turned off (``light_refit=False``), so that it
    /// reweights the flagged simulations only.
    #[getter]
    fn approximate(&self) -> Option<String> {
        self.inner.approximate.clone()
    }

    /// MCMC diagnostics of a single-segment fit's posterior, as
    /// ``CompartmentalFit.diagnostics``.
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        diagnostics_dict(
            py,
            &self
                .one("diagnostics", "diagnostics_frame()")?
                .fit
                .fit
                .diagnostics,
        )
    }

    /// Parameter names of a single-segment fit.
    #[getter]
    fn parameter_names(&self) -> PyResult<Vec<String>> {
        Ok(self
            .one("parameter_names", "")?
            .fit
            .fit
            .posterior
            .names()
            .to_vec())
    }

    /// Posterior draws of a single-segment fit, one list per draw.
    #[getter]
    fn draws(&self) -> PyResult<Vec<Vec<f64>>> {
        let posterior = &self.one("draws", "")?.fit.fit.posterior;
        let p = posterior.dim().max(1);
        Ok(posterior.values().chunks(p).map(<[f64]>::to_vec).collect())
    }

    /// Number of chains of a single-segment fit.
    #[getter]
    fn chains(&self) -> PyResult<usize> {
        Ok(self.one("chains", "")?.fit.fit.posterior.chains())
    }

    /// Each simulation's Pareto ``k`` in a single-segment fit.
    #[getter]
    fn pareto_k(&self) -> PyResult<Vec<f64>> {
        Ok(self.one("pareto_k", "")?.pareto_k.clone())
    }

    /// The simulations of a single-segment fit refitted with NUTS
    /// (0-based).
    #[getter]
    fn refitted(&self) -> PyResult<Vec<usize>> {
        Ok(self.one("refitted", "")?.refitted.clone())
    }

    /// The simulations of a single-segment fit that moment matching
    /// rescued (0-based).
    #[getter]
    fn moment_matched(&self) -> PyResult<Vec<usize>> {
        Ok(self.one("moment_matched", "")?.moment_matched.clone())
    }

    /// The simulations of a single-segment fit due for a refit beyond
    /// ``max_refits``, reweighted instead (0-based).
    #[getter]
    fn unrefitted(&self) -> PyResult<Vec<usize>> {
        Ok(self.one("unrefitted", "")?.unrefitted.clone())
    }

    /// How many of a single-segment fit's refits had a diagnostic warning.
    #[getter]
    fn refit_warnings(&self) -> PyResult<usize> {
        Ok(self.one("refit_warnings", "")?.refit_warnings)
    }

    /// The Pareto ``k`` above which a simulation is flagged.
    #[getter]
    fn k_threshold(&self) -> f64 {
        self.inner.segments.fits[0].k_threshold
    }

    /// The share of a single-segment fit's simulations whose ``k`` is
    /// above ``k_threshold`` (or NaN).
    #[getter]
    fn flagged_share(&self) -> PyResult<f64> {
        Ok(self.one("flagged_share", "totals_frame()")?.flagged_share())
    }

    /// Posterior summary of a single-segment fit, as
    /// ``CompartmentalFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let one = &self.one("summary()", "")?.fit.fit;
        let s = py.detach(|| one.summary()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// PSIS-LOO of a single-segment fit's observed cells, as
    /// ``CompartmentalFit.loo``.
    ///
    /// Returns
    /// -------
    /// Elpd
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the fit has several segments.
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let one = &self.one("loo()", "")?.fit.fit;
        let l = py.detach(|| one.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l.loo))
    }

    /// One row per segment: the key columns, then ``max_rhat``,
    /// ``min_ess_bulk``, ``min_ess_tail``, ``divergences`` and
    /// ``warnings`` (the segment's messages joined by ``"; "``, empty when
    /// every threshold holds). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn diagnostics_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        segment_diagnostics_frame(py, &self.inner.segments, |s| &s.fit.fit.diagnostics)
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``premium``, ``latest_paid``, ``opening_ultimate``,
    /// ``opening_reserve``, and the ``cdr_mean`` and ``cdr_std_dev`` of the
    /// claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the totals of ``to_frame()``'s
    /// columns, the ``flagged_share``, and the ``cdr_mean`` and
    /// ``cdr_std_dev`` of the segment's total claims development result.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The one-year view of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint claims
    /// development result (same dimensions).
    ///
    /// Returns
    /// -------
    /// CompartmentalOneYearFits
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        Ok(Self {
            inner: self.inner.segment(&keys).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        let segments = &self.inner.segments;
        let detail = match segments.fits.as_slice() {
            [one] => format!(
                ", flagged_share={:?}, refitted={}",
                one.flagged_share(),
                one.refitted.len()
            ),
            _ => String::new(),
        };
        format!(
            "CompartmentalOneYearFits({}origins={}, n_sims={}{detail})",
            segments_prefix(segments),
            self.origins().len(),
            self.inner.cdr.n_sims(),
        )
    }
}
