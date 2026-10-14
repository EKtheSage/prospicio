//! Python bindings, built with maturin from `python/`.
//!
//! Wrappers convert arguments, release the GIL around any real work, and map
//! [`prospicio_core::Error`] to Python exceptions. They hold no numerical code.
//! Each lane keeps its wrappers in its own module; this file only registers
//! them.
//!
//! The `///` comments on `#[pyclass]` and `#[pymethods]` items are the Python
//! docstrings, written in numpydoc style. `cargo xtask python` copies them
//! into `python/prospicio/prospicio_native.pyi` and the Python docs site
//! (docs/architecture.md, "Documentation").

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod aggregate;
mod counts;
mod credibility;
mod discrete;
mod distributions;
mod experience;
mod models;
mod natural;
mod pareto;
mod payout;
mod pnl;
mod pricing;
mod ratemaking;
mod reinsurance;
mod reserving;
mod risk;

fn to_py(e: prospicio_core::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pymodule]
mod prospicio_native {
    #[pymodule_export]
    use super::aggregate::{
        PyApproximation, PyBivariateReport, PyClash, PyCompoundMoments, PyCompoundReport,
        PyEventLossTable, PyEventSet, PyGridSize, PyJointGrid, PyOccurrencePml, PyPicks,
        PyRenewalRuin, PyRuin, compound_moments, eventual_ruin, fft, fft_auto, fft_bivariate,
        layer_loss_picks, occurrence_exceedance, occurrence_pml, panjer, recommend_grid,
        round_bucket, simulate_events, wiener_hopf,
    };
    #[pymodule_export]
    use super::counts::PyCount;
    #[pymodule_export]
    use super::credibility::{PyBuhlmannStraubFit, PyLimitedFluctuation, buhlmann_straub};
    #[pymodule_export]
    use super::discrete::PyDiscrete;
    #[pymodule_export]
    use super::distributions::{
        PyDiscretizationReport, PyGrid, PyLognormal, PyNegativeBinomial, PyPoisson,
        PyPredictiveDistribution, PySampled, from_json, to_json,
    };
    #[pymodule_export]
    use super::experience::{
        PyAsIf, PyBurningCost, PyDevelopmentPattern, PyIndexClause, PyLargeLossExperience,
        PyPreparedLosses,
    };
    #[pymodule_export]
    use super::models::{
        PyBayesGlm, PyBayesGlmFit, PyBayesStacking, PyCoding, PyCvPath, PyDesign, PyElasticNet,
        PyElasticNetFit, PyElpd, PyGam, PyGamFit, PyGlm, PyGlmFit, PyHierarchicalStacking,
        PyStackingFit, PyTerms, actual_vs_expected, crps, deviance, elpd_loo, elpd_waic, gini,
        group_k_fold, k_fold, ks_uniform, lift, log_score, lppd, mcmc_diagnostics, pinball_loss,
        pit, pit_from_draws, pit_histogram, pseudo_bma_weights, simulate_from_means,
        stacking_weights, time_ordered,
    };
    #[pymodule_export]
    use super::natural::{
        PyNaturalPrice, PyPentagon, PyPortfolio, PyPricingBounds, PyStandAlone,
        calibrate_classical, classical_premium, distortion_envelope,
    };
    #[pymodule_export]
    use super::pareto::{
        PyBeta, PyBinomial, PyBurr, PyCustom, PyGamma, PyGeneralizedPareto, PyInverseGamma,
        PyInverseGaussian, PyLogAffinePareto, PyLoglogistic, PyMixture, PyPareto,
        PyPiecewisePareto, PyTruncated, PyTweedie, PyWeibull, claim_count,
        local_pareto_to_piecewise,
    };
    #[pymodule_export]
    use super::payout::{PyPaymentPattern, PyPayoutPrice, price_over_payout};
    #[pymodule_export]
    use super::pnl::{
        PyBook, PyExpense, PyGroup, PyLedger, PyLeg, derive_premium, occurrence_source,
        pnl_program, round_premium, size_premium, stack,
    };
    #[pymodule_export]
    use super::pricing::{
        PyCollectiveModel, PyMbbefd, PyPortfolioPrice, PyPrice, PyQuote, PyRiskProfile,
        PyTabulatedCurve, PyTowerModel, alpha_between_frequencies,
        alpha_between_frequency_and_layer, alpha_between_layers, fit_pml_curve, fit_references,
        ilf, loss_elimination_ratio, match_tower, pareto_extrapolation, price, price_portfolio,
        severity_exposure_curve,
    };
    #[pymodule_export]
    use super::ratemaking::{
        PyLossRatioExperience, PyPeriod, PyProvisions, PyPurePremiumExperience, PyRateHistory,
        PyRateLevel, PyTrendFit, credibility_weighted, premium_based_expenses, trend_factor,
        trended_present_rates, year_fraction,
    };
    #[pymodule_export]
    use super::reinsurance::{
        PyAggregateLayerStatistics, PyAnnualGrids, PyCessionStatistics, PyLayer, PyLayerStatistics,
        PyLossSensitivePremium, PyTower, PyTowerGrids, PyTowerPrice, PyViewPrice,
        aggregate_layer_statistics, occurrence_cession_statistics, occurrence_layer_statistics,
    };
    #[pymodule_export]
    use super::reserving::{
        PyBayesBf, PyBayesBfFit, PyBayesOdp, PyBayesOdpFit, PyBayesOdpFits, PyBenktander,
        PyBornhuetterFerguson, PyCapeCod, PyCapeCodFit, PyChainLadder, PyChainLadderFit,
        PyClaimsDevelopmentResult, PyClarkCapeCod, PyClarkFit, PyClarkLdf, PyCompartmental,
        PyCompartmentalFit, PyCompartmentalPriors, PyExpectedLoss, PyExpectedLossFit, PyMack,
        PyMackBootstrap, PyMackBootstrapFit, PyMackFit, PyMeyers, PyMeyersFit, PyMeyersFits,
        PyOdpBootstrap, PyOdpBootstrapFit, PyOneYearFit, PyTailBondy, PyTailConstant, PyTailCurve,
        PyTailLogLinear, PyTriangle,
    };
    #[pymodule_export]
    use super::risk::{
        PyAllocation, PyArchimedeanCopula, PyDistortion, PyGaussianCopula, PyGpd, PyPotTail,
        PyStudentTCopula, allocate, calibrate, capital, covar, entropic, esscher,
        esscher_allocation, hill, iman_conover, marginal_expected_shortfall, mean_excess, simulate,
    };
}
