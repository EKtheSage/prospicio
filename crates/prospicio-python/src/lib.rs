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
mod book;
mod cashflow;
mod casualty;
mod counts;
mod credibility;
mod discrete;
mod distributions;
mod excess;
mod experience;
mod grouped_fit;
mod models;
mod natural;
#[cfg(feature = "nn")]
mod nn;
mod normal;
mod odp_glm;
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
        PyEventLossTable, PyEventSet, PyExposure, PyGridAudit, PyGridSize, PyJointGrid,
        PyOccurrencePml, PyPicks, PyRenewalRuin, PyRuin, PyTimeline, audit_grid,
        audit_portfolio_grid, compound_moments, eventual_ruin, fft, fft_auto, fft_bivariate,
        layer_loss_picks, occurrence_exceedance, occurrence_pml, panjer, recommend_grid,
        recommend_portfolio_grid, round_bucket, simulate_events, wiener_hopf,
        year_correlation_for_ratio, ylt_hash,
    };
    #[pymodule_export]
    use super::book::{PyBookCashflows, PyDealBook, PyDealRun};
    #[pymodule_export]
    use super::cashflow::{
        PyCashflowTerms, PyCashflows, PyCollateralRequirement, PyPremiumAdjustment,
    };
    #[pymodule_export]
    use super::casualty::{
        PyCasualtyProfile, PyCasualtyRating, PyElfTable, PyIlfTable, PyParetoAnchor, poisson_tail,
    };
    #[pymodule_export]
    use super::counts::PyCount;
    #[pymodule_export]
    use super::credibility::{
        PyBuhlmannStraubFit, PyHierarchicalCredibilityFit, PyLimitedFluctuation, buhlmann_straub,
        credibility_premiums, hierarchical_credibility,
    };
    #[pymodule_export]
    use super::discrete::PyDiscrete;
    #[pymodule_export]
    use super::distributions::{
        PyDiscretizationReport, PyGrid, PyLognormal, PyNegativeBinomial, PyPoisson,
        PyPredictiveDistribution, PySampled, from_json, to_json,
    };
    #[pymodule_export]
    use super::excess::{
        PyExcessLayer, PyExcessPattern, PyParetoShape, PyScaleShift, layer_at_shift,
        layer_exposure, layer_share,
    };
    #[pymodule_export]
    use super::experience::{
        PyAsIf, PyBurningCost, PyDevelopmentPattern, PyIndexClause, PyLargeLossExperience,
        PyPreparedLosses,
    };
    #[pymodule_export]
    use super::grouped_fit::{
        PyGroupedFit, PyLevDistanceFit, fit_grouped, fit_lev_distance, grouped_goodness_of_fit,
    };
    #[pymodule_export]
    use super::models::{
        PyBayesGlm, PyBayesGlmFit, PyBayesGlmm, PyBayesGlmmFit, PyBayesStacking, PyCoding,
        PyCvPath, PyDesign, PyDoubleGlm, PyDoubleGlmFit, PyDraws, PyElasticNet, PyElasticNetFit,
        PyElpd, PyGam, PyGamFit, PyGlm, PyGlmFit, PyGlmm, PyGlmmFit, PyHierarchicalStacking,
        PyMarsFit, PyNegativeBinomialFit, PyNegativeBinomialGlm, PyRelaxedCv, PyStackingFit,
        PyTargetEncoding, PyTerms, PyTweedieFit, PyTweedieGlm, actual_vs_expected,
        confusion_matrix, crps, deviance, double_lift, elpd_loo, elpd_waic, f_test, gini,
        group_k_fold, k_fold, ks_uniform, lift, likelihood_ratio_test, log_score, loo_pit,
        loss_ratio_chart, lppd, mars, mcmc_diagnostics, normalized_gini, permutation, pinball_loss,
        pit, pit_from_draws, pit_histogram, ppc_check, pseudo_bma_weights, quantile_residuals, roc,
        simulate_from_means, stacking_weights, target_encode, time_ordered, tweedie_profile, vif,
    };
    #[pymodule_export]
    use super::natural::{
        PyNaturalPrice, PyPentagon, PyPortfolio, PyPricingBounds, PyStandAlone,
        calibrate_classical, classical_premium, distortion_envelope,
    };
    #[cfg(feature = "nn")]
    #[pymodule_export]
    use super::nn::{
        PyAttentionCann, PyAttentionCannFit, PyCann, PyCannFit, PyOnnxModel, PyOnnxPredictor,
        PyTokenAttention,
    };
    #[pymodule_export]
    use super::normal::{PyNormal, PyOgive};
    #[pymodule_export]
    use super::odp_glm::{PyOdpGlm, PyOdpGlmFit};
    #[pymodule_export]
    use super::pareto::{
        PyAffine, PyBeta, PyBinomial, PyBurr, PyCustom, PyGamma, PyGeneralizedPareto,
        PyInverseGamma, PyInverseGaussian, PyLogAffinePareto, PyLoglogistic, PyMixture, PyPareto,
        PyPiecewisePareto, PyPolicyLimits, PySeverityFit, PySizeOfLoss, PyTruncated, PyTweedie,
        PyWeibull, claim_count, fit_severity, goodness_of_fit, local_pareto_to_piecewise,
        product_limit,
    };
    #[pymodule_export]
    use super::payout::{
        PyIrrResult, PyPaymentPattern, PyPayoutPrice, irr, leverage, moic, npv, price_over_payout,
        tiered_fee,
    };
    #[pymodule_export]
    use super::pnl::{
        PyBook, PyExpense, PyGroup, PyLedger, PyLeg, derive_premium, occurrence_source,
        pnl_program, pnl_program_stated, round_premium, size_premium, stack,
    };
    #[pymodule_export]
    use super::pricing::{
        PyCollectiveModel, PyMbbefd, PyPortfolioPrice, PyPrice, PyQuote, PyRiskProfile,
        PyTabulatedCurve, PyTowerModel, alpha_between_frequencies,
        alpha_between_frequency_and_layer, alpha_between_layers, censored_las, fit_pml_curve,
        fit_references, ilf, layer_cv, loss_elimination_ratio, match_tower, pareto_extrapolation,
        price, price_portfolio, severity_exposure_curve,
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
        PyBacktest, PyBayesBf, PyBayesBfFit, PyBayesBfNormalised, PyBayesBfNormalisedFit,
        PyBayesOdp, PyBayesOdpFit, PyBayesOdpFits, PyBayesOneYearFit, PyBayesOneYearFits,
        PyBenktander, PyBornhuetterFerguson, PyCapeCod, PyCapeCodFit, PyChainLadder,
        PyChainLadderFit, PyClaimsDevelopmentResult, PyClarkCapeCod, PyClarkFit, PyClarkLdf,
        PyCompartmental, PyCompartmentalCandidate, PyCompartmentalFit, PyCompartmentalFits,
        PyCompartmentalOneYearFit, PyCompartmentalOneYearFits, PyCompartmentalPriors,
        PyExpectedLoss, PyExpectedLossFit, PyGlmCandidate, PyLayerSharePrior, PyMack,
        PyMackBootstrap, PyMackBootstrapFit, PyMackFit, PyMeyers, PyMeyersFit, PyMeyersFits,
        PyOdpBootstrap, PyOdpBootstrapFit, PyOneYearFit, PyTailBondy, PyTailConstant, PyTailCurve,
        PyTailLogLinear, PyTriangle, diagonal_backtest,
    };
    #[pymodule_export]
    use super::risk::{
        PyAllocation, PyArchimedeanCopula, PyDistortion, PyGaussianCopula, PyGpd,
        PyLevelAllocation, PyNestedArchimedeanCopula, PyPairCopula, PyPotTail, PyStudentTCopula,
        PyVineCopula, allocate, bootstrap_se, calibrate, capital, co_statistics, covar, entropic,
        equal_risk, esscher, esscher_allocation, haircut, hill, iman_conover,
        marginal_expected_shortfall, mean_excess, rearrange, risk_ratios, shapley, simulate,
        statistics, threshold_diagnostics, worst_var,
    };
}
