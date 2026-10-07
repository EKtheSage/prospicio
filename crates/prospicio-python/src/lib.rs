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
mod distributions;
mod models;
mod pareto;
mod pricing;
mod reinsurance;
mod reserving;
mod risk;

fn to_py(e: prospicio_core::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pymodule]
mod prospicio_native {
    #[pymodule_export]
    use super::aggregate::{PyCompoundReport, PyEventSet, fft, panjer, simulate_events};
    #[pymodule_export]
    use super::distributions::{
        PyDiscretizationReport, PyGrid, PyLognormal, PyNegativeBinomial, PyPoisson,
        PyPredictiveDistribution, PySampled, from_json, to_json,
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
    use super::pareto::{
        PyBinomial, PyCustom, PyGamma, PyGeneralizedPareto, PyLogAffinePareto, PyLoglogistic,
        PyMixture, PyPareto, PyPiecewisePareto, PyTweedie, PyWeibull, claim_count,
        local_pareto_to_piecewise,
    };
    #[pymodule_export]
    use super::pricing::{
        PyCollectiveModel, PyMbbefd, PyPortfolioPrice, PyPrice, PyRiskProfile, PyTabulatedCurve,
        PyTowerModel, alpha_between_frequencies, alpha_between_frequency_and_layer,
        alpha_between_layers, fit_pml_curve, fit_references, ilf, loss_elimination_ratio,
        match_tower, pareto_extrapolation, price, price_portfolio, severity_exposure_curve,
    };
    #[pymodule_export]
    use super::reinsurance::{PyLayer, PyTower, PyTowerGrids};
    #[pymodule_export]
    use super::reserving::{
        PyBenktander, PyBornhuetterFerguson, PyCapeCod, PyCapeCodFit, PyChainLadder,
        PyChainLadderFit, PyClaimsDevelopmentResult, PyClarkCapeCod, PyClarkFit, PyClarkLdf,
        PyExpectedLoss, PyExpectedLossFit, PyMack, PyMackBootstrap, PyMackFit, PyOdpBootstrap,
        PyOdpBootstrapFit, PyOneYearFit, PyTailBondy, PyTailConstant, PyTailCurve, PyTailLogLinear,
        PyTriangle,
    };
    #[pymodule_export]
    use super::risk::{
        PyAllocation, PyArchimedeanCopula, PyDistortion, PyGaussianCopula, PyGpd, PyPotTail,
        PyStudentTCopula, allocate, capital, covar, entropic, esscher, esscher_allocation, hill,
        iman_conover, marginal_expected_shortfall, mean_excess, simulate,
    };
}
