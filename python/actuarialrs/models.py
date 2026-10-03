"""Models: terms and design matrices, GLMs and GAMs, metrics, resampling,
and MCMC diagnostics (docs/design/models.md)."""

from .actuarialrs_native import (
    Coding,
    Design,
    Gam,
    GamFit,
    Glm,
    GlmFit,
    Terms,
    crps,
    deviance,
    gini,
    group_k_fold,
    k_fold,
    lift,
    mcmc_diagnostics,
    time_ordered,
)

__all__ = [
    "Terms",
    "Coding",
    "Design",
    "Glm",
    "GlmFit",
    "Gam",
    "GamFit",
    "deviance",
    "gini",
    "lift",
    "crps",
    "k_fold",
    "group_k_fold",
    "time_ordered",
    "mcmc_diagnostics",
]
