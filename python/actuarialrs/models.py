"""Models: terms and design matrices, GLMs, elastic nets and GAMs, metrics, resampling,
tuning, and MCMC diagnostics (docs/design/models.md)."""

import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .actuarialrs_native import (
    Coding,
    CvPath,
    Design,
    ElasticNet,
    ElasticNetFit,
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
    ks_uniform,
    lift,
    log_score,
    mcmc_diagnostics,
    pit,
    pit_from_draws,
    pit_histogram,
    time_ordered,
)

__all__ = [
    "Terms",
    "Coding",
    "Design",
    "CvPath",
    "Glm",
    "GlmFit",
    "ElasticNet",
    "ElasticNetFit",
    "Gam",
    "GamFit",
    "deviance",
    "gini",
    "lift",
    "crps",
    "log_score",
    "pit",
    "pit_from_draws",
    "pit_histogram",
    "ks_uniform",
    "k_fold",
    "group_k_fold",
    "time_ordered",
    "mcmc_diagnostics",
    "deviance_score",
    "cross_validate",
    "grid_search",
    "random_search",
    "log_uniform",
    "SearchResult",
]


def deviance_score(family, theta=None, power=None):
    """A score for ``cross_validate``: the family's mean deviance on the
    test rows, ``sum(w * d) / sum(w)`` with the test design's weights.

    Parameters
    ----------
    family : str
        As ``Glm``.
    theta : float, optional
    power : float, optional

    Returns
    -------
    callable
        ``score(y_test, predicted, test_design) -> float``.

    Examples
    --------
    >>> from actuarialrs.models import Design, deviance_score
    >>> d = Design([[1.0, 1.0]], ["(Intercept)"])
    >>> deviance_score("gaussian")([1.0, 3.0], [2.0, 2.0], d)
    1.0
    """

    def score(y, pred, design):
        w = design.weights
        return deviance(family, y, pred, weights=w, theta=theta, power=power) / sum(w)

    return score


def cross_validate(model, design, y, splits, score, n_jobs=None):
    """Fits ``model`` on each split's training rows and scores it on the
    test rows.

    Folds run on threads; the Rust fits release the GIL, so they run in
    parallel.

    Parameters
    ----------
    model : object
        Anything with ``fit(design, y)`` returning an object with
        ``predict(design)``: ``Glm``, ``ElasticNet``, ``Gam``.
    design : Design
    y : list of float
    splits : list of (list of int, list of int)
        ``(train, test)`` rows, as ``k_fold`` returns.
    score : callable
        ``score(y_test, predicted, test_design) -> float``, a loss (lower is
        better), such as ``deviance_score("poisson")``.
    n_jobs : int, optional
        Threads; one per split by default.

    Returns
    -------
    list of float
        One score per split.

    Examples
    --------
    >>> from actuarialrs.models import Design, Glm, cross_validate, deviance_score, k_fold
    >>> d = Design([[1.0] * 6, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], ["(Intercept)", "x"])
    >>> y = [1.0, 2.9, 5.1, 7.0, 8.9, 11.2]
    >>> scores = cross_validate(Glm("gaussian"), d, y, k_fold(6, 3, 1), deviance_score("gaussian"))
    >>> len(scores)
    3
    """
    if len(y) != design.n_rows:
        raise ValueError(f"{len(y)} responses for {design.n_rows} design rows")

    def one(split):
        train, test = split
        fitted = model.fit(design.select(train), [y[i] for i in train])
        test_design = design.select(test)
        return score([y[i] for i in test], fitted.predict(test_design), test_design)

    with ThreadPoolExecutor(max_workers=n_jobs or max(1, len(splits))) as pool:
        return list(pool.map(one, splits))


@dataclass
class SearchResult:
    """The result of ``grid_search`` or ``random_search``.

    Attributes
    ----------
    scores : list of (object, float)
        Each candidate with its mean score over the splits.
    best : int
        Index of the lowest mean score.
    """

    scores: list
    best: int

    @property
    def best_candidate(self):
        """The candidate with the lowest mean score."""
        return self.scores[self.best][0]


def grid_search(candidates, make, design, y, splits, score, n_jobs=None):
    """Scores every candidate by ``cross_validate`` on the same splits and
    picks the lowest mean score.

    Parameters
    ----------
    candidates : list
        Hyperparameter values, in any form ``make`` accepts.
    make : callable
        ``make(candidate) -> model``.
    design : Design
    y : list of float
    splits : list of (list of int, list of int)
    score : callable
        As ``cross_validate``.
    n_jobs : int, optional

    Returns
    -------
    SearchResult

    Examples
    --------
    >>> from actuarialrs.models import Design, ElasticNet, deviance_score, grid_search, k_fold
    >>> x = [i / 10 for i in range(40)]
    >>> d = Design([[1.0] * 40, x], ["(Intercept)", "x"])
    >>> y = [1.0 + 2.0 * v for v in x]
    >>> found = grid_search([0.0, 1.0, 10.0], lambda lam: ElasticNet("gaussian", lam=lam),
    ...                     d, y, k_fold(40, 4, 1), deviance_score("gaussian"))
    >>> found.best_candidate
    0.0
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    scores = []
    for c in candidates:
        s = cross_validate(make(c), design, y, splits, score, n_jobs)
        scores.append((c, sum(s) / len(s)))
    best = min(range(len(scores)), key=lambda i: scores[i][1])
    return SearchResult(scores, best)


def log_uniform(rng, low, high):
    """A draw log-uniform between ``low`` and ``high``, for learning rates
    and penalties.

    Parameters
    ----------
    rng : random.Random
    low, high : float
        Positive bounds.

    Returns
    -------
    float
    """
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def random_search(n, seed, draw, make, design, y, splits, score, n_jobs=None):
    """Draws ``n`` candidates with ``draw(rng)`` from ``random.Random(seed)``
    and scores them as ``grid_search`` does: with several hyperparameters,
    random candidates cover each range better than a grid of the same size.

    Parameters
    ----------
    n : int
    seed : int
    draw : callable
        ``draw(rng) -> candidate``, for example
        ``lambda rng: log_uniform(rng, 1e-4, 1.0)``.
    make, design, y, splits, score, n_jobs
        As ``grid_search``.

    Returns
    -------
    SearchResult
    """
    rng = random.Random(seed)
    return grid_search([draw(rng) for _ in range(n)], make, design, y, splits, score, n_jobs)
