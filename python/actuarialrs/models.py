"""Models: terms and design matrices, GLMs, elastic nets and GAMs, metrics, resampling,
tuning, and MCMC diagnostics (docs/design/models.md)."""

import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .actuarialrs_native import (
    actual_vs_expected,
    BayesGlm,
    BayesGlmFit,
    BayesStacking,
    HierarchicalStacking,
    StackingFit,
    Coding,
    CvPath,
    Design,
    ElasticNet,
    ElasticNetFit,
    Elpd,
    Gam,
    GamFit,
    Glm,
    GlmFit,
    Terms,
    crps,
    deviance,
    elpd_loo,
    elpd_waic,
    gini,
    group_k_fold,
    k_fold,
    ks_uniform,
    lift,
    log_score,
    lppd,
    mcmc_diagnostics,
    pit,
    pit_from_draws,
    pit_histogram,
    pseudo_bma_weights,
    simulate_from_means,
    stacking_weights,
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
    "simulate_from_means",
    "elpd_loo",
    "elpd_waic",
    "lppd",
    "Elpd",
    "deviance_score",
    "cross_validate",
    "grid_search",
    "random_search",
    "log_uniform",
    "SearchResult",
    "compare",
    "Comparison",
    "stacking_weights",
    "pseudo_bma_weights",
    "actual_vs_expected",
    "BayesGlm",
    "BayesGlmFit",
    "BayesStacking",
    "HierarchicalStacking",
    "StackingFit",
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


def _mean_se(values):
    k = len(values)
    m = sum(values) / k
    if k < 2:
        return m, math.nan
    var = sum((v - m) ** 2 for v in values) / (k - 1)
    return m, math.sqrt(var / k)


@dataclass
class Comparison:
    """The result of ``compare``: every model's score on every metric and
    split.

    Attributes
    ----------
    models : list of str
    metrics : list of str
    split_scores : dict
        ``split_scores[(model, metric)]`` is the list of per-split scores.
    """

    models: list
    metrics: list
    split_scores: dict

    def best(self, metric):
        """The model with the lowest mean ``metric``."""
        return min(self.models, key=lambda m: self.mean(m, metric))

    def mean(self, model, metric):
        """Mean score over the splits."""
        return _mean_se(self.split_scores[(model, metric)])[0]

    def std_error(self, model, metric):
        """Standard deviation of the split scores over the square root of
        their number."""
        return _mean_se(self.split_scores[(model, metric)])[1]

    def difference_std_error(self, model, metric):
        """Standard error of the per-split difference from the best model.

        The splits are shared, so the paired difference is much less noisy
        than either mean: a model within about two of these of the best is
        not clearly worse.
        """
        best = self.split_scores[(self.best(metric), metric)]
        d = [a - b for a, b in zip(self.split_scores[(model, metric)], best)]
        return _mean_se(d)[1]

    def table(self):
        """One row per model and metric: ``model``, ``metric``, ``mean``,
        ``std_error`` and ``difference_std_error``, as a list of dicts
        (pass it to ``pandas.DataFrame`` for a frame)."""
        return [
            {
                "model": m,
                "metric": k,
                "mean": self.mean(m, k),
                "std_error": self.std_error(m, k),
                "difference_std_error": self.difference_std_error(m, k),
            }
            for m in self.models
            for k in self.metrics
        ]


def compare(models, design, y, splits, scores, n_jobs=None):
    """Fits every model on each split's training rows and scores its test
    predictions on every metric: one table across engines.

    Parameters
    ----------
    models : dict
        Name to model; each model has ``fit(design, y)`` returning an object
        with ``predict(design)``, as for ``cross_validate``.
    design : Design
    y : list of float
    splits : list of (list of int, list of int)
    scores : dict
        Name to ``score(y_test, predicted, test_design) -> float``, losses
        (lower is better).
    n_jobs : int, optional
        Threads; one per split by default.

    Returns
    -------
    Comparison

    Examples
    --------
    >>> from actuarialrs.models import Design, ElasticNet, Glm, compare, deviance_score, k_fold
    >>> x = [i / 10 for i in range(40)]
    >>> d = Design([[1.0] * 40, x], ["(Intercept)", "x"])
    >>> y = [1.0 + 2.0 * v + (0.3 if i % 3 else -0.6) for i, v in enumerate(x)]
    >>> c = compare({"glm": Glm("gaussian"), "ridge": ElasticNet("gaussian", alpha=0.0, lam=5.0)},
    ...             d, y, k_fold(40, 4, 1), {"deviance": deviance_score("gaussian")})
    >>> c.best("deviance")
    'glm'
    >>> [row["model"] for row in c.table()]
    ['glm', 'ridge']
    """
    if not models or not scores or not splits:
        raise ValueError("compare needs at least one model, score and split")
    if len(y) != design.n_rows:
        raise ValueError(f"{len(y)} responses for {design.n_rows} design rows")

    def one(split):
        train, test = split
        train_design, test_design = design.select(train), design.select(test)
        y_train, y_test = [y[i] for i in train], [y[i] for i in test]
        out = {}
        for name, model in models.items():
            pred = model.fit(train_design, y_train).predict(test_design)
            for metric, score in scores.items():
                out[(name, metric)] = score(y_test, pred, test_design)
        return out

    with ThreadPoolExecutor(max_workers=n_jobs or max(1, len(splits))) as pool:
        per_split = list(pool.map(one, splits))
    split_scores = {
        (m, k): [s[(m, k)] for s in per_split] for m in models for k in scores
    }
    return Comparison(list(models), list(scores), split_scores)
