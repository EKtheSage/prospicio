"""Models: terms and design matrices, GLMs, elastic nets and GAMs, neural networks (the
CANN, the attention CANN and ONNX import), metrics, resampling, tuning, and MCMC
diagnostics (docs/design/models.md)."""

import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .prospicio_native import (
    actual_vs_expected,
    BayesGlm,
    BayesGlmFit,
    BayesGlmm,
    BayesGlmmFit,
    BayesStacking,
    HierarchicalStacking,
    StackingFit,
    Coding,
    CvPath,
    RelaxedCv,
    Design,
    ElasticNet,
    ElasticNetFit,
    Elpd,
    Gam,
    GamFit,
    Glm,
    GlmFit,
    DoubleGlm,
    Draws,
    DoubleGlmFit,
    Glmm,
    GlmmFit,
    NegativeBinomialFit,
    NegativeBinomialGlm,
    TargetEncoding,
    Terms,
    TweedieFit,
    TweedieGlm,
    crps,
    deviance,
    elpd_loo,
    elpd_waic,
    confusion_matrix,
    conformal_cqr,
    conformal_plus,
    conformal_split,
    balance_correction,
    double_lift,
    isotonic,
    murphy,
    recalibrate,
    reliability,
    Isotonic,
    f_test,
    gini,
    loss_ratio_chart,
    normalized_gini,
    roc,
    likelihood_ratio_test,
    vif,
    group_k_fold,
    k_fold,
    ks_uniform,
    lift,
    log_score,
    loo_pit,
    lppd,
    mars,
    MarsFit,
    mcmc_diagnostics,
    permutation,
    pinball_loss,
    pit,
    pit_from_draws,
    pit_histogram,
    ppc_check,
    pseudo_bma_weights,
    quantile_residuals,
    simulate_from_means,
    target_encode,
    stacking_weights,
    time_ordered,
    time_splits,
    tweedie_profile,
)

try:  # The neural networks, behind the native build's default `nn` feature.
    from .prospicio_native import (
        AttentionCann,
        AttentionCannFit,
        Cann,
        CannFit,
        OnnxModel,
        OnnxPredictor,
        TokenAttention,
    )
except ImportError:  # pragma: no cover - a build without `nn`
    _NN = []
else:
    _NN = [
        "Cann",
        "CannFit",
        "AttentionCann",
        "AttentionCannFit",
        "TokenAttention",
        "OnnxModel",
        "OnnxPredictor",
    ]

__all__ = [
    "Terms",
    "TargetEncoding",
    "target_encode",
    "mars",
    "MarsFit",
    "Coding",
    "Design",
    "CvPath",
    "RelaxedCv",
    "Glm",
    "GlmFit",
    "TweedieGlm",
    "TweedieFit",
    "tweedie_profile",
    "NegativeBinomialGlm",
    "NegativeBinomialFit",
    "DoubleGlm",
    "DoubleGlmFit",
    "Glmm",
    "GlmmFit",
    "ElasticNet",
    "ElasticNetFit",
    "Gam",
    "GamFit",
    "deviance",
    "gini",
    "normalized_gini",
    "double_lift",
    "conformal_split",
    "conformal_cqr",
    "conformal_plus",
    "conformal_intervals",
    "isotonic",
    "Isotonic",
    "recalibrate",
    "balance_correction",
    "reliability",
    "murphy",
    "loss_ratio_chart",
    "roc",
    "confusion_matrix",
    "f_test",
    "likelihood_ratio_test",
    "vif",
    "lift",
    "crps",
    "pinball_loss",
    "pinball_score",
    "log_score",
    "pit",
    "pit_from_draws",
    "pit_histogram",
    "quantile_residuals",
    "loo_pit",
    "ppc_check",
    "ks_uniform",
    "k_fold",
    "group_k_fold",
    "time_ordered",
    "time_splits",
    "mcmc_diagnostics",
    "Draws",
    "read_stan_csv",
    "draws_from_arrays",
    "draws_from_inference_data",
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
    "permutation",
    "permutation_importance",
    "PermutationImportance",
    "interaction_screen",
    "InteractionScreen",
    "partial_dependence",
    "PartialDependence",
    "stacking_weights",
    "pseudo_bma_weights",
    "actual_vs_expected",
    "BayesGlm",
    "BayesGlmFit",
    "BayesGlmm",
    "BayesGlmmFit",
    "BayesStacking",
    "HierarchicalStacking",
    "StackingFit",
] + _NN


def pinball_score(alpha):
    """A score for ``cross_validate`` and ``compare``: the weighted mean
    pinball loss of predicted ``alpha`` quantiles on the test rows, with the
    test design's weights. For a quantile model, such as
    ``Booster("quantile", alpha=...)``.

    Parameters
    ----------
    alpha : float
        In ``(0, 1)``.

    Returns
    -------
    callable
        ``score(y_test, predicted, test_design) -> float``.

    Examples
    --------
    >>> from prospicio.models import Design, pinball_score
    >>> d = Design([[1.0, 1.0]], ["(Intercept)"])
    >>> pinball_score(0.9)([1.0, 0.0], [0.0, 1.0], d)
    0.5
    """

    def score(y, pred, design):
        return pinball_loss(list(y), list(pred), alpha, list(design.weights))

    return score


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
    >>> from prospicio.models import Design, deviance_score
    >>> d = Design([[1.0, 1.0]], ["(Intercept)"])
    >>> deviance_score("gaussian")([1.0, 3.0], [2.0, 2.0], d)
    1.0
    """

    def score(y, pred, design):
        w = design.weights
        return deviance(family, y, pred, weights=w, theta=theta, power=power) / sum(w)

    return score


def conformal_intervals(model, design, y, test_design, method="cv+", alpha=0.1, splits=None,
                        score="absolute", family=None, n_jobs=None):
    """Conformal prediction intervals around any model, with coverage of at
    least ``1 - alpha`` (split, CQR) or ``1 - 2 alpha`` (CV+, jackknife+) in
    finite samples under exchangeability.

    Parameters
    ----------
    model : object
        For ``"split"`` a fitted model (``predict(design)``), scored on the
        calibration set ``design``, ``y``. For ``"cqr"`` a pair of fitted
        lower and upper quantile models. For ``"cv+"`` and ``"jackknife+"``
        a specification with ``fit(design, y)``, fitted on each split's
        training rows on threads, as ``cross_validate`` does.
    design : Design
        The calibration set (split, CQR) or the training set (CV+,
        jackknife+).
    y : list of float
    test_design : Design
    method : {"split", "cqr", "cv+", "jackknife+"}, default "cv+"
    alpha : float, default 0.1
    splits : list of (list of int, list of int), optional
        For ``"cv+"``: the folds, each row in one test set; ``k_fold(n, 5,
        0)`` by default. ``"jackknife+"`` leaves out one row a fit.
    score : {"absolute", "pearson"}, default "absolute"
        For ``"split"``; ``"pearson"`` needs ``family``.
    family : str, optional
    n_jobs : int, optional

    Returns
    -------
    dict
        ``lo`` and ``hi``, one a test row.

    Examples
    --------
    >>> from prospicio.models import Design, Glm, conformal_intervals
    >>> x = [float(i % 7) for i in range(40)]
    >>> y = [1.0 + 0.5 * v + ((i * 31) % 7) - 3.0 for i, v in enumerate(x)]
    >>> d = Design([[1.0] * 40, x], ["(Intercept)", "x"])
    >>> t = Design([[1.0], [3.0]], ["(Intercept)", "x"])
    >>> i = conformal_intervals(Glm("gaussian"), d, y, t, method="jackknife+")
    >>> i["lo"][0] < 2.5 < i["hi"][0]
    True
    """
    if len(y) != design.n_rows:
        raise ValueError(f"{len(y)} responses for {design.n_rows} design rows")
    if method == "split":
        return conformal_split(y, model.predict(design), model.predict(test_design), alpha=alpha,
                               score=score, family=family)
    if method == "cqr":
        lower, upper = model
        return conformal_cqr(y, lower.predict(design), upper.predict(design),
                             lower.predict(test_design), upper.predict(test_design), alpha=alpha)
    if method not in ("cv+", "jackknife+"):
        raise ValueError(f'method must be "split", "cqr", "cv+" or "jackknife+", not {method!r}')
    n = len(y)
    if method == "jackknife+":
        splits = [([j for j in range(n) if j != i], [i]) for i in range(n)]
    elif splits is None:
        splits = k_fold(n, 5, 0)
    folds = [-1] * n
    for k, (_, test) in enumerate(splits):
        for i in test:
            if folds[i] != -1:
                raise ValueError("each row must be in exactly one split's test set")
            folds[i] = k
    if -1 in folds:
        raise ValueError("each row must be in exactly one split's test set")

    def one(split):
        train, test = split
        fitted = model.fit(design.select(train), [y[i] for i in train])
        return fitted.predict(design.select(test)), fitted.predict(test_design)

    with ThreadPoolExecutor(max_workers=n_jobs or min(32, max(1, len(splits)))) as pool:
        results = list(pool.map(one, splits))
    oof = [0.0] * n
    for (_, test), (held, _) in zip(splits, results):
        for i, p in zip(test, held):
            oof[i] = p
    return conformal_plus(y, oof, folds, [t for _, t in results], alpha=alpha)


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
    >>> from prospicio.models import Design, Glm, cross_validate, deviance_score, k_fold
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
    >>> from prospicio.models import Design, ElasticNet, deviance_score, grid_search, k_fold
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


def read_stan_csv(paths):
    """Reads CmdStan CSV output, one file per chain, into ``Draws``.

    Comments are skipped, saved warmup draws dropped, ``theta.1.2`` named
    ``theta[1,2]``; ``lp__`` is kept as a variable, the sampler's other
    ``__`` columns are in ``Draws.sampler`` and the ``divergent__`` draws
    are counted per chain.

    Parameters
    ----------
    paths : str, path or list of them
        One CSV file per chain.

    Returns
    -------
    Draws
    """
    if isinstance(paths, (str, bytes)) or hasattr(paths, "__fspath__"):
        paths = [paths]
    return Draws.read_stan_csv([str(p) for p in paths])


def _as_lists(x):
    """A nested list from an array-like (numpy, xarray, lists)."""
    values = getattr(x, "values", x)
    if hasattr(values, "tolist"):
        values = values.tolist()
    return values


def _shape(x):
    shape = []
    while isinstance(x, (list, tuple)):
        shape.append(len(x))
        if not x:
            break
        x = x[0]
    return shape


def _flatten(x):
    if isinstance(x, (list, tuple)):
        for v in x:
            yield from _flatten(v)
    else:
        yield float(x)


def _element_names(name, shape):
    """``name`` for a scalar, else ``name[i,j,...]`` (1-based, row-major)."""
    if not shape:
        return [name]
    names = [()]
    for n in shape:
        names = [idx + (i,) for idx in names for i in range(1, n + 1)]
    return [f"{name}[{','.join(str(i) for i in idx)}]" for idx in names]


def _arrays_to_chains(arrays):
    names, per_var = [], []
    n_chains = n_draws = None
    for name, value in arrays.items():
        x = _as_lists(value)
        shape = _shape(x)
        if len(shape) < 2:
            raise ValueError(f"{name}: an array needs the shape (chain, draw, ...)")
        if n_chains is None:
            n_chains, n_draws = shape[0], shape[1]
        elif shape[:2] != [n_chains, n_draws]:
            raise ValueError(
                f"{name} has {shape[0]} chains of {shape[1]} draws, "
                f"the first variable {n_chains} of {n_draws}"
            )
        elems = _element_names(name, shape[2:])
        names.extend(elems)
        per_var.append(
            [[list(_flatten(x[c][d])) for d in range(n_draws)] for c in range(n_chains)]
        )
    if not names:
        raise ValueError("no variables given")
    chains = [
        [[v for block in per_var for v in block[c][d]] for d in range(n_draws)]
        for c in range(n_chains)
    ]
    return names, chains


def draws_from_arrays(arrays, log_likelihood=None, divergences=None):
    """``Draws`` from arrays shaped ``(chain, draw, ...)``.

    A variable of shape ``(chain, draw)`` is one name; one of shape
    ``(chain, draw, n, m, ...)`` becomes the names ``theta[1,1]``,
    ``theta[1,2]``, ... (1-based, row-major, as ``posterior`` names them).

    Parameters
    ----------
    arrays : dict
        Variable name to an array-like (numpy array, nested lists).
    log_likelihood : array-like, optional
        The pointwise log-likelihood shaped ``(chain, draw, ...)``, held as
        ``log_lik[...]`` for ``Draws.loo``.
    divergences : array-like, optional
        Either one count per chain or a ``(chain, draw)`` array of
        divergent transitions (booleans or 0/1).

    Returns
    -------
    Draws

    Examples
    --------
    >>> from prospicio.models import draws_from_arrays
    >>> d = draws_from_arrays({"mu": [[0.1, 0.3, 0.2, 0.0], [0.2, 0.1, 0.4, 0.3]]})
    >>> d.n_chains, d.draws_per_chain
    (2, 4)
    """
    arrays = dict(arrays)
    if log_likelihood is not None:
        if "log_lik" in arrays:
            raise ValueError("arrays already hold a variable named log_lik")
        arrays["log_lik"] = log_likelihood
    names, chains = _arrays_to_chains(arrays)
    div = None
    if divergences is not None:
        d = _as_lists(divergences)
        div = [int(sum(bool(v) for v in c)) if isinstance(c, (list, tuple)) else int(c) for c in d]
    return Draws(names, chains, div)


def draws_from_inference_data(idata, var_names=None, log_likelihood=None):
    """``Draws`` from an arviz ``InferenceData``, as PyMC, nutpie and
    ``arviz.from_netcdf`` or ``arviz.from_zarr`` return it.

    Reads the ``posterior`` group, the ``log_likelihood`` group (held as
    ``log_lik[...]`` for ``Draws.loo``) and ``sample_stats.diverging``.
    arviz itself is not imported: any object with those groups, each a
    mapping of names to arrays shaped ``(chain, draw, ...)``, works.

    Parameters
    ----------
    idata : arviz.InferenceData
    var_names : list of str, optional
        The posterior variables to keep; all of them by default.
    log_likelihood : str, optional
        Which variable of the ``log_likelihood`` group to take when it has
        several (for a model with several observed variables).

    Returns
    -------
    Draws
    """
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        raise ValueError("the InferenceData has no posterior group")
    names = list(var_names) if var_names is not None else list(posterior.data_vars)
    arrays = {n: posterior[n] for n in names}
    ll = None
    group = getattr(idata, "log_likelihood", None)
    if group is not None:
        observed = list(group.data_vars)
        if log_likelihood is not None:
            ll = group[log_likelihood]
        elif len(observed) == 1:
            ll = group[observed[0]]
        elif observed:
            raise ValueError(
                f"the log_likelihood group has {observed}; name one with log_likelihood="
            )
    elif log_likelihood is not None:
        raise ValueError("the InferenceData has no log_likelihood group")
    stats = getattr(idata, "sample_stats", None)
    diverging = None
    if stats is not None and "diverging" in stats.data_vars:
        diverging = stats["diverging"]
    return draws_from_arrays(arrays, log_likelihood=ll, divergences=diverging)


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
    >>> from prospicio.models import Design, ElasticNet, Glm, compare, deviance_score, k_fold
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


# Model-agnostic explanations (CAS Monograph 16, chapter 9). They need only
# ``predict``, so they work for any model; a frame is a dict of columns
# (lists or NumPy arrays) and ``build(frame)`` turns one into a Design.


def _take(column, rows):
    try:
        return column[rows]  # NumPy fancy indexing keeps the array type
    except TypeError:
        return [column[i] for i in rows]


def _constant(column, value):
    n = len(column)
    if type(column).__module__ == "numpy":
        import numpy

        return numpy.full(n, value)
    return [value] * n


def _frame_rows(frame):
    lengths = {len(v) for v in frame.values()}
    if len(lengths) != 1:
        raise ValueError(f"the frame's columns have different lengths: {sorted(lengths)}")
    return lengths.pop()


def _groups(features, frame):
    """Normalizes ``features`` to ``[(label, names)]``: a name, a list of
    names shuffled together (labelled by joining them with ``+``), or a dict
    from label to names."""
    items = features.items() if isinstance(features, dict) else (
        (f if isinstance(f, str) else "+".join(f), f) for f in features
    )
    out = []
    for label, names in items:
        names = (names,) if isinstance(names, str) else tuple(names)
        for name in names:
            if name not in frame:
                raise ValueError(f"no column {name!r} in the frame")
        out.append((label, names))
    if not out:
        raise ValueError("features must not be empty")
    if len({label for label, _ in out}) != len(out):
        raise ValueError("feature labels must be distinct")
    return out


def _shuffled(frame, names, perm):
    return {k: (_take(v, perm) if k in names else v) for k, v in frame.items()}


def _per_split(model, frame, y, build, splits, one, n_jobs):
    """Runs ``one(fitted, test_frame, y_test, split_index)`` on each split's
    test rows after fitting ``model`` on its training rows."""
    if not splits:
        raise ValueError("splits must not be empty")
    if len(y) != _frame_rows(frame):
        raise ValueError(f"{len(y)} responses for {_frame_rows(frame)} frame rows")

    def run(item):
        i, (train, test) = item
        train_frame = {k: _take(v, train) for k, v in frame.items()}
        fitted = model.fit(build(train_frame), [y[j] for j in train])
        test_frame = {k: _take(v, test) for k, v in frame.items()}
        return one(fitted, test_frame, [y[j] for j in test], i)

    with ThreadPoolExecutor(max_workers=n_jobs or max(1, len(splits))) as pool:
        return list(pool.map(run, enumerate(splits)))


def _losses(fitted, frame, y, build, score, groups, n_repeats, seed, split):
    """The score on ``frame`` and, for each group of names, the mean increase
    over ``n_repeats`` shuffles; repeat ``r`` of split ``i`` shuffles with
    ``permutation(n, seed, i * n_repeats + r)``, the same for every group."""
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    n = _frame_rows(frame)
    if len(y) != n:
        raise ValueError(f"{len(y)} responses for {n} frame rows")

    def scored(f):
        design = build(f)
        return score(y, fitted.predict(design), design)

    base = scored(frame)
    total = [0.0] * len(groups)
    for r in range(n_repeats):
        perm = permutation(n, seed, split * n_repeats + r)
        for g, names in enumerate(groups):
            total[g] += scored(_shuffled(frame, names, perm)) - base
    return base, [t / n_repeats for t in total]


@dataclass
class PermutationImportance:
    """The result of ``permutation_importance``.

    Attributes
    ----------
    features : list of str
        The labels of the features or groups shuffled.
    by_split : list of list of float
        ``by_split[i][j]``: on split ``i``, the increase in the score (a loss)
        when feature ``j`` is shuffled, averaged over the repeats.
    base : list of float
        The score on each split with nothing shuffled.
    """

    features: list
    by_split: list
    base: list

    @property
    def importance(self):
        """The mean increase over the splits, per feature."""
        return [_mean_se([s[j] for s in self.by_split])[0] for j in range(len(self.features))]

    @property
    def std_error(self):
        """The splits' standard deviation over the square root of their
        number, per feature (NaN for one split)."""
        return [_mean_se([s[j] for s in self.by_split])[1] for j in range(len(self.features))]

    @property
    def relative(self):
        """Importance as a percent of the largest, the monograph's scale."""
        imp = self.importance
        top = max(imp)
        return [100.0 * v / top if top > 0 else math.nan for v in imp]

    @staticmethod
    def combine(results):
        """One result from several over the same features, such as one per
        fold computed with a design built differently in each fold."""
        results = list(results)
        if not results or any(r.features != results[0].features for r in results):
            raise ValueError("results must be non-empty and over the same features")
        return PermutationImportance(
            results[0].features,
            [s for r in results for s in r.by_split],
            [b for r in results for b in r.base],
        )

    def table(self):
        """One row per feature, most important first: ``feature``,
        ``importance``, ``std_error`` and ``relative``, as a list of dicts."""
        rows = [
            {"feature": f, "importance": i, "std_error": s, "relative": r}
            for f, i, s, r in zip(self.features, self.importance, self.std_error, self.relative)
        ]
        return sorted(rows, key=lambda row: -row["importance"])


def permutation_importance(
    model, frame, y, build, score, features, splits=None, n_repeats=1, seed=0, n_jobs=None
):
    """Permutation feature importance: how much a score worsens when a
    feature's column is shuffled, which breaks its link to the response and
    leaves every other column in place (CAS Monograph 16, section 9.2).

    It needs only predictions, so it works for any model. A list of names
    shuffles them together with one permutation (a group, such as the
    columns of one encoding). Shuffles come from ``permutation``, so Python
    and R give the same values on the same data.

    Parameters
    ----------
    model : object
        A fitted model with ``predict(design)``; with ``splits``, a model
        with ``fit(design, y)`` (as for ``cross_validate``), fitted on each
        split's training rows and explained on its test rows.
    frame : dict
        Column name to a list or NumPy array of raw values, all one length:
        the rows to explain (the whole data with ``splits``).
    y : list of float
    build : callable
        ``build(frame) -> Design``, the design from raw columns (with any
        offset and weights). Shuffling a raw column moves every design
        column built from it.
    score : callable
        ``score(y, predicted, design) -> float``, a loss (lower is better),
        such as ``deviance_score("poisson")``; a gain such as a pseudo-R2
        goes in as its negative.
    features : list or dict
        Names of columns, lists of names shuffled together, or a dict from
        a label to names. Never the response, exposure or weights.
    splits : list of (list of int, list of int), optional
        ``(train, test)`` rows, as ``k_fold`` returns.
    n_repeats : int, default 1
        Shuffles per split, averaged.
    seed : int, default 0
        Repeat ``r`` of split ``i`` shuffles with
        ``permutation(n, seed, i * n_repeats + r)``.
    n_jobs : int, optional
        Threads with ``splits``; one per split by default.

    Returns
    -------
    PermutationImportance

    Examples
    --------
    >>> from prospicio.models import Design, Glm, deviance_score, permutation_importance
    >>> x = [i / 10 for i in range(40)]
    >>> frame = {"x": x, "noise": [((7 * i) % 11) / 11 for i in range(40)]}
    >>> y = [1.0 + 2.0 * v + (0.3 if i % 3 else -0.6) for i, v in enumerate(x)]
    >>> build = lambda f: Design([[1.0] * len(f["x"]), list(f["x"])], ["(Intercept)", "x"])
    >>> fit = Glm("gaussian").fit(build(frame), y)
    >>> imp = permutation_importance(fit, frame, y, build, deviance_score("gaussian"), ["x", "noise"])
    >>> imp.importance[0] > 1.0, imp.importance[1]
    (True, 0.0)
    """
    groups = _groups(features, frame)
    names = [g for _, g in groups]
    labels = [label for label, _ in groups]

    if splits is None:
        base, loss = _losses(model, frame, y, build, score, names, n_repeats, seed, 0)
        return PermutationImportance(labels, [loss], [base])

    def one(fitted, test_frame, y_test, i):
        return _losses(fitted, test_frame, y_test, build, score, names, n_repeats, seed, i)

    out = _per_split(model, frame, y, build, splits, one, n_jobs)
    return PermutationImportance(labels, [loss for _, loss in out], [base for base, _ in out])


@dataclass
class InteractionScreen:
    """The result of ``interaction_screen``.

    Attributes
    ----------
    features : list of str
    pairs : list of (str, str)
    single : list of list of float
        ``single[i][j]``: on split ``i``, the loss from shuffling feature
        ``j`` alone, ``L_A``.
    joint : list of list of float
        ``joint[i][k]``: on split ``i``, the loss from shuffling pair ``k``'s
        two features together, ``L_AB``.
    """

    features: list
    pairs: list
    single: list
    joint: list

    def _strengths(self, i):
        index = {f: j for j, f in enumerate(self.features)}
        return [
            self.single[i][index[a]] + self.single[i][index[b]] - self.joint[i][k]
            for k, (a, b) in enumerate(self.pairs)
        ]

    @property
    def by_split(self):
        """``L_A + L_B - L_AB`` per split and pair."""
        return [self._strengths(i) for i in range(len(self.single))]

    @property
    def strength(self):
        """``L_A + L_B - L_AB`` averaged over the splits, per pair: about 0
        when neither feature interacts with anything, positive when they
        interact with each other or with a common third feature."""
        by = self.by_split
        return [_mean_se([s[k] for s in by])[0] for k in range(len(self.pairs))]

    @property
    def std_error(self):
        """The splits' standard error of each pair's strength."""
        by = self.by_split
        return [_mean_se([s[k] for s in by])[1] for k in range(len(self.pairs))]

    @staticmethod
    def combine(results):
        """One result from several over the same features and pairs."""
        results = list(results)
        if not results or any(
            (r.features, r.pairs) != (results[0].features, results[0].pairs) for r in results
        ):
            raise ValueError("results must be non-empty and over the same features and pairs")
        return InteractionScreen(
            results[0].features,
            results[0].pairs,
            [s for r in results for s in r.single],
            [s for r in results for s in r.joint],
        )

    def table(self):
        """One row per pair, strongest first: ``a``, ``b``, ``strength``,
        ``std_error``, and the mean losses ``loss_a``, ``loss_b`` and
        ``loss_ab``, as a list of dicts."""
        index = {f: j for j, f in enumerate(self.features)}
        mean = lambda rows, j: _mean_se([r[j] for r in rows])[0]
        rows = [
            {
                "a": a,
                "b": b,
                "strength": s,
                "std_error": e,
                "loss_a": mean(self.single, index[a]),
                "loss_b": mean(self.single, index[b]),
                "loss_ab": mean(self.joint, k),
            }
            for k, ((a, b), s, e) in enumerate(zip(self.pairs, self.strength, self.std_error))
        ]
        return sorted(rows, key=lambda row: -row["strength"])


def interaction_screen(
    model, frame, y, build, score, features, pairs=None, splits=None, n_repeats=1, seed=0,
    n_jobs=None,
):
    """The pairwise interaction screen by permutations (CAS Monograph 16,
    section 9.4): with ``L_A``, ``L_B`` and ``L_AB`` the losses from
    shuffling feature ``A``, feature ``B`` and both together with the same
    permutation, ``L_A + L_B - L_AB`` is about zero if neither interacts with
    anything and positive if they interact with each other or with a common
    third feature. Run it on a flexible model (a boosting model) to find the
    pairs a GLM might need.

    Parameters
    ----------
    model, frame, y, build, score, splits, n_repeats, seed, n_jobs
        As for ``permutation_importance``.
    features : list or dict
        As for ``permutation_importance``: names, groups, or labels to names.
    pairs : list of (str, str), optional
        Pairs of feature labels to screen; every pair by default.

    Returns
    -------
    InteractionScreen

    Examples
    --------
    >>> from prospicio.models import Design, Glm, deviance_score, interaction_screen
    >>> a = [float(i % 5) for i in range(40)]
    >>> b = [float(i % 3) for i in range(40)]
    >>> y = [1.0 + u + v + 2.0 * u * v + (0.1 if i % 2 else -0.1) for i, (u, v) in enumerate(zip(a, b))]
    >>> build = lambda f: Design([[1.0] * len(f["a"]), list(f["a"]), list(f["b"]),
    ...                           [u * v for u, v in zip(f["a"], f["b"])]], ["(Intercept)", "a", "b", "ab"])
    >>> fit = Glm("gaussian").fit(build({"a": a, "b": b}), y)
    >>> s = interaction_screen(fit, {"a": a, "b": b}, y, build, deviance_score("gaussian"), ["a", "b"])
    >>> s.pairs, s.strength[0] > 0
    ([('a', 'b')], True)
    """
    groups = _groups(features, frame)
    labels = [label for label, _ in groups]
    by_label = dict(groups)
    if pairs is None:
        pairs = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    pairs = [tuple(p) for p in pairs]
    for p in pairs:
        if len(p) != 2 or p[0] == p[1] or any(f not in by_label for f in p):
            raise ValueError(f"{p!r} is not a pair of two different feature labels")
    if not pairs:
        raise ValueError("interaction_screen needs at least two features")
    shuffles = [by_label[f] for f in labels] + [by_label[a] + by_label[b] for a, b in pairs]

    def run(fitted, test_frame, y_test, i):
        _, loss = _losses(fitted, test_frame, y_test, build, score, shuffles, n_repeats, seed, i)
        return loss[: len(labels)], loss[len(labels) :]

    if splits is None:
        out = [run(model, frame, y, 0)]
    else:
        out = _per_split(model, frame, y, build, splits, run, n_jobs)
    return InteractionScreen(labels, pairs, [s for s, _ in out], [j for _, j in out])


@dataclass
class PartialDependence:
    """The result of ``partial_dependence``.

    Attributes
    ----------
    feature : str
    values : list
    by_split : list of list of float
        ``by_split[i][v]``: on split ``i``, the mean prediction with the
        feature set to ``values[v]`` on every row.
    """

    feature: str
    values: list
    by_split: list

    @property
    def mean(self):
        """The mean prediction at each value, averaged over the splits."""
        return [_mean_se([s[v] for s in self.by_split])[0] for v in range(len(self.values))]

    def relative(self, value):
        """The mean predictions divided by the one at ``value``: for a
        log-link model without interactions, exactly the feature's
        relativities."""
        m = self.mean
        return [v / m[self.values.index(value)] for v in m]

    @staticmethod
    def combine(results):
        """One result from several over the same feature and values."""
        results = list(results)
        if not results or any(
            (r.feature, r.values) != (results[0].feature, results[0].values) for r in results
        ):
            raise ValueError("results must be non-empty and over the same feature and values")
        return PartialDependence(
            results[0].feature, results[0].values, [s for r in results for s in r.by_split]
        )


def partial_dependence(model, frame, build, feature, values, y=None, splits=None, n_jobs=None):
    """Partial dependence: the mean prediction with ``feature`` set to each
    value on every row and every other column as it is (CAS Monograph 16,
    section 9.3). An average over the portfolio, so an effect that differs
    by another feature (an interaction) is blended across it.

    Parameters
    ----------
    model : object
        A fitted model with ``predict(design)``; with ``splits``, a model
        with ``fit(design, y)``, explained on each split's test rows.
    frame : dict
        As for ``permutation_importance``.
    build : callable
        ``build(frame) -> Design``.
    feature : str
        One column of ``frame``.
    values : list
        The values to set it to.
    y : list of float, optional
        The response, needed only with ``splits`` to fit.
    splits : list of (list of int, list of int), optional
    n_jobs : int, optional

    Returns
    -------
    PartialDependence

    Examples
    --------
    >>> import math
    >>> from prospicio.models import Design, Glm, partial_dependence
    >>> x = [float(i % 4) for i in range(40)]
    >>> z = [float(i % 3) for i in range(40)]
    >>> y = [math.exp(0.2 * u + 0.1 * v) * (1.1 if i % 2 else 0.9) for i, (u, v) in enumerate(zip(x, z))]
    >>> build = lambda f: Design([[1.0] * len(f["x"]), list(f["x"]), list(f["z"])], ["(Intercept)", "x", "z"])
    >>> fit = Glm("poisson").fit(build({"x": x, "z": z}), y)
    >>> pd = partial_dependence(fit, {"x": x, "z": z}, build, "x", [0.0, 1.0])
    >>> round(pd.relative(0.0)[1] / math.exp(fit.coefficients[1]), 12)
    1.0
    """
    if feature not in frame:
        raise ValueError(f"no column {feature!r} in the frame")
    values = list(values)
    if not values:
        raise ValueError("values must not be empty")

    def run(fitted, f, *_):
        out = []
        for v in values:
            pred = fitted.predict(build(dict(f, **{feature: _constant(f[feature], v)})))
            out.append(sum(pred) / len(pred))
        return out

    if splits is None:
        by_split = [run(model, frame)]
    else:
        if y is None:
            raise ValueError("partial_dependence needs y to fit on the splits")
        by_split = _per_split(model, frame, y, build, splits, run, n_jobs)
    return PartialDependence(feature, values, by_split)
