"""Gradient boosting through LightGBM or XGBoost, behind the model
protocol (docs/design/models.md: "Gradient boosting stays an adapter").

The adapter converts a ``Design`` and responses into the engine's inputs,
calls the engine, and wraps its output in the shared objects: means from
``predict``, joint draws from ``predict_distribution``. It adds no
evaluation logic: ``compare``, ``cross_validate`` and the metrics in
``prospicio.models`` take a ``Booster`` like any other model.

- The design's offset is the engine's starting score (LightGBM
  ``init_score``, XGBoost ``base_margin``), on the link scale: log
  exposure for a frequency. A constant is added so the trees start from
  the weighted mean response, as the engines' own ``boost_from_average``
  does without an offset: ``log(sum w y / sum w exp(offset))`` for the log
  link, ``sum w (y - offset) / sum w`` for the identity.
- XGBoost computes margins in single precision, so its means agree with a
  double-precision calculation to about 1e-6 relative.
- The design's weights are the engine's sample weights.
- ``predict_distribution`` draws each row's response from the family
  around the fitted mean (process noise); with ``n_boot`` bootstrap
  refits, each simulation also picks one refit's means (parameter
  uncertainty).
- ``dispersion_model=True`` adds a distributional head: a second booster
  (gamma objective, log link) fitted to the Pearson residuals
  ``w (y - mu)^2 / V(mu)``, whose expectation is each row's dispersion.
  The residuals come from 5-fold cross-fitted means, so a mean model that
  fits the training rows closely does not shrink them.
  ``predict_distribution`` then uses one dispersion per row
  (``predict_dispersion``), as a double GLM does (Smyth, 1989).
- ``family="quantile"`` fits the ``alpha`` quantile instead of the mean
  (LightGBM ``quantile``, XGBoost ``reg:quantileerror``), starting from
  the weighted ``alpha`` quantile of the responses. Score it with
  ``models.pinball_score``; ``predict_quantiles`` puts several fits
  together as non-crossing quantile sets.

- ``monotone`` and ``interaction_constraints`` take feature names and are
  written in each engine's own form on the design's columns
  (``BoosterFit.engine_params``); ``BoosterFit.check_monotone`` confirms a
  fitted model's partial dependence on a grid, since the engines enforce
  the constraints split by split.

LightGBM and XGBoost are optional: install the one you use
(``pip install lightgbm`` or ``pip install xgboost``).
"""

import random
import re

from .prospicio_native import simulate_from_means

__all__ = ["Booster", "BoosterFit", "predict_quantiles"]

# family -> (LightGBM objective, XGBoost objective, link)
_OBJECTIVES = {
    "poisson": ("poisson", "count:poisson", "log"),
    "gamma": ("gamma", "reg:gamma", "log"),
    "tweedie": ("tweedie", "reg:tweedie", "log"),
    "gaussian": ("regression", "reg:squarederror", "identity"),
    "quantile": ("quantile", "reg:quantileerror", "identity"),
}
_ENGINES = ("lightgbm", "xgboost")


def _variance(family, mu, power):
    if family == "poisson":
        return mu
    if family == "gamma":
        return mu * mu
    if family == "tweedie":
        return mu**power
    return 1.0


class Booster:
    """Gradient-boosted trees for a response family, through LightGBM or
    XGBoost.

    Parameters
    ----------
    family : {"poisson", "gamma", "tweedie", "gaussian", "quantile"}, default "poisson"
        The engine's objective; a log link for the first three. ``"quantile"``
        fits the ``alpha`` quantile of the response (no link, no offset).
    engine : {"lightgbm", "xgboost"}, default "lightgbm"
    power : float, optional
        Tweedie variance power in (1, 2); required for ``"tweedie"``.
    alpha : float, optional
        Quantile level in (0, 1); required for ``"quantile"``.
    dispersion_model : bool, default False
        Fit a dispersion per row (gamma, Tweedie and Gaussian families).
    n_rounds : int, default 200
        Boosting rounds.
    learning_rate : float, default 0.05
    params : dict, optional
        Further engine parameters (``num_leaves``, ``max_depth``, ...),
        passed as they are; give constraints through ``monotone`` and
        ``interaction_constraints`` instead.
    monotone : dict, optional
        ``{feature: 1, -1 or 0}``: the prediction non-decreasing (1) or
        non-increasing (-1) in the feature. A feature is a design column's
        name or the variable of a monotone term (``hinge(x,k)``,
        ``step(x,k)``, ``log(x/b)``), and the constraint goes on every such
        column. A factor, a polynomial or spline, a feature in an
        interaction column and an unknown name are errors.
    interaction_constraints : list of list of str, optional
        Groups of features allowed to interact in a tree; each feature's
        columns (factor levels included) enter its groups.
    n_boot : int, default 0
        Bootstrap refits for parameter uncertainty in
        ``predict_distribution``; 0 gives process noise only.
    seed : int, default 0
        Seeds the engine and the bootstrap resamples.

    Examples
    --------
    >>> from prospicio.boosting import Booster
    >>> from prospicio.models import Design
    >>> x = [i % 10 / 10 for i in range(200)]
    >>> d = Design([x], ["x"], offset=[0.0] * 200)
    >>> y = [float(i % 3 == 0) + v for i, v in enumerate(x)]
    >>> fit = Booster("poisson", n_rounds=20).fit(d, y)
    >>> len(fit.predict(d))
    200
    """

    def __init__(self, family="poisson", engine="lightgbm", power=None, n_rounds=200,
                 learning_rate=0.05, params=None, n_boot=0, seed=0, alpha=None,
                 dispersion_model=False, monotone=None, interaction_constraints=None):
        if family not in _OBJECTIVES:
            raise ValueError(f"family must be one of {sorted(_OBJECTIVES)}, got {family!r}")
        if engine not in _ENGINES:
            raise ValueError(f"engine must be one of {list(_ENGINES)}, got {engine!r}")
        if family == "tweedie" and not (power is not None and 1.0 < power < 2.0):
            raise ValueError("tweedie needs power in (1, 2)")
        if family == "quantile" and not (alpha is not None and 0.0 < alpha < 1.0):
            raise ValueError("quantile needs alpha in (0, 1)")
        if dispersion_model and family in ("poisson", "quantile"):
            raise ValueError(f"a dispersion model needs a gamma, tweedie or gaussian family, "
                             f"not {family}")
        if n_rounds < 1 or n_boot < 0:
            raise ValueError("n_rounds must be positive and n_boot non-negative")
        self.family = family
        self.engine = engine
        self.power = power
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.params = dict(params or {})
        self.n_boot = n_boot
        self.seed = seed
        self.alpha = alpha
        self.dispersion_model = dispersion_model
        self.monotone = dict(monotone or {})
        self.interaction_constraints = [list(g) for g in (interaction_constraints or [])]
        if any(v not in (-1, 0, 1) for v in self.monotone.values()):
            raise ValueError("monotone values must be 1, -1 or 0")
        given = {"monotone_constraints", "monotone_constraint", "mc", "interaction_constraints"}
        clash = sorted(given & set(self.params))
        if clash and (self.monotone or self.interaction_constraints):
            raise ValueError(f"params sets {clash}; give constraints through monotone and "
                             "interaction_constraints only")

    def engine_params(self, names):
        """The constraint parameters the engine gets for a design with
        columns ``names``: LightGBM ``monotone_constraints`` (a list in
        column order) and ``interaction_constraints`` (lists of column
        indices), or XGBoost's tuple string and nested list string.

        Parameters
        ----------
        names : list of str

        Returns
        -------
        dict
        """
        names = list(names)
        out = {}
        if self.monotone:
            signs = [0] * len(names)
            for feature, sign in self.monotone.items():
                for j in _monotone_columns(names, feature):
                    signs[j] = int(sign)
            out["monotone_constraints"] = (signs if self.engine == "lightgbm"
                                           else "(" + ",".join(map(str, signs)) + ")")
        if self.interaction_constraints:
            groups = [sorted({j for f in g for j in _feature_columns(names, f)})
                      for g in self.interaction_constraints]
            out["interaction_constraints"] = (groups if self.engine == "lightgbm"
                                              else "[" + ", ".join("[" + ", ".join(map(str, g)) + "]"
                                                                   for g in groups) + "]")
        return out

    def _train(self, x, y, w, offset, seed, constraints=None):
        lgb_obj, xgb_obj, _ = _OBJECTIVES[self.family]
        if self.engine == "lightgbm":
            import lightgbm as lgb

            params = {"objective": lgb_obj, "learning_rate": self.learning_rate,
                      "seed": seed, "deterministic": True, "verbosity": -1}
            if self.family == "tweedie":
                params["tweedie_variance_power"] = self.power
            if self.family == "quantile":
                params["alpha"] = self.alpha
            params.update(self.params)
            params.update(constraints or {})
            data = lgb.Dataset(x, label=y, weight=w, init_score=offset, free_raw_data=False)
            return lgb.train(params, data, num_boost_round=self.n_rounds)
        import xgboost as xgb

        params = {"objective": xgb_obj, "eta": self.learning_rate, "seed": seed}
        if self.family == "tweedie":
            params["tweedie_variance_power"] = self.power
        if self.family == "quantile":
            params["quantile_alpha"] = self.alpha
        params.update(self.params)
        params.update(constraints or {})
        data = xgb.DMatrix(x, label=y, weight=w, base_margin=offset)
        return xgb.train(params, data, num_boost_round=self.n_rounds)

    def fit(self, design, y):
        """Fits the booster on ``design``'s columns, offset and weights.

        Parameters
        ----------
        design : Design
        y : list of float

        Returns
        -------
        BoosterFit
        """
        import numpy as np

        if len(y) != design.n_rows:
            raise ValueError(f"{len(y)} responses for {design.n_rows} design rows")
        constraints = self.engine_params(design.names)
        x = _matrix(design)
        y = np.asarray(y, dtype=float)
        w = np.asarray(design.weights, dtype=float)
        offset = np.asarray(design.offset, dtype=float)
        if self.family == "quantile":
            if np.any(offset != 0.0):
                raise ValueError("a quantile booster takes no offset: put exposure in as a "
                                 "feature, or model the response per unit of exposure")
            base = _weighted_quantile(y, w, self.alpha)
        else:
            base = _base_score(self.family, y, w, offset)
        model = self._train(x, y, w, offset + base, self.seed, constraints)
        boots = []
        rng = random.Random(self.seed)
        n = len(y)
        for b in range(self.n_boot):
            rows = np.array([rng.randrange(n) for _ in range(n)])
            boots.append(self._train(x[rows], y[rows], w[rows], offset[rows] + base, self.seed + 1 + b,
                                     constraints))
        fit = BoosterFit(self, design.names, model, boots, 1.0, base)
        if self.family not in ("poisson", "quantile"):
            mu = np.asarray(fit.predict(design))
            v = np.array([_variance(self.family, m, self.power) for m in mu])
            fit.dispersion = float(np.sum(w * (y - mu) ** 2 / v) / n)
        if self.dispersion_model:
            fit.dispersion_fit = self._fit_dispersion(x, y, w, offset, base, constraints)
        return fit

    def _fit_dispersion(self, x, y, w, offset, base, constraints=None):
        """A gamma booster for the Pearson residuals of 5-fold cross-fitted
        means: ``E[w (y - mu)^2 / V(mu)] = phi`` row by row."""
        import numpy as np

        n = len(y)
        rng = random.Random(self.seed + 7919)
        folds = np.array([rng.randrange(5) for _ in range(n)])
        mu = np.empty(n)
        link = _OBJECTIVES[self.family][2]
        for k in range(5):
            test = folds == k
            if not test.any():
                continue
            train = ~test
            model = self._train(x[train], y[train], w[train], offset[train] + base, self.seed,
                                constraints)
            eta = _raw_margin(self.engine, model, x[test], offset[test] + base)
            mu[test] = np.exp(eta) if link == "log" else eta
        v = np.array([_variance(self.family, m, self.power) for m in mu])
        z = w * (y - mu) ** 2 / v
        z = np.maximum(z, 1e-12 * z.mean())
        spec = Booster("gamma", engine=self.engine, n_rounds=self.n_rounds,
                       learning_rate=self.learning_rate, params=self.params, seed=self.seed)
        zbase = float(np.log(z.mean()))
        model = spec._train(x, z, np.ones(n), np.full(n, zbase), self.seed)
        return (model, zbase)


class BoosterFit:
    """A fitted ``Booster``: means and joint predictive draws.

    Attributes
    ----------
    model
        The engine's booster (``lightgbm.Booster`` or ``xgboost.Booster``).
    base : float
        The constant added to the offset so the trees start from the mean.
    dispersion : float
        1 for the Poisson; otherwise Pearson's estimate on the training
        rows, ``sum w (y - mu)^2 / V(mu) / n`` (no degrees-of-freedom
        correction: trees have no fixed parameter count).
    dispersion_fit
        With ``dispersion_model=True``, the dispersion booster and its
        starting log level; otherwise ``None``.
    """

    def __init__(self, spec, names, model, boots, dispersion, base):
        self.spec = spec
        self.names = list(names)
        self.model = model
        self.boots = boots
        self.dispersion = dispersion
        self.base = base
        self.dispersion_fit = None

    def _means(self, model, design):
        import numpy as np

        if list(design.names) != self.names:
            raise ValueError(f"design columns {design.names} differ from the fit's {self.names}")
        x = _matrix(design)
        offset = np.asarray(design.offset, dtype=float) + self.base
        eta = _raw_margin(self.spec.engine, model, x, offset)
        link = _OBJECTIVES[self.spec.family][2]
        return (np.exp(eta) if link == "log" else eta).tolist()

    def predict(self, design):
        """Fitted means for the rows of ``design`` (the ``alpha`` quantiles
        for ``family="quantile"``).

        Parameters
        ----------
        design : Design
            Same columns as the training design; its offset applies.

        Returns
        -------
        list of float
        """
        return self._means(self.model, design)

    def check_monotone(self, design, feature, grid=None, direction=None, tol=1e-9):
        """The partial dependence of the predictions on ``feature`` (a
        design column) over ``grid``: the mean prediction over the rows of
        ``design`` with the column set to each value, and where it moves
        against ``direction`` (the fit's monotone constraint by default).
        The engines enforce a constraint split by split, so this confirms
        it on the model as fitted.

        Parameters
        ----------
        design : Design
        feature : str
        grid : list of float, optional
            Twenty quantiles of the column by default.
        direction : {1, -1}, optional
        tol : float, default 1e-9
            Relative tolerance on a step against the direction.

        Returns
        -------
        dict
            ``grid``, ``mean``, ``direction``, ``violations`` (indices ``i``
            whose step from ``i - 1`` goes against the direction) and
            ``monotone``.
        """
        import numpy as np

        from .models import Design

        names = list(design.names)
        if feature not in names:
            raise ValueError(f"{feature!r} is not a column of the design")
        if direction is None:
            direction = self.spec.monotone.get(feature)
        if direction not in (1, -1):
            raise ValueError("give direction 1 or -1, or fit with a monotone constraint on the feature")
        j = names.index(feature)
        cols = [list(design.column(k)) for k in range(len(names))]
        if grid is None:
            grid = sorted(set(np.quantile(cols[j], np.linspace(0, 1, 20)).tolist()))
        means = []
        for v in grid:
            c = [col if k != j else [float(v)] * design.n_rows for k, col in enumerate(cols)]
            d = Design(c, names, offset=list(design.offset), weights=list(design.weights))
            means.append(float(np.mean(self.predict(d))))
        scale = max(abs(m) for m in means) or 1.0
        bad = [i for i in range(1, len(means)) if direction * (means[i] - means[i - 1]) < -tol * scale]
        return {"grid": list(grid), "mean": means, "direction": direction, "violations": bad,
                "monotone": not bad}

    def predict_dispersion(self, design):
        """Each row's dispersion from the dispersion model, or the constant
        ``dispersion`` when the booster was fitted without one.

        Parameters
        ----------
        design : Design

        Returns
        -------
        list of float
        """
        import numpy as np

        if self.dispersion_fit is None:
            return [self.dispersion] * design.n_rows
        if list(design.names) != self.names:
            raise ValueError(f"design columns {design.names} differ from the fit's {self.names}")
        model, zbase = self.dispersion_fit
        x = _matrix(design)
        return np.exp(_raw_margin(self.spec.engine, model, x, np.full(len(x), zbase))).tolist()

    def predict_distribution(self, design, n_sims, seed):
        """Joint draws across the rows, keyed ``row = 0, 1, ...``: each row's
        response from the family around its mean (process noise), with the
        means of one bootstrap refit per simulation when ``n_boot > 0``
        (parameter uncertainty).

        Parameters
        ----------
        design : Design
        n_sims : int
        seed : int

        Returns
        -------
        PredictiveDistribution
        """
        if self.spec.family == "quantile":
            raise ValueError("a quantile booster predicts quantiles, not a distribution; "
                             "see predict_quantiles")
        models = self.boots or [self.model]
        means = [self._means(m, design) for m in models]
        return simulate_from_means(self.spec.family, means, n_sims, seed,
                                   dispersion=self.predict_dispersion(design),
                                   weights=list(design.weights),
                                   power=self.spec.power)


def predict_quantiles(fits, design):
    """Quantile sets from several quantile boosters on the same rows.

    Each fit predicts its own ``alpha`` quantile independently, so the
    predictions can cross (a 90% quantile below the 50% one on some row).
    Each row's predictions are sorted across the levels, which never makes
    any of them worse in pinball loss (Chernozhukov, Fernández-Val and
    Galichon, 2010).

    Parameters
    ----------
    fits : list of BoosterFit
        From ``Booster("quantile", alpha=...)``, with distinct ``alpha``.
    design : Design

    Returns
    -------
    dict
        ``{alpha: list of float}`` in increasing ``alpha``.
    """
    alphas = [f.spec.alpha for f in fits]
    if any(f.spec.family != "quantile" for f in fits) or len(set(alphas)) != len(alphas):
        raise ValueError("predict_quantiles needs quantile fits with distinct alpha")
    order = sorted(range(len(fits)), key=lambda i: alphas[i])
    preds = [fits[i].predict(design) for i in order]
    rows = [sorted(r) for r in zip(*preds)]
    return {alphas[i]: [r[k] for r in rows] for k, i in enumerate(order)}


def _raw_margin(engine, model, x, start):
    """The model's link-scale prediction plus ``start`` (offset + base)."""
    if engine == "lightgbm":
        return model.predict(x, raw_score=True) + start
    import xgboost as xgb

    return model.predict(xgb.DMatrix(x, base_margin=start), output_margin=True)


def _weighted_quantile(y, w, alpha):
    import numpy as np

    order = np.argsort(y)
    cw = np.cumsum(w[order])
    k = int(np.searchsorted(cw, alpha * cw[-1]))
    return float(y[order][min(k, len(y) - 1)])


def _base_score(family, y, w, offset):
    import numpy as np

    if _OBJECTIVES[family][2] == "log":
        total = float(np.sum(w * y))
        if total <= 0:
            raise ValueError("the weighted response total must be positive for a log link")
        return float(np.log(total / np.sum(w * np.exp(offset))))
    return float(np.sum(w * (y - offset)) / np.sum(w))


def _matrix(design):
    import numpy as np

    return np.column_stack([design.column(j) for j in range(len(design.names))])


_FUNCTION_COLUMN = re.compile(r"^(\w+)\(([^,/)]+)")
_MONOTONE_FUNCTIONS = {"hinge", "step", "log"}


def _column_feature(name):
    """A single-term column's feature and its kind: ``"plain"`` (the column
    itself), ``"monotone"`` (a monotone transform), ``"factor"`` (a level,
    bin or ordinal indicator) or ``"other"`` (a polynomial or spline)."""
    m = _FUNCTION_COLUMN.match(name)
    if m:
        return m.group(2), "monotone" if m.group(1) in _MONOTONE_FUNCTIONS else "other"
    if "[" in name:
        return name[:name.index("[")], "factor"
    if "^" in name:
        return name[:name.index("^")], "other"
    return name, "plain"


def _split_interaction(name):
    """An interaction column's parts, split at ``:`` outside brackets."""
    parts, depth, cur = [], 0, ""
    for ch in name:
        depth += ch in "[(" 
        depth -= ch in "])"
        if ch == ":" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return parts + [cur]


def _feature_columns(names, feature):
    """Every column a feature produces, interactions included."""
    cols = [j for j, n in enumerate(names)
            if any(_column_feature(p)[0] == feature for p in _split_interaction(n))]
    if not cols:
        raise ValueError(f"{feature!r} is not a feature of the design (columns: {names})")
    return cols


def _monotone_columns(names, feature):
    """The columns a monotone constraint on a feature goes on."""
    cols = []
    for j, n in enumerate(names):
        parts = _split_interaction(n)
        hits = [_column_feature(p) for p in parts if _column_feature(p)[0] == feature]
        if not hits:
            continue
        if len(parts) > 1:
            raise ValueError(f"{feature!r} enters the interaction column {n!r}; constrain its "
                             "main effect without the interaction, or give the column itself")
        kind = hits[0][1]
        if kind == "factor":
            raise ValueError(f"{feature!r} is a factor ({n!r}); a monotone constraint needs a numeric feature")
        if kind == "other":
            raise ValueError(f"{n!r} is not a monotone transform of {feature!r}")
        cols.append(j)
    if not cols:
        raise ValueError(f"{feature!r} is not a feature of the design (columns: {names})")
    return cols
