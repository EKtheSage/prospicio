"""Gradient boosting through LightGBM or XGBoost, behind the model
protocol (docs/design/models.md: "Gradient boosting stays an adapter").

The adapter converts a ``Design`` and responses into the engine's inputs,
calls the engine, and wraps its output in the shared objects: means from
``predict``, joint draws from ``predict_distribution``. It adds no
evaluation logic: ``compare``, ``cross_validate`` and the metrics in
``actuarialrs.models`` take a ``Booster`` like any other model.

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

LightGBM and XGBoost are optional: install the one you use
(``pip install lightgbm`` or ``pip install xgboost``).
"""

import random

from .actuarialrs_native import simulate_from_means

__all__ = ["Booster", "BoosterFit"]

# family -> (LightGBM objective, XGBoost objective, link)
_OBJECTIVES = {
    "poisson": ("poisson", "count:poisson", "log"),
    "gamma": ("gamma", "reg:gamma", "log"),
    "tweedie": ("tweedie", "reg:tweedie", "log"),
    "gaussian": ("regression", "reg:squarederror", "identity"),
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
    family : {"poisson", "gamma", "tweedie", "gaussian"}, default "poisson"
        The engine's objective; a log link for the first three.
    engine : {"lightgbm", "xgboost"}, default "lightgbm"
    power : float, optional
        Tweedie variance power in (1, 2); required for ``"tweedie"``.
    n_rounds : int, default 200
        Boosting rounds.
    learning_rate : float, default 0.05
    params : dict, optional
        Further engine parameters (``num_leaves``, ``max_depth``,
        ``monotone_constraints``, ...), passed as they are.
    n_boot : int, default 0
        Bootstrap refits for parameter uncertainty in
        ``predict_distribution``; 0 gives process noise only.
    seed : int, default 0
        Seeds the engine and the bootstrap resamples.

    Examples
    --------
    >>> from actuarialrs.boosting import Booster
    >>> from actuarialrs.models import Design
    >>> x = [i % 10 / 10 for i in range(200)]
    >>> d = Design([x], ["x"], offset=[0.0] * 200)
    >>> y = [float(i % 3 == 0) + v for i, v in enumerate(x)]
    >>> fit = Booster("poisson", n_rounds=20).fit(d, y)
    >>> len(fit.predict(d))
    200
    """

    def __init__(self, family="poisson", engine="lightgbm", power=None, n_rounds=200,
                 learning_rate=0.05, params=None, n_boot=0, seed=0):
        if family not in _OBJECTIVES:
            raise ValueError(f"family must be one of {sorted(_OBJECTIVES)}, got {family!r}")
        if engine not in _ENGINES:
            raise ValueError(f"engine must be one of {list(_ENGINES)}, got {engine!r}")
        if family == "tweedie" and not (power is not None and 1.0 < power < 2.0):
            raise ValueError("tweedie needs power in (1, 2)")
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

    def _train(self, x, y, w, offset, seed):
        lgb_obj, xgb_obj, _ = _OBJECTIVES[self.family]
        if self.engine == "lightgbm":
            import lightgbm as lgb

            params = {"objective": lgb_obj, "learning_rate": self.learning_rate,
                      "seed": seed, "deterministic": True, "verbosity": -1}
            if self.family == "tweedie":
                params["tweedie_variance_power"] = self.power
            params.update(self.params)
            data = lgb.Dataset(x, label=y, weight=w, init_score=offset, free_raw_data=False)
            return lgb.train(params, data, num_boost_round=self.n_rounds)
        import xgboost as xgb

        params = {"objective": xgb_obj, "eta": self.learning_rate, "seed": seed}
        if self.family == "tweedie":
            params["tweedie_variance_power"] = self.power
        params.update(self.params)
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
        x = _matrix(design)
        y = np.asarray(y, dtype=float)
        w = np.asarray(design.weights, dtype=float)
        offset = np.asarray(design.offset, dtype=float)
        base = _base_score(self.family, y, w, offset)
        model = self._train(x, y, w, offset + base, self.seed)
        boots = []
        rng = random.Random(self.seed)
        n = len(y)
        for b in range(self.n_boot):
            rows = np.array([rng.randrange(n) for _ in range(n)])
            boots.append(self._train(x[rows], y[rows], w[rows], offset[rows] + base, self.seed + 1 + b))
        fit = BoosterFit(self, design.names, model, boots, 1.0, base)
        if self.family != "poisson":
            mu = np.asarray(fit.predict(design))
            v = np.array([_variance(self.family, m, self.power) for m in mu])
            fit.dispersion = float(np.sum(w * (y - mu) ** 2 / v) / n)
        return fit


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
    """

    def __init__(self, spec, names, model, boots, dispersion, base):
        self.spec = spec
        self.names = list(names)
        self.model = model
        self.boots = boots
        self.dispersion = dispersion
        self.base = base

    def _means(self, model, design):
        import numpy as np

        if list(design.names) != self.names:
            raise ValueError(f"design columns {design.names} differ from the fit's {self.names}")
        x = _matrix(design)
        offset = np.asarray(design.offset, dtype=float) + self.base
        if self.spec.engine == "lightgbm":
            eta = model.predict(x, raw_score=True) + offset
        else:
            import xgboost as xgb

            eta = model.predict(xgb.DMatrix(x, base_margin=offset), output_margin=True)
        link = _OBJECTIVES[self.spec.family][2]
        return (np.exp(eta) if link == "log" else eta).tolist()

    def predict(self, design):
        """Fitted means for the rows of ``design``.

        Parameters
        ----------
        design : Design
            Same columns as the training design; its offset applies.

        Returns
        -------
        list of float
        """
        return self._means(self.model, design)

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
        models = self.boots or [self.model]
        means = [self._means(m, design) for m in models]
        return simulate_from_means(self.spec.family, means, n_sims, seed,
                                   dispersion=self.dispersion, weights=list(design.weights),
                                   power=self.spec.power)


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
