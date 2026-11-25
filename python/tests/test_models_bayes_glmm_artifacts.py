"""Posterior draws and Bayesian GLMM fits round-trip through JSON and pickle (#978)."""

import math
import pickle

import pytest

from prospicio.models import BayesGlmm, BayesGlmmFit, Design, Draws


def test_draws_round_trip():
    d = Draws(["mu", "sigma"], [[[0.1, 1.0], [0.2, math.nan]], [[0.0, 0.9], [0.3, 1.2]]], divergences=[1, 0])
    for back in (Draws.from_json(d.to_json()), pickle.loads(pickle.dumps(d))):
        assert isinstance(back, Draws)
        assert back.to_json() == d.to_json()
    with pytest.raises(ValueError):
        Draws.from_json(d.to_json().replace('"format_version":1', '"format_version":2'))


def test_bayes_glmm_round_trip():
    y = [float((i * 7) % 5 + i % 4) for i in range(40)]
    groups = {"site": [f"s{i % 4}" for i in range(40)]}
    d = Design([[1.0] * 40], ["(Intercept)"])
    fit = BayesGlmm(chains=1, tune=60, draws=20, seed=3).fit(d, groups, y)
    for back in (BayesGlmmFit.from_json(fit.to_json()), pickle.loads(pickle.dumps(fit))):
        assert isinstance(back, BayesGlmmFit)
        assert back.predict(d, groups) == fit.predict(d, groups)
        assert back.to_json() == fit.to_json()
    with pytest.raises(ValueError):
        BayesGlmmFit.from_json(fit.to_json().replace("risk_rs.bayes_glmm_fit", "risk_rs.glm_fit"))
