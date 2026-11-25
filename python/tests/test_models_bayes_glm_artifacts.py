"""Bayesian GLM fits round-trip through JSON and pickle (#978)."""

import pickle

import pytest

from prospicio.models import BayesGlm, BayesGlmFit, Design


def data():
    x = [(i % 5) / 2 - 1 for i in range(30)]
    y = [1 + (i % 5) * 0.7 + (i * 7 % 3) for i in range(30)]
    return Design([[1.0] * 30, x], ["(Intercept)", "x"]), y


@pytest.mark.parametrize("family", ["poisson", "gamma"])
def test_round_trip(family):
    d, y = data()
    fit = BayesGlm(family, link="log", chains=2, tune=60, draws=25, seed=7).fit(d, y)
    for back in (BayesGlmFit.from_json(fit.to_json()), pickle.loads(pickle.dumps(fit))):
        assert isinstance(back, BayesGlmFit)
        assert back.posterior_mean == fit.posterior_mean
        assert back.predict(d) == fit.predict(d)
        assert back.to_json() == fit.to_json()


def test_refuses_other_documents():
    d, y = data()
    text = BayesGlm("poisson", chains=1, tune=40, draws=10).fit(d, y).to_json()
    with pytest.raises(ValueError):
        BayesGlmFit.from_json(text.replace("risk_rs.bayes_glm_fit", "risk_rs.glm_fit"))
    with pytest.raises(ValueError):
        BayesGlmFit.from_json(text.replace('"format_version":1', '"format_version":2'))
