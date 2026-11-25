"""GLMM fits round-trip through JSON and pickle (#978)."""

import pickle

import pytest

from prospicio.models import Design, Glmm, GlmmFit


def data():
    n = 48
    x = [(i * 7 % 11) / 10 - 0.5 for i in range(n)]
    groups = {"g": [f"g{i % 8}" for i in range(n)], "h": [f"h{i % 3}" for i in range(n)]}
    y = [float(round(1.5 + i * 13 % 7 + (i % 8) * 0.5 + x[i])) for i in range(n)]
    return Design([[1.0] * n, x], ["(Intercept)", "x"]), groups, y


@pytest.mark.parametrize("family", ["poisson", "gamma"])
def test_round_trip(family):
    d, groups, y = data()
    fit = Glmm(family).fit(d, groups, y)
    for back in (GlmmFit.from_json(fit.to_json()), pickle.loads(pickle.dumps(fit))):
        assert isinstance(back, GlmmFit)
        assert back.coefficients == fit.coefficients
        assert back.predict(d, groups) == fit.predict(d, groups)
        assert back.to_json() == fit.to_json()


def test_quadrature_round_trip():
    d, groups, y = data()
    fit = Glmm(nodes=5).fit(d, {"g": groups["g"]}, y)
    assert GlmmFit.from_json(fit.to_json()).to_json() == fit.to_json()


def test_refuses_other_documents():
    d, groups, y = data()
    text = Glmm().fit(d, {"g": groups["g"]}, y).to_json()
    with pytest.raises(ValueError):
        GlmmFit.from_json(text.replace("risk_rs.glmm_fit", "risk_rs.glm_fit"))
    with pytest.raises(ValueError):
        GlmmFit.from_json(text.replace('"format_version":1', '"format_version":2'))
