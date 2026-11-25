"""MARS fits round-trip through JSON and pickle (#978)."""

import pickle

import pytest

from prospicio.models import MarsFit, mars


def data():
    a = [i / 2 for i in range(80)]
    b = [float((i * 17) % 23) for i in range(80)]
    y = [
        2 + 0.3 * u + 1.5 * max(0.0, u - 20) + 0.2 * max(0.0, 8 - v) + (i * 7 % 5) / 10
        for i, (u, v) in enumerate(zip(a, b))
    ]
    return {"a": a, "b": b}, y


@pytest.mark.parametrize("family", [None, "gamma"])
def test_round_trip(family):
    x, y = data()
    fit = mars(x, y, threshold=0.0, prune=True, family=family)
    for back in (MarsFit.from_json(fit.to_json()), pickle.loads(pickle.dumps(fit))):
        assert isinstance(back, MarsFit)
        assert back.coefficients == fit.coefficients
        assert back.knots("a") == fit.knots("a")
        assert back.to_json() == fit.to_json()


def test_refuses_other_documents():
    x, y = data()
    text = mars(x, y).to_json()
    with pytest.raises(ValueError):
        MarsFit.from_json(text.replace("risk_rs.mars_fit", "risk_rs.glm_fit"))
    with pytest.raises(ValueError):
        MarsFit.from_json(text.replace('"format_version":1', '"format_version":2'))
