"""Tweedie fits round-trip through JSON and pickle (#978)."""

import pickle

import pytest

from prospicio.models import Design, TweedieFit, TweedieGlm


def _fit():
    x = [float(i % 2) for i in range(80)]
    y = [0.0 if i % 3 == 0 else 0.5 + (i * 7) % 5 * (1.0 + x[i]) for i in range(80)]
    d = Design([[1.0] * 80, x], ["(Intercept)", "x"])
    return d, TweedieGlm().fit(d, y)


def test_round_trip():
    d, fit = _fit()
    for back in (TweedieFit.from_json(fit.to_json()), pickle.loads(pickle.dumps(fit))):
        assert isinstance(back, TweedieFit)
        assert back.to_json() == fit.to_json()
        assert back.power == fit.power
        assert back.dispersion == fit.dispersion
        assert back.interval == fit.interval


def test_bad_artifacts_are_refused():
    _, fit = _fit()
    with pytest.raises(ValueError):
        TweedieFit.from_json(fit.to_json().replace("risk_rs.tweedie_fit", "risk_rs.glm_fit"))
    with pytest.raises(ValueError):
        TweedieFit.from_json(fit.to_json().replace('"format_version":1', '"format_version":2', 1))
