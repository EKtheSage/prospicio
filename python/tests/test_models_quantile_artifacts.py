"""Quantile and expectile fits round-trip through JSON and pickle (#978)."""

import pickle

import pytest

from prospicio.models import (
    Design,
    ExpectileFit,
    ExpectileRegression,
    QuantileFit,
    QuantileRegression,
)


def data():
    x = [i / 4 for i in range(40)]
    y = [1.0 + 0.5 * v + ((i * 37) % 11) / 3 for i, v in enumerate(x)]
    return Design([[1.0] * 40, x], ["(Intercept)", "x"]), y


@pytest.mark.parametrize("spec, cls", [(QuantileRegression(0.9), QuantileFit),
                                       (ExpectileRegression(0.2), ExpectileFit)])
def test_fits_round_trip(spec, cls):
    d, y = data()
    fit = spec.fit(d, y)
    for back in (cls.from_json(fit.to_json()), pickle.loads(pickle.dumps(fit))):
        assert back.coefficients == fit.coefficients
        assert back.predict(d) == fit.predict(d)
        assert back.to_json() == fit.to_json()


def test_formats_are_not_interchangeable():
    d, y = data()
    with pytest.raises(ValueError):
        ExpectileFit.from_json(QuantileRegression(0.5).fit(d, y).to_json())
