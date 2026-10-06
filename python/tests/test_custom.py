import math
import pickle

import pytest

from actuarialrs.aggregate import simulate_events
from actuarialrs.distributions import Custom, Grid, Lognormal, Mixture, Poisson

LN = Lognormal.from_mean_cv(1000.0, 1.0)


def ln_cdf(x):
    return LN.cdf(x)


def ln_quantile(p):
    return LN.quantile(p)


@pytest.mark.parametrize("quantile", [None, ln_quantile])
def test_matches_the_native_family(quantile):
    d = Custom(ln_cdf, quantile, name="lognormal")
    assert d.mean() == pytest.approx(LN.mean(), rel=1e-7)
    assert d.variance() == pytest.approx(LN.variance(), rel=1e-4)
    for limit in [100.0, 1000.0, 5000.0]:
        assert d.lev(limit) == pytest.approx(LN.lev(limit), rel=1e-9)
    assert d.layer(2000.0, 1000.0) == pytest.approx(LN.layer(2000.0, 1000.0), rel=1e-8)
    assert d.quantile(0.99) == pytest.approx(LN.quantile(0.99), rel=1e-12)
    assert d.has_quantile == (quantile is not None)
    assert d.name == "lognormal"


def test_goes_where_a_severity_goes():
    d = Custom(ln_cdf, ln_quantile)
    # Same quantile function, same streams: the same simulated years.
    a = simulate_events(Poisson(3.0), d, 2_000, 5)
    b = simulate_events(Poisson(3.0), LN, 2_000, 5)
    assert a.totals().mean() == pytest.approx(b.totals().mean(), rel=1e-12)
    grid, report = Grid.local_moment(d, 100.0, 400)
    assert grid.mean() == pytest.approx(LN.mean(), rel=1e-3)
    m = Mixture([(0.5, d), (0.5, LN)])
    assert m.mean() == pytest.approx(LN.mean(), rel=1e-6)


def test_errors_are_raised_or_kept():
    def boom(x):
        if x > 50:
            raise RuntimeError("boom")
        return x / 100

    with pytest.raises(ValueError, match="boom"):
        Custom(boom)
    with pytest.raises(ValueError, match="does not reach"):
        Custom(lambda x: 0.5 * (1 - math.exp(-x)))
    with pytest.raises(TypeError):
        Custom(3.0)

    def late(x):
        if x == 12345.0:
            raise ValueError("late")
        return 1 - math.exp(-x)

    d = Custom(late)
    assert d.last_error is None
    assert math.isnan(d.cdf(12345.0))
    assert "late" in d.last_error


def test_pickles_with_module_level_functions():
    d = Custom(ln_cdf, ln_quantile, name="ln")
    back = pickle.loads(pickle.dumps(d))
    assert back.mean() == d.mean() and back.name == "ln"
    assert "Custom(name=\"ln\"" in repr(d)
