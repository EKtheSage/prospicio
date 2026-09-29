import math
import pickle

import pytest
from scipy import stats

import actuarialrs as ar


def test_matches_scipy():
    d = ar.distributions.Lognormal(7.0, 0.5)
    ref = stats.lognorm(s=0.5, scale=math.exp(7.0))
    assert d.mean() == pytest.approx(ref.mean(), rel=1e-13)
    assert d.variance() == pytest.approx(ref.var(), rel=1e-12)
    for p in (0.01, 0.5, 0.995):
        assert d.quantile(p) == pytest.approx(ref.ppf(p), rel=1e-12)
    assert d.cdf(1500.0) == pytest.approx(ref.cdf(1500.0), rel=1e-12)


def test_from_mean_cv():
    d = ar.distributions.Lognormal.from_mean_cv(1000.0, 0.5)
    assert d.mean() == pytest.approx(1000.0)
    assert d.std() == pytest.approx(500.0)


def test_invalid_input_raises_value_error():
    with pytest.raises(ValueError, match="sdlog"):
        ar.distributions.Lognormal(0.0, -1.0)
    with pytest.raises(ValueError, match="probability"):
        ar.distributions.Lognormal(0.0, 1.0).quantile(2.0)


def test_sample_reproducible_and_matches_rust_stream():
    d = ar.distributions.Lognormal(0.0, 1.0)
    a = d.sample(5, seed=42, stream=3)
    assert a == d.sample(5, seed=42, stream=3)
    assert a != d.sample(5, seed=42, stream=4)
    # Same values are pinned in the Rust and R tests.
    assert a[:3] == [1.0007760893701914, 1.6293872534754683, 1.0763869265482304]


def test_pickle_round_trip():
    d = ar.distributions.Lognormal(1.5, 0.25)
    back = pickle.loads(pickle.dumps(d))
    assert (back.meanlog, back.sdlog) == (1.5, 0.25)
    assert repr(back) == "Lognormal(meanlog=1.5, sdlog=0.25)"
