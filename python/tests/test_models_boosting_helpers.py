"""The arithmetic the boosters share with R, from the Rust core (#977)."""

import math

import pytest

from prospicio.models import (
    Design,
    base_score,
    deviance,
    deviance_score,
    mean_deviance,
    monotone_violations,
    pearson_dispersion,
    pearson_terms,
    weighted_quantile,
)


def test_weighted_quantile_is_the_first_value_reaching_the_weight():
    y = [5.0, 1.0, 3.0, 2.0]
    w = [1.0, 1.0, 1.0, 1.0]
    assert weighted_quantile(y, w, 0.25) == 1.0
    assert weighted_quantile(y, w, 0.26) == 2.0
    assert weighted_quantile(y, w, 1.0) == 5.0
    with pytest.raises(ValueError):
        weighted_quantile(y, w, 1.5)


def test_base_score_is_the_weighted_mean_on_the_link_scale():
    y, w = [1.0, 2.0, 6.0], [1.0, 2.0, 1.0]
    off = [0.0, 0.0, math.log(2.0)]
    assert base_score("log", y, w, off) == pytest.approx(math.log(11.0 / 5.0), abs=1e-15)
    assert base_score("identity", y, w, [0.0, 0.0, 1.0]) == pytest.approx(2.5, abs=1e-15)
    with pytest.raises(ValueError, match="positive"):
        base_score("log", [0.0], [1.0], [0.0])


def test_pearson_dispersion_uses_the_family_variance():
    y, mu, w = [1.0, 3.0], [2.0, 2.0], [1.0, 2.0]
    assert pearson_dispersion("gamma", y, mu, w) == pytest.approx((0.25 + 0.5) / 2)
    terms = pearson_terms("tweedie", y, mu, w, power=1.5)
    assert terms[1] == pytest.approx(2.0 / 2.0**1.5)


def test_monotone_violations_are_relative_to_the_largest_value():
    assert monotone_violations([1.0, 2.0, 1.5, 3.0], 1) == [2]
    assert monotone_violations([3.0, 2.0, 2.2], -1, 0.1) == []
    with pytest.raises(ValueError):
        monotone_violations([1.0], 0)


def test_mean_deviance_and_deviance_score_agree():
    y, mu, w = [1.0, 3.0, 2.0], [2.0, 2.0, 2.5], [1.0, 2.0, 3.0]
    expected = deviance("poisson", y, mu, weights=w) / sum(w)
    assert mean_deviance("poisson", y, mu, weights=w) == pytest.approx(expected, rel=1e-15)
    d = Design([[1.0, 1.0, 1.0]], ["(Intercept)"], weights=w)
    assert deviance_score("poisson")(y, mu, d) == pytest.approx(expected, rel=1e-15)
