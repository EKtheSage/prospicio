"""One vocabulary for option names across Python, R and Excel (#834, step 1).

Each spelling an option accepts in any of the three front ends is accepted
here too and gives the same result as the canonical name.
"""

import math

import pytest

from prospicio.distributions import Count, Sampled
from prospicio.models import Design, ZeroInflated
from prospicio.reserving import (
    BenchmarkCurve,
    ChainLadder,
    Mack,
    MunichChainLadder,
    PaidIncurredChain,
    Triangle,
)
from prospicio.risk import calibrate

ROWS = [
    [110.0, 290.0, 370.0, 420.0, 440.0, 452.0],
    [95.0, 300.0, 390.0, 425.0, 447.0],
    [130.0, 320.0, 410.0, 470.0],
    [105.0, 305.0, 395.0],
    [120.0, 310.0],
    [125.0],
]


def triangle(grain="Y"):
    return Triangle.from_long(
        [2020 + i for i, row in enumerate(ROWS) for _ in row],
        [12 * (d + 1) for row in ROWS for d in range(len(row))],
        {
            "paid": [v for row in ROWS for v in row],
            "incurred": [1.3 * v for row in ROWS for v in row],
        },
        origin_grain=grain,
        development_grain=grain,
    )


def test_count_distribution_negbin():
    import csv
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "validation" / "data" / "biochemists.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    y = [float(r["art"]) for r in rows]
    x = Design(
        [[1.0] * len(rows), [float(r["ment"]) for r in rows]], ["(Intercept)", "ment"]
    )
    a = ZeroInflated("negative_binomial").fit(x, y)
    b = ZeroInflated("negbin").fit(x, y)
    assert a.count == b.count == "negative_binomial"
    assert a.count_coefficients == b.count_coefficients


@pytest.mark.parametrize("name", ["loglinear", "log_linear"])
def test_sigma_interpolation_spellings(name):
    tri = triangle()
    want = Mack(sigma_interpolation="log-linear").fit(tri, "paid").total_reserve
    assert Mack(sigma_interpolation=name).fit(tri, "paid").total_reserve == want
    m = MunichChainLadder(sigma_interpolation_paid=name, sigma_interpolation_incurred=name)
    base = MunichChainLadder().fit(tri, "paid", "incurred")
    assert m.fit(tri, "paid", "incurred").ultimate_paid == base.ultimate_paid


@pytest.mark.parametrize("name", ["log-linear", "log_linear"])
def test_variance_extrapolation_spellings(name):
    tri = triangle()
    want = PaidIncurredChain().fit(tri, "paid", "incurred").total_reserve
    got = PaidIncurredChain(extrapolation=name).fit(tri, "paid", "incurred").total_reserve
    assert got == want


@pytest.mark.parametrize("grain", ["y", " Y ", "year", "Year"])
def test_grain_spellings(grain):
    want = ChainLadder().fit(triangle("Y"), "paid").ultimate
    assert ChainLadder().fit(triangle(grain), "paid").ultimate == want


def test_mixing_ig():
    a = Count.mixed_poisson(10.0, 0.5, mixing="inverse_gaussian")
    b = Count.mixed_poisson(10.0, 0.5, mixing="ig")
    assert a.pmf(3) == b.pmf(3)


def test_mixing_names():
    # The mixing's names come from one table (#834, step 2).
    with pytest.raises(ValueError, match='mixing must be "gamma", "inverse_gaussian", "ig", "beta", '
                       '"sichel", "sichel_skewness", "sichel_gamma" or "sichel_ig", got "lognormal"'):
        Count.mixed_poisson(10.0, 0.5, mixing="lognormal")
    a = Count.mixed_poisson(10.0, 0.5, mixing="sichel", skewness=1.5)
    b = Count.mixed_poisson(10.0, 0.5, mixing="sichel_skewness", skewness=1.5)
    assert a.pmf(3) == b.pmf(3)


def test_bonus_malus_mixing_ig():
    from prospicio.pricing import BonusMalus

    scale = BonusMalus.rule(6, 5, down=1, up_per_claim=2)
    a = scale.optimal_relativities(0.1, cv=0.8, mixing="inverse_gaussian")
    assert scale.optimal_relativities(0.1, cv=0.8, mixing="ig") == a


@pytest.mark.parametrize(
    "canonical, other",
    [("ldf", "age-to-age"), ("cdf", "age-to-ultimate"),
     ("percent_developed", "percent"), ("percent_developed", "percent developed")],
)
def test_pattern_kind_spellings(canonical, other):
    values = {
        "ldf": {12: 2.5, 24: 1.3, 36: 1.1, 48: 1.05, 60: 1.02},
        "cdf": {12: 4.0, 24: 1.6, 36: 1.25, 48: 1.12, 60: 1.04},
        "percent_developed": {12: 0.25, 24: 0.625, 36: 0.8, 48: 0.89, 60: 0.96},
    }[canonical]
    tri = triangle()
    want = ChainLadder(ldf=values, ldf_kind=canonical).fit(tri, "paid").ultimate
    assert ChainLadder(ldf=values, ldf_kind=other).fit(tri, "paid").ultimate == want


@pytest.mark.parametrize("other", ["percent", "percent developed"])
def test_benchmark_kind_spellings(other):
    bench = {12: 0.18, 24: 0.47, 36: 0.66, 48: 0.78, 60: 0.855, 72: 0.90}
    want = BenchmarkCurve().fit(bench, "percent_developed").omega
    assert BenchmarkCurve().fit(bench, other).omega == want


@pytest.mark.parametrize(
    "name", ["ph", "proportional_hazard", "Proportional Hazard", "proportional-hazard"]
)
def test_distortion_family_spellings(name):
    x = Sampled([22, 28, 36, 40, 40, 40, 40, 55, 65, 100])
    premium = (46.6 + 0.15 * 100) / 1.15
    d = calibrate(name, x, premium)
    assert math.isclose(d.measure(x), premium, rel_tol=1e-9)
    with pytest.raises(ValueError, match="unknown distortion family"):
        calibrate("bitvar", x, premium)
