"""A growth curve fitted to a benchmark pattern (``BenchmarkCurve``,
``docs/design/reserving-v02.md``, decision 20). The Rust core is checked
against scipy's ``least_squares`` in
``validation/tests/reserving_benchmark_curve.rs``; these tests check the
bindings' arguments, the same reference values and the pattern in the chain
ladder and the expected-loss methods."""

import pytest

from prospicio.reserving import (
    BenchmarkCurve,
    BornhuetterFerguson,
    ChainLadder,
    ClarkLdf,
    Triangle,
)

AGES = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
PERCENT = [0.18, 0.47, 0.66, 0.78, 0.855, 0.90, 0.93, 0.951, 0.965, 0.975]
WEIGHTS = [1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
BENCH = dict(zip(AGES, PERCENT))

ROWS = [
    [110.0, 290.0, 370.0, 420.0, 440.0, 452.0],
    [95.0, 300.0, 390.0, 425.0, 447.0],
    [130.0, 320.0, 410.0, 470.0],
    [105.0, 305.0, 395.0],
    [120.0, 310.0],
    [125.0],
]


def triangle():
    return Triangle.from_long(
        [2020 + i for i, row in enumerate(ROWS) for _ in row],
        [12 * (d + 1) for row in ROWS for d in range(len(row))],
        {
            "paid": [v for row in ROWS for v in row],
            "premium": [800.0 for row in ROWS for _ in row],
        },
    )


@pytest.mark.parametrize(
    ("settings", "weighted", "omega", "theta", "sse"),
    [
        ({}, False, 2.09195678319105, 25.5547370205669, 0.000775555676218485),
        (
            {"curve": "weibull", "max_age": 240.0, "origin_width": 12.0},
            False,
            1.00079725545721,
            28.350802255576,
            0.000447194375150825,
        ),
        ({"origin_width": 12.0}, True, 1.6187953142168, 18.9910413059938, 0.00480838215217264),
    ],
)
def test_matches_scipy_least_squares(settings, weighted, omega, theta, sse):
    weights = dict(zip(AGES, WEIGHTS)) if weighted else None
    fit = BenchmarkCurve(**settings).fit(BENCH, weights=weights)
    assert fit.omega == pytest.approx(omega, rel=1e-7)
    assert fit.theta == pytest.approx(theta, rel=1e-7)
    assert fit.sum_of_squares == pytest.approx(sse, rel=1e-9)
    assert fit.ages == [float(a) for a in AGES]
    assert fit.observed == PERCENT
    for o, f, r in zip(fit.observed, fit.fitted, fit.residuals):
        assert r == pytest.approx(o - f, abs=1e-15)


def test_factors_to_ultimate_and_weights_by_age():
    by_percent = BenchmarkCurve().fit(BENCH)
    by_cdf = BenchmarkCurve().fit({a: 1.0 / p for a, p in BENCH.items()}, kind="cdf")
    assert by_cdf.omega == pytest.approx(by_percent.omega, rel=1e-7)
    # An age not listed in weights gets 1.
    partial = BenchmarkCurve().fit(BENCH, weights={24: 2.0})
    assert partial.weights == [1.0, 2.0] + [1.0] * 8
    # Ages need not be whole months.
    half = BenchmarkCurve().fit({6.5: 0.1, 18.5: 0.4, 30.5: 0.6})
    assert half.ages == [6.5, 18.5, 30.5]


def test_recovers_clark_and_reproduces_its_ultimates():
    tri = triangle()
    clark = ClarkLdf(curve="weibull").fit(tri, "paid")
    bench = {12 * k: clark.growth(12 * k) for k in range(1, 9)}
    fit = BenchmarkCurve(curve="weibull", origin_width=12).fit(bench)
    assert fit.omega == pytest.approx(clark.omega, rel=1e-7)
    assert fit.theta == pytest.approx(clark.theta, rel=1e-7)
    ages = [12, 24, 36, 48, 60, 72]
    cl = ChainLadder(**fit.selected(ages)).fit(tri, "paid")
    assert cl.pattern_tail == pytest.approx(fit.cdf(72), rel=1e-12)
    assert cl.ultimate == pytest.approx(clark.ultimate, rel=1e-7)
    # The expected-loss methods take the same pattern.
    bf = BornhuetterFerguson(apriori=0.6, **fit.selected(ages)).fit(tri, "paid", "premium")
    latest = [row[-1] for row in ROWS]
    for u, l, a in zip(bf.ultimate, latest, reversed(ages)):
        expected = l + 0.6 * 800.0 * (1.0 - fit.percent_developed(a))
        assert u == pytest.approx(expected, rel=1e-9)


def test_values_and_selected():
    fit = BenchmarkCurve(max_age=180.0).fit(BENCH)
    assert fit.percent_developed(180) == 1.0
    assert fit.cdf(12) == pytest.approx(1.0 / fit.percent_developed(12))
    assert fit.curve_age(12) == 12.0
    s = fit.selected([12, 24])
    assert s["ldf_kind"] == "percent_developed" and s["ldf_weight"] == 1.0
    assert s["ldf"] == {12: fit.percent_developed(12), 24: fit.percent_developed(24)}
    assert "BenchmarkCurve(curve=\"loglogistic\"" in repr(BenchmarkCurve())


def test_errors():
    with pytest.raises(ValueError, match="no level"):
        BenchmarkCurve().fit(BENCH, kind="ldf")
    with pytest.raises(ValueError, match="kind"):
        BenchmarkCurve().fit(BENCH, kind="percent")
    with pytest.raises(ValueError, match="unknown growth curve"):
        BenchmarkCurve(curve="gompertz")
    with pytest.raises(ValueError, match="not an age of the pattern"):
        BenchmarkCurve().fit(BENCH, weights={18: 1.0})
    with pytest.raises(ValueError, match="at least two ages"):
        BenchmarkCurve().fit({12: 0.5})
    with pytest.raises(ValueError, match="max_age"):
        BenchmarkCurve(max_age=60.0).fit(BENCH)
    with pytest.raises(TypeError, match="dict"):
        BenchmarkCurve().fit(PERCENT)
    fit = BenchmarkCurve().fit(BENCH)
    with pytest.raises(ValueError, match="given twice"):
        fit.selected([12, 12])
