"""Methods on claim counts (``docs/design/reserving-v02.md``, decisions 33 and
34): frequency-severity, disposal rates, case outstanding development and
the Berquist-Sherman adjustments.

The Rust core is checked against Friedland's exhibits and chainladder-python
in ``validation/tests/reserving_counts.rs``; these tests reproduce printed
totals on XYZ Insurer through the bindings and check their arguments and
errors. ``R/prospicio/tests/test-reserving-counts.R`` asserts the same
numbers."""

import csv
import math
from pathlib import Path

import pytest

from prospicio.reserving import (
    CaseOutstanding,
    ChainLadder,
    Disposal,
    FrequencySeverity,
    Triangle,
    Trended,
    average_case,
    berquist_sherman_adjust,
    case_adequacy_adjust,
    case_development_factor,
    exponential_trend,
    pooled_trend,
    ratio_column,
    settlement_rate_adjust,
    trend_by_age,
)

DATA = Path(__file__).resolve().parents[2] / "validation" / "data" / "friedland_xyz.csv"
COLUMNS = ["reported", "paid", "reported_counts", "closed_counts", "premium"]


def xyz(first=1998, last=2008, max_age=132):
    with DATA.open() as f:
        rows = [
            r
            for r in csv.DictReader(line for line in f if not line.startswith("#"))
            if first <= int(r["origin"]) <= last and int(r["development"]) <= max_age
        ]
    return Triangle.from_long(
        [int(r["origin"]) for r in rows],
        [int(r["development"]) for r in rows],
        {c: [float(r[c]) for r in rows] for c in COLUMNS},
    )


def latest(triangle, column):
    return list(ChainLadder().fit(triangle, column).latest)


def small():
    return Triangle.from_long(
        [2020, 2020, 2021],
        [12, 24, 12],
        {
            "reported": [100.0, 150.0, 120.0],
            "paid": [50.0, 100.0, 60.0],
            "reported_counts": [10.0, 12.0, 11.0],
            "closed_counts": [5.0, 10.0, 6.0],
        },
    )


def test_frequency_severity_equals_the_chain_ladder_when_the_patterns_agree():
    # Rows that are multiples of one row: counts and severities develop on
    # consistent patterns, so counts times severity is the chain ladder.
    base = {12: 10.0, 24: 16.0, 36: 20.0}
    sev = {12: 5.0, 24: 6.0, 36: 7.0}
    origin, age, amounts, counts = [], [], [], []
    for k, (o, scale) in enumerate([(2019, 1.0), (2020, 2.0), (2021, 3.0)]):
        for a in list(base)[: 3 - k]:
            origin.append(o)
            age.append(a)
            counts.append(base[a] * scale)
            amounts.append(base[a] * sev[a] * scale)
    tri = Triangle.from_long(origin, age, {"amounts": amounts, "counts": counts})
    fit = FrequencySeverity("amounts", "counts").fit(tri)
    cl = ChainLadder().fit(tri, "amounts")
    assert fit.ultimate == pytest.approx(list(cl.ultimate), rel=1e-12)
    assert fit.total_reserve == pytest.approx(cl.total_reserve, rel=1e-12)


def test_trended_frequency_and_severity():
    tri = small()
    fit = FrequencySeverity(
        "reported",
        "reported_counts",
        ultimate_counts=Trended(0.5, 2021, 0.0),
        severity=Trended(20.0, 2021, 0.1),
        exposure="paid",
    ).fit(tri)
    # 100 paid at 0.5 a unit, a severity of 20 / 1.1 for 2020; 60 and 20 for 2021.
    assert fit.ultimate == pytest.approx([50.0 * 20.0 / 1.1, 30.0 * 20.0])
    assert fit.origins == ["2020", "2021"]


def test_friedland_xyz_case_outstanding():
    fit = CaseOutstanding("paid", "reported", 1.10, paid_on_case=2, case_on_case=2).fit(xyz())
    # Exhibit II's 513,944 and the 233 it left out of 2000.
    assert fit.total_ultimate == pytest.approx(513944.0 + 0.501 * 465.0, abs=8.0)
    assert fit.ibnr[-1] == pytest.approx(fit.ultimate[-1] - fit.latest_incurred[-1])
    assert fit.reserves[-1] == pytest.approx(fit.ultimate[-1] - fit.latest_paid[-1])


def test_friedland_xyz_disposal():
    tri = xyz(2001, 2008, 96)
    reported = [1455.0, 1554.0, 1629.0, 2258.0, 2390.0, 1657.0, 1264.0, 1036.0]
    cdf = [1.000, 1.000, 1.001, 1.002, 1.005, 1.013, 1.035, 1.131]
    rates = [0.244, 0.572, 0.704, 0.814, 0.910, 0.952, 0.982, 0.994]
    fit = Disposal(
        "closed_counts",
        "paid",
        [r * c for r, c in zip(reported, cdf)],
        disposal_rates={12 + 12 * k: r for k, r in enumerate(rates)},
        severities=2,
        tail_severity_from=84,
        trend=0.05,
        base=2008,
        factors=[0.670] * 5 + [0.750, 1.0, 1.0],
    ).fit(tri)
    assert fit.total_ultimate == pytest.approx(385676.0, abs=6.0)
    assert fit.ages == [12 + 12 * k for k in range(8)]
    assert fit.disposal_rates[-1][1] is None
    assert len(fit.selected_severities) == 9


def test_friedland_xyz_berquist_sherman():
    tri = xyz()
    closed = ("closed_counts", ChainLadder(n_periods=2))
    reported = ("reported_counts", ChainLadder(n_periods=2))
    case = case_adequacy_adjust(
        tri,
        "reported",
        "paid",
        0.05,
        reported="reported_counts",
        closed="closed_counts",
        max_age=84,
    )
    assert case.columns == tri.columns
    # The latest diagonal does not change.
    assert latest(case, "reported") == latest(tri, "reported")
    settled = settlement_rate_adjust(tri, "paid", "closed_counts", reported)
    assert latest(settled, "paid") == latest(tri, "paid")
    both = berquist_sherman_adjust(
        tri,
        "reported",
        "paid",
        "reported_counts",
        "closed_counts",
        0.05,
        reported,
        max_age=84,
    )
    assert latest(both, "reported") == latest(tri, "reported")
    extra = average_case(
        tri, "reported", "paid", 0.05, reported="reported_counts", closed="closed_counts"
    )
    assert {"average_case", "adjusted_average_case"} <= set(extra.columns)
    # A mean of two projections is accepted anywhere ultimate counts are.
    settlement_rate_adjust(tri, "paid", "closed_counts", [closed, reported])


def test_trends():
    assert exponential_trend([0.0, 1.0, 2.0], [100.0, 110.0, 121.0])["rate"] == pytest.approx(0.1)
    assert exponential_trend([0.0], [100.0]) is None
    tri = ratio_column(xyz(), "paid", "closed_counts", "average_paid")
    by_age = trend_by_age(tri, "average_paid")
    assert round(by_age[12]["rate"] * 100, 1) == 8.1
    assert math.isfinite(pooled_trend(tri, "average_paid")["rate"])
    assert case_development_factor(1.2, 2.0) == pytest.approx(1.5)


def test_arguments():
    tri = small()
    with pytest.raises(TypeError, match="disposal_rates must be"):
        Disposal("closed_counts", "paid", [10.0, 12.0], disposal_rates="latest")
    with pytest.raises(TypeError, match="ultimate_counts must be"):
        Disposal("closed_counts", "paid", 1.0).fit(tri)
    with pytest.raises(ValueError, match="not an origin"):
        Disposal("closed_counts", "paid", [10.0, 12.0], base=1999).fit(tri)
    with pytest.raises(ValueError, match="needs exposure"):
        FrequencySeverity("paid", "closed_counts", ultimate_counts=Trended(1.0, 2021, 0.0)).fit(tri)
    with pytest.raises(ValueError, match="not both"):
        case_adequacy_adjust(tri, "reported", "paid", 0.1, open_counts="x", reported="y")
    with pytest.raises(ValueError, match="interpolation"):
        settlement_rate_adjust(tri, "paid", "closed_counts", [10.0, 12.0], interpolation="spline")
    with pytest.raises(TypeError, match="severity must be"):
        FrequencySeverity("paid", "closed_counts", severity=1.0).fit(tri)
