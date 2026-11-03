"""Friedland's exhibits and the weighted selection of ultimates
(docs/design/reserving-v02.md, decision 30; issue #557).

Friedland (2010), Estimating Unpaid Claims Using Basic Techniques: chapter 7,
Exhibit I (U.S. Industry Auto) and chapter 15, Exhibit I, sheets 2 and 6 (XYZ
Insurer), to the printed digits; chainladder-python's VotingChainladder on
GenIns, from validation/reference/reserving_exhibits_python.csv.
"""

import csv
import math
from pathlib import Path

import pytest

from prospicio.reserving import (
    BornhuetterFerguson,
    CapeCod,
    ChainLadder,
    DevelopmentExhibit,
    ExpectedLoss,
    SelectedUltimates,
    Triangle,
    UltimateExhibit,
    select_ultimates,
)

ROOT = Path(__file__).resolve().parents[2]
AGES = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
CDF = {
    "reported": [1.292, 1.110, 1.051, 1.023, 1.011, 1.006, 1.003, 1.001, 1.000, 1.000],
    "paid": [2.390, 1.404, 1.184, 1.085, 1.040, 1.020, 1.011, 1.006, 1.004, 1.002],
}


def read_triangle(name, columns):
    rows = [r for r in (ROOT / "validation" / "data" / f"{name}.csv").read_text().splitlines()
            if not r.startswith("#")]
    table = list(csv.DictReader(rows))
    return Triangle.from_long(
        [int(r["origin"]) for r in table],
        [int(r["development"]) for r in table],
        {c: [float(r[c]) for r in table] for c in columns},
    )


@pytest.fixture(scope="module")
def us():
    return read_triangle("friedland_us_auto", ["reported", "paid", "premium"])


def printed(values, digits):
    return [round(v, digits) for v in values]


def test_development_exhibit_reproduces_chapter_7(us):
    # Chapter 7, Exhibit I, sheet 1, parts 3 and 4.
    selected = [1.164, 1.056, 1.027, 1.012, 1.005, 1.003, 1.002, 1.001, 1.000]
    ex = DevelopmentExhibit(us, "reported", ChainLadder(ldf=dict(zip(AGES, selected)), ldf_kind="ldf"))
    assert list(ex.averages) == [
        "simple latest 5", "simple latest 3", "medial latest 5x1",
        "volume latest 5", "volume latest 3", "geometric latest 4",
    ]
    assert printed(ex.averages["medial latest 5x1"], 3) == [
        1.165, 1.057, 1.027, 1.010, 1.004, 1.003, 1.002, 1.001, 1.000]
    assert printed(ex.averages["geometric latest 4"], 3) == [
        1.164, 1.057, 1.027, 1.011, 1.004, 1.003, 1.002, 1.001, 1.000]
    assert ex.selected == selected
    assert printed(ex.cdf, 3) == CDF["reported"]
    assert ex.link_labels[0] == "12-24" and ex.link_labels[-1] == "120-ult"
    assert round(ex.link_ratios[0][0], 3) == 1.166
    assert math.isnan(ex.link_ratios[9][0])
    # The percents use the book's rounded CDFs.
    ex = DevelopmentExhibit(us, "paid", ChainLadder(ldf=dict(zip(AGES, CDF["paid"])), ldf_kind="cdf"),
                            averages={})
    assert printed([100 * p for p in ex.percent_developed], 1) == [
        41.8, 71.2, 84.5, 92.2, 96.2, 98.0, 98.9, 99.4, 99.6, 99.8]
    assert ex.tail == pytest.approx(1.002)
    pd = pytest.importorskip("pandas")
    ratios, factors = DevelopmentExhibit(us, "paid").to_frames()
    assert isinstance(ratios, pd.DataFrame)
    assert list(factors.index)[-3:] == ["selected", "cdf", "percent_developed"]
    assert math.isnan(factors.loc["simple latest 5", "120-ult"])


def test_ultimate_exhibit_reproduces_sheet_4(us):
    def cdf(column):
        return ChainLadder(ldf=dict(zip(AGES, CDF[column])), ldf_kind="cdf")

    apriori = dict(zip(range(1998, 2008), [0.75] * 5 + [0.65] * 5))
    ex = UltimateExhibit.from_triangle(us, reported="reported", paid="paid")
    ex = ex.with_method("development reported", cdf("reported").fit(us, "reported"))
    ex = ex.with_method("development paid", cdf("paid").fit(us, "paid"))
    ex = ex.with_method("expected claims", ExpectedLoss(apriori=apriori, ldf=dict(zip(AGES, CDF["reported"])),
                                                        ldf_kind="cdf").fit(us, "reported", "premium"))
    ex = ex.with_method("bf paid", BornhuetterFerguson(apriori=apriori, ldf=dict(zip(AGES, CDF["paid"])),
                                                       ldf_kind="cdf").fit(us, "paid", "premium"))
    ex = ex.with_method("cape cod", CapeCod(ldf=dict(zip(AGES, CDF["reported"])), ldf_kind="cdf")
                        .fit(us, "reported", "premium"))
    assert ex.ages == AGES[::-1]
    assert round(sum(ex.case_outstanding)) == 45431219
    for name, book in [
        ("development reported", (569172456, 25690869, 71122088)),
        ("development paid", (572041099, 28559512, 73990731)),
        ("expected claims", (569281839, 25800252, 71231471)),
        ("bf paid", (570568198, 27086611, 72517830)),
        ("cape cod", (570800677, 27319090, 72750309)),
    ]:
        got = (round(sum(ex.ultimate(name))), round(sum(ex.ibnr(name))), round(sum(ex.unpaid(name))))
        assert got == book, name
    assert [round(v) for v in ex.ibnr("development paid")][:3] == [-2829, 18769, 22495]
    table = ex.to_frame("ibnr")
    assert list(table["development reported"].round())[-1] == 14265240
    with pytest.raises(ValueError, match="quantity"):
        ex.to_frame("case")


XYZ = {
    "development both": [15822, 25107, 37246, 38798, 48169, 44373, 70780, 71362, 53392, 42680, 41531],
    "bf reported": [15822, 25107, 37246, 38798, 48169, 44373, 70792, 71554, 53906, 40300, 36842],
    "bf paid": [15975, 25128, 37294, 39274, 47313, 45070, 71688, 77898, 56031, 34988, 33988],
    "frequency-severity 2": [math.nan] * 9 + [30512, 30140],
}
XYZ_REPORTED = [15822, 25107, 37246, 38798, 48169, 44373, 70288, 70655, 48804, 31732, 18632]
XYZ_PAID = [15822, 24817, 36782, 38519, 44437, 39320, 52811, 40026, 22819, 11865, 3409]
XYZ_SELECTED = [15822, 25107, 37246, 38798, 48169, 44373, 70780, 74726, 54968, 40300, 33491]


def xyz():
    ex = UltimateExhibit(list(range(1998, 2009)), [132 - 12 * k for k in range(11)],
                         reported=XYZ_REPORTED, paid=XYZ_PAID)
    for name, ult in XYZ.items():
        ex = ex.with_method(name, [float(u) for u in ult])
    return ex


def test_chapter_15_selection_by_origin_and_by_age():
    ex = xyz()
    by_origin = {
        "development both": [1.0] * 7 + [0.0] * 4,
        "bf reported": [0.0] * 7 + [0.5, 0.5, 1.0, 0.5],
        "bf paid": [0.0] * 7 + [0.5, 0.5, 0.0, 0.0],
        "frequency-severity 2": [0.0] * 10 + [0.5],
    }
    by_age = {
        60: {"development both": 1.0},
        12: {"bf reported": 0.5, "frequency-severity 2": 0.5},
        36: {"bf reported": 0.5, "bf paid": 0.5},
        24: {"bf reported": 1.0},
    }
    for sel in [ex.select(by_origin), ex.select(by_age=by_age)]:
        assert isinstance(sel, SelectedUltimates)
        assert all(abs(a - b) <= 0.5 for a, b in zip(sel.ultimate, XYZ_SELECTED))
        assert abs(sel.total_ultimate - 483781) <= 1
        assert abs(sel.total_ibnr - 34155) <= 1
        assert sel.weights["frequency-severity 2"][-1] == 0.5
    table = select_ultimates(ex, by_origin)
    assert abs(table["ultimate"]["total"] - 483781) <= 1 if not isinstance(table, dict) else True


def test_weights_must_sum_to_one():
    ex = xyz()
    with pytest.raises(ValueError, match="sum to 0.75"):
        ex.select({"development both": 0.5, "bf reported": 0.25})
    with pytest.raises(ValueError, match="no ultimate there"):
        ex.select("frequency-severity 2")
    with pytest.raises(ValueError, match="not both"):
        ex.select("bf paid", by_age={12: {"bf paid": 1.0}})
    with pytest.raises(ValueError, match="no method"):
        ex.select("chain ladder")
    sel = ex.select({"bf paid": 1.0})
    assert sel.ultimate == [float(u) for u in XYZ["bf paid"]]


def voting_reference():
    path = ROOT / "validation" / "reference" / "reserving_exhibits_python.csv"
    rows = [r for r in path.read_text().splitlines() if not r.startswith("#")]
    return list(csv.DictReader(rows))


def test_voting_chainladder_parity():
    tri = read_triangle("genins_premium", ["paid", "premium"])
    fits = {
        "cl": ChainLadder().fit(tri, "paid"),
        "bf": BornhuetterFerguson(apriori=0.75).fit(tri, "paid", "premium"),
        "cc": CapeCod().fit(tri, "paid", "premium"),
    }
    ex = UltimateExhibit.from_triangle(tri, paid="paid")
    for name, fit in fits.items():
        ex = ex.with_method(name, fit)
    phased = [[1.0, 0.0, 0.0]] * 4 + [[0.5, 0.5, 0.0]] * 3 + [[0.0, 0.5, 0.5]] * 2 + [[0.0, 0.0, 1.0]]
    weights = {
        "phased": {m: [r[k] for r in phased] for k, m in enumerate(["cl", "bf", "cc"])},
        "blend": {"cl": 0.5, "bf": 0.25, "cc": 0.25},
        "single_bf": "bf",
    }
    selected = {case: ex.select(w) for case, w in weights.items()}
    n = 0
    for row in voting_reference():
        sel = selected[row["method"]]
        q = row["quantity"]
        expected = float(row["expected"])
        if q.startswith("ultimate:"):
            got = ex.ultimate(q.split(":", 1)[1])[ex.origins.index(row["arg"])]
        elif q == "ultimate":
            got = sel.ultimate[ex.origins.index(row["arg"])]
        elif q == "reserve":
            got = sel.unpaid[ex.origins.index(row["arg"])]
        elif q == "total_ultimate":
            got = sel.total_ultimate
        else:
            got = sel.total_unpaid
        assert got == pytest.approx(expected, rel=1e-9), (row["method"], q, row["arg"])
        n += 1
    assert n == 156
    table = select_ultimates(fits, weights["blend"], triangle=tri, paid="paid", total=False)
    got = list(table["ultimate"])
    assert got == pytest.approx(selected["blend"].ultimate)


def test_fit_with_other_origins_is_refused(us):
    ex = UltimateExhibit.from_triangle(us, paid="paid")
    other = read_triangle("genins_premium", ["paid"])
    with pytest.raises(ValueError, match="origins"):
        ex.with_method("cl", ChainLadder().fit(other, "paid"))
    with pytest.raises(ValueError, match="triangle"):
        select_ultimates({"cl": ChainLadder().fit(us, "paid")}, "cl")


def test_fit_without_ultimate_is_refused(us):
    # The bootstrap and Bayesian fits have origins but no point ultimate:
    # their mean ultimates go in as numbers.
    ex = UltimateExhibit.from_triangle(us, paid="paid")

    class NoUltimate:
        origins = ex.origins

    with pytest.raises(TypeError, match="no ultimate per origin"):
        ex.with_method("bayes", NoUltimate())
    means = [1.0] * len(ex.origins)
    assert ex.with_method("bayes", means).ultimate("bayes") == means
