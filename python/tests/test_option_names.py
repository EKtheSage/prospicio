"""One vocabulary for option names across Python, R and Excel (#834, step 1).

Each spelling an option accepts in any of the three front ends is accepted
here too and gives the same result as the canonical name.
"""

import math

import pytest

from prospicio.distributions import Count, PredictiveDistribution, Sampled
from prospicio.models import Design, ZeroInflated
from prospicio.reserving import (
    BenchmarkCurve,
    ChainLadder,
    LatestDiagonal,
    Meyers,
    Mack,
    MunichChainLadder,
    PaidIncurredChain,
    Triangle,
)
from prospicio.risk import Distortion, calibrate, capital

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


def test_family_and_link_names():
    # The GLM family and link names come from one table (#834, step 2).
    from prospicio.models import Glm

    with pytest.raises(ValueError, match='family must be "gaussian", "normal", "poisson", .* or "tweedie", got "weibull"'):
        Glm("weibull")
    with pytest.raises(ValueError, match='link must be "identity", "log", .* or "power", got "sqrt"'):
        Glm("poisson", link="sqrt")
    x = [float(i % 5) for i in range(30)]
    design = Design([[1.0] * 30, x], ["(Intercept)", "x"])
    y = [1.0 + 0.3 * v + (i % 3) * 0.1 for i, v in enumerate(x)]
    a = Glm("gaussian").fit(design, y).coefficients
    assert Glm("normal").fit(design, y).coefficients == a


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


@pytest.mark.parametrize("name", ["euler", "natural"])
def test_allocation_method_spellings(name):
    # Excel's `natural` is the Euler allocation (AllocationMethod::NAMES, #956).
    pd = PredictiveDistribution(
        ["lob"],
        [("motor",), ("property",)],
        [[1.0, 2.0], [4.0, 1.0], [2.0, 5.0], [3.0, 6.0]],
    )
    want = capital(pd, Distortion.tvar(0.5), "euler").allocated
    got = capital(pd, Distortion.tvar(0.5), name)
    assert got.allocated == want
    assert got.method == "euler"
    with pytest.raises(ValueError, match="method must be"):
        capital(pd, Distortion.tvar(0.5), "nope")


def test_cashflow_option_spellings():
    # Excel reads a cash-flow option in any case, with `_`, `-` or a space between
    # the words; Python and R do too (prospicio_pricing::cashflow::option_key, #956).
    from prospicio.pricing import CashflowTerms, CollateralRequirement, PremiumAdjustment

    def ramp(m):
        return [min((t + 1) / m, 1.0) for t in range(36)]

    def flows(base, adjustment, rule, reference, source, interest, fee, release):
        terms = CashflowTerms(
            100.0,
            30,
            ramp(12),
            ramp(24),
            ramp(36),
            CollateralRequirement({base: 1.0, "reserve": 0.2}),
            adjustments=[PremiumAdjustment(18, 0.1, base=adjustment)],
            funding_rule=rule,
            reference=reference,
            funding_source=source,
            annual_yield=0.04,
            interest_basis=interest,
            fee=([0.0, 100.0], [0.1, 0.2]),
            fee_basis=fee,
            equity_release=release,
        )
        return terms.run([0.6, 1.5]).flows

    want = flows(
        "unearned-premium",
        "to-date",
        "obligation-increase",
        "running-max",
        "operating-first",
        "opening",
        "profit-over-premium",
        "at-commutation",
    )
    got = flows(
        "Unearned_Premium",
        "TO DATE",
        "Obligation Increase",
        "running_max",
        " Operating-First ",
        "OPENING",
        "profit_over_premium",
        "At Commutation",
    )
    assert got == want
    assert CollateralRequirement({"Unearned Premium": 0.3}).terms == [("unearned-premium", 0.3)]
    with pytest.raises(ValueError, match="base must be"):
        CollateralRequirement({"nope": 1.0})
    with pytest.raises(ValueError, match="fee basis must be"):
        flows("fixed", "ultimate", "shortfall", "previous-date", "investor", "average", "nope", "monthly")

@pytest.mark.parametrize(
    "name, canonical",
    [("lp", "lp"), ("last_piece", "lp"), ("wd", "wd"), ("whole_distribution", "wd")],
)
def test_truncation_type_spellings(name, canonical):
    # Excel's `last_piece` and `whole_distribution` (Truncation::NAMES, #956).
    from prospicio.distributions import PiecewisePareto

    want = PiecewisePareto([1000.0, 2000.0], [1.0, 2.0], truncation=5000.0, truncation_type=canonical)
    got = PiecewisePareto([1000.0, 2000.0], [1.0, 2.0], truncation=5000.0, truncation_type=name)
    assert got.truncation_type == canonical
    assert got.survival(1500.0) == want.survival(1500.0)
    with pytest.raises(ValueError, match="truncation_type must be"):
        PiecewisePareto([1.0], [1.0], truncation=2.0, truncation_type="nope")

@pytest.mark.parametrize(
    "name, canonical",
    [
        ("top-down", "peel-top-down"),
        ("Top_Down", "peel-top-down"),
        ("peel top down", "peel-top-down"),
        ("bottom-up", "peel-bottom-up"),
        ("peel_bottom_up", "peel-bottom-up"),
    ],
)
def test_book_face_spellings(name, canonical):
    # Excel's short names for the peel faces (Face::NAMES, #956).
    from prospicio.aggregate import Book, Expense
    from prospicio.reinsurance import Layer, Tower

    gross = ([2, 3, 4, 5, 6, 8], [0.25, 0.3, 0.09, 0.2, 0.12, 0.04])
    high = Layer.stop_loss("High", 3.0, 5.0).with_deposit_premium(1.0).with_ceding_commission(0.1)
    low = Layer.stop_loss("Low", 2.0, 3.0).with_deposit_premium(0.5)
    book = Book("Ex6", 10.0, [Expense("premium", 0.1)])
    want = book.ledger(gross, Tower([low, high]), canonical)
    got = book.ledger(gross, Tower([low, high]), name)
    assert got.result.mean() == want.result.mean()
    assert got.sheet()["step"] == want.sheet()["step"]
    with pytest.raises(ValueError, match="face must be"):
        book.ledger(gross, None, "sideways")

@pytest.mark.parametrize("name", ["in_force", "inforce", "In-Force", "in force"])
def test_rate_change_applies_spellings(name):
    # Excel's `inforce`, and its case and separator rules (Applies::NAMES, #956).
    from prospicio.pricing import Period, RateHistory

    effective = [2010.5, 2011.0, 2011.5, 2012.25]
    changes = [0.05, 0.10, -0.05, -0.01]
    period = Period.calendar(2011.0, 2012.0)
    want = RateHistory(effective, changes, applies=["written", "written", "in_force", "written"])
    got = RateHistory(effective, changes, applies=["written", "written", name, "written"])
    assert (
        got.rate_level(period, 1.0).on_level_factor == want.rate_level(period, 1.0).on_level_factor
    )
    assert got.rate_level(period, 1.0).in_force_changes == [0, 0, 1, 0, 1]
    with pytest.raises(ValueError, match="applies must be"):
        RateHistory([2011.0], [0.1], applies=["sometimes"])

@pytest.mark.parametrize("name", ["loglogistic", "log-logistic", "log_logistic"])
def test_growth_curve_spellings(name):
    # Excel's spellings of the loglogistic growth curve (#956).
    bench = {12: 0.18, 24: 0.47, 36: 0.66, 48: 0.78, 60: 0.855, 72: 0.90}
    want = BenchmarkCurve().fit(bench, "percent_developed").omega
    assert BenchmarkCurve(curve=name).fit(bench, "percent_developed").omega == want
    with pytest.raises(ValueError, match="growth curve must be"):
        BenchmarkCurve(curve="gompertz")


@pytest.mark.parametrize("name", ["csr", "CSR", "Csr"])
def test_meyers_model_in_any_case(name):
    # Python took the model in any case; R now does too (#956).
    assert Meyers(name).model == "csr"
    with pytest.raises(ValueError, match="model must be"):
        Meyers("ipi")


@pytest.mark.parametrize("name", ["no_development", "none"])
def test_past_pattern_end_spellings(name):
    # Excel's "none" for "no_development" (#956).
    d = LatestDiagonal([2023, 2024], [24, 12], [1.0, 2.0], past_pattern_end=name)
    assert d.past_pattern_end == "no_development"
    with pytest.raises(ValueError, match="past_pattern_end must be"):
        LatestDiagonal([2023, 2024], [24, 12], [1.0, 2.0], past_pattern_end="hold")


@pytest.mark.parametrize("name", ["means_linear", "linear"])
def test_elt_bucketing_spellings(name):
    # The means-alone rule in Python's and Excel's spelling (#956).
    from prospicio.aggregate import EventLossTable

    elt = EventLossTable(
        [0.5, 0.25, 0.05], [130.0, 270.5, 1234.5], [0.0] * 3, [0.0] * 3, [260.0, 541.0, 2469.0]
    )
    want, _ = elt.severity_grid(10.0, 256, "means_linear")
    got, report = elt.severity_grid(10.0, 256, name)
    assert list(got.probs) == list(want.probs)
    assert report["bucketing"] == "means_linear"
    with pytest.raises(ValueError, match="bucketing must be"):
        elt.severity_grid(10.0, 256, "nope")


@pytest.mark.parametrize("name", ["r_chainladder", "r-chainladder", "chainladder", "r"])
def test_paid_incurred_convention_spellings(name):
    # Excel's spellings of R ChainLadder's convention, and Python's own (#956).
    tri = triangle()
    want = PaidIncurredChain(convention="r_chainladder").fit(tri, "paid", "incurred")
    got = PaidIncurredChain(convention=name).fit(tri, "paid", "incurred")
    assert got.total_reserve == want.total_reserve
    assert PaidIncurredChain(convention=name).convention == "r_chainladder"
    with pytest.raises(ValueError, match="convention must be"):
        PaidIncurredChain(convention="nope")


def test_cover_basis_aliases():
    from prospicio.aggregate import EventSet, Exposure, Timeline

    events = EventSet.from_years([[100.0]], times=[[13.5 / 24]])
    timeline = Timeline(events, 24)
    shares = [[0.0] * 24, [0.0] * 24]
    shares[0][13], shares[1][13] = 0.25, 0.75
    for canonical, alias in [("risks_attaching", "rad"), ("losses_occurring", "lod")]:
        a = timeline.years(canonical, Exposure(shares))
        b = timeline.years(alias, Exposure(shares))
        assert [y.events(0) for y in a] == [y.events(0) for y in b]
    with pytest.raises(ValueError, match='cover basis must be "losses_occurring", "lod", "risks_attaching" or "rad", got "claims_made"'):
        timeline.years("claims_made", Exposure(shares))
