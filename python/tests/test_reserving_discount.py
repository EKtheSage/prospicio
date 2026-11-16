"""Discounted reserves and the IFRS 17 risk adjustment (#553, PR 2;
docs/design/reserving-ifrs17.md): the Python bindings of
prospicio_reserving::discount, checked by hand and against prospicio's own
PaymentPattern (validation target 3)."""

import math

import pytest

from helpers import dataset
from prospicio.pricing import PaymentPattern
from prospicio.reserving import (
    ChainLadder,
    DiscountCurve,
    DiscountedReserves,
    Mack,
    MackBootstrap,
    NotSupportedError,
    OdpBootstrap,
    Triangle,
)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1.0)


@pytest.fixture(scope="module")
def raa():
    return dataset("raa")


@pytest.fixture(scope="module")
def boot(raa):
    return MackBootstrap(n_sims=500, seed=3).fit(raa, "values")


@pytest.fixture(scope="module")
def selected(raa):
    """Mack's bootstrap with RAA's own factors fully selected: every
    simulation pays along the chain ladder's pattern, with no parameter
    error, and the pattern cumulative at each age by hand."""
    cl = ChainLadder().fit(raa, "values")
    ages = raa.development
    ldf = dict(zip(ages[:-1], cl.ldf))
    fit = MackBootstrap(n_sims=300, seed=5, ldf=ldf, ldf_kind="ldf").fit(raa, "values")
    pattern = [1.0 / math.prod(cl.ldf[d:]) for d in range(len(ages))]
    n = len(ages)
    # RAA is square: origin w is at its age n - 1 - w.
    latest = [n - 1 - w for w in range(n)]
    return fit, ages, pattern, latest, cl


def test_a_zero_rate_leaves_every_draw_as_it_was(boot):
    for timing in ["mid", "end"]:
        d = boot.discounted(basis="paid", rate=0.0, timing=timing)
        assert isinstance(d, DiscountedReserves)
        assert d.timing == timing
        assert d.unpaid.draw_matrix() == boot.reserves.draw_matrix()
        for a, b in zip(d.discounted.draw_matrix(), boot.reserves.draw_matrix()):
            assert all(close(x, y) for x, y in zip(a, b))
        assert close(d.runoff[0], d.best_estimate)
        assert d.runoff[-1] == 0.0
        assert all(close(f, 1.0) for f in d.discount_factors)
    # A curve at 0 everywhere is the same.
    zero = boot.discounted(basis="paid", curve=DiscountCurve([1.0, 10.0], [0.0, 0.0]))
    assert close(zero.best_estimate, boot.reserves.mean())


def test_a_flat_rate_is_the_hand_built_cash_flow(selected):
    fit, ages, pattern, latest, _ = selected
    rate = 0.03
    d = fit.discounted(basis="paid", rate=rate)  # mid-period by default
    assert d.timing == "mid"
    unit = []
    for a in latest:
        rest = 1.0 - pattern[a]
        if rest <= 1e-12:
            unit.append((1 + rate) ** -0.5)
            continue
        unit.append(
            sum(
                (pattern[k] - pattern[k - 1]) / rest * (1 + rate) ** -((k - a) - 0.5)
                for k in range(a + 1, len(pattern))
            )
        )
    for pv, r in zip(d.discounted.draw_matrix(), fit.reserves.draw_matrix()):
        for w in range(len(unit)):
            assert close(pv[w], r[w] * unit[w]), (w, pv[w], r[w])
    for w, f in enumerate(d.discount_factors):
        assert f == 1.0 if latest[w] == len(ages) - 1 else close(f, unit[w])
    assert d.best_estimate < fit.reserves.mean()
    # The flat curve as a curve gives the same.
    same = fit.discounted(basis="paid", curve=DiscountCurve.flat(rate))
    assert close(same.best_estimate, d.best_estimate, 1e-12)


def test_each_origins_factor_is_the_payment_patterns(selected):
    """Validation target 3. PaymentPattern pays each share at its stated
    time and ``discounted`` with ``timing="end"`` pays each development
    period's increment at the period's end, so a pattern built from the
    factors to ultimate at the ages in years, valued from each origin's
    latest age, has the same discount factor as that origin's."""
    fit, ages, _, latest, cl = selected
    rate = 0.04
    d = fit.discounted(basis="paid", rate=rate, timing="end")
    years = [a / 12 for a in ages]
    cdf = list(cl.cdf)
    assert len(cdf) == len(years) and close(cdf[-1], 1.0)
    pattern = PaymentPattern.from_development(years, cdf)
    checked = 0
    for w, a in enumerate(latest):
        if a == len(ages) - 1:
            assert d.discount_factors[w] == 1.0  # nothing to pay
            continue
        want = pattern.unpaid_from(years[a]).discount_factor(rate)
        assert close(d.discount_factors[w], want, 1e-12), (w, d.discount_factors[w], want)
        checked += 1
    assert checked == len(ages) - 1
    # The discounted total is the undiscounted total weighted by the factors.
    for pv, r in zip(d.discounted.draw_matrix(), fit.reserves.draw_matrix()):
        assert close(sum(pv), sum(f * x for f, x in zip(d.discount_factors, r)))


def test_the_incurred_basis_adds_the_case_reserves(selected):
    fit, _, pattern, _, cl = selected
    latest_incurred = list(cl.latest)
    latest_paid = [0.8 * x for x in latest_incurred]
    case = [i - p for i, p in zip(latest_incurred, latest_paid)]
    d = fit.discounted(basis="incurred", rate=0.0, paid_pattern=pattern, latest_paid=latest_paid)
    for u, r in zip(d.unpaid.draw_matrix(), fit.reserves.draw_matrix()):
        assert all(close(a, b + c) for a, b, c in zip(u, r, case))
    assert close(d.best_estimate, fit.reserves.mean() + sum(case))
    # Every latest paid at the latest incurred: the IBNR alone, timed as the
    # paid basis times it.
    ibnr = fit.discounted(basis="incurred", rate=0.03, paid_pattern=pattern,
                          latest_paid=latest_incurred)
    paid = fit.discounted(basis="paid", rate=0.03)
    assert close(ibnr.best_estimate, paid.best_estimate, 1e-12)


def test_the_risk_adjustment_and_its_implied_confidence(boot):
    d = boot.discounted(basis="paid", rate=0.03)
    total = d.discounted.total()
    n = d.discounted.n_sims
    ra = d.risk_adjustment("confidence", 0.75)
    assert close(ra.amount, total.quantile(0.75) - d.best_estimate)
    assert ra.best_estimate == d.best_estimate
    assert 0.75 <= ra.implied_confidence < 0.75 + 1 / n
    assert ra.capital is None and ra.segments == []
    tv = d.risk_adjustment("tvar", 0.75)
    assert tv.amount > ra.amount
    assert tv.implied_confidence > 0.75


def test_the_cost_of_capital_under_both_charges(boot):
    rate, coc = 0.03, 0.06
    d = boot.discounted(basis="paid", rate=rate)
    run = d.runoff
    k0 = 1000.0
    by_hand = {
        "end": coc * sum(k0 * be / run[0] * (1 + rate) ** -(t + 1) for t, be in enumerate(run)),
        "start": coc * sum(k0 * be / run[0] * (1 + rate) ** -t for t, be in enumerate(run)),
    }
    for charge, want in by_hand.items():
        got = d.risk_adjustment("cost_of_capital", rate=coc, capital=k0, charge=charge)
        assert close(got.amount, want), charge
        assert got.capital == k0
        same = d.risk_adjustment("cost_of_capital", rate=coc, capital=("amount", k0), charge=charge)
        assert same.amount == got.amount
        assert close(got.implied_confidence,
                     sum(x <= d.best_estimate + got.amount for x in d.discounted.total().draws)
                     / d.discounted.n_sims)
    assert by_hand["start"] > by_hand["end"]
    # The end of year is the default; a VaR capital is Q_q less the best estimate.
    var = d.risk_adjustment("cost_of_capital", rate=coc, capital=("var", 0.995))
    assert close(var.capital, d.discounted.total().quantile(0.995) - d.best_estimate)
    end = d.risk_adjustment("cost_of_capital", rate=coc, capital=var.capital)
    assert close(var.amount, end.amount)


def test_segments_are_reported_stand_alone(raa):
    rows = [(o, a, v) for o, a, v in zip(*raa_long())]
    tri = Triangle.from_long(
        [o for o, _, _ in rows] * 2,
        [a for _, a, _ in rows] * 2,
        {"paid": [v for _, _, v in rows] + [0.5 * v for _, _, v in rows]},
        keys={"lob": ["A"] * len(rows) + ["B"] * len(rows)},
    )
    fit = OdpBootstrap(n_sims=400, seed=2).fit(tri, "paid")
    d = fit.discounted(basis="paid", rate=0.03)
    assert d.segment_labels == [["A"], ["B"]]
    assert len(d.segment_runoff(1)) > 1
    ra = d.risk_adjustment("confidence", 0.9)
    assert [s.label for s in ra.segments] == [["A"], ["B"]]
    assert ra.segments[0].result.amount > ra.segments[1].result.amount > 0
    # A capital amount is the total's: no segment results.
    assert d.risk_adjustment("cost_of_capital", rate=0.06, capital=500.0).segments == []
    with pytest.raises(ValueError, match="no segment"):
        d.segment_runoff(2)


def raa_long():
    from helpers import read_csv, VALIDATION

    rows = read_csv(VALIDATION / "data" / "raa.csv")
    return (
        [int(r["origin"]) for r in rows],
        [int(r["development"]) for r in rows],
        [float(r["value"]) for r in rows],
    )


def test_the_errors(raa, boot, selected):
    with pytest.raises(TypeError):
        boot.discounted(rate=0.03)  # the basis has no default
    with pytest.raises(ValueError, match="exactly one of rate"):
        boot.discounted(basis="paid")
    with pytest.raises(ValueError, match="exactly one of rate"):
        boot.discounted(basis="paid", rate=0.03, curve=DiscountCurve.flat(0.03))
    with pytest.raises(ValueError, match="timing"):
        boot.discounted(basis="paid", rate=0.03, timing="start")
    with pytest.raises(ValueError, match="basis"):
        boot.discounted(basis="reported", rate=0.03)
    with pytest.raises(ValueError, match="needs paid_pattern"):
        boot.discounted(basis="incurred", rate=0.03)
    with pytest.raises(ValueError, match="for basis=\"incurred\""):
        boot.discounted(basis="paid", rate=0.03, latest_paid=[0.0] * 10)
    _, _, pattern, _, cl = selected
    with pytest.raises(ValueError):  # one latest paid too few
        boot.discounted(basis="incurred", rate=0.03, paid_pattern=pattern,
                        latest_paid=list(cl.latest)[1:])
    with pytest.raises(ValueError):  # a paid pattern one age short
        boot.discounted(basis="incurred", rate=0.03, paid_pattern=pattern[1:],
                        latest_paid=list(cl.latest))
    with pytest.raises(ValueError):
        DiscountCurve([2.0, 1.0], [0.03, 0.04])
    with pytest.raises(NotSupportedError, match="simulations"):
        Mack().fit(raa, "values").discounted(basis="paid", rate=0.03)
    d = boot.discounted(basis="paid", rate=0.03)
    with pytest.raises(ValueError, match="needs p"):
        d.risk_adjustment("confidence")
    with pytest.raises(ValueError, match="needs capital"):
        d.risk_adjustment("cost_of_capital", rate=0.06)
    with pytest.raises(ValueError, match="method"):
        d.risk_adjustment("expected_shortfall", 0.9)
    with pytest.raises(TypeError, match="capital"):
        d.risk_adjustment("cost_of_capital", rate=0.06, capital="lots")


def test_the_curve():
    c = DiscountCurve([1.0, 5.0], [0.02, 0.04])
    assert c.maturities == [1.0, 5.0] and c.rates == [0.02, 0.04]
    assert close(c.rate(3.0), 0.03) and c.rate(0.5) == 0.02 and c.rate(9.0) == 0.04
    assert close(c.factor(2.0), 1.025 ** -2.0)
    assert c == DiscountCurve([1.0, 5.0], [0.02, 0.04])
    assert DiscountCurve.flat(0.03) == DiscountCurve([1.0], [0.03])
