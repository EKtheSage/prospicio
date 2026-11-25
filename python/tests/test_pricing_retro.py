"""Retrospective rating: the insurance charge and savings (Table M, Table L)
and the basic premium that balances a plan (#981)."""

import math

import pytest

from prospicio.aggregate import fft
from prospicio.distributions import Gamma, Grid, Poisson
from prospicio.pricing import InsuranceCharge, RetroPlan

YEARS = [0.0, 10.0, 30.0, 50.0, 70.0, 90.0, 120.0, 150.0, 200.0, 280.0]


def test_a_sample_and_the_grid_of_its_values_agree():
    sample = InsuranceCharge.from_sample([250.0, 0.0, 100.0, 50.0])
    grid = InsuranceCharge.from_grid(Grid(50.0, [0.25, 0.25, 0.25, 0.0, 0.0, 0.25]))
    assert sample.expected == 100.0 and sample.loss_elimination == 0.0
    assert sample.charge(1.0) == pytest.approx(0.375, abs=1e-15)
    assert sample.savings(0.5) == pytest.approx(0.125, abs=1e-15)
    for r in [0.0, 0.1, 0.5, 1.0, 1.7, 2.5, 3.0]:
        assert grid.charge(r) == pytest.approx(sample.charge(r), abs=1e-15)
        assert grid.savings(r) == pytest.approx(sample.savings(r), abs=1e-15)
    table = sample.table([0.0, 0.5, 1.0])
    assert table == {
        "entry_ratio": [0.0, 0.5, 1.0],
        "charge": [1.0, 0.625, 0.375],
        "savings": [0.0, 0.125, 0.375],
    }
    assert "expected=100" in repr(sample)


def test_an_fft_aggregate_keeps_the_identity():
    severity, _ = Grid.local_moment(Gamma(1.0, 10.0), 0.25, 1024)
    aggregate, report = fft(Poisson(4.0), severity, 4096)
    assert report.aliasing_error < 1e-12
    table = InsuranceCharge.from_grid(aggregate)
    last = 1.0
    for k in range(1, 17):
        r = 0.25 * k
        assert table.charge(r) - table.savings(r) == pytest.approx(1.0 - r, abs=1e-12)
        assert table.charge(r) < last
        last = table.charge(r)


def test_table_l_adds_the_cost_of_the_limit():
    limited = InsuranceCharge.from_sample([110.0, 100.0, 0.0, 200.0])
    table = limited.table_l(132.5)
    k = 30.0 / 132.5
    assert table.expected == 132.5
    assert table.loss_elimination == pytest.approx(k, abs=1e-15)
    assert table.charge(0.0) == 1.0
    assert table.charge(1.0) == pytest.approx((67.5 / 4 + 30.0) / 132.5, abs=1e-15)
    assert table.charge(10.0) == pytest.approx(k, abs=1e-15)
    assert table.charge(1.0) - table.savings(1.0) == pytest.approx(0.0, abs=1e-15)
    # The limited table itself is unchanged.
    assert limited.expected == 102.5 and limited.loss_elimination == 0.0


@pytest.mark.parametrize(
    "minimum, maximum", [(80.0, 220.0), (0.0, 180.0), (60.0, math.inf), (60.0, 400.0)]
)
def test_a_balanced_plan_collects_the_guaranteed_cost_premium(minimum, maximum):
    plan = RetroPlan(100.0, 30.0, loss_conversion=1.1, tax_multiplier=1.03,
                     minimum=minimum, maximum=maximum)
    b = plan.balance(InsuranceCharge.from_sample(YEARS))
    terms = plan.loss_sensitive_premium(b["basic"])
    mean = sum(terms.premium(YEARS)) / len(YEARS)
    assert plan.guaranteed_cost() == pytest.approx(1.03 * 130.0, abs=1e-12)
    assert mean == pytest.approx(plan.guaranteed_cost(), abs=1e-9)
    assert b["expected_premium"] == pytest.approx(plan.guaranteed_cost(), abs=1e-12)
    textbook = 30.0 - 0.1 * 100.0 + 1.1 * 100.0 * b["net_insurance_charge"]
    assert b["basic"] == pytest.approx(textbook, abs=1e-9)
    if math.isinf(maximum):
        assert math.isinf(b["entry_ratio_maximum"])


def test_a_plan_with_a_loss_limit_balances_against_table_l():
    limited = [min(x, 150.0) for x in YEARS]
    plan = RetroPlan(100.0, 30.0, 1.1, 1.03, 80.0, 200.0)
    b = plan.balance(InsuranceCharge.from_sample(limited).table_l(100.0))
    terms = plan.loss_sensitive_premium(b["basic"])
    mean = sum(terms.premium(limited)) / len(limited)
    assert mean == pytest.approx(plan.guaranteed_cost(), abs=1e-9)
    with pytest.raises(ValueError, match="Table L"):
        plan.balance(InsuranceCharge.from_sample(limited))


def test_defaults_and_refusals():
    # The defaults: no conversion, no tax, no minimum, no maximum.
    plan = RetroPlan(100.0, 30.0)
    b = plan.balance(InsuranceCharge.from_sample(YEARS))
    assert b["basic"] == pytest.approx(30.0, abs=1e-9) and b["entry_ratio_minimum"] < 0
    assert plan.loss_sensitive_premium(30.0).premium([50.0]) == [80.0]
    with pytest.raises(ValueError):
        plan.loss_sensitive_premium(-1.0)
    # A minimum that would need a negative basic premium is refused.
    with pytest.raises(ValueError, match="negative basic premium"):
        RetroPlan(100.0, 30.0, 1.1, 1.03, minimum=120.0).balance(InsuranceCharge.from_sample(YEARS))
    with pytest.raises(ValueError):
        InsuranceCharge.from_sample([])
    with pytest.raises(ValueError):
        InsuranceCharge.from_sample([1.0, -2.0])
    with pytest.raises(ValueError):
        InsuranceCharge.from_sample([10.0, 30.0]).table_l(19.0)
    with pytest.raises(ValueError):
        RetroPlan(100.0, 30.0, minimum=100.0, maximum=100.0)
    table = InsuranceCharge.from_sample(YEARS)
    with pytest.raises(ValueError, match="minimum"):
        RetroPlan(100.0, 30.0, 1.1, 1.03, 134.0, 300.0).balance(table)
    with pytest.raises(ValueError, match="maximum"):
        RetroPlan(100.0, 30.0, 1.1, 1.03, 50.0, 133.0).balance(table)
