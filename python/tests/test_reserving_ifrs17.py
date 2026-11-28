"""The IFRS 17 groups in Python: the general model, the PAA and reinsurance held.

The Rust tests in `validation/tests/reserving_ifrs17*.rs` check the
measurement against the standard's examples; these check the bindings: the
dictionaries in and out, the saved state and the errors.
"""

import json
import pickle

import pytest

from prospicio.reserving import DiscountCurve, Ifrs17Group, PaaGroup, ReinsuranceHeld


def example_1a():
    return Ifrs17Group(
        {"premiums": [900.0, 0.0, 0.0], "claims": [200.0] * 3}, 120.0, DiscountCurve.flat(0.05)
    )


def test_example_1a_recognition_and_first_year():
    g = example_1a()
    assert g.period == 0
    assert g.balances["csm"] == pytest.approx(235.35, abs=0.01)
    year = g.roll_forward(g.as_expected(), {"claims": [200.0] * 2}, 80.0)
    assert year["period"] == 1
    assert g.period == 1
    closing = year["closing"]
    assert closing["csm"] == pytest.approx(164.75, abs=0.01)
    assert closing["liability"] == pytest.approx(616.63, abs=0.01)
    assert year["profit"] == pytest.approx(
        year["insurance_service_result"] - year["finance_expenses_in_profit"]
    )
    assert "csm_recognised" in year["by_component"]["line"]


def test_the_saved_state_rolls_on_identically():
    g = example_1a()
    g.roll_forward(g.as_expected(), {"claims": [200.0] * 2}, 80.0)
    text = g.to_json()
    assert json.loads(text)["format"] == "risk_rs.ifrs17_group"
    for copy in (Ifrs17Group.from_json(text), pickle.loads(pickle.dumps(g))):
        assert copy.balances == g.balances
        assert copy.period == g.period
        a = copy.roll_forward(copy.as_expected(), {"claims": [200.0]}, 40.0)
        b = Ifrs17Group.from_json(text).roll_forward(g.as_expected(), {"claims": [200.0]}, 40.0)
        assert a == b


def test_unknown_keys_and_timings_are_refused():
    with pytest.raises(ValueError, match="unknown key"):
        Ifrs17Group({"premium": [1.0]}, 1.0, DiscountCurve.flat(0.0))
    with pytest.raises(ValueError, match="timing"):
        Ifrs17Group(
            {"premiums": [1.0], "premium_timing": "middle"}, 1.0, DiscountCurve.flat(0.0)
        )


def test_paa_year():
    g = PaaGroup(1_000.0, 100.0, [1.0, 1.0])
    year = g.roll_forward(1_000.0, 100.0, 300.0, 200.0, 100.0, 10.0)
    assert year["insurance_revenue"] == pytest.approx(500.0)
    assert year["insurance_service_expenses"] == pytest.approx(360.0)
    copy = PaaGroup.from_json(g.to_json())
    assert copy.balances == g.balances
    assert pickle.loads(pickle.dumps(g)).balances == g.balances


def test_reinsurance_held_follows_its_underlying_group():
    g = example_1a()
    held = ReinsuranceHeld(
        {"premiums": [30.0, 0.0, 0.0], "recoveries": [20.0] * 3},
        5.0,
        DiscountCurve.flat(0.0),
        underlying=[(g, 0.1)],
    )
    assert held.balances["asset"] == pytest.approx(0.0, abs=1e-9)
    g.roll_forward(g.as_expected(), {"claims": [200.0] * 2}, 80.0)
    year = held.roll_forward(held.as_expected(), {"recoveries": [20.0] * 2}, 3.0, underlying=[g])
    assert held.period == 1
    assert year["period"] == 1
    copy = ReinsuranceHeld.from_json(held.to_json())
    assert copy.balances == held.balances
    assert pickle.loads(pickle.dumps(held)).balances == held.balances


def test_reinsurance_held_refuses_a_group_from_another_period():
    g = example_1a()
    held = ReinsuranceHeld(
        {"premiums": [30.0, 0.0, 0.0], "recoveries": [20.0] * 3},
        5.0,
        DiscountCurve.flat(0.0),
        underlying=[(g, 0.1)],
    )
    with pytest.raises(ValueError):
        held.roll_forward(held.as_expected(), {"recoveries": [20.0] * 2}, 3.0, underlying=[g])
