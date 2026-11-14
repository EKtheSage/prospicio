"""The running premium-weighted loss ratio against the four values the AP
pricing demo prints (#806), on its inputs in
validation/data/ap_pricing_demo_inputs.json, the rows the Rust test reads."""

import json

import pytest
from helpers import VALIDATION
from prospicio.pricing import cumulative_loss_ratio

PRINTED = {1: 0.610000000000, 2: 0.594705882353, 3: 0.638120662040, 14: 0.637034303676}


def test_matches_the_demos_printed_values():
    inputs = json.loads((VALIDATION / "data" / "ap_pricing_demo_inputs.json").read_text())
    lr = cumulative_loss_ratio(inputs["loss_ratios"], inputs["earned_premium"])
    assert len(lr) == 14
    for row, want in PRINTED.items():
        assert lr[row - 1] == pytest.approx(want, abs=5e-13)
