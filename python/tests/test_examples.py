"""The worked examples in examples/ run, with fewer simulations."""

import os
import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_one_year_view(monkeypatch, capsys):
    monkeypatch.setitem(os.environ, "ACTUARIALRS_EXAMPLE_SIMS", "4000")
    ns = runpy.run_path(str(EXAMPLES / "one_year_view.py"))
    out = capsys.readouterr().out
    assert "4. TVaR 99%" in out
    # Simulation agrees with exposure rating; the allocation adds up.
    result, profile = ns["result"], ns["profile"]
    surplus = result.marginal(("ceded", "surplus")).mean()
    assert surplus == pytest.approx(profile.expected_surplus_loss(2e6, 4.0), rel=0.05)
    for a in ns["allocations"].values():
        assert sum(a.allocated) == pytest.approx(a.total, rel=1e-9)
    assert ns["saved"] > 0
