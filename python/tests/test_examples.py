"""The worked notebooks in examples/ run, with fewer simulations.

Each notebook's code cells run in order in one namespace, without Jupyter;
cells tagged ``plot`` are skipped, so CI needs no plotting library.
"""

import csv
import json
import os
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def run_notebook(path):
    cells = json.loads(path.read_text())["cells"]
    ns = {}
    for cell in cells:
        if cell["cell_type"] != "code" or "plot" in cell["metadata"].get("tags", []):
            continue
        exec(compile("".join(cell["source"]), f"{path.name}", "exec"), ns)
    return ns


def test_one_year_view(monkeypatch, capsys):
    monkeypatch.setitem(os.environ, "PROSPICIO_EXAMPLE_SIMS", "4000")
    monkeypatch.chdir(EXAMPLES)
    ns = run_notebook(EXAMPLES / "one_year_view.ipynb")
    assert "reinsurance cuts the company's TVaR" in capsys.readouterr().out
    # Simulation agrees with exposure rating; the allocation adds up.
    surplus = ns["surplus_loss"].mean()
    assert surplus == pytest.approx(ns["profile"].expected_surplus_loss(2e6, 4.0), rel=0.05)
    for a in ns["allocations"].values():
        assert sum(a.allocated) == pytest.approx(a.total, rel=1e-9)
    assert ns["saved"] > 0


def test_reinsurance_pricing(monkeypatch, capsys):
    monkeypatch.chdir(EXAMPLES)
    ns = run_notebook(EXAMPLES / "reinsurance_pricing.ipynb")
    assert "comparisons are inside their tolerance" in capsys.readouterr().out
    # Every comparison with aggregate's numbers is inside its stated tolerance.
    comparisons = ns["COMPARISONS"]
    assert len(comparisons) > 300
    assert [c["name"] for c in comparisons if not c["ok"]] == []
    # A few values, read from the reference file here and not through the notebook's helpers.
    root = EXAMPLES.parent / "validation" / "reference" / "re_pricing_aggregate.csv"
    with open(root) as f:
        ref = {(r["example"], r["quantity"]): float(r["value"]) for r in csv.DictReader(f)}
    assert ns["gross"].mean() == pytest.approx(ref["tower_gross", "gross_mean"], rel=1e-9)
    assert ns["gross"].quantile(0.99) == ref["tower_gross", "gross_q0.99"]
    assert ns["net"].mean() == pytest.approx(ref["tower_program", "net_mean"], rel=1e-9)
    assert ns["limited"].mean() == pytest.approx(ref["agg_limit_reinst", "agglimit2800_ceded_mean"], rel=1e-7)
    assert ns["after_aad"].mean() == pytest.approx(ref["stop_loss_aad", "after_aad_mean"], rel=1e-9)
    assert ns["retro_grids"].expected_premium[0] == pytest.approx(
        ref["variable_retro", "retro_premium_mean"], rel=1e-9)
    assert ns["commission"] == pytest.approx(ref["variable_pc", "pc_amount_mean"], rel=1e-9)
    assert ns["slide_commission"] == pytest.approx(ref["variable_slide", "slide_commission_amount_mean"], rel=1e-9)
    assert ns["priced"]["gross", "ph"].measure(ns["gross"]) == pytest.approx(
        ref["risk_pricing_gross", "unlimited_price_ask_ph"], rel=1e-6)
    # The exact closed forms the grid means approach.
    assert ns["severity"].lev(5000.0) * 10 == pytest.approx(ref["tower_gross", "gross_mean_exact"], rel=1e-12)


def test_notebooks_are_saved_with_outputs():
    for path in EXAMPLES.glob("*.ipynb"):
        cells = json.loads(path.read_text())["cells"]
        shows = [c for c in cells if c["cell_type"] == "code"
                 and ("print(" in "".join(c["source"]) or "plt.show" in "".join(c["source"]))]
        assert shows and all(c["outputs"] for c in shows), f"{path.name} is saved without results"
