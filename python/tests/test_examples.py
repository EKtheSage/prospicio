"""The worked notebooks in examples/ run, with fewer simulations.

Each notebook's code cells run in order in one namespace, without Jupyter;
cells tagged ``plot`` are skipped, so CI needs no plotting library.
"""

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


def test_notebooks_are_saved_with_outputs():
    for path in EXAMPLES.glob("*.ipynb"):
        cells = json.loads(path.read_text())["cells"]
        shows = [c for c in cells if c["cell_type"] == "code"
                 and ("print(" in "".join(c["source"]) or "plt.show" in "".join(c["source"]))]
        assert shows and all(c["outputs"] for c in shows), f"{path.name} is saved without results"
