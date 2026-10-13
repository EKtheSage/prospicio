"""The worked notebooks in examples/ run, with fewer simulations.

Each notebook's code cells run in order in one namespace, without Jupyter;
cells tagged ``plot`` are skipped, so CI needs no plotting library.
"""

import csv
import json
import math
import os
import subprocess
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def run_notebook(path):
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
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


def test_submission_evaluation(monkeypatch, capsys):
    monkeypatch.setitem(os.environ, "PROSPICIO_EXAMPLE_SIMS", "4000")
    monkeypatch.chdir(EXAMPLES)
    ns = run_notebook(EXAMPLES / "submission_evaluation.ipynb")
    capsys.readouterr()
    with open(EXAMPLES / "data" / "answer_key.json") as f:
        key = json.load(f)

    # The files reconcile, and what is read from them agrees with how they were generated.
    assert [name for name, ok in ns["checks"] if not ok] == []
    assert ns["onlevel"][2025] == pytest.approx(key["onlevel_premium"]["2025"], rel=0.005)
    assert ns["exposure_trend"].annual_change == pytest.approx(key["exposure_growth"], abs=0.002)
    assert ns["SEVERITY_TREND"] == pytest.approx(key["severity_trend"], abs=0.005)

    # The Pareto fit allows for truncation and censoring: it lands within two standard errors of the truth,
    # and the fit that ignores the per-claim reporting thresholds is far from it.
    from prospicio.distributions import Pareto

    se = ns["ALPHA"] / math.sqrt(ns["n_uncensored"])
    assert abs(ns["ALPHA"] - key["alpha"]) < 2 * se
    ignoring = Pareto.fit(ns["losses"], ns["T"], censored=ns["censored"]).alpha
    assert ns["ALPHA"] - ignoring > 0.2

    # Layer losses: the blend lies between its two inputs, with credibility in (0, 1), and the grid reproduces it.
    for z, e, x, b, grid in zip(ns["weights"], ns["experience_loss"], ns["exposure_loss"], ns["blended"],
                                ns["grids"].ceded):
        assert 0.0 < z < 1.0
        assert min(e, x) <= b <= max(e, x)
        assert grid.mean() == pytest.approx(b, rel=5e-3)
    # The simulated program agrees with the grid for the working layer.
    assert ns["simulated"].marginal(["ceded", ns["NAMES"][0]]).mean() == pytest.approx(
        ns["grids"].ceded[0].mean(), rel=0.15)

    # Technical premiums: loss plus a capital charge, so above the expected loss; the one-function chain
    # reproduces the step-by-step result.
    for row, base_row in zip(ns["priced"], ns["base"]["rows"]):
        assert row["technical"] > row["loss"]
        assert base_row["technical"] == pytest.approx(row["technical"], rel=1e-7)
    # Prices fall with the attachment as a rate on line. The case is built so that the decision differs by
    # layer: quote the working layer, counter the middle one, refer the top one to the portfolio view.
    rols = [row["rol"] for row in ns["priced"]]
    assert rols == sorted(rols, reverse=True)
    assert [verdict.split()[0] for verdict in ns["closing"]] == ["quote", "counter", "refer"]

    # Against the generator's truth: the blended layer losses, and the technical premium they lead to.
    for name, blended in zip(ns["NAMES"], ns["blended"]):
        truth = key["layer_expected_loss_2026_occurrence"][name]
        assert blended == pytest.approx(truth, rel=0.10 if name == "5m xs 5m" else 0.25), name
    for row, true_row in zip(ns["priced"], ns["true_rows"]):
        assert row["technical"] == pytest.approx(true_row["technical"], rel=0.10), row["name"]


def test_submission_files_match_generator(tmp_path):
    # The committed files are what examples/data/make_submission.py writes: the CSV files byte for byte (a
    # checkout under core.autocrlf may turn their LF into CRLF), the answer key to rounding, since its
    # full-precision floats come through libm and could differ in the last digit between platforms.
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_submission", EXAMPLES / "data" / "make_submission.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generator.OUT = tmp_path / "submission"
    generator.KEY = tmp_path / "answer_key.json"
    generator.main()
    committed = EXAMPLES / "data" / "submission"
    names = sorted(p.name for p in generator.OUT.iterdir())
    assert names == sorted(p.name for p in committed.iterdir())
    for name in names:
        assert (generator.OUT / name).read_bytes() == (committed / name).read_bytes().replace(b"\r\n", b"\n"), name

    def close(a, b):
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
        if isinstance(a, float):
            return a == pytest.approx(b, rel=1e-9)
        return a == b

    with open(generator.KEY) as f, open(EXAMPLES / "data" / "answer_key.json") as g:
        assert close(json.load(f), json.load(g))


def staged(path):
    """The notebook as git stores it: the index copy, after the nbstripout
    filter, so a local run that left outputs in the working tree does not fail
    this check. Falls back to the file outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "show", f":{path.relative_to(EXAMPLES.parent).as_posix()}"],
            cwd=EXAMPLES.parent, capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        out = path.read_bytes()
    return json.loads(out.decode("utf-8"))


def test_notebooks_are_committed_without_outputs():
    # Results are published on the docs site, which executes the notebooks
    # (`cargo xtask docs`). The repository keeps code only, so re-running a
    # notebook changes nothing git tracks (the nbstripout filter, .gitattributes).
    paths = sorted(EXAMPLES.glob("*.ipynb"))
    assert paths
    for path in paths:
        nb = staged(path)
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert code, path.name
        assert all(c["outputs"] == [] for c in code), f"{path.name} is committed with outputs"
        assert all(c["execution_count"] is None for c in code), f"{path.name} has execution counts"
        assert "version" not in nb["metadata"].get("language_info", {}), path.name
