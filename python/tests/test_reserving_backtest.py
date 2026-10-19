"""The diagonal backtest (docs/design/reserving-bayes.md, decision 21).

The metrics and the diagnostics against their definitions are the Rust
tests of prospicio-reserving's backtest.rs; these tests check the binding:
the tables, the models it accepts, the diagnostics of the samplers, the
reproducibility by seed and the errors.
"""

import csv
import math
from pathlib import Path

import pytest

from prospicio.models import Glm, Terms
from prospicio.reserving import (
    Backtest,
    BayesOdp,
    GlmCandidate,
    Meyers,
    Triangle,
    diagonal_backtest,
)

VALIDATION = Path(__file__).resolve().parents[2] / "validation"

METRICS = [
    "crps",
    "coverage",
    "actual_vs_expected",
    "total_crps",
    "mean_log_density",
    "total_percentile",
]

# One chain of a few draws: the sampler runs in milliseconds in a release
# build and fails its diagnostics, which the backtest reports and scores.
TINY = dict(chains=1, tune=40, draws=40)


def rows_of(name):
    with open(VALIDATION / "data" / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


@pytest.fixture(scope="module")
def raa():
    rows = rows_of("raa")
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows],
        development=[int(r["development"]) for r in rows],
        values={"values": [float(r["value"]) for r in rows]},
    )


@pytest.fixture(scope="module")
def comauto():
    """Group 353's upper triangle, as Meyers fits it, and its premium."""
    rows = rows_of("clrd_comauto_353")
    upper = [r for r in rows if int(r["origin"]) - 1988 + int(r["development"]) // 12 <= 10]
    tri = Triangle.from_long(
        origin=[int(r["origin"]) for r in upper],
        development=[int(r["development"]) for r in upper],
        values={"paid": [float(r["paid"]) for r in upper]},
    )
    premium = {int(r["origin"]): float(r["premium"]) for r in rows}
    return tri, [premium[o] for o in sorted(premium)]


@pytest.fixture(scope="module")
def odp_bt(raa):
    return diagonal_backtest(raa, "values", [GlmCandidate.odp()], 2, n_sims=500, seed=3)


def col(table, name):
    """A column of a table, as a list, from a DataFrame or a dict of lists."""
    return list(table[name])


def test_the_glm_candidate():
    odp = GlmCandidate.odp()
    assert odp.name == "odp"
    assert repr(odp) == "GlmCandidate(name='odp')"
    custom = GlmCandidate(Terms().intercept().factor("origin"), Glm("poisson", "log"))
    assert custom.name == "glm"
    assert isinstance(custom.terms, Terms) and isinstance(custom.glm, Glm)
    assert GlmCandidate(Terms().intercept(), Glm("poisson", "log"), name="flat").name == "flat"


def test_glm_tables_and_shapes(odp_bt):
    bt = odp_bt
    assert isinstance(bt, Backtest)
    assert bt.models == ["odp"]
    assert bt.metrics == METRICS
    assert bt.n_splits == 2
    assert bt.diagonals == ["1989", "1990"]
    assert repr(bt) == (
        "Backtest(models=['odp'], diagonals=['1989', '1990'], metrics=6, warnings=0)"
    )

    scores = bt.scores()
    assert list(scores.columns if hasattr(scores, "columns") else scores) == [
        "model",
        "diagonal",
        "metric",
        "value",
    ]
    assert len(col(scores, "value")) == 1 * 2 * len(METRICS)
    assert set(col(scores, "model")) == {"odp"}
    assert col(scores, "diagonal") == ["1989"] * 6 + ["1990"] * 6
    assert col(scores, "metric") == METRICS * 2

    mean = bt.mean()
    assert list(mean.columns if hasattr(mean, "columns") else mean) == ["model", *METRICS]
    assert col(mean, "model") == ["odp"]
    by = {(d, m): v for d, m, v in zip(col(scores, "diagonal"), col(scores, "metric"), col(scores, "value"))}
    for metric in METRICS:
        expected = (by["1989", metric] + by["1990", metric]) / 2
        assert col(mean, metric)[0] == pytest.approx(expected, rel=1e-12)


def test_glm_scores_are_in_range_and_consistent(odp_bt, raa):
    bt = odp_bt
    scores = bt.scores()
    by = {(d, m): v for d, m, v in zip(col(scores, "diagonal"), col(scores, "metric"), col(scores, "value"))}
    for d in bt.diagonals:
        assert all(math.isfinite(by[d, m]) for m in METRICS), d
        assert 0.0 <= by[d, "coverage"] <= 1.0
        assert 0.0 <= by[d, "total_percentile"] <= 1.0
        assert by[d, "crps"] > 0.0 and by[d, "total_crps"] > 0.0
        assert by[d, "actual_vs_expected"] > 0.0

    # The latest diagonal has 10 cells and the one before it 9; the rows
    # the models cannot fit are excluded from the scoring.
    assert [len(r) + e for r, e in zip(bt.scored_rows, bt.excluded)] == [9, 10]
    assert all(r == sorted(r) and len(set(r)) == len(r) for r in bt.scored_rows)

    # The log densities are the scored rows', diagonals in order, and the
    # mean log density of a diagonal is their mean over its rows.
    lpd = bt.log_densities
    assert list(lpd) == ["odp"]
    n_first = len(bt.scored_rows[0])
    assert len(lpd["odp"]) == n_first + len(bt.scored_rows[1])
    first = lpd["odp"][:n_first]
    assert by["1989", "mean_log_density"] == pytest.approx(sum(first) / n_first, rel=1e-12)

    # A GLM has no sampler: no diagnostics and no warnings.
    assert bt.warnings == []
    d = bt.diagnostics()
    assert len(col(d, "model")) == 0
    assert list(d.columns if hasattr(d, "columns") else d) == [
        "model",
        "diagonal",
        "max_rhat",
        "min_ess_bulk",
        "min_ess_tail",
        "divergences",
        "n_warnings",
    ]


def test_scores_replay_by_seed(raa, odp_bt):
    again = diagonal_backtest(raa, "values", [GlmCandidate.odp()], 2, n_sims=500, seed=3)
    assert col(again.scores(), "value") == col(odp_bt.scores(), "value")
    other = diagonal_backtest(raa, "values", [GlmCandidate.odp()], 2, n_sims=500, seed=4)
    assert col(other.scores(), "value") != col(odp_bt.scores(), "value")


def test_names_and_several_models(raa):
    named = {
        "odp": GlmCandidate.odp(),
        "origin only": GlmCandidate(Terms().intercept().factor("origin"), Glm("poisson", "log")),
    }
    bt = diagonal_backtest(raa, "values", named, 1, n_sims=200)
    assert bt.models == ["odp", "origin only"]
    assert list(bt.log_densities) == ["odp", "origin only"]
    assert col(bt.mean(), "model") == ["odp", "origin only"]
    # Both models are scored on the same rows.
    assert len(bt.log_densities["odp"]) == len(bt.log_densities["origin only"])

    # A list takes the models' own names; a dict renames them.
    listed = diagonal_backtest(
        raa, "values", [GlmCandidate.odp(), GlmCandidate(Terms().intercept(), Glm("poisson", "log"))], 1, n_sims=200
    )
    assert listed.models == ["odp", "glm"]


def test_bayes_odp_reports_diagnostics_and_is_scored(raa):
    settings = BayesOdp(**TINY)
    bt = diagonal_backtest(raa, "values", {"bayes": settings}, 1, n_sims=200, seed=1)
    assert bt.models == ["bayes"]
    assert bt.diagonals == ["1990"]
    scores = bt.scores()
    by = dict(zip(col(scores, "metric"), col(scores, "value")))
    assert all(math.isfinite(by[m]) for m in METRICS), by
    assert 0.0 <= by["total_percentile"] <= 1.0

    d = bt.diagnostics()
    assert col(d, "model") == ["bayes"] and col(d, "diagonal") == ["1990"]
    assert col(d, "min_ess_bulk")[0] > 0.0
    # 40 draws cannot pass the ESS threshold: the failure is reported and
    # the fit is scored all the same.
    warned = bt.warnings
    assert col(d, "n_warnings")[0] == len(warned) > 0
    assert all(m == "bayes" and diag == "1990" for m, diag, _ in warned)
    assert any("ESS" in message for _, _, message in warned)
    assert "warnings=" in repr(bt) and "warnings=0" not in repr(bt)

    # The backtest's n_sims and seed replace the settings'.
    again = diagonal_backtest(raa, "values", [BayesOdp(n_sims=7, seed=9, **TINY)], 1, n_sims=200, seed=1)
    assert again.models == ["bayes_odp"]
    assert col(again.scores(), "value") == col(scores, "value")


def test_meyers_needs_premium_and_reports_diagnostics(comauto):
    tri, premium = comauto
    with pytest.raises(ValueError, match="Meyers needs premium"):
        diagonal_backtest(tri, "paid", [Meyers(**TINY)], 1)
    bt = diagonal_backtest(
        tri, "paid", [Meyers("csr", **TINY)], 1, premium=premium, n_sims=200, seed=2
    )
    assert bt.models == ["meyers_csr"]
    assert bt.diagonals == ["1997"]
    scores = bt.scores()
    by = dict(zip(col(scores, "metric"), col(scores, "value")))
    assert all(math.isfinite(by[m]) for m in METRICS), by
    d = bt.diagnostics()
    assert col(d, "model") == ["meyers_csr"]
    assert col(d, "n_warnings")[0] == len(bt.warnings) > 0
    assert len(bt.log_densities["meyers_csr"]) == len(bt.scored_rows[0])


def test_errors(raa):
    odp = GlmCandidate.odp()
    with pytest.raises(ValueError, match="two models are named 'odp'.*dict"):
        diagonal_backtest(raa, "values", [odp, GlmCandidate.odp()], 1)
    with pytest.raises(TypeError, match="BayesOdp, a Meyers or a GlmCandidate, got str"):
        diagonal_backtest(raa, "values", ["odp"], 1)
    with pytest.raises(TypeError, match="BayesOdp, a Meyers or a GlmCandidate, got int"):
        diagonal_backtest(raa, "values", {"x": 1}, 1)
    with pytest.raises(TypeError, match="dict of name to model or a list"):
        diagonal_backtest(raa, "values", 5, 1)
    with pytest.raises(ValueError, match="at least one model"):
        diagonal_backtest(raa, "values", [], 1)
    with pytest.raises(ValueError, match="at least one model"):
        diagonal_backtest(raa, "values", {}, 1)
    with pytest.raises(ValueError):
        diagonal_backtest(raa, "values", [odp], 0)
    with pytest.raises(ValueError):
        diagonal_backtest(raa, "values", [odp], 1, interval=1.0)
    with pytest.raises(ValueError):
        diagonal_backtest(raa, "nope", [odp], 1)
    with pytest.raises(ValueError):
        diagonal_backtest(raa, "values", [odp], 1, premium=[1.0, 2.0])
