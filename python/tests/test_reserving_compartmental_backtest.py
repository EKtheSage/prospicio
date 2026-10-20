"""Compartmental in the diagonal backtest
(docs/design/reserving-compartmental.md, decision 8, "As built").

``Compartmental.candidate`` puts the model next to the other models of
``diagonal_backtest`` (the paid increments of the held-out diagonals) and
``CompartmentalCandidate.outstanding_backtest`` scores the held-out
outstanding of the same diagonals on the same fits. The seed sweeps and the
targets are in validation/tests/reserving_compartmental_segments.rs; these
tests check the binding on company 337 workers' compensation (the papers'
triangle) with a small sampler: the tables' shapes, finite metrics, the
fits shared between the two tables, replay by seed, the names and the
errors. Model 2 is left to the Rust validation (its sampler is too slow for
a test).
"""

import csv
import math
import time
import warnings
from pathlib import Path

import pytest

from prospicio.reserving import (
    Backtest,
    Compartmental,
    CompartmentalCandidate,
    CompartmentalPriors,
    GlmCandidate,
    Triangle,
    diagonal_backtest,
)

VALIDATION = Path(__file__).resolve().parents[2] / "validation"
GM = CompartmentalPriors.gesmann_morris()

N_SIMS = 400
N_DIAGONALS = 3
SEED = 3


def training_triangle(with_outstanding=False):
    """Company 337 at calendar years up to 1996, and 1997 with its premium
    only (the backtest's frame, as the Rust validation builds it), with an
    ``outstanding`` column of incurred less paid when asked."""
    path = VALIDATION / "data" / "clrd_wkcomp_337.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    premium_1997 = next(float(r["premium"]) for r in rows if r["origin"] == "1997")
    rows = [r for r in rows if int(r["origin"]) + int(r["development"]) // 12 - 1 <= 1996]
    assert len(rows) == 45
    nan = float("nan")
    origin = [int(r["origin"]) for r in rows] + [1997]
    age = [int(r["development"]) for r in rows] + [12]
    paid = [float(r["paid"]) for r in rows] + [nan]
    incurred = [float(r["incurred"]) for r in rows] + [nan]
    premium = [float(r["premium"]) for r in rows] + [premium_1997]
    columns = {"paid": paid, "incurred": incurred, "premium": premium}
    if with_outstanding:
        columns["outstanding"] = [i - p for i, p in zip(incurred, paid)]
    return Triangle.from_long(origin, age, columns)


def small_model(**kw):
    return Compartmental(GM, chains=2, tune=300, draws=300, n_sims=N_SIMS, seed=SEED, **kw)


def candidate(tri, model=None):
    return (model or small_model()).candidate(tri, "paid", "premium", incurred="incurred")


def col(table, name):
    if isinstance(table, dict):
        return list(table[name])
    return table[name].tolist()


@pytest.fixture(scope="module")
def tri():
    return training_triangle()


@pytest.fixture(scope="module")
def tables(tri):
    """The paid table (beside the ODP GLM) and the outstanding table of one
    candidate, with the time each took."""
    cand = candidate(tri)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        paid = diagonal_backtest(
            tri, "paid", [GlmCandidate.odp(), cand], N_DIAGONALS,
            n_sims=N_SIMS, seed=SEED,
        )
        t_paid = time.perf_counter() - t0
        t0 = time.perf_counter()
        outstanding = cand.outstanding_backtest(N_DIAGONALS, n_sims=N_SIMS, seed=SEED)
        t_os = time.perf_counter() - t0
    return cand, paid, outstanding, t_paid, t_os


def test_the_candidate_and_its_names(tri):
    cand = candidate(tri)
    assert isinstance(cand, CompartmentalCandidate)
    assert cand.name == "compartmental_gm1"
    assert (cand.paid, cand.premium) == ("paid", "premium")
    assert (cand.outstanding, cand.outstanding_from) == ("incurred", "incurred")
    assert "compartmental_gm1" in repr(cand)
    model_2 = Compartmental.model_2(CompartmentalPriors.gesmann_morris_2())
    assert model_2.candidate(tri, "paid", "premium", outstanding="incurred").name == (
        "compartmental_gm2"
    )
    as_column = small_model().candidate(tri, "paid", "premium", outstanding="incurred")
    assert as_column.outstanding_from == "column"


def test_the_paid_table_has_the_expected_cells(tables):
    _, paid, _, _, _ = tables
    assert isinstance(paid, Backtest)
    assert paid.models == ["odp", "compartmental_gm1"]
    assert paid.n_splits == N_DIAGONALS
    assert paid.diagonals == ["1994", "1995", "1996"]
    # 5, 6 and 7 cells over the three diagonals, as in the Rust validation.
    assert [len(r) for r in paid.scored_rows] == [5, 6, 7]
    assert paid.excluded == [2, 2, 2]
    scores = paid.scores()
    assert len(col(scores, "value")) == 2 * len(paid.metrics) * N_DIAGONALS
    means = paid.mean()
    for metric in ("crps", "coverage", "actual_vs_expected", "total_crps"):
        assert all(math.isfinite(v) for v in col(means, metric)), metric
    lpd = col(means, "mean_log_density")
    assert all(math.isfinite(v) for v in lpd)
    # The ODP GLM is deterministic and has no sampler; the candidate has one.
    diag = paid.diagnostics()
    sampled = [m for m in col(diag, "model") if m == "compartmental_gm1"]
    assert len(sampled) == N_DIAGONALS
    assert all(math.isnan(v) for m, v in zip(col(diag, "model"), col(diag, "min_ess_bulk")) if m == "odp")
    assert all(v > 0 for m, v in zip(col(diag, "model"), col(diag, "min_ess_bulk")) if m != "odp")


def test_the_outstanding_table_scores_the_same_cells(tables):
    _, paid, os, _, _ = tables
    assert os.models == ["compartmental_gm1_outstanding"]
    assert os.n_splits == N_DIAGONALS
    assert os.diagonals == paid.diagonals
    assert os.scored_rows == paid.scored_rows
    assert os.excluded == paid.excluded
    means = os.mean()
    for metric in os.metrics:
        assert all(math.isfinite(v) for v in col(means, metric)), metric
    coverage = col(means, "coverage")[0]
    assert 0.0 <= coverage <= 1.0
    # The outstanding is on a larger scale than a paid increment, and the
    # fit that forecasts it is the paid table's: the same diagnostics.
    p_diag = paid.diagnostics()
    o_diag = os.diagnostics()
    p_ess = [v for m, v in zip(col(p_diag, "model"), col(p_diag, "min_ess_bulk")) if m != "odp"]
    assert col(o_diag, "min_ess_bulk") == p_ess
    assert col(o_diag, "max_rhat") == [
        v for m, v in zip(col(p_diag, "model"), col(p_diag, "max_rhat")) if m != "odp"
    ]


def test_the_two_tables_share_their_fits(tables):
    # This timing is the proof that the fits are shared: equal diagnostics
    # (checked above) would also follow from a refit on the same seed. The
    # outstanding table adds forecasts, not NUTS runs: it is a small
    # fraction of the time of the paid table, which sampled once per split.
    _, _, _, t_paid, t_os = tables
    assert t_os < 0.5 * t_paid, (t_os, t_paid)


def test_the_scores_agree_with_the_rust_validation(tables):
    # The validation's targets (reserving_compartmental_segments.rs) are for
    # the model's default sampler and 4,000 simulations; this sampler is a
    # small one, so only the scale of the scores is held to them: each
    # diagonal's actual over expected within a quarter of the target, and
    # the outstanding table's scores in the validation's order of size.
    _, paid, os, _, _ = tables
    ae = {
        (m, d): v
        for m, d, metric, v in zip(
            col(paid.scores(), "model"),
            col(paid.scores(), "diagonal"),
            col(paid.scores(), "metric"),
            col(paid.scores(), "value"),
        )
        if metric == "actual_vs_expected"
    }
    for diagonal, target in zip(paid.diagonals, (1.3404, 1.0679, 1.1531)):
        assert abs(ae[("compartmental_gm1", diagonal)] / target - 1) < 0.25, diagonal
    # The ODP GLM is deterministic: the validation's targets to 0.001.
    for diagonal, target in zip(paid.diagonals, (0.89353, 0.9217, 1.0152)):
        assert ae[("odp", diagonal)] == pytest.approx(target, abs=0.001)
    crps = col(os.mean(), "crps")[0]
    total_crps = col(os.mean(), "total_crps")[0]
    assert 1881.9 / 2 < crps < 1881.9 * 2
    assert 3617.3 / 2 < total_crps < 3617.3 * 2


def test_an_outstanding_column_scores_as_incurred_less_paid():
    # outstanding= reads the case reserves from a column; here the column is
    # incurred less paid, so both tables equal those of incurred= on the
    # same seed, cell for cell.
    tri = training_triangle(with_outstanding=True)
    by_column = small_model().candidate(tri, "paid", "premium", outstanding="outstanding")
    by_incurred = candidate(tri)
    assert by_column.outstanding_from == "column"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tables = [
            (
                diagonal_backtest(tri, "paid", [c], 1, n_sims=200, seed=SEED),
                c.outstanding_backtest(1, n_sims=200, seed=SEED),
            )
            for c in (by_column, by_incurred)
        ]
    (paid_c, os_c), (paid_i, os_i) = tables
    for got, want in ((paid_c, paid_i), (os_c, os_i)):
        values = col(got.scores(), "value")
        assert all(math.isfinite(v) for v in values)
        assert values == col(want.scores(), "value")
        assert got.scored_rows == want.scored_rows
    assert len(os_c.scored_rows[0]) == 7


def test_a_named_candidate_and_replay_by_seed(tri):
    first = candidate(tri)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = diagonal_backtest(
            tri, "paid", {"gm1": first}, 1, n_sims=200, seed=SEED
        )
        b = diagonal_backtest(
            tri, "paid", {"gm1": candidate(tri)}, 1, n_sims=200, seed=SEED
        )
    assert a.models == ["gm1"]
    assert col(a.scores(), "value") == col(b.scores(), "value")


def test_errors(tri):
    # exactly one of outstanding and incurred
    with pytest.raises(ValueError, match="exactly one"):
        small_model().candidate(tri, "paid", "premium")
    with pytest.raises(ValueError, match="exactly one"):
        small_model().candidate(
            tri, "paid", "premium", outstanding="incurred", incurred="incurred"
        )
    cand = candidate(tri)
    # the backtest's column and triangle must be the candidate's
    with pytest.raises(ValueError, match="another triangle or paid column"):
        diagonal_backtest(tri, "incurred", [cand], 1, n_sims=100)
    other = Triangle.from_long([2020, 2021, 2021], [12, 12, 24], {"paid": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="another triangle or paid column"):
        diagonal_backtest(other, "paid", [cand], 1, n_sims=100)
    # more diagonals than the triangle holds
    with pytest.raises(ValueError, match="earlier period"):
        cand.outstanding_backtest(50, n_sims=100)
    with pytest.raises(TypeError, match="CompartmentalCandidate"):
        diagonal_backtest(tri, "paid", [small_model()], 1, n_sims=100)
