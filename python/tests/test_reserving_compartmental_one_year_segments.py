"""Compartmental.one_year_segments, the one-year view of Gesmann and Morris'
model 1 over segments (docs/design/reserving-compartmental.md, decision 8,
and reserving-bayes.md, decision 20).

The seeds, the single-segment identity and the rank correlation are checked
in the Rust unit tests and in validation/tests/reserving_compartmental_one_year.rs;
these tests check the binding on company 337 workers' compensation under
two keys, with tiny sampler budgets: the result's fields and frames, the
single-segment accessors, a segment against its part of the joint result,
independence and rank correlation between segments, replay by seed, the
warnings and the errors.
"""

import csv
import warnings
from pathlib import Path

import pytest

from prospicio.distributions import PredictiveDistribution
from prospicio.reserving import (
    Compartmental,
    CompartmentalOneYearFits,
    CompartmentalPriors,
    Triangle,
)

VALIDATION = Path(__file__).resolve().parents[2] / "validation"
GM = CompartmentalPriors.gesmann_morris()

N_SIMS = 300
SHORT = dict(chains=2, tune=100, draws=100, n_sims=N_SIMS)


def rows_337():
    path = VALIDATION / "data" / "clrd_wkcomp_337.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    premium_1997 = next(float(r["premium"]) for r in rows if r["origin"] == "1997")
    rows = [r for r in rows if int(r["origin"]) + int(r["development"]) // 12 - 1 <= 1996]
    assert len(rows) == 45
    return rows, premium_1997


def lines(names):
    """Company 337 once per name, as the segments of a ``lob`` key."""
    rows, premium_1997 = rows_337()
    nan = float("nan")
    keys, origin, age, paid, incurred, premium = [], [], [], [], [], []
    for name in names:
        keys += [name] * 46
        origin += [int(r["origin"]) for r in rows] + [1997]
        age += [int(r["development"]) for r in rows] + [12]
        paid += [float(r["paid"]) for r in rows] + [nan]
        incurred += [float(r["incurred"]) for r in rows] + [nan]
        premium += [float(r["premium"]) for r in rows] + [premium_1997]
    return Triangle.from_long(
        origin, age,
        {"paid": paid, "incurred": incurred, "premium": premium},
        keys={"lob": keys},
    )


def totals_by_segment(pd):
    comps = [c[0] for c in pd.components()]
    names = list(dict.fromkeys(comps))
    totals = {n: [0.0] * pd.n_sims for n in names}
    for row_i, row in enumerate(pd.draw_matrix()):
        for name, v in zip(comps, row):
            totals[name][row_i] += v
    return [totals[n] for n in names]


def spearman(x, y):
    def ranks(xs):
        order = sorted(range(len(xs)), key=xs.__getitem__)
        r = [0.0] * len(xs)
        for k, i in enumerate(order):
            r[i] = float(k)
        return r

    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5


def view(tri, **settings):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # short chains miss a threshold
        model = {"seed": 5, **SHORT, **settings.pop("model", {})}
        return Compartmental(GM, **model).one_year_segments(
            tri, "paid", "premium", incurred="incurred",
            **{"refit": "reweight_only", **settings},
        )


@pytest.fixture(scope="module")
def two():
    return lines(["a", "b"])


@pytest.fixture(scope="module")
def views(two):
    return view(two)


def test_fields_and_frames(views):
    assert isinstance(views, CompartmentalOneYearFits)
    assert views.keys == ["lob"] and views.index == ["a", "b"]
    assert len(views.origins) == 20 and views.origins[:2] == ["1988", "1989"]
    assert len(views.premium) == len(views.latest_paid) == 20
    assert len(views.opening_ultimate) == len(views.opening_reserve) == 20
    assert views.approximate is None and views.k_threshold == 0.7
    cdr = views.cdr
    assert isinstance(cdr, PredictiveDistribution)
    assert cdr.dims == ["lob", "origin"]
    assert cdr.n_components == 20 and cdr.n_sims == N_SIMS
    # Every origin of the training triangle has a cell in the year (1988's 120
    # months, 1997's first), and the result varies.
    assert all(any(row[j] != 0.0 for row in cdr.draw_matrix()) for j in range(20))

    long = views.to_frame()
    assert list(long.columns) == [
        "lob", "origin", "premium", "latest_paid", "opening_ultimate",
        "opening_reserve", "cdr_mean", "cdr_std_dev",
    ]
    assert len(long) == 20
    assert long["cdr_mean"].sum() == pytest.approx(cdr.mean(), rel=1e-9, abs=1e-6)
    totals = views.totals_frame()
    assert list(totals["lob"]) == ["a", "b"]
    assert "flagged_share" in totals.columns
    assert totals["opening_reserve"].sum() == pytest.approx(sum(views.opening_reserve))
    d = views.diagnostics_frame()
    assert list(d["lob"]) == ["a", "b"] and "warnings" in d.columns
    assert repr(views).startswith("CompartmentalOneYearFits(segments=2, origins=20, n_sims=300")


def test_single_segment_accessors_need_one_segment(views):
    for field in ["diagnostics", "parameter_names", "draws", "chains", "pareto_k",
                  "refitted", "moment_matched", "unrefitted", "refit_warnings",
                  "flagged_share"]:
        with pytest.raises(ValueError, match=r"single-segment fit.*segment\(\.\.\.\)"):
            getattr(views, field)
    with pytest.raises(ValueError, match="single-segment fit"):
        views.summary()
    with pytest.raises(ValueError, match="single-segment fit"):
        views.loo()


def test_one_segment_is_its_part_of_the_joint_result(views):
    one = views.segment(lob="b")
    assert isinstance(one, CompartmentalOneYearFits) and one.index == ["b"]
    assert one.cdr.n_components == 10
    assert one.cdr.draw_matrix() == [row[10:] for row in views.cdr.draw_matrix()]
    assert one.premium == views.premium[10:]
    assert len(one.pareto_k) == N_SIMS and one.refitted == [] and one.unrefitted == []
    assert one.moment_matched == []
    assert 0.0 <= one.flagged_share <= 1.0
    assert one.flagged_share == views.totals_frame()["flagged_share"][1]
    assert one.chains == 2 and len(one.draws) == 200
    assert one.parameter_names[0] == "k_er"
    assert list(one.summary()["parameter"]) == one.parameter_names
    assert len(one.loo().pointwise) > 0
    assert one.totals_frame()["cdr_std_dev"][0] == pytest.approx(
        views.totals_frame()["cdr_std_dev"][1]
    )
    assert repr(one).startswith("CompartmentalOneYearFits(origins=10, n_sims=300, flagged_share=")


def test_independent_segments_are_independent(views):
    # The same triangle under two names: shared random numbers would make
    # their totals almost perfectly rank correlated.
    a, b = totals_by_segment(views.cdr)
    assert abs(spearman(a, b)) < 4.5 / N_SIMS**0.5


def test_rank_correlation_moves_whole_rows(two, views):
    target = 0.5
    ranked = view(two, model=dict(
        dependence="rank_correlation", spearman=[[1.0, target], [target, 1.0]]))
    a, b = totals_by_segment(ranked.cdr)
    assert spearman(a, b) == pytest.approx(target, abs=0.06)
    # Each segment keeps its simulations as a set.
    for mine, theirs in zip(totals_by_segment(ranked.cdr), totals_by_segment(views.cdr)):
        assert sorted(mine) == pytest.approx(sorted(theirs))


def test_replay_by_seed_and_the_budget(two):
    a, b = view(two), view(two)
    assert a.cdr.draw_matrix() == b.cdr.draw_matrix()
    c = view(two, model=dict(seed=6))
    assert a.cdr.draw_matrix() != c.cdr.draw_matrix()
    # A smaller n_sims is the first simulations of a larger one.
    small = view(two, n_sims=50)
    assert small.cdr.n_sims == 50
    assert small.cdr.draw_matrix() == a.cdr.draw_matrix()[:50]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        refit0 = Compartmental(GM, seed=5, **SHORT).one_year_segments(
            two, "paid", "premium", incurred="incurred", refit="refit_all",
            max_refits=0, n_sims=20,
        )
    assert any("unrefitted" in str(w.message) or "max_refits" in str(w.message) for w in caught)
    assert refit0.cdr.n_sims == 20
    assert refit0.segment(lob="a").refitted == []
    assert len(refit0.segment(lob="a").unrefitted) == 20


def test_errors(two):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="Synchronized"):
            Compartmental(GM, dependence="synchronized", **SHORT).one_year_segments(
                two, "paid", "premium", incurred="incurred")
        with pytest.raises(ValueError):
            Compartmental(GM, dependence="rank_correlation",
                          spearman=[[1.0, 1.1], [1.1, 1.0]], **SHORT).one_year_segments(
                two, "paid", "premium", incurred="incurred")
        with pytest.raises(ValueError, match="refit"):
            Compartmental(GM, **SHORT).one_year_segments(
                two, "paid", "premium", incurred="incurred", refit="nonsense")
        with pytest.raises(ValueError, match="exactly one|outstanding"):
            Compartmental(GM, **SHORT).one_year_segments(two, "paid", "premium")
