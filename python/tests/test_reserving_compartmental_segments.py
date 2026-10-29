"""Compartmental.fit_segments, Gesmann and Morris' model 1 over segments
(docs/design/reserving-compartmental.md, decision 8).

The seed sweeps, the twins and the diagonal backtest are in
validation/tests/reserving_compartmental_segments.rs; these tests check the
binding: settings, the joint reserves, future paid and future outstanding
cells and the frames, one segment against ``fit``, independence and rank
correlation between segments, replay by seed, the diagnostics warning and
the errors, on company 337 workers' compensation (the papers' triangle)
under two keys.
"""

import csv
import warnings
from pathlib import Path

import pytest

from prospicio.distributions import PredictiveDistribution
from prospicio.models import Elpd
from prospicio.reserving import (
    Compartmental,
    CompartmentalFit,
    CompartmentalFits,
    CompartmentalPriors,
    Triangle,
)

VALIDATION = Path(__file__).resolve().parents[2] / "validation"
GM = CompartmentalPriors.gesmann_morris()

N_SIMS = 2000


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
    """Each segment's total per simulation, in index order."""
    comps = [c[0] for c in pd.components()]
    names = list(dict.fromkeys(comps))
    totals = {n: [0.0] * pd.n_sims for n in names}
    for row_i, row in enumerate(pd.draw_matrix()):
        for name, v in zip(comps, row):
            totals[name][row_i] += v
    return [totals[n] for n in names]


def ranks(xs):
    order = sorted(range(len(xs)), key=xs.__getitem__)
    r = [0.0] * len(xs)
    for k, i in enumerate(order):
        r[i] = float(k)
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5


@pytest.fixture(scope="module")
def two():
    return lines(["a", "b"])


@pytest.fixture(scope="module")
def fits(two):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the diagnostics are checked below
        return Compartmental(GM, n_sims=N_SIMS, seed=5).fit_segments(
            two, "paid", "premium", incurred="incurred"
        )


def test_settings():
    m = Compartmental(GM)
    assert m.dependence == "independent" and m.spearman is None
    m = Compartmental(
        GM, dependence="rank_correlation", spearman=[[1.0, 0.5], [0.5, 1.0]]
    )
    assert m.dependence == "rank_correlation"
    assert m.spearman == [[1.0, 0.5], [0.5, 1.0]]
    m2 = Compartmental.model_2(
        CompartmentalPriors.gesmann_morris_2(),
        dependence="rank_correlation",
        spearman=[[1.0, 0.2], [0.2, 1.0]],
    )
    assert m2.dependence == "rank_correlation" and m2.likelihood == "lognormal"
    assert Compartmental(GM, dependence="synchronized").dependence == "synchronized"


def test_fields_and_frames(fits):
    assert isinstance(fits, CompartmentalFits)
    assert fits.keys == ["lob"]
    assert fits.index == ["a", "b"]
    assert len(fits.origins) == 20 and fits.origins[:2] == ["1988", "1989"]
    assert len(fits.premium) == 20 and len(fits.latest_paid) == 20
    assert fits.rate_swaps == [0, 0]

    for pd, dims in [
        (fits.reserves, ["lob", "origin"]),
        (fits.paid, ["lob", "origin", "age"]),
        (fits.outstanding, ["lob", "origin", "age"]),
    ]:
        assert isinstance(pd, PredictiveDistribution)
        assert pd.dims == dims and pd.n_sims == N_SIMS
    assert fits.reserves.n_components == 20
    assert fits.paid.n_components == fits.outstanding.n_components == 2 * 55

    long = fits.to_frame()
    assert list(long.columns) == [
        "lob", "origin", "premium", "latest_paid", "mean", "std_dev",
        "ultimate_loss_ratio",
    ]
    assert len(long) == 20
    assert long["mean"].sum() == pytest.approx(fits.reserves.mean(), rel=1e-9)
    totals = fits.totals_frame()
    assert list(totals["lob"]) == ["a", "b"]
    assert totals["mean"].sum() == pytest.approx(fits.reserves.mean(), rel=1e-9)
    assert len(fits.paid_frame()) == 110 and len(fits.outstanding_frame()) == 110
    assert set(fits.paid_frame()["lob"]) == {"a", "b"}

    d = fits.diagnostics_frame()
    assert list(d.columns) == [
        "lob", "max_rhat", "min_ess_bulk", "min_ess_tail", "divergences", "warnings",
    ]
    assert list(d["lob"]) == ["a", "b"]
    assert repr(fits).startswith("CompartmentalFits(segments=2, origins=20, n_sims=2000")


def test_a_reserve_is_the_paid_at_the_last_age_less_the_latest(fits):
    # Each simulation's reserve of an origin is its last future paid cell
    # less the latest paid, in every segment.
    comps = fits.paid.components()
    last = {}
    for i, (lob, origin, age) in enumerate(comps):
        if (lob, origin) not in last or int(age) > int(comps[last[(lob, origin)]][2]):
            last[(lob, origin)] = i
    keyed = {c: i for i, c in enumerate(fits.reserves.components())}
    latest = dict(zip(zip(fits.to_frame()["lob"], fits.to_frame()["origin"]), fits.latest_paid))
    paid, res = fits.paid.draw_matrix(), fits.reserves.draw_matrix()
    for key, i in last.items():
        j = keyed[key]
        for row in range(0, 200, 40):
            assert paid[row][i] - latest[key] == pytest.approx(res[row][j], rel=1e-9)


def test_single_segment_fields_need_one_segment(fits):
    for field in ["diagnostics", "parameter_names", "draws", "chains"]:
        with pytest.raises(ValueError, match=r"single-segment fit.*segment\(\.\.\.\)"):
            getattr(fits, field)
    with pytest.raises(ValueError, match="single-segment fit"):
        fits.summary()
    with pytest.raises(ValueError, match="single-segment fit"):
        fits.loo()


def test_one_segment(fits, two):
    one = fits.segment(lob="b")
    assert isinstance(one, CompartmentalFits)
    assert one.index == ["b"]
    assert one.reserves.n_components == 10 and one.paid.n_components == 55
    # Its part of the joint distributions.
    assert one.reserves.draw_matrix() == [row[10:] for row in fits.reserves.draw_matrix()]
    assert one.paid.draw_matrix() == [row[55:] for row in fits.paid.draw_matrix()]
    assert one.premium == fits.premium[10:]

    # The posterior of one segment: the same model as ``fit`` with its own
    # seeds, so a close posterior but other draws.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alone = Compartmental(GM, n_sims=N_SIMS, seed=5).fit(
            lines(["b"]), "paid", "premium", incurred="incurred"
        )
    assert isinstance(alone, CompartmentalFit)
    assert one.premium == alone.premium
    assert one.parameter_names == alone.parameter_names
    assert one.chains == 4 and len(one.draws) == 8000
    assert one.draws != alone.draws
    s, a = one.summary(), alone.summary()
    assert list(s["parameter"]) == one.parameter_names
    assert s.loc[0, "mean"] == pytest.approx(a.loc[0, "mean"], rel=0.05)
    assert one.totals_frame()["mean"][0] == pytest.approx(
        alone.totals_frame()["mean"][0], rel=0.05
    )
    loo = one.loo()
    assert isinstance(loo, Elpd) and len(loo.pointwise) > 0


def test_independent_segments_are_independent(fits):
    # The same triangle under two names: shared random numbers would make
    # their totals almost perfectly rank correlated.
    a, b = totals_by_segment(fits.reserves)
    assert abs(spearman(a, b)) < 4.5 / N_SIMS**0.5
    assert sum(a) / sum(b) == pytest.approx(1.0, rel=0.02)


def test_rank_correlation_moves_the_whole_row(two, fits):
    independent = fits
    target = 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = Compartmental(
            GM, n_sims=N_SIMS, seed=5, dependence="rank_correlation",
            spearman=[[1.0, target], [target, 1.0]],
        ).fit_segments(two, "paid", "premium", incurred="incurred")
    a, b = totals_by_segment(fits.reserves)
    assert spearman(a, b) == pytest.approx(target, abs=0.04)
    # The future paid cells moved with the reserves: the last age's total
    # paid of a segment ranks exactly as its total reserve.
    paid = fits.paid
    comps = paid.components()
    for seg, reserve in zip(["a", "b"], (a, b)):
        cols = [i for i, c in enumerate(comps) if c[0] == seg and c[2] == 120]
        assert cols, "the oldest age is 120 months"
        last = [sum(row[i] for i in cols) for row in paid.draw_matrix()]
        assert spearman(last, reserve) == pytest.approx(1.0, abs=1e-9)
    # The paid patterns moved with the rows (reserving-compartmental.md
    # decision 14): the same posterior draws as the independent fit's, in
    # another order, each ending at 1 at the square's last age.
    for seg in ("a", "b"):
        moved = fits.patterns.pattern_draws(segment=seg)
        base = independent.patterns.pattern_draws(segment=seg)
        assert len(moved) == N_SIMS
        assert all(row[-1] == 1.0 for row in moved)
        assert sorted(map(tuple, moved)) == sorted(map(tuple, base))
        assert moved != base


def test_replay_by_seed(two):
    settings = dict(chains=2, tune=200, draws=200, n_sims=300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = lambda seed: Compartmental(GM, seed=seed, **settings).fit_segments(  # noqa: E731
            two, "paid", "premium", incurred="incurred"
        )
        a, b, c = run(7), run(7), run(8)
    assert a.reserves.draw_matrix() == b.reserves.draw_matrix()
    assert a.paid.draw_matrix() == b.paid.draw_matrix()
    assert a.segment(lob="a").draws == b.segment(lob="a").draws
    assert a.reserves.draw_matrix() != c.reserves.draw_matrix()


def test_failed_diagnostics_warn_with_segment_label(two):
    # 2 x 10 kept draws cannot reach a bulk ESS of 200 (100 per chain).
    with pytest.warns(UserWarning, match=r"Compartmental: the MCMC diagnostics failed.*a: .*ESS"):
        f = Compartmental(GM, chains=2, tune=50, draws=10, n_sims=100, seed=1).fit_segments(
            two, "paid", "premium", incurred="incurred"
        )
    assert (f.diagnostics_frame()["warnings"] != "").all()
    assert f.reserves.n_sims == 100


def test_errors(two):
    with pytest.raises(ValueError, match="dependence must be"):
        Compartmental(GM, dependence="copula")
    with pytest.raises(ValueError, match="needs a spearman matrix"):
        Compartmental(GM, dependence="rank_correlation")
    with pytest.raises(ValueError, match="only with dependence"):
        Compartmental(GM, spearman=[[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="square"):
        Compartmental(GM, dependence="rank_correlation", spearman=[[1.0, 0.5]])
    # Synchronized is refused before any sampling.
    with pytest.raises(ValueError, match="Synchronized.*no residuals"):
        Compartmental(GM, dependence="synchronized").fit_segments(
            two, "paid", "premium", incurred="incurred"
        )
    # A Spearman matrix of the wrong size for the segments.
    three = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError):
        Compartmental(
            GM, chains=2, tune=50, draws=50, n_sims=100,
            dependence="rank_correlation", spearman=three,
        ).fit_segments(two, "paid", "premium", incurred="incurred")
    with pytest.raises(ValueError, match="exactly one of outstanding"):
        Compartmental(GM).fit_segments(two, "paid", "premium")
    with pytest.raises(ValueError, match="missing"):
        Compartmental(GM).fit_segments(two, "missing", "premium", incurred="incurred")
    # fit still takes one segment only.
    with pytest.raises(ValueError):
        Compartmental(GM).fit(two, "paid", "premium", incurred="incurred")
