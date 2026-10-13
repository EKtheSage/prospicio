"""BayesOdp.fit_segments, the Bayesian ODP over segments
(docs/design/reserving-bayes.md, decision 11).

The seed sweep and the diagonal backtest are in
validation/tests/reserving_bayes_segments.rs; these tests check the
binding: settings, the joint reserves and frames, one segment against
``fit``, independence and rank correlation between lines, replay by seed,
the diagnostics warning and the errors.
"""

import csv
import warnings
from pathlib import Path

import pytest

from prospicio.distributions import PredictiveDistribution
from prospicio.models import Elpd
from prospicio.reserving import BayesOdp, BayesOdpFits, ChainLadderFit, Triangle

VALIDATION = Path(__file__).resolve().parents[2] / "validation"

N_SIMS = 4000


def clrd(lines):
    """The lines of clrd_lines.csv in ``lines`` ((line, name) pairs) as the
    segments of a ``lob`` key."""
    with open(VALIDATION / "data" / "clrd_lines.csv", newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    keys, origin, age, paid = [], [], [], []
    for line, name in lines:
        for r in rows:
            if r["lob"] == line:
                keys.append(name)
                origin.append(int(r["origin"]))
                age.append(int(r["development"]))
                paid.append(float(r["paid"]))
    return Triangle.from_long(origin, age, {"paid": paid}, keys={"lob": keys})


def line_totals(fits):
    """Each segment's total reserve per simulation, in index order."""
    comps = [c[0] for c in fits.reserves.components()]
    names = list(dict.fromkeys(comps))
    totals = {n: [0.0] * fits.reserves.n_sims for n in names}
    for row_i, row in enumerate(fits.reserves.draw_matrix()):
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
def two_lines():
    return clrd([("comauto", "comauto"), ("wkcomp", "wkcomp")])


@pytest.fixture(scope="module")
def fits(two_lines):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # both lines pass every diagnostic
        return BayesOdp(n_sims=N_SIMS, seed=5).fit_segments(two_lines, "paid")


def test_settings():
    m = BayesOdp()
    assert m.dependence == "independent" and m.spearman is None
    assert "dependence" not in repr(m)
    m = BayesOdp(dependence="rank_correlation", spearman=[[1.0, 0.5], [0.5, 1.0]])
    assert m.dependence == "rank_correlation"
    assert m.spearman == [[1.0, 0.5], [0.5, 1.0]]
    assert repr(m).endswith('dependence="rank_correlation", spearman=[[1.0, 0.5], [0.5, 1.0]])')
    assert BayesOdp(dependence="synchronized").dependence == "synchronized"


def test_fields_and_frames(fits):
    assert isinstance(fits, BayesOdpFits)
    assert fits.keys == ["lob"]
    assert fits.index == ["comauto", "wkcomp"]
    assert isinstance(fits.chain_ladder, ChainLadderFit)
    assert len(fits.origins) == 20 and fits.origins[:2] == ["1988", "1989"]
    assert fits.development == [12 * k for k in range(1, 11)]

    r = fits.reserves
    assert isinstance(r, PredictiveDistribution)
    assert r.dims == ["lob", "origin"]
    assert r.n_sims == N_SIMS and r.n_components == 20
    assert r.provenance()["model"] == "bayes_odp"

    long = fits.to_frame()
    assert list(long.columns) == [
        "lob", "origin", "latest", "ultimate", "reserve", "mean", "std_dev",
    ]
    assert len(long) == 20
    assert long["mean"].sum() == pytest.approx(r.mean(), rel=1e-9)
    totals = fits.totals_frame()
    assert list(totals["lob"]) == ["comauto", "wkcomp"] and "scale" in totals.columns
    assert totals["mean"].sum() == pytest.approx(r.mean(), rel=1e-9)
    # Each line's posterior mean reserve sits near its chain ladder's.
    for _, row in totals.iterrows():
        assert row["mean"] == pytest.approx(row["reserve"], rel=0.03)
    assert len(fits.development_frame()) == 20

    d = fits.diagnostics_frame()
    assert list(d.columns) == [
        "lob", "max_rhat", "min_ess_bulk", "min_ess_tail", "divergences", "warnings",
    ]
    assert list(d["lob"]) == ["comauto", "wkcomp"]
    assert (d["max_rhat"] < 1.01).all() and (d["warnings"] == "").all()
    assert repr(fits).startswith("BayesOdpFits(segments=2, origins=20, n_sims=4000")


def test_single_segment_fields_need_one_segment(fits):
    for field in ["scale", "diagnostics", "parameter_names", "draws", "chains"]:
        with pytest.raises(ValueError, match=r"single-segment fit.*segment\(\.\.\.\)"):
            getattr(fits, field)
    with pytest.raises(ValueError, match="single-segment fit"):
        fits.summary()
    with pytest.raises(ValueError, match="single-segment fit"):
        fits.loo()


def test_one_segment(fits, two_lines):
    one = fits.segment(lob="wkcomp")
    assert isinstance(one, BayesOdpFits)
    assert one.index == ["wkcomp"]
    assert one.reserves.dims == ["lob", "origin"] and one.reserves.n_components == 10
    # Its part of the joint reserves.
    whole = fits.reserves.draw_matrix()
    assert one.reserves.draw_matrix() == [row[10:] for row in whole]

    # The posterior of one segment: the same model as ``fit``, with its own
    # seeds, so the same scale and a close posterior but other draws.
    alone = BayesOdp(n_sims=N_SIMS, seed=5).fit(clrd([("wkcomp", "wkcomp")]), "paid")
    assert one.scale == pytest.approx(alone.scale, rel=1e-12)
    assert one.parameter_names == alone.parameter_names
    assert one.chains == 4 and len(one.draws) == 4000
    assert one.draws != alone.draws
    assert one.diagnostics["warnings"] == []
    s = one.summary()
    assert list(s["parameter"]) == one.parameter_names
    a = alone.summary()
    assert s.loc[0, "mean"] == pytest.approx(a.loc[0, "mean"], abs=0.02)
    loo = one.loo()
    assert isinstance(loo, Elpd) and len(loo.pointwise) == 55
    assert loo.elpd == pytest.approx(alone.loo().elpd, abs=1.0)


def test_independent_lines_are_independent():
    # The same line under two names: shared random numbers would make their
    # totals almost perfectly rank correlated. 4.5 / sqrt(n_sims) is the
    # tolerance of validation/tests/reserving_bayes_segments.rs, sized from
    # a 40-seed sweep (sd about 1 / sqrt(n_sims)).
    twins = clrd([("wkcomp", "a"), ("wkcomp", "b")])
    fits = BayesOdp(n_sims=N_SIMS, seed=11).fit_segments(twins, "paid")
    a, b = line_totals(fits)
    assert abs(spearman(a, b)) < 4.5 / N_SIMS**0.5
    assert sum(a) / sum(b) == pytest.approx(1.0, rel=0.01)


def test_rank_correlation(two_lines):
    target = 0.5
    fits = BayesOdp(
        n_sims=N_SIMS, seed=5, dependence="rank_correlation",
        spearman=[[1.0, target], [target, 1.0]],
    ).fit_segments(two_lines, "paid")
    a, b = line_totals(fits)
    # Iman-Conover reaches the target within 0.009 at 10,000 simulations
    # over 40 seeds (sd at most 0.0029); 0.03 allows for 4,000.
    assert spearman(a, b) == pytest.approx(target, abs=0.03)


def test_replays_by_seed(two_lines):
    settings = dict(chains=2, tune=200, draws=200, n_sims=500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = BayesOdp(seed=7, **settings).fit_segments(two_lines, "paid")
        b = BayesOdp(seed=7, **settings).fit_segments(two_lines, "paid")
        c = BayesOdp(seed=8, **settings).fit_segments(two_lines, "paid")
    assert a.reserves.draw_matrix() == b.reserves.draw_matrix()
    assert a.segment(lob="comauto").draws == b.segment(lob="comauto").draws
    assert a.reserves.draw_matrix() != c.reserves.draw_matrix()


def test_failed_diagnostics_warn_with_segment_label(two_lines):
    # 2 x 10 kept draws cannot reach a bulk ESS of 200 (100 per chain).
    with pytest.warns(UserWarning, match=r"MCMC diagnostics failed.*comauto: .*ESS"):
        f = BayesOdp(chains=2, tune=50, draws=10, n_sims=100, seed=1).fit_segments(
            two_lines, "paid"
        )
    d = f.diagnostics_frame()
    assert (d["warnings"] != "").all()
    assert f.reserves.n_sims == 100


def test_errors(two_lines):
    with pytest.raises(ValueError, match="dependence must be"):
        BayesOdp(dependence="copula")
    with pytest.raises(ValueError, match="needs a spearman matrix"):
        BayesOdp(dependence="rank_correlation")
    with pytest.raises(ValueError, match="only with dependence"):
        BayesOdp(spearman=[[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="square"):
        BayesOdp(dependence="rank_correlation", spearman=[[1.0, 0.5]])
    # Synchronized is refused before any sampling.
    with pytest.raises(ValueError, match="Synchronized.*no residuals"):
        BayesOdp(dependence="synchronized").fit_segments(two_lines, "paid")
    # A Spearman matrix of the wrong size for the segments.
    three = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError):
        BayesOdp(
            chains=2, tune=50, draws=50, n_sims=100,
            dependence="rank_correlation", spearman=three,
        ).fit_segments(two_lines, "paid")
    with pytest.raises(ValueError, match="missing"):
        BayesOdp().fit_segments(two_lines, "missing")
    # fit still takes one segment only.
    with pytest.raises(ValueError, match="segment"):
        BayesOdp().fit(two_lines, "paid")
