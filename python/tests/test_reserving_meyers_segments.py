"""Meyers.fit_segments, Meyers' CRC, CSR and CAY over segments
(docs/design/reserving-bayes.md, decisions 11 and 13).

The seed sweep and the diagonal backtest are in
validation/tests/reserving_meyers_segments.rs; these tests check the
binding: settings, the joint ultimates and reserves and the frames, one
segment against ``fit``, independence and rank correlation between lines,
replay by seed, the diagnostics warning and the errors, on the CLRD lines
(paid losses and net earned premium).
"""

import csv
import warnings
from pathlib import Path

import pytest

from prospicio.distributions import PredictiveDistribution
from prospicio.models import Elpd
from prospicio.reserving import ChainLadderFit, Meyers, MeyersFits, Triangle

VALIDATION = Path(__file__).resolve().parents[2] / "validation"

N_SIMS = 4000

# Tolerances from seeds 0 to 59 at 4,000 simulations (each at least 4 sds
# and 1.25 times the largest deviation there): a segment against `fit` of
# that segment alone, logelr's posterior mean (sd 0.0001), the mean total
# reserve (sd 0.18%) and elpd_loo (sd 0.52, largest 1.42); twin lines' mean
# totals (sd 0.19%).
LOGELR_TOL = 0.001
TOTAL_TOL = 0.01
LOO_TOL = 2.5
TWIN_TOL = 0.01


def clrd(lines):
    """The lines of clrd_lines.csv in ``lines`` ((line, name) pairs) as the
    segments of a ``lob`` key, with columns ``paid`` and ``premium``."""
    with open(VALIDATION / "data" / "clrd_lines.csv", newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    keys, origin, age, paid, premium = [], [], [], [], []
    for line, name in lines:
        for r in rows:
            if r["lob"] == line:
                keys.append(name)
                origin.append(int(r["origin"]))
                age.append(int(r["development"]))
                paid.append(float(r["paid"]))
                premium.append(float(r["premium"]))
    return Triangle.from_long(
        origin, age, {"paid": paid, "premium": premium}, keys={"lob": keys}
    )


def line_totals(pd):
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
def two_lines():
    return clrd([("comauto", "comauto"), ("wkcomp", "wkcomp")])


@pytest.fixture(scope="module")
def fits(two_lines):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # both lines pass every diagnostic
        return Meyers(n_sims=N_SIMS, seed=5).fit_segments(two_lines, "paid", "premium")


def test_settings():
    m = Meyers()
    assert m.dependence == "independent" and m.spearman is None
    assert "dependence" not in repr(m)
    m = Meyers("csr", dependence="rank_correlation", spearman=[[1.0, 0.5], [0.5, 1.0]])
    assert m.dependence == "rank_correlation"
    assert m.spearman == [[1.0, 0.5], [0.5, 1.0]]
    assert repr(m).endswith('dependence="rank_correlation", spearman=[[1.0, 0.5], [0.5, 1.0]])')
    assert Meyers(dependence="synchronized").dependence == "synchronized"


def test_fields_and_frames(fits):
    assert isinstance(fits, MeyersFits)
    assert fits.model == "crc"
    assert fits.keys == ["lob"]
    assert fits.index == ["comauto", "wkcomp"]
    assert isinstance(fits.chain_ladder, ChainLadderFit)
    assert len(fits.origins) == 20 and fits.origins[:2] == ["1988", "1989"]
    assert fits.development == [12 * k for k in range(1, 11)]
    assert len(fits.premium) == 20 and fits.premium[0] == 913636.0

    r, u = fits.reserves, fits.ultimates
    for pd in (r, u):
        assert isinstance(pd, PredictiveDistribution)
        assert pd.dims == ["lob", "origin"]
        assert pd.n_sims == N_SIMS and pd.n_components == 20
        assert pd.provenance()["model"].startswith("meyers")
    # Each simulation's ultimates less its reserves are the latest values.
    latest = fits.to_frame()["latest"].tolist()
    for ur, rr in zip(u.draw_matrix()[:50], r.draw_matrix()[:50]):
        for a, b, c in zip(ur, rr, latest):
            assert a - b == pytest.approx(c, rel=1e-9)

    long = fits.to_frame()
    assert list(long.columns) == [
        "lob", "origin", "latest", "ultimate", "reserve", "premium", "ulr", "mean", "std_dev",
        "ultimate_mean", "ultimate_std_dev", "ulr_mean",
    ]
    assert list(long["ulr"]) == [u / p for u, p in zip(long["ultimate"], long["premium"])]
    assert list(long["ulr_mean"]) == [
        u / p for u, p in zip(long["ultimate_mean"], long["premium"])
    ]
    assert len(long) == 20
    assert long["mean"].sum() == pytest.approx(r.mean(), rel=1e-9)
    assert long["ultimate_mean"].sum() == pytest.approx(u.mean(), rel=1e-9)
    totals = fits.totals_frame()
    assert list(totals["lob"]) == ["comauto", "wkcomp"]
    assert "premium" in totals.columns and "ultimate_mean" in totals.columns
    assert totals["mean"].sum() == pytest.approx(r.mean(), rel=1e-9)
    assert totals["premium"].sum() == pytest.approx(sum(fits.premium), rel=1e-12)
    assert list(totals["ulr_mean"]) == list(totals["ultimate_mean"] / totals["premium"])
    assert len(fits.development_frame()) == 20

    d = fits.diagnostics_frame()
    assert list(d.columns) == [
        "lob", "max_rhat", "min_ess_bulk", "min_ess_tail", "divergences", "warnings",
    ]
    assert list(d["lob"]) == ["comauto", "wkcomp"]
    assert (d["max_rhat"] < 1.01).all() and (d["warnings"] == "").all()
    assert repr(fits).startswith('MeyersFits(model="crc", segments=2, origins=20, n_sims=4000')


def test_single_segment_fields_need_one_segment(fits):
    for field in ["diagnostics", "parameter_names", "draws", "chains"]:
        with pytest.raises(ValueError, match=r"single-segment fit.*segment\(\.\.\.\)"):
            getattr(fits, field)
    with pytest.raises(ValueError, match="single-segment fit"):
        fits.summary()
    with pytest.raises(ValueError, match="single-segment fit"):
        fits.loo()


def test_one_segment(fits):
    one = fits.segment(lob="wkcomp")
    assert isinstance(one, MeyersFits)
    assert one.index == ["wkcomp"]
    assert one.reserves.dims == ["lob", "origin"] and one.reserves.n_components == 10
    # Its part of the joint ultimates and reserves.
    assert one.reserves.draw_matrix() == [row[10:] for row in fits.reserves.draw_matrix()]
    assert one.ultimates.draw_matrix() == [row[10:] for row in fits.ultimates.draw_matrix()]
    assert one.premium == fits.premium[10:]

    # The posterior of one segment: the same model as ``fit``, with its own
    # seeds, so the same premium and a close posterior but other draws.
    alone = Meyers(n_sims=N_SIMS, seed=5).fit(clrd([("wkcomp", "wkcomp")]), "paid", "premium")
    assert one.premium == alone.premium
    assert one.parameter_names == alone.parameter_names
    assert one.chains == 4 and len(one.draws) == 8000
    assert one.draws != alone.draws
    assert one.diagnostics["warnings"] == []
    s = one.summary()
    assert list(s["parameter"]) == one.parameter_names
    a = alone.summary()
    assert s.loc[0, "mean"] == pytest.approx(a.loc[0, "mean"], abs=LOGELR_TOL)
    assert one.totals_frame()["mean"][0] == pytest.approx(
        alone.totals_frame()["mean"][0], rel=TOTAL_TOL
    )
    loo = one.loo()
    assert isinstance(loo, Elpd) and len(loo.pointwise) == 55
    assert loo.elpd == pytest.approx(alone.loo().elpd, abs=LOO_TOL)


def test_independent_lines_are_independent():
    # The same line under two names: shared random numbers would make their
    # totals almost perfectly rank correlated. 4.5 / sqrt(n_sims) is the
    # tolerance of validation/tests/reserving_meyers_segments.rs (sd 0.0083
    # at 10,000 simulations over 60 seeds, about 1 / sqrt(n_sims)).
    twins = clrd([("wkcomp", "a"), ("wkcomp", "b")])
    fits = Meyers(n_sims=N_SIMS, seed=11).fit_segments(twins, "paid", "premium")
    a, b = line_totals(fits.reserves)
    assert abs(spearman(a, b)) < 4.5 / N_SIMS**0.5
    assert sum(a) / sum(b) == pytest.approx(1.0, rel=TWIN_TOL)


def test_rank_correlation(two_lines):
    target = 0.5
    fits = Meyers(
        "csr", n_sims=N_SIMS, seed=5, dependence="rank_correlation",
        spearman=[[1.0, target], [target, 1.0]],
    ).fit_segments(two_lines, "paid", "premium")
    a, b = line_totals(fits.reserves)
    # Iman-Conover reaches the target within 0.0081 at 10,000 simulations
    # over 60 seeds (sd at most 0.0030); 0.03 allows for 4,000.
    assert spearman(a, b) == pytest.approx(target, abs=0.03)
    # The ultimates move with the reserves.
    ua, ub = line_totals(fits.ultimates)
    assert spearman(ua, a) == pytest.approx(1.0, abs=1e-12)
    assert spearman(ub, b) == pytest.approx(1.0, abs=1e-12)


def test_cay_and_replay_by_seed(two_lines):
    settings = dict(chains=2, tune=200, draws=200, n_sims=500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = Meyers("cay", seed=7, **settings).fit_segments(two_lines, "paid", "premium")
        b = Meyers("cay", seed=7, **settings).fit_segments(two_lines, "paid", "premium")
        c = Meyers("cay", seed=8, **settings).fit_segments(two_lines, "paid", "premium")
    assert a.model == "cay"
    assert a.segment(lob="comauto").parameter_names[-11] == "rho"
    assert a.reserves.draw_matrix() == b.reserves.draw_matrix()
    assert a.segment(lob="comauto").draws == b.segment(lob="comauto").draws
    assert a.reserves.draw_matrix() != c.reserves.draw_matrix()


def test_failed_diagnostics_warn_with_segment_label(two_lines):
    # 2 x 10 kept draws cannot reach a bulk ESS of 200 (100 per chain).
    with pytest.warns(UserWarning, match=r"Meyers: the MCMC diagnostics failed.*comauto: .*ESS"):
        f = Meyers(chains=2, tune=50, draws=10, n_sims=100, seed=1).fit_segments(
            two_lines, "paid", "premium"
        )
    d = f.diagnostics_frame()
    assert (d["warnings"] != "").all()
    assert f.reserves.n_sims == 100


def test_errors(two_lines):
    with pytest.raises(ValueError, match="dependence must be"):
        Meyers(dependence="copula")
    with pytest.raises(ValueError, match="needs a spearman matrix"):
        Meyers(dependence="rank_correlation")
    with pytest.raises(ValueError, match="only with dependence"):
        Meyers(spearman=[[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="square"):
        Meyers(dependence="rank_correlation", spearman=[[1.0, 0.5]])
    # Synchronized is refused before any sampling.
    with pytest.raises(ValueError, match="Synchronized.*no residuals"):
        Meyers(dependence="synchronized").fit_segments(two_lines, "paid", "premium")
    # A Spearman matrix of the wrong size for the segments.
    three = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError):
        Meyers(
            chains=2, tune=50, draws=50, n_sims=100,
            dependence="rank_correlation", spearman=three,
        ).fit_segments(two_lines, "paid", "premium")
    with pytest.raises(ValueError, match="missing"):
        Meyers().fit_segments(two_lines, "missing", "premium")
    with pytest.raises(ValueError, match="missing"):
        Meyers().fit_segments(two_lines, "paid", "missing")
    # fit still takes one segment only.
    with pytest.raises(ValueError, match="segment"):
        Meyers().fit(two_lines, "paid", "premium")
