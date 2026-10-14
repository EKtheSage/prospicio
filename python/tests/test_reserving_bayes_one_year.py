"""BayesOdp.one_year and one_year_segments, the one-year view of the Bayesian
ODP chain ladder by PSIS reweighting (docs/design/reserving-bayes.md,
decision 16).

The comparison with the ODP bootstrap's one-year view over seeds and the
reweighting against refits are in
validation/tests/reserving_bayes_one_year.rs; these tests check the binding:
the settings, the fields and frames, the lifetime fit against ``fit``, which
simulations are refitted, segments, replay by seed, the warnings and the
errors.
"""

import csv
import math
import warnings
from pathlib import Path

import pytest

from prospicio.distributions import PredictiveDistribution
from prospicio.reserving import (
    BayesOdp,
    BayesOdpFit,
    BayesOneYearFit,
    BayesOneYearFits,
    ChainLadder,
    ChainLadderFit,
    OdpBootstrap,
    Triangle,
)

VALIDATION = Path(__file__).resolve().parents[2] / "validation"

N_SIMS = 2000

TINY = dict(
    origin=[2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    development=[12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
)
PAID = [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0]


def dataset(name):
    with open(VALIDATION / "data" / f"{name}.csv", newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows],
        development=[int(r["development"]) for r in rows],
        values={"values": [float(r["value"]) for r in rows]},
    )


def tiny(keys=False):
    if not keys:
        return Triangle.from_long(TINY["origin"], TINY["development"], {"paid": PAID})
    return Triangle.from_long(
        TINY["origin"] * 2,
        TINY["development"] * 2,
        {"paid": PAID + [2 * v for v in PAID]},
        keys={"lob": ["Auto"] * 10 + ["Home"] * 10},
    )


def total_draws(dist):
    return [sum(row) for row in dist.draw_matrix()]


def sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


@pytest.fixture(scope="module")
def genins():
    return dataset("genins")


@pytest.fixture(scope="module")
def view(genins):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # GenIns passes every diagnostic
        return BayesOdp(n_sims=N_SIMS, seed=3).one_year(genins, "values", refit="reweight_only")


def test_fields(view, genins):
    assert isinstance(view, BayesOneYearFit)
    assert isinstance(view.fit, BayesOdpFit)
    assert isinstance(view.chain_ladder, ChainLadderFit)
    assert view.origins == [str(y) for y in range(2001, 2011)]
    assert view.development == [12 * k for k in range(1, 11)]
    assert view.latest == view.chain_ladder.latest
    for u, lat, r in zip(view.opening_ultimate, view.latest, view.opening_reserve):
        assert u == pytest.approx(lat + r, rel=1e-12)
    # The opening reserve is the posterior mean of the expected run-off,
    # the mean of the lifetime predictive reserve up to Monte Carlo error.
    assert sum(view.opening_reserve) == pytest.approx(view.fit.reserves.mean(), rel=0.02)

    cdr = view.cdr
    assert isinstance(cdr, PredictiveDistribution)
    assert cdr.n_sims == N_SIMS
    assert cdr.components() == [(o,) for o in view.origins]
    # 2001 is fully developed: no cell in the year, no change.
    assert all(row[0] == 0.0 for row in cdr.draw_matrix())
    # The mean is zero up to Monte Carlo error.
    assert abs(cdr.mean()) < 5 * cdr.variance() ** 0.5 / math.sqrt(N_SIMS)
    assert cdr.variance() ** 0.5 < view.fit.reserves.variance() ** 0.5

    assert len(view.pareto_k) == N_SIMS
    assert view.refitted == [] and view.refit_warnings == 0
    assert view.k_threshold == 0.7
    flagged = sum(1 for k in view.pareto_k if math.isnan(k) or k > 0.7) / N_SIMS
    assert view.flagged_share == pytest.approx(flagged, abs=1e-15)
    # About a quarter of GenIns' simulations are flagged (26% over seeds).
    assert 0.1 < view.flagged_share < 0.45
    assert repr(view).startswith(f"BayesOneYearFit(origins=10, n_sims={N_SIMS}, flagged_share=")


def test_the_lifetime_fit_is_fits(view, genins):
    fit = BayesOdp(n_sims=N_SIMS, seed=3).fit(genins, "values")
    assert view.fit.draws == fit.draws
    assert view.fit.reserves.draw_matrix() == fit.reserves.draw_matrix()


def test_against_the_odp_bootstrap(view, genins):
    # The validation test pins this over 80 seeds (ratio 1.006 on GenIns);
    # here a loose check that the binding returns the same quantity.
    boot = OdpBootstrap(n_sims=N_SIMS, seed=3).one_year(genins, "values", ChainLadder())
    assert view.cdr.variance() ** 0.5 == pytest.approx(boot.cdr.variance() ** 0.5, rel=0.12)


def test_frames(view):
    pd = pytest.importorskip("pandas")
    df = view.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "origin",
        "latest",
        "opening_ultimate",
        "opening_reserve",
        "cdr_mean",
        "cdr_std_dev",
    ]
    assert len(df) == 10
    assert df["opening_reserve"].tolist() == pytest.approx(view.opening_reserve)
    totals = view.totals_frame()
    assert len(totals) == 1
    for col in ["opening_reserve", "scale", "flagged_share", "cdr_mean", "cdr_std_dev"]:
        assert col in totals.columns
    assert totals["flagged_share"][0] == view.flagged_share
    assert totals["scale"][0] == pytest.approx(view.fit.scale)
    assert totals["cdr_std_dev"][0] == pytest.approx(view.cdr.variance() ** 0.5, rel=1e-9)
    assert len(view.development_frame()) == 10


def test_replay_by_seed():
    tri = tiny()
    model = BayesOdp(n_sims=300, seed=11)
    a = model.one_year(tri, "paid", refit="reweight_only")
    b = model.one_year(tri, "paid", refit="reweight_only")
    assert a.cdr.draw_matrix() == b.cdr.draw_matrix()
    assert a.pareto_k == b.pareto_k


def test_refits():
    tri = tiny()
    model = BayesOdp(tune=200, draws=200, n_sims=12, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # short chains may miss a threshold
        only = model.one_year(tri, "paid", refit="reweight_only")
        flagged = model.one_year(tri, "paid", k_threshold=0.3)
        every = model.one_year(tri, "paid", refit="refit_all")
    # The flagged simulations are those above the threshold, refitted or not.
    expected = [i for i, k in enumerate(flagged.pareto_k) if math.isnan(k) or k > 0.3]
    assert flagged.refitted == expected
    assert flagged.k_threshold == 0.3
    assert every.refitted == list(range(12))
    # The years and their Pareto k do not depend on the refits.
    assert only.pareto_k == flagged.pareto_k == every.pareto_k
    # A refitted row is refit_all's, the others reweight_only's.
    rows_only, rows_flagged = only.cdr.draw_matrix(), flagged.cdr.draw_matrix()
    rows_every = every.cdr.draw_matrix()
    for i in range(12):
        assert rows_flagged[i] == (rows_every[i] if i in expected else rows_only[i])


def test_refit_warning():
    # Twenty draws per chain cannot reach the ESS threshold: the fit warns,
    # and so do its refits.
    model = BayesOdp(chains=2, tune=20, draws=20, n_sims=3, seed=1)
    with pytest.warns(UserWarning) as record:
        view = model.one_year(tiny(), "paid", refit="refit_all")
    messages = [str(w.message) for w in record]
    assert any("the MCMC diagnostics failed" in m for m in messages)
    refits = [m for m in messages if "refits in the one-year view" in m]
    assert len(refits) == 1
    assert f"{view.refit_warnings} of 3 refits" in refits[0]
    assert view.refit_warnings > 0


@pytest.fixture(scope="module")
def views():
    model = BayesOdp(tune=300, draws=300, n_sims=1000, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # short chains may miss a threshold
        return model.one_year_segments(tiny(keys=True), "paid", refit="reweight_only")


def test_segments(views):
    assert isinstance(views, BayesOneYearFits)
    assert views.keys == ["lob"]
    assert views.index == ["Auto", "Home"]
    assert views.origins == ["2020", "2021", "2022", "2023"] * 2
    assert views.development == [12, 24, 36, 48]
    assert views.cdr.dims == ["lob", "origin"]
    assert len(views.opening_reserve) == 8
    assert views.latest == [170.0, 180.0, 175.0, 130.0, 340.0, 360.0, 350.0, 260.0]
    assert views.k_threshold == 0.7
    assert repr(views).startswith("BayesOneYearFits(segments=2, origins=8, n_sims=1000")
    for field in ["scale", "pareto_k", "refitted", "refit_warnings", "flagged_share"]:
        with pytest.raises(ValueError, match="single-segment"):
            getattr(views, field)
    with pytest.raises(ValueError, match="single-segment"):
        views.diagnostics

    home = views.segment(lob="Home")
    assert isinstance(home, BayesOneYearFits)
    assert home.cdr.dims == ["lob", "origin"]
    assert len(home.pareto_k) == 1000
    assert home.refitted == []
    assert home.scale > 0
    assert set(home.diagnostics) >= {"max_rhat", "warnings"}
    assert repr(home).startswith("BayesOneYearFits(origins=4, n_sims=1000, flagged_share=")


def test_segment_frames(views):
    pytest.importorskip("pandas")
    df = views.to_frame()
    assert list(df.columns[:2]) == ["lob", "origin"]
    assert len(df) == 8
    totals = views.totals_frame()
    assert totals["lob"].tolist() == ["Auto", "Home"]
    assert "flagged_share" in totals.columns and "scale" in totals.columns
    assert totals["flagged_share"][1] == views.segment(lob="Home").flagged_share
    diag = views.diagnostics_frame()
    assert diag["lob"].tolist() == ["Auto", "Home"]
    assert len(views.development_frame()) == 8


def test_rank_correlation():
    tri = tiny(keys=True)
    model = BayesOdp(
        tune=300,
        draws=300,
        n_sims=2000,
        seed=6,
        dependence="rank_correlation",
        spearman=[[1.0, 0.8], [0.8, 1.0]],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        views = model.one_year_segments(tri, "paid", refit="reweight_only")
    auto = total_draws(views.segment(lob="Auto").cdr)
    home = total_draws(views.segment(lob="Home").cdr)
    rx = sorted(range(len(auto)), key=auto.__getitem__)
    ry = sorted(range(len(home)), key=home.__getitem__)
    ra, rh = [0] * len(auto), [0] * len(home)
    for k, i in enumerate(rx):
        ra[i] = k
    for k, i in enumerate(ry):
        rh[i] = k
    m = (len(ra) - 1) / 2
    rho = sum((a - m) * (b - m) for a, b in zip(ra, rh)) / math.sqrt(
        sum((a - m) ** 2 for a in ra) * sum((b - m) ** 2 for b in rh)
    )
    assert rho == pytest.approx(0.8, abs=0.05)
    assert sd(auto) > 0 and sd(home) > 0


def test_errors(genins):
    model = BayesOdp(n_sims=10)
    with pytest.raises(ValueError, match="refit must be"):
        model.one_year(genins, "values", refit="sometimes")
    with pytest.raises(ValueError, match="k_threshold"):
        model.one_year(genins, "values", k_threshold=float("nan"))
    with pytest.raises(ValueError):
        model.one_year(tiny(keys=True), "paid")
    with pytest.raises(ValueError, match="(?i)synchroni"):
        BayesOdp(n_sims=10, dependence="synchronized").one_year_segments(tiny(keys=True), "paid")
