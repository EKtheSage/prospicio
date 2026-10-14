"""Gesmann and Morris' compartmental model 1 (docs/design/reserving-compartmental.md).

The parity with the paper's brms code in Stan and its printed posterior is
validation/tests/reserving_compartmental.rs; these tests check the
binding: the priors and their preset, the settings, the fit's fields and
frames, the prior predictive check, replay by seed, the diagnostics
warning and the errors, on company 337 workers' compensation as the
papers fit it (accident years 1988 to 1997 at calendar years up to 1996,
outstanding = incurred - paid, direct premium).
"""

import csv
import math
import warnings
from pathlib import Path

import pytest

from prospicio.distributions import PredictiveDistribution
from prospicio.models import Elpd
from prospicio.reserving import (
    Compartmental,
    CompartmentalFit,
    CompartmentalPriors,
    Triangle,
)

VALIDATION = Path(__file__).resolve().parents[2] / "validation"
GM = CompartmentalPriors.gesmann_morris()

# The default chains (4 x 2,000 draws) take about 5 seconds in a release
# build and pass every diagnostic on company 337 at seed 0.
QUICK = dict(n_sims=2000)


def premium_1997():
    path = VALIDATION / "data" / "clrd_wkcomp_337.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return next(float(r["premium"]) for r in rows if r["origin"] == "1997")


@pytest.fixture(scope="module")
def tri():
    path = VALIDATION / "data" / "clrd_wkcomp_337.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    rows = [r for r in rows if int(r["origin"]) + int(r["development"]) // 12 - 1 <= 1996]
    assert len(rows) == 45
    nan = float("nan")
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows] + [1997],
        development=[int(r["development"]) for r in rows] + [12],
        values={
            "paid": [float(r["paid"]) for r in rows] + [nan],
            "incurred": [float(r["incurred"]) for r in rows] + [nan],
            "premium": [float(r["premium"]) for r in rows] + [premium_1997()],
        },
    )


@pytest.fixture(scope="module")
def fit(tri):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the defaults pass every diagnostic
        return Compartmental(GM, **QUICK).fit(tri, "paid", "premium", incurred="incurred")


def test_the_preset_and_replace():
    p = CompartmentalPriors.gesmann_morris()
    assert (p.k_er, p.k_p, p.rlr, p.rrf) == ((3.0, 0.1), (1.0, 0.1), (0.7, 0.2), (0.8, 0.1))
    assert p.rlr_effect_sd == ("student_t", 10.0, 0.2)
    assert p.rrf_effect_sd == ("student_t", 10.0, 0.1)
    assert p.lkj_eta == 1.0
    assert p.log_sigma == [("student_t", 1.0, 0.0, 1000.0)] * 2
    assert p == GM
    q = p.replace(k_er=(2.0, 0.3), rrf_effect_sd=("normal", 0.5),
                  log_sigma=[("normal", 8.0, 2.0), ("student_t", 3, 7, 2)])
    assert q.k_er == (2.0, 0.3) and q.rrf_effect_sd == ("normal", 0.5)
    assert q.log_sigma == [("normal", 8.0, 2.0), ("student_t", 3.0, 7.0, 2.0)]
    assert q.k_p == p.k_p and q != p
    assert repr(p).startswith("CompartmentalPriors(k_er=(3.0, 0.1), k_p=(1.0, 0.1)")
    built = CompartmentalPriors(
        k_er=(3.0, 0.1), k_p=(1.0, 0.1), rlr=(0.7, 0.2), rrf=(0.8, 0.1),
        rlr_effect_sd=("student_t", 10, 0.2), rrf_effect_sd=("student_t", 10, 0.1),
        lkj_eta=1.0, log_sigma=[("student_t", 1, 0, 1000)] * 2)
    assert built == p


def test_bad_priors():
    with pytest.raises(TypeError):
        CompartmentalPriors(k_er=(3.0, 0.1))  # every prior is required
    with pytest.raises(ValueError, match="k_er must be"):
        GM.replace(k_er=(-1.0, 0.1))
    with pytest.raises(ValueError, match="rlr_effect_sd must be"):
        GM.replace(rlr_effect_sd=("cauchy", 1.0))
    with pytest.raises(ValueError, match="log_sigma must hold two"):
        GM.replace(log_sigma=[("normal", 0.0, 1.0)])
    with pytest.raises(ValueError, match=r"log_sigma\[1\] must be"):
        GM.replace(log_sigma=[("normal", 0.0, 1.0), ("normal", 0.0)])


def test_priors_are_required():
    with pytest.raises(TypeError):
        Compartmental()
    with pytest.raises(TypeError):
        Compartmental(None)


def test_settings_and_defaults():
    m = Compartmental(GM)
    assert m.priors == GM
    assert (m.likelihood, m.scale, m.paid_response, m.fitted) == (
        "normal", "amount", "cumulative", "paid_and_outstanding")
    assert (m.rate_order, m.start) == ("free", "data")
    assert (m.chains, m.tune, m.draws, m.n_sims, m.seed) == (4, 1000, 2000, 10_000, 0)
    assert (m.target_accept, m.max_depth) == (0.99, 10)
    m = Compartmental(GM, rate_order="Ordered", start="prior_centre", seed=3, chains=2)
    assert (m.rate_order, m.start, m.seed, m.chains) == ("ordered", "prior_centre", 3, 2)
    assert repr(m).startswith('Compartmental(likelihood="normal", scale="amount"')
    assert 'rate_order="ordered"' in repr(m)
    with pytest.raises(ValueError, match="unknown rate_order"):
        Compartmental(GM, rate_order="sorted")
    with pytest.raises(ValueError, match="unknown fitted"):
        Compartmental(GM, fitted="everything")
    with pytest.raises(ValueError, match="n_sims must be positive"):
        Compartmental(GM, n_sims=0)


@pytest.mark.parametrize("setting", [
    dict(fitted="paid_and_incurred"),
    dict(fitted="paid_only"),
    dict(fitted="incurred_only"),
])
def test_settings_not_yet_built(tri, setting):
    m = Compartmental(GM, **setting)  # accepted and named
    assert getattr(m, next(iter(setting))) == next(iter(setting.values()))
    with pytest.raises(ValueError, match="not yet built"):
        m.fit(tri, "paid", "premium", incurred="incurred")


def test_fit_errors(tri):
    m = Compartmental(GM)
    with pytest.raises(ValueError, match="exactly one of outstanding"):
        m.fit(tri, "paid", "premium")
    with pytest.raises(ValueError, match="exactly one of outstanding"):
        m.fit(tri, "paid", "premium", outstanding="incurred", incurred="incurred")
    with pytest.raises(ValueError):
        m.fit(tri, "paid", "no_such_column", incurred="incurred")
    with pytest.raises(ValueError, match="lkj_eta"):
        Compartmental(GM.replace(lkj_eta=0.0)).fit(tri, "paid", "premium", incurred="incurred")


def test_prior_predictive(tri):
    paths = Compartmental(GM, seed=4).prior_predictive(tri, "premium", n_sims=500)
    assert set(paths) == {"outstanding", "paid"}
    for pd in paths.values():
        assert isinstance(pd, PredictiveDistribution)
        assert pd.dims == ["origin", "age"] and pd.n_sims == 500
        assert len(pd.components()) == 100  # the 10 x 10 square
    # The mean paid of 1988 at 120 months over its premium sits about the
    # preset's ultimate loss ratio, 0.7 x 0.8.
    paid_120 = paths["paid"].marginal(("1988", 120)).mean()
    assert 0.3 < paid_120 / 104_437 < 0.9
    again = Compartmental(GM, seed=4).prior_predictive(tri, "premium", n_sims=500)
    assert again["paid"].draw_matrix() == paths["paid"].draw_matrix()
    with pytest.raises(ValueError):
        Compartmental(GM).prior_predictive(tri, "premium", n_sims=0)


def test_fit_fields(fit):
    assert isinstance(fit, CompartmentalFit)
    years = [str(y) for y in range(1988, 1998)]
    assert fit.origins == years
    assert fit.population_origins == ["1997"]
    assert fit.premium[0] == 104_437.0 and len(fit.premium) == 10
    assert fit.latest_paid[-1] == 0.0 and fit.latest_paid[0] > 0
    assert fit.rate_swaps == 0

    r = fit.reserves
    assert isinstance(r, PredictiveDistribution)
    assert r.dims == ["origin"] and r.n_sims == 2000
    assert r.components() == [(y,) for y in years]
    assert r.provenance()["model"] == "compartmental_gm1"
    # Seed 0 at the defaults gives 173,390 (sd 10,167), Stan 173,341.
    assert r.mean() == pytest.approx(173_390, rel=0.02)

    for cells in (fit.paid, fit.outstanding):
        assert cells.dims == ["origin", "age"] and cells.n_sims == 2000
        assert cells.components()[0] == ("1988", 120)
        assert len(cells.components()) == 55  # 1 + 2 + ... + 10
    # The reserve of an origin is its paid at 120 months less its latest paid.
    paid = fit.paid
    k = paid.components().index(("1990", 120))
    j = r.components().index(("1990",))
    first = paid.draw_matrix()[0][k] - fit.latest_paid[2]
    assert r.draw_matrix()[0][j] == pytest.approx(first, rel=1e-12)

    names = fit.parameter_names
    assert names[:6] == ["k_er", "k_p", "RLR", "RRF", "sigma_os", "sigma_pd"]
    assert "b_ker" in names and "rho" in names and names[-1] == "log_sigma_pd"
    assert fit.chains == 4
    assert len(fit.draws) == 8000 and all(len(row) == len(names) for row in fit.draws)
    assert len(fit.ultimate_loss_ratio) == 10
    assert all(0.3 < u < 1.5 for u in fit.ultimate_loss_ratio)

    d = fit.diagnostics
    assert d["warnings"] == [] and d["divergences"] == 0 and d["max_rhat"] < 1.01
    assert set(d) == {"max_rhat", "min_ess_bulk", "min_ess_tail", "divergences", "warnings"}
    assert repr(fit).startswith("CompartmentalFit(origins=10, draws=8000, n_sims=2000")


def test_summary_and_loo(fit):
    s = fit.summary()
    assert list(s.columns) == ["parameter", "mean", "sd", "q05", "q50", "q95",
                               "rhat", "ess_bulk", "ess_tail"]
    b = s.set_index("parameter").loc["b_kp"]
    # Stan: b_kp -8.488 (sd 0.358).
    assert b["mean"] == pytest.approx(-8.488, abs=0.1)
    loo = fit.loo()
    assert isinstance(loo, Elpd)
    assert len(loo.pointwise) == 90  # 45 outstanding cells, then 45 paid
    # Stan: elpd_loo -811.0.
    assert loo.elpd == pytest.approx(-811.0, abs=2)
    by = fit.loo_by_type()
    assert by["n_outstanding"] == 45
    assert by["elpd_outstanding"] + by["elpd_paid"] == pytest.approx(loo.elpd, rel=1e-12)
    assert by["elpd_outstanding"] == pytest.approx(sum(loo.pointwise[:45]), rel=1e-12)


def test_frames(fit):
    f = fit.to_frame()
    assert list(f.columns) == ["origin", "premium", "latest_paid", "mean", "std_dev",
                               "ultimate_loss_ratio"]
    # Every origin has paid to come: 1988 is observed up to 108 months.
    assert len(f) == 10 and (f["mean"] > 0).all()
    assert f["latest_paid"].iloc[-1] == 0.0
    t = fit.totals_frame()
    assert t["mean"].iloc[0] == pytest.approx(fit.reserves.mean(), rel=1e-12)
    for frame in (fit.paid_frame(), fit.outstanding_frame()):
        assert list(frame.columns) == ["origin", "development", "mean", "std_dev",
                                       "q05", "q50", "q95"]
        assert len(frame) == 55
        assert (frame["q05"] <= frame["q50"]).all() and (frame["q50"] <= frame["q95"]).all()


def test_replay_by_seed(tri):
    small = dict(chains=2, tune=100, draws=100, n_sims=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = Compartmental(GM, seed=7, **small).fit(tri, "paid", "premium", incurred="incurred")
        b = Compartmental(GM, seed=7, **small).fit(tri, "paid", "premium", incurred="incurred")
        c = Compartmental(GM, seed=8, **small).fit(tri, "paid", "premium", incurred="incurred")
    assert a.draws == b.draws
    assert a.reserves.draw_matrix() == b.reserves.draw_matrix()
    assert a.draws != c.draws


def test_failed_diagnostics_warn(tri):
    m = Compartmental(GM, chains=2, tune=20, draws=20, n_sims=100)
    with pytest.warns(UserWarning, match="Compartmental: the MCMC diagnostics failed"):
        fit = m.fit(tri, "paid", "premium", incurred="incurred")
    assert fit.diagnostics["warnings"]
    assert not math.isnan(fit.reserves.mean())
