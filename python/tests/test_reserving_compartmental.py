"""Gesmann and Morris' compartmental models 1 and 2 (docs/design/reserving-compartmental.md).

The parity with the paper's brms code in Stan and its printed posterior is
validation/tests/reserving_compartmental.rs (model 1) and
reserving_compartmental_gm2.rs (model 2); these tests check the
binding: the priors and their presets, the settings, the fit's fields and
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
    dict(fitted="paid_only"),
    dict(fitted="incurred_only"),
])
def test_settings_not_yet_built(tri, setting):
    m = Compartmental(GM, **setting)  # accepted and named
    assert getattr(m, next(iter(setting))) == next(iter(setting.values()))
    with pytest.raises(ValueError, match="not yet built"):
        m.fit(tri, "paid", "premium", incurred="incurred")


def test_paid_and_incurred(tri):
    # Incurred replaces outstanding as the second response, from the
    # incurred column or from outstanding plus paid (the same data).
    short = dict(chains=2, tune=100, draws=100, n_sims=200, seed=5)
    m = Compartmental(GM, fitted="paid_and_incurred", **short)
    assert m.fitted == "paid_and_incurred"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = m.fit(tri, "paid", "premium", incurred="incurred")
    assert a.reserves.provenance()["model"] == "compartmental"
    assert a.reserves.n_components == 10
    # the future incurred cells are above the future paid in the mean
    inc = a.outstanding.marginal(("1990", 120)).mean()
    assert inc > 0.9 * a.paid.marginal(("1990", 120)).mean()


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


# ---------------------------------------------------------------------------
# Model 2 (PR C): the log-normal likelihood on loss ratios with incremental
# paid, and the hierarchy of CompartmentalPriors.gesmann_morris_2().
# ---------------------------------------------------------------------------

GM2 = CompartmentalPriors.gesmann_morris_2()
HALF = ("student_t", 10.0, 0.3)


def test_the_model_2_preset():
    p = GM2
    assert (p.k_er, p.k_p, p.rlr, p.rrf) == ((3.0, 0.1), (1.0, 0.1), (0.7, 0.2), (0.8, 0.1))
    assert p.rlr_effect_sd == ("student_t", 10.0, 0.7)
    assert p.rrf_effect_sd == ("student_t", 10.0, 0.5)
    assert p.ker_effect_sd == HALF and p.kp_effect_sd == HALF
    assert p.development_effect_sd == [HALF, HALF, ("student_t", 10.0, 0.7),
                                       ("student_t", 10.0, 0.5)]
    assert p.log_sigma == [("normal", math.log(0.2), 0.2)] * 2
    # Model 1 has none of the new effects.
    assert GM.ker_effect_sd is None and GM.kp_effect_sd is None
    assert GM.development_effect_sd == [None] * 4
    built = CompartmentalPriors(
        k_er=(3.0, 0.1), k_p=(1.0, 0.1), rlr=(0.7, 0.2), rrf=(0.8, 0.1),
        rlr_effect_sd=("student_t", 10, 0.7), rrf_effect_sd=("student_t", 10, 0.5),
        lkj_eta=1.0, log_sigma=[("normal", math.log(0.2), 0.2)] * 2,
        ker_effect_sd=HALF, kp_effect_sd=HALF,
        development_effect_sd=[HALF, HALF, ("student_t", 10, 0.7), ("student_t", 10, 0.5)])
    assert built == p
    assert 'kp_effect_sd=("student_t", 10.0, 0.3)' in repr(p)
    assert repr(GM).endswith("ker_effect_sd=None, kp_effect_sd=None, development_effect_sd=None)")


def test_replace_the_effects():
    # None keeps an effect, False removes it, a tuple sets it.
    assert GM2.replace(k_er=(2.0, 0.3)).kp_effect_sd == HALF
    q = GM2.replace(ker_effect_sd=False, kp_effect_sd=("normal", 0.2))
    assert q.ker_effect_sd is None and q.kp_effect_sd == ("normal", 0.2)
    # A development_effect_sd replaces all four.
    q = GM2.replace(development_effect_sd=[None, None, ("normal", 0.5), None])
    assert q.development_effect_sd == [None, None, ("normal", 0.5), None]
    one = GM.replace(development_effect_sd=[None, HALF, None, None])
    assert one.development_effect_sd[1] == HALF and one.ker_effect_sd is None
    off = GM2.replace(ker_effect_sd=False, kp_effect_sd=False,
                      development_effect_sd=[None] * 4,
                      rlr_effect_sd=GM.rlr_effect_sd, rrf_effect_sd=GM.rrf_effect_sd,
                      log_sigma=GM.log_sigma)
    assert off == GM


def test_bad_effect_priors():
    with pytest.raises(ValueError, match="ker_effect_sd must be"):
        GM.replace(ker_effect_sd=("cauchy", 1.0))
    with pytest.raises(ValueError, match="ker_effect_sd must be"):
        GM.replace(ker_effect_sd=0.3)
    with pytest.raises(ValueError, match="not True"):
        GM.replace(kp_effect_sd=True)
    with pytest.raises(ValueError, match="development_effect_sd must be None or four"):
        GM.replace(development_effect_sd=[HALF])
    with pytest.raises(ValueError, match=r"development_effect_sd\[2\] must be"):
        GM.replace(development_effect_sd=[None, None, ("normal",), None])


def test_model_2_settings():
    m = Compartmental.model_2(GM2)
    assert m.priors == GM2
    assert (m.likelihood, m.scale, m.paid_response, m.fitted) == (
        "lognormal", "loss_ratio", "incremental", "paid_and_outstanding")
    assert (m.chains, m.tune, m.draws, m.target_accept, m.max_depth) == (4, 2000, 8000, 0.999, 10)
    assert (m.rate_order, m.start, m.n_sims, m.seed) == ("free", "data", 10_000, 0)
    m = Compartmental.model_2(GM2, draws=500, seed=3, rate_order="ordered")
    assert (m.draws, m.seed, m.rate_order, m.likelihood) == (500, 3, "ordered", "lognormal")
    with pytest.raises(TypeError):
        Compartmental.model_2()


@pytest.fixture(scope="module")
def fit2(tri):
    # Shorter chains than model_2's (4 x 2,000 warm-up and 8,000 draws,
    # about a minute, warning-free at seed 0; the Rust validation runs
    # them): about 20 seconds in a release build. They may miss an ESS
    # threshold.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Compartmental.model_2(GM2, tune=1000, draws=2000, n_sims=2000).fit(
            tri, "paid", "premium", incurred="incurred")


def test_model_2_fit(fit2):
    years = [str(y) for y in range(1988, 1998)]
    assert fit2.origins == years and fit2.population_origins == ["1997"]
    assert fit2.reserves.provenance()["model"] == "compartmental_gm2"
    # Stan (4 x 25,000 draws): total reserve 152,242 (sd 13,938).
    assert fit2.reserves.mean() == pytest.approx(152_242, rel=0.03)
    names = fit2.parameter_names
    assert names[:6] == ["k_er", "k_p", "RLR", "RRF", "sigma_os", "sigma_pd"]
    for o in years[:-1]:
        assert f"k_er[{o}]" in names and f"k_p[{o}]" in names and f"RLR[{o}]" in names
    # Development effects at each observed age, 12 to 108 months; 120 has no
    # cell and takes a new effect in each simulation.
    for a in range(12, 120, 12):
        for p in ("ker", "kp", "RLR", "RRF"):
            assert f"dev_{p}[{a}]" in names
    assert "dev_kp[120]" not in names
    for n in ("tau_ker", "tau_kp", "tau_dev_ker", "tau_dev_kp", "tau_dev_RLR", "tau_dev_RRF",
              "rho", "b_kp"):
        assert n in names
    swaps = fit2.cell_rate_swaps
    assert swaps is not None and 0.0 <= swaps <= 1.0
    s = fit2.summary().set_index("parameter")
    # Stan: b_kp -5.064 (sd 1.722).
    assert s.loc["b_kp", "mean"] == pytest.approx(-5.064, abs=0.5)
    # Stan: elpd_loo 204.29, on the loss ratios.
    loo = fit2.loo()
    assert loo.elpd == pytest.approx(204.29, abs=3)
    assert len(loo.pointwise) == 90
    f = fit2.to_frame()
    assert len(f) == 10 and (f["mean"] > 0).all()
    assert len(fit2.paid_frame()) == 55


def test_model_1_has_no_cell_rate_swaps(fit):
    assert fit.cell_rate_swaps is None


@pytest.mark.parametrize("setting", [
    dict(scale="loss_ratio"),
    dict(paid_response="incremental"),
    dict(likelihood="lognormal"),
    dict(likelihood="lognormal", scale="loss_ratio", paid_response="incremental"),
])
def test_mixed_settings_fit(tri, setting):
    # Model 1's hierarchy under each other setting; short chains.
    priors = GM.replace(log_sigma=GM2.log_sigma) if "likelihood" in setting else GM
    m = Compartmental(priors, chains=2, tune=100, draws=100, n_sims=200, **setting)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = m.fit(tri, "paid", "premium", incurred="incurred")
    assert all(math.isfinite(v) for v in fit.reserves.draw_matrix()[0])
    assert fit.cell_rate_swaps is None  # the rates are shared


def small_triangle(paid_88_12=9.6, incurred_88_12=62.7):
    paid = [[paid_88_12, 22.8, 33.3], [7.9, 19.5], [8.7]]
    incurred = [[incurred_88_12, 64.0, 65.6], [62.1, 56.7], [64.0]]
    premium = [104.4, 88.9, 86.0]
    rows = [(1988 + w, 12 * (d + 1), paid[w][d], incurred[w][d], premium[w])
            for w in range(3) for d in range(len(paid[w]))]
    origin, age, pd, inc, prem = map(list, zip(*rows))
    return Triangle.from_long(origin, age, {"paid": pd, "incurred": inc, "premium": prem})


def test_model_2_response_errors():
    m = Compartmental.model_2(GM2, chains=1, tune=10, draws=10, n_sims=10)
    # An outstanding of 0 has no log.
    zero = small_triangle(incurred_88_12=9.6)
    with pytest.raises(ValueError, match="origin 1988 at development age 12 months has "
                                         "outstanding 0"):
        m.fit(zero, "paid", "premium", incurred="incurred")
    # Incremental paid needs the paid of the age before.
    gap = small_triangle(paid_88_12=float("nan"))
    with pytest.raises(ValueError, match="incremental paid needs the paid of the age before"):
        m.fit(gap, "paid", "premium", outstanding="incurred")


def test_model_2_prior_predictive(tri):
    paths = Compartmental.model_2(GM2, seed=4).prior_predictive(tri, "premium", n_sims=300)
    assert paths["paid"].dims == ["origin", "age"] and paths["paid"].n_sims == 300
    assert len(paths["outstanding"].components()) == 100
    paid_120 = paths["paid"].marginal(("1988", 120)).mean()
    assert 0.2 < paid_120 / 104_437 < 1.5
    again = Compartmental.model_2(GM2, seed=4).prior_predictive(tri, "premium", n_sims=300)
    assert again["paid"].draw_matrix() == paths["paid"].draw_matrix()


def test_one_year(tri, fit):
    from prospicio.reserving import CompartmentalOneYearFit

    model = Compartmental(GM, **QUICK)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        view = model.one_year(
            tri, "paid", "premium", incurred="incurred", refit="reweight_only"
        )
    assert isinstance(view, CompartmentalOneYearFit)
    assert isinstance(view.fit, CompartmentalFit)
    # The lifetime fit is fit's.
    assert view.fit.reserves.draw_matrix() == fit.reserves.draw_matrix()
    assert view.origins == fit.origins and len(view.origins) == 10
    assert isinstance(view.cdr, PredictiveDistribution)
    assert view.cdr.n_sims == 2000 and len(view.cdr.components()) == 10
    assert len(view.pareto_k) == 2000
    assert view.refitted == [] and view.moment_matched == [] and view.unrefitted == []
    assert 0.0 <= view.flagged_share <= 1.0 and view.k_threshold == 0.7
    assert view.cdr.variance() < view.fit.reserves.variance()
    for o, l, r, u in zip(view.origins, view.latest_paid, view.opening_reserve, view.opening_ultimate):
        assert math.isclose(l + r, u)
    frame = view.to_frame()
    assert list(frame.columns) == [
        "origin", "premium", "latest_paid", "opening_ultimate", "opening_reserve",
        "cdr_mean", "cdr_std_dev",
    ]
    totals = view.totals_frame()
    assert "flagged_share" in totals.columns
    assert "CompartmentalOneYearFit(origins=10" in repr(view)
    # Moment matching without refits (max_refits=0) lists and warns about
    # what it leaves.
    small = Compartmental(GM, n_sims=40)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        budget = small.one_year(tri, "paid", "premium", incurred="incurred", max_refits=0)
    assert budget.refitted == []
    flagged = [i for i, k in enumerate(budget.pareto_k) if math.isnan(k) or k > 0.7]
    assert sorted(budget.moment_matched + budget.unrefitted) == flagged
    if budget.unrefitted:
        assert any("beyond max_refits" in str(w.message) for w in record)
    with pytest.raises(ValueError, match="exactly one"):
        model.one_year(tri, "paid", "premium")


def test_one_year_defaults(tri):
    # The user's decisions of 2026-10-16 and 2026-10-21: at most 2,000
    # one-year simulations by default (n_sims sets the count; a smaller
    # count gives the first simulations of a larger one), and model 2's
    # default view refits its flagged simulations with the light refit;
    # light_refit=False reweights only, with a warning that it is
    # approximate.
    model = Compartmental(GM)
    assert model.n_sims == 10_000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        capped = model.one_year(tri, "paid", "premium", incurred="incurred",
                                refit="reweight_only")
        first = model.one_year(tri, "paid", "premium", incurred="incurred",
                               refit="reweight_only", n_sims=300)
    assert len(capped.pareto_k) == 2000 and capped.cdr.n_sims == 2000
    assert first.pareto_k == capped.pareto_k[:300]
    assert capped.approximate is None
    assert dict(capped.cdr.provenance()["parameters"])["one_year_sims"] == "2000"
    with pytest.raises(ValueError, match="n_sims"):
        model.one_year(tri, "paid", "premium", incurred="incurred", n_sims=0)
    # Model 2 with short chains (they may miss a diagnostic threshold).
    short = Compartmental.model_2(GM2, chains=2, tune=200, draws=200, target_accept=0.9)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        view = short.one_year(tri, "paid", "premium", incurred="incurred", n_sims=50,
                              light_refit=False)
    assert len(view.pareto_k) == 50
    assert view.refitted == [] and view.moment_matched == [] and view.unrefitted == []
    assert "approximate" in view.approximate
    assert dict(view.cdr.provenance()["parameters"])["refit"] == "ReweightOnly"
    assert any("approximate" in str(w.message) for w in record)
    # Refits stay a choice: an explicit mode carries no warning.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        chosen = short.one_year(tri, "paid", "premium", incurred="incurred", n_sims=50,
                                refit="reweight_only")
    assert chosen.approximate is None
    assert not any("approximate" in str(w.message) for w in record)
    assert chosen.pareto_k == view.pareto_k
    # Model 2's default is the light refit: no warning, a few simulations.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        light = short.one_year(tri, "paid", "premium", incurred="incurred", n_sims=2)
    assert light.approximate is None
    assert not any("approximate" in str(w.message) for w in record)
    assert len(light.refitted) == len(light.pareto_k) - len(light.unrefitted) >= 1
    params = dict(light.cdr.provenance()["parameters"])
    assert params["refit"] == "RefitFlagged" and params["light_refit"] == "4x150+250"


def test_light_refit(tri):
    # A warm-started, shorter refit (decision 24), model 2's default: it
    # refits its flagged simulations and is not approximate.
    short = Compartmental.model_2(GM2, chains=2, tune=200, draws=200, target_accept=0.9)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        view = short.one_year(tri, "paid", "premium", incurred="incurred", n_sims=4,
                              light_refit=(2, 60, 60))
    assert view.approximate is None
    assert not any("approximate" in str(w.message) for w in record)
    assert len(view.refitted) == 4
    params = dict(view.cdr.provenance()["parameters"])
    assert params["refit"] == "RefitFlagged" and params["light_refit"] == "2x60+60"
    # An explicit refit mode without a light refit (or False): the fit's
    # own sampler, no such parameter.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # short chains may miss a threshold
        plain = short.one_year(tri, "paid", "premium", incurred="incurred", n_sims=4,
                               refit="reweight_only", light_refit=False)
        # True takes the default sampler.
        default = short.one_year(tri, "paid", "premium", incurred="incurred", n_sims=2,
                                 light_refit=True)
    assert "light_refit" not in dict(plain.cdr.provenance()["parameters"])
    assert dict(default.cdr.provenance()["parameters"])["light_refit"] == "4x150+250"
    for bad in [(2, 60), "yes", (0, 60, 60), (2, 60, 1)]:
        with pytest.raises(ValueError, match="light_refit"):
            short.one_year(tri, "paid", "premium", incurred="incurred", n_sims=2,
                           light_refit=bad)
