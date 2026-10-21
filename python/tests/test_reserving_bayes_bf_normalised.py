"""BayesBfNormalised, the Bayesian Bornhuetter-Ferguson with a normalised
pattern (docs/design/reserving-bayes.md, decision 23).

The limits, the contrast with BayesBf and the one-year view are checked in
validation/tests/reserving_bayes_bf_normalised.rs; these tests check the
binding: settings, the fit's fields and frames, the two limits of the prior's
strength and the prior's reach, replay by seed, the diagnostics warning, the
one-year view and the errors, on EVW's triangle (validation/data/evw2012.csv)
with its prior ultimates as the exposure.
"""

import csv
import math
import warnings
from pathlib import Path

import pytest

from prospicio.distributions import PredictiveDistribution
from prospicio.models import Elpd
from prospicio.reserving import (
    BayesBf,
    BayesBfNormalised,
    BayesBfNormalisedFit,
    BayesOdp,
    BayesOneYearFit,
    ChainLadderFit,
    Triangle,
)

VALIDATION = Path(__file__).resolve().parents[2] / "validation"

QUICK = dict(n_sims=2000)


def evw():
    with open(VALIDATION / "data" / "evw2012.csv", newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows],
        development=[int(r["development"]) for r in rows],
        values={k: [float(r[k]) for r in rows] for k in ("paid", "prior")},
    )


@pytest.fixture(scope="module")
def tri():
    return evw()


@pytest.fixture(scope="module")
def fit(tri):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # EVW's triangle passes every diagnostic
        return BayesBfNormalised(seed=3, **QUICK).fit(tri, "paid", "prior")


def test_settings_and_defaults():
    m = BayesBfNormalised()
    assert (m.apriori, m.cv, m.development_sd) == (1.0, 0.1, 10.0)
    assert m.process == "gamma"
    assert (m.chains, m.tune, m.draws, m.n_sims, m.seed) == (4, 1000, 1000, 10_000, 0)
    assert (m.target_accept, m.max_depth) == (0.95, 10)
    m = BayesBfNormalised(apriori=0.7, cv=math.inf, process="none", chains=3, seed=9)
    assert (m.apriori, m.cv, m.process, m.chains, m.seed) == (0.7, math.inf, "none", 3, 9)
    assert repr(m).startswith("BayesBfNormalised(apriori=0.7, cv=inf, development_sd=10.0")


def test_fit_fields(fit):
    assert isinstance(fit, BayesBfNormalisedFit)
    assert isinstance(fit.chain_ladder, ChainLadderFit)
    assert fit.origins == [str(y) for y in range(2001, 2011)]
    assert fit.development == [12 * k for k in range(1, 11)]
    # EVW (2012) print phi = 14,714.
    assert fit.scale == pytest.approx(14714, rel=1e-3)
    assert fit.exposure[0] == 11653101.0
    assert fit.prior_ultimate == fit.exposure  # apriori 1

    r = fit.reserves
    assert isinstance(r, PredictiveDistribution)
    assert r.n_sims == 2000
    assert r.components() == [(o,) for o in fit.origins]
    assert r.provenance()["model"] == "bayes_bf_normalised"

    # The shares sum to 1; the expected ultimates are near the prior's mean
    # in total (the data inform them, so not equal).
    pattern = fit.pattern
    assert len(pattern) == 10 and sum(pattern) == pytest.approx(1.0, abs=1e-9)
    assert all(p > 0 for p in pattern) and pattern[0] > pattern[-1]
    assert len(fit.expected_ultimate) == 10

    names = fit.parameter_names
    assert names[0] == "alpha[1]" and names[9] == "alpha[10]"
    # The first age's share is the corner: no beta[1].
    assert names[10] == "beta[2]" and names[-1] == "beta[10]"
    assert len(names) == 19
    assert fit.chains == 4
    draws = fit.draws
    assert len(draws) == 4000 and all(len(row) == 19 for row in draws)

    d = fit.diagnostics
    assert set(d) == {"max_rhat", "min_ess_bulk", "min_ess_tail", "divergences", "warnings"}
    assert d["warnings"] == [] and d["divergences"] == 0
    assert repr(fit).startswith("BayesBfNormalisedFit(origins=10, draws=4000, n_sims=2000")


def test_summary_and_frames(fit):
    s = fit.summary()
    assert list(s.columns) == [
        "parameter", "mean", "sd", "q05", "q50", "q95", "rhat", "ess_bulk", "ess_tail",
    ]
    assert list(s["parameter"]) == fit.parameter_names

    long = fit.to_frame()
    assert list(long.columns) == [
        "origin", "latest", "ultimate", "reserve", "exposure", "prior_ultimate", "mean",
        "std_dev",
    ]
    assert len(long) == 10
    assert list(long["prior_ultimate"]) == fit.prior_ultimate
    assert long["mean"].sum() == pytest.approx(fit.reserves.mean(), rel=1e-9)
    totals = fit.totals_frame()
    assert len(totals) == 1
    assert {"exposure", "prior_ultimate", "scale"} <= set(totals.columns)
    assert totals.loc[0, "mean"] == pytest.approx(fit.reserves.mean(), rel=1e-9)
    assert len(fit.development_frame()) == 10


def test_loo(fit):
    loo = fit.loo()
    assert isinstance(loo, Elpd)
    assert len(loo.pointwise) == 55 and len(loo.pareto_k) == 55
    assert math.isfinite(loo.elpd)


def test_limits_and_reach_of_the_prior(tri):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # A flat prior is the Bayesian chain ladder: BayesOdp's reserve
        # within Monte Carlo error, and EVW's 6,049,398 at a = 0.
        flat = BayesBfNormalised(cv=math.inf, development_sd=100, seed=1, **QUICK).fit(
            tri, "paid", "prior"
        )
        odp = BayesOdp(seed=1, **QUICK).fit(tri, "paid")
        # A very tight prior: the shares S / (M + lambda) given the prior
        # ultimates, whose reserve on this triangle is 8,987,282, far above
        # the classical Bornhuetter-Ferguson's 7,356,578 (the pattern follows
        # the ultimates).
        tight = BayesBfNormalised(cv=0.001, development_sd=100, seed=1, **QUICK).fit(
            tri, "paid", "prior"
        )
        mid = BayesBfNormalised(cv=0.1, seed=1, **QUICK).fit(tri, "paid", "prior")
        # The prior is on the ultimate: its level moves the reserve, where
        # BayesBf's does not.
        low = BayesBfNormalised(apriori=0.6, cv=0.05, seed=1, **QUICK).fit(tri, "paid", "prior")
        high = BayesBfNormalised(apriori=1.2, cv=0.05, seed=1, **QUICK).fit(tri, "paid", "prior")
        old_low = BayesBf(apriori=0.6, cv=0.05, seed=1, **QUICK).fit(tri, "paid", "prior")
        old_high = BayesBf(apriori=1.2, cv=0.05, seed=1, **QUICK).fit(tri, "paid", "prior")
    assert flat.reserves.mean() == pytest.approx(odp.reserves.mean(), rel=0.02)
    assert flat.reserves.mean() == pytest.approx(6049398, rel=0.015)
    # The posterior mean of the shares sits 2.5% below their mode.
    assert tight.reserves.mean() == pytest.approx(8987282, rel=0.04)
    assert flat.reserves.mean() < mid.reserves.mean() < tight.reserves.mean()
    assert high.reserves.mean() > 1.5 * low.reserves.mean()
    assert old_high.reserves.mean() < 1.1 * old_low.reserves.mean()


def test_replays_by_seed(tri):
    settings = dict(chains=2, tune=200, draws=200, n_sims=500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = BayesBfNormalised(seed=7, **settings).fit(tri, "paid", "prior")
        b = BayesBfNormalised(seed=7, **settings).fit(tri, "paid", "prior")
        c = BayesBfNormalised(seed=8, **settings).fit(tri, "paid", "prior")
    assert a.draws == b.draws
    assert a.reserves.draw_matrix() == b.reserves.draw_matrix()
    assert a.draws != c.draws
    assert a.reserves.draw_matrix() != c.reserves.draw_matrix()


def test_failed_diagnostics_warn_but_fit(tri):
    # 2 x 10 kept draws cannot reach a bulk ESS of 200 (100 per chain).
    with pytest.warns(UserWarning, match="BayesBfNormalised: the MCMC diagnostics failed.*ESS"):
        f = BayesBfNormalised(chains=2, tune=50, draws=10, n_sims=100, seed=1).fit(
            tri, "paid", "prior"
        )
    assert any("bulk ESS" in w for w in f.diagnostics["warnings"])
    assert f.reserves.n_sims == 100


def test_one_year(tri):
    model = BayesBfNormalised(apriori=0.9, cv=0.2, tune=300, draws=300, n_sims=1000, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        view = model.one_year(tri, "paid", "prior", refit="reweight_only")
        fit = model.fit(tri, "paid", "prior")
    assert isinstance(view, BayesOneYearFit)
    assert isinstance(view.fit, BayesBfNormalisedFit)
    # The posterior and lifetime reserves are fit's, draw for draw.
    assert view.fit.draws == fit.draws
    assert view.fit.reserves.draw_matrix() == fit.reserves.draw_matrix()
    assert view.origins == [str(y) for y in range(2001, 2011)]
    assert len(view.pareto_k) == 1000 and view.refitted == []
    # 2001 is at the last age: no cell in the year, no change.
    assert all(row[0] == 0.0 for row in view.cdr.draw_matrix())
    assert view.cdr.variance() < fit.reserves.variance()
    frame = view.to_frame()
    assert list(frame["prior_ultimate"]) == pytest.approx(fit.prior_ultimate)
    assert "opening_reserve" in frame.columns and "cdr_std_dev" in frame.columns
    with pytest.raises(ValueError, match="refit must be"):
        model.one_year(tri, "paid", "prior", refit="sometimes")
    with pytest.raises(ValueError, match="k_threshold"):
        model.one_year(tri, "paid", "prior", k_threshold=float("nan"))


def test_errors(tri):
    with pytest.raises(ValueError, match="n_sims must be positive"):
        BayesBfNormalised(n_sims=0)
    with pytest.raises(ValueError, match="process"):
        BayesBfNormalised(process="lognormal")
    with pytest.raises(OverflowError):
        BayesBfNormalised(seed=-1)
    for name, value in [("apriori", 0.0), ("cv", 0.0), ("cv", float("nan")),
                        ("development_sd", math.inf)]:
        with pytest.raises(ValueError, match=name):
            BayesBfNormalised(**{name: value}, **QUICK).fit(tri, "paid", "prior")
    with pytest.raises(ValueError):
        BayesBfNormalised(**QUICK).fit(tri, "paid", "missing")
    with pytest.raises(ValueError):
        BayesBfNormalised(**QUICK).fit(tri, "missing", "prior")

    # The youngest origin's increments sum to -5: a flat prior has no
    # log-linear mean there, an informative one still does.
    bad = Triangle.from_long(
        [2020] * 3 + [2021] * 2 + [2022],
        [12, 24, 36, 12, 24, 12],
        {"paid": [100.0, 150.0, 160.0, 110.0, 170.0, -5.0], "premium": [200.0] * 6},
    )
    with pytest.raises(ValueError, match="origin 2022"):
        BayesBfNormalised(cv=math.inf, **QUICK).fit(bad, "paid", "premium")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = BayesBfNormalised(apriori=0.8, cv=0.1, **QUICK).fit(bad, "paid", "premium")
    assert ok.reserves.n_sims == 2000
    # ... and PSIS-LOO has no density at a negative increment.
    with pytest.raises(ValueError, match="non-negative increments"):
        ok.loo()
