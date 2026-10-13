"""BayesBf, the Bayesian Bornhuetter-Ferguson (docs/design/reserving-bayes.md,
decision 9).

The parity with the same model in Stan and with England, Verrall and
Wuthrich (2012) is validation/tests/reserving_bayes_bf.rs; these tests
check the binding: settings, the fit's fields and frames, the two limits of
the prior's strength, replay by seed, the diagnostics warning and the
errors, on EVW's triangle (validation/data/evw2012.csv) with its prior
ultimates as the exposure.
"""

import csv
import math
import warnings
from pathlib import Path

import pytest

from prospicio.distributions import PredictiveDistribution
from prospicio.models import Elpd
from prospicio.reserving import BayesBf, BayesBfFit, BayesOdp, ChainLadderFit, Triangle

VALIDATION = Path(__file__).resolve().parents[2] / "validation"

# The default chains (4 x 1,000 draws) take about 0.1 s on EVW's triangle
# in a release build and pass every diagnostic.
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
        return BayesBf(seed=3, **QUICK).fit(tri, "paid", "prior")


def test_settings_and_defaults():
    m = BayesBf()
    assert (m.apriori, m.cv, m.development_sd) == (1.0, 0.1, 10.0)
    assert m.process == "gamma"
    assert (m.chains, m.tune, m.draws, m.n_sims, m.seed) == (4, 1000, 1000, 10_000, 0)
    assert (m.target_accept, m.max_depth) == (0.95, 10)
    m = BayesBf(apriori=0.7, cv=math.inf, process="none", chains=3, seed=9)
    assert (m.apriori, m.cv, m.process, m.chains, m.seed) == (0.7, math.inf, "none", 3, 9)
    assert repr(m).startswith("BayesBf(apriori=0.7, cv=inf, development_sd=10.0")


def test_fit_fields(fit):
    assert isinstance(fit, BayesBfFit)
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
    assert r.provenance()["model"] == "bayes_bf"
    # EVW print 6,145,526 at a = 100 (cv 0.1). Over seeds 0 to 29 the
    # mean sits +0.26% above it (the default development_sd's bias) with a
    # seed-to-seed sd of 0.15%: 1.5% is the bias plus 8 sds.
    assert r.mean() == pytest.approx(6145526, rel=0.015)

    names = fit.parameter_names
    assert names[0] == "alpha[1]" and names[9] == "alpha[10]"
    assert names[10] == "beta[1]" and names[-1] == "beta[10]"
    assert len(names) == 20
    assert fit.chains == 4
    draws = fit.draws
    assert len(draws) == 4000 and all(len(row) == 20 for row in draws)

    d = fit.diagnostics
    assert set(d) == {"max_rhat", "min_ess_bulk", "min_ess_tail", "divergences", "warnings"}
    assert d["warnings"] == [] and d["divergences"] == 0
    assert repr(fit).startswith("BayesBfFit(origins=10, draws=4000, n_sims=2000")


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


def test_limits_of_the_prior(tri):
    # Over seeds 0 to 29 each mean below sat within 0.7% of its target,
    # with seed-to-seed sds of 0.15% to 0.23%; the tolerances are at least
    # 6 sds beyond the mean offset.
    #
    # A flat prior is the Bayesian chain ladder: BayesOdp's reserve within
    # Monte Carlo error (the two vague priors differ slightly).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flat = BayesBf(cv=math.inf, development_sd=100, seed=1, **QUICK).fit(tri, "paid", "prior")
        odp = BayesOdp(seed=1, **QUICK).fit(tri, "paid")
        tight = BayesBf(cv=0.001, seed=1, **QUICK).fit(tri, "paid", "prior")
        mid = BayesBf(cv=0.1, seed=1, **QUICK).fit(tri, "paid", "prior")
    assert flat.reserves.mean() == pytest.approx(odp.reserves.mean(), rel=0.02)
    # EVW print 6,049,398 at a = 0.
    assert flat.reserves.mean() == pytest.approx(6049398, rel=0.015)
    # A very tight prior: EVW's closed-form strong-prior limit, 6,644,047.
    assert tight.reserves.mean() == pytest.approx(6644047, rel=0.015)
    assert flat.reserves.mean() < mid.reserves.mean() < tight.reserves.mean()


def test_replays_by_seed(tri):
    settings = dict(chains=2, tune=200, draws=200, n_sims=500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = BayesBf(seed=7, **settings).fit(tri, "paid", "prior")
        b = BayesBf(seed=7, **settings).fit(tri, "paid", "prior")
        c = BayesBf(seed=8, **settings).fit(tri, "paid", "prior")
    assert a.draws == b.draws
    assert a.reserves.draw_matrix() == b.reserves.draw_matrix()
    assert a.draws != c.draws
    assert a.reserves.draw_matrix() != c.reserves.draw_matrix()


def test_failed_diagnostics_warn_but_fit(tri):
    # 2 x 10 kept draws cannot reach a bulk ESS of 200 (100 per chain).
    with pytest.warns(UserWarning, match="BayesBf: the MCMC diagnostics failed.*ESS"):
        f = BayesBf(chains=2, tune=50, draws=10, n_sims=100, seed=1).fit(tri, "paid", "prior")
    assert any("bulk ESS" in w for w in f.diagnostics["warnings"])
    assert f.reserves.n_sims == 100


def test_errors(tri):
    with pytest.raises(ValueError, match="n_sims must be positive"):
        BayesBf(n_sims=0)
    with pytest.raises(ValueError, match="process"):
        BayesBf(process="lognormal")
    with pytest.raises(OverflowError):
        BayesBf(seed=-1)
    for name, value in [("apriori", 0.0), ("cv", 0.0), ("cv", float("nan")),
                        ("development_sd", math.inf)]:
        with pytest.raises(ValueError, match=name):
            BayesBf(**{name: value}, **QUICK).fit(tri, "paid", "prior")
    with pytest.raises(ValueError):
        BayesBf(**QUICK).fit(tri, "paid", "missing")
    with pytest.raises(ValueError):
        BayesBf(**QUICK).fit(tri, "missing", "prior")

    # The youngest origin's increments sum to -5: a flat prior has no
    # log-linear mean there, an informative one still does.
    bad = Triangle.from_long(
        [2020] * 3 + [2021] * 2 + [2022],
        [12, 24, 36, 12, 24, 12],
        {"paid": [100.0, 150.0, 160.0, 110.0, 170.0, -5.0], "premium": [200.0] * 6},
    )
    with pytest.raises(ValueError, match="origin 2022"):
        BayesBf(cv=math.inf, **QUICK).fit(bad, "paid", "premium")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = BayesBf(apriori=0.8, cv=0.1, **QUICK).fit(bad, "paid", "premium")
    assert ok.reserves.n_sims == 2000
    # ... and PSIS-LOO has no density at a negative increment.
    with pytest.raises(ValueError, match="non-negative increments"):
        ok.loo()
