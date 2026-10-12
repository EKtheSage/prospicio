"""BayesOdp, the Bayesian ODP chain ladder (docs/design/reserving-bayes.md).

The parity with the same model in Stan is validation/tests/reserving_bayes.rs;
these tests check the binding: settings, the fit's fields and frames,
replay by seed, the diagnostics warning and the errors.
"""

import csv
import warnings
from pathlib import Path

import pytest

from prospicio.distributions import PredictiveDistribution
from prospicio.models import Elpd
from prospicio.reserving import BayesOdp, BayesOdpFit, ChainLadderFit, OdpBootstrap, Triangle

VALIDATION = Path(__file__).resolve().parents[2] / "validation"

# The default chains (4 x 1,000 draws) take well under a second on GenIns
# in a release build and pass every diagnostic; shorter ones need not.
QUICK = dict(n_sims=2000)


def dataset(name):
    with open(VALIDATION / "data" / f"{name}.csv", newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows],
        development=[int(r["development"]) for r in rows],
        values={"values": [float(r["value"]) for r in rows]},
    )


@pytest.fixture(scope="module")
def genins():
    return dataset("genins")


@pytest.fixture(scope="module")
def fit(genins):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # GenIns passes every diagnostic
        return BayesOdp(seed=3, **QUICK).fit(genins, "values")


def test_settings_and_defaults():
    m = BayesOdp()
    assert (m.intercept_sd, m.origin_sd, m.development_sd) == (10.0, 10.0, 10.0)
    assert m.process == "gamma"
    assert (m.chains, m.tune, m.draws, m.n_sims, m.seed) == (4, 1000, 1000, 10_000, 0)
    assert (m.target_accept, m.max_depth) == (0.8, 10)
    m = BayesOdp(origin_sd=2.0, process="none", chains=3, seed=9)
    assert (m.origin_sd, m.process, m.chains, m.seed) == (2.0, "none", 3, 9)
    assert repr(m).startswith("BayesOdp(intercept_sd=10.0, origin_sd=2.0")


def test_fit_fields(fit, genins):
    assert isinstance(fit, BayesOdpFit)
    assert isinstance(fit.chain_ladder, ChainLadderFit)
    assert fit.origins == [str(y) for y in range(2001, 2011)]
    assert fit.development == [12 * k for k in range(1, 11)]
    # The plug-in scale is the ODP bootstrap's.
    boot = OdpBootstrap(n_sims=10, seed=0).fit(genins, "values")
    assert fit.scale == pytest.approx(boot.scale, rel=1e-12)
    assert fit.scale == pytest.approx(52601.3615, rel=1e-8)

    r = fit.reserves
    assert isinstance(r, PredictiveDistribution)
    assert r.n_sims == 2000
    assert r.components() == [(o,) for o in fit.origins]
    assert r.provenance()["model"] == "bayes_odp"
    # The posterior mean reserve sits near the chain ladder's (+0.8% with
    # the defaults; Monte Carlo error with these short chains).
    cl = fit.chain_ladder.total_reserve
    assert r.mean() == pytest.approx(cl, rel=0.05)

    names = fit.parameter_names
    assert names[0] == "c" and names[1] == "a[2]" and names[-1] == "b[10]"
    assert len(names) == 19
    assert fit.chains == 4
    draws = fit.draws
    assert len(draws) == 4000 and all(len(row) == 19 for row in draws)

    d = fit.diagnostics
    assert set(d) == {"max_rhat", "min_ess_bulk", "min_ess_tail", "divergences", "warnings"}
    assert d["warnings"] == [] and d["divergences"] == 0
    assert d["max_rhat"] < 1.01 and d["min_ess_bulk"] >= 400
    assert repr(fit).startswith("BayesOdpFit(origins=10, draws=4000, n_sims=2000")


def test_summary_and_frames(fit):
    s = fit.summary()
    assert list(s.columns) == [
        "parameter", "mean", "sd", "q05", "q50", "q95", "rhat", "ess_bulk", "ess_tail",
    ]
    assert list(s["parameter"]) == fit.parameter_names
    assert s.loc[0, "mean"] == pytest.approx(12.49, abs=0.1)

    long = fit.to_frame()
    assert list(long.columns) == ["origin", "latest", "ultimate", "reserve", "mean", "std_dev"]
    assert len(long) == 10
    assert long["mean"].sum() == pytest.approx(fit.reserves.mean(), rel=1e-9)
    totals = fit.totals_frame()
    assert len(totals) == 1 and "scale" in totals.columns
    assert totals.loc[0, "mean"] == pytest.approx(fit.reserves.mean(), rel=1e-9)
    assert len(fit.development_frame()) == 10


def test_loo(fit):
    loo = fit.loo()
    assert isinstance(loo, Elpd)
    assert len(loo.pointwise) == 55 and len(loo.pareto_k) == 55
    # -749.9 (se 6.0) with the defaults.
    assert loo.elpd == pytest.approx(-750, abs=10)


def test_replays_by_seed(genins):
    settings = dict(chains=2, tune=200, draws=200, n_sims=500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = BayesOdp(seed=7, **settings).fit(genins, "values")
        b = BayesOdp(seed=7, **settings).fit(genins, "values")
        c = BayesOdp(seed=8, **settings).fit(genins, "values")
    assert a.draws == b.draws
    assert a.reserves.draw_matrix() == b.reserves.draw_matrix()
    assert a.draws != c.draws
    assert a.reserves.draw_matrix() != c.reserves.draw_matrix()


def test_failed_diagnostics_warn_but_fit(genins):
    # 2 x 10 kept draws cannot reach a bulk ESS of 200 (100 per chain).
    with pytest.warns(UserWarning, match="MCMC diagnostics failed.*ESS"):
        f = BayesOdp(chains=2, tune=50, draws=10, n_sims=100, seed=1).fit(genins, "values")
    assert f.diagnostics["warnings"]
    assert any("bulk ESS" in w for w in f.diagnostics["warnings"])
    assert f.reserves.n_sims == 100


def test_errors(genins):
    with pytest.raises(ValueError, match="n_sims must be positive"):
        BayesOdp(n_sims=0)
    with pytest.raises(ValueError, match="process"):
        BayesOdp(process="lognormal")
    with pytest.raises(OverflowError):
        BayesOdp(seed=-1)
    with pytest.raises(ValueError, match="origin_sd"):
        BayesOdp(origin_sd=-1.0, **QUICK).fit(genins, "values")
    with pytest.raises(ValueError):
        BayesOdp(**QUICK).fit(genins, "missing")

    # An origin whose increments sum to less than zero has no log-linear mean.
    tri = Triangle.from_long(
        [2020] * 3 + [2021] * 2 + [2022],
        [12, 24, 36, 12, 24, 12],
        [100.0, 150.0, 160.0, 110.0, 170.0, -5.0],
    )
    with pytest.raises(ValueError, match="origin 2022"):
        BayesOdp(**QUICK).fit(tri, "values")

    # Several segments wait for fit_segments.
    two = Triangle.from_long(
        [2020, 2020, 2021] * 2,
        [12, 24, 12] * 2,
        {"paid": [100.0, 150.0, 200.0, 10.0, 20.0, 30.0]},
        keys={"lob": ["Auto"] * 3 + ["Home"] * 3},
    )
    with pytest.raises(ValueError):
        BayesOdp(**QUICK).fit(two, "paid")


def test_loo_needs_non_negative_increments():
    # RAA has a negative increment, where the density does not exist.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raa = BayesOdp(chains=2, tune=200, draws=200, n_sims=100).fit(dataset("raa"), "values")
    with pytest.raises(ValueError, match="non-negative increments"):
        raa.loo()
