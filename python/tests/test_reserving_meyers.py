"""Meyers' CRC and CSR models (docs/design/reserving-bayes.md, decision 7).

The parity with Meyers' own Stan code and his printed numbers is
validation/tests/reserving_meyers.rs; these tests check the binding:
settings, the fit's fields and frames, replay by seed, the diagnostics
warning and the errors, on Meyers' group 353 Commercial Auto data.
"""

import csv
import warnings
from pathlib import Path

import pytest

from prospicio.distributions import PredictiveDistribution
from prospicio.models import Elpd
from prospicio.reserving import ChainLadderFit, Meyers, MeyersFit, Triangle

VALIDATION = Path(__file__).resolve().parents[2] / "validation"

# The default chains (4 x 2,000 draws) take under a second per fit in a
# release build and pass every diagnostic on group 353.
QUICK = dict(n_sims=2000)


def comauto(upper=True):
    """Group 353: the upper triangle (1988 + w at ages up to 120 - 12 w) by
    default, as Meyers fits it, or the full 10 x 10 square."""
    path = VALIDATION / "data" / "clrd_comauto_353.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    if upper:
        rows = [r for r in rows if int(r["origin"]) - 1988 + int(r["development"]) // 12 <= 10]
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows],
        development=[int(r["development"]) for r in rows],
        values={k: [float(r[k]) for r in rows] for k in ("paid", "incurred", "premium")},
    )


@pytest.fixture(scope="module")
def tri():
    return comauto()


@pytest.fixture(scope="module")
def crc(tri):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the defaults pass every diagnostic
        return Meyers(**QUICK).fit(tri, "paid", "premium")


@pytest.fixture(scope="module")
def csr(tri):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return Meyers("csr", **QUICK).fit(tri, "paid", "premium")


def test_settings_and_defaults():
    m = Meyers()
    assert m.model == "crc"
    assert (m.logelr_mean, m.logelr_sd) == (-0.4, 3.162)
    assert (m.origin_sd, m.development_sd, m.gamma_sd) == (3.162, 3.162, 0.05)
    assert (m.chains, m.tune, m.draws, m.n_sims, m.seed) == (4, 1000, 2000, 10_000, 0)
    assert (m.target_accept, m.max_depth) == (0.95, 10)
    m = Meyers("CSR", gamma_sd=0.1, chains=3, seed=9)
    assert (m.model, m.gamma_sd, m.chains, m.seed) == ("csr", 0.1, 3, 9)
    assert repr(m).startswith('Meyers(model="csr", logelr_mean=-0.4')


def test_crc_fit_fields(crc):
    assert isinstance(crc, MeyersFit)
    assert crc.model == "crc"
    assert isinstance(crc.chain_ladder, ChainLadderFit)
    assert crc.origins == [str(y) for y in range(1988, 1998)]
    assert crc.development == [12 * k for k in range(1, 11)]
    assert crc.premium[:3] == [5812.0, 4908.0, 5454.0]

    u, r = crc.ultimates, crc.reserves
    for pd in (u, r):
        assert isinstance(pd, PredictiveDistribution)
        assert pd.n_sims == 2000
        assert pd.components() == [(o,) for o in crc.origins]
        assert pd.provenance()["model"] == "meyers_crc"
    # Meyers (2019, Table 5.2): CRC paid total ultimate 40,121 (sd 2,487).
    assert u.mean() == pytest.approx(40_121, rel=0.02)
    # 1988 is observed at the oldest age and keeps its value.
    first = [row[0] for row in u.draw_matrix()]
    assert min(first) == max(first) == 3912.0
    # Reserve = ultimate - latest, simulation by simulation.
    latest = sum(crc.chain_ladder.to_frame()["latest"])
    assert r.mean() == pytest.approx(u.mean() - latest, rel=1e-9)

    names = crc.parameter_names
    assert names[0] == "logelr" and names[1] == "alpha[2]" and names[10] == "beta[1]"
    assert names[19] == "sigma[1]" and names[-1] == "sigma[10]" and "gamma" not in names
    assert len(names) == 29
    assert crc.chains == 4
    draws = crc.draws
    assert len(draws) == 8000 and all(len(row) == 29 for row in draws)

    d = crc.diagnostics
    assert set(d) == {"max_rhat", "min_ess_bulk", "min_ess_tail", "divergences", "warnings"}
    assert d["warnings"] == [] and d["divergences"] == 0 and d["max_rhat"] < 1.01
    assert repr(crc).startswith('MeyersFit(model="crc", origins=10, draws=8000, n_sims=2000')


def test_summary_frames_and_loo(crc):
    s = crc.summary()
    assert list(s.columns) == [
        "parameter", "mean", "sd", "q05", "q50", "q95", "rhat", "ess_bulk", "ess_tail",
    ]
    assert list(s["parameter"]) == crc.parameter_names

    long = crc.to_frame()
    assert list(long.columns)[:4] == ["origin", "latest", "ultimate", "reserve"]
    for col in ("premium", "mean", "std_dev", "ultimate_mean", "ultimate_std_dev"):
        assert col in long.columns
    assert len(long) == 10
    assert long["mean"].sum() == pytest.approx(crc.reserves.mean(), rel=1e-9)
    assert long["ultimate_mean"].sum() == pytest.approx(crc.ultimates.mean(), rel=1e-9)
    totals = crc.totals_frame()
    assert len(totals) == 1
    assert totals.loc[0, "premium"] == pytest.approx(sum(crc.premium))
    assert totals.loc[0, "ultimate_mean"] == pytest.approx(crc.ultimates.mean(), rel=1e-9)
    assert len(crc.development_frame()) == 10

    loo = crc.loo()
    assert isinstance(loo, Elpd)
    assert len(loo.pointwise) == 55 and len(loo.pareto_k) == 55
    # Meyers (2019, Table 6.4): elpd_loo 47.80; ours is 47.90 at seed 0.
    assert loo.elpd == pytest.approx(47.8, abs=2.0)


def test_csr(csr):
    assert csr.model == "csr"
    names = csr.parameter_names
    assert len(names) == 30 and names[19] == "gamma" and names[20] == "sigma[1]"
    assert csr.ultimates.provenance()["model"] == "meyers_csr"
    # Meyers (2019, Tables 7.1 and 7.2): gamma 0.0446, total 37,597.
    s = csr.summary().set_index("parameter")
    assert s.loc["gamma", "mean"] == pytest.approx(0.0446, abs=0.01)
    assert csr.ultimates.mean() == pytest.approx(37_597, rel=0.02)


def test_replays_by_seed(tri):
    settings = dict(chains=2, tune=200, draws=200, n_sims=500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = Meyers(seed=7, **settings).fit(tri, "incurred", "premium")
        b = Meyers(seed=7, **settings).fit(tri, "incurred", "premium")
        c = Meyers(seed=8, **settings).fit(tri, "incurred", "premium")
    assert a.draws == b.draws
    assert a.ultimates.draw_matrix() == b.ultimates.draw_matrix()
    assert a.draws != c.draws
    assert a.ultimates.draw_matrix() != c.ultimates.draw_matrix()


def test_failed_diagnostics_warn_but_fit(tri):
    # 2 x 10 kept draws cannot reach a bulk ESS of 200 (100 per chain).
    with pytest.warns(UserWarning, match="Meyers: the MCMC diagnostics failed.*ESS"):
        f = Meyers(chains=2, tune=50, draws=10, n_sims=100, seed=1).fit(tri, "paid", "premium")
    assert any("bulk ESS" in w for w in f.diagnostics["warnings"])
    assert f.reserves.n_sims == 100


def test_errors(tri):
    with pytest.raises(ValueError, match="n_sims must be positive"):
        Meyers(n_sims=0)
    with pytest.raises(ValueError, match="crc"):
        Meyers("cay")
    with pytest.raises(OverflowError):
        Meyers(seed=-1)
    with pytest.raises(ValueError, match="gamma_sd"):
        Meyers(gamma_sd=0.0, **QUICK).fit(tri, "paid", "premium")
    with pytest.raises(ValueError, match="logelr_mean"):
        Meyers(logelr_mean=float("nan"), **QUICK).fit(tri, "paid", "premium")
    with pytest.raises(ValueError):
        Meyers(**QUICK).fit(tri, "paid", "missing")

    # The log-normal model needs positive cells.
    zero = Triangle.from_long(
        [2020] * 3 + [2021] * 2 + [2022],
        [12, 24, 36, 12, 24, 12],
        {"paid": [100.0, 150.0, 160.0, 0.0, 170.0, 130.0], "premium": [200.0] * 6},
    )
    with pytest.raises(ValueError, match="positive cumulative values"):
        Meyers(**QUICK).fit(zero, "paid", "premium")

    # Six cells cannot identify CRC's eight parameters.
    small = Triangle.from_long(
        [2020] * 3 + [2021] * 2 + [2022],
        [12, 24, 36, 12, 24, 12],
        {"paid": [100.0, 150.0, 160.0, 110.0, 170.0, 130.0], "premium": [200.0] * 6},
    )
    with pytest.raises(ValueError, match="more cells than parameters"):
        Meyers(**QUICK).fit(small, "paid", "premium")
