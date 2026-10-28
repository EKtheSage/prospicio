"""Ultimates and loss ratios per simulation from reserving fits (#602,
`reserving-v02.md` decision 26): index ``i`` is simulation ``i`` across
origins, calls and fits with the same seed, and a tower on the draws cedes
what it cedes on the fit's own distribution."""

import csv
from pathlib import Path

import pytest

from prospicio.distributions import PredictiveDistribution
from prospicio.reinsurance import Layer, Tower
from prospicio.reserving import BayesOdp, MackBootstrap, OdpBootstrap, Triangle

VALIDATION = Path(__file__).resolve().parents[2] / "validation"


def raa_rows():
    with open(VALIDATION / "data" / "raa.csv", newline="") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


@pytest.fixture(scope="module")
def raa():
    rows = raa_rows()
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows],
        development=[int(r["development"]) for r in rows],
        values={"paid": [float(r["value"]) for r in rows]},
    )


@pytest.fixture(scope="module")
def two_lines():
    rows = raa_rows()
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows] * 2,
        development=[int(r["development"]) for r in rows] * 2,
        values={"paid": [float(r["value"]) * s for s in (1.0, 2.0) for r in rows]},
        keys={"lob": ["A"] * len(rows) + ["B"] * len(rows)},
    )


@pytest.fixture(scope="module")
def boot(raa):
    return OdpBootstrap(n_sims=2000, seed=602).fit(raa, "paid")


def column(pd, j):
    return [row[j] for row in pd.draw_matrix()]


def test_each_origin_is_its_latest_plus_its_reserve_in_the_same_simulation(boot):
    latest = boot.chain_ladder.latest
    for j, origin in enumerate(boot.origins):
        ultimate = boot.ultimate_draws(origin)
        reserve = column(boot.reserves, j)
        assert len(ultimate) == 2000
        assert ultimate == pytest.approx([latest[j] + r for r in reserve], rel=1e-12)
    # An origin by its label, an integer or a key tuple.
    assert boot.ultimate_draws(1990) == boot.ultimate_draws("1990") == boot.ultimate_draws(("1990",))
    # The loss ratio on the origin's premium, in the same order.
    ratio = boot.loss_ratio_draws(1990, 25_000.0)
    assert ratio == pytest.approx([u / 25_000.0 for u in boot.ultimate_draws(1990)], rel=1e-15)
    # The total is summed simulation by simulation.
    total = boot.ultimate_draws()
    rows = boot.reserves.draw_matrix()
    assert total == pytest.approx([sum(latest) + sum(r) for r in rows], rel=1e-12)
    assert boot.loss_ratio_draws(None, 400_000.0) == pytest.approx(
        [t / 400_000.0 for t in total], rel=1e-15)
    # The joint ultimates keep the reserves' layout.
    ultimates = boot.ultimates
    assert isinstance(ultimates, PredictiveDistribution)
    assert ultimates.components() == boot.reserves.components()
    assert column(ultimates, 9) == pytest.approx(boot.ultimate_draws(1990), rel=1e-15)


def test_the_same_seed_gives_the_same_order(raa, boot):
    again = OdpBootstrap(n_sims=2000, seed=602).fit(raa, "paid")
    assert again.ultimate_draws(1985) == boot.ultimate_draws(1985)
    other = OdpBootstrap(n_sims=2000, seed=603).fit(raa, "paid")
    assert other.ultimate_draws(1985) != boot.ultimate_draws(1985)


def test_a_tower_on_the_draws_cedes_what_it_cedes_on_the_fits_distribution(boot):
    limit, attachment = 30_000.0, 220_000.0
    tower = Tower([Layer.stop_loss("adc", limit, attachment)])
    on_draws = tower.apply_annual(boot.ultimate_draws()).marginal(["ceded", "adc"]).draws
    on_fit = tower.apply_aggregate(boot.ultimates).marginal(["ceded", "adc"]).draws
    assert on_draws == pytest.approx(on_fit, rel=1e-12, abs=1e-6)
    expected = sum(on_draws) / len(on_draws)
    assert 0.0 < expected < limit


def test_segments_take_the_full_key(two_lines):
    fits = MackBootstrap(n_sims=500, seed=4).fit(two_lines, "paid")
    a = fits.ultimate_draws(("A", 1990))
    b = fits.ultimate_draws(("B", "1990"))
    latest = fits.chain_ladder.latest
    assert latest[19] == 2.0 * latest[9]
    assert a == pytest.approx([latest[9] + r for r in column(fits.reserves, 9)], rel=1e-12)
    assert b == pytest.approx([latest[19] + r for r in column(fits.reserves, 19)], rel=1e-12)
    with pytest.raises(ValueError, match="1990"):
        fits.ultimate_draws(1990)
    with pytest.raises(ValueError, match="premium"):
        fits.loss_ratio_draws(("A", 1990), 0.0)


def test_a_bayesian_fit(raa):
    fit = BayesOdp(n_sims=1000, seed=3).fit(raa, "paid")
    latest = fit.chain_ladder.latest
    for j, origin in enumerate(fit.origins):
        assert fit.ultimate_draws(origin) == pytest.approx(
            [latest[j] + r for r in column(fit.reserves, j)], rel=1e-12)
    again = BayesOdp(n_sims=1000, seed=3).fit(raa, "paid")
    assert again.ultimate_draws() == fit.ultimate_draws()
