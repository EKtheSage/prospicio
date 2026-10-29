"""A development pattern per simulation from the bootstraps and the Bayesian
fits (`reserving-v02.md` decision 28, #600, and Meyers' models, #678): row
``i`` is simulation ``i`` of the fit's ``reserves``, and ``monthly`` puts the
rows on the cash-flow engine's paid patterns."""

import csv
import math
from pathlib import Path

import pytest

import prospicio.pricing as P
from prospicio.reserving import (
    DevelopmentPatterns,
    MackBootstrap,
    Meyers,
    OdpBootstrap,
    TailConstant,
    Triangle,
    UnderwritingBasis,
)

VALIDATION = Path(__file__).resolve().parents[2] / "validation"
N = 500


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
def odp(raa):
    return OdpBootstrap(n_sims=N, seed=600).fit(raa, "paid")


@pytest.fixture(scope="module", params=["odp", "mack"])
def fit(request, raa):
    cls = OdpBootstrap if request.param == "odp" else MackBootstrap
    return cls(n_sims=N, seed=600).fit(raa, "paid")


def test_each_simulation_ends_at_one_and_the_mean_follows_the_chain_ladder(fit):
    p = fit.patterns
    assert isinstance(p, DevelopmentPatterns)
    assert p.n_sims == N
    assert p.segments == [()]
    ages = p.ages()
    draws = p.pattern_draws()
    assert len(draws) == N
    for row in draws:
        assert len(row) == len(ages)
        assert row[-1] == 1.0
    mean = p.mean()
    want = [1.0 / c for c in fit.chain_ladder.cdf]
    assert len(mean) == len(want)
    assert mean == pytest.approx(want, abs=0.03)


def test_a_tail_adds_one_age_at_one(raa):
    plain = OdpBootstrap(n_sims=N, seed=600).fit(raa, "paid").patterns
    tailed = OdpBootstrap(n_sims=N, seed=600, tail=TailConstant(1.05)).fit(raa, "paid").patterns
    assert len(tailed.ages()) == len(plain.ages()) + 1
    assert tailed.ages()[-1] == plain.ages()[-1] + 12
    for row in tailed.pattern_draws():
        assert row[-1] == 1.0
        assert row[-2] < 1.0


def test_monthly_patterns_are_non_decreasing_and_end_at_one(odp):
    fit = odp
    rows = fit.patterns.monthly(120, monotone="running_max")
    assert len(rows) == N
    for row in rows:
        assert len(row) == 120
        assert row[-1] == pytest.approx(1.0, abs=1e-12)
        assert all(b >= a for a, b in zip(row, row[1:]))
    # An underwriting basis and the other reading also give whole rows.
    uw = fit.patterns.monthly(240, basis=UnderwritingBasis(), monotone="running_max")
    assert len(uw) == N and all(len(r) == 240 for r in uw)
    ln = fit.patterns.monthly(120, basis="origin_ln_linear", monotone="running_max")
    assert len(ln) == N
    assert fit.patterns.monthly(120, basis="origin", monotone="running_max") == rows


def test_monthly_errors(fit):
    p = fit.patterns
    with pytest.raises(ValueError):
        p.monthly(24, monotone="running_max")
    with pytest.raises(ValueError, match="monotone"):
        p.monthly(120, monotone="sometimes")
    with pytest.raises(ValueError, match="basis"):
        p.monthly(120, basis="calendar")
    # A fit of one segment answers to any label.
    assert p.ages(("A",)) == p.ages()


def test_refuse_raises_when_a_simulation_falls_else_equals_running_max(fit):
    p = fit.patterns
    falls = any(b < a for row in p.pattern_draws() for a, b in zip(row, row[1:]))
    if falls:
        with pytest.raises(ValueError):
            p.monthly(120, basis="origin_ln_linear")
    else:
        assert p.monthly(120) == p.monthly(120, monotone="running_max")


def test_segments_are_named_by_their_label(two_lines):
    fits = OdpBootstrap(n_sims=200, seed=600).fit(two_lines, "paid")
    p = fits.patterns
    assert p.n_sims == 200
    assert sorted(p.segments) == [("A",), ("B",)]
    a = p.pattern_draws(("A",))
    assert len(a) == 200
    assert p.pattern_draws("A") == a == p.pattern_draws(["A"])
    assert p.ages(("B",)) == p.ages(("A",))
    with pytest.raises(ValueError):
        p.pattern_draws()
    with pytest.raises(ValueError):
        p.pattern_draws(("C",))
    assert len(p.monthly(120, ("A",), monotone="running_max")) == 200
    # A segment picked from the fit keeps its label and needs none.
    alone = fits.segment(lob="A").patterns
    assert alone.pattern_draws() == a == alone.pattern_draws("A")


def terms_of(months, paid):
    ramp = lambda m: [min((t + 1) / m, 1.0) for t in range(months)]
    return P.CashflowTerms(
        100.0, 30, ramp(12), ramp(36), paid, P.CollateralRequirement({"reserve": 1.0})
    )


def test_patterns_drive_the_cash_flow_engine(odp):
    fit = odp
    patterns = fit.patterns.monthly(120, monotone="running_max")
    u = fit.loss_ratio_draws(None, 400_000.0)
    own = patterns[0]
    terms = terms_of(120, own)
    out = terms.run(u, paid_patterns=patterns)
    assert len(out) == N
    # Every simulation on the terms' own pattern is the plain run.
    same = terms.run(u, paid_patterns=[own] * N)
    plain = terms.run(u)
    assert same.flows == plain.flows
    # The patterns differ across simulations, so the flows do too.
    assert out.flows != plain.flows


def test_a_share_that_is_not_positive_is_refused_or_lifted(raa):
    # Mack's bootstrap draws its factors from normal distributions; on RAA a
    # few of 5,000 simulations draw a negative first factor, so the share at
    # 12 months is negative. The default refuses it; running_max gives it
    # the next positive share.
    p = MackBootstrap(n_sims=5_000, seed=600).fit(raa, "paid").patterns
    rows = p.pattern_draws()
    assert any(r[0] <= 0.0 for r in rows)
    with pytest.raises(ValueError, match="not positive"):
        p.monthly(120, basis="origin_ln_linear")
    held = p.monthly(120, basis="origin_ln_linear", monotone="running_max")
    assert len(held) == 5_000
    for row in held:
        assert row[0] > 0.0
        assert all(b >= a for a, b in zip(row, row[1:]))
        assert row[-1] == 1.0


def comauto_353(keys=None):
    """Meyers' group 353 Commercial Auto paid, the upper triangle, with its
    premium; ``keys`` names copies of it, each doubling the one before."""
    with open(VALIDATION / "data" / "clrd_comauto_353.csv", newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    rows = [r for r in rows if int(r["origin"]) - 1988 + int(r["development"]) // 12 <= 10]
    names = keys or [None]
    return Triangle.from_long(
        origin=[int(r["origin"]) for _ in names for r in rows],
        development=[int(r["development"]) for _ in names for r in rows],
        values={
            k: [float(r[k]) * 2.0**i for i in range(len(names)) for r in rows]
            for k in ("paid", "premium")
        },
        keys={"lob": [n for n in names for _ in rows]} if keys else None,
    )


# Meyers' default chains pass every diagnostic on group 353.
MEYERS = dict(n_sims=1000)


@pytest.mark.parametrize("model", ["crc", "csr", "cay"])
def test_meyers_patterns_end_at_one_at_the_oldest_age(model):
    fit = Meyers(model, seed=678, **MEYERS).fit(comauto_353(), "paid", "premium")
    p = fit.patterns
    assert isinstance(p, DevelopmentPatterns)
    assert p.n_sims == 1000 == fit.reserves.n_sims
    assert p.ages() == [12 * (d + 1) for d in range(10)]
    rows = p.pattern_draws()
    assert len(rows) == 1000
    for row in rows:
        assert row[-1] == 1.0
        assert all(x > 0.0 and math.isfinite(x) for x in row)
    # The simulations take different posterior draws.
    assert len({tuple(r) for r in rows}) > 100
    # Nothing keeps beta_d rising, and on group 353 a late share falls in
    # nearly every simulation: the default refuses, running_max holds.
    with pytest.raises(ValueError, match="falls"):
        p.monthly(120)
    held = p.monthly(120, monotone="running_max")
    assert len(held) == 1000
    for row in held:
        assert all(b >= a for a, b in zip(row, row[1:]))
        assert row[-1] == 1.0
    # Each simulation's loss ratio is paid by its own pattern.
    u = fit.loss_ratio_draws(None, 50_000.0)
    terms = terms_of(120, held[0])
    out = terms.run(u, paid_patterns=held)
    assert len(out) == 1000
    assert out.flows != terms.run(u).flows


def test_meyers_segments_have_their_own_patterns():
    tri = comauto_353(["A", "B"])
    fits = Meyers(seed=679, **MEYERS).fit_segments(tri, "paid", "premium")
    p = fits.patterns
    assert p.n_sims == 1000
    assert sorted(p.segments) == [("A",), ("B",)]
    a, b = p.pattern_draws("A"), p.pattern_draws("B")
    assert all(r[-1] == 1.0 for r in a + b)
    # Each segment is sampled on its own seeds.
    assert a != b
    with pytest.raises(ValueError):
        p.pattern_draws()
    alone = fits.segment(lob="B").patterns
    assert alone.pattern_draws() == b
