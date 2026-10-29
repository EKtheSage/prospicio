"""A development pattern per posterior draw from Meyers' CRC, CSR and CAY
(`reserving-bayes.md` decision 27, #678): row ``i`` is simulation ``i`` of the
fit's ``reserves`` and ``ultimates``, and ``monthly`` puts the rows on the
cash-flow engine's paid patterns. The formula and the alignment with the
reserves are checked in Rust (``meyers.rs``); these tests check the binding on
Meyers' group 353 Commercial Auto data."""

import csv
from pathlib import Path

import pytest

from prospicio.reserving import DevelopmentPatterns, Meyers, Triangle

VALIDATION = Path(__file__).resolve().parents[2] / "validation"
N = 400
QUICK = dict(tune=300, draws=300, n_sims=N, seed=678)


def comauto_rows():
    path = VALIDATION / "data" / "clrd_comauto_353.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return [r for r in rows if int(r["origin"]) - 1988 + int(r["development"]) // 12 <= 10]


def build(rows, lines=("",)):
    scale = {"": 1.0, "A": 1.0, "B": 2.0}
    origin, development, paid, premium, lob = [], [], [], [], []
    for line in lines:
        for r in rows:
            origin.append(int(r["origin"]))
            development.append(int(r["development"]))
            paid.append(float(r["paid"]) * scale[line])
            premium.append(float(r["premium"]) * scale[line])
            lob.append(line)
    keys = {"lob": lob} if lines != ("",) else None
    return Triangle.from_long(
        origin=origin,
        development=development,
        values={"paid": paid, "premium": premium},
        keys=keys,
    )


@pytest.fixture(scope="module")
def tri():
    return build(comauto_rows())


@pytest.fixture(scope="module", params=["crc", "csr", "cay"])
def fit(request, tri):
    return Meyers(request.param, **QUICK).fit(tri, "paid", "premium")


def test_each_simulation_ends_at_one(fit):
    p = fit.patterns
    assert isinstance(p, DevelopmentPatterns)
    assert p.n_sims == N
    assert p.segments == [()]
    assert p.ages() == [12 * d for d in range(1, 11)]
    draws = p.pattern_draws()
    # One pattern per simulation of the reserves.
    assert len(draws) == N == len(fit.ultimate_draws())
    for row in draws:
        assert len(row) == 10
        # The oldest age is the reference, so there is no tail.
        assert row[-1] == 1.0
        assert all(x > 0 for x in row)
    mean = p.mean()
    assert len(mean) == 10 and mean[-1] == 1.0
    # Group 353 develops: the mean pattern rises from well below 1.
    assert mean[0] < 0.7
    assert mean[-2] > 0.9


def test_a_refit_on_the_same_seed_repeats_patterns_and_ultimates(fit):
    # A determinism check only. That the patterns draw no random number, so
    # the reserves are what they were, is checked by the Rust replay test
    # (``each_simulations_pattern_is_its_posterior_draws``).
    again = Meyers(fit.model, **QUICK).fit(build(comauto_rows()), "paid", "premium")
    assert again.patterns.pattern_draws() == fit.patterns.pattern_draws()
    assert again.ultimate_draws() == fit.ultimate_draws()
    assert len(fit.ultimate_draws()) == N


def test_monthly_gives_whole_rows_and_a_falling_one_is_refused_or_held(fit):
    p = fit.patterns
    rows = p.monthly(120, monotone="running_max")
    assert len(rows) == N
    for row in rows:
        assert len(row) == 120
        assert row[-1] == pytest.approx(1.0, abs=1e-12)
        assert all(b >= a for a, b in zip(row, row[1:]))
    falls = any(b < a for row in p.pattern_draws() for a, b in zip(row, row[1:]))
    if falls:
        with pytest.raises(ValueError):
            p.monthly(120)
    else:
        assert p.monthly(120) == rows
    with pytest.raises(ValueError):
        p.monthly(24, monotone="running_max")


def test_segments_are_named_by_their_label():
    tri = build(comauto_rows(), lines=("A", "B"))
    fits = Meyers("crc", **QUICK).fit_segments(tri, "paid", "premium")
    p = fits.patterns
    assert p.n_sims == N
    assert sorted(p.segments) == [("A",), ("B",)]
    a, b = p.pattern_draws(("A",)), p.pattern_draws(("B",))
    assert len(a) == len(b) == N
    assert a != b  # each segment has its own seeds
    assert all(row[-1] == 1.0 for row in a + b)
    one = fits.segment(lob="B").patterns
    assert one.pattern_draws() == b
    with pytest.raises(ValueError):
        p.pattern_draws(("C",))


def test_a_rank_correlation_moves_the_patterns_with_the_reserves():
    tri = build(comauto_rows(), lines=("A", "B"))
    rho = [[1.0, 0.8], [0.8, 1.0]]
    free = Meyers("crc", **QUICK).fit_segments(tri, "paid", "premium")
    ranked = Meyers("crc", dependence="rank_correlation", spearman=rho, **QUICK).fit_segments(tri, "paid", "premium")
    # Each ranked row's reserves are some free row j's, and its pattern is
    # that same row j's pattern.
    moved = 0
    for label in ("A", "B"):
        free_reserves = free.segment(lob=label).reserves.draw_matrix()
        ranked_reserves = ranked.segment(lob=label).reserves.draw_matrix()
        source = {tuple(row): j for j, row in enumerate(free_reserves)}
        assert len(source) == N
        free_patterns = free.patterns.pattern_draws((label,))
        ranked_patterns = ranked.patterns.pattern_draws((label,))
        for i, row in enumerate(ranked_reserves):
            j = source[tuple(row)]
            assert ranked_patterns[i] == free_patterns[j], (label, i, j)
            moved += i != j
    assert moved > N // 4
