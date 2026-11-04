"""The posterior predictive of a new underwriting year
(docs/design/reserving-bayes.md, decision 28, issue #598):
BayesBfNormalisedFit.new_year and MeyersFit.new_year.

The exact distributions (the conjugate case on a square, the tight-prior
limit, the pooling of identical origins, Meyers' implied moments) are checked
in validation/tests/reserving_bayes_new_year.rs; these tests check the
binding: the result's fields, that the new year without a common level is
the prior, the pairing with the patterns and their monthly emergence, replay
by seed and the errors.
"""

import csv
import math
from pathlib import Path

import pytest

from prospicio.distributions import PredictiveDistribution
from prospicio.reserving import (
    BayesBfNormalised,
    DevelopmentPatterns,
    Meyers,
    NewYear,
    Triangle,
)

VALIDATION = Path(__file__).resolve().parents[2] / "validation"


def comauto():
    """Group 353's upper triangle, as Meyers fits it."""
    path = VALIDATION / "data" / "clrd_comauto_353.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    rows = [r for r in rows if int(r["origin"]) - 1988 + int(r["development"]) // 12 <= 10]
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows],
        development=[int(r["development"]) for r in rows],
        values={k: [float(r[k]) for r in rows] for k in ("paid", "incurred", "premium")},
    )


@pytest.fixture(scope="module")
def tri():
    return comauto()


def column(pd):
    return [row[0] for row in pd.draw_matrix()]


def moments(xs):
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def test_without_a_common_level_the_new_year_is_the_prior(tri):
    fit = BayesBfNormalised(apriori=0.7, cv=0.25, process="none", n_sims=200, seed=1).fit(
        tri, "paid", "premium"
    )
    year = fit.new_year(5000.0, n_sims=40_000, seed=3)
    assert isinstance(year, NewYear)
    assert year.origin == "1998"
    assert year.premium == 5000.0
    assert isinstance(year.loss_ratio, PredictiveDistribution)
    assert isinstance(year.patterns, DevelopmentPatterns)
    lr = column(year.loss_ratio)
    assert len(lr) == 40_000
    mean, sd = moments(lr)
    assert abs(mean / 0.7 - 1) < 0.005
    assert abs(sd / (0.7 * 0.25) - 1) < 0.02
    for u, x in zip(column(year.ultimate), lr):
        assert u == pytest.approx(5000.0 * x, rel=1e-12)
    # The seed replays and moves the draws.
    assert column(fit.new_year(5000.0, n_sims=40_000, seed=3).loss_ratio) == lr
    assert column(fit.new_year(5000.0, n_sims=40_000, seed=4).loss_ratio) != lr
    # Each simulation's pattern ends at 1, and the monthly emergence pairs up.
    draws = year.patterns.pattern_draws()
    assert len(draws) == 40_000 and all(row[-1] == 1.0 for row in draws[:100])
    # The monthly reading costs milliseconds a draw (knowledge/findings/
    # pattern-draws-monthly-cost.md), so it is checked on fewer simulations.
    small = fit.new_year(5000.0, n_sims=300, seed=3)
    monthly = small.patterns.monthly(120, monotone="running_max")
    assert len(monthly) == 300 and len(monthly[0]) == 120 and monthly[0][-1] == 1.0


def test_a_common_level_follows_the_data(tri):
    fit = BayesBfNormalised(
        apriori=1.2, cv=0.2, common_cv=10.0, process="none", n_sims=200, seed=1
    ).fit(tri, "paid", "premium")
    mean, sd = moments(column(fit.new_year(5000.0, n_sims=20_000, seed=1).loss_ratio))
    # Group 353's paid loss ratio is about 0.75, far below the prior's 1.2.
    assert 0.6 < mean < 0.95
    assert sd / mean >= 0.2


def test_meyers_new_year_and_its_level(tri):
    fit = Meyers(n_sims=200, seed=1).fit(tri, "paid", "premium")
    year = fit.new_year(5000.0, n_sims=20_000, seed=2)
    assert year.origin == "1998"
    logs = [math.log(x) for x in column(year.loss_ratio)]
    _, sd = moments(logs)
    assert 0.1 < sd < 0.5
    prior = fit.new_year(5000.0, n_sims=20_000, seed=2, level="prior")
    _, sd = moments([math.log(x) for x in column(prior.loss_ratio)])
    assert sd > 3.0  # Meyers' origin_sd of 3.162
    with pytest.raises(ValueError, match="exchangeable"):
        fit.new_year(5000.0, level="hierarchical")


def test_errors(tri):
    fit = BayesBfNormalised(cv=math.inf, n_sims=200, seed=1).fit(tri, "paid", "premium")
    with pytest.raises(ValueError, match="finite cv"):
        fit.new_year(5000.0)
    with pytest.raises(ValueError, match="premium"):
        fit.new_year(0.0)
    with pytest.raises(ValueError, match="n_sims"):
        fit.new_year(5000.0, n_sims=0)
