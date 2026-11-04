"""IndustryPrior and PatternPrior, the industry prior for the Bayesian BF
from other companies' triangles (docs/design/reserving-bayes.md, decision
28, #604).

The recovery of simulated companies' parameters is checked in
validation/tests/reserving_industry_prior.rs; these tests check the
binding on four companies with exact cells, whose prior has a closed form,
the same numbers the R twin checks
(R/prospicio/tests/test-reserving-industry-prior.R).
"""

import math

import pytest

from prospicio.reserving import (
    BayesBfNormalised,
    BayesBfNormalisedFit,
    IndustryPrior,
    LayerSharePrior,
    PatternPrior,
    Triangle,
)

SHARES = [0.4, 0.3, 0.2, 0.1]
LEVELS = [0.6, 0.8, 0.7, 0.9]
WOBBLE = [1.1, 0.9, 1.0, 1.0]
PREMIUM = 100.0


def rows(level):
    """Four origins at four ages, exactly `premium * level * wobble * share`
    cumulated: each company's chain ladder returns its pattern and loss
    ratios."""
    origin, ages, paid = [], [], []
    for w, wobble in enumerate(WOBBLE):
        cum = 0.0
        for d in range(4 - w):
            cum += PREMIUM * level * wobble * SHARES[d]
            origin.append(2010 + w)
            ages.append(12 * (d + 1))
            paid.append(cum)
    return origin, ages, paid


def company(level, premium=False):
    origin, ages, paid = rows(level)
    values = {"paid": paid}
    if premium:
        values["premium"] = [PREMIUM] * len(paid)
    return Triangle.from_long(origin, ages, values)


def closed_form():
    """apriori, within cv and between cv of the four companies (the Rust
    docs of IndustryPrior): r = 0.02 / 3, n = 4."""
    apriori = sum(LEVELS) / 4
    r = 0.02 / 3
    within2 = r / (1 - r / 4)
    var = sum((x - apriori) ** 2 for x in LEVELS) / 3
    between2 = (1 + var / apriori**2) / (1 + within2 / 4) - 1
    return apriori, math.sqrt(within2), math.sqrt(between2)


@pytest.fixture(scope="module")
def prior():
    return IndustryPrior.from_triangles(
        [company(x) for x in LEVELS], "paid", [[PREMIUM] * 4] * 4, min_developed=0.0
    )


def test_exact_companies_give_the_closed_form(prior):
    apriori, within, between = closed_form()
    assert prior.apriori == pytest.approx(apriori, abs=1e-12)
    assert prior.within_cv == pytest.approx(within, abs=1e-12)
    assert prior.cv == pytest.approx(within, abs=1e-12)
    assert prior.between_cv == pytest.approx(between, abs=1e-12)
    assert prior.common_cv == prior.between_cv
    assert prior.process_cv == pytest.approx(0.0, abs=1e-6)
    assert prior.company_loss_ratios == pytest.approx(LEVELS, abs=1e-12)
    assert prior.apriori_se == pytest.approx(
        math.sqrt(sum((x - apriori) ** 2 for x in LEVELS) / 3 / 4), abs=1e-12
    )
    # Identical patterns: the log share ratios' sd is floored at 0.01.
    assert prior.pattern.ages == [12, 24, 36, 48]
    assert prior.pattern.mean == pytest.approx([math.log(s / 0.4) for s in SHARES[1:]])
    assert prior.pattern.sd == [0.01] * 3
    assert prior.percent_developed == pytest.approx([0.4, 0.7, 0.9, 1.0])
    report = prior.report
    assert (report["n_companies"], report["n_companies_used"], report["n_origins_used"]) == (
        4,
        4,
        16,
    )
    assert report["origins_used"] == [4] * 4
    assert report["companies_by_age"] == [4] * 3
    assert any("only 4 companies" in w for w in report["warnings"])
    assert repr(prior).startswith("IndustryPrior(apriori=0.7500")


def test_settings_segments_and_errors(prior):
    # min_developed 0.5 leaves out the newest origin (0.4 developed).
    default = IndustryPrior.from_triangles(
        [company(x) for x in LEVELS], "paid", [[PREMIUM] * 4] * 4
    )
    assert default.report["origins_used"] == [3] * 4
    warnings = IndustryPrior.from_triangles(
        [company(x) for x in LEVELS], "paid", [[PREMIUM] * 4] * 4, min_companies=3
    ).report["warnings"]
    assert not any("only" in w for w in warnings)
    # The identical patterns' sd is floored, with a warning per age.
    assert sum("set to 0.01" in w for w in warnings) == 3
    # One triangle with a segment per company, premium from a column.
    origin, ages, paid, names = [], [], [], []
    for k, x in enumerate(LEVELS):
        o, a, p = rows(x)
        origin += o
        ages += a
        paid += p
        names += [f"c{k}"] * len(p)
    stacked = Triangle.from_long(
        origin,
        ages,
        {"paid": paid, "premium": [PREMIUM] * len(paid)},
        keys={"company": names},
    )
    by_segment = IndustryPrior.from_segments(stacked, "paid", "premium", min_developed=0.0)
    assert by_segment.apriori == pytest.approx(prior.apriori, abs=1e-12)
    assert by_segment.between_cv == pytest.approx(prior.between_cv, abs=1e-12)
    with pytest.raises(ValueError, match="at least two companies"):
        IndustryPrior.from_triangles([company(0.6)], "paid", [[PREMIUM] * 4])
    with pytest.raises(ValueError, match="premiums"):
        IndustryPrior.from_triangles([company(0.6), company(0.7)], "paid", [[PREMIUM] * 4])
    with pytest.raises(ValueError, match="min_developed"):
        IndustryPrior.from_triangles(
            [company(0.6), company(0.7)], "paid", [[PREMIUM] * 4] * 2, min_developed=2.0
        )


def test_the_pattern_prior_and_the_cut():
    p = PatternPrior.from_shares([12, 24, 36], [0.5, 0.3, 0.2], [0.1, 0.2])
    assert p.shares() == pytest.approx([0.5, 0.3, 0.2], abs=1e-12)
    assert p.percent_developed() == pytest.approx([0.5, 0.8, 1.0], abs=1e-12)
    assert p.sd == [0.1, 0.2]
    assert PatternPrior([12, 24], [0.0], [1.0]).shares() == pytest.approx([0.5, 0.5])
    assert p.truncated(24).mean == pytest.approx([math.log(0.6)])
    with pytest.raises(ValueError, match="positive"):
        PatternPrior([12, 24], [0.0], [0.0])
    with pytest.raises(ValueError):
        p.truncated(12)
    assert repr(p).startswith("PatternPrior(ages=[12, 24, 36]")


def test_the_model_carries_the_prior(prior):
    base = BayesBfNormalised(chains=1, tune=100, draws=100, n_sims=200, seed=1)
    model = prior.model(base)
    assert (model.apriori, model.cv, model.common_cv) == (
        prior.apriori,
        prior.cv,
        prior.between_cv,
    )
    assert (model.chains, model.n_sims, model.seed) == (1, 200, 1)
    assert model.pattern_prior.mean == prior.pattern.mean
    assert prior.to_bf_normalised_settings().pattern_prior.ages == [12, 24, 36, 48]
    assert "pattern_prior=PatternPrior" in repr(model)
    assert BayesBfNormalised().pattern_prior is None
    # A layer's prior keeps the base model's pattern prior, as its fit does.
    layered = LayerSharePrior(0.65, 0.2, apriori_cv=0.1, share_cv=0.3).model(model)
    assert layered.pattern_prior.mean == prior.pattern.mean
    # A cut at 36 months: the loss ratio developed to it, the tail factor.
    cut = prior.at_age(36)
    assert cut.apriori == pytest.approx(prior.apriori * 0.9, abs=1e-12)
    assert cut.tail_factor == pytest.approx(1 / 0.9, abs=1e-12)
    assert cut.pattern.ages == [12, 24, 36]
    assert any("36 months" in w for w in cut.report["warnings"])


def test_a_fit_under_the_industry_prior_takes_its_pattern(prior):
    """A short chain on a program whose cells wobble about the industry
    pattern (so its ODP scale is positive): the pattern prior (sd 0.01)
    holds the posterior pattern at the industry shares."""
    base = BayesBfNormalised(chains=2, tune=500, draws=500, n_sims=200, seed=1)
    origin, ages, paid = [], [], []
    noise = [1.04, 0.97, 1.02, 0.99, 0.96, 1.03, 1.01, 0.98, 1.02, 0.97]
    k = 0
    for w in range(4):
        cum = 0.0
        for d in range(4 - w):
            cum += PREMIUM * 0.7 * SHARES[d] * noise[k]
            k += 1
            origin.append(2010 + w)
            ages.append(12 * (d + 1))
            paid.append(cum)
    tri = Triangle.from_long(origin, ages, {"paid": paid, "premium": [PREMIUM] * len(paid)})
    fit = prior.fit(tri, "paid", "premium", model=base)
    assert isinstance(fit, BayesBfNormalisedFit)
    assert fit.pattern == pytest.approx(SHARES, rel=0.02)
    # The same through BayesBfNormalised(pattern_prior=...).
    model = BayesBfNormalised(
        apriori=prior.apriori,
        cv=prior.cv,
        common_cv=prior.common_cv,
        chains=2,
        tune=500,
        draws=500,
        n_sims=200,
        seed=1,
        pattern_prior=prior.pattern,
    )
    again = model.fit(tri, "paid", "premium")
    assert again.pattern == fit.pattern
    # Ages other than the triangle's are an error that names the cut.
    with pytest.raises(ValueError, match="truncated"):
        prior.at_age(36).fit(tri, "paid", "premium", model=base)
