"""LayerSharePrior, the Bayesian BF prior of an excess layer from the layer
share and its uncertainty (docs/design/reserving-excess.md, PR 4).

The prior's moments, the tight-prior limit and the widening of the
predictive interval are checked in
validation/tests/reserving_bayes_excess_prior.rs; these tests check the
binding and that Python gives the numbers the Rust test and the R twin give
(R/prospicio/tests/test-reserving-bayes-excess-prior.R): the same cv, the
same prior ultimates, the same layer exposure, and the same prior moments of
the total with a share error common to all origins (decision 25 of
docs/design/reserving-bayes.md, #518).
"""

import csv
import math
from pathlib import Path

import pytest

from prospicio.distributions import Pareto
from prospicio.reserving import (
    BayesBfNormalised,
    BayesBfNormalisedFit,
    BayesOneYearFit,
    ExcessLayer,
    LayerSharePrior,
    Triangle,
    layer_share,
)

VALIDATION = Path(__file__).resolve().parents[2] / "validation"
SHARE = 0.25


@pytest.fixture(scope="module")
def tri():
    """EVW's triangle: `paid` scaled by the share is the layer's losses and
    `prior` is the ground-up exposure."""
    with open(VALIDATION / "data" / "evw2012.csv", newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows],
        development=[int(r["development"]) for r in rows],
        values={
            "layer": [SHARE * float(r["paid"]) for r in rows],
            "prior": [float(r["prior"]) for r in rows],
        },
    )


def test_the_cv_is_that_of_a_product_of_independents():
    p = LayerSharePrior(0.65, SHARE, apriori_cv=0.1, share_cv=0.3)
    assert p.cv**2 == pytest.approx(1.01 * 1.09 - 1.0, abs=1e-14)
    assert p.layer_apriori(0) == pytest.approx(0.65 * SHARE, abs=1e-15)
    assert (p.apriori, p.apriori_cv, p.share_cv, p.shares) == (0.65, 0.1, 0.3, [SHARE])
    assert p.share_dependence == "common" and p.has_common_factor
    # One factor known: the other's cv alone.
    assert LayerSharePrior(0.5, 0.2, apriori_cv=0.0, share_cv=0.3).cv == pytest.approx(0.3)
    assert LayerSharePrior(0.5, 0.2, share_cv=math.inf).cv == math.inf
    assert repr(p).startswith("LayerSharePrior(apriori=0.65, shares=[0.25]")
    assert 'share_dependence="common"' in repr(p)


def test_the_prior_moments_of_the_total_are_exact():
    # The same exposures, shares and cvs as the R twin. Closed forms from
    # first principles, m_i the origins' prior ultimates.
    exposure = [100.0, 150.0, 220.0, 90.0]
    shares = [0.2, 0.25, 0.3, 0.2]
    ce, cs = 0.25, 0.7
    m = [0.6 * e * s for e, s in zip(exposure, shares)]
    total, squares = sum(m), sum(x * x for x in m)
    common = LayerSharePrior(0.6, shares, apriori_cv=ce, share_cv=cs)
    mean, sd = common.prior_total(exposure)
    assert mean == pytest.approx(total, abs=1e-12)
    want = ce**2 * (1 + cs**2) * squares + cs**2 * total**2
    assert sd == pytest.approx(math.sqrt(want), rel=1e-12)
    per = LayerSharePrior(
        0.6, shares, apriori_cv=ce, share_cv=cs, share_dependence="per_origin"
    )
    assert per.share_dependence == "per_origin" and not per.has_common_factor
    want = ((1 + ce**2) * (1 + cs**2) - 1) * squares
    assert per.prior_total(exposure)[1] == pytest.approx(math.sqrt(want), rel=1e-12)
    assert sd > per.prior_total(exposure)[1]
    # The same numbers in the R twin.
    assert (mean, sd) == pytest.approx((84.9, 61.2), rel=1e-3)
    assert LayerSharePrior(0.6, shares, share_cv=math.inf).prior_total(exposure)[1] == math.inf
    with pytest.raises(ValueError, match="exposure"):
        common.prior_total([1.0, 2.0])
    with pytest.raises(ValueError, match="share_dependence"):
        LayerSharePrior(0.6, shares, share_dependence="shared")


def test_the_three_routes_to_a_share():
    a = LayerSharePrior.from_share_se(0.6, 0.25, 0.05)
    assert a.share_cv == pytest.approx(0.2) and a.apriori_cv == 0.1
    sev, layer = Pareto(1.0, 2.0), ExcessLayer(4.0)
    assert layer_share(sev, layer) == pytest.approx(0.125, abs=1e-12)
    b = LayerSharePrior.from_severity(0.6, sev, layer, share_cv=0.2)
    assert b.shares[0] == pytest.approx(0.125, abs=1e-12)
    per_origin = LayerSharePrior(0.6, [0.1 * (k + 1) for k in range(3)])
    assert per_origin.layer_apriori(2) == pytest.approx(0.6 * 0.3)
    with pytest.raises(ValueError, match="out of range"):
        per_origin.layer_apriori(3)


def test_errors():
    with pytest.raises(ValueError, match="apriori"):
        LayerSharePrior(0.0, 0.2)
    with pytest.raises(ValueError, match="share"):
        LayerSharePrior(0.5, 1.5)
    with pytest.raises(ValueError, match="share_cv"):
        LayerSharePrior(0.5, 0.2, share_cv=-1.0)
    with pytest.raises(TypeError, match="shares"):
        LayerSharePrior(0.5, "x")


def test_the_layer_exposure_and_the_model(tri):
    p = LayerSharePrior(0.5, SHARE, apriori_cv=0.1, share_cv=0.3)
    layered = p.triangle(tri, "prior")
    assert "layer_exposure" in layered.columns
    base = BayesBfNormalised(seed=3, n_sims=500)
    m = p.model(base)
    # A common share error: the loss ratio's cv, and the share's on the
    # multiplier.
    assert (m.apriori, m.cv, m.common_cv, m.seed, m.n_sims) == (0.5, 0.1, 0.3, 3, 500)
    per = LayerSharePrior(0.5, SHARE, apriori_cv=0.1, share_cv=0.3, share_dependence="per_origin")
    pm = per.model(base)
    assert (pm.cv, pm.common_cv) == (pytest.approx(p.cv), None)
    # A known loss ratio cannot be shared.
    with pytest.raises(ValueError, match="apriori_cv"):
        LayerSharePrior(0.5, SHARE, apriori_cv=0.0, share_cv=0.3).model(base)
    fit = p.fit(tri, "layer", "prior", model=base)
    assert isinstance(fit, BayesBfNormalisedFit)
    # The multiplier is the last parameter of the fit.
    assert fit.parameter_names[-1] == "log_share"
    assert per.fit(tri, "layer", "prior", model=base).parameter_names[-1] != "log_share"
    # The prior ultimate is the ground-up exposure times apriori times the
    # share: the same numbers as the Rust and R tests.
    ground_up = BayesBfNormalised(n_sims=100).fit(
        tri, "layer", "prior"
    )
    assert fit.exposure == pytest.approx([SHARE * e for e in ground_up.exposure])
    assert fit.prior_ultimate == pytest.approx([0.5 * SHARE * e for e in ground_up.exposure])
    # The same as fitting the model by hand on the layer exposure.
    by_hand = m.fit(layered, "layer", "layer_exposure")
    assert fit.reserves.mean() == pytest.approx(by_hand.reserves.mean())


def test_the_share_uncertainty_widens_the_interval(tri):
    base = BayesBfNormalised(seed=3, n_sims=2000)
    for dependence in ("common", "per_origin"):
        sds = []
        for share_cv in (1e-3, 0.5, 1.2):
            p = LayerSharePrior(
                0.35, SHARE, apriori_cv=0.1, share_cv=share_cv, share_dependence=dependence
            )
            fit = p.fit(tri, "layer", "prior", model=base)
            sds.append(fit.reserves.variance() ** 0.5)
        # A share error widens the interval over a known share; independent
        # priors keep widening it, a common one is learned from all origins
        # (the Rust validation reports the numbers).
        assert sds[0] < sds[1]
        if dependence == "per_origin":
            assert sds[1] < sds[2]


def test_the_one_year_view(tri):
    p = LayerSharePrior(0.5, SHARE, apriori_cv=0.1, share_cv=0.3)
    base = BayesBfNormalised(seed=1, n_sims=300)
    view = p.one_year(tri, "layer", "prior", model=base, refit="reweight_only")
    assert isinstance(view, BayesOneYearFit)
    assert len(view.pareto_k) == 300
    assert view.cdr.variance() < view.fit.reserves.variance()
