import math

import pytest

from prospicio.distributions import Grid, Lognormal, PredictiveDistribution, Sampled
from prospicio.risk import (
    ArchimedeanCopula,
    Distortion,
    GaussianCopula,
    StudentTCopula,
    allocate,
    calibrate,
    capital,
    covar,
    entropic,
    esscher,
    esscher_allocation,
    iman_conover,
    marginal_expected_shortfall,
    simulate,
)

X = [10.0, 20.0, 30.0, 40.0, 50.0]


def test_distortions():
    s = Sampled(X)
    assert Distortion.tvar(0.6).measure(s) == pytest.approx(s.tvar(0.6), rel=1e-12)
    for d in (Distortion.wang(0.0), Distortion.proportional_hazard(1.0), Distortion.dual_power(1.0)):
        assert d.measure(s) == pytest.approx(30.0, rel=1e-12)
    assert Distortion.dual_power(2.0).g(0.3) == pytest.approx(0.51)
    assert sum(Distortion.wang(0.5).weights(100)) == pytest.approx(1.0)
    g = Grid(1.0, [0.5, 0.25, 0.25])
    assert Distortion.tvar(0.5).measure(g) == 1.5
    with pytest.raises(ValueError):
        Distortion.proportional_hazard(1.5)
    with pytest.raises(TypeError):
        Distortion.tvar(0.5).measure([1.0, 2.0])
    assert repr(Distortion.wang(0.25)) == "Distortion.wang(0.25)"


# Monograph 15's InsCo: ten equally likely totals, assets 100, priced at a
# 15% cost of capital (validation/scripts/aggregate_distortions.py).
INSCO = Sampled([22.0, 28.0, 36.0, 40.0, 40.0, 40.0, 40.0, 55.0, 65.0, 100.0])
INSCO_P = (46.6 + 0.15 * 100.0) / 1.15


def test_more_distortion_families():
    ccoc = Distortion.ccoc(0.15)
    assert ccoc.measure(INSCO) == pytest.approx(INSCO_P, rel=1e-12)
    assert ccoc.mass == pytest.approx(0.15 / 1.15)
    bt = Distortion.bitvar(0.0, 1.0, 0.15 / 1.15)
    assert bt.measure(INSCO) == pytest.approx(INSCO_P, rel=1e-12)
    others = [
        Distortion.weighted_tvar([0.1, 0.5, 0.9], [0.2, 0.5, 0.3]),
        Distortion.capped_linear(0.05, 1.4),
        Distortion.capped_log_linear(0.1, 0.8),
        Distortion.lep(0.02, 0.2),
        Distortion.linear_yield(0.03, 0.5),
        Distortion.beta(0.6, 1.5),
        Distortion.mixture([Distortion.wang(0.3), ccoc], [0.4, 0.6]),
        Distortion.minimum([Distortion.proportional_hazard(0.5), Distortion.tvar(0.6)]),
        Distortion.convex([(0.01, 0.05), (0.1, 0.25)]),
    ]
    for d in others:
        assert d.g(0.0) == 0.0 and d.g(1.0) == pytest.approx(1.0)
        assert 46.6 <= d.measure(INSCO) <= 100.0
        assert d.g_dual(0.3) <= 0.3 + 1e-12
        assert d.g(d.g_inv(0.4)) >= 0.4 - 1e-12
        assert repr(d).startswith("Distortion.")
    with pytest.raises(ValueError):
        Distortion.bitvar(0.9, 0.2, 0.5)
    with pytest.raises(ValueError):
        Distortion.mixture([ccoc], [0.5, 0.5])


def test_calibrate_matches_aggregate_on_insco():
    # aggregate 1.0.1's parameters, to its own calibration tolerance.
    want = {"ph": 0.7204792831878889, "wang": 0.3427309477594301, "dual": 1.5951514670652984}
    for family, param in want.items():
        d = calibrate(family, INSCO, INSCO_P)
        assert d.measure(INSCO) == pytest.approx(INSCO_P, rel=1e-12)
        assert d.g(0.3) == pytest.approx(
            {"ph": Distortion.proportional_hazard, "wang": Distortion.wang, "dual": Distortion.dual_power}[
                family
            ](param).g(0.3),
            rel=1e-6,
        )
    ccoc = calibrate("ccoc", INSCO, INSCO_P)
    assert repr(ccoc).startswith("Distortion.ccoc(0.15")
    # Capped at assets of 65, the premium must lie below 65.
    capped = calibrate("tvar", INSCO, 50.0, assets=65.0)
    assert capped.measure(Sampled([min(x, 65.0) for x in INSCO.draws])) == pytest.approx(50.0, rel=1e-9)
    with pytest.raises(ValueError):
        calibrate("tvar", INSCO, 70.0, assets=65.0)
    with pytest.raises(ValueError):
        calibrate("nope", INSCO, 50.0)


def test_allocation_adds_up():
    c = GaussianCopula([[1.0, 0.5], [0.5, 1.0]])
    pd = simulate(c, [Lognormal.from_mean_cv(100.0, 0.3), Lognormal.from_mean_cv(50.0, 1.5)], 5_000, 3)
    for d in (Distortion.tvar(0.99), Distortion.wang(0.5)):
        co = allocate(pd, d)
        assert len(co) == 2
        assert sum(co) == pytest.approx(d.measure(pd), rel=1e-9)


def test_copulas_and_simulation():
    r = [[1.0, 0.6], [0.6, 1.0]]
    for c in (GaussianCopula(r), StudentTCopula(r, 4.0), ArchimedeanCopula("gumbel", 2.0, 2)):
        u = c.sample(2_000, 5)
        assert len(u) == 2_000 and all(0.0 < x < 1.0 for row in u for x in row)
        assert c.sample(10, 5) == u[:10]
    with pytest.raises(ValueError):
        GaussianCopula([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError):
        ArchimedeanCopula("gauss", 2.0, 2)
    with pytest.raises(TypeError):
        simulate("not a copula", [], 10, 1)
    with pytest.raises(ValueError):
        simulate(GaussianCopula(r), [Lognormal(0.0, 1.0)], 10, 1)
    pd = simulate(GaussianCopula(r), [Lognormal(0.0, 1.0), Lognormal(0.0, 1.0)], 100, 1)
    assert pd.dims == ["component"]
    assert pd.components() == [(0,), (1,)]


def test_iman_conover_keeps_marginals():
    n = 4_000
    independent = GaussianCopula([[1.0, 0.0], [0.0, 1.0]])
    pd = simulate(independent, [Lognormal(0.0, 0.5), Lognormal(1.0, 1.0)], n, 9)
    joined = iman_conover(pd, [[1.0, 0.8], [0.8, 1.0]], 2)
    for k in [(0,), (1,)]:
        assert sorted(joined.marginal(k).draws) == sorted(pd.marginal(k).draws)
    a, b = joined.marginal((0,)).draws, joined.marginal((1,)).draws
    ranks = lambda v: {i: r for r, i in enumerate(sorted(range(n), key=v.__getitem__))}
    ra, rb = ranks(a), ranks(b)
    mean = (n - 1) / 2
    cov = sum((ra[i] - mean) * (rb[i] - mean) for i in range(n))
    var = sum((r - mean) ** 2 for r in range(n))
    assert cov / var == pytest.approx(6 / math.pi * math.asin(0.4), abs=0.02)


def test_evt():
    from prospicio.risk import Gpd, PotTail

    g = Gpd(0.25, 2.0)
    assert g.cdf(g.quantile(0.9)) == pytest.approx(0.9)
    fit = Gpd.fit([g.quantile((i - 0.5) / 2000) for i in range(1, 2001)])
    assert fit.xi == pytest.approx(0.25, abs=0.02)
    with pytest.raises(ValueError):
        Gpd.fit([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        Gpd(0.1, 0.0)
    s = Sampled([Lognormal(0.0, 1.0).quantile((i - 0.5) / 20_000) for i in range(1, 20_001)])
    tail = PotTail.fit(s, 0.9)
    assert tail.p_exceed == pytest.approx(0.1, abs=1e-3)
    assert tail.var(0.9999) > s.var(0.9999) * 0.5
    assert tail.tvar(0.99) > tail.var(0.99)
    with pytest.raises(ValueError):
        tail.var(0.5)


def test_capital_methods():
    pd = PredictiveDistribution(
        ["lob"],
        [("a",), ("b",), ("c",)],
        [[float((i * 37) % 101), float((i * 53) % 97), float((i * i) % 89)] for i in range(300)],
    )
    d = Distortion.tvar(0.9)
    euler = capital(pd, d)
    assert euler.method == "euler"
    assert euler.allocated == allocate(pd, d)
    for method in ["euler", "covariance", "proportional", "shapley"]:
        a = capital(pd, d, method)
        assert abs(sum(a.allocated) - a.total) < 1e-9
        assert abs(sum(a.diversification()) - a.diversification_benefit()) < 1e-9
    marginal = capital(pd, d, "marginal")
    assert sum(marginal.allocated) < marginal.total
    with pytest.raises(ValueError):
        capital(pd, d, "nope")


def test_exponential_utility_and_systemic_measures():
    k = 3.0
    u = [(i + 0.5) / 100_000 for i in range(100_000)]
    want = (math.exp(k) * (k - 1.0) + 1.0) / (k * math.expm1(k))
    assert Distortion.exponential(k).measure(Sampled(u)) == pytest.approx(want, abs=1e-6)
    assert repr(Distortion.exponential(k)) == "Distortion.exponential(3.0)"
    with pytest.raises(ValueError):
        Distortion.exponential(0.0)

    s = Sampled(X)
    assert entropic(s, 1e-9) == pytest.approx(30.0, abs=1e-6)
    assert entropic(X, 50.0) == pytest.approx(50.0, abs=0.1)
    assert esscher(s, 0.0) == pytest.approx(30.0)
    with pytest.raises(ValueError):
        entropic(X, 0.0)
    with pytest.raises(TypeError):
        esscher("x", 0.1)

    pd = PredictiveDistribution(
        ["lob"],
        [("a",), ("b",)],
        [[float((i * 37) % 101), float((i * 53) % 97)] for i in range(300)],
    )
    assert marginal_expected_shortfall(pd, 0.9) == allocate(pd, Distortion.tvar(0.9))
    alloc = esscher_allocation(pd, 0.02)
    assert sum(alloc) == pytest.approx(esscher(pd, 0.02))
    # At p = 0 every simulation is in distress: the unconditional VaR.
    assert covar(pd, ("a",), 0.0, 0.5) == pd.total().var(0.5)
    with pytest.raises(ValueError):
        covar(pd, ("z",), 0.9, 0.5)
