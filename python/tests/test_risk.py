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


def test_distortion_family_and_parameters():
    from prospicio.risk import Distortion

    d = Distortion.proportional_hazard(0.7)
    assert (d.family, d.parameters) == ("proportional_hazard", {"rho": 0.7})
    c = Distortion.capped_linear(0.1, 2.0)
    assert c.family == "capped_linear" and c.parameters == {"r0": 0.1, "slope": 2.0}
    assert Distortion.mixture([d, c], [0.5, 0.5]).parameters == {}


def test_copula_cdfs():
    import math

    from prospicio.risk import ArchimedeanCopula, GaussianCopula, StudentTCopula

    g = GaussianCopula([[1.0, 0.6], [0.6, 1.0]])
    assert g.cdf([0.5, 0.5]) == pytest.approx(0.25 + math.asin(0.6) / (2 * math.pi), abs=1e-15)
    assert g.cdf([0.3, 1.0]) == 0.3 and g.cdf([0.0, 0.7]) == 0.0
    c = ArchimedeanCopula("gumbel", 2.0, 3)
    want = math.exp(-sum((-math.log(u)) ** 2 for u in (0.4, 0.6, 0.7)) ** 0.5)
    assert c.cdf([0.4, 0.6, 0.7]) == pytest.approx(want, rel=1e-14)
    with pytest.raises(ValueError):
        c.cdf([0.4, 0.6])
    with pytest.raises(ValueError):
        StudentTCopula([[1.0, 0.5], [0.5, 1.0]], 4.0).cdf([0.5, 0.5])


def test_nested_archimedean_copula():
    from prospicio.risk import ArchimedeanCopula, NestedArchimedeanCopula

    tree = (1.5, [0, (3.0, [1, 2]), (2.0, [3, 4])])
    c = NestedArchimedeanCopula("gumbel", tree)
    assert c.dim == 5 and c.tree == tree
    # A coordinate at 1 does not constrain; one node is the flat copula.
    assert c.cdf([1.0, 0.5, 1.0, 1.0, 1.0]) == pytest.approx(0.5)
    flat = ArchimedeanCopula("clayton", 2.0, 3)
    one = NestedArchimedeanCopula("clayton", (2.0, [0, 1, 2]))
    assert one.sample(20, 4) == flat.sample(20, 4)
    assert one.cdf([0.3, 0.5, 0.8]) == pytest.approx(flat.cdf([0.3, 0.5, 0.8]), rel=1e-14)
    with pytest.raises(ValueError):
        NestedArchimedeanCopula("gumbel", (3.0, [0, (2.0, [1, 2])]))
    with pytest.raises(ValueError):
        NestedArchimedeanCopula("frank", (2.0, [0, 1]))
    # It joins marginals like any copula.
    from prospicio.distributions import Lognormal
    from prospicio.risk import simulate

    pd = simulate(c, [Lognormal.from_mean_cv(1.0, 0.5)] * 5, 2000, 3)
    assert pd.mean() == pytest.approx(5.0, rel=0.05)


def test_vine_copula():
    from prospicio.distributions import Lognormal
    from prospicio.risk import PairCopula, VineCopula, simulate

    g, c, f = PairCopula("gumbel", 2.0), PairCopula("clayton", 1.5, 90), PairCopula("frank", 3.0)
    v = VineCopula("d", [0, 1, 2], [[g, c], [f]])
    assert v.dim == 3 and v.kind == "d" and v.order == [0, 1, 2]
    u = [0.2, 0.7, 0.4]
    want = (math.log(g.density(0.2, 0.7)) + math.log(c.density(0.7, 0.4))
            + math.log(f.density(g.h2(0.2, 0.7), c.h1(0.4, 0.7))))
    assert v.log_density(u) == pytest.approx(want, rel=1e-12)
    # The inverse Rosenblatt transform of the first uniform alone is itself.
    assert v.inverse_rosenblatt([0.3, 0.5, 0.5])[0] == pytest.approx(0.3)
    assert c.h2_inverse(c.h2(0.3, 0.6), 0.6) == pytest.approx(0.3, abs=1e-12)
    with pytest.raises(ValueError):
        PairCopula("gaussian", 0.5, 90)
    with pytest.raises(ValueError):
        VineCopula("d", [0, 1, 2], [[g]])
    pd = simulate(v, [Lognormal.from_mean_cv(1.0, 0.5)] * 3, 2000, 5)
    assert pd.mean() == pytest.approx(3.0, rel=0.05)


def test_iman_conover_t_scores_and_shuffled_rows():
    import random

    from prospicio.distributions import PredictiveDistribution
    from prospicio.risk import iman_conover

    rng = random.Random(4)
    n = 5000
    rows = [[rng.lognormvariate(0, 0.5), rng.lognormvariate(0, 1.0)] for _ in range(n)]
    pd = PredictiveDistribution(["lob"], [(0,), (1,)], rows)
    r = [[1.0, 0.3], [0.3, 1.0]]

    def joint(p):
        a, b = p.marginal((0,)).draws, p.marginal((1,)).draws
        qa, qb = sorted(a)[n - n // 100], sorted(b)[n - n // 100]
        return sum(1 for x, y in zip(a, b) if x >= qa and y >= qb)

    normal = iman_conover(pd, r, seed=5)
    t2 = iman_conover(pd, r, seed=5, dof=2.0)
    assert joint(t2) > 1.5 * joint(normal)
    first = normal.marginal((0,)).draws
    assert first == sorted(first)
    shuffled = iman_conover(pd, r, seed=5, shuffle_rows=True)
    assert shuffled.marginal((0,)).draws != sorted(first)
    assert sorted(shuffled.marginal((0,)).draws) == sorted(first)
    with pytest.raises(ValueError):
        iman_conover(pd, r, seed=5, dof=2.0, scores=[0.0] * n)
def _simple_discrete():
    # Pricing Insurance Risk's Simple Discrete Example: 16 equally likely rows.
    x1 = {0: 2, 8: 1, 10: 1}
    x2 = {0: 2, 1: 1, 90: 1}
    rows = [[float(a), float(b)] for a, n in x1.items() for b, m in x2.items() for _ in range(n * m)]
    return PredictiveDistribution(["unit"], [("X1",), ("X2",)], rows)


def test_covariance_margin_keeps_the_mean():
    from prospicio.risk import Distortion, capital

    pd = _simple_discrete()
    d = Distortion.tvar(0.75)
    cov = capital(pd, d, "covariance")
    margin = capital(pd, d, "covariance_margin")
    assert margin.method == "covariance_margin"
    assert sum(margin.allocated) == pytest.approx(margin.total)
    for mean, c, m in zip((4.5, 22.75), cov.allocated, margin.allocated):
        assert m == pytest.approx(mean + (cov.total - 27.25) * c / cov.total)


def test_shapley_of_any_cost():
    import statistics

    from prospicio.risk import Distortion, capital, shapley

    pd = _simple_discrete()
    d = Distortion.tvar(0.75)
    want = capital(pd, d, "shapley")
    got = shapley(pd, lambda x: d.measure(Sampled(x)))
    assert got.allocated == pytest.approx(want.allocated)
    a = shapley(pd, lambda x: statistics.fmean(x) + 0.5 * statistics.pstdev(x))
    # Two units: phi_1 = (v(X1) + v(S) - v(X2)) / 2.
    assert a.allocated[0] == pytest.approx(0.5 * (a.standalone[0] + a.total - a.standalone[1]))
    with pytest.raises(ZeroDivisionError):
        shapley(pd, lambda x: 1 / 0)
    with pytest.raises(ValueError):
        shapley(pd, lambda x: float("nan"))


def test_haircut_and_equal_risk_match_pricing_insurance_risk():
    # Exercises 205 and 206: the premium of 80 by stand-alone TVaR.
    from prospicio.risk import equal_risk, haircut

    pd = _simple_discrete()
    h = haircut(pd, 80.0, "tvar", 0.75)
    assert h.measure == "tvar" and h.standalone == [10.0, 90.0]
    assert h.allocated == pytest.approx([8.0, 72.0])
    e = equal_risk(pd, 80.0, "tvar")
    assert e.level == pytest.approx(0.6796, abs=5e-5)
    assert e.allocated == pytest.approx([9.56, 70.44], abs=5e-3)
    for measure in ("var", "epd"):
        a = equal_risk(pd, 80.0, measure)
        assert sum(a.allocated) == pytest.approx(80.0)
    with pytest.raises(ValueError):
        equal_risk(pd, 1e6, "tvar")
    with pytest.raises(ValueError):
        haircut(pd, 80.0, "es", 0.5)


def test_worst_var_by_rearrangement():
    # Pricing Insurance Risk, Exercise 59, from the guide's rotated starts.
    from prospicio.risk import rearrange, worst_var

    m = [Lognormal.from_mean_cv(10.0, cv) for cv in (1.0, 2.0, 3.0)]
    for rotate, want in [(7, 353.8), (13, 353.6), (19, 352.9)]:
        assert worst_var(m, 0.99, n=40, rotate=rotate)["value"] == pytest.approx(want, abs=0.05)
    a = worst_var(m, 0.99, n=40, seed=1)
    assert a == worst_var(m, 0.99, n=40, seed=1)
    assert rearrange([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])["value"] == 5.0
    with pytest.raises(ValueError):
        rearrange([[1.0]], rotate=1, seed=1)


def test_statistics_match_numpy_linear_and_the_integral_tvar():
    np = pytest.importorskip("numpy")
    from prospicio.risk import statistics

    x = [7.0, 0.0, 100.0, 3.0, 1.0, 0.0, 45.0, 7.0, 12.0, 20.0]
    pct = [0, 10, 25, 50, 90, 99, 100]
    st = statistics(
        x, pct, tvar_levels=[0, 50, 90, 99], quantile="linear", tvar="linear"
    )
    for q in pct:
        assert st[f"Percentile {q}"] == pytest.approx(
            np.quantile(x, q / 100, method="linear"), abs=1e-12
        )
    # The integral of the linear quantile function over [p, 1], numerically.
    for q in [0, 50, 90, 99]:
        u = np.linspace(q / 100, 1, 200_001)
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        tvar = trapezoid(np.quantile(x, u, method="linear"), u) / (1 - q / 100)
        assert st[f"TVaR {q}"] == pytest.approx(tvar, rel=1e-6)
    assert st["Mean"] == 19.5
    assert st["SD"] == pytest.approx(np.std(x))
    assert st["Mean + 2 SD"] == pytest.approx(19.5 + 2 * np.std(x))
    # Sampled's own methods agree, and its defaults are unchanged.
    s = Sampled(x)
    assert s.var(0.25, method="linear") == st["Percentile 25"]
    assert s.tvar(0.9, method="linear") == st["TVaR 90"]
    assert s.var(0.25) == np.quantile(x, 0.25, method="inverted_cdf")
    with pytest.raises(ValueError):
        s.var(0.5, method="nearest")


def test_statistics_of_a_predictive_distribution_by_component():
    from prospicio.risk import statistics

    rows = [[float(i), 2.0 * i] for i in range(1, 11)]
    pd = PredictiveDistribution(["lob"], [(0,), (1,)], rows)
    cols = statistics(pd, [50], tvar_levels=[90], sd_multiples=[1])
    assert cols["lob"] == [0, 1]
    assert cols["Mean"] == [5.5, 11.0]
    assert cols["Percentile 50"] == [5.0, 10.0]
    assert cols["TVaR 90"] == pytest.approx([10.0, 20.0], rel=1e-12)
    assert list(cols)[:3] == ["lob", "Mean", "SD"]
