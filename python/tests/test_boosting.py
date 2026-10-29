import math
import random

import pytest

from prospicio.boosting import Booster
from prospicio.models import Design, Glm, compare, deviance_score, k_fold, simulate_from_means

np = pytest.importorskip("numpy")


def portfolio(n=2000, seed=11):
    """Claim counts with exposure: rate 0.1 for x < 0.5, 0.3 above."""
    rng = random.Random(seed)
    x = [rng.random() for _ in range(n)]
    exposure = [0.2 + 0.8 * rng.random() for _ in range(n)]
    y = []
    for v, e in zip(x, exposure):
        lam = (0.1 if v < 0.5 else 0.3) * e
        # Poisson by inversion.
        u, k, p = rng.random(), 0, math.exp(-lam)
        c = p
        while u > c:
            k += 1
            p *= lam / k
            c += p
        y.append(float(k))
    d = Design([x], ["x"], offset=[math.log(e) for e in exposure])
    return d, y, exposure


@pytest.mark.parametrize("engine", ["lightgbm", "xgboost"])
def test_offset_is_the_exposure(engine):
    pytest.importorskip(engine)
    d, y, exposure = portfolio()
    fit = Booster("poisson", engine=engine, n_rounds=150, learning_rate=0.1,
                  params={"max_depth": 2} if engine == "xgboost" else {"num_leaves": 3}).fit(d, y)
    mu = fit.predict(d)
    # Rates per unit exposure recover the two levels.
    low = [m / e for m, e, v in zip(mu, exposure, d.column(0)) if v < 0.45]
    high = [m / e for m, e, v in zip(mu, exposure, d.column(0)) if v > 0.55]
    assert abs(sum(low) / len(low) - 0.1) < 0.03
    assert abs(sum(high) / len(high) - 0.3) < 0.06
    # Doubling the exposure (offset + log 2) doubles every mean.
    d2 = Design([d.column(0)], ["x"], offset=[o + math.log(2.0) for o in d.offset])
    # XGBoost's margins are single precision.
    rel = 1e-9 if engine == "lightgbm" else 1e-5
    assert fit.predict(d2) == pytest.approx([2.0 * m for m in mu], rel=rel)


def test_lightgbm_adapter_is_the_engine():
    lgb = pytest.importorskip("lightgbm")
    d, y, _ = portfolio(500)
    fit = Booster("poisson", n_rounds=30, params={"num_leaves": 4}).fit(d, y)
    x = np.column_stack([d.column(0)])
    base = math.log(sum(y) / sum(math.exp(o) for o in d.offset))
    assert fit.base == pytest.approx(base, rel=1e-12)
    init = np.array(d.offset) + base
    data = lgb.Dataset(x, label=np.array(y), weight=np.ones(len(y)), init_score=init)
    raw = lgb.train({"objective": "poisson", "learning_rate": 0.05, "seed": 0, "deterministic": True,
                     "verbosity": -1, "num_leaves": 4}, data, num_boost_round=30)
    want = np.exp(raw.predict(x, raw_score=True) + init)
    assert fit.predict(d) == pytest.approx(want.tolist(), rel=1e-12)


@pytest.mark.parametrize("family,power", [("gamma", None), ("tweedie", 1.5), ("gaussian", None)])
def test_other_families_fit_and_estimate_dispersion(family, power):
    pytest.importorskip("lightgbm")
    rng = random.Random(3)
    x = [rng.random() for _ in range(800)]
    y = [rng.gammavariate(2.0, (100.0 + 100.0 * v) / 2.0) for v in x]
    d = Design([x], ["x"], offset=[0.0] * 800)
    fit = Booster(family, power=power, n_rounds=100, params={"num_leaves": 4}).fit(d, y)
    mu = fit.predict(d)
    assert abs(sum(mu) / sum(y) - 1.0) < 0.05
    if family == "gamma":
        # Gamma(shape 2): phi = 1 / shape.
        assert abs(fit.dispersion - 0.5) < 0.1
    assert fit.dispersion > 0


def test_predictive_distribution_and_bootstrap():
    pytest.importorskip("lightgbm")
    d, y, _ = portfolio(1000)
    plain = Booster("poisson", n_rounds=50).fit(d, y)
    boot = Booster("poisson", n_rounds=50, n_boot=8, seed=4).fit(d, y)
    p = plain.predict_distribution(d, 4000, 1)
    b = boot.predict_distribution(d, 4000, 1)
    assert p.components()[:2] == [(0,), (1,)]
    assert abs(p.mean() / sum(plain.predict(d)) - 1.0) < 0.02
    # Bootstrap refits add parameter uncertainty to the total.
    assert b.variance() > p.variance()


def test_simulate_from_means_and_compare():
    pd = simulate_from_means("gamma", [[100.0, 200.0]], 20_000, 2, dispersion=0.25)
    assert abs(pd.mean() / 300.0 - 1.0) < 0.02
    with pytest.raises(ValueError):
        simulate_from_means("poisson", [[-1.0]], 10, 1)
    pytest.importorskip("lightgbm")
    d, y, _ = portfolio(600)
    dx = Design([[1.0] * 600, d.column(0)], ["(Intercept)", "x"], offset=d.offset)
    c = compare({"glm": Glm("poisson"), "gbm": Booster("poisson", n_rounds=50)},
                dx, y, k_fold(600, 3, 1), {"deviance": deviance_score("poisson")})
    assert {row["model"] for row in c.table()} == {"glm", "gbm"}


def test_rejects_bad_specs():
    with pytest.raises(ValueError):
        Booster("binomial")
    with pytest.raises(ValueError):
        Booster("poisson", engine="catboost")
    with pytest.raises(ValueError):
        Booster("tweedie")


@pytest.mark.parametrize("engine", ["lightgbm", "xgboost"])
def test_quantile_objective(engine):
    pytest.importorskip(engine)
    from prospicio.boosting import predict_quantiles
    from prospicio.models import pinball_loss, pinball_score

    rng = random.Random(7)
    x = [rng.random() for _ in range(3000)]
    # Spread grows with x: the quantiles fan out.
    y = [100.0 + 200.0 * v * rng.gauss(0.0, 1.0) for v in x]
    d = Design([x], ["x"], offset=[0.0] * len(x))
    params = {"num_leaves": 4} if engine == "lightgbm" else {"max_depth": 2}
    fits = [Booster("quantile", engine=engine, alpha=a, n_rounds=200, learning_rate=0.1,
                    params=params).fit(d, y) for a in (0.9, 0.1, 0.5)]
    q = predict_quantiles(fits, d)
    assert list(q) == [0.1, 0.5, 0.9]
    assert all(lo <= mid <= hi for lo, mid, hi in zip(q[0.1], q[0.5], q[0.9]))
    inside = sum(lo <= v <= hi for v, lo, hi in zip(y, q[0.1], q[0.9])) / len(y)
    assert abs(inside - 0.8) < 0.04
    # The fitted 90% quantile beats the flat one it started from.
    flat = [fits[0].base] * len(y)
    assert pinball_loss(y, fits[0].predict(d), 0.9) < 0.9 * pinball_loss(y, flat, 0.9)
    assert pinball_score(0.9)(y, fits[0].predict(d), d) == pytest.approx(
        pinball_loss(y, fits[0].predict(d), 0.9))
    with pytest.raises(ValueError):
        fits[0].predict_distribution(d, 10, 1)


def test_quantile_rejects_bad_input():
    with pytest.raises(ValueError):
        Booster("quantile")
    with pytest.raises(ValueError):
        Booster("quantile", alpha=1.0)
    pytest.importorskip("lightgbm")
    d, y, _ = portfolio(200)
    with pytest.raises(ValueError, match="offset"):
        Booster("quantile", alpha=0.5, n_rounds=5).fit(d, y)


@pytest.mark.parametrize("engine", ["lightgbm", "xgboost"])
def test_dispersion_model(engine):
    pytest.importorskip(engine)
    rng = random.Random(5)
    x = [rng.random() for _ in range(4000)]
    # Gamma severities: shape 4 (phi 0.25) below x = 0.5, shape 1 (phi 1) above.
    y = [rng.gammavariate(4.0 if v < 0.5 else 1.0, 1000.0 / (4.0 if v < 0.5 else 1.0)) for v in x]
    d = Design([x], ["x"], offset=[0.0] * len(x))
    params = {"num_leaves": 4} if engine == "lightgbm" else {"max_depth": 2}
    fit = Booster("gamma", engine=engine, n_rounds=100, learning_rate=0.1, params=params,
                  dispersion_model=True).fit(d, y)
    phi = fit.predict_dispersion(d)
    low = [p for p, v in zip(phi, x) if v < 0.45]
    high = [p for p, v in zip(phi, x) if v > 0.55]
    assert abs(sum(low) / len(low) - 0.25) < 0.06
    assert abs(sum(high) / len(high) - 1.0) < 0.2
    # Per-row dispersion reaches the draws: the high-phi row is wider.
    two = Design([[0.2, 0.8]], ["x"], offset=[0.0, 0.0])
    pd = fit.predict_distribution(two, 20_000, 3)
    var = [pd.marginal((j,)).variance() for j in (0, 1)]
    mu = fit.predict(two)
    for v, m, p in zip(var, mu, fit.predict_dispersion(two)):
        assert v == pytest.approx(p * m * m, rel=0.05)
    # Without a dispersion model it is the constant.
    plain = Booster("gamma", engine=engine, n_rounds=20, params=params).fit(d, y)
    assert plain.predict_dispersion(two) == [plain.dispersion] * 2


def test_dispersion_model_needs_a_dispersion_family():
    with pytest.raises(ValueError):
        Booster("poisson", dispersion_model=True)


def _chalk():
    import csv
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples" / "data" / "chalk_portfolio.csv"
    with open(path) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    cols = ["age", "veh_age", "exper", "noise"]
    x = [[float(r[c]) for r in rows] for c in cols]
    off = [math.log(float(r["exposure"])) for r in rows]
    y = [float(r["claims"]) for r in rows]
    return Design(x, cols, offset=off), y


@pytest.mark.parametrize("engine", ["lightgbm", "xgboost"])
def test_monotone_constraints_on_the_chalk_portfolio(engine):
    """Typed monotone and interaction constraints (#575): unconstrained, the
    partial dependence on experience wiggles against its trend and on vehicle
    age with it; constrained, both are monotone on a fine grid, and the
    typed constraint is the engine's own raw parameter."""
    pytest.importorskip(engine)
    d, y = _chalk()
    params = {"num_leaves": 15} if engine == "lightgbm" else {"max_depth": 4}
    grid = list(np.linspace(min(d.column(2)), max(d.column(2)), 60))
    vgrid = list(np.linspace(min(d.column(1)), max(d.column(1)), 60))
    free = Booster("poisson", engine=engine, n_rounds=150, params=params, seed=1).fit(d, y)
    assert not free.check_monotone(d, "exper", grid=grid, direction=1)["monotone"]
    assert not free.check_monotone(d, "veh_age", grid=vgrid, direction=-1)["monotone"]
    spec = Booster("poisson", engine=engine, n_rounds=150, params=params, seed=1,
                   monotone={"exper": 1, "veh_age": -1},
                   interaction_constraints=[["age", "exper"], ["veh_age", "noise"]])
    fit = spec.fit(d, y)
    assert fit.check_monotone(d, "exper", grid=grid)["monotone"]
    assert fit.check_monotone(d, "veh_age", grid=vgrid)["monotone"]
    raw = spec.engine_params(d.names)
    if engine == "lightgbm":
        assert raw == {"monotone_constraints": [0, -1, 1, 0], "interaction_constraints": [[0, 2], [1, 3]]}
    else:
        assert raw == {"monotone_constraints": "(0,-1,1,0)", "interaction_constraints": "[[0, 2], [1, 3]]"}
    by_hand = Booster("poisson", engine=engine, n_rounds=150, params={**params, **raw}, seed=1).fit(d, y)
    assert fit.predict(d) == by_hand.predict(d)


def test_constraints_reject_factors_unknown_names_and_clashes():
    """A constraint on a factor, a polynomial or an interaction column, an
    unknown name and a clash with params are errors (#575)."""
    from prospicio.models import Terms

    data = {"si": [1.0, 2.0, 3.0, 4.0], "region": ["A", "B", "A", "B"], "age": [20.0, 30.0, 40.0, 50.0]}
    names = Terms().factor("region").numeric("si").poly("age", 2).hinges("si", [2.0]).fit(data).names
    assert names == ["region[B]", "si", "age", "age^2", "hinge(si,2)"]
    b = Booster("poisson", monotone={"si": 1}, interaction_constraints=[["region", "si"]])
    assert b.engine_params(names) == {"monotone_constraints": [0, 1, 0, 0, 1],
                                      "interaction_constraints": [[0, 1, 4]]}
    with pytest.raises(ValueError, match="factor"):
        Booster("poisson", monotone={"region": 1}).engine_params(names)
    with pytest.raises(ValueError, match="monotone transform"):
        Booster("poisson", monotone={"age": 1}).engine_params(names)
    with pytest.raises(ValueError, match="not a feature"):
        Booster("poisson", monotone={"bm_class": 1}).engine_params(names)
    inter = Terms().numeric("si").numeric("age").interaction("si", "age").fit(data).names
    with pytest.raises(ValueError, match="interaction"):
        Booster("poisson", monotone={"si": 1}).engine_params(inter)
    with pytest.raises(ValueError, match="params"):
        Booster("poisson", monotone={"si": 1}, params={"monotone_constraints": [1]})
    with pytest.raises(ValueError, match="1, -1 or 0"):
        Booster("poisson", monotone={"si": 2})
