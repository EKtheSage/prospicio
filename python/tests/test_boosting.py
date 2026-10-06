import math
import random

import pytest

from actuarialrs.boosting import Booster
from actuarialrs.models import Design, Glm, compare, deviance_score, k_fold, simulate_from_means

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
