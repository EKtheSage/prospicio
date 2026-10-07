import math

import pytest

from prospicio.models import (
    compare,
    cross_validate,
    deviance_score,
    grid_search,
    log_uniform,
    random_search,
    ElasticNet,
    Design,
    Gam,
    GamFit,
    ElasticNetFit,
    Glm,
    GlmFit,
    Terms,
    crps,
    deviance,
    gini,
    group_k_fold,
    k_fold,
    lift,
    mcmc_diagnostics,
    time_ordered,
)

DATA = {
    "region": ["N", "N", "S", "S", "W", "W"],
    "exposure": [1.0, 2.0, 1.0, 1.0, 0.5, 1.5],
}
CLAIMS = [1.0, 2.0, 4.0, 2.0, 1.0, 3.0]


def test_poisson_glm_on_factors_is_the_observed_rates():
    coding = Terms().intercept().factor("region").fit(DATA)
    assert coding.names == ["(Intercept)", "region[S]", "region[W]"]
    offset = [math.log(e) for e in DATA["exposure"]]
    d = coding.design(DATA, offset=offset)
    fit = Glm("poisson", "log").fit(d, CLAIMS)
    # Saturated in region: each rate is claims / exposure in the region.
    assert abs(fit.coefficients[0] - math.log(3 / 3)) < 1e-10
    assert abs(fit.coefficients[1] - math.log(6 / 2)) < 1e-10
    assert abs(fit.coefficients[2] - math.log(4 / 2)) < 1e-10
    assert fit.dispersion == 1.0
    assert len(fit.std_errors) == 3 and all(se > 0 for se in fit.std_errors)
    assert abs(fit.aic - (-2 * fit.log_likelihood + 6)) < 1e-12
    # Predictions on new data, and an unseen level is an error.
    new = coding.design({"region": ["W"], "exposure": [2.0]}, offset=[math.log(2.0)])
    assert abs(fit.predict(new)[0] - 4.0) < 1e-9
    with pytest.raises(ValueError):
        coding.design({"region": ["E"], "exposure": [1.0]})
    pd = fit.predict_distribution(new, 2000, 1)
    assert pd.n_sims == 2000


def test_quasi_poisson_and_family_parameters():
    coding = Terms().intercept().factor("region").fit(DATA)
    d = coding.design(DATA)
    fit = Glm("poisson", dispersion="pearson").fit(d, CLAIMS)
    assert fit.dispersion != 1.0
    with pytest.raises(ValueError):
        Glm("tweedie")
    assert "tweedie" in repr(Glm("tweedie", "log", power=1.5))


def test_gam_smooths_a_curve():
    x = [i / 99 for i in range(100)]
    y = [math.sin(6 * v) + 2.0 for v in x]
    d = Design([[1.0] * 100, x], ["(Intercept)", "x"])
    fit = Gam(Glm("gaussian"), [("x", 12)]).fit(d, y)
    assert max(abs(a - b) for a, b in zip(fit.predict(d), y)) < 0.02
    assert fit.names[0] == "(Intercept)" and fit.names[1] == "s(x).1"
    fixed = Gam(Glm("gaussian"), ["x"], smoothing=[1e8]).fit(d, y)
    assert abs(fixed.edf - 2.0) < 1e-4


def test_metrics_and_splits():
    assert deviance("gaussian", [1.0, 3.0], [2.0, 2.0]) == 2.0
    assert abs(gini([0.0, 1.0], [0.1, 0.9]) - 0.5) < 1e-15
    bands = lift([0.0, 1.0, 2.0, 3.0], [0.1, 0.9, 2.1, 2.9], bands=2)
    assert [round(b["actual"]) for b in bands] == [1, 5]
    assert crps([3.0], 1.0) == 2.0
    folds = k_fold(10, 5, 1)
    assert sorted(i for _, test in folds for i in test) == list(range(10))
    for train, test in group_k_fold(["a", "a", "b", "c", "c", "d"], 2, 3):
        assert not {0, 1} & set(train) or not {0, 1} & set(test)
    assert time_ordered([0, 1, 2, 1, 2, 2], 1) == [([0, 1, 3], [2, 4, 5])]


def test_mcmc_diagnostics():
    a = [float((i * 37) % 101) for i in range(400)]
    b = [float((i * 53 + 7) % 101) for i in range(400)]
    d = mcmc_diagnostics([a, b])
    assert d["rhat"] < 1.01 and d["ess_bulk"] > 100
    shifted = [v + 200.0 for v in b]
    assert mcmc_diagnostics([a, shifted])["rhat"] > 1.5


def test_elastic_net_path():
    x1 = [i / 4 for i in range(40)]
    x2 = [math.sin(7.3 * i) for i in range(40)]
    y = [2.0 + 0.8 * a + 0.3 * b + math.cos(3.1 * i) for i, (a, b) in enumerate(zip(x1, x2))]
    d = Design([[1.0] * 40, x1, x2], ["(Intercept)", "x1", "x2"])
    net = ElasticNet("gaussian", alpha=0.5)
    lams = net.lambda_path(d, y, n=6, min_ratio=1e-3)
    assert len(lams) == 6 and lams[0] == net.lambda_max(d, y)
    fits = net.path(d, y, lams)
    assert all(abs(b) < 1e-10 for b in fits[0].coefficients[1:])
    assert fits[-1].df == 2 and fits[-1].deviance < fits[0].deviance
    # At lam = 0 it is the GLM.
    glm = Glm("gaussian").fit(d, y)
    zero = net.with_lam(0.0).fit(d, y)
    assert all(abs(a - b) < 1e-8 for a, b in zip(zero.coefficients, glm.coefficients))
    assert len(zero.predict(d)) == 40
    pd = zero.predict_distribution(d, 50, 1)
    assert pd.n_sims == 50
    with pytest.raises(ValueError):
        ElasticNet("gaussian", alpha=2.0).fit(d, y)


def test_cross_validation_and_search():
    x1 = [i / 4 for i in range(60)]
    x2 = [math.sin(7.3 * i) for i in range(60)]
    y = [2.0 + 0.8 * a + math.cos(3.1 * i) for i, (a, b) in enumerate(zip(x1, x2))]
    d = Design([[1.0] * 60, x1, x2], ["(Intercept)", "x1", "x2"])
    splits = k_fold(60, 5, 1)
    score = deviance_score("gaussian")
    glm_scores = cross_validate(Glm("gaussian"), d, y, splits, score)
    assert len(glm_scores) == 5 and all(s > 0 for s in glm_scores)
    # Threads give the same scores as one thread.
    assert cross_validate(Glm("gaussian"), d, y, splits, score, n_jobs=1) == glm_scores
    # The elastic net's own path cross-validation agrees with refitting.
    net = ElasticNet("gaussian", alpha=1.0)
    lams = net.lambda_path(d, y, n=8, min_ratio=1e-3)
    cv = net.cross_validate(d, y, lams, splits)
    assert cv.lam_1se >= cv.lam_min and len(cv.mean) == 8 and len(cv.fold_scores) == 5
    k = 3
    refit = cross_validate(net.with_lam(lams[k]), d, y, splits, score)
    fold_w = [len(t) for _, t in splits]
    weighted = sum(w * s for w, s in zip(fold_w, refit)) / sum(fold_w)
    assert abs(weighted - cv.mean[k]) < 1e-6 * cv.mean[k]
    found = grid_search(lams, lambda lam: net.with_lam(lam), d, y, splits, score)
    assert found.best_candidate == found.scores[found.best][0]
    rnd = random_search(5, 1, lambda rng: log_uniform(rng, 1e-3, 1.0),
                        lambda lam: net.with_lam(lam), d, y, splits, score)
    assert len(rnd.scores) == 5 and all(1e-3 <= c <= 1.0 for c, _ in rnd.scores)


def test_pit_and_log_score():
    from prospicio.models import ks_uniform, log_score, pit, pit_from_draws, pit_histogram

    import random

    rng = random.Random(3)
    # Poisson(3) counts by inversion.
    def draw(lam):
        u, k, p = rng.random(), 0, math.exp(-lam)
        c = p
        while u > c:
            k += 1
            p *= lam / k
            c += p
        return float(k)

    y = [draw(3.0) for _ in range(2000)]
    p = pit("poisson", y, [3.0] * 2000, seed=1)
    assert ks_uniform(p) < 1.63 / math.sqrt(2000)
    assert ks_uniform(pit("poisson", y, [4.5] * 2000)) > 3 / math.sqrt(2000)
    assert sum(pit_histogram(p, 10)) == 2000
    assert log_score("poisson", y, [3.0] * 2000) < log_score("poisson", y, [4.5] * 2000)
    assert pit_from_draws([1.0, 2.0, 3.0, 4.0], 2.5) == 0.5


def test_elpd_loo_and_waic():
    import random

    from prospicio.models import elpd_loo, elpd_waic, lppd

    rng = random.Random(1)
    y = [rng.gauss(0, 1) for _ in range(20)] + [5.0]
    draws = [rng.gauss(sum(y) / len(y), 1 / math.sqrt(len(y))) for _ in range(400)]
    ll = [[-0.5 * (yi - m) ** 2 - 0.5 * math.log(2 * math.pi) for yi in y] for m in draws]
    loo = elpd_loo(ll)
    waic = elpd_waic(ll)
    assert loo.elpd < lppd(ll) and abs(loo.ic + 2 * loo.elpd) < 1e-12
    assert len(loo.pareto_k) == 21 and waic.pareto_k is None
    # The outlier has the largest Pareto k and the lowest ELPD.
    assert max(range(21), key=lambda i: loo.pareto_k[i]) == 20
    assert min(range(21), key=lambda i: loo.pointwise[i]) == 20
    assert abs(loo.elpd - waic.elpd) < 0.5


def test_robust_covariance():
    # Two groups of a Poisson: each intercept's HC0 variance is
    # sum((y - ybar)**2) / (n ybar)**2.
    d = Design([[1.0] * 6, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], ["(Intercept)", "x"])
    y = [1.0, 2.0, 6.0, 1.0, 4.0, 2.0]
    fit = Glm("poisson").fit(d, y)
    v = fit.robust_covariance(d, y)
    assert v[0][0] == pytest.approx(14.0 / 81.0, rel=1e-10)
    assert v[0][1] == pytest.approx(v[1][0])
    hc1 = fit.robust_covariance(d, y, "HC1")
    assert hc1[1][1] == pytest.approx(v[1][1] * 6 / 4, rel=1e-12)
    # Singleton clusters are HC1.
    se = fit.robust_std_errors(d, y, "cluster", groups=["a", "b", "c", "d", "e", "f"])
    assert se == pytest.approx([math.sqrt(hc1[j][j]) for j in range(2)], rel=1e-10)
    with pytest.raises(ValueError):
        fit.robust_covariance(d, y, "cluster")
    with pytest.raises(ValueError):
        fit.robust_covariance(d, y, "HC3")
    with pytest.raises(ValueError):
        fit.robust_covariance(d, y[1:])


def test_compare_models():
    x = [i / 10 for i in range(60)]
    d = Design([[1.0] * 60, x], ["(Intercept)", "x"])
    y = [1.0 + 2.0 * v + ((i * 7) % 5 - 2) * 0.2 for i, v in enumerate(x)]
    splits = k_fold(60, 5, 3)
    mse = lambda t, p, _: sum((a - b) ** 2 for a, b in zip(t, p)) / len(t)
    models = {"glm": Glm("gaussian"), "lasso": ElasticNet("gaussian", lam=1.0)}
    c = compare(models, d, y, splits, {"mse": mse, "dev": deviance_score("gaussian")})
    assert c.models == ["glm", "lasso"] and c.metrics == ["mse", "dev"]
    assert c.split_scores[("glm", "mse")] == cross_validate(Glm("gaussian"), d, y, splits, mse)
    assert c.best("mse") == "glm"
    assert c.difference_std_error("glm", "mse") == 0.0
    assert c.difference_std_error("lasso", "mse") > 0.0
    rows = c.table()
    assert len(rows) == 4 and rows[0]["model"] == "glm" and rows[0]["metric"] == "mse"
    with pytest.raises(ValueError):
        compare({}, d, y, splits, {"mse": mse})


def test_glm_fit_save_and_load():
    import pickle

    d = Design([[1.0] * 6, [0.1, 0.7, 1.3, 2.2, 2.9, 3.4]], ["(Intercept)", "x"])
    y = [0.0, 2.3, 0.0, 4.4, 5.2, 6.0]
    fit = Glm("tweedie", power=1.4).fit(d, y)
    back = GlmFit.from_json(fit.to_json())
    assert back.coefficients == fit.coefficients and back.covariance == fit.covariance
    assert back.predict(d) == fit.predict(d)
    assert back.input_hash == fit.input_hash and fit.input_hash.startswith("blake3:")
    again = pickle.loads(pickle.dumps(fit))
    assert again.to_json() == fit.to_json()
    with pytest.raises(ValueError):
        GlmFit.from_json('{"format": "other"}')


def test_stacking_and_blending():
    from prospicio.distributions import PredictiveDistribution
    from prospicio.models import pseudo_bma_weights, stacking_weights

    a = [-0.1, -0.1, -3.0, -3.0]
    b = [-3.0, -3.0, -0.1, -0.1]
    assert stacking_weights([a, b]) == pytest.approx([0.5, 0.5], abs=1e-9)
    worse = [v - 1.0 for v in a]
    assert stacking_weights([a, worse]) == [1.0, 0.0]
    plain = pseudo_bma_weights([a, worse], bootstrap=False)
    assert plain[0] == pytest.approx(1 / (1 + math.exp(-4.0)))
    plus = pseudo_bma_weights([a, worse], n_draws=500, seed=3)
    assert sum(plus) == pytest.approx(1.0) and plus[0] > 0.5
    with pytest.raises(ValueError):
        stacking_weights([a])
    pa = PredictiveDistribution(["lob"], [("x",), ("y",)], [[1.0, -1.0]] * 200)
    pb = PredictiveDistribution(["lob"], [("x",), ("y",)], [[2.0, -2.0]] * 200)
    mix = PredictiveDistribution.blend([pa, pb], [1.0, 1.0], 5)
    assert all(t == 0.0 for t in mix.total().draws)
    with pytest.raises(ValueError):
        PredictiveDistribution.blend([pa, pb], [1.0], 5)


def test_gam_and_elastic_net_save_and_load():
    import pickle

    x = [i / 6 for i in range(60)]
    d = Design([[1.0] * 60, x], ["(Intercept)", "x"])
    y = [2.0 + math.sin(v) for v in x]
    gam = Gam(Glm("gaussian"), [("x", 8)]).fit(d, y)
    back = GamFit.from_json(gam.to_json())
    assert back.predict(d) == gam.predict(d) and back.input_hash == gam.input_hash
    assert pickle.loads(pickle.dumps(gam)).to_json() == gam.to_json()
    net = ElasticNet("gaussian", alpha=0.5, lam=0.05).fit(d, y)
    nback = ElasticNetFit.from_json(net.to_json())
    assert nback.coefficients == net.coefficients
    assert pickle.loads(pickle.dumps(net)).predict(d) == net.predict(d)
    with pytest.raises(ValueError):
        GamFit.from_json(net.to_json())


def test_actual_vs_expected():
    from prospicio.models import actual_vs_expected

    periods = ["2021", "2021", "2022", "2022", "2023", "2023", "2024", "2024"]
    y = [4.0, 6.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0]
    m = actual_vs_expected(periods, "poisson", y, [5.0] * 8)
    assert [p["period"] for p in m["periods"]] == ["2021", "2022", "2023", "2024"]
    assert [p["actual"] for p in m["periods"]] == [10.0, 11.0, 12.0, 13.0]
    assert m["trend"] == pytest.approx(0.1)
    assert m["trend_std_error"] == pytest.approx(math.sqrt(0.1 / 5))
    assert m["total"]["z"] == pytest.approx(6 / math.sqrt(40))
    g = actual_vs_expected(periods, "tweedie", y, [5.0] * 8, weights=[2.0] * 8, dispersion=0.5, power=1.5)
    assert g["periods"][0]["std_dev"] == pytest.approx(math.sqrt(0.5 * 4 * 5 ** 1.5))
    with pytest.raises(ValueError):
        actual_vs_expected(periods[:3], "poisson", y, [5.0] * 8)


def test_bayes_glm():
    from prospicio.models import BayesGlm, elpd_loo, stacking_weights

    x = [(i % 4) - 1.5 for i in range(80)]
    y = [[1.0, 2.0, 3.0, 5.0][i % 4] for i in range(80)]
    d = Design([[1.0] * 80, x], ["(Intercept)", "x"])
    spec = BayesGlm("poisson", chains=2, tune=300, draws=300, seed=4)
    fit = spec.fit(d, y)
    again = spec.fit(d, y)
    assert fit.coefficient_draws == again.coefficient_draws
    summary = fit.summary()
    assert [s["name"] for s in summary] == ["(Intercept)", "x"]
    assert all(s["rhat"] < 1.05 for s in summary)
    mle = Glm("poisson").fit(d, y)
    assert abs(fit.posterior_mean[1] - mle.coefficients[1]) < 0.5 * mle.std_errors[1]
    assert len(fit.coefficient_draws) == 600 and fit.divergences == 0
    loo = fit.loo(d, y)
    assert len(loo.pointwise) == 80
    ll = fit.log_likelihood(d, y)
    assert len(ll) == 600 and len(ll[0]) == 80
    assert elpd_loo(ll).elpd == pytest.approx(loo.elpd, abs=0.5)
    pd = fit.predict_distribution(d, 200, 1)
    assert pd.n_sims == 200
    g = BayesGlm("gaussian", chains=2, tune=300, draws=300).fit(d, y)
    assert g.summary()[-1]["name"] == "dispersion"
    flat = BayesGlm("poisson", chains=2, tune=300, draws=300).fit(
        Design([[1.0] * 80], ["(Intercept)"]), y)
    w = stacking_weights([loo.pointwise, flat.loo(Design([[1.0] * 80], ["(Intercept)"]), y).pointwise])
    assert w[0] > 0.9
    with pytest.raises(ValueError):
        BayesGlm("poisson", dispersion_scale=1.0, dispersion=None, draws=1).fit(d, y)


def test_bayesian_and_hierarchical_stacking():
    from prospicio.distributions import PredictiveDistribution
    from prospicio.models import BayesStacking, HierarchicalStacking

    x = [i / 99 - 0.5 for i in range(100)]
    a = [-0.5 if v < 0 else -2.0 for v in x]
    b = [-2.0 if v < 0 else -0.5 for v in x]
    fit = HierarchicalStacking(chains=2, tune=300, draws=300, seed=2).fit([a, b], [x])
    w = fit.weights([[-0.4, 0.0, 0.4]])
    assert len(w) == 3 and all(abs(sum(r) - 1) < 1e-12 for r in w)
    assert w[0][0] > 0.7 and w[2][0] < 0.3
    assert fit.divergences == 0 and all(r < 1.05 for r, _ in fit.rhat_ess())
    assert len(fit.alpha_draws) == 600 and len(fit.beta_draws) == 600
    bayes = BayesStacking(chains=2, tune=300, draws=300).fit([a, b])
    (row,) = bayes.weights()
    assert abs(row[0] - 0.5) < 0.1
    with pytest.raises(ValueError):
        HierarchicalStacking().fit([a, b], [x[:3]])
    pa = PredictiveDistribution(["lob"], [("x",), ("y",)], [[1.0, 1.0]] * 50)
    pb = PredictiveDistribution(["lob"], [("x",), ("y",)], [[2.0, 2.0]] * 50)
    mix = PredictiveDistribution.blend_by_component([pa, pb], [[1.0, 0.0], [0.0, 1.0]], 3)
    assert all(r == [1.0, 2.0] for r in mix.draw_matrix())


def test_hierarchical_stacking_with_partial_pooling():
    from prospicio.models import HierarchicalStacking

    n = 120
    region = [i % 4 for i in range(n)]
    dummies = [[float(g == r) for g in region] for r in (1, 2, 3)]
    x = [((i * 37) % 101) / 100 - 0.5 for i in range(n)]
    a = [-0.5 if g < 2 else -2.0 for g in region]
    b = [-2.0 if g < 2 else -0.5 for g in region]
    spec = HierarchicalStacking(
        discrete=3, partial_pooling=True, adaptive=4.0, chains=2, tune=300, draws=300, seed=9
    )
    fit = spec.fit([a, b], dummies + [x])
    # Region 0 (no dummy set) and region 3.
    w = fit.weights([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    assert w[0][0] > 0.7 and w[1][0] < 0.3
    with pytest.raises(ValueError):
        HierarchicalStacking(discrete=5).fit([a, b], [x])


def test_over_dispersed_poisson_accepts_negative_responses():
    # Log link and a group dummy: the fitted means are the group means.
    d = Design([[1.0] * 6, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], ["(Intercept)", "b"])
    y = [5.0, -1.0, 4.0, 2.0, 3.0, 4.0]
    fit = Glm("poisson", dispersion="pearson").fit(d, y)
    assert fit.coefficients[0] == pytest.approx(math.log(8 / 3), rel=1e-10)
    assert fit.coefficients[1] == pytest.approx(math.log(3 / (8 / 3)), rel=1e-10)
    with pytest.raises(ValueError):
        Glm("poisson").fit(d, y)


def test_glm_mean_preserving_parameter_draws():
    d = Design([[1.0] * 6, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], ["(Intercept)", "x"])
    fit = Glm("poisson", dispersion="pearson").fit(d, [3.0, 9.0, 1.0, 8.0, 20.0, 5.0])
    want = sum(fit.predict(d))
    centred = fit.predict_distribution(d, 20_000, 1, parameters="mean_preserving")
    fixed = fit.predict_distribution(d, 20_000, 1, parameters="fixed")
    normal = fit.predict_distribution(d, 20_000, 1)
    assert sum(centred.total().draws) / 20_000 == pytest.approx(want, rel=0.02)
    assert sum(fixed.total().draws) / 20_000 == pytest.approx(want, rel=0.02)
    assert sum(normal.total().draws) > sum(centred.total().draws)
    with pytest.raises(ValueError):
        fit.predict_distribution(d, 10, 1, parameters="other")
