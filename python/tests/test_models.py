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
    # The GAM's AIC counts its edf; at λ → ∞ it is the line's.
    line = Glm("gaussian").fit(d, y)
    assert fixed.log_likelihood == pytest.approx(line.log_likelihood, abs=1e-4)
    assert fit.aic == pytest.approx(-2 * fit.log_likelihood + 2 * fit.edf, rel=1e-14)
    assert GamFit.from_json(fit.to_json()).log_likelihood == fit.log_likelihood
    assert fit.reml_scale is None
    # REML: a noisy curve, its dispersion estimated with the smoothing.
    noisy = [v + 0.1 * math.sin(97.0 * i) for i, v in enumerate(y)]
    reml = Gam(Glm("gaussian"), [("x", 12)], smoothing="reml").fit(d, noisy)
    assert 0.002 < reml.reml_scale < 0.02 and 4 < reml.edf < 12


def test_gam_cyclic_smooth():
    t = [(i + 0.5) / 20 for i in range(240)]
    y = [2.0 + math.sin(math.tau * v / 12) for v in t]
    d = Design([[1.0] * 240, t], ["(Intercept)", "month"])
    fit = Gam(Glm("gaussian"), [{"column": "month", "bs": "cp", "period": (0.0, 12.0)}]).fit(d, y)
    ends = fit.predict(Design([[1.0, 1.0], [0.0, 12.0]], ["(Intercept)", "month"]))
    assert abs(ends[0] - ends[1]) < 1e-12 and abs(ends[0] - 2.0) < 0.01
    # The period wraps: month 13 is month 1.
    p = fit.predict(Design([[1.0, 1.0], [1.0, 13.0]], ["(Intercept)", "month"]))
    assert abs(p[0] - p[1]) < 1e-12
    with pytest.raises(ValueError):
        Gam(Glm("gaussian"), [{"column": "month", "bs": "zz"}])
    with pytest.raises(ValueError):
        Gam(Glm("gaussian"), [{"column": "month", "period": (0.0, 12.0)}])


def test_gam_monotone_smooth():
    x = [i / 199 for i in range(200)]
    y = [v * v + 0.05 * math.sin(1.7 * i) for i, v in enumerate(x)]
    d = Design([[1.0] * 200, x], ["(Intercept)", "x"])
    up = Gam(Glm("gaussian"), [{"column": "x", "bs": "mpi"}]).fit(d, y)
    assert all(b >= a - 1e-12 for a, b in zip(up.fitted, up.fitted[1:]))
    down = Gam(Glm("gaussian"), [{"column": "x", "bs": "mpd"}]).fit(d, [-v for v in y])
    assert all(b <= a + 1e-12 for a, b in zip(down.fitted, down.fitted[1:]))
    with pytest.raises(ValueError):
        Gam(Glm("gaussian"), [{"column": "x", "bs": "mpi"}], smoothing="reml").fit(d, y)


def test_gam_tensor_smooth():
    n = 200
    x = [((i * 37) % 101) / 100 for i in range(n)]
    z = [((i * 53) % 97) / 96 for i in range(n)]
    y = [math.sin(3 * a) * math.cos(2 * b) + 0.05 * math.sin(91.0 * i)
         for i, (a, b) in enumerate(zip(x, z))]
    d = Design([[1.0] * n, x, z], ["(Intercept)", "x", "z"])
    fit = Gam(Glm("gaussian"), [], smoothing="reml", tensors=[["x", "z"]]).fit(d, y)
    assert len(fit.lambdas) == 2 and fit.names[1] == "te(x,z).1"
    assert max(abs(a - b) for a, b in zip(fit.fitted, y)) < 0.2
    sized = Gam(Glm("gaussian"), [], tensors=[(["x", "z"], [5, 6])], smoothing=[1.0, 1.0]).fit(d, y)
    assert len(sized.names) == 1 + 29


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


def test_posterior_predictive_checks():
    from prospicio.models import BayesGlm, loo_pit, ppc_check

    x = [(i % 4) - 1.5 for i in range(80)]
    y = [[1.0, 2.0, 3.0, 5.0][i % 4] for i in range(80)]
    d = Design([[1.0] * 80, x], ["(Intercept)", "x"])
    fit = BayesGlm("poisson", chains=2, tune=300, draws=300, seed=4).fit(d, y)
    reps = fit.replicate(d, seed=2)
    assert len(reps) == 600 and len(reps[0]) == 80
    assert reps == fit.replicate(d, seed=2)
    checks = fit.posterior_predictive_check(d, y, ["mean", "sd", 0.9, "proportion_zero"], seed=2)
    assert [c["name"] for c in checks] == ["mean", "sd", "q0.9", "proportion_zero"]
    assert 0.05 < checks[0]["p_value"] < 0.95
    # These counts are far less spread than a Poisson's: the observed sd
    # sits low in its replicates, and the zeros never appear.
    assert checks[1]["p_value"] > 0.95
    assert checks[3]["observed"] == 0.0
    # The same checks from the replicates directly.
    again = ppc_check(y, reps, ["mean", "sd", 0.9, "proportion_zero"])
    assert [c["p_value"] for c in again] == [c["p_value"] for c in checks]
    grouped = ppc_check(y, reps, ["mean"], groups=[i % 4 for i in range(80)])
    assert [c["name"] for c in grouped] == ["mean[0]", "mean[1]", "mean[2]", "mean[3]"]
    p = fit.loo_pit(d, y, seed=2)
    assert len(p["pit"]) == 80 and all(0.0 <= v <= 1.0 for v in p["pit"])
    direct = loo_pit(y, reps, fit.log_likelihood(d, y), seed=2)
    assert len(direct["pareto_k"]) == 80 and direct["ks"] >= 0.0
    with pytest.raises(ValueError):
        ppc_check(y, reps, ["median"])
    with pytest.raises(ValueError):
        ppc_check(y, [r[:10] for r in reps])


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


def _quine():
    import csv
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "validation" / "data" / "quine.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    names = ["EthN", "SexM", "AgeF1", "AgeF2", "AgeF3", "LrnSL"]
    columns = [[1.0] * len(rows)] + [[float(r[n]) for r in rows] for n in names]
    return Design(columns, ["(Intercept)", *names]), [float(r["Days"]) for r in rows]


def test_negative_binomial_theta_matches_mass_glm_nb():
    # MASS 7.3-60: glm.nb(Days ~ Eth + Sex + Age + Lrn, quine).
    from prospicio.models import NegativeBinomialGlm

    design, y = _quine()
    fit = NegativeBinomialGlm().fit(design, y)
    assert fit.theta == pytest.approx(1.27489264581101, rel=1e-6)
    assert fit.se_theta == pytest.approx(0.161035178973597, rel=1e-5)
    assert fit.log_likelihood == pytest.approx(-546.575509144992, rel=1e-9)
    assert fit.aic == pytest.approx(2 * 546.575509144992 + 2 * 8, rel=1e-9)
    assert fit.glm.coefficients[0] == pytest.approx(2.89458001692052, rel=1e-6)
    assert len(fit.predict(design)) == len(y)


def test_tweedie_power_is_the_profile_maximum():
    from prospicio.models import TweedieGlm, tweedie_profile

    design, y = _quine()
    fit = TweedieGlm().fit(design, y)
    lo, hi = fit.interval
    assert lo < fit.power < hi
    profile = tweedie_profile(design, y, [1.1, 1.3, 1.5, 1.7, 1.9])
    assert profile["power"] == pytest.approx(fit.power, abs=1e-3)
    assert profile["max_log_likelihood"] == pytest.approx(fit.log_likelihood, rel=1e-6)
    assert max(profile["log_likelihood"]) <= fit.log_likelihood + 1e-9


def test_nested_tests_bic_and_vif():
    import csv
    import pathlib

    from prospicio.models import f_test, likelihood_ratio_test, vif

    path = pathlib.Path(__file__).resolve().parents[2] / "validation" / "data" / "quine.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    names = ["EthN", "SexM", "AgeF1", "AgeF2", "AgeF3", "LrnSL"]

    def design(k):
        cols = [[1.0] * len(rows)] + [[float(r[n]) for r in rows] for n in names[:k]]
        return Design(cols, ["(Intercept)", *names[:k]])

    y = [math.log(float(r["Days"]) + 1.0) for r in rows]
    small = Glm("gaussian", "identity").fit(design(2), y)
    big = Glm("gaussian", "identity").fit(design(6), y)
    # R: anova(small, big, test = "F") and test = "Chisq".
    f = f_test(small, big)
    assert f["statistic"] == pytest.approx(1.2950554093611, rel=1e-10)
    assert f["p_value"] == pytest.approx(0.274969956093846, rel=1e-9)
    assert (f["df"], f["df_resid"]) == (4.0, 139.0)
    lr = likelihood_ratio_test(small, big)
    assert lr["p_value"] == pytest.approx(0.26930041235005, rel=1e-9)
    assert lr["df_resid"] == math.inf
    days = [float(r["Days"]) for r in rows]
    assert Glm("poisson").fit(design(6), days).bic == pytest.approx(2320.06887663732, rel=1e-11)
    v = vif(design(6))
    assert math.isnan(v[0])
    assert v[3] == pytest.approx(2.09706084577139, rel=1e-11)
    with pytest.raises(ValueError):
        f_test(big, small)


def test_model_charts():
    from prospicio.models import confusion_matrix, double_lift, loss_ratio_chart, normalized_gini, roc

    r = roc([0.0, 0.0, 1.0, 1.0], [0.1, 0.4, 0.35, 0.8])
    assert r["auc"] == 0.75 and r["thresholds"][0] == math.inf
    y = [float(i % 2) for i in range(40)]
    score = [((i * 37) % 41) / 41 + 0.3 * y[i] for i in range(40)]
    ng = normalized_gini(y, score)
    assert roc(y, score)["auc"] == pytest.approx(0.5 * (1 + ng), rel=1e-12)
    c = confusion_matrix(y, score, 0.5)
    assert c["true_positive"] + c["false_negative"] == 20
    losses = [1.0, 0.0, 3.0, 2.0, 5.0, 1.0]
    dl = double_lift(losses, [1, 1, 2, 2, 4, 1], [2, 1, 2, 1, 3, 2], bands=2)
    assert sum(b["exposure"] * b["actual_relativity"] for b in dl) == pytest.approx(6.0)
    lr = loss_ratio_chart(losses, [1, 1, 2, 2, 4, 1], [2.0] * 6, bands=2)
    assert lr[1]["loss_ratio"] == pytest.approx(lr[1]["actual"] / lr[1]["premium"])


def test_glm_diagnostics_match_r():
    # R: glm(Days ~ ., poisson, quine), rows 1, 50 and 146; R's working
    # weights are those of its last iteration, so the tolerance is loose.
    import csv
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "validation" / "data" / "quine.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    names = ["EthN", "SexM", "AgeF1", "AgeF2", "AgeF3", "LrnSL"]
    d = Design([[1.0] * len(rows)] + [[float(r[n]) for r in rows] for n in names], ["(Intercept)", *names])
    y = [float(r["Days"]) for r in rows]
    fit = Glm("poisson").fit(d, y)
    g = fit.diagnostics(d, y)
    want = {
        "deviance": [-6.01850067312018, 1.73227012015060, 5.22406442044183],
        "leverage": [0.1043236171522393, 0.0450395922904775, 0.0355182959781139],
        "cooks_distance": [0.3963575691741380, 0.0248951016628828, 0.2198985132111394],
    }
    for key, values in want.items():
        assert [g[key][i] for i in (0, 49, 145)] == pytest.approx(values, rel=1e-6), key
    se = fit.prediction_se(d)
    assert [se[i] for i in (0, 49, 145)] == pytest.approx(
        [1.620656205333161, 0.698117923072381, 0.694808641916209], rel=1e-6
    )
    assert sum(r * r for r in g["deviance"]) == pytest.approx(fit.deviance, rel=1e-10)
    e = fit.sum_prediction_error(d.select([0]))
    assert e["rmsep"] ** 2 == pytest.approx(e["parameter_variance"] + e["process_variance"])
    n, p = len(y), len(fit.coefficients)
    assert fit.gcv(d, y) == pytest.approx(n * sum(r * r for r in g["response"]) / (n - p) ** 2)
    assert fit.log_likelihood_at(d, y, 1.0) == pytest.approx(fit.log_likelihood, rel=1e-12)
def test_terms_build_rating_variables():
    data = {"age": [20.0, 30.0, 45.0, 60.0], "region": ["N", "S", "W", "S"], "size": ["small", "large", "medium", "small"]}
    coding = (
        Terms()
        .intercept()
        .interaction("region", "age")
        .poly("age", 2)
        .hinges("age", [25.0])
        .steps("age", [30.0])
        .bins("age", [30.0, 50.0])
        .log("age", 40.0)
        .ordinal("size", ["small", "medium", "large"])
        .fit(data)
    )
    assert coding.names == [
        "(Intercept)", "region[S]:age", "region[W]:age", "age", "age^2", "hinge(age,25)",
        "step(age,30)", "age[(30,50]]", "age[(50,Inf)]", "log(age/40)", "size[>=medium]", "size[>=large]",
    ]
    d = coding.design({"age": [70.0], "region": ["W"], "size": ["medium"]})
    row = [d.column(i)[0] for i in range(len(coding.names))]
    assert row[:9] == [1.0, 0.0, 70.0, 70.0, 4900.0, 45.0, 1.0, 0.0, 1.0]
    assert row[9] == pytest.approx(math.log(70 / 40))
    assert row[10:] == [1.0, 0.0]
    with pytest.raises(ValueError):
        Terms().hinges("age", [50.0, 25.0]).fit(data)
