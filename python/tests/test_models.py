import math

import pytest

from actuarialrs.models import (
    cross_validate,
    deviance_score,
    grid_search,
    log_uniform,
    random_search,
    ElasticNet,
    Design,
    Gam,
    Glm,
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
