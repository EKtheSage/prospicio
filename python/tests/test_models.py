import math
import pathlib

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
    Draws,
    draws_from_arrays,
    draws_from_inference_data,
    read_stan_csv,
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


def _validation():
    return pathlib.Path(__file__).resolve().parents[2] / "validation"


def _reference_rows(name):
    import csv

    with open(_validation() / "reference" / name) as f:
        return list(csv.DictReader(f))


def _near(actual, row):
    expected = float(row["expected"])
    tol = max(float(row["abs_tol"]), float(row["rel_tol"]) * abs(expected))
    assert abs(actual - expected) <= tol, (row, actual)


def test_stan_csv_matches_posterior_and_loo():
    # Real CmdStan output and the values posterior and loo give on it
    # (validation/scripts/r_stan_csv.R, #395).
    dirname = _validation() / "data" / "stan_eight_schools"
    d = read_stan_csv([dirname / f"chain-{c}.csv" for c in range(1, 5)])
    assert (d.n_chains, d.draws_per_chain, d.warmup_dropped) == (4, 250, 150)
    assert "divergent__" in d.sampler.names
    summary = {s["name"]: s for s in d.summary()}
    loo = d.loo()
    r_eff = d.relative_efficiency()
    est = {"elpd_loo": loo.elpd, "p_loo": loo.p, "looic": loo.ic, "se_elpd_loo": loo.se}
    for row in _reference_rows("stan_csv_posterior.csv"):
        v, q = row["variable"], row["quantity"]
        if v.startswith("chain "):
            _near(d.divergences[int(v[6:]) - 1], row)
        elif v == "log_lik":
            _near(est[q], row)
        elif v.startswith("log_lik[") and q in ("r_eff", "elpd_loo", "pareto_k"):
            i = int(v[8:-1]) - 1
            _near({"r_eff": r_eff, "elpd_loo": loo.pointwise, "pareto_k": loo.pareto_k}[q][i], row)
        else:
            _near(summary[v][q], row)


def test_draws_from_arrays_match_posterior():
    # The existing chain fixture through the common type: (chain, draw)
    # arrays, as an InferenceData's variables arrive.
    import csv

    with open(_validation() / "data" / "mcmc_chains.csv") as f:
        rows = list(csv.DictReader(f))
    names = [k for k in rows[0] if k not in ("chain", "iteration")]
    arrays = {n: [[float(r[n]) for r in rows if r["chain"] == str(c)] for c in range(4)] for n in names}
    d = draws_from_arrays(arrays)
    for row in _reference_rows("mcmc_posterior.csv"):
        _near(d.diagnostics(row["variable"])[row["quantity"]], row)


class _Group:
    """A stand-in for an xarray Dataset: data_vars and item access."""

    def __init__(self, **arrays):
        self._arrays = arrays
        self.data_vars = list(arrays)

    def __getitem__(self, name):
        return self._arrays[name]


class _InferenceData:
    def __init__(self, **groups):
        for k, v in groups.items():
            setattr(self, k, v)


def test_draws_from_inference_data_and_containers():
    chain = lambda c: [[0.1 * c + 0.01 * d, 1.0 + d] for d in range(6)]
    theta = [chain(c) for c in range(2)]  # (chain, draw, 2)
    mu = [[0.5 * c - 0.1 * d for d in range(6)] for c in range(2)]
    ll = [[[-1.0 - 0.01 * d, -2.0 + 0.02 * c] for d in range(6)] for c in range(2)]
    diverging = [[False, True, False, False, False, False], [False] * 6]
    idata = _InferenceData(
        posterior=_Group(mu=mu, theta=theta),
        log_likelihood=_Group(y=ll),
        sample_stats=_Group(diverging=diverging),
    )
    d = draws_from_inference_data(idata)
    assert d.names == ["mu", "theta[1]", "theta[2]", "log_lik[1]", "log_lik[2]"]
    assert d.divergences == [1, 0]
    assert d.chains("theta[2]")[1] == [1.0 + i for i in range(6)]
    assert d.log_likelihood()[0] == [-1.0, -2.0]
    assert len(d.loo().pointwise) == 2
    # Matrices name their elements row-major, 1-based.
    m = draws_from_arrays({"b": [[[[1.0, 2.0], [3.0, 4.0]]] * 4] * 2})
    assert m.names == ["b[1,1]", "b[1,2]", "b[2,1]", "b[2,2]"]
    assert m.chains("b[2,1]") == [[3.0] * 4] * 2
    with pytest.raises(ValueError):
        draws_from_arrays({"a": [[1.0] * 4] * 2, "b": [[1.0] * 5] * 2})
    with pytest.raises(ValueError):
        draws_from_inference_data(_InferenceData(posterior=_Group(mu=mu), log_likelihood=_Group(y=ll, z=ll)))
    with pytest.raises(ValueError):
        Draws(["a", "a"], [[[1.0, 2.0]] * 4])


def test_draws_from_arviz():
    az = pytest.importorskip("arviz")
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(1)
    groups = {
        "posterior": {"mu": rng.normal(size=(4, 200)), "theta": rng.normal(size=(4, 200, 3))},
        "log_likelihood": {"y": rng.normal(-1.0, 0.1, size=(4, 200, 5))},
        "sample_stats": {"diverging": np.zeros((4, 200), dtype=bool)},
    }
    try:
        idata = az.from_dict(groups)  # arviz 1.x: a DataTree
    except TypeError:
        idata = az.from_dict(**groups)  # arviz 0.x: an InferenceData
    d = draws_from_inference_data(idata)
    assert d.names[:4] == ["mu", "theta[1]", "theta[2]", "theta[3]"]
    assert d.n_chains == 4 and d.draws_per_chain == 200
    s = {r["name"]: r for r in d.summary()}
    ref = az.summary(idata, var_names=["mu"], round_to="none")
    assert abs(s["mu"]["mean"] - ref.loc["mu", "mean"]) < 1e-12
    assert abs(s["mu"]["ess_bulk"] - ref.loc["mu", "ess_bulk"]) < 1e-6 * ref.loc["mu", "ess_bulk"]
    assert len(d.loo().pointwise) == 5


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
    # Randomized quantile residuals (#464): a count's are reproducible by
    # seed; a gamma's are exact, as R's statmod::qresiduals.
    from prospicio.models import quantile_residuals

    q = fit.quantile_residuals(d, y, seed=3)
    assert q == fit.quantile_residuals(d, y, seed=3)
    assert q == pytest.approx(quantile_residuals("poisson", y, fit.predict(d), seed=3), rel=1e-12)
    y1 = [v + 1.0 for v in y]
    gamma = Glm("gamma", link="log").fit(d, y1)
    qg = gamma.quantile_residuals(d, y1)
    assert [qg[i] for i in (0, 49, 145)] == pytest.approx(
        [-1.566963914736516, 0.714809090957266, 1.470914293998306], rel=1e-6
    )


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


def test_orthogonal_polys_and_natural_splines():
    # R: poly(x, 3) and lm(y ~ ns(x, knots = c(10, 25))).
    x = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 37.0, 46.0]
    y = [math.sin(v / 7) * 10 + v / 3 for v in x]
    poly = Terms().orthogonal_poly("x", 3).fit({"x": x})
    assert poly.design({"x": [50.0]}).column(2)[0] == pytest.approx(1.26850684550873050, rel=1e-12)
    spline = Terms().intercept().natural_spline("x", [10.0, 25.0]).fit({"x": x})
    assert spline.names == ["(Intercept)", "spline(x)1", "spline(x)2", "spline(x)3"]
    fit = Glm("gaussian", "identity").fit(spline.design({"x": x}), y)
    new = fit.predict(spline.design({"x": [0.0, 20.0, 50.0]}))
    assert new == pytest.approx([-0.72122511127239264, 7.71926159866044337, 22.49421993091769423], rel=1e-9)


def test_target_encode_out_of_fold():
    from prospicio.models import target_encode

    values = [f"l{i % 7}" for i in range(60)]
    y = [float((i * 13) % 5) for i in range(60)]
    splits = k_fold(60, 5, 3)
    rows, enc = target_encode(values, y, [2.0] * 60, splits, 4.0)
    y2 = list(y)
    y2[10] += 100.0
    rows2, _ = target_encode(values, y2, [2.0] * 60, splits, 4.0)
    assert rows2[10] == rows[10]
    a = sum(y[i] for i in range(60) if i % 7 == 0)
    n = sum(1 for i in range(60) if i % 7 == 0)
    z = n / (n + 4.0)
    assert enc.encode(["l0", "unseen"]) == pytest.approx([z * a / (2.0 * n) + 1 - z, 1.0])
    with pytest.raises(ValueError):
        target_encode(values, y, [2.0] * 60, splits[1:], 4.0)


def _glmm_reference():
    import csv
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "validation"
    with open(root / "data" / "glmm_salamanders.csv") as f:
        rows = list(csv.DictReader(f))
    with open(root / "reference" / "glmm_r.csv") as f:
        ref = {
            (r["model"], r["quantity"], r["arg"]): float(r["expected"])
            for r in csv.DictReader(line for line in f if not line.startswith("#"))
            if r["dataset"] == "salamanders"
        }
    # Treatment coding with R's references: spp GP, mined yes.
    spp = ["PR", "DM", "EC-A", "EC-L", "DES-L", "DF"]
    names = ["(Intercept)"] + [f"spp{s}" for s in spp] + ["minedno"]
    columns = [[1.0] * len(rows)]
    columns += [[float(r["spp"] == s) for r in rows] for s in spp]
    columns.append([float(r["mined"] == "no") for r in rows])
    y = [float(r["count"]) for r in rows]
    return rows, ref, names, columns, y


def test_glmm_matches_glmmtmb_and_glmer():
    # validation/scripts/r_glmm.R: glmmTMB 1.1.15.2 and lme4 2.0.6 on
    # glmmTMB's Salamanders, count ~ spp + mined + (1 | site).
    from prospicio.models import Design, Glmm

    rows, ref, names, columns, y = _glmm_reference()
    design = Design(columns, names)
    site = [r["site"] for r in rows]
    fit = Glmm().fit(design, {"site": site}, y)
    assert fit.converged
    assert fit.names == names
    for name, b, se in zip(names, fit.coefficients, fit.standard_errors):
        assert b == pytest.approx(ref[("poisson", "coef", name)], rel=2e-5, abs=2e-6)
        assert se == pytest.approx(ref[("poisson", "se", name)], rel=2e-4)
    sd = ref[("poisson", "sd", "site")]
    assert fit.sd[0] == pytest.approx(sd, rel=2e-5)
    assert fit.sd_standard_errors[0] == pytest.approx(
        sd * ref[("poisson", "se_log_sd", "site")], rel=2e-4
    )
    assert fit.dispersion == 1.0
    assert fit.log_likelihood == pytest.approx(ref[("poisson", "loglik", "")], rel=1e-8)
    effects = fit.random_effects["site"]
    assert effects["terms"] == ["(Intercept)"]
    j = effects["levels"].index("R-1")
    assert effects["values"]["(Intercept)"][j] == pytest.approx(
        ref[("poisson", "ranef", "site|R-1")], rel=1e-4, abs=1e-5
    )
    assert effects["prediction_variance"]["(Intercept)"][j] == pytest.approx(
        ref[("poisson", "prediction_var", "site|R-1")], rel=1e-3
    )
    assert effects["conditional_variance"]["(Intercept)"][j] == pytest.approx(
        ref[("poisson_glmer", "conditional_var", "site|R-1")], rel=1e-3
    )
    # A level the fit has not seen, or no groups, predicts the population mean.
    row = Design([[c[0]] for c in columns], names)
    population = math.exp(sum(b * c[0] for b, c in zip(fit.coefficients, columns)))
    assert fit.predict(row, {"site": ["nowhere"]})[0] == pytest.approx(population, rel=1e-12)
    assert fit.predict(row)[0] == pytest.approx(population, rel=1e-12)
    u = effects["values"]["(Intercept)"][effects["levels"].index(site[0])]
    assert fit.predict(row, {"site": [site[0]]})[0] == pytest.approx(
        population * math.exp(u), rel=1e-12
    )
    assert fit.fitted[0] == pytest.approx(population * math.exp(u), rel=1e-9)

    agq = Glmm(nodes=9).fit(design, {"site": site}, y)
    assert agq.log_likelihood == pytest.approx(ref[("poisson_agq9", "loglik", "")], rel=1e-8)
    assert agq.sd[0] == pytest.approx(ref[("poisson_agq9", "sd", "site")], rel=2e-5)


def test_glmm_random_slopes_match_glmmtmb():
    # validation/scripts/r_glmm.R: glmmTMB 1.1.15.2 on the slopes book,
    # count ~ x + w + (1 + x | territory) + offset(log(exposure)), and the
    # slope independent of the intercept, (1 + x || territory).
    import csv
    import pathlib

    from prospicio.models import Design, Glmm

    root = pathlib.Path(__file__).resolve().parents[2] / "validation"
    with open(root / "data" / "glmm_slopes.csv") as f:
        rows = list(csv.DictReader(f))
    with open(root / "reference" / "glmm_r.csv") as f:
        ref = {
            (r["model"], r["quantity"], r["arg"]): float(r["expected"])
            for r in csv.DictReader(line for line in f if not line.startswith("#"))
            if r["dataset"] == "slopes"
        }
    x = [float(r["x"]) for r in rows]
    w = [float(r["w"]) for r in rows]
    offset = [math.log(float(r["exposure"])) for r in rows]
    design = Design([[1.0] * len(rows), x, w], ["(Intercept)", "x", "w"], offset=offset)
    y = [float(r["count"]) for r in rows]
    territory = [r["territory"] for r in rows]
    groups = {"territory": {"labels": territory, "slopes": {"x": x}}}
    fit = Glmm().fit(design, groups, y)
    assert fit.converged
    assert fit.group_names == ["territory", "territory"]
    assert fit.term_names == ["(Intercept)", "x"]
    for b, name in zip(fit.coefficients, fit.names):
        assert b == pytest.approx(ref[("poisson", "coef", name)], rel=2e-5, abs=2e-6)
    assert fit.sd == pytest.approx(
        [ref[("poisson", "sd", "territory")], ref[("poisson", "sd", "territory|x")]], rel=2e-5
    )
    name, terms, matrix = fit.correlations[0]
    assert (name, terms) == ("territory", ["(Intercept)", "x"])
    assert matrix[1][0] == pytest.approx(
        ref[("poisson", "cor", "territory|x|(Intercept)")], rel=2e-4
    )
    assert fit.log_likelihood == pytest.approx(ref[("poisson", "loglik", "")], rel=1e-8)
    effects = fit.random_effects["territory"]
    assert effects["terms"] == ["(Intercept)", "x"]
    j = effects["levels"].index("t07")
    assert effects["values"]["x"][j] == pytest.approx(
        ref[("poisson", "ranef", "territory|t07|x")], rel=1e-4, abs=1e-5
    )
    assert fit.predict(design, groups) == pytest.approx(fit.fitted, rel=1e-9)
    independent = {"territory": {"labels": territory, "slopes": {"x": x}, "correlated": False}}
    diag = Glmm().fit(design, independent, y)
    assert diag.log_likelihood == pytest.approx(ref[("poisson_diag", "loglik", "")], rel=1e-8)
    assert diag.random_effects["territory"]["terms"] == ["(Intercept)", "x"]
    assert diag.predict(design, independent) == pytest.approx(diag.fitted, rel=1e-9)
    with pytest.raises(ValueError, match="single term"):
        Glmm(nodes=5).fit(design, groups, y)
    with pytest.raises(ValueError, match="unknown key"):
        Glmm().fit(design, {"territory": {"labels": territory, "slope": {"x": x}}}, y)


def test_bayes_glmm_on_the_slopes_book_is_near_the_laplace_fit():
    # The NUTS fit of (1 + x | territory) on the slopes book: with 800 rows
    # its posterior means sit near the Laplace fit.
    import csv
    import pathlib

    from prospicio.models import BayesGlmm, Design

    root = pathlib.Path(__file__).resolve().parents[2] / "validation"
    with open(root / "data" / "glmm_slopes.csv") as f:
        rows = list(csv.DictReader(f))
    x = [float(r["x"]) for r in rows]
    w = [float(r["w"]) for r in rows]
    offset = [math.log(float(r["exposure"])) for r in rows]
    design = Design([[1.0] * len(rows), x, w], ["(Intercept)", "x", "w"], offset=offset)
    y = [float(r["count"]) for r in rows]
    groups = {"territory": {"labels": [r["territory"] for r in rows], "slopes": {"x": x}}}
    fit = BayesGlmm(chains=4, tune=500, draws=500, seed=2026).fit(design, groups, y)
    assert fit.divergences == 0
    assert fit.parameter_names == [
        "(Intercept)",
        "x",
        "w",
        "sd[territory|(Intercept)]",
        "sd[territory|x]",
        "cor[territory|x,(Intercept)]",
    ]
    summary = fit.summary()
    assert all(s["rhat"] < 1.05 for s in summary)
    laplace = fit.laplace
    for s, b in zip(summary, laplace.coefficients):
        assert abs(s["mean"] - b) < 0.2 * s["sd"]
    assert len(fit.parameter_draws) == 2000
    assert len(fit.effect_draws[0]) == len(fit.effect_names) == 60
    effects = fit.random_effects["territory"]
    assert effects["terms"] == ["(Intercept)", "x"]
    modes = laplace.random_effects["territory"]["values"]
    for term in effects["terms"]:
        for m, s, b in zip(effects["mean"][term], effects["sd"][term], modes[term]):
            assert abs(m - b) < 0.4 * s
    mu = fit.predict(design, groups)
    assert len(mu) == len(rows) and all(v > 0 for v in mu)


def test_double_glm_matches_corrected_dglm():
    # validation/scripts/r_dglm.R: dglm 1.8.6 with its mean step corrected,
    # on the simulated book; the gamma with the log link by REML and the
    # Poisson with an exposure offset by ML.
    import csv
    import math
    import pathlib

    from prospicio.models import Design, DoubleGlm

    root = pathlib.Path(__file__).resolve().parents[2] / "validation"
    with open(root / "data" / "glmm_simulated.csv") as f:
        rows = list(csv.DictReader(f))
    with open(root / "reference" / "dglm_r.csv") as f:
        ref = {
            (r["case"], r["method"], r["quantity"], r["arg"]): float(r["expected"])
            for r in csv.DictReader(line for line in f if not line.startswith("#"))
        }
    n = len(rows)
    x = [float(r["x"]) for r in rows]
    urban = [float(r["urban"] == "yes") for r in rows]
    names = ["(Intercept)", "x", "urbanyes"]
    mean = Design([[1.0] * n, x, urban], names)
    by_urban = Design([[1.0] * n, urban], ["(Intercept)", "urbanyes"])
    offset = [math.log(float(r["exposure"])) for r in rows]
    cases = [
        ("gamma", "reml", DoubleGlm("gamma", link="log", method="reml"), mean, mean, "severity"),
        (
            "poisson_offset",
            "ml",
            DoubleGlm("poisson", method="ml"),
            Design([[1.0] * n, x, urban], names, offset=offset),
            by_urban,
            "count",
        ),
    ]
    for case, method, spec, x_design, z_design, column in cases:
        fit = spec.fit(x_design, z_design, [float(r[column]) for r in rows])
        assert fit.method == method
        for name, b, se in zip(fit.names, fit.coefficients, fit.standard_errors):
            assert b == pytest.approx(ref[(case, method, "coef", name)], rel=1e-8)
            assert se == pytest.approx(ref[(case, method, "se", name)], rel=1e-8)
        for name, g, se in zip(
            fit.dispersion_names, fit.dispersion_coefficients, fit.dispersion_standard_errors
        ):
            assert g == pytest.approx(ref[(case, method, "dcoef", name)], rel=1e-8)
            assert se == pytest.approx(ref[(case, method, "dse", name)], rel=1e-8)
        assert fit.minus_two_log_likelihood == pytest.approx(
            ref[(case, method, "m2loglik", "")], rel=1e-10
        )
        assert fit.reml_criterion == pytest.approx(
            ref[(case, method, "reml_criterion", "")], rel=1e-10
        )
        assert fit.predict(x_design) == pytest.approx(fit.fitted, rel=1e-12)
        assert fit.predict_dispersion(z_design) == pytest.approx(fit.dispersion_fitted, rel=1e-12)


def test_double_glm_rejects_what_it_does_not_fit():
    from prospicio.models import Design, DoubleGlm

    with pytest.raises(ValueError, match="ml or reml"):
        DoubleGlm("gamma", method="pql")
    d = Design([[1.0] * 4], ["(Intercept)"])
    with pytest.raises(ValueError, match="double GLM takes"):
        DoubleGlm("binomial").fit(d, d, [0.0, 1.0, 1.0, 0.0])


def test_glmm_rejects_what_it_does_not_fit():
    from prospicio.models import Design, Glmm

    with pytest.raises(ValueError, match="poisson, gamma or tweedie"):
        Glmm("binomial")
    with pytest.raises(ValueError, match="tweedie needs power"):
        Glmm("tweedie")
    design = Design([[1.0] * 4], ["(Intercept)"])
    with pytest.raises(ValueError, match="grouping factor"):
        Glmm().fit(design, {}, [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="labels for"):
        Glmm().fit(design, {"g": ["a", "b"]}, [1.0, 2.0, 3.0, 4.0])


def test_bounded_and_relaxed_elastic_nets_match_glmnet():
    # validation/scripts/r_glmnet_relax.R: glmnet 5.1 on the policy data,
    # lasso; its gamma = 0 is gamma = 1e-5, as the files record.
    import csv
    import math
    import pathlib

    from prospicio.models import Design, ElasticNet

    root = pathlib.Path(__file__).resolve().parents[2] / "validation"
    with open(root / "data" / "glm_policies.csv") as f:
        rows = list(csv.DictReader(f))
    with open(root / "reference" / "elastic_net_relax_glmnet.csv") as f:
        ref = list(csv.DictReader(f))
    with open(root / "reference" / "elastic_net_relax_cv_glmnet.csv") as f:
        cv_ref = list(csv.DictReader(f))
    n = len(rows)
    age = [float(r["age"]) for r in rows]
    names = ["(Intercept)", "age", "age2", "region[B]", "region[C]", "region[D]"]
    columns = [[1.0] * n, age, [a * a / 100 for a in age]] + [
        [float(r["region"] == level) for r in rows] for level in "BCD"
    ]
    offset = [math.log(float(r["exposure"])) for r in rows]
    poisson = Design(columns, names, offset=offset)
    gaussian = Design(columns, names)
    claims = [float(r["claims"]) for r in rows]
    gauss = [float(r["gauss"]) for r in rows]
    inf = math.inf
    cases = {
        "poisson_nonneg": (ElasticNet("poisson", lower_limits=[0.0] * 6), poisson, claims),
        "gaussian_bounded": (
            ElasticNet(
                "gaussian",
                lower_limits=[-inf, -inf, -inf, -inf, -2.0, -inf],
                upper_limits=[inf, inf, 0.1, inf, inf, 5.0],
            ),
            gaussian,
            gauss,
        ),
        "poisson_nonneg_relaxed": (
            ElasticNet("poisson", lower_limits=[0.0] * 6),
            poisson,
            claims,
        ),
        "gaussian_relaxed": (ElasticNet("gaussian"), gaussian, gauss),
        "poisson_relaxed": (ElasticNet("poisson"), poisson, claims),
    }
    checked = 0
    for r in ref:
        if r["quantity"] != "coef":
            continue
        net, d, y = cases[r["case"]]
        fit = net.with_lam(float(r["arg"])).fit(d, y)
        if r["gamma"]:
            fit = fit.relax(d, y, float(r["gamma"]))
            assert fit.gamma == float(r["gamma"])
        got = fit.coefficients[fit.names.index(r["term"])]
        assert got == pytest.approx(float(r["expected"]), rel=1e-7, abs=1e-9)
        checked += 1
    assert checked == 192
    bounded = cases["gaussian_bounded"][0]
    assert bounded.upper_limits[2] == 0.1
    # A relaxed fit round-trips as a version 2 artifact.
    relaxed = ElasticNet("poisson", lam=0.01).fit(poisson, claims).relax(poisson, claims, 0.5)
    text = relaxed.to_json()
    assert '"format_version":2' in text
    assert ElasticNetFit.from_json(text).coefficients == relaxed.coefficients
    with pytest.raises(ValueError):
        relaxed.relax(poisson, claims, 0.0)

    splits = [
        ([i for i in range(n) if i % 5 != f], [i for i in range(n) if i % 5 == f])
        for f in range(5)
    ]
    for case in ["poisson_relaxed", "gaussian_relaxed"]:
        net, d, y = cases[case]
        mine = [r for r in cv_ref if r["case"] == case]
        lams = [float(r["arg"]) for r in mine if r["quantity"] == "mean" and r["gamma"] == "1"]
        gammas = sorted({float(r["gamma"]) for r in mine if r["gamma"]})
        cv = net.cross_validate_relaxed(d, y, lams, gammas, splits)
        assert cv.gammas == gammas and len(cv.paths) == 5
        for r in mine:
            q = r["quantity"]
            if q in ("mean", "se"):
                path = cv.paths[gammas.index(float(r["gamma"]))]
                got = getattr(path, q)[lams.index(float(r["arg"]))]
                assert got == pytest.approx(float(r["expected"]), rel=1e-6, abs=1e-12)
            else:
                got = getattr(cv, q.replace("lambda", "lam"))
                assert got == pytest.approx(float(r["expected"]), rel=1e-12)


# Model-agnostic explanations (#355) on a portfolio with a planted
# young-driver high-power interaction. R's test-models.R builds the same
# rows and checks the same pinned values.
def explain_portfolio():
    n = 300
    age = [18.0 + (37 * i) % 60 for i in range(n)]
    power = [["low", "mid", "high"][(i // 7) % 3] for i in range(n)]
    noise = [((13 * i) % 17) / 17 for i in range(n)]
    exposure = [0.5 + ((11 * i) % 10) / 20 for i in range(n)]
    y = []
    for i in range(n):
        young, high = age[i] < 25, power[i] == "high"
        mu = exposure[i] * math.exp(0.5 + 0.8 * young + 0.3 * high + 0.5 * (young and high))
        y.append(float(math.floor(mu * (1 + 0.6 * math.sin(1.7 * i)) + 0.5)))
    return {"age": age, "power": power, "noise": noise, "exposure": exposure}, y


def explain_design(f, interaction=False):
    young = [float(a < 25) for a in f["age"]]
    high = [float(p == "high") for p in f["power"]]
    cols = [[1.0] * len(young), young, [float(p == "mid") for p in f["power"]], high]
    names = ["(Intercept)", "young", "power[mid]", "power[high]"]
    if interaction:
        cols, names = cols + [[u * v for u, v in zip(young, high)]], names + ["young:high"]
    return Design(cols, names, offset=[math.log(e) for e in f["exposure"]])


def test_permutation_importance_is_the_hand_loop():
    from prospicio.models import permutation, permutation_importance

    frame, y = explain_portfolio()
    fit = Glm("poisson").fit(explain_design(frame), y)
    score = deviance_score("poisson")
    imp = permutation_importance(
        fit, frame, y, explain_design, score, ["age", "power", "noise", ["age", "power"]],
        n_repeats=2, seed=5,
    )
    assert imp.features == ["age", "power", "noise", "age+power"]

    def scored(f):
        d = explain_design(f)
        return score(y, fit.predict(d), d)

    base = scored(frame)
    for j, names in enumerate([["age"], ["power"], ["noise"], ["age", "power"]]):
        hand = 0.0
        for r in range(2):
            perm = permutation(len(y), 5, r)
            shuffled = {k: ([v[i] for i in perm] if k in names else v) for k, v in frame.items()}
            hand += (scored(shuffled) - base) / 2
        assert abs(imp.importance[j] - hand) < 1e-14
    # The GLM does not use noise, so shuffling it changes nothing.
    assert imp.importance[2] == 0.0
    assert imp.base == [base] and math.isnan(imp.std_error[0])
    assert imp.table()[0]["feature"] == "age+power"
    assert max(imp.relative) == 100.0
    # Pinned, and the same in R.
    assert abs(imp.importance[0] - 0.48670981297841526) < 1e-10
    assert abs(imp.importance[1] - 0.13778344497063177) < 1e-10


def test_explanations_average_over_folds():
    from prospicio.models import PermutationImportance, permutation_importance

    frame, y = explain_portfolio()
    splits = k_fold(len(y), 4, 2)
    score = deviance_score("poisson")
    imp = permutation_importance(
        Glm("poisson"), frame, y, explain_design, score, ["age", "power"], splits=splits, seed=1
    )
    assert len(imp.by_split) == 4 and len(imp.base) == 4
    # Fold 0 by hand: its fitted model explained on its test rows. Fold i of
    # a cross-validated run shuffles on stream i, a single fit on stream 0.
    train, test = splits[0]
    sub = lambda rows: {k: [v[j] for j in rows] for k, v in frame.items()}
    fit = Glm("poisson").fit(explain_design(sub(train)), [y[j] for j in train])
    one = permutation_importance(
        fit, sub(test), [y[j] for j in test], explain_design, score, ["age", "power"], seed=1
    )
    assert one.by_split[0] == imp.by_split[0] and one.base[0] == imp.base[0]
    both = PermutationImportance.combine([imp, imp])
    assert len(both.by_split) == 8 and both.importance == imp.importance
    assert abs(imp.importance[0] - 0.4464341484022761) < 1e-10


def test_partial_dependence_of_an_additive_glm_is_its_relativities():
    from prospicio.models import partial_dependence

    frame, y = explain_portfolio()
    fit = Glm("poisson").fit(explain_design(frame), y)
    pd = partial_dependence(fit, frame, explain_design, "power", ["low", "mid", "high"])
    rel = pd.relative("low")
    assert abs(rel[1] - math.exp(fit.coefficients[2])) < 1e-12
    assert abs(rel[2] - math.exp(fit.coefficients[3])) < 1e-12
    on_folds = partial_dependence(
        Glm("poisson"), frame, explain_design, "age", [20.0, 40.0], y=y, splits=k_fold(len(y), 3, 1)
    )
    assert len(on_folds.by_split) == 3 and on_folds.mean[0] > on_folds.mean[1]
    with pytest.raises(ValueError):
        partial_dependence(Glm("poisson"), frame, explain_design, "age", [20.0], splits=[([0], [1])])


def test_explanations_keep_numpy_columns():
    np = pytest.importorskip("numpy")
    from prospicio.models import partial_dependence, permutation_importance

    frame, y = explain_portfolio()
    arrays = {k: np.asarray(v) for k, v in frame.items()}

    def build(f):  # NumPy comparisons, as a design builder on arrays writes them
        assert isinstance(f["age"], np.ndarray)
        return explain_design({k: v.tolist() for k, v in f.items()})

    fit = Glm("poisson").fit(explain_design(frame), y)
    score = deviance_score("poisson")
    a = permutation_importance(fit, arrays, y, build, score, ["age", "noise"], seed=3)
    b = permutation_importance(fit, frame, y, explain_design, score, ["age", "noise"], seed=3)
    assert a.importance == b.importance
    pd = partial_dependence(fit, arrays, build, "age", [20.0, 40.0])
    assert pd.mean == partial_dependence(fit, frame, explain_design, "age", [20.0, 40.0]).mean


def test_interaction_screen_finds_the_planted_pair():
    from prospicio.models import interaction_screen

    frame, y = explain_portfolio()
    score = deviance_score("poisson")
    features = ["age", "power", "noise"]
    screens = {}
    for planted in [False, True]:
        build = lambda f, planted=planted: explain_design(f, interaction=planted)
        fit = Glm("poisson").fit(build(frame), y)
        screens[planted] = interaction_screen(
            fit, frame, y, build, score, features, n_repeats=20, seed=0
        )
    s = screens[True]
    assert s.pairs == [("age", "power"), ("age", "noise"), ("power", "noise")]
    # The deviance is not additive on the log scale, so even the additive
    # GLM's pair is not zero; the planted term lifts it above it.
    assert s.strength[0] > 0 > screens[False].strength[0]
    # Noise is unused: shuffling it with another feature is shuffling that one.
    assert s.strength[1] == 0.0 and s.strength[2] == 0.0
    row = next(r for r in s.table() if (r["a"], r["b"]) == ("age", "power"))
    assert abs(row["loss_a"] + row["loss_b"] - row["loss_ab"] - row["strength"]) < 1e-14
    assert abs(s.strength[0] - 0.020418701731271027) < 1e-10
    build = lambda f: explain_design(f, interaction=True)
    fit = Glm("poisson").fit(build(frame), y)
    only = interaction_screen(fit, frame, y, build, score, features, pairs=[("power", "age")],
                              n_repeats=20, seed=0)
    assert only.pairs == [("power", "age")] and abs(only.strength[0] - s.strength[0]) < 1e-14
    with pytest.raises(ValueError):
        interaction_screen(fit, frame, y, build, score, ["age"], pairs=[("age", "age")])


def test_permutation_importance_of_the_monographs_toy_table():
    # CAS Monograph 16, Tables 9.1 to 9.4: frequency is age / 100 and the
    # model has learned exactly that; seed 3 shuffles three rows as the
    # monograph does (ages 40, 20, 30).
    from prospicio.models import permutation, permutation_importance

    assert permutation(3, 3) == [2, 0, 1]

    class Toy:
        def predict(self, design):
            return [a / 100 for a in design.column(0)]

    frame = {"age": [20.0, 30.0, 40.0], "height": [1.4, 1.5, 1.6]}
    y = [0.2, 0.3, 0.4]
    build = lambda f: Design([list(f["age"]), list(f["height"])], ["age", "height"])
    mae = lambda yy, p, d: sum(abs(a - b) for a, b in zip(yy, p)) / len(yy)
    imp = permutation_importance(Toy(), frame, y, build, mae, ["age", "height"], seed=3)
    assert round(imp.importance[0], 3) == 0.133 and imp.importance[1] == 0.0
    assert imp.relative == [100.0, 0.0]


def test_mars_matches_earth_on_the_chalk_portfolio():
    # R earth 5.3: earth(claims / exposure ~ age + veh_age + exper + noise,
    # degree = 1, pmethod = "none"), unweighted and weighted by exposure
    # (validation/scripts/r_mars_earth.R).
    import csv
    from pathlib import Path

    from prospicio.models import Glm, Terms, mars

    path = Path(__file__).resolve().parents[2] / "examples" / "data" / "chalk_portfolio.csv"
    with open(path) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    names = ["age", "veh_age", "exper", "noise"]
    x = {n: [float(r[n]) for r in rows] for n in names}
    w = [float(r["exposure"]) for r in rows]
    y = [float(r["claims"]) / e for r, e in zip(rows, w)]
    fit = mars(x, y)
    assert fit.terms == [
        ("age", 28.0, "up"),
        ("age", 28.0, "down"),
        ("veh_age", 5.0, "up"),
        ("veh_age", 5.0, "down"),
    ]
    assert fit.rss == pytest.approx(7229.20770861623, rel=1e-10)
    assert fit.gcv == pytest.approx(0.36178591949671, rel=1e-10)
    assert fit.termination == "threshold"
    weighted = mars(x, y, weights=w, threshold=0.0, max_terms=15)
    assert weighted.knots("exper") == [2.0, 7.0, 26.0]
    assert weighted.rss == pytest.approx(5166.57719014687, rel=1e-10)
    # The knots feed Terms: a weighted refit on them has earth's RSS.
    terms = weighted.add_terms(Terms().intercept())
    design = terms.fit(x).design(x, weights=w)
    assert Glm("gaussian").fit(design, y).deviance == pytest.approx(5166.57719014687, rel=1e-9)
    pruned = mars(x, y, threshold=0.0, max_terms=15, prune=True)
    assert pruned.rss == pytest.approx(7223.67323614061, rel=1e-10)
    assert len(pruned.terms) == 6


def test_isotonic_recalibration_balance_and_murphy():
    """Auto-calibration checks and repairs (#545)."""
    import csv
    from pathlib import Path

    from prospicio.models import balance_correction, isotonic, murphy, recalibrate, reliability

    ref = Path(__file__).resolve().parents[2] / "validation" / "reference" / "isotonic_sklearn.csv"

    def nums(s):
        return [float(v) for v in s.split(";")] if s else None

    with open(ref) as f:
        for r in csv.DictReader(f):
            if r["case"] == "isotonic":
                fit = isotonic(nums(r["x"]), nums(r["y"]), nums(r["w"]), increasing=r["increasing"] == "1")
                assert fit.predict(nums(r["query"])) == pytest.approx(nums(r["expected"]), rel=1e-12, abs=1e-12)
            else:
                b = balance_correction(nums(r["y"]), nums(r["query"]), nums(r["w"]))
                assert b == pytest.approx(float(r["expected"]), rel=1e-14)
    m = recalibrate([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 1.0, 2.0], y_min=0.1)
    assert m.predict([1.0, 2.5, 4.0]) == pytest.approx([0.1, 1.0, 2.0])
    assert isotonic([1.0, 3.0], [1.0, 3.0], interpolation="step").predict([2.0]) == [1.0]
    y = [float((i * 37) % 11) for i in range(200)]
    pred = [1.0 + ((i * 37) % 11) * 0.7 + (i % 3) for i in range(200)]
    for family in ("gaussian", "poisson"):
        d = murphy(y, pred, family=family)
        assert d["score"] == pytest.approx(d["miscalibration"] - d["discrimination"] + d["uncertainty"], abs=1e-12)
        rc = isotonic(pred, y).predict(pred)
        assert abs(murphy(y, rc, family=family)["miscalibration"]) < 1e-12
    bands = reliability(y, pred, bins=4)
    assert sum(b["count"] for b in bands) == 200
    assert sum(b["weight"] * b["mean_actual"] for b in bands) == pytest.approx(sum(y))
    with pytest.raises(ValueError):
        isotonic([1.0], [1.0], interpolation="cubic")


def test_conformal_intervals_match_mapie_and_cover():
    """Conformal prediction intervals (#546)."""
    import csv
    from pathlib import Path

    from prospicio.models import (
        Design,
        Glm,
        conformal_cqr,
        conformal_intervals,
        conformal_plus,
        conformal_split,
        k_fold,
    )

    ref = Path(__file__).resolve().parents[2] / "validation" / "reference" / "conformal_mapie.csv"

    def nums(s):
        return [float(v) for v in s.split(";")]

    with open(ref) as f:
        for r in csv.DictReader(f):
            a, y = float(r["alpha"]), nums(r["y"])
            if r["case"] == "split":
                i = conformal_split(y, nums(r["pred"]), nums(r["test"]), alpha=a)
            elif r["case"] == "cqr":
                p, t = r["pred"].split("|"), r["test"].split("|")
                i = conformal_cqr(y, nums(p[0]), nums(p[1]), nums(t[0]), nums(t[1]), alpha=a)
            else:
                folds = [int(v) for v in nums(r["folds"])]
                i = conformal_plus(y, nums(r["pred"]), folds, [nums(t) for t in r["test"].split("|")], alpha=a)
            assert i["lo"] == pytest.approx(nums(r["lo"]), rel=1e-12, abs=1e-12)
            assert i["hi"] == pytest.approx(nums(r["hi"]), rel=1e-12, abs=1e-12)
    # The model-level wrapper equals the array functions on the same folds.
    n = 60
    x = [float((i * 7) % 13) for i in range(n)]
    y = [1.0 + 0.5 * v + ((i * 31) % 7) - 3.0 for i, v in enumerate(x)]
    d = Design([[1.0] * n, x], ["(Intercept)", "x"])
    t = Design([[1.0] * 3, [0.5, 6.0, 12.5]], ["(Intercept)", "x"])
    splits = k_fold(n, 5, 3)
    got = conformal_intervals(Glm("gaussian"), d, y, t, method="cv+", splits=splits)
    oof, folds, per = [0.0] * n, [0] * n, []
    for k, (train, test) in enumerate(splits):
        f = Glm("gaussian").fit(d.select(train), [y[i] for i in train])
        for i, p in zip(test, f.predict(d.select(test))):
            oof[i], folds[i] = p, k
        per.append(f.predict(t))
    assert got == conformal_plus(y, oof, folds, per, alpha=0.1)
    fitted = Glm("gaussian").fit(d, y)
    s = conformal_intervals(fitted, d, y, t, method="split")
    assert all(lo < hi for lo, hi in zip(s["lo"], s["hi"]))
    with pytest.raises(ValueError, match="family"):
        conformal_split(y, y, [1.0], score="pearson")
    with pytest.raises(ValueError, match="method"):
        conformal_intervals(fitted, d, y, t, method="naive")


def test_time_splits_window_horizon_gap_step():
    """Rolling windows, horizons and gaps in time splits (#549)."""
    import csv
    from pathlib import Path

    from prospicio.models import time_ordered, time_splits

    periods = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    for n in range(1, 6):
        assert time_splits(periods, n) == time_ordered(periods, n)
    ref = Path(__file__).resolve().parents[2] / "validation" / "reference" / "time_splits_sklearn.csv"
    with open(ref) as f:
        for r in csv.DictReader(f):
            h = int(r["horizon"])
            s = time_splits(list(range(int(r["n"]))), int(r["n_test"]), window=int(r["window"]) if r["window"] else None,
                            horizon=h, gap=int(r["gap"]), step=h)[int(r["split"])]
            assert s == ([int(v) for v in r["train"].split()], [int(v) for v in r["test"].split()])
    with pytest.raises(ValueError, match="training period"):
        time_splits(periods, 2, horizon=3, step=3, gap=1)
