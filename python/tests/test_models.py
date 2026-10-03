import math

import pytest

from actuarialrs.models import (
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
