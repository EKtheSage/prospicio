"""The age-period-cohort reserving GLM and the separation method's GLM form
(issue #979, docs/design/reserving-inflation.md, PR 7): the Python wrappers
of validation/tests/reserving_apc.rs, checked against its Python `apc` 1.0.2
reference on Taylor-Ashe (GenIns) and R `apc`'s documented I0 total."""

import math
import pickle
from collections import Counter
from functools import lru_cache

import pytest
from helpers import VALIDATION, dataset, read_csv
from prospicio.reserving import (
    ApcGlm,
    ApcGlmFit,
    FutureCells,
    NotSupportedError,
    OdpGlm,
    OdpGlmFit,
    Separation,
    Triangle,
)

ROWS = read_csv(VALIDATION / "reference" / "reserving_apc_python.csv")
EFFECTS = {"poisson_apc": "APC", "poisson_ap": "AP", "poisson_pc": "PC", "poisson_ac": "AC"}


def close(a, b, rel=1e-6):
    return abs(a - b) <= rel * max(1.0, abs(b))


@lru_cache(maxsize=None)
def genins():
    return dataset("genins")


@lru_cache(maxsize=None)
def apc(effects, extrapolation="I0"):
    return ApcGlm(effects=effects, extrapolation=extrapolation).fit(genins(), "values")


@lru_cache(maxsize=None)
def separation_glm(method):
    future = {"separation_rate": {"future_rate": 0.05}, "separation_trend": {"trend_periods": 4}}
    return Separation(**future[method]).glm(genins(), "values")


def fit_of(method):
    """The fit a reference method names: `poisson_<effects>` with I0, or
    `poisson_<effects>_<rule>`."""
    if method in EFFECTS:
        return apc(EFFECTS[method])
    base, rule = method.rsplit("_", 1)
    return apc(EFFECTS[base], rule.upper())


def error_row(error, quantity, arg):
    """A process or parameter variance from a prediction error dict: an
    origin's (arg: the year) or the total's (no arg)."""
    i = error["origin"].index(arg or "Total")
    se = error["process_se" if quantity == "process_variance" else "parameter_se"][i]
    return se**2


def got(row):
    method, quantity, arg = row["method"], row["quantity"], row["arg"]
    if method.startswith("separation"):
        glm = separation_glm(method)
        if quantity == "total_reserve":
            return glm.total_reserve
        return error_row(glm.prediction_error(), quantity, arg)
    fit = fit_of(method)
    if quantity in ("process_variance", "parameter_variance"):
        return error_row(fit.prediction_error(), quantity, arg)
    if quantity in ("dd_age", "dd_cohort", "dd_period"):
        values = {
            "dd_age": fit.age_second_differences,
            "dd_cohort": fit.accident_second_differences,
            "dd_period": fit.calendar_second_differences,
        }[quantity]
        return values[int(arg) - 2]
    if quantity in ("fitted", "future"):
        key = dict(part.split("=") for part in arg.split(";"))
        if quantity == "fitted":
            return fit.fitted[fit.origins.index(key["origin"])][fit.ages.index(int(key["age"]))]
        cells = fit.future.cells
        at = [
            v
            for o, a, v in zip(cells["origin"], cells["age"], cells["value"])
            if o == key["origin"] and a == int(key["age"])
        ]
        assert len(at) == 1, row
        return at[0]
    if quantity == "reserve":
        return fit.reserves[fit.origins.index(arg)]
    return {
        "deviance": lambda: fit.deviance,
        "level": lambda: fit.level,
        "slope_age": lambda: fit.age_slope,
        "slope_cohort": lambda: fit.accident_slope,
        "total_reserve": lambda: fit.total_reserve,
        "documented_total": lambda: fit.total_reserve,
    }[quantity]()


@pytest.mark.parametrize("method", sorted({r["method"] for r in ROWS}))
def test_the_fits_match_python_apc(method):
    for row in (r for r in ROWS if r["method"] == method):
        want = float(row["expected"])
        tol = max(float(row["abs_tol"]), float(row["rel_tol"]) * abs(want))
        value = got(row)
        assert abs(value - want) <= tol, (row, value)


def test_the_reference_has_every_quantity():
    # A generator that drops a whole quantity must fail here.
    assert Counter(r["quantity"] for r in ROWS) == {
        "dd_age": 24,
        "dd_cohort": 24,
        "dd_period": 24,
        "deviance": 4,
        "documented_total": 1,
        "fitted": 220,
        "future": 450,
        "level": 4,
        "parameter_variance": 120,
        "process_variance": 120,
        "reserve": 100,
        "slope_age": 4,
        "slope_cohort": 4,
        "total_reserve": 12,
    }


def test_the_i0_total_is_r_apcs_documented_one():
    assert abs(apc("APC", "I0").total_reserve - 19_061_428) <= 1.0


def test_the_defaults_are_apc_i0_and_odp_glms():
    model, odp = ApcGlm(), OdpGlm()
    assert (model.effects, model.extrapolation, model.family) == ("APC", "I0", "poisson")
    assert model.power is None
    assert model.tolerance == odp.tolerance
    assert model.max_iterations == odp.max_iterations
    assert ApcGlm(effects="pc", extrapolation="i2") == ApcGlm(effects="PC", extrapolation="I2")
    assert ApcGlm(effects="ap").effects == "AP"
    assert ApcGlm(extrapolation="i1").extrapolation == "I1"
    tweedie = ApcGlm(family="tweedie", power=1.5)
    assert (tweedie.family, tweedie.power) == ("tweedie", 1.5)
    assert pickle.loads(pickle.dumps(tweedie)) == tweedie
    assert repr(ApcGlm()) == 'ApcGlm(effects="APC", extrapolation="I0", family="poisson")'


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"effects": "APCX"}, "effects must be"),
        ({"effects": "A"}, "effects must be"),
        ({"extrapolation": "I3"}, "extrapolation must be"),
        ({"family": "normal"}, "family must be"),
        ({"family": "Poisson"}, "family must be"),
        ({"family": "gamma", "power": 1.5}, "power is the Tweedie's"),
        ({"family": "tweedie"}, "tweedie needs power"),
        ({"family": "tweedie", "power": 2.5}, "power"),
    ],
)
def test_bad_settings_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ApcGlm(**kwargs)


def test_the_fit_exposes_the_canonical_parameters():
    fit = apc("APC")
    k = len(fit.origins)
    assert isinstance(fit, ApcGlmFit)
    assert (fit.effects, fit.extrapolation) == ("APC", "I0")
    assert fit.ages == [12 * (j + 1) for j in range(k)]
    assert len(fit.coefficients) == len(fit.names) == 3 * k - 3
    assert fit.names[:3] == ["level", "slope_age", "slope_accident"]
    assert fit.coefficients[:3] == [fit.level, fit.age_slope, fit.accident_slope]
    assert fit.coefficients[3:] == (
        fit.age_second_differences
        + fit.accident_second_differences
        + fit.calendar_second_differences
    )
    cov = fit.covariance
    assert len(cov) == len(cov[0]) == 3 * k - 3
    assert all(cov[i][i] > 0 for i in range(len(cov)))
    assert len(fit.future_calendar_second_differences) == 2
    assert fit.dispersion > 0
    for i, row in enumerate(fit.fitted):
        assert all((cell is None) == (i + j >= k) for j, cell in enumerate(row))
    assert fit.reserves[0] == 0.0
    assert math.isclose(sum(fit.reserves), fit.total_reserve, rel_tol=1e-12)
    assert "ApcGlmFit(effects=" in repr(fit)


def test_the_odp_fit_carries_the_draws():
    fit = apc("APC", "I1")
    odp = fit.odp_fit
    assert isinstance(odp, OdpGlmFit)
    assert odp.glm.names == fit.names
    assert odp.reserves == fit.reserves
    assert odp.dispersion == fit.dispersion
    assert len(odp.residuals()["origin"]) == 55
    direct = fit.predict_distribution(200, 7)
    assert direct.dims == ["origin", "development"]
    assert direct.n_components == 45
    assert direct.draw_matrix() == odp.predict_distribution(200, 7).draw_matrix()


@pytest.mark.parametrize("effects", ["APC", "AP", "PC", "AC"])
def test_the_future_cells_sum_to_the_reserves(effects):
    fit = apc(effects, "I2")
    future = fit.future
    assert isinstance(future, FutureCells)
    assert len(future) == 45
    assert len(future.reserves()) == len(fit.reserves) == 10
    assert all(close(a, b, 1e-12) for a, b in zip(future.reserves(), fit.reserves))
    assert future == FutureCells.from_odp_glm(fit.odp_fit)


def test_the_ac_model_is_the_chain_ladder():
    fit = apc("AC")
    odp = OdpGlm().fit(genins(), "values")
    assert fit.calendar_second_differences == []
    assert fit.future_calendar_second_differences == [0.0, 0.0]
    assert len(fit.reserves) == len(odp.reserves) == 10
    assert all(close(a, b) for a, b in zip(fit.reserves, odp.reserves))
    assert close(fit.deviance, apc("AC", "I1").deviance, 1e-12)


def test_i0_needs_four_origins():
    rows = [[100.0, 60.0, 20.0], [110.0, 70.0], [125.0]]
    tri = Triangle.from_long(
        [2001 + i for i, row in enumerate(rows) for _ in row],
        [12 * (j + 1) for row in rows for j in range(len(row))],
        {"paid": [v for row in rows for v in row]},
        cumulative=False,
    )
    with pytest.raises(ValueError):
        ApcGlm().fit(tri, "paid")


def test_a_triangle_that_is_not_full_raises():
    # The first age is two development periods, not one.
    rows = [[100.0, 60.0, 20.0, 5.0], [110.0, 70.0, 24.0], [125.0, 75.0], [140.0]]
    tri = Triangle.from_long(
        [2001 + i for i, row in enumerate(rows) for _ in row],
        [12 * (j + 2) for row in rows for j in range(len(row))],
        {"paid": [v for row in rows for v in row]},
        cumulative=False,
    )
    with pytest.raises(ValueError, match="the first age must be one development period"):
        ApcGlm(extrapolation="I2").fit(tri, "paid")


@pytest.mark.parametrize("future", [{"future_rate": 0.05}, {"trend_periods": 4}])
def test_the_separation_glm_has_the_arithmetic_reserves(future):
    model = Separation(**future)
    glm = model.glm(genins(), "values")
    fit = model.fit(genins(), "values")
    assert isinstance(glm, OdpGlmFit)
    assert len(glm.reserves) == len(fit.reserves) == 10
    assert all(close(a, b) for a, b in zip(glm.reserves, fit.reserves))
    assert close(glm.total_reserve, fit.total_reserve)
    assert len(glm.future_means) == len(fit.future.cells["value"]) == 45
    assert all(close(a, b) for a, b in zip(glm.future_means, fit.future.cells["value"]))


def test_the_ap_model_fits_as_the_separation_glm():
    # Both have development and calendar effects on the same cells, so their
    # fits agree; only the future calendar effects differ.
    ap = apc("AP", "I2")
    glm = separation_glm("separation_rate")
    assert close(ap.deviance, glm.glm.deviance)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "geometric", "future_rate": 0.05},
        {"kind": "geometric_lognormal", "future_rate": 0.05},
        {"normaliser": [1.0] * 10, "future_rate": 0.05},
    ],
)
def test_the_separation_glm_is_arithmetic_without_a_normaliser(kwargs):
    with pytest.raises(NotSupportedError, match="separation prediction error"):
        Separation(**kwargs).glm(genins(), "values")
