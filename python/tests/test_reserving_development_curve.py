"""Development curves over every age (``DevelopmentCurve``,
``docs/design/reserving-v02.md``, decision 31, issue #556). The Rust core
is checked against R's ``lm`` and ``nls`` in
``validation/tests/reserving_curves.rs``; these tests check the bindings'
arguments, the same reference rows and the fitted pattern in the chain
ladder."""

import math

import pytest

from prospicio.reserving import (
    ChainLadder,
    DevelopmentCurve,
    DevelopmentCurveFit,
    Mack,
    TailCurve,
)
from helpers import VALIDATION, dataset, read_csv

SHAPES = ["exponential", "inverse_power", "hoerl", "sherman", "loglogistic", "weibull"]


@pytest.fixture(scope="module")
def triangles():
    return {name: dataset(name) for name in ["raa", "genins"]}


def test_matches_r_lm_and_nls(triangles):
    # validation/scripts/reserving_curves_r.R: R's lm and nls on the
    # volume-weighted factors of RAA and GenIns.
    rows = read_csv(VALIDATION / "reference" / "reserving_curves_r.csv")
    fits = {}
    params = {"a": 0, "omega": 0, "b": 1, "theta": 1, "c": 2}
    for row in rows:
        key = (row["dataset"], row["method"])
        if key not in fits:
            fits[key] = DevelopmentCurve(shape=row["method"]).fit(
                triangles[row["dataset"]], "values"
            )
        fit = fits[key]
        quantity = row["quantity"]
        if quantity.startswith("param_"):
            got = fit.parameters[params[quantity[len("param_") :]]]
        elif quantity == "fitted":
            got = fit.fitted[int(row["arg"])]
        elif quantity == "tail":
            got = fit.tail()
        else:
            got = getattr(fit, quantity)
        want = float(row["expected"])
        assert got == pytest.approx(want, rel=float(row["rel_tol"])), row


def test_the_fit_in_the_chain_ladder(triangles):
    raa = triangles["raa"]
    fit = DevelopmentCurve(shape="sherman", keep_observed_until=48).fit(raa, "values")
    assert isinstance(fit, DevelopmentCurveFit)
    assert fit.shape == "sherman" and fit.keep_observed_until == 48
    assert fit.attachment == 3
    plain = ChainLadder().fit(raa, "values")
    assert fit.observed == pytest.approx(plain.ldf, rel=1e-15)
    assert fit.ldf[:3] == fit.observed[:3] and fit.ldf[3:] == fit.fitted[3:]
    selected = fit.selected()
    assert selected["ldf_kind"] == "ldf" and selected["ldf_weight"] == 1.0
    assert list(selected["ldf"]) == [48, 60, 72, 84, 96, 108]
    cl = ChainLadder(**selected, tail=fit.tail()).fit(raa, "values")
    assert cl.ldf == pytest.approx(fit.ldf, rel=1e-15)
    assert cl.tail == pytest.approx(fit.tail(), rel=1e-12)
    # The curve's factors are selected: Mack charges them no parameter error.
    mack = Mack(**selected).fit(raa, "values")
    assert mack.total_parameter_risk < Mack().fit(raa, "values").total_parameter_risk
    # Statistics on the factor scale.
    assert fit.n_points == 9 and fit.n_parameters == 3
    assert all(fit.in_fit)
    sse = sum(r * r for r in fit.residuals)
    assert fit.sum_of_squares == pytest.approx(sse, rel=1e-12)
    assert fit.factor(9) == pytest.approx(fit.tail(1), rel=1e-15)
    assert "DevelopmentCurveFit(shape=\"sherman\"" in repr(fit)


def test_compare_lists_every_shape(triangles):
    table = DevelopmentCurve().compare(triangles["genins"], "values")
    assert table["shape"] == SHAPES
    assert table["n_parameters"] == [2, 2, 3, 3, 2, 2]
    assert table["n_points"] == [9] * 6
    hoerl = table["sum_of_squares"][2]
    assert hoerl < table["sum_of_squares"][0] and hoerl < table["sum_of_squares"][1]
    assert all(r <= 1 for r in table["r_squared"])
    some = DevelopmentCurve(fit_period=(24, None)).compare(
        triangles["genins"], "values", ["hoerl", "sherman"]
    )
    assert some["shape"] == ["hoerl", "sherman"] and some["n_points"] == [8, 8]


def test_settings_and_errors(triangles):
    curve = DevelopmentCurve(shape="hoerl", fit_period=(24, 96), keep_observed_until=36)
    assert curve.shape == "hoerl"
    assert curve.fit_period == (24, 96)
    assert curve.keep_observed_until == 36
    assert repr(curve) == (
        'DevelopmentCurve(shape="hoerl", fit_period=(24, 96), keep_observed_until=36)'
    )
    fit = curve.fit(triangles["raa"], "values", average="simple")
    assert fit.in_fit == [False] + [True] * 6 + [False, False]
    assert math.isfinite(fit.adjusted_r_squared)
    with pytest.raises(ValueError, match="shape must be"):
        DevelopmentCurve(shape="cubic")
    with pytest.raises(ValueError, match="past the last factor"):
        DevelopmentCurve(keep_observed_until=120).fit(triangles["raa"], "values")
    with pytest.raises(ValueError, match="four factors"):
        DevelopmentCurve(shape="sherman", fit_period=(84, None)).fit(triangles["raa"], "values")


def test_tail_curve_takes_every_shape(triangles):
    raa = triangles["raa"]
    for shape in SHAPES:
        tail = TailCurve(curve=shape)
        assert tail.curve == shape
        fit = ChainLadder(tail=tail).fit(raa, "values")
        curve = DevelopmentCurve(shape=shape).fit(raa, "values")
        assert fit.tail == pytest.approx(curve.tail(), rel=1e-12)
    with pytest.raises(ValueError, match="curve must be"):
        TailCurve(curve="cubic")
