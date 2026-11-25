"""Taylor's separation method (issue #979, docs/design/reserving-inflation.md,
PR 5): the Python wrappers of validation/tests/reserving_separation.rs, with
its statsmodels reference and Taylor's (1977) printed Example 1."""

import math
import pickle

import pytest
from helpers import dataset, reference
from prospicio.reserving import (
    CalendarIndex,
    FutureCells,
    Separation,
    SeparationFit,
    Triangle,
)

REFERENCE = "reserving_separation_statsmodels.csv"
METHODS = ["arithmetic", "geometric", "geometric_lognormal"]


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def want(name, method, quantity, arg=""):
    return reference(REFERENCE, dataset=name, method=method, quantity=quantity, arg=arg)


def fit_of(name, method):
    return Separation(kind=method, future_rate=0.05).fit(dataset(name), "values")


@pytest.mark.parametrize("name", ["genins", "abc"])
@pytest.mark.parametrize("method", METHODS)
def test_the_fit_matches_statsmodels(name, method):
    fit = fit_of(name, method)
    assert isinstance(fit, SeparationFit)
    for age, r in zip(fit.ages, fit.proportions):
        assert close(r, want(name, method, "proportion", age))
    for year, lam in zip(fit.calendar, fit.indices):
        assert close(lam, want(name, method, "index", year))
    for year, reserve in zip(fit.origins, fit.reserves):
        assert close(reserve, want(name, method, "reserve", year))
    assert close(fit.total_reserve, want(name, method, "total_reserve"))
    assert close(sum(fit.proportions), 1.0, 1e-12)
    if method == "arithmetic":
        assert fit.log_variance is None
    else:
        assert close(fit.log_variance, want(name, method, "log_variance"))


@pytest.mark.parametrize("method", METHODS)
def test_the_fitted_and_future_cells_match_statsmodels(method):
    fit = fit_of("genins", method)
    n = len(fit.origins)
    fitted = fit.fitted
    assert len(fitted) == n and all(len(row) == n for row in fitted)
    for i, year in enumerate(fit.origins):
        for j, age in enumerate(fit.ages):
            cell = fitted[i][j]
            if i + j < n:
                assert close(cell, want("genins", method, "fitted", f"origin={year}.0;age={age}.0"))
            else:
                assert cell is None
    cells = fit.future.cells
    assert len(cells["value"]) == n * (n - 1) // 2
    for origin, age, value in zip(cells["origin"], cells["age"], cells["value"]):
        arg = f"origin={origin};age={age}"
        assert close(value, want("genins", method, "future", arg))
    assert isinstance(fit.future, FutureCells)
    assert all(close(a, b, 1e-12) for a, b in zip(fit.future.reserves(), fit.reserves))
    assert fit.reserves[0] == 0.0
    assert close(sum(fit.reserves), fit.total_reserve, 1e-12)


def test_the_future_indices_continue_at_the_rate():
    fit = fit_of("genins", "arithmetic")
    future = fit.future_indices
    assert future["period"][0] == "2011"
    last = fit.indices[-1]
    for t, lam in enumerate(future["index"], start=1):
        assert close(lam, last * 1.05**t, 1e-12)
    assert fit.calendar[0] == "2001" and len(fit.calendar) == 10
    assert fit.origins[0] == "2001"
    assert fit.ages[:2] == [12, 24]
    assert fit.normaliser == [1.0] * 10


def taylor_example_1():
    rows = [[50.4, 78.6, 87.6, 92.4], [58.0, 87.2, 96.9], [59.5, 92.7], [66.2]]
    return Triangle.from_long(
        [2001 + i for i, row in enumerate(rows) for _ in row],
        [12 * (j + 1) for row in rows for j in range(len(row))],
        {"s": [v for row in rows for v in row]},
    )


def test_taylor_1977_example_1():
    fit = Separation(future_rate=0.10).fit(taylor_example_1(), "s")
    for got, printed in zip(fit.proportions, [0.5835, 0.2878, 0.0866, 0.0421]):
        assert abs(got - printed) <= 5e-5
    for got, printed in zip(fit.indices, [86.4, 98.9, 102.0, 113.9]):
        assert abs(got - printed) <= 0.05
    fitted = [
        [50.4, 28.5, 8.8, 4.8],
        [57.7, 29.4, 9.9, None],
        [59.5, 32.8, None, None],
        [66.5, None, None, None],
    ]
    for row, printed_row in zip(fit.fitted, fitted):
        for got, printed in zip(row, printed_row):
            assert (got is None and printed is None) or abs(got - printed) <= 0.06
    for got, printed in zip(fit.future.cells["value"], [5.3, 10.8, 5.8, 36.1, 11.9, 6.4]):
        assert abs(got - printed) <= 0.06


def test_a_normaliser_scales_the_indices():
    plain = fit_of("genins", "arithmetic")
    scaled = Separation(normaliser=[2.0] * 10, future_rate=0.05).fit(dataset("genins"), "values")
    assert scaled.normaliser == [2.0] * 10
    for a, b in zip(plain.proportions, scaled.proportions):
        assert close(a, b, 1e-12)
    for a, b in zip(plain.indices, scaled.indices):
        assert close(a, 2.0 * b, 1e-12)
    assert close(plain.total_reserve, scaled.total_reserve, 1e-12)
    with pytest.raises(ValueError, match="normaliser"):
        Separation(normaliser=[1.0, 2.0], future_rate=0.05).fit(dataset("genins"), "values")


def test_the_future_comes_from_a_rate_an_index_or_a_trend():
    tri = dataset("genins")
    base = Separation(future_rate=0.05).fit(tri, "values")
    # An annual index at 5% a year gives the same indices as the rate.
    cpi = CalendarIndex("2001", [1.05**k for k in range(10)], future_rate=0.05)
    by_index = Separation(future_index=cpi).fit(tri, "values")
    for a, b in zip(base.future_indices["index"], by_index.future_indices["index"]):
        assert close(a, b, 1e-9)
    assert by_index.proportions == base.proportions
    trended = Separation(trend_periods=3).fit(tri, "values")
    last3 = [math.log(x) for x in trended.indices[-3:]]
    slope = (last3[2] - last3[0]) / 2
    for t, lam in enumerate(trended.future_indices["index"], start=1):
        assert close(lam, trended.indices[-1] * math.exp(slope * t), 1e-9)
    with pytest.raises(ValueError, match="trend"):
        Separation(trend_periods=1).fit(tri, "values")
    with pytest.raises(ValueError, match="rate"):
        Separation(future_rate=-1.0).fit(tri, "values")


def test_the_settings_are_checked():
    with pytest.raises(ValueError, match="future"):
        Separation()
    with pytest.raises(ValueError, match="only one"):
        Separation(future_rate=0.05, trend_periods=3)
    with pytest.raises(ValueError, match="kind"):
        Separation(kind="harmonic", future_rate=0.05)
    with pytest.raises(ValueError, match="trend_periods"):
        Separation(trend_periods=-3)
    sep = Separation("geometric_lognormal", [1.0, 2.0], trend_periods=2)
    assert sep.kind == "geometric_lognormal"
    assert sep.normaliser == [1.0, 2.0]
    assert sep.trend_periods == 2
    assert sep.future_rate is None and sep.future_index is None
    cpi = CalendarIndex("2020", [1.0, 1.1], future_rate=0.1)
    assert Separation(future_index=cpi).future_index == cpi
    assert Separation(future_rate=0.05).future_rate == 0.05
    assert Separation(future_rate=0.05).kind == "arithmetic"
    assert "Separation(" in repr(sep)


def test_the_settings_pickle_and_compare():
    cpi = CalendarIndex("2020", [1.0, 1.1], future_rate=0.1)
    for sep in [
        Separation(future_rate=0.05),
        Separation("geometric", [1.0, 2.0, 3.0], trend_periods=2),
        Separation("geometric_lognormal", future_index=cpi),
    ]:
        assert pickle.loads(pickle.dumps(sep)) == sep
    assert Separation(future_rate=0.05) != Separation(future_rate=0.06)
    fit = fit_of("genins", "arithmetic")
    assert fit == fit_of("genins", "arithmetic")
    assert "SeparationFit(" in repr(fit)


def test_a_triangle_that_is_not_full_is_refused():
    # Two origins, three ages: not a full triangle.
    tri = Triangle.from_long(
        [2020, 2020, 2020, 2021], [12, 24, 36, 12], {"paid": [100.0, 150.0, 170.0, 110.0]}
    )
    with pytest.raises(ValueError, match="full triangle"):
        Separation(future_rate=0.05).fit(tri, "paid")
    with pytest.raises(ValueError, match="column"):
        Separation(future_rate=0.05).fit(dataset("genins"), "nope")


def two_by_two(second):
    return Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {"paid": [100.0, second, 110.0]})


def test_the_geometric_methods_need_positive_increments_and_three_origins():
    # A negative increment: 100, then 90 cumulative.
    negative = two_by_two(90.0)
    with pytest.raises(ValueError, match="positive"):
        Separation("geometric", future_rate=0.05).fit(negative, "paid")
    ok = two_by_two(150.0)
    two = Separation("geometric", future_rate=0.05).fit(ok, "paid")
    assert two.log_variance is None
    with pytest.raises(ValueError, match="3 origins"):
        Separation("geometric_lognormal", future_rate=0.05).fit(ok, "paid")


def test_a_segment_is_chosen_by_its_key_values():
    keys = {"lob": ["a", "a", "a", "b", "b", "b"]}
    tri = Triangle.from_long(
        [2020, 2020, 2021, 2020, 2020, 2021],
        [12, 24, 12, 12, 24, 12],
        {"paid": [100.0, 150.0, 110.0, 10.0, 25.0, 12.0]},
        keys=keys,
    )
    sep = Separation(future_rate=0.05)
    with pytest.raises(ValueError, match="segment"):
        sep.fit(tri, "paid")
    a = sep.fit(tri, "paid", segment={"lob": "a"})
    assert a == sep.fit(two_by_two(150.0), "paid")
    b = sep.fit(tri, "paid", segment={"lob": "b"})
    assert b != a
    assert b.origins == ["2020", "2021"]
    with pytest.raises(ValueError):
        sep.fit(tri, "paid", segment={"lob": "c"})
