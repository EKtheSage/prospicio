import csv
import datetime
import math
from pathlib import Path

import pytest

from actuarialrs.reserving import ChainLadder, ChainLadderFit, Mack, MackFit, Triangle

VALIDATION = Path(__file__).resolve().parents[2] / "validation"

# Reference method -> (average, sigma_interpolation), as in
# validation/tests/reserving.rs.
METHODS = {
    "chain_ladder": ("volume", "log-linear"),
    "mack": ("volume", "log-linear"),
    "chain_ladder_simple": ("simple", "log-linear"),
    "mack_alpha0": ("simple", "log-linear"),
    "mack_alpha2": ("regression", "log-linear"),
    "mack_sigma_mack": ("volume", "mack"),
}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def dataset(name):
    rows = read_csv(VALIDATION / "data" / f"{name}.csv")
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows],
        development=[int(r["development"]) for r in rows],
        values={"values": [float(r["value"]) for r in rows]},
    )


@pytest.fixture(scope="module")
def triangles():
    return {name: dataset(name) for name in ["raa", "genins", "abc"]}


def evaluate(fit, quantity, arg):
    def origin():
        return fit.origins.index(str(int(float(arg))))

    per_age = {"ata_factor": fit.ldf, "cdf": fit.cdf, "sigma": fit.sigma, "f_se": fit.std_err}
    if quantity in per_age:
        return per_age[quantity][int(float(arg))]
    per_origin = {
        "ultimate": fit.ultimate,
        "reserve": fit.reserve,
        "se": fit.standard_error,
        "process_risk": fit.process_risk,
        "parameter_risk": fit.parameter_risk,
    }
    if quantity in per_origin:
        return per_origin[quantity][origin()]
    return getattr(fit, quantity)


@pytest.mark.parametrize("reference", ["reserving_chainladder_r.csv", "reserving_chainladder_python.csv"])
def test_matches_reference(triangles, reference):
    fits = {}
    cases = read_csv(VALIDATION / "reference" / reference)
    assert cases
    for case in cases:
        key = (case["dataset"], case["method"])
        if key not in fits:
            average, sigma = METHODS[case["method"]]
            fits[key] = Mack(average, sigma).fit(triangles[case["dataset"]], "values")
        got = evaluate(fits[key], case["quantity"], case["arg"])
        want = float(case["expected"])
        abs_tol = float(case["abs_tol"] or 0)
        rel_tol = float(case["rel_tol"] or 0)
        err = abs(got - want)
        assert got == want or err <= abs_tol or err <= rel_tol * abs(want), (case, got)


def test_chain_ladder_totals_every_average(triangles):
    # R ChainLadder 0.2.21 (validation/reference/reserving_chainladder_r.csv).
    raa = triangles["raa"]
    cl = ChainLadder().fit(raa, "values")
    assert isinstance(cl, ChainLadderFit)
    assert cl.total_ultimate == pytest.approx(213_122.22826121017, rel=1e-9)
    assert cl.total_reserve == pytest.approx(52_135.228261210155, rel=1e-9)
    assert cl.reserve[0] == 0.0 and len(cl.cdf) == 10 and cl.cdf[-1] == 1.0
    assert cl.origins[0] == "1981" and cl.development[0] == 12
    for average in ["simple", "regression"]:
        fit = ChainLadder(average=average).fit(raa, "values")
        mack = Mack(average=average).fit(raa, "values")
        assert fit.total_reserve == pytest.approx(mack.total_reserve, rel=1e-12)
    # The chain ladder is Mack's projection.
    mack = Mack().fit(raa, "values")
    assert mack.chain_ladder.ultimate == cl.ultimate
    assert mack.ultimate == cl.ultimate


def test_tail_scales_ultimates(triangles):
    raa = triangles["raa"]
    base = ChainLadder().fit(raa, "values")
    tailed = ChainLadder(tail=1.05).fit(raa, "values")
    assert tailed.tail == 1.05
    assert tailed.ultimate == pytest.approx([u * 1.05 for u in base.ultimate], rel=1e-12)


def test_mack_fit(triangles):
    fit = Mack(sigma_interpolation="mack").fit(triangles["raa"], "values")
    assert isinstance(fit, MackFit)
    assert fit.total_standard_error == pytest.approx(26_909.01, abs=0.01)
    assert fit.total_cv == pytest.approx(0.5161, abs=1e-4)
    total = math.hypot(fit.total_process_risk, fit.total_parameter_risk)
    assert fit.total_standard_error == pytest.approx(total, rel=1e-12)
    se = [math.hypot(p, q) for p, q in zip(fit.process_risk, fit.parameter_risk)]
    assert fit.standard_error == pytest.approx(se, rel=1e-12)


def test_dataset_shapes(triangles):
    for name, n, first, latest in [
        ("raa", 10, "1981", 160_987.0),
        ("genins", 10, "2001", 34_358_090.0),
        ("abc", 11, "1977", 10_221_194.0),
    ]:
        tri = triangles[name]
        assert tri.shape == (1, 1, n, n)
        assert tri.origins[0] == first
        assert tri.index == ["Total"] and tri.columns == ["values"]
        assert tri.is_cumulative
        assert sum(tri.latest_diagonal()[0][0]) == latest


def small():
    return Triangle.from_long(
        origin=[2020, 2020, 2020, 2021, 2021, 2022],
        development=[12, 24, 36, 12, 24, 12],
        values={"paid": [100.0, 150.0, 160.0, 110.0, 170.0, 120.0]},
    )


def test_accessors_and_values():
    tri = small()
    assert tri.shape == (1, 1, 3, 3)
    assert tri.development == [12, 24, 36]
    assert tri.valuation == "2022-12"
    assert tri.origin_grain == "Y" and tri.development_grain == "Y"
    v = tri.values[0][0]
    assert v[0] == [100.0, 150.0, 160.0]
    assert v[2][0] == 120.0 and math.isnan(v[2][1]) and math.isnan(v[2][2])
    assert tri.latest_diagonal() == [[[160.0, 170.0, 120.0]]]
    assert "Triangle(shape=(1, 1, 3, 3)" in repr(tri)


def test_incremental_cumulative_round_trip():
    tri = small()
    inc = tri.to_incremental()
    assert not inc.is_cumulative
    assert inc.values[0][0][0] == [100.0, 50.0, 10.0]
    assert inc.to_cumulative() == tri
    assert tri.to_cumulative() == tri
    # Incremental input builds the same triangle.
    long = inc.to_long()
    again = Triangle.from_long(long["origin"], long["development"], {"paid": long["paid"]}, cumulative=False)
    assert again.to_cumulative() == tri


def test_long_round_trip():
    tri = small()
    long = tri.to_long()
    assert list(long) == ["index", "origin", "development", "paid"]
    assert long["origin"][0] == datetime.date(2020, 1, 1)
    assert long["index"] == ["Total"] * 6
    back = Triangle.from_long(long["origin"], long["development"], {"paid": long["paid"]}, index=long["index"])
    assert back == tri


def test_link_ratios():
    lr = small().link_ratios()
    assert lr.shape == (1, 1, 3, 2)
    assert lr.values[0][0][0] == pytest.approx([1.5, 160.0 / 150.0])
    assert math.isnan(lr.values[0][0][2][0])


def test_multi_part_index_slice_and_frame():
    data = {
        "lob": ["Auto", "Auto", "Auto", "Home", "Home"],
        "state": ["CA", "CA", "CA", "NY", "NY"],
        "year": [2020, 2020, 2021, 2020, 2021],
        "age": [12, 24, 12, 12, 12],
        "paid": [100.0, 150.0, 110.0, 50.0, 60.0],
        "incurred": [120.0, 160.0, 130.0, 70.0, 80.0],
    }
    tri = Triangle.from_frame(data, "year", "age", ["paid", "incurred"], index=["lob", "state"])
    assert tri.index == [("Auto", "CA"), ("Home", "NY")]
    assert tri.columns == ["paid", "incurred"]
    assert tri.shape == (2, 2, 2, 2)
    home = tri.slice(index=("Home", "NY"), columns="incurred")
    assert home.shape == (1, 1, 2, 2)
    assert home.values[0][0][0][0] == 70.0
    both = tri.slice(index=[("Home", "NY"), ("Auto", "CA")])
    assert both.index == [("Home", "NY"), ("Auto", "CA")]
    long = tri.to_long()
    assert long["index"][0] == ("Auto", "CA")
    back = Triangle.from_long(long["origin"], long["development"],
                              {"paid": long["paid"], "incurred": long["incurred"]}, index=long["index"])
    assert back == tri
    with pytest.raises(ValueError, match="slice to one"):
        ChainLadder().fit(tri, "paid")
    with pytest.raises(ValueError, match="no column or index named"):
        tri.slice(columns="nope")
    with pytest.raises(ValueError, match="twice"):
        tri.slice(columns=["paid", "paid"])


def test_pandas_frame_and_datetimes():
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    df = pd.DataFrame(
        {
            "accident": pd.to_datetime(["2021-02-15", "2021-02-15", "2021-05-01"]),
            "valued": pd.to_datetime(["2021-03-31", "2021-06-30", "2021-06-30"]),
            "paid": [10.0, 25.0, 7.0],
        }
    )
    tri = Triangle.from_frame(
        df, "accident", "valued", "paid",
        origin_grain="Q", development_grain="Q", development_is_valuation=True,
    )
    assert tri.origins == ["2021Q1", "2021Q2"]
    assert tri.development == [3, 6]
    assert tri.valuation == "2021-06"
    assert np.asarray(tri.values).shape == (1, 1, 2, 2)
    # numpy datetime64 and integer arrays work directly.
    same = Triangle.from_long(
        np.array(["2021-02-15", "2021-02-15", "2021-05-01"], dtype="datetime64[D]"),
        np.array([3, 6, 3]),
        {"paid": np.array([10.0, 25.0, 7.0])},
        origin_grain="Q",
        development_grain="Q",
    )
    assert same == tri
    frame = tri.to_frame()
    assert list(frame.columns) == ["index", "origin", "development", "paid"]
    back = Triangle.from_frame(frame, "origin", "development", "paid", index="index",
                               origin_grain="Q", development_grain="Q")
    assert back == tri


def test_grain():
    q = Triangle.from_long(
        origin=[2020, 2020, 2020, 2020],
        development=[3, 6, 9, 12],
        values=[1.0, 2.0, 3.0, 4.0],
        origin_grain="Q",
        development_grain="Q",
    )
    y = q.grain("Y", "Y")
    assert y.origins == ["2020"] and y.development == [12]
    assert y.values[0][0][0] == [4.0]
    assert q.grain("Y").development_grain == "Q"
    with pytest.raises(ValueError, match="grain"):
        y.grain("Q")
    with pytest.raises(ValueError, match="grain must be"):
        q.grain("W")


def test_errors():
    with pytest.raises(ValueError, match="rows"):
        Triangle.from_long([2020, 2021], [12], [1.0, 2.0])
    with pytest.raises(ValueError, match="development grid"):
        Triangle.from_long([2020, 2020], [12, 18], [1.0, 2.0])
    with pytest.raises(ValueError, match="infinite"):
        Triangle.from_long([2020], [12], [math.inf])
    with pytest.raises(ValueError, match="positive"):
        Triangle.from_long([2020], [-12], [1.0])
    with pytest.raises(TypeError, match="dates or integer years"):
        Triangle.from_long(["2020"], [12], [1.0])
    with pytest.raises(ValueError, match="average"):
        ChainLadder(average="median")
    with pytest.raises(ValueError, match="sigma_interpolation"):
        Mack(sigma_interpolation="linear")
    with pytest.raises(ValueError, match="tail"):
        ChainLadder(tail=0.0).fit(small(), "paid")
    with pytest.raises(ValueError, match="no column or index named"):
        ChainLadder().fit(small(), "incurred")
    two = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], [1.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="at least 3"):
        Mack().fit(two, "values")


def test_frame_round_trip_multi_part_index():
    pd = pytest.importorskip("pandas")
    data = {
        "lob": ["Auto", "Auto", "Home"],
        "state": ["CA", "CA", "NY"],
        "year": [2020, 2020, 2020],
        "age": [12, 24, 12],
        "paid": [100.0, 150.0, 50.0],
    }
    tri = Triangle.from_frame(data, "year", "age", "paid", index=["lob", "state"])
    frame = tri.to_frame()
    assert isinstance(frame, pd.DataFrame)
    back = Triangle.from_frame(frame, "origin", "development", "paid", index="index")
    assert back.index == [("Auto", "CA"), ("Home", "NY")]
    assert back == tri


def test_numpy_integer_years():
    np = pytest.importorskip("numpy")
    years = list(np.array([2020, 2020, 2021]))
    tri = Triangle.from_long(years, [12, 24, 12], [1.0, 2.0, 3.0])
    assert tri == Triangle.from_long([2020, 2020, 2021], [12, 24, 12], [1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="dates or integer years"):
        Triangle.from_long([True], [12], [1.0])
    with pytest.raises(TypeError, match="dates or integer years"):
        Triangle.from_long([2020.0], [12], [1.0])
