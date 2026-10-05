import csv
import datetime
import math
from pathlib import Path

import pytest

from actuarialrs.distributions import PredictiveDistribution
from actuarialrs.reserving import (
    ChainLadder,
    ChainLadderFit,
    Mack,
    MackFit,
    OdpBootstrap,
    OdpBootstrapFit,
    Triangle,
)

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
        assert tri.keys == [] and tri.index == ["Total"] and tri.columns == ["values"]
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
    assert tri.valuation == datetime.date(2022, 12, 31)
    assert tri.origin_grain == "Y" and tri.development_grain == "Y"
    v = tri.values[0][0]
    assert v[0] == [100.0, 150.0, 160.0]
    assert v[2][0] == 120.0 and math.isnan(v[2][1]) and math.isnan(v[2][2])
    assert tri.latest_diagonal() == [[[160.0, 170.0, 120.0]]]
    assert repr(tri).startswith("Triangle: paid (cumulative, valuation 2022-12)")


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
    assert list(long) == ["origin", "development", "paid"]
    assert long["origin"][0] == datetime.date(2020, 1, 1)
    back = Triangle.from_long(long["origin"], long["development"], {"paid": long["paid"]})
    assert back == tri
    assert tri.select() == tri
    assert tri.group_by([]) == tri


def two_keys():
    """Two lobs x two states, paid and incurred; Home / NY has no paid
    value in 2021."""
    data = {
        "lob": ["Auto", "Auto", "Auto", "Auto", "Home", "Home", "Home"],
        "state": ["CA", "CA", "CA", "NY", "CA", "NY", "NY"],
        "year": [2020, 2020, 2021, 2020, 2020, 2020, 2021],
        "age": [12, 24, 12, 12, 12, 12, 12],
        "paid": [100.0, 150.0, 110.0, 50.0, 30.0, 20.0, math.nan],
        "incurred": [120.0, 160.0, 130.0, 60.0, 35.0, 25.0, 15.0],
    }
    return Triangle.from_frame(data, "year", "age", ["paid", "incurred"], keys=["lob", "state"])


def same(a, b):
    return a == b or (math.isnan(a) and math.isnan(b))


def test_view_matches_values():
    tri = two_keys()
    values = tri.values
    for i, (lob, state) in enumerate(tri.index):
        for c, column in enumerate(tri.columns):
            v = tri.view(column, lob=lob, state=state)
            assert list(v.index) == tri.origins
            assert list(v.columns) == tri.development
            assert v.index.name == "origin" and v.columns.name == "development"
            for o in range(len(tri.origins)):
                for d in range(len(tri.development)):
                    assert same(float(v.iat[o, d]), values[i][c][o][d])


def test_view_matches_to_frame():
    tri = two_keys()
    long = tri.to_frame()
    for _, row in long.iterrows():
        v = tri.view("incurred", lob=row["lob"], state=row["state"])
        origin = str(row["origin"].year)
        assert same(float(v.loc[origin, row["development"]]), row["incurred"])


def test_view_one_segment_and_errors():
    tri = small()
    assert tri.view().loc["2020", 36] == 160.0
    assert tri.view("paid").equals(tri.view())
    keyed = two_keys()
    with pytest.raises(ValueError, match="2 segments match"):
        keyed.view("paid", lob="Auto")
    with pytest.raises(ValueError, match="2 columns"):
        keyed.view(lob="Auto", state="CA")
    with pytest.raises(ValueError, match="no column named"):
        keyed.view("x", lob="Auto", state="CA")
    with pytest.raises(ValueError, match="no key named"):
        keyed.view("paid", line="Auto")
    with pytest.raises(ValueError, match="no segment has"):
        keyed.view("paid", lob="Boat")


def test_summary_of_two_keys_and_two_measures():
    s = two_keys().summary()
    assert list(s.columns) == [
        "lob",
        "state",
        "column",
        "n_origins",
        "first_origin",
        "last_origin",
        "valuation",
        "latest",
        "cumulative",
    ]
    rows = list(zip(s["lob"], s["state"], s["column"], s["n_origins"].tolist(), s["latest"].tolist()))
    assert rows == [
        ("Auto", "CA", "paid", 2, 260.0),
        ("Auto", "CA", "incurred", 2, 290.0),
        ("Auto", "NY", "paid", 1, 50.0),
        ("Auto", "NY", "incurred", 1, 60.0),
        ("Home", "CA", "paid", 1, 30.0),
        ("Home", "CA", "incurred", 1, 35.0),
        ("Home", "NY", "paid", 1, 20.0),
        ("Home", "NY", "incurred", 2, 40.0),
    ]
    assert s["first_origin"].tolist()[6:] == ["2020", "2020"]
    assert s["last_origin"].tolist()[6:] == ["2020", "2021"]
    assert s["valuation"].tolist()[6:] == [datetime.date(2020, 12, 31), datetime.date(2021, 12, 31)]
    assert s["cumulative"].all()
    # The latest totals are the latest diagonal's sums.
    diagonal = two_keys().latest_diagonal()
    totals = [sum(v for v in diagonal[i][c] if not math.isnan(v)) for i in range(4) for c in range(2)]
    assert s["latest"].tolist() == totals
    # Incremental: the sum of the increments, the same totals.
    inc = two_keys().to_incremental().summary()
    assert inc["latest"].tolist() == pytest.approx(totals)
    assert not inc["cumulative"].any()


def test_views_without_pandas(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pandas", None)
    tri = two_keys()
    v = tri.view("paid", lob="Auto", state="CA")
    assert list(v) == ["origin", 12, 24]
    assert v["origin"] == ["2020", "2021"] and v[12] == [100.0, 110.0]
    assert v[24][0] == 150.0 and math.isnan(v[24][1])
    s = tri.summary()
    assert isinstance(s, dict) and s["n_origins"] == [2, 2, 1, 1, 1, 1, 1, 2]
    assert s["lob"][0] == "Auto" and s["first_origin"][0] == "2020"


def test_summary_key_clash():
    tri = Triangle.from_long([2020], [12], {"paid": [1.0]}, keys={"column": ["a"]})
    with pytest.raises(ValueError, match="clashes"):
        tri.summary()


def words(text):
    return [line.split() for line in text.splitlines()]


def test_printout_of_one_segment_is_the_grid():
    lines = words(repr(small()))
    assert lines[0][:2] == ["Triangle:", "paid"]
    assert lines[1] == ["12", "24", "36"]
    assert lines[2] == ["2020", "100", "150", "160"]
    assert lines[4] == ["2022", "120"]
    assert "<table" in small()._repr_html_()
    ratios = words(small().link_ratios().to_string())
    assert ratios[2] == ["2020", "1.500", "1.067"]


def test_printout_of_several_segments_is_the_summary():
    tri = two_keys()
    lines = words(repr(tri))
    assert lines[1] == ["lob", "state", "column", "n_origins", "first_origin", "last_origin", "valuation", "latest"]
    assert lines[2] == ["Auto", "CA", "paid", "2", "2020", "2021", "2021-12", "260"]
    assert len(lines) == 2 + 8
    html = tri._repr_html_()
    assert html.count("<tr>") == 1 + 8 and "Home" in html
    one = words(repr(tri.select(lob="Home", state="NY", columns="paid")))
    assert one[0][:3] == ["Triangle:", "paid,", "lob=Home,"]
    assert one[2] == ["2020", "20"]


def test_printout_of_empty_holey_and_large_triangles():
    # A segment with no observed value, and one with a hole at age 24.
    holey = Triangle.from_long(
        [2020, 2020, 2020], [12, 36, 12], {"paid": [1.0, 3.0, math.nan]}, keys={"lob": ["Auto", "Auto", "Home"]}
    )
    assert "Home" in repr(holey)
    first = holey.summary()["first_origin"]
    assert first[0] == "2020" and first.isna().tolist() == [False, True]
    assert words(repr(holey.select(lob="Home")))[2] == ["2020"]
    assert words(repr(holey.select(lob="Auto")))[2] == ["2020", "1", "3"]
    # Forty origins and ages: truncated to 20 rows and 12 ages around "...".
    origin = [2000 + k for k in range(40) for _ in range(40 - k)]
    age = [12 * (d + 1) for k in range(40) for d in range(40 - k)]
    big = Triangle.from_long(origin, age, [1.0] * len(origin))
    lines = words(repr(big))
    assert len(lines) == 1 + 1 + 20 + 1 + 1
    assert lines[-1] == ["[40", "origins", "x", "40", "ages]"]
    assert len(lines[1]) == 12 + 1 and "..." in lines[1]
    assert len(words(big.to_string(max_rows=0, max_cols=0))) == 42
    assert big.view().shape == (40, 40)


def test_link_ratios():
    lr = small().link_ratios()
    assert lr.shape == (1, 1, 3, 2)
    assert lr.values[0][0][0] == pytest.approx([1.5, 160.0 / 150.0])
    assert math.isnan(lr.values[0][0][2][0])


def test_named_keys_select_and_frame():
    data = {
        "lob": ["Auto", "Auto", "Auto", "Home", "Home"],
        "state": ["CA", "CA", "CA", "NY", "NY"],
        "year": [2020, 2020, 2021, 2020, 2021],
        "age": [12, 24, 12, 12, 12],
        "paid": [100.0, 150.0, 110.0, 50.0, 60.0],
        "incurred": [120.0, 160.0, 130.0, 70.0, 80.0],
    }
    tri = Triangle.from_frame(data, "year", "age", ["paid", "incurred"], keys=["lob", "state"])
    assert tri.keys == ["lob", "state"]
    assert tri.index == [("Auto", "CA"), ("Home", "NY")]
    assert repr(tri).startswith("Triangle: 2 segments x 2 columns, keys lob, state")
    assert tri.columns == ["paid", "incurred"]
    assert tri.shape == (2, 2, 2, 2)
    home = tri.select(lob="Home", state="NY", columns="incurred")
    assert home.shape == (1, 1, 2, 2)
    assert home.values[0][0][0][0] == 70.0
    both = tri.select(state=["NY", "CA"])
    assert both.index == [("Auto", "CA"), ("Home", "NY")]
    assert both.keys == ["lob", "state"]
    long = tri.to_long()
    assert list(long) == ["lob", "state", "origin", "development", "paid", "incurred"]
    assert long["lob"] == ["Auto"] * 3 + ["Home"] * 2
    assert long["state"] == ["CA"] * 3 + ["NY"] * 2
    back = Triangle.from_long(long["origin"], long["development"],
                              {"paid": long["paid"], "incurred": long["incurred"]},
                              keys={"lob": long["lob"], "state": long["state"]})
    assert back == tri
    assert Triangle.from_frame(long, "origin", "development", ["paid", "incurred"], keys=tri.keys) == tri
    # Home is observed at 12 months only: it fits on its own single age
    # (no development, so no reserve), not on the triangle's two.
    home_fit = ChainLadder().fit(tri, "paid").segment(lob="Home", state="NY")
    assert home_fit.ldf == []
    assert all(r == 0.0 for r in home_fit.reserve)
    with pytest.raises(ValueError, match="no column named"):
        tri.select(columns="nope")
    with pytest.raises(ValueError, match="twice"):
        tri.select(columns=["paid", "paid"])


def lob_coverage_long():
    """Two keys (lob x coverage) with paid and incurred: Auto from RAA and
    Home from GenIns (scaled), origins re-based to 2011, each coverage a
    different share of the line with its own tilt by origin."""
    long = {"lob": [], "coverage": [], "year": [], "age": [], "paid": [], "incurred": []}
    for lob, name, scale in [("Auto", "raa", 1.0), ("Home", "genins", 1e-3)]:
        rows = read_csv(VALIDATION / "data" / f"{name}.csv")
        first = min(int(r["origin"]) for r in rows)
        for coverage, share in [("BI", 0.7), ("PD", 0.3)]:
            for k, r in enumerate(rows):
                offset = int(r["origin"]) - first
                paid = float(r["value"]) * scale * share * (1 + share * offset / 20)
                long["lob"].append(lob)
                long["coverage"].append(coverage)
                long["year"].append(2011 + offset)
                long["age"].append(int(r["development"]))
                long["paid"].append(paid)
                long["incurred"].append(paid * 1.2 + 10.0 * (k % 3))
    return long


def test_select_and_group_by_lob_and_coverage():
    long = lob_coverage_long()
    measures = ["paid", "incurred"]
    tri = Triangle.from_frame(long, "year", "age", measures, keys=["lob", "coverage"])
    assert tri.shape == (4, 2, 10, 10)

    # Selection by name; values may be one or a list, ANDed across keys.
    auto_bi = tri.select(lob="Auto", coverage="BI")
    assert auto_bi.index == [("Auto", "BI")]
    assert tri.select(lob="Auto", coverage=["PD", "BI"]).index == [("Auto", "BI"), ("Auto", "PD")]
    paid_bi = tri.select(coverage="BI", columns=["incurred", "paid"])
    assert paid_bi.index == [("Auto", "BI"), ("Home", "BI")]
    assert paid_bi.columns == ["incurred", "paid"]
    assert tri.select(columns="paid").shape == (4, 1, 10, 10)

    # Grouping equals building with fewer keys, and totals are sums of segments.
    by_lob = tri.group_by(["lob"])
    assert by_lob.keys == ["lob"]
    assert by_lob.index == ["Auto", "Home"]
    assert by_lob == Triangle.from_frame(long, "year", "age", measures, keys=["lob"])
    total = tri.group_by([])
    assert total.index == ["Total"]
    assert total == Triangle.from_frame(long, "year", "age", measures)
    for c in range(2):
        segments = sum(x for i in tri.latest_diagonal() for x in i[c])
        assert sum(x for i in by_lob.latest_diagonal() for x in i[c]) == pytest.approx(segments)
        assert sum(total.latest_diagonal()[0][c]) == pytest.approx(segments)
    swapped = tri.group_by(["coverage", "lob"])
    assert swapped.index[0] == ("BI", "Auto")

    # Chain ladder on a group equals chain ladder on the summed triangle.
    auto_rows = [i for i, lob in enumerate(long["lob"]) if lob == "Auto"]
    summed = Triangle.from_frame({k: [v[i] for i in auto_rows] for k, v in long.items()},
                                 "year", "age", measures)
    for column in measures:
        grouped = ChainLadder().fit(by_lob.select(lob="Auto"), column)
        direct = ChainLadder().fit(summed, column)
        assert grouped.ultimate == pytest.approx(direct.ultimate, rel=1e-12)
        assert grouped.ldf == pytest.approx(direct.ldf, rel=1e-12)

    # Incremental triangles are summed as cumulative values.
    assert tri.to_incremental().group_by("lob") == by_lob.to_incremental()

    with pytest.raises(ValueError, match="no key named line"):
        tri.select(line="Auto")
    with pytest.raises(ValueError, match="no segment has coverage = \"GL\""):
        tri.select(coverage="GL")
    with pytest.raises(ValueError, match="supplied twice"):
        tri.select(lob=["Auto", "Auto"])
    with pytest.raises(ValueError, match="at least one value"):
        tri.select(lob=[])
    with pytest.raises(ValueError, match="no column named reported"):
        tri.select(columns="reported")
    with pytest.raises(ValueError, match="no key named line"):
        tri.group_by(["line"])
    with pytest.raises(ValueError, match="key lob is supplied twice"):
        tri.group_by(["lob", "lob"])


def test_every_segment_at_once():
    long = lob_coverage_long()
    tri = Triangle.from_frame(long, "year", "age", ["paid", "incurred"], keys=["lob", "coverage"])
    labels = tri.index

    # Each segment's fit equals fitting that segment alone, exactly.
    cl = ChainLadder().fit(tri, "paid")
    mack = Mack().fit(tri, "paid")
    assert cl.keys == ["lob", "coverage"] and cl.index == labels
    assert len(cl.origins) == len(cl.reserve) == len(mack.standard_error) == 40
    for k, (lob, coverage) in enumerate(labels):
        alone = tri.select(lob=lob, coverage=coverage)
        one_cl, one_mack = ChainLadder().fit(alone, "paid"), Mack().fit(alone, "paid")
        rows = slice(10 * k, 10 * k + 10)
        assert cl.reserve[rows] == one_cl.reserve
        assert cl.origins[rows] == one_cl.origins
        assert mack.standard_error[rows] == one_mack.standard_error
        seg = mack.segment(lob=lob, coverage=coverage)
        assert seg.index == [(lob, coverage)]
        assert seg.ldf == one_mack.ldf
        assert seg.total_standard_error == one_mack.total_standard_error
    assert cl.total_reserve == pytest.approx(sum(cl.reserve), rel=1e-12)

    # Long results.
    frame = mack.to_frame()
    assert list(frame.columns) == ["lob", "coverage", "origin", "latest", "ultimate", "reserve",
                                   "process_risk", "parameter_risk", "standard_error"]
    assert len(frame) == 40 and frame["origin"].iloc[10] == "2011"
    assert list(frame["reserve"]) == mack.reserve
    totals = mack.totals_frame()
    assert list(totals["lob"]) == ["Auto", "Auto", "Home", "Home"]
    home_pd = mack.segment(lob="Home", coverage="PD")
    assert totals["standard_error"].iloc[3] == home_pd.total_standard_error
    assert totals["reserve"].iloc[3] == home_pd.total_reserve
    dev = cl.development_frame()
    assert list(dev.columns) == ["lob", "coverage", "development", "ldf", "cdf", "sigma", "std_err"]
    assert len(dev) == 40
    assert list(dev["ldf"].iloc[30:39]) == cl.segment(lob="Home", coverage="PD").ldf
    assert math.isnan(dev["ldf"].iloc[39])

    # Per-age fields and the totals' standard errors need one segment.
    with pytest.raises(ValueError, match="4 segments; use development_frame"):
        cl.ldf
    with pytest.raises(ValueError, match="use totals_frame"):
        mack.total_standard_error
    with pytest.raises(ValueError, match="2 segments match"):
        cl.segment(lob="Auto")
    with pytest.raises(ValueError, match="no key named line"):
        cl.segment(line="Auto")
    with pytest.raises(ValueError, match="no segment has lob = \"Farm\""):
        cl.segment(lob="Farm", coverage="BI")
    assert "segments=4" in repr(cl)

    # The bootstrap: one joint distribution over lob x coverage x origin.
    boot = OdpBootstrap(n_sims=2000, seed=3).fit(tri, "paid")
    reserves = boot.reserves
    assert reserves.dims == ["lob", "coverage", "origin"]
    assert len(reserves.components()) == 40
    assert reserves.components()[10] == ("Auto", "PD", "2011")
    again = OdpBootstrap(n_sims=2000, seed=3).fit(tri, "paid")
    one = boot.segment(lob="Home", coverage="BI").reserves
    assert one.dims == reserves.dims and len(one.components()) == 10
    assert one.draw_matrix() == again.segment(lob="Home", coverage="BI").reserves.draw_matrix()
    by_lob = reserves.aggregate(["lob"])
    assert by_lob.components() == [("Auto",), ("Home",)]
    assert by_lob.mean() == pytest.approx(reserves.mean(), rel=1e-12)
    totals = boot.totals_frame()
    assert list(totals.columns) == ["lob", "coverage", "latest", "ultimate", "reserve", "scale",
                                    "mean", "std_dev"]
    alone = OdpBootstrap(n_sims=10).fit(tri.select(lob="Home", coverage="BI"), "paid")
    assert totals["scale"].iloc[2] == alone.scale
    assert boot.segment(lob="Home", coverage="BI").scale == alone.scale
    frame = boot.to_frame()
    assert len(frame) == 40
    draws = boot.reserves.draw_matrix()
    column = [row[10] for row in draws]
    assert frame["mean"].iloc[10] == pytest.approx(sum(column) / len(column), rel=1e-12)
    assert totals["mean"].iloc[2] == pytest.approx(one.mean(), rel=1e-12)
    with pytest.raises(ValueError, match="use totals_frame"):
        boot.scale
    with pytest.raises(ValueError, match="use segment"):
        boot.residuals


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
    assert tri.valuation == datetime.date(2021, 6, 30)
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
    assert list(frame.columns) == ["origin", "development", "paid"]
    back = Triangle.from_frame(frame, "origin", "development", "paid",
                               origin_grain="Q", development_grain="Q")
    assert back == tri


def test_select_with_numpy_and_pandas_values():
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    df = pd.DataFrame({"lob": ["A", "A", "B", "C"], "company": [7, 8, 8, 8],
                       "year": 2020, "age": 12, "paid": [1.0, 2.0, 3.0, 4.0]})
    tri = Triangle.from_frame(df, "year", "age", "paid", keys=["lob", "company"])
    # Arrays, Series and tuples are lists of values; numbers match as str().
    wanted = df.loc[df["paid"] > 2.5, "lob"].unique()
    assert tri.select(lob=wanted).index == [("B", "8"), ("C", "8")]
    assert tri.select(lob=pd.Series(["B", "C"]), company=np.int64(8)).shape[0] == 2
    assert tri.select(lob=("A",), company=[7]).index == [("A", "7")]


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
    # The development grain defaults to the origin grain.
    assert q.grain("Y") == y
    assert q.grain("Y", "Q").development_grain == "Q"
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
    with pytest.raises(ValueError, match="no column named"):
        ChainLadder().fit(small(), "incurred")
    two = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], [1.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="at least 3"):
        Mack().fit(two, "values")


def test_frame_round_trip_named_keys():
    pd = pytest.importorskip("pandas")
    data = {
        "lob": ["Auto", "Auto", "Home"],
        "state": ["CA", "CA", "NY"],
        "year": [2020, 2020, 2020],
        "age": [12, 24, 12],
        "paid": [100.0, 150.0, 50.0],
    }
    tri = Triangle.from_frame(data, "year", "age", "paid", keys=["lob", "state"])
    frame = tri.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["lob", "state", "origin", "development", "paid"]
    back = Triangle.from_frame(frame, "origin", "development", "paid", keys=["lob", "state"])
    assert back.index == [("Auto", "CA"), ("Home", "NY")]
    assert back == tri
    # One key: labels are plain strings; key order follows the argument.
    one = Triangle.from_frame(frame, "origin", "development", "paid", keys="lob")
    assert one.keys == ["lob"] and one.index == ["Auto", "Home"]
    swapped = Triangle.from_frame(frame, "origin", "development", "paid", keys=["state", "lob"])
    assert swapped.index == [("CA", "Auto"), ("NY", "Home")]
    assert list(swapped.to_frame().columns)[:2] == ["state", "lob"]


def test_key_validation():
    origin, dev = [2020, 2020], [12, 24]
    with pytest.raises(ValueError, match="key lob is supplied twice"):
        Triangle.from_frame({"lob": ["A", "A"], "o": origin, "d": dev, "paid": [1.0, 2.0]},
                            "o", "d", "paid", keys=["lob", "lob"])
    with pytest.raises(ValueError, match="paid is both a key and a value column"):
        Triangle.from_long(origin, dev, {"paid": [1.0, 2.0]}, keys={"paid": ["A", "A"]})
    with pytest.raises(ValueError, match="column lob has 1 rows, expected 2"):
        Triangle.from_long(origin, dev, {"paid": [1.0, 2.0]}, keys={"lob": ["A"]})
    with pytest.raises(ValueError, match="missing values"):
        Triangle.from_long(origin, dev, {"paid": [1.0, 2.0]}, keys={"lob": ["A", None]})
    with pytest.raises(ValueError, match="missing values"):
        Triangle.from_long(origin, dev, {"paid": [1.0, 2.0]}, keys={"lob": ["A", math.nan]})
    with pytest.raises(TypeError, match="dict"):
        Triangle.from_long(origin, dev, {"paid": [1.0, 2.0]}, keys=["A", "A"])
    clash = Triangle.from_long(origin, dev, {"paid": [1.0, 2.0]}, keys={"origin": ["A", "A"]})
    with pytest.raises(ValueError, match="clashes"):
        clash.to_long()
    # Non-string key values are stored as strings.
    codes = Triangle.from_long(origin, dev, {"paid": [1.0, 2.0]}, keys={"company": [7, 7]})
    assert codes.index == ["7"] and codes.to_long()["company"] == ["7", "7"]


def test_pandas_missing_keys():
    pd = pytest.importorskip("pandas")
    base = {"o": [2020, 2020], "d": [12, 24], "paid": [1.0, 2.0]}
    missing = [
        pd.array(["A", pd.NA], dtype="string"),
        pd.array([7, pd.NA], dtype="Int64"),
        pd.to_datetime(["2020-01-01", None]),
        pd.Categorical(["A", None]),
    ]
    for column in missing:
        frame = pd.DataFrame({"lob": column, **base})
        with pytest.raises(ValueError, match="missing values"):
            Triangle.from_frame(frame, "o", "d", "paid", keys="lob")
    with pytest.raises(ValueError, match="missing values"):
        Triangle.from_long(base["o"], base["d"], {"paid": base["paid"]}, keys={"lob": ["A", pd.NA]})
    with pytest.raises(ValueError, match="missing values"):
        Triangle.from_long(base["o"], base["d"], {"paid": base["paid"]}, keys={"lob": ["A", pd.NaT]})


def test_numpy_integer_years():
    np = pytest.importorskip("numpy")
    years = list(np.array([2020, 2020, 2021]))
    tri = Triangle.from_long(years, [12, 24, 12], [1.0, 2.0, 3.0])
    assert tri == Triangle.from_long([2020, 2020, 2021], [12, 24, 12], [1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="dates or integer years"):
        Triangle.from_long([True], [12], [1.0])
    with pytest.raises(TypeError, match="dates or integer years"):
        Triangle.from_long([2020.0], [12], [1.0])


def test_valuation_is_last_day_of_month():
    tri = Triangle.from_long(
        origin=[datetime.date(2024, 2, 1)],
        development=[1],
        values=[1.0],
        origin_grain="M",
        development_grain="M",
    )
    assert tri.valuation == datetime.date(2024, 2, 29)


# ODP bootstrap against R ChainLadder's BootChainLadder
# (validation/reference/reserving_bootstrap_r.csv). The reference tolerances
# for simulated quantities assume 20000 simulations with this seed, as in
# validation/tests/reserving.rs.
BOOT_SIMS = 20_000
BOOT_SEED = 20_261_004


@pytest.fixture(scope="module")
def raa_boot(triangles):
    return OdpBootstrap(n_sims=BOOT_SIMS, seed=BOOT_SEED).fit(triangles["raa"], "values")


def bootstrap_reference(quantity, method="odp_bootstrap"):
    rows = read_csv(VALIDATION / "reference" / "reserving_bootstrap_r.csv")
    return [r for r in rows if r["dataset"] == "raa" and r["method"] == method and r["quantity"] == quantity]


def test_bootstrap_scale_and_residuals_match_r(raa_boot):
    assert isinstance(raa_boot, OdpBootstrapFit)
    (scale,) = bootstrap_reference("scale")
    assert raa_boot.scale == pytest.approx(float(scale["expected"]), rel=1e-9)
    residuals = bootstrap_reference("residual")
    assert len(residuals) == 55
    origins = raa_boot.origins
    for row in residuals:
        year, k = row["arg"].split(":")
        got = raa_boot.residuals[origins.index(year)][int(k)]
        want = float(row["expected"])
        assert got == pytest.approx(want, rel=1e-9, abs=1e-9), row
    # Laid out like Triangle.values: nan below the latest diagonal.
    assert len(raa_boot.residuals) == 10 and len(raa_boot.residuals[0]) == 10
    assert math.isnan(raa_boot.residuals[9][1]) and math.isnan(raa_boot.fitted[9][1])
    # Fitted incrementals add up to each origin's latest cumulative value.
    latest = raa_boot.chain_ladder.latest
    for o, row in enumerate(raa_boot.fitted):
        assert sum(m for m in row if not math.isnan(m)) == pytest.approx(latest[o], rel=1e-12)


def test_bootstrap_reserves_match_r(raa_boot):
    reserves = raa_boot.reserves
    assert isinstance(reserves, PredictiveDistribution)
    assert reserves.n_sims == BOOT_SIMS
    total = {r["quantity"]: r for r in bootstrap_reference("mean_total", "odp_bootstrap_gamma")}
    total.update({r["quantity"]: r for r in bootstrap_reference("sd_total", "odp_bootstrap_gamma")})
    got = {"mean_total": reserves.mean(), "sd_total": math.sqrt(reserves.variance())}
    for quantity, row in total.items():
        assert abs(got[quantity] - float(row["expected"])) <= float(row["abs_tol"]), (row, got[quantity])
    # The chain ladder the bootstrap is centred on.
    assert raa_boot.chain_ladder.total_reserve == pytest.approx(52_135.228261210155, rel=1e-9)


def test_bootstrap_components_are_origins(raa_boot):
    reserves = raa_boot.reserves
    assert reserves.dims == ["origin"]
    assert reserves.components() == [(str(y),) for y in range(1981, 1991)]
    assert raa_boot.origins == [str(y) for y in range(1981, 1991)]
    assert raa_boot.development == list(range(12, 121, 12))
    # The fully developed origin has no reserve.
    draws = reserves.draw_matrix()
    assert len(draws) == BOOT_SIMS and len(draws[0]) == 10
    assert all(row[0] == 0.0 for row in draws)


def test_bootstrap_is_reproducible(triangles):
    raa = triangles["raa"]
    a = OdpBootstrap(n_sims=500, seed=7).fit(raa, "values")
    b = OdpBootstrap(n_sims=500, seed=7).fit(raa, "values")
    c = OdpBootstrap(n_sims=500, seed=8).fit(raa, "values")
    assert a.reserves.draw_matrix() == b.reserves.draw_matrix()
    assert a.reserves.draw_matrix() != c.reserves.draw_matrix()
    assert a.reserves.provenance()["model"] == "odp_bootstrap"


def test_bootstrap_process_error(triangles):
    raa = triangles["raa"]
    gamma = OdpBootstrap(n_sims=5_000, seed=1, process="gamma").fit(raa, "values")
    param = OdpBootstrap(n_sims=5_000, seed=1, process="none").fit(raa, "values")
    assert OdpBootstrap().process == "gamma" and OdpBootstrap().n_sims == 10_000
    assert OdpBootstrap(process="none").process == "none"
    # Process error adds variance; the residuals and scale do not change.
    assert param.reserves.variance() < gamma.reserves.variance()
    assert param.scale == gamma.scale
    assert "OdpBootstrap(n_sims=10000, seed=0, process=\"gamma\")" == repr(OdpBootstrap())
    assert repr(param).startswith("OdpBootstrapFit(origins=10, n_sims=5000")


def test_bootstrap_errors(triangles):
    with pytest.raises(ValueError, match="process must be"):
        OdpBootstrap(process="poisson")
    with pytest.raises(ValueError, match="n_sims must be positive"):
        OdpBootstrap(n_sims=0)
    with pytest.raises(OverflowError):
        OdpBootstrap(n_sims=-1)
    data = {
        "lob": ["Auto"] * 3 + ["Home"] * 3,
        "year": [2020, 2020, 2021] * 2,
        "age": [12, 24, 12] * 2,
        "paid": [100.0, 150.0, 110.0, 50.0, 70.0, 60.0],
    }
    multi = Triangle.from_frame(data, "year", "age", "paid", keys="lob")
    with pytest.raises(ValueError, match="segment Auto: bootstrap: too few observed cells"):
        OdpBootstrap(n_sims=10).fit(multi, "paid")
    with pytest.raises(ValueError, match="no column named"):
        OdpBootstrap(n_sims=10).fit(triangles["raa"], "paid")
    with pytest.raises(ValueError, match="degrees of freedom"):
        two = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], [1.0, 2.0, 1.0])
        OdpBootstrap(n_sims=10).fit(two, "values")
