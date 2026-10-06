import csv
import datetime
import math
from pathlib import Path

import pytest

from actuarialrs.distributions import PredictiveDistribution
from actuarialrs.reserving import (
    Benktander,
    BornhuetterFerguson,
    CapeCod,
    CapeCodFit,
    ChainLadder,
    ChainLadderFit,
    ClaimsDevelopmentResult,
    ClarkCapeCod,
    ClarkFit,
    ClarkLdf,
    ExpectedLoss,
    ExpectedLossFit,
    Mack,
    MackFit,
    OdpBootstrap,
    OdpBootstrapFit,
    OneYearFit,
    TailBondy,
    TailConstant,
    TailCurve,
    TailLogLinear,
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


# Reference method -> Mack keyword arguments, as in
# validation/tests/reserving_tails.rs. mack_tail_given adds tail_sigma and
# tail_std_err from the fitted pattern.
TAIL_METHODS = {
    # R ChainLadder.
    "mack_tail_loglinear": dict(tail=TailLogLinear()),
    "mack_tail_loglinear_sigma_mack": dict(tail=TailLogLinear(), sigma_interpolation="mack"),
    "mack_tail_loglinear_alpha2": dict(tail=TailLogLinear(), average="regression"),
    "mack_tail_constant": dict(tail=1.05),
    "mack_tail_constant_sigma_mack": dict(tail=1.05, sigma_interpolation="mack"),
    "mack_tail_given": dict(tail=1.05),
    # chainladder-python.
    "tail_constant": dict(tail=TailConstant(1.05)),
    "tail_constant_decay": dict(tail=TailConstant(1.1, decay=0.75)),
    "tail_constant_attach": dict(tail=TailConstant(1.05, attachment_age=72)),
    "tail_constant_below_one": dict(tail=TailConstant(0.98)),
    "tail_curve_exponential": dict(tail=TailCurve()),
    "tail_curve_inverse_power": dict(tail=TailCurve("inverse_power")),
    "tail_curve_fit_period": dict(tail=TailCurve(fit_period=(36, 108), extrap_periods=50)),
    "tail_curve_off_grid": dict(tail=TailCurve(fit_period=(30, 102))),
    "tail_curve_attach": dict(tail=TailCurve(attachment_age=60)),
    "tail_bondy": dict(tail=TailBondy()),
    "tail_bondy_generalized": dict(tail=TailBondy(earliest_age=36)),
    "tail_bondy_off_grid": dict(tail=TailBondy(earliest_age=30)),
    "tail_bondy_attach": dict(tail=TailBondy(earliest_age=36, attachment_age=72)),
}


def given_tail(tri):
    """R's mack_tail_given inputs: twice the last sigma, half the last standard error."""
    fit = ChainLadder().fit(tri, "values")
    return fit.sigma[-1] * 2, fit.std_err[-1] / 2


def evaluate_tail(fit, quantity, arg):
    if quantity == "ldf":
        return (fit.ldf + fit.tail_ldf)[int(arg)]
    if quantity == "cdf":
        k = int(arg)
        if k < len(fit.cdf):
            return fit.cdf[k]
        return math.prod((fit.ldf + fit.tail_ldf)[k:])
    if quantity == "tail_factor":
        return fit.tail
    return evaluate(fit, quantity, arg)


@pytest.mark.parametrize("reference", ["reserving_tails_r.csv", "reserving_tails_python.csv"])
def test_tails_match_reference(triangles, reference):
    fits = {}
    cases = read_csv(VALIDATION / "reference" / reference)
    assert cases
    for case in cases:
        tri = triangles[case["dataset"]]
        quantity = case["quantity"]
        if quantity.startswith("given_tail_"):
            sigma, std_err = given_tail(tri)
            got = sigma if quantity == "given_tail_sigma" else std_err
        else:
            key = (case["dataset"], case["method"])
            if key not in fits:
                kw = dict(TAIL_METHODS[case["method"]])
                if case["method"] == "mack_tail_given":
                    kw["tail_sigma"], kw["tail_std_err"] = given_tail(tri)
                fits[key] = Mack(**kw).fit(tri, "values")
            got = evaluate_tail(fits[key], quantity, case["arg"])
        want = float(case["expected"])
        abs_tol = float(case["abs_tol"] or 0)
        rel_tol = float(case["rel_tol"] or 0)
        err = abs(got - want)
        assert got == want or err <= abs_tol or err <= rel_tol * abs(want), (case, got)


def test_tail_arguments(triangles):
    raa = triangles["raa"]
    assert ChainLadder().tail == 1.0
    assert ChainLadder(tail=1.05).tail == 1.05
    curve = TailCurve("inverse_power", fit_period=(36, None), attachment_age=60)
    assert ChainLadder(tail=curve).tail.curve == "inverse_power"
    assert curve.fit_period == (36, None) and curve.extrap_periods == 100
    assert "TailBondy(earliest_age=36" in repr(Mack(tail=TailBondy(36)))
    # A constant with the default decay and attachment is just its factor.
    assert ChainLadder(tail=TailConstant(1.05)).tail == 1.05
    assert isinstance(ChainLadder(tail=TailConstant(1.05, decay=0.75)).tail, TailConstant)
    # The selected factors replace the estimated ones from the attachment age.
    fit = ChainLadder(tail=curve).fit(raa, "values")
    base = ChainLadder().fit(raa, "values")
    assert fit.ldf[:4] == base.ldf[:4] and fit.ldf[4] != base.ldf[4]
    assert fit.estimated_ldf == base.ldf and fit.tail_attachment_age == 60
    assert base.tail_attachment_age == 120
    tailed_mack = Mack(tail=curve).fit(raa, "values")
    assert tailed_mack.estimated_ldf == base.ldf and tailed_mack.tail_attachment_age == 60
    assert math.prod(fit.tail_ldf) == pytest.approx(fit.tail, rel=1e-12)
    assert fit.cdf[-1] == pytest.approx(fit.tail, rel=1e-12)
    # Without a tail above 1 there is no tail risk.
    mack = Mack().fit(raa, "values")
    assert (mack.tail, mack.tail_sigma, mack.tail_std_err, mack.standard_error[0]) == (1.0, 0.0, 0.0, 0.0)
    given = Mack(tail=1.05, tail_sigma=1.5, tail_std_err=0.003).fit(raa, "values")
    assert (given.tail_sigma, given.tail_std_err) == (1.5, 0.003)
    with pytest.raises(TypeError, match="tail must be"):
        ChainLadder(tail="1.05")
    with pytest.raises(ValueError, match="curve must be"):
        TailCurve("weibull")
    with pytest.raises(ValueError, match="tail"):
        ChainLadder(tail=TailCurve(fit_period=(108, None))).fit(raa, "values")
    with pytest.raises(ValueError, match="tail"):
        Mack(tail=1.05, tail_sigma=-1.0).fit(raa, "values")


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
    # Each segment's tail is in totals_frame(); the tail's factors need one.
    assert list(totals.columns)[-3:] == ["tail", "tail_sigma", "tail_std_err"]
    assert list(totals["tail"]) == [1.0] * 4 and list(totals["tail_sigma"]) == [0.0] * 4
    assert list(cl.totals_frame().columns)[-3:] == ["tail", "tail_sigma", "tail_std_err"]
    with pytest.raises(ValueError, match="4 segments; use totals_frame"):
        mack.tail_sigma
    with pytest.raises(ValueError, match="4 segments; use totals_frame"):
        cl.tail
    with pytest.raises(ValueError, match=r"4 segments; use segment\(\.\.\.\)$"):
        cl.tail_ldf
    with pytest.raises(ValueError, match=r"4 segments; use segment\(\.\.\.\)$"):
        mack.estimated_ldf
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


# Reference method -> sigma_interpolation, as in
# validation/tests/reserving_cdr.rs.
CDR_METHODS = {"cdr": "log-linear", "cdr_sigma_mack": "mack"}


def cdr_value(mack, cdr, quantity, arg):
    """The value of a reserving_cdr_r.csv case. R reports one calendar year
    per age, so years past the run-off are zero."""
    if quantity == "cdr_se":
        year, k = arg.split(":")
        years, k = cdr.by_calendar_year, int(k)
        return years[k - 1][cdr.origins.index(year)] if k <= len(years) else 0.0
    if quantity == "total_cdr_se":
        years, k = cdr.total_by_calendar_year, int(arg)
        return years[k - 1] if k <= len(years) else 0.0
    per_origin = {
        "reserve": mack.reserve,
        "one_year_se": cdr.one_year_standard_error,
        "mack_se": mack.standard_error,
    }
    if quantity in per_origin:
        return per_origin[quantity][cdr.origins.index(arg)]
    return {
        "total_reserve": mack.total_reserve,
        "total_one_year_se": cdr.total_one_year_standard_error,
        "total_mack_se": mack.total_standard_error,
    }[quantity]


def test_claims_development_result_matches_r():
    # Every row of R ChainLadder's CDR(MackChainLadder(tri), dev = "all").
    fits = {}
    cases = read_csv(VALIDATION / "reference" / "reserving_cdr_r.csv")
    assert cases
    for case in cases:
        key = (case["dataset"], case["method"])
        if key not in fits:
            mack = Mack(sigma_interpolation=CDR_METHODS[case["method"]]).fit(
                dataset(case["dataset"]), "values"
            )
            fits[key] = (mack, mack.claims_development_result())
        got = cdr_value(*fits[key], case["quantity"], case["arg"])
        want = float(case["expected"])
        tol = max(float(case["abs_tol"] or 0), float(case["rel_tol"] or 0) * abs(want))
        assert got == want or abs(got - want) <= tol, (case, got)


def test_claims_development_result_mw2008():
    # Merz and Wuthrich (2008), Table 4, printed to the unit.
    mack = Mack(sigma_interpolation="mack").fit(dataset("mw2008"), "values")
    cdr = mack.claims_development_result()
    assert isinstance(cdr, ClaimsDevelopmentResult)
    assert mack.total_reserve == pytest.approx(2_237_826, abs=1)
    assert cdr.total_one_year_standard_error == pytest.approx(81_080, abs=1)
    assert cdr.total_run_off_standard_error == pytest.approx(108_401, abs=1)
    assert cdr.origins[0] == "2001" and cdr.one_year_standard_error[0] == 0.0
    assert cdr.one_year_standard_error == cdr.by_calendar_year[0]
    assert cdr.total_one_year_standard_error == cdr.total_by_calendar_year[0]
    assert len(cdr.by_calendar_year) == len(cdr.total_by_calendar_year) == 8
    assert cdr.run_off_standard_error == pytest.approx(mack.standard_error, rel=1e-9, abs=1e-9)
    assert "ClaimsDevelopmentResult(origins=9" in repr(cdr)


def test_claims_development_result_frame():
    pytest.importorskip("pandas")
    cdr = Mack().fit(dataset("raa"), "values").claims_development_result()
    frame = cdr.to_frame()
    assert list(frame.columns) == ["origin"] + [f"cdr_{k}" for k in range(1, 10)] + ["run_off"]
    assert list(frame["origin"]) == cdr.origins
    assert list(frame["cdr_1"]) == cdr.one_year_standard_error
    assert list(frame["run_off"]) == cdr.run_off_standard_error


def test_claims_development_result_errors(triangles):
    simple = Mack(average="simple").fit(triangles["raa"], "values")
    with pytest.raises(ValueError, match="claims development result: needs volume-weighted"):
        simple.claims_development_result()
    # Merz and Wuthrich assume no tail: a factor other than 1, or a factor
    # of 1 that replaces estimated factors, is an error.
    no_tail = "claims development result: needs no tail factor"
    for tail in [1.05, TailConstant(1.05), TailLogLinear(), TailConstant(1.0, attachment_age=84)]:
        fit = Mack(tail=tail).fit(triangles["raa"], "values")
        with pytest.raises(ValueError, match=no_tail):
            fit.claims_development_result()
    explicit = Mack(tail=1.0).fit(triangles["raa"], "values").claims_development_result()
    default = Mack().fit(triangles["raa"], "values").claims_development_result()
    assert explicit.by_calendar_year == default.by_calendar_year
    tri = Triangle.from_frame(
        lob_coverage_long(), "year", "age", ["paid", "incurred"], keys=["lob", "coverage"]
    )
    mack = Mack().fit(tri, "paid")
    with pytest.raises(ValueError, match="4 segments; use segment"):
        mack.claims_development_result()
    one = mack.segment(lob="Home", coverage="PD")
    assert one.claims_development_result().total_run_off_standard_error == pytest.approx(
        one.total_standard_error, rel=1e-9
    )
# Expected-loss methods: parity with chainladder-python 0.10.1
# (validation/reference/reserving_expected_loss_python.csv).

PREMIUM_DATASETS = ["clrd_wkcomp", "genins_premium"]


def premium_rows(name):
    return read_csv(VALIDATION / "data" / f"{name}.csv")


def premium_dataset(name):
    rows = premium_rows(name)
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows],
        development=[int(r["development"]) for r in rows],
        values={c: [float(r[c]) for r in rows] for c in ["paid", "premium"]},
    )


def expected_loss_method(method):
    """The estimator a reference `method` (name;key=value;...) describes."""
    name, *settings = method.split(";")
    kw = dict(s.split("=") for s in settings)
    average = kw.pop("average")
    cls = {
        "expected_loss": ExpectedLoss,
        "bornhuetter_ferguson": BornhuetterFerguson,
        "benktander": Benktander,
        "cape_cod": CapeCod,
    }[name]
    kw = {k: int(v) if k == "n_iters" else float(v) for k, v in kw.items()}
    return cls(average=average, **kw)


def test_expected_loss_matches_chainladder_python():
    tris = {name: premium_dataset(name) for name in PREMIUM_DATASETS}
    cases = read_csv(VALIDATION / "reference" / "reserving_expected_loss_python.csv")
    assert cases
    fits = {}
    for case in cases:
        key = (case["dataset"], case["method"])
        if key not in fits:
            model = expected_loss_method(case["method"])
            fits[key] = model.fit(tris[case["dataset"]], "paid", "premium")
        fit = fits[key]
        quantity, arg = case["quantity"], case["arg"]
        if quantity.startswith("total_"):
            got = getattr(fit, quantity)
        else:
            per_origin = {
                "ultimate": lambda: fit.ultimate,
                "reserve": lambda: fit.reserve,
                "apriori": lambda: fit.trended_apriori,
                "detrended_apriori": lambda: fit.apriori,
            }[quantity]()
            got = per_origin[fit.origins.index(arg)]
        want = float(case["expected"])
        err = abs(got - want)
        abs_tol, rel_tol = float(case["abs_tol"] or 0), float(case["rel_tol"] or 0)
        assert got == want or err <= abs_tol or err <= rel_tol * abs(want), (case, got)


def test_expected_loss_family_identities():
    tri = premium_dataset("clrd_wkcomp")
    el = ExpectedLoss(apriori=0.7).fit(tri, "paid", "premium")
    bf = BornhuetterFerguson(apriori=0.7).fit(tri, "paid", "premium")
    assert isinstance(el, ExpectedLossFit) and isinstance(bf, ExpectedLossFit)
    assert Benktander(apriori=0.7, n_iters=0).fit(tri, "paid", "premium").ultimate == el.ultimate
    assert Benktander(apriori=0.7, n_iters=1).fit(tri, "paid", "premium").ultimate == bf.ultimate
    assert el.ultimate == pytest.approx([0.7 * e for e in el.exposure], rel=1e-15)
    assert el.apriori == [0.7] * 10
    assert el.exposure[0] == 1_691_130.0
    cl = ChainLadder().fit(tri, "paid")
    many = Benktander(apriori=0.7, n_iters=10_000).fit(tri, "paid", "premium")
    assert many.ultimate == pytest.approx(cl.ultimate, rel=1e-12)
    assert bf.chain_ladder.ultimate == cl.ultimate
    assert bf.ldf == cl.ldf and bf.cdf == cl.cdf and bf.latest == cl.latest
    assert bf.development == cl.development
    assert bf.total_reserve == pytest.approx(sum(bf.reserve), rel=1e-12)
    # Cape Cod without decay keeps each origin's own loss ratio: the chain ladder.
    cc = CapeCod(trend=0.05, decay=0.0).fit(tri, "paid", "premium")
    assert isinstance(cc, CapeCodFit)
    assert cc.ultimate == pytest.approx(cl.ultimate, rel=1e-12)
    assert cc.trended_apriori[-1] == cc.apriori[-1]
    assert cc.trended_apriori[0] / cc.apriori[0] == pytest.approx(1.05**9, rel=1e-12)
    assert cc.expected_loss.apriori == cc.apriori
    assert repr(Benktander(apriori=0.7, n_iters=3)) == (
        'Benktander(apriori=0.7, n_iters=3, average="volume", '
        'sigma_interpolation="log-linear", tail=1.0)'
    )
    assert repr(CapeCod()).startswith("CapeCod(trend=0.0, decay=1.0, ")
    assert repr(cc).startswith("CapeCodFit(origins=10, total_ultimate=")
    model = CapeCod(trend=0.05, decay=0.75, average="simple", tail=1.01)
    assert (model.trend, model.decay, model.average, model.tail) == (0.05, 0.75, "simple", 1.01)
    assert BornhuetterFerguson(apriori=0.6).apriori == 0.6
    assert Benktander(n_iters=4).n_iters == 4
    # The tail is ChainLadder's: a number or a tail estimator.
    bondy = BornhuetterFerguson(apriori=0.7, tail=TailBondy()).fit(tri, "paid", "premium")
    assert bondy.cdf == ChainLadder(tail=TailBondy()).fit(tri, "paid").cdf
    assert bondy.cdf != bf.cdf
    assert isinstance(CapeCod(tail=TailBondy()).tail, TailBondy)
    assert "tail=TailBondy(" in repr(Benktander(tail=TailBondy()))
    with pytest.raises(TypeError, match="tail must be"):
        ExpectedLoss(tail="1.05")


def test_expected_loss_every_segment_at_once():
    pytest.importorskip("pandas")
    origin, development, paid, premium, lob = [], [], [], [], []
    for name in PREMIUM_DATASETS:
        for r in premium_rows(name):
            origin.append(int(r["origin"]))
            development.append(int(r["development"]))
            paid.append(float(r["paid"]))
            premium.append(float(r["premium"]))
            lob.append(name)
    both = Triangle.from_long(
        origin, development, {"paid": paid, "premium": premium}, keys={"lob": lob}
    )
    assert both.index == PREMIUM_DATASETS
    wkcomp = premium_dataset("clrd_wkcomp")
    for model in [BornhuetterFerguson(apriori=0.6), CapeCod(trend=0.02, decay=0.8)]:
        fit = model.fit(both, "paid", "premium")
        assert fit.keys == ["lob"] and len(fit.origins) == 20
        alone = model.fit(wkcomp, "paid", "premium")
        seg = fit.segment(lob="clrd_wkcomp")
        assert seg.exposure == alone.exposure
        # Cape Cod trends to the triangle's valuation (2010 here, 1997
        # alone) and back, so the two agree to rounding.
        assert seg.ultimate == pytest.approx(alone.ultimate, rel=1e-14)
        assert fit.ultimate[:10] == seg.ultimate
        frame = fit.to_frame()
        assert list(frame["ultimate"]) == fit.ultimate
        assert list(frame["reserve"]) == fit.reserve
        assert list(frame["apriori"]) == fit.apriori
        totals = fit.totals_frame()
        assert totals["reserve"].iloc[0] == pytest.approx(alone.total_reserve, rel=1e-12)
        assert totals["exposure"].iloc[0] == sum(alone.exposure)
        assert len(fit.development_frame()) == 20
        with pytest.raises(ValueError, match="2 segments; use development_frame"):
            fit.ldf
        assert "segments=2" in repr(fit)
    cc = CapeCod().fit(both, "paid", "premium")
    assert list(cc.to_frame().columns) == [
        "lob", "origin", "latest", "ultimate", "reserve", "exposure", "apriori", "trended_apriori",
    ]
    # The chain ladder's ultimate is not the method's.
    assert cc.chain_ladder.ultimate != cc.ultimate


def test_expected_loss_errors():
    tri = Triangle.from_long(
        [2020, 2020, 2021],
        [12, 24, 12],
        {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, float("nan")]},
    )
    msg = "origin 2021 has no observed, finite, positive exposure in column premium"
    with pytest.raises(ValueError, match=msg):
        BornhuetterFerguson().fit(tri, "paid", "premium")
    with pytest.raises(ValueError, match="no column named exposure"):
        ExpectedLoss().fit(tri, "paid", "exposure")
    good = Triangle.from_long(
        [2020, 2020, 2021],
        [12, 24, 12],
        {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
    )
    with pytest.raises(ValueError, match="apriori = 0 is invalid"):
        BornhuetterFerguson(apriori=0.0).fit(good, "paid", "premium")
    with pytest.raises(ValueError, match="decay = 2 is invalid"):
        CapeCod(decay=2.0).fit(good, "paid", "premium")
    with pytest.raises(ValueError, match="trend = -1 is invalid"):
        CapeCod(trend=-1.0).fit(good, "paid", "premium")
    with pytest.raises(OverflowError):
        Benktander(n_iters=-1)
    with pytest.raises(ValueError):
        ExpectedLoss(average="median")


# Clark's growth curves: parity with R ChainLadder 0.2.21 and
# chainladder-python 0.10.1 (validation/reference/reserving_clark_*.csv; see
# validation/tests/reserving_clark.rs for the tolerances).


def clark_triangle(name):
    return premium_dataset(name) if name == "genins_premium" else dataset(name)


def clark_fit(tri, dataset_name, method):
    """Fits a reference `method` (name;curve=...;max_age=...[;optim=...])."""
    name, *settings = method.split(";")
    kw = dict(s.split("=") for s in settings)
    max_age = None if kw["max_age"] == "inf" else float(kw["max_age"])
    if name == "clark_cape_cod":
        return ClarkCapeCod(curve=kw["curve"], max_age=max_age).fit(tri, "paid", "premium")
    column = "paid" if dataset_name == "genins_premium" else "values"
    return ClarkLdf(curve=kw["curve"], max_age=max_age).fit(tri, column)


def clark_value(fit, quantity, arg):
    p = len(fit.covariance)
    scalars = {
        "omega": lambda: fit.omega,
        "theta": lambda: fit.theta,
        "elr": lambda: fit.elr,
        "sigma2": lambda: fit.scale,
        "omega_se": lambda: math.sqrt(fit.covariance[p - 2][p - 2]),
        "theta_se": lambda: math.sqrt(fit.covariance[p - 1][p - 1]),
        "elr_se": lambda: math.sqrt(fit.covariance[0][0]),
        "total_ultimate": lambda: fit.total_ultimate,
        "total_reserve": lambda: fit.total_reserve,
        "total_process_se": lambda: fit.total_process_risk,
        "total_parameter_se": lambda: fit.total_parameter_risk,
        "total_standard_error": lambda: fit.total_standard_error,
        "ldf": lambda: fit.growth(float(arg) + 12) / fit.growth(float(arg)),
    }
    if quantity in scalars:
        return scalars[quantity]()
    per_origin = {
        "expected_ultimate": fit.expected_ultimate,
        "ultimate": fit.ultimate,
        "reserve": fit.reserve,
        "process_se": fit.process_risk,
        "parameter_se": fit.parameter_risk,
        "standard_error": fit.standard_error,
    }[quantity]
    return per_origin[fit.origins.index(arg)]


@pytest.mark.parametrize("reference", ["reserving_clark_r.csv", "reserving_clark_python.csv"])
def test_clark_matches_reference(reference):
    cases = read_csv(VALIDATION / "reference" / reference)
    assert cases
    tris, fits = {}, {}
    for case in cases:
        name = case["dataset"]
        # The optimizer setting of the R reference does not change our fit.
        method = ";".join(p for p in case["method"].split(";") if not p.startswith("optim="))
        if name not in tris:
            tris[name] = clark_triangle(name)
        if (name, method) not in fits:
            fits[(name, method)] = clark_fit(tris[name], name, method)
        got = clark_value(fits[(name, method)], case["quantity"], case["arg"])
        want = float(case["expected"])
        err = abs(got - want)
        abs_tol, rel_tol = float(case["abs_tol"] or 0), float(case["rel_tol"] or 0)
        assert got == want or err <= abs_tol or err <= rel_tol * abs(want), (case, got)


def test_clark_fit_fields(triangles):
    raa = triangles["raa"]
    fit = ClarkLdf().fit(raa, "values")
    assert isinstance(fit, ClarkFit)
    assert fit.curve == "loglogistic" and fit.max_age is None and fit.elr is None
    assert fit.exposure is None
    assert fit.latest == ChainLadder().fit(raa, "values").latest
    # Every origin starts at age 0, so U_i = latest / G(latest age); the
    # 1990 origin is at 12 months, 6 from the average date of loss.
    assert fit.expected_ultimate[-1] == pytest.approx(fit.latest[-1] / fit.growth(12), rel=1e-12)
    assert fit.growth(float("inf")) == 1.0
    assert fit.reserve == pytest.approx([u - l for u, l in zip(fit.ultimate, fit.latest)])
    assert fit.total_reserve == pytest.approx(sum(fit.reserve), rel=1e-12)
    se = [math.hypot(p, q) for p, q in zip(fit.process_risk, fit.parameter_risk)]
    assert fit.standard_error == pytest.approx(se, rel=1e-12)
    assert len(fit.covariance) == 12 and fit.scale > 0
    # RAA has 55 observed incremental values (act_reserving's unit test).
    assert fit.n_observations == 55 and fit.origin_width == 12.0
    assert repr(ClarkLdf(curve="weibull", max_age=240)) == 'ClarkLdf(curve="weibull", max_age=240.0)'
    assert repr(ClarkCapeCod()) == 'ClarkCapeCod(curve="loglogistic", max_age=None)'
    assert repr(fit).startswith('ClarkFit(method="ldf", curve="loglogistic", origins=10, ')
    cc = ClarkCapeCod(curve="weibull").fit(premium_dataset("genins_premium"), "paid", "premium")
    assert cc.exposure[0] == 10_000_000.0
    assert cc.expected_ultimate == [cc.elr * e for e in cc.exposure]
    assert len(cc.covariance) == 3 and cc.curve == "weibull"
    assert repr(cc).startswith('ClarkFit(method="cape_cod", curve="weibull", ')


def test_clark_every_segment_at_once():
    pytest.importorskip("pandas")
    origin, development, paid, premium, lob = [], [], [], [], []
    for name, factor in [("a", 1.0), ("b", 10.0)]:
        for r in premium_rows("genins_premium"):
            origin.append(int(r["origin"]))
            development.append(int(r["development"]))
            paid.append(float(r["paid"]) * factor)
            premium.append(float(r["premium"]))
            lob.append(name)
    both = Triangle.from_long(
        origin, development, {"paid": paid, "premium": premium}, keys={"lob": lob}
    )
    alone = premium_dataset("genins_premium")
    for model, fit_one in [
        (ClarkLdf(max_age=240), lambda m, t: m.fit(t, "paid")),
        (ClarkCapeCod(curve="weibull"), lambda m, t: m.fit(t, "paid", "premium")),
    ]:
        fit = fit_one(model, both)
        single = fit_one(model, alone)
        a, b = fit.segment(lob="a"), fit.segment(lob="b")
        assert a.ultimate == pytest.approx(single.ultimate, rel=1e-9)
        assert a.omega == pytest.approx(single.omega, rel=1e-9)
        # Ten times the losses: the same curve, ten times the amounts and
        # standard errors.
        assert b.theta == pytest.approx(a.theta, rel=1e-7)
        assert b.total_standard_error == pytest.approx(10 * a.total_standard_error, rel=1e-6)
        assert len(fit.origins) == 20 and fit.keys == ["lob"]
        frame = fit.to_frame()
        assert list(frame["standard_error"]) == fit.standard_error
        totals = fit.totals_frame()
        assert list(totals["omega"]) == [a.omega, b.omega]
        assert totals["reserve"].iloc[0] == pytest.approx(a.total_reserve, rel=1e-12)
        with pytest.raises(ValueError, match="2 segments; use totals_frame"):
            fit.omega
        with pytest.raises(ValueError, match="2 segments; use segment"):
            fit.growth(12)
        with pytest.raises(ValueError, match="2 segments; use segment"):
            fit.n_observations
        assert fit.origin_width == 12.0
        if fit.exposure is None:
            assert fit.elr is None
        else:
            with pytest.raises(ValueError, match="2 segments; use totals_frame"):
                fit.elr
        assert "segments=2" in repr(fit)
    cc = ClarkCapeCod().fit(both, "paid", "premium")
    assert list(cc.to_frame().columns) == [
        "lob", "origin", "latest", "ultimate", "reserve", "exposure", "expected_ultimate",
        "process_risk", "parameter_risk", "standard_error",
    ]
    assert "elr" in cc.totals_frame().columns


def test_clark_errors(triangles):
    raa = triangles["raa"]
    with pytest.raises(ValueError, match="unknown growth curve"):
        ClarkLdf(curve="gompertz")
    with pytest.raises(ValueError, match="max_age = 119 is invalid"):
        ClarkLdf(max_age=119).fit(raa, "values")
    with pytest.raises(ValueError, match="at least 4 development ages"):
        three = Triangle.from_long([2020] * 3 + [2021] * 2 + [2022], [12, 24, 36, 12, 24, 12], [1.0, 2, 3, 1, 2, 1])
        ClarkLdf().fit(three, "values")
    with pytest.raises(ValueError, match="no column named premium"):
        ClarkCapeCod().fit(raa, "values", "premium")


# The simulated one-year view (OdpBootstrap.one_year), as in
# validation/tests/reserving_one_year_bootstrap.rs: same seed and number of
# simulations, so the measured ratio to R's Merz-Wuthrich CDR(1)S.E. is the
# same; the tolerance is five Monte Carlo standard errors of the SD.
ONE_YEAR_SIMS = 20_000
ONE_YEAR_SEED = 20_261_006


def test_one_year_chain_ladder_against_merz_wuthrich(triangles):
    genins = triangles["genins"]
    fit = OdpBootstrap(n_sims=ONE_YEAR_SIMS, seed=ONE_YEAR_SEED).one_year(genins, "values", ChainLadder())
    assert isinstance(fit, OneYearFit)
    rows = read_csv(VALIDATION / "reference" / "reserving_cdr_r.csv")
    (mw,) = [
        float(r["expected"])
        for r in rows
        if r["dataset"] == "genins" and r["method"] == "cdr" and r["quantity"] == "total_one_year_se"
    ]
    sd = math.sqrt(fit.cdr.variance())
    assert abs(sd / mw - 1.0234) < 0.0277
    # The opening ultimate is the chain ladder's; the CDR is centred near 0.
    cl = ChainLadder().fit(genins, "values")
    assert fit.opening_ultimate == cl.ultimate
    assert fit.opening_reserve == pytest.approx(cl.reserve, abs=1e-6)
    assert abs(fit.cdr.mean()) < 0.1 * sd


def test_one_year_fields_and_frames(triangles):
    raa = triangles["raa"]
    fit = OdpBootstrap(n_sims=500, seed=1).one_year(raa, "values", ChainLadder(tail=1.05))
    assert fit.origins == [str(y) for y in range(1981, 1991)]
    assert fit.keys == [] and fit.index == ["Total"]
    assert fit.cdr.dims == ["origin"]
    assert fit.cdr.components() == [(str(y),) for y in range(1981, 1991)]
    assert fit.cdr.n_sims == 500
    assert fit.scale == pytest.approx(983.635027030873, rel=1e-9)
    assert fit.latest == ChainLadder().fit(raa, "values").latest
    # The bootstrap's chain ladder has no tail; the method's does.
    assert fit.chain_ladder.tail == 1.0
    assert fit.opening_ultimate == ChainLadder(tail=1.05).fit(raa, "values").ultimate
    # 1981 is at the last age: no new cell, and a constant tail does not move.
    assert all(row[0] == 0.0 for row in fit.cdr.draw_matrix())
    assert fit.cdr.provenance()["model"] == "odp_bootstrap_one_year"
    assert repr(fit).startswith("OneYearFit(origins=10, n_sims=500, scale=983.63")
    again = OdpBootstrap(n_sims=500, seed=1).one_year(raa, "values", ChainLadder(tail=1.05))
    other = OdpBootstrap(n_sims=500, seed=2).one_year(raa, "values", ChainLadder(tail=1.05))
    assert fit.cdr.draw_matrix() == again.cdr.draw_matrix()
    assert fit.cdr.draw_matrix() != other.cdr.draw_matrix()
    pytest.importorskip("pandas")
    frame = fit.to_frame()
    assert list(frame.columns) == [
        "origin",
        "latest",
        "opening_ultimate",
        "opening_reserve",
        "cdr_mean",
        "cdr_std_dev",
    ]
    assert frame["cdr_std_dev"].iloc[0] == 0.0
    totals = fit.totals_frame()
    assert list(totals.columns) == [
        "latest",
        "opening_ultimate",
        "opening_reserve",
        "scale",
        "cdr_mean",
        "cdr_std_dev",
    ]
    assert totals["cdr_std_dev"].iloc[0] == pytest.approx(math.sqrt(fit.cdr.variance()), rel=1e-6)


def test_one_year_expected_loss_methods():
    tri = premium_dataset("genins_premium")
    boot = OdpBootstrap(n_sims=300, seed=4)
    methods = [
        BornhuetterFerguson(apriori=0.6),
        Benktander(apriori=0.6, n_iters=2),
        CapeCod(trend=0.02, decay=0.9),
    ]
    for method in methods:
        fit = boot.one_year(tri, "paid", method, exposure="premium")
        assert fit.opening_ultimate == method.fit(tri, "paid", "premium").ultimate, method
        assert fit.cdr.variance() > 0.0, method
    # The expected loss ratio ultimate ignores the losses: its CDR is zero.
    elr = boot.one_year(tri, "paid", ExpectedLoss(apriori=0.6), exposure="premium")
    assert all(x == 0.0 for row in elr.cdr.draw_matrix() for x in row)


def test_one_year_every_segment_at_once():
    data = {
        "lob": ["Auto"] * 6 + ["Home"] * 6,
        "year": [2020, 2020, 2020, 2021, 2021, 2022] * 2,
        "age": [12, 24, 36, 12, 24, 12] * 2,
        "paid": [100.0, 150.0, 165.0, 110.0, 170.0, 120.0, 50.0, 80.0, 85.0, 60.0, 90.0, 70.0],
    }
    tri = Triangle.from_frame(data, "year", "age", "paid", keys="lob")
    fit = OdpBootstrap(n_sims=400, seed=5).one_year(tri, "paid", ChainLadder())
    assert fit.keys == ["lob"] and fit.index == ["Auto", "Home"]
    assert fit.cdr.dims == ["lob", "origin"]
    assert len(fit.cdr.components()) == 6
    assert fit.cdr.aggregate(["lob"]).components() == [("Auto",), ("Home",)]
    with pytest.raises(ValueError, match="scale needs a single-segment fit"):
        fit.scale
    home = fit.segment(lob="Home")
    assert home.cdr.components() == [("Home", "2020"), ("Home", "2021"), ("Home", "2022")]
    assert home.opening_ultimate == fit.opening_ultimate[3:]
    assert repr(fit).startswith("OneYearFit(segments=2, origins=6, n_sims=400")


def test_one_year_errors(triangles):
    tri = premium_dataset("genins_premium")
    boot = OdpBootstrap(n_sims=10)
    with pytest.raises(TypeError, match="method must be ChainLadder, ExpectedLoss"):
        boot.one_year(tri, "paid", Mack())
    with pytest.raises(ValueError, match="BornhuetterFerguson needs an exposure column"):
        boot.one_year(tri, "paid", BornhuetterFerguson())
    with pytest.raises(ValueError, match="ChainLadder takes no exposure column"):
        boot.one_year(tri, "paid", ChainLadder(), exposure="premium")
    with pytest.raises(ValueError, match="no column named exposure"):
        boot.one_year(tri, "paid", CapeCod(), exposure="exposure")
    with pytest.raises(ValueError, match="no column named"):
        boot.one_year(triangles["raa"], "paid", ChainLadder())
