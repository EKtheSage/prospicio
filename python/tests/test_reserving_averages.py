"""Averages over the latest periods, excluding high and low link ratios, and
the geometric average (``docs/design/reserving-v02.md``, decision 13). The
Rust core is checked against chainladder-python in
``validation/tests/reserving_averages.rs``; these tests check the bindings'
arguments, properties, repr and errors, and the same RAA numbers through
Python."""

import csv
import math
from pathlib import Path

import pytest

from prospicio.reserving import (
    Benktander,
    BornhuetterFerguson,
    CapeCod,
    ChainLadder,
    ExpectedLoss,
    Mack,
    MackBootstrap,
    OdpBootstrap,
    Triangle,
)

VALIDATION = Path(__file__).resolve().parents[2] / "validation"
REFERENCE = VALIDATION / "reference" / "reserving_averages_python.csv"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


@pytest.fixture(scope="module")
def raa():
    rows = read_csv(VALIDATION / "data" / "raa.csv")
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows],
        development=[int(r["development"]) for r in rows],
        values={"values": [float(r["value"]) for r in rows]},
    )


def settings(method):
    """``average=volume;n_periods=3`` as keyword arguments."""
    out = {}
    for part in method.split(";"):
        name, value = part.split("=")
        out[name] = value if name == "average" else int(value)
    return out


CASES = {}
for row in read_csv(REFERENCE):
    if row["dataset"] == "raa":
        CASES.setdefault(row["method"], []).append(row)


def test_defaults_keep_the_current_fit(raa):
    plain = ChainLadder().fit(raa, "values")
    cl = ChainLadder()
    assert (cl.n_periods, cl.drop_high, cl.drop_low) == (None, 0, 0)
    assert "n_periods" not in repr(cl) and "drop_" not in repr(cl)
    assert repr(cl).startswith("ChainLadder(average=\"volume\", sigma_interpolation=")
    assert plain.excluded is False
    assert plain.n_links == [9, 8, 7, 6, 5, 4, 3, 2, 1]
    same = ChainLadder(n_periods=None, drop_high=0, drop_low=0).fit(raa, "values")
    assert same.ldf == plain.ldf


@pytest.mark.parametrize("cls", [ChainLadder, Mack, MackBootstrap])
def test_settings_round_trip_and_repr(cls):
    m = cls(n_periods=3, drop_high=1, drop_low=2)
    assert (m.n_periods, m.drop_high, m.drop_low) == (3, 1, 2)
    assert "n_periods=3, drop_high=1, drop_low=2" in repr(m)
    only = cls(drop_low=1)
    assert only.n_periods is None and only.drop_high == 0
    assert "drop_low=1" in repr(only) and "n_periods" not in repr(only)


@pytest.mark.parametrize("cls", [ExpectedLoss, BornhuetterFerguson, Benktander, CapeCod])
def test_expected_loss_methods_take_the_settings(cls):
    m = cls(n_periods=2, drop_high=1)
    assert (m.n_periods, m.drop_high, m.drop_low) == (2, 1, 0)
    assert "n_periods=2, drop_high=1" in repr(m)
    assert "n_periods" not in repr(cls())


def test_geometric_round_trips():
    assert ChainLadder(average="geometric").average == "geometric"
    assert 'average="geometric"' in repr(ChainLadder(average="geometric"))
    with pytest.raises(ValueError, match="geometric"):
        ChainLadder(average="harmonic")


@pytest.mark.parametrize("method", sorted(CASES))
def test_raa_matches_chainladder_python(raa, method):
    cl = ChainLadder(**settings(method)).fit(raa, "values")
    ages = cl.development
    checked = set()
    for case in CASES[method]:
        q, arg, want = case["quantity"], case["arg"], float(case["expected"])
        if q in ("ldf", "n_links", "sigma", "std_err"):
            i = ages.index(int(float(arg)))
            got = {
                "ldf": cl.ldf,
                "n_links": cl.n_links,
                "sigma": cl.sigma,
                "std_err": cl.std_err,
            }[q][i]
        elif q == "ultimate":
            got = cl.ultimate[cl.origins.index(str(int(float(arg))))]
        elif q == "total_ultimate":
            got = sum(cl.ultimate)
        else:
            continue
        checked.add(q)
        if math.isnan(want):
            assert math.isnan(got)
        else:
            assert got == pytest.approx(want, rel=1e-9), (method, q, arg)
    assert {"ldf", "n_links"} <= checked


def test_excluded_and_n_links(raa):
    assert ChainLadder(n_periods=3).fit(raa, "values").n_links == [3] * 7 + [2, 1]
    assert ChainLadder(n_periods=3).fit(raa, "values").excluded is True
    # More diagonals than the triangle has leaves every link ratio.
    big = ChainLadder(n_periods=20).fit(raa, "values")
    assert big.excluded is False and big.n_links == [9, 8, 7, 6, 5, 4, 3, 2, 1]
    # Drops are skipped at an age where they would leave none.
    assert ChainLadder(drop_high=1, drop_low=1).fit(raa, "values").n_links[-2:] == [2, 1]
    assert Mack(n_periods=4).fit(raa, "values").n_links[0] == 4
    assert Mack(n_periods=4).fit(raa, "values").excluded is True


def test_geometric_is_the_geometric_mean_of_the_link_ratios(raa):
    fit = ChainLadder(average="geometric").fit(raa, "values")
    links = raa.link_ratios().to_frame() if hasattr(raa.link_ratios(), "to_frame") else None
    if links is None:
        pytest.skip("link ratios not available as a frame")
    for i, age in enumerate(fit.development[:-1]):
        col = [v for v in links[links["development"] == age]["values"] if not math.isnan(v)]
        want = math.exp(sum(math.log(v) for v in col) / len(col))
        assert fit.ldf[i] == pytest.approx(want, rel=1e-12)


def test_geometric_is_for_factors_only(raa):
    ChainLadder(average="geometric").fit(raa, "values")
    for model in (Mack(average="geometric"), MackBootstrap(n_sims=10, average="geometric")):
        with pytest.raises(ValueError, match="geometric.*not supported"):
            model.fit(raa, "values")


def test_bootstraps_and_one_year_views_reject_exclusions(raa):
    with pytest.raises(ValueError, match="n_periods.*not supported"):
        MackBootstrap(n_sims=10, n_periods=3).fit(raa, "values")
    with pytest.raises(ValueError, match="not supported"):
        OdpBootstrap(n_sims=10).one_year(raa, "values", method=ChainLadder(n_periods=3))


def test_zero_periods_is_an_invalid_setting(raa):
    with pytest.raises(ValueError, match="n_periods"):
        ChainLadder(n_periods=0).fit(raa, "values")


def test_a_cumulative_pattern_may_sit_below_one():
    # Friedland's XYZ reported pattern falls below 1 after 96 months.
    years = [2019] * 4 + [2020] * 3 + [2021] * 2 + [2022]
    ages = [12, 24, 36, 48, 12, 24, 36, 12, 24, 12]
    values = [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0]
    tri = Triangle.from_long(years, ages, values)
    cdf = {12: 1.6, 24: 1.1, 36: 0.99, 48: 1.0}
    fit = ChainLadder(ldf=cdf, ldf_kind="cdf").fit(tri, "values")
    assert fit.ldf == pytest.approx([1.6 / 1.1, 1.1 / 0.99, 0.99])
    # Not monotone, in either direction.
    ChainLadder(ldf={12: 0.9, 24: 1.2, 36: 1.0}, ldf_kind="cdf").fit(tri, "values")
