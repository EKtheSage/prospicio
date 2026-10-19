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


def test_n_links_and_excluded_need_a_single_segment(raa):
    cells = [(o, 12 * d) for o in range(1981, 1991) for d in range(1, 1992 - o)]
    data = {
        "lob": ["a"] * len(cells) + ["b"] * len(cells),
        "origin": [o for o, _ in cells] * 2,
        "development": [age for _, age in cells] * 2,
        "paid": [float(v) for v in range(1, len(cells) + 1)] * 2,
    }
    tri = Triangle.from_frame(data, "origin", "development", "paid", keys="lob")
    fit = ChainLadder(n_periods=3).fit(tri, "paid")
    for name in ("n_links", "excluded"):
        with pytest.raises(ValueError, match=f"{name} needs a single-segment fit"):
            getattr(fit, name)
    assert fit.segment(lob="a").excluded is True
    assert fit.segment(lob="b").n_links[0] == 3


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


def test_geometric_is_rejected_by_the_one_year_views(raa):
    with pytest.raises(ValueError, match="geometric.*not supported"):
        MackBootstrap(n_sims=10, average="geometric").one_year(raa, "values", ChainLadder())
    with pytest.raises(ValueError, match="geometric.*not supported"):
        OdpBootstrap(n_sims=10).one_year(raa, "values", ChainLadder(average="geometric"))


def test_odp_bootstrap_settings_round_trip(raa):
    boot = OdpBootstrap(n_sims=10, n_periods=3, drop_high=1, drop_low=1)
    assert (boot.n_periods, boot.drop_high, boot.drop_low) == (3, 1, 1)
    assert "n_periods=3, drop_high=1, drop_low=1" in repr(boot)
    plain = OdpBootstrap(n_sims=10)
    assert (plain.n_periods, plain.drop_high, plain.drop_low) == (None, 0, 0)
    assert "n_periods" not in repr(plain) and "drop" not in repr(plain)
    with pytest.raises(ValueError, match="n_periods"):
        OdpBootstrap(n_sims=10, n_periods=0).fit(raa, "values")


def test_the_bootstraps_follow_the_exclusions(raa):
    for cls in (OdpBootstrap, MackBootstrap):
        plain = cls(n_sims=200, seed=7).fit(raa, "values")
        # Settings that leave out no link ratio change nothing, seed for seed.
        same = cls(n_sims=200, seed=7, n_periods=50).fit(raa, "values")
        assert same.reserves.mean() == plain.reserves.mean()
        cut = cls(n_sims=200, seed=7, n_periods=3).fit(raa, "values")
        dropped = cls(n_sims=200, seed=7, drop_high=1, drop_low=1).fit(raa, "values")
        assert cut.reserves.mean() != plain.reserves.mean()
        assert dropped.reserves.mean() != plain.reserves.mean()


def test_odp_scale_is_biased_low_under_exclusions(raa):
    plain = OdpBootstrap(n_sims=10, seed=1).fit(raa, "values")
    cut = OdpBootstrap(n_sims=10, seed=1, n_periods=3).fit(raa, "values")
    assert plain.scale == pytest.approx(983.6, abs=0.1)
    assert cut.scale == pytest.approx(694.0, abs=0.1)


def test_the_one_year_views_follow_the_method_s_exclusions(raa):
    method = ChainLadder(n_periods=3)
    for boot in (OdpBootstrap(n_sims=200, seed=3), MackBootstrap(n_sims=200, seed=3)):
        inherited = boot.one_year(raa, "values", method)
        assert inherited.cdr.variance() > 0
        # Naming the method's own exclusions is the same as inheriting them.
        own = type(boot)(n_sims=200, seed=3, n_periods=3).one_year(raa, "values", method)
        assert own.cdr.mean() == inherited.cdr.mean()
        # The window moves with the coming diagonal, so the mean CDR is not
        # the unrestricted view's.
        plain = boot.one_year(raa, "values", ChainLadder())
        assert inherited.cdr.mean() != plain.cdr.mean()


def test_a_bootstrap_whose_exclusions_differ_from_the_method_s_errors(raa):
    for cls in (OdpBootstrap, MackBootstrap):
        with pytest.raises(ValueError, match="exclusions"):
            cls(n_sims=10, n_periods=3).one_year(raa, "values", ChainLadder(n_periods=4))
        with pytest.raises(ValueError, match="exclusions"):
            cls(n_sims=10, drop_high=1).one_year(raa, "values", ChainLadder())


def test_the_closed_form_still_rejects_the_exclusions(raa):
    fit = Mack(n_periods=3).fit(raa, "values")
    with pytest.raises(ValueError, match="MackBootstrap"):
        fit.claims_development_result()


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
