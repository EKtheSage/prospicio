"""Recoveries, ALAE and ULAE (``docs/design/reserving-v02.md``, decision 37).

The Rust core is checked exhibit by exhibit in
``validation/tests/reserving_expense.rs``; these tests reproduce Friedland's
headline numbers through the Python bindings (*Estimating Unpaid Claims Using
Basic Techniques*, chapters 14, 16 and 17) and check the bindings' arguments
and errors. ``R/prospicio/tests/test-reserving-expense.R`` asserts the same
numbers."""

import csv
import math
from pathlib import Path

import pytest

from prospicio.reserving import (
    ChainLadder,
    RatioDevelopment,
    Triangle,
    ulae_classical,
    ulae_conger_nolibos,
    ulae_kittel,
)

DATA = Path(__file__).resolve().parents[2] / "validation" / "data"
AGES = [12 * k for k in range(1, 12)]


def book(name):
    with open(DATA / name, newline="") as file:
        rows = list(csv.DictReader(file))
    columns = [c for c in rows[0] if c not in ("origin", "age")]
    return Triangle.from_long(
        [int(r["origin"]) for r in rows],
        [int(r["age"]) for r in rows],
        {c: [float(r[c]) for r in rows] for c in columns},
    )


def printed(cdf):
    return ChainLadder(ldf=dict(zip(AGES, cdf)), ldf_kind="cdf")


def averaged(tri, reported_cdf, paid_cdf):
    """The selected claims: the average of the two chain ladders."""
    reported = printed(reported_cdf).fit(tri, "reported").ultimate
    paid = printed(paid_cdf).fit(tri, "paid").ultimate
    return [(r + p) / 2 for r, p in zip(reported, paid)]


def physical_damage():
    tri = book("friedland_auto_physical_damage.csv")
    claims = averaged(
        tri,
        [1.115, 1.001, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1.279, 1.005, 1.001, 1, 1, 1, 1, 1, 1, 1, 1],
    )
    return tri, claims


def property_damage():
    tri = book("friedland_auto_property_damage.csv")
    claims = averaged(
        tri,
        [1.101, 0.990, 0.989, 0.991, 0.993, 0.998, 0.999, 0.999, 1, 1, 1],
        [1.584, 1.029, 1.007, 1.004, 1.002, 1.001, 1.001, 1.001, 1, 1, 1],
    )
    return tri, claims


def close(got, want, tolerance):
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert abs(g - w) <= tolerance, (g, w)


SS_RATIO_CDF = [1.499, 1.009, 1, 1, 1, 1, 1, 1, 1, 1, 1]
ALAE_RATIO_CDF = [1.332, 1.201, 1.140, 1.086, 1.050, 1.021, 1.007, 1.003, 1.002, 1, 1]


def test_salvage_and_subrogation_ratio_method():
    tri, claims = physical_damage()
    fit = RatioDevelopment(printed(SS_RATIO_CDF), selected={2008: 0.345}).fit(
        tri, "received_ss", "paid", claims
    )
    assert fit.origins[-1] == "2008"
    assert fit.ages == AGES
    assert fit.selected_ratio[-1] == 0.345
    assert fit.ratios[-1][1] is None
    # 5,957 / 16,822 x 1.009 x 16,893: the unrounded ratio gives 6,036.
    assert abs(fit.ultimate[9] - 6036) <= 1
    assert abs(fit.total_ultimate - 44934) <= 2
    assert abs(fit.total_unpaid - 3054) <= 2
    close(fit.unpaid[-2:], [79, 2966], 1)


def test_the_development_settings_reach_the_ratio():
    tri, claims = physical_damage()
    medial = ChainLadder(average="simple", n_periods=5, drop_high=1, drop_low=1)
    fit = RatioDevelopment(medial).fit(tri, "received_ss", "paid", claims)
    close(fit.averages[:2], [1.486, 1.009], 0.0005)
    close(fit.cdf, SS_RATIO_CDF, 0.0015)


def test_alae_ratio_method_below_paid():
    tri, claims = property_damage()
    fit = RatioDevelopment(printed(ALAE_RATIO_CDF), selected={2007: 0.0077, 2008: 0.0077}).fit(
        tri, "paid_alae", "paid", claims
    )
    assert abs(fit.total_ultimate - 12485) <= 2
    # The selected ratio puts 2007 below its paid ALAE, as printed.
    assert abs(fit.unpaid[9] + 94) <= 1
    assert abs(fit.ultimate[8] - 1302) <= 1
    assert abs(fit.total_unpaid - 800) <= 2


def test_alae_additive_method():
    tri, claims = property_damage()
    latest_3 = ChainLadder(average="simple", n_periods=3)
    fit = RatioDevelopment(latest_3, additive=True).fit(tri, "paid_alae", "paid", claims)
    assert abs(fit.total_ultimate - 13215) <= 2
    assert abs(fit.total_unpaid - 1530) <= 2
    assert round(fit.cdf[0], 4) == 0.0019
    # A selected difference and a tail move the additions.
    moved = RatioDevelopment(
        latest_3, additive=True, additive_selected={12: 0.001}, additive_tail=0.0001
    ).fit(tri, "paid_alae", "paid", claims)
    assert moved.factors[0] == 0.001
    assert moved.cdf[-1] == 0.0001


def test_ratio_development_defaults_to_the_simple_average():
    tri, claims = property_damage()
    fit = RatioDevelopment().fit(tri, "paid_alae", "paid", claims)
    simple = ChainLadder(average="simple")
    assert fit.factors == RatioDevelopment(simple).fit(tri, "paid_alae", "paid", claims).factors


def test_ratio_development_refuses_bad_arguments():
    tri, claims = property_damage()
    with pytest.raises(ValueError, match="additive"):
        RatioDevelopment(additive_tail=0.1)
    with pytest.raises(ValueError):
        RatioDevelopment(selected={1990: 0.1}).fit(tri, "paid_alae", "paid", claims)
    with pytest.raises(ValueError):
        RatioDevelopment().fit(tri, "paid_alae", "paid", claims[:-1])
    with pytest.raises(Exception, match="additive|simple|average"):
        RatioDevelopment(ChainLadder(), additive=True).fit(tri, "paid_alae", "paid", claims)


XYZ_ULAE = [14.352e6, 15.321e6, 16.870e6, 17.112e6, 17.331e6]
XYZ_PAID = [333e6, 358e6, 334e6, 347e6, 391e6]
XYZ_INCURRED = [535.213e6, 492.265e6, 435.985e6, 432.966e6, 475.300e6]


def test_classical():
    fit = ulae_classical(XYZ_ULAE, XYZ_PAID, [603e6], [316e6], ratio=0.045)
    assert fit.ratio == 0.045
    assert round(fit.total_ratio, 3) == 0.046
    assert fit.total == pytest.approx(27_787_500)
    pure = ulae_classical(XYZ_ULAE, XYZ_PAID, [603e6], [316e6], ratio=0.045, pure_ibnr=[19e6])
    assert pure.total == pytest.approx(21_105_000)
    assert pure.expected_claims is None


def test_classical_opening_share():
    # Friedland's erratum to page 388: 25% opening, 75% of the ratio on case.
    fit = ulae_classical(XYZ_ULAE, XYZ_PAID, [100.0], [10.0], ratio=0.1, opening=0.25)
    assert fit.total == pytest.approx(0.1 * (0.75 * 100 + 10))


def test_kittel():
    fit = ulae_kittel(XYZ_ULAE, XYZ_PAID, XYZ_INCURRED, [603e6], [316e6], ratio=0.040)
    assert fit.basis[0] == pytest.approx(434_106_500)
    assert fit.total == pytest.approx(24_700_000)
    pure = ulae_kittel(
        XYZ_ULAE, XYZ_PAID, XYZ_INCURRED, [603e6], [316e6], ratio=0.040, pure_ibnr=[19e6]
    )
    assert pure.total == pytest.approx(18_760_000)


def test_the_ratio_defaults_to_the_total():
    fit = ulae_classical(XYZ_ULAE, XYZ_PAID, [603e6], [316e6])
    assert fit.ratio == fit.total_ratio == sum(XYZ_ULAE) / sum(XYZ_PAID)


PQR_ULAE = [1978.0, 4820.0, 8558.0, 12039.0, 13143.0, 15286.0]
PQR_PAID = [4590.0, 14600.0, 38390.0, 58297.0, 86074.0, 105466.0]
PQR_ULT_REPORTED = [27200.0, 76700.0, 106900.0, 154300.0, 163100.0, 176400.0]
PQR_ULTIMATE = [28600.0, 79200.0, 108400.0, 156700.0, 163400.0, 177100.0]


@pytest.mark.parametrize(
    ("weights", "bf", "development"),
    [((0.6, 0.4), 16767, 17152), ((0.7, 0.3, 0.0), 12795, 12201)],
)
def test_conger_nolibos(weights, bf, development):
    fit = ulae_conger_nolibos(
        PQR_ULAE,
        PQR_PAID,
        PQR_ULT_REPORTED,
        [292_130.0],
        [113_853.0],
        weights=weights,
        ratio=0.100,
        pure_ibnr=[8800.0],
        ultimate=PQR_ULTIMATE,
    )
    assert abs(fit.expected_claims - 15516) <= 0.5
    assert abs(fit.bornhuetter_ferguson - bf) <= 0.5
    assert abs(fit.development - development) <= 0.5
    assert abs(fit.total - bf) <= 0.5


def test_conger_nolibos_simplification():
    fit = ulae_conger_nolibos(
        PQR_ULAE,
        PQR_PAID,
        PQR_ULTIMATE,
        [292_130.0],
        [113_853.0],
        weights=(0.6, 0.4),
        ratio=0.100,
        pure_ibnr=[0.04 * 177_100],
    )
    assert abs(fit.total - 16664) <= 0.5
    assert fit.bornhuetter_ferguson is None


def test_conger_nolibos_closing_share():
    fit = ulae_conger_nolibos(
        [10.0],
        [100.0],
        [200.0],
        [30.0],
        [20.0],
        weights=(0.5, 0.25, 0.25),
        pure_ibnr=[8.0],
        closed=[80.0],
        unclosed=[60.0],
    )
    assert fit.basis == [0.5 * 200 + 0.25 * 100 + 0.25 * 80]
    assert fit.total == pytest.approx(fit.ratio * (0.5 * 8 + 0.25 * 50 + 0.25 * 60))


def test_ulae_refuses_bad_arguments():
    with pytest.raises(ValueError, match="weights"):
        ulae_conger_nolibos([1.0], [1.0], [1.0], [1.0], [1.0], weights=(1.0,))
    with pytest.raises(TypeError, match="weights"):
        ulae_conger_nolibos([1.0], [1.0], [1.0], [1.0], [1.0], weights="even")
    with pytest.raises(ValueError):
        ulae_conger_nolibos([1.0], [1.0], [1.0], [1.0], [1.0], weights=(0.6, 0.4))
    with pytest.raises(ValueError):
        ulae_classical([1.0, 2.0], [1.0], [1.0], [1.0])
    with pytest.raises(ValueError):
        ulae_classical([1.0], [1.0], [1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        ulae_classical([1.0], [1.0], [1.0], [1.0], opening=1.5)
    with pytest.raises(ValueError):
        ulae_classical([math.nan], [1.0], [1.0], [1.0])
