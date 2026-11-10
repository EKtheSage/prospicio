"""Ratemaking: the parallelogram method and trend, against Werner and
Modlin, *Basic Ratemaking* (5th ed., CAS, 2016). The full parity suite is
in ``validation/tests/pricing.rs``."""

import math

import pytest

from prospicio.pricing import Period, RateHistory, TrendFit, trend_factor, year_fraction


def test_year_fraction_half_month_grid():
    assert year_fraction(2013, 6, 30) == 2013.5
    assert year_fraction(2011, 11, 15) == 2011.875
    assert year_fraction(2015, 12, 31) == 2016.0
    with pytest.raises(ValueError):
        year_fraction(2011, 2, 29)


def test_trend_lengths():
    cy = Period.calendar(2011.0, 2012.0)
    assert cy.kind == "calendar"
    assert Period.policy(2013.0, 2014.0).average_written_date(1.0) - cy.average_written_date(1.0) == 2.5
    assert Period.policy(2015.0, 2016.0).average_earned_date(0.5) - cy.average_earned_date(0.5) == 4.25
    with pytest.raises(ValueError):
        Period.calendar(2012.0, 2011.0)


def test_parallelogram_table_5_19():
    h = RateHistory([2010.5, 2011.0, 2012.25], [0.05, 0.10, -0.01])
    level = h.rate_level(Period.calendar(2011.0, 2012.0), 1.0)
    assert level.shares == pytest.approx([0.125, 0.375, 0.5], abs=1e-14)
    assert level.written_changes == [0, 1, 2]
    assert level.average_level == pytest.approx(1.09625, abs=1e-12)
    assert level.on_level_factor == pytest.approx(1.14345 / 1.09625, abs=1e-12)
    six = h.rate_level(Period.calendar(2011.0, 2012.0), 0.5)
    assert six.shares == pytest.approx([0.25, 0.75], abs=1e-14)
    py = h.rate_level(Period.policy(2012.0, 2013.0), 1.0)
    assert round(py.on_level_factor, 4) == 0.9975


def test_law_change_table_5_23():
    h = RateHistory(
        [2010.5, 2011.0, 2011.5, 2012.25],
        [0.05, 0.10, -0.05, -0.01],
        applies=["written", "written", "in_force", "written"],
    )
    level = h.rate_level(Period.calendar(2011.0, 2012.0), 1.0)
    assert level.shares == pytest.approx([0.125, 0.25, 0.125, 0.125, 0.375], abs=1e-14)
    assert level.in_force_changes == [0, 0, 1, 0, 1]
    assert level.on_level_factor == pytest.approx(1.017084, abs=1e-6)
    with pytest.raises(ValueError):
        RateHistory([2011.0], [0.1], applies=["sometimes"])


def test_from_levels():
    h = RateHistory.from_levels([2011.25, 2012.5], [0.95, 1.045])
    assert h.current_level == pytest.approx(1.045, abs=1e-15)
    with pytest.raises(ValueError):
        RateHistory([2012.0, 2011.0], [0.1, 0.1])


def test_trend_fits():
    t = [i / 4 for i in range(8)]
    fit = TrendFit.exponential(t, [100 * 1.03**x for x in t])
    assert fit.kind == "exponential"
    assert fit.annual_change == pytest.approx(0.03, abs=1e-12)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.n == 8
    assert fit.fitted(1.0) == pytest.approx(103.0)
    line = TrendFit.linear(t, [5 + 2 * x for x in t])
    assert line.annual_change == pytest.approx(2.0)
    with pytest.raises(ValueError):
        TrendFit.exponential([0.0, 1.0], [1.0, -1.0])
    assert trend_factor(0.02, 2.5) == pytest.approx(math.pow(1.02, 2.5))


def test_loss_ratio_indication_appendix_a():
    from prospicio.pricing import (
        LossRatioExperience,
        Provisions,
        credibility_weighted,
        premium_based_expenses,
        trended_present_rates,
    )

    e = LossRatioExperience(
        [1_122_372, 1_154_508, 1_280_545, 1_369_976, 1_397_750],
        [856_495, 867_184, 835_120, 821_509, 797_866],
        on_level_factor=[1.2161, 1.2176, 1.1311, 1.0892, 1.0991],
        premium_trend=[1.1342, 1.1116, 1.0879, 1.0663, 1.0452],
        development=[1.0000, 0.9799, 1.0003, 1.0282, 1.0966],
        loss_trend=[0.9912, 0.9962, 1.0012, 1.0062, 1.0113],
        lae_factor=1.143,
    )
    assert [round(x) for x in e.projected_losses_by_year] == [970_359, 967_578, 955_974, 971_450, 1_011_357]
    assert round(e.loss_ratio, 3) == 0.619
    p = Provisions(0.17, 0.05)
    assert round(p.variable_permissible_loss_ratio, 3) == 0.78
    assert round(p.loss_ratio_change(0.619, 0.113), 3) == -0.062
    complement = trended_present_rates(0.132, 0.05, 1.005 / 1.02)
    assert round(complement, 3) == 0.062
    assert credibility_weighted(-0.06, 1.0, complement) == pytest.approx(-0.06)
    fixed, variable = premium_based_expenses([0.063, 0.085, 0.112], [0.75, 0.75, 0.0])
    assert fixed == pytest.approx(0.063 * 0.75 + 0.085 * 0.75)
    assert variable == pytest.approx(0.063 * 0.25 + 0.085 * 0.25 + 0.112)
    with pytest.raises(ValueError):
        LossRatioExperience([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        LossRatioExperience([1.0], [1.0], development=[1.0, 2.0])
    with pytest.raises(ValueError):
        Provisions(0.9, 0.2)


def test_pure_premium_indication_appendix_b():
    from prospicio.pricing import Provisions, PurePremiumExperience

    e = PurePremiumExperience(
        [12_760, 12_766, 12_805, 12_834, 13_411],
        [5_161_624, 4_820_968, 4_112_172, 5_052_052, 6_559_224],
        development=[1.0000, 1.0012, 1.0055, 1.0186, 1.0555],
        loss_trend=[1.1939, 1.1705, 1.1476, 1.1251, 1.1030],
        lae_factor=1.012,
    )
    assert round(e.projected_losses) == 30_343_206
    assert round(e.pure_premium, 2) == 469.88
    p = Provisions(0.138, 0.05)
    assert round(p.pure_premium_rate(493.63 + 29.11 + 74.57 + 15.68, 77.83), 2) == 850.76


AP_YEARS = list(range(2012, 2028))
AP_RATES = [r / 100 for r in [4.5, 4.0, 3.5, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 6.5, 5.5, 5.0, 4.5, 4.0, 4.0, 4.0]]
AP_INCEPTION = 2026.0027397260274


@pytest.mark.parametrize(
    "basis, kind, cutoff, want",
    [
        ("RAD", "loss", None, [1.965720460376, 1.549905768665, 1.060710030249]),
        ("RAD", "premium", None, [1.970440084121, 1.557339412121, 1.060710030249]),
        ("LOD", "loss", None, [1.927547497255, 1.519807645895, 1.040111758173]),
        ("LOD", "premium", None, [1.932175469096, 1.527096933018, 1.040111758173]),
        ("RAD", "loss", 6, [1.965720460376, 1.549905768665, 1.071161637407]),
        ("RAD", "premium", 6, [1.970440084121, 1.557339412121, 1.072446776656]),
        ("LOD", "loss", 9, [1.927547497255, 1.519807645895, 1.045223518981]),
        ("LOD", "premium", 9, [1.932175469096, 1.527096933018, 1.045850340865]),
    ],
)
def test_schedule_trend_reproduces_the_ap_vectors(basis, kind, cutoff, want):
    from prospicio.pricing import schedule_trend

    t = schedule_trend(list(range(2012, 2026)), AP_YEARS, AP_RATES, AP_INCEPTION,
                       basis=basis, kind=kind, cutoff=cutoff)
    for k, w in zip([0, 6, 13], want):
        assert t["factor"][k] == pytest.approx(w, rel=1e-12)


def test_schedule_trend_detail_extension_and_errors():
    from prospicio.pricing import schedule_trend

    t = schedule_trend([2012, 2025], AP_YEARS, AP_RATES, 2026.0)
    assert list(t) == ["year", "start", "target", "factor"]
    assert t["start"] == [2012.5, 2025.5] and t["target"] == [2027.0, 2027.0]
    assert t["factor"] == pytest.approx([1.965509247180, 1.060596058827], rel=1e-12)
    for kind in ["loss", "premium"]:
        e = schedule_trend([2008, 2010, 2025], [2012, None, 2013, 2014], [0.05, None, 0.06, 0.07],
                           2016.0027397260274, kind=kind)
        assert e["factor"] == pytest.approx([1.7281053820990948, 1.5093941672627256, 1.0], rel=1e-12)
    with pytest.raises(ValueError, match="not both"):
        schedule_trend([2012], [2012, 2013], [0.05, None], 2016.0)
    with pytest.raises(ValueError, match="cutoff"):
        schedule_trend([2012], [2012], [0.05], 2016.0, cutoff=13)
    with pytest.raises(ValueError, match="basis"):
        schedule_trend([2012], [2012], [0.05], 2016.0, basis="XYZ")


def test_cumulative_loss_ratio():
    from prospicio.pricing import cumulative_loss_ratio

    got = cumulative_loss_ratio([0.0, 0.6, None, 0.9, 0.3], [0.0, 100.0, None, 300.0, 600.0])
    assert got == pytest.approx([0.0, 0.6, 0.6, 0.825, 0.51], abs=1e-15)
    with pytest.raises(ValueError, match="not both"):
        cumulative_loss_ratio([0.5, 0.6], [1.0, None])
