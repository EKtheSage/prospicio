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
