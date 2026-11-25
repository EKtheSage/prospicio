"""Earned exposure from policy dates (#981): Werner and Modlin's chapter 3
to 5 examples through the Python binding."""

import datetime as dt

import pytest

from prospicio.pricing import calendar_windows, earned_exposure, in_force

EFFECTIVE = ["2010-10-01", "2011-01-01", "2011-04-01", "2011-07-01", "2011-10-01", "2012-01-01"]
EXPIRATION = ["2011-10-01", "2012-01-01", "2012-04-01", "2012-07-01", "2012-10-01", "2013-01-01"]
ONES = [1.0] * 6
YEARS = calendar_windows(2010, 2012)


def test_calendar_windows():
    assert YEARS == ["2010-01-01", "2011-01-01", "2012-01-01", "2013-01-01"]
    assert len(calendar_windows(2024, 2024, by="quarter")) == 5
    assert len(calendar_windows(2024, 2024, by="month")) == 13
    with pytest.raises(ValueError, match="window length"):
        calendar_windows(2024, 2024, by="week")


def test_chapter_4_on_the_half_month_grid():
    # Tables 4.7 and 4.11.
    t = earned_exposure(EFFECTIVE, EXPIRATION, ONES, YEARS, method="half_month")
    assert list(t) == ["label", "from", "to", "written", "earned", "unearned", "in_force"]
    assert t["from"] == YEARS[:3] and t["to"] == YEARS[1:]
    assert t["written"] == [1.0, 4.0, 1.0]
    assert t["earned"] == pytest.approx([0.25, 3.25, 2.5], abs=1e-12)
    assert t["in_force"] == [1.0, 4.0, 1.0]


def test_exact_days_and_date_objects():
    # datetime.date and datetime.datetime are taken as their dates.
    effective = [dt.date.fromisoformat(d) for d in EFFECTIVE]
    expiration = [dt.datetime.fromisoformat(d + "T09:30:00") for d in EXPIRATION]
    windows = [dt.date(2010, 1, 1), dt.date(2011, 1, 1), dt.date(2012, 1, 1), dt.date(2013, 1, 1)]
    t = earned_exposure(effective, expiration, ONES, windows)
    assert t["earned"][0] == pytest.approx(92 / 365, abs=1e-15)
    assert t["earned"] == pytest.approx([0.25, 3.25, 2.5], abs=6e-3)


def test_policy_years_as_of_a_date_and_in_force():
    # Tables 4.9 and 4.13, as of 12/31/12, and Table 4.15.
    t = earned_exposure(EFFECTIVE, EXPIRATION, ONES, YEARS, kind="policy", valuation="2013-01-01")
    assert t["written"] == [1.0, 4.0, 1.0]
    assert t["earned"] == pytest.approx([1.0, 4.0, 1.0], abs=1e-12)
    snapshots = [in_force(EFFECTIVE, EXPIRATION, ONES, d) for d in ["2011-01-01", "2011-06-15", "2012-01-01"]]
    assert snapshots == [2.0, 3.0, 4.0]


def test_a_cancellation_and_a_mid_term_change():
    # Policy D cancelled on March 31, 2012: 400 in 2011 and -100 in 2012.
    cancel = [None, None, None, "2012-04-01", None, None]
    amounts = [0.0, 0.0, 0.0, 400.0, 0.0, 0.0]
    t = earned_exposure(EFFECTIVE, EXPIRATION, amounts, YEARS, method="half_month", cancellation=cancel)
    assert t["written"] == pytest.approx([0.0, 400.0, -100.0], abs=1e-12)
    assert t["earned"] == pytest.approx([0.0, 200.0, 100.0], abs=1e-12)
    by_policy_year = earned_exposure(EFFECTIVE, EXPIRATION, amounts, YEARS, kind="policy",
                                     method="half_month", cancellation=cancel)
    assert by_policy_year["written"] == pytest.approx([0.0, 300.0, 0.0], abs=1e-12)
    # Table 3.1's Policy C: 1,000, then 1,200 full term from January 1, 2011.
    change = [(0, dt.date(2011, 1, 1), 1200.0)]
    t = earned_exposure(["2010-07-01"], ["2011-07-01"], [1000.0], calendar_windows(2010, 2011),
                        method="half_month", changes=change)
    assert t["written"] == pytest.approx([1000.0, 100.0], abs=1e-12)
    assert t["earned"] == pytest.approx([500.0, 600.0], abs=1e-12)
    assert in_force(["2010-07-01"], ["2011-07-01"], [1000.0], "2010-12-31", changes=change) == 1000.0
    assert in_force(["2010-07-01"], ["2011-07-01"], [1000.0], "2011-01-01", changes=change) == 1200.0


def test_twenty_fourths_and_labels():
    # Table 4.24: 240 written in January and in December 2010.
    t = earned_exposure(["2010-01-09", "2010-12-28"], ["2011-01-09", "2011-12-28"], [240.0, 240.0],
                        calendar_windows(2010, 2011), method="twenty_fourths", label=["jan", "dec"])
    assert t["label"] == ["dec", "dec", "jan", "jan"]
    assert t["earned"] == pytest.approx([10.0, 230.0, 230.0, 10.0], abs=1e-10)


def test_refusals():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        earned_exposure(["2023-02-29"], ["2024-01-01"], [1.0], YEARS)
    with pytest.raises(ValueError):
        earned_exposure([20230101], ["2024-01-01"], [1.0], YEARS)
    with pytest.raises(ValueError, match="amount has 5 entries"):
        earned_exposure(EFFECTIVE, EXPIRATION, ONES[:5], YEARS)
    with pytest.raises(ValueError, match="method"):
        earned_exposure(EFFECTIVE, EXPIRATION, ONES, YEARS, method="weekly")
    with pytest.raises(ValueError, match="kind"):
        earned_exposure(EFFECTIVE, EXPIRATION, ONES, YEARS, kind="fiscal")
    with pytest.raises(ValueError, match="policy 7"):
        earned_exposure(EFFECTIVE, EXPIRATION, ONES, YEARS, changes=[(6, "2011-06-01", 2.0)])
