"""The general liability experience ratio plan (#981), by hand: ISO
publishes no worked exhibit, so the worksheet's sums are taken one by one."""

import pytest

from prospicio.pricing import experience_ratio_modification


def test_the_worksheet_by_hand():
    x = experience_ratio_modification(
        [200_000.0, 220_000.0, 240_000.0], [0.10, 0.30, 0.60],
        [0, 0, 1], [90_000.0, 10_000.0, 30_000.0], [20_000.0, 1_000.0, 0.0],
        maximum_single_limit=100_000.0, expected_ratio=0.75, credibility=0.4,
        detrend_factor=[0.95, 1.0, 1.0],
    )
    # 110,000 held to the MSL of 100,000; 11,000; 30,000.
    assert x["years"]["reported"] == [111_000.0, 30_000.0, 0.0]
    unreported = [200_000 * 0.95 * 0.75 * 0.10, 220_000 * 0.75 * 0.30, 240_000 * 0.75 * 0.60]
    assert x["years"]["expected_unreported"] == pytest.approx(unreported)
    aer = (141_000.0 + sum(unreported)) / 660_000.0
    assert x["actual_ratio"] == pytest.approx(aer, rel=1e-14)
    assert x["modification"] == pytest.approx(1 + 0.4 * (aer - 0.75) / 0.75, rel=1e-14)
    assert x["credit_debit"] == pytest.approx(x["modification"] - 1, rel=1e-14)
    assert x["subject_loss_cost"] == 660_000.0


def test_no_credibility_gives_unity():
    x = experience_ratio_modification(
        [100_000.0], [0.4], [0], [500_000.0], [0.0],
        maximum_single_limit=float("inf"), expected_ratio=0.5, credibility=0.0,
    )
    assert x["modification"] == 1.0


def test_bad_inputs_are_errors():
    with pytest.raises(ValueError, match="year"):
        experience_ratio_modification(
            [100_000.0], [0.4], [1], [1.0], [0.0],
            maximum_single_limit=1e5, expected_ratio=0.5, credibility=0.5,
        )
    with pytest.raises(ValueError):
        experience_ratio_modification(
            [100_000.0], [0.4], [0], [1.0], [0.0],
            maximum_single_limit=1e5, expected_ratio=0.5, credibility=1.5,
        )
