"""The workers compensation split plan (#981), against NCCI's "ABCs of
Experience Rating" (Exhibits A, B and H)."""

import pytest

from prospicio.pricing import split_plan_modification


def test_the_worksheet_summary_gives_its_modification():
    x = split_plan_modification(
        ["all"], [100.0], [176_190.0], [56_172.0 / 176_190.0],
        [49_985.0, 17_359.0, 17_000.0, 16_225.0],
        weight=0.14, ballast=47_400.0, split_point=18_000.0,
        per_claim_limit=250_000.0,
    )
    assert x["actual_primary"] == pytest.approx(68_584.0)
    assert x["actual_excess"] == pytest.approx(31_985.0)
    assert round(x["adjusted_actual"]) == 223_677
    assert round(x["adjusted_expected"]) == 223_590
    assert round(x["modification"], 2) == 1.00


def test_claims_split_after_the_limit_and_medical_only_at_30_percent():
    x = split_plan_modification(
        ["8810"], [1_000_000.0], [1.0], [0.3],
        [500_000.0, 100_000.0, 5_000.0, 20_000.0],
        weight=0.2, ballast=10_000.0, split_point=18_000.0,
        medical_only=[False, False, False, True], per_claim_limit=250_000.0,
    )
    assert x["claims"]["limited"] == [250_000.0, 100_000.0, 5_000.0, 20_000.0]
    assert x["claims"]["primary"] == pytest.approx([18_000.0, 18_000.0, 5_000.0, 5_400.0])
    assert x["claims"]["excess"] == pytest.approx([232_000.0, 82_000.0, 0.0, 600.0])
    assert x["classes"]["expected"] == [10_000.0]


def test_an_accident_is_limited_together():
    x = split_plan_modification(
        ["8810"], [1_000_000.0], [1.0], [0.3], [200_000.0, 150_000.0],
        weight=0.2, ballast=0.0, split_point=18_000.0,
        accident=["a", "a"], per_accident_limit=300_000.0,
    )
    assert sum(x["claims"]["limited"]) == pytest.approx(300_000.0)
    assert x["claims"]["primary"] == [18_000.0, 18_000.0]


def test_bad_input_is_refused():
    with pytest.raises(ValueError, match="payroll has 2 entries"):
        split_plan_modification(["a"], [1.0, 2.0], [1.0], [0.3], [], 0.2, 0.0, 18_000.0)
    with pytest.raises(ValueError):
        split_plan_modification(["a"], [100.0], [1.0], [0.3], [], 1.5, 0.0, 18_000.0)
