"""The Solvency II counterparty default module for type 1 exposures
(prospicio.reinsurance.solvency2_*, #980) against
validation/reference/reinsurer_default_solvency2.csv, which
validation/scripts/aggregate_reinsurer_default.py computes from the text of
Delegated Regulation (EU) 2015/35, Arts 192, 199, 200 and 201."""

import math

import pytest
from helpers import VALIDATION, read_csv

from prospicio.reinsurance import (
    Solvency2Type1,
    solvency2_probability_by_ratio,
    solvency2_probability_by_step,
    solvency2_reinsurance_lgd,
    solvency2_type1,
)

ROWS = read_csv(VALIDATION / "reference" / "reinsurer_default_solvency2.csv")


def params(row):
    return {k: float(v) for k, v in (p.split("=") for p in row["params"].split(";"))}


def exposures(row):
    p = params(row)
    return [(p[f"pd_{i}"], p[f"lgd_{i}"]) for i in range(1, int(p["n"]) + 1)]


def evaluate(row):
    kind, quantity, arg = row["distribution"], row["quantity"], row["arg"]
    if (kind, quantity) == ("probability_by_step", "probability"):
        return solvency2_probability_by_step(int(arg))
    if (kind, quantity) == ("probability_by_ratio", "probability"):
        return solvency2_probability_by_ratio(float(arg))
    if (kind, quantity) == ("reinsurance_lgd", "lgd"):
        p = params(row)
        return solvency2_reinsurance_lgd(
            p["recoverables"], p["risk_mitigation"], p["collateral"],
            lgd_share=p["lgd_share"], factor=p["factor"],
        )
    if kind == "type1" and quantity in ("v_inter", "v_intra", "sigma", "total_lgd", "scr"):
        return getattr(solvency2_type1(exposures(row)), quantity)
    raise LookupError(f"no evaluation for {kind} {quantity}")


def test_every_row_is_read():
    assert len(ROWS) == 78


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r['distribution']}-{r['arg']}-{r['quantity']}")
def test_matches_the_articles(row):
    want = float(row["expected"])
    got = evaluate(row)
    tol = max(float(row["abs_tol"]), float(row["rel_tol"]) * abs(want))
    assert got == want or abs(got - want) <= tol, (got, want)


def test_type1_result():
    t = solvency2_type1([(0.012, 50.0)])
    assert isinstance(t, Solvency2Type1)
    assert math.isclose(t.v_inter + t.v_intra, 0.012 * 0.988 * 2500.0, rel_tol=1e-12)
    assert t.scr == 5.0 * t.sigma
    assert t.total_lgd == 50.0
    assert "Solvency2Type1" in repr(t)
    assert solvency2_type1([]).scr == 0.0


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        solvency2_probability_by_step(7)
    with pytest.raises(OverflowError):
        solvency2_probability_by_step(-1)
    with pytest.raises(ValueError):
        solvency2_probability_by_ratio(float("nan"))
    with pytest.raises(ValueError):
        solvency2_probability_by_ratio(-0.5)
    with pytest.raises(ValueError):
        solvency2_reinsurance_lgd(100.0, 0.0, -1.0)
    with pytest.raises(ValueError):
        solvency2_reinsurance_lgd(100.0, 0.0, 0.0, lgd_share=1.5)
    with pytest.raises(ValueError):
        solvency2_reinsurance_lgd(100.0, 0.0, 0.0, factor=2.0)
    with pytest.raises(ValueError):
        solvency2_type1([(1.5, 10.0)])
    with pytest.raises(ValueError):
        solvency2_type1([(0.01, -10.0)])
