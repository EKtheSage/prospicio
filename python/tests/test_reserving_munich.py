"""The Munich chain ladder (Quarg and Mack 2004;
``docs/design/reserving-v02.md``, decision 33). The Rust core is checked
against R ChainLadder and chainladder-python in
``validation/tests/reserving_munich.rs``; these tests check the bindings'
arguments, results, frames and errors against the same references through
Python."""

import math

import pytest

from prospicio.reserving import (
    ChainLadder,
    MackFit,
    MunichChainLadder,
    MunichFit,
    Triangle,
)
from helpers import VALIDATION, read_csv


ROWS = read_csv(VALIDATION / "data" / "mcl.csv")


def mcl(incurred=None):
    paid = [float(r["paid"]) for r in ROWS]
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in ROWS],
        development=[int(r["development"]) for r in ROWS],
        values={
            "paid": paid,
            "incurred": incurred or [float(r["incurred"]) for r in ROWS],
        },
    )


def reference(name, test):
    return {
        (r["quantity"], r["arg"]): float(r["expected"])
        for r in read_csv(VALIDATION / "reference" / name)
        if r["test"] == test
    }


def test_matches_r_with_mack_sigma():
    # R ChainLadder's MunichChainLadder(MCLpaid, MCLincurred,
    # est.sigmaP = "Mack", est.sigmaI = "Mack").
    ref = reference("reserving_munich_r.csv", "mack")
    fit = MunichChainLadder(
        sigma_interpolation_paid="mack", sigma_interpolation_incurred="mack"
    ).fit(mcl(), "paid", "incurred")
    assert isinstance(fit, MunichFit)
    assert fit.lambda_paid == pytest.approx(ref[("lambda_paid", "")], rel=1e-12)
    assert fit.lambda_incurred == pytest.approx(ref[("lambda_incurred", "")], rel=1e-12)
    for o, year in enumerate(range(2001, 2008)):
        assert fit.ultimate_paid[o] == pytest.approx(ref[("ultimate_paid", str(year))], rel=1e-12)
        assert fit.ultimate_incurred[o] == pytest.approx(
            ref[("ultimate_incurred", str(year))], rel=1e-12
        )
    for k, age in enumerate(fit.development):
        assert fit.q[k] == pytest.approx(ref[("q", str(age))], rel=1e-12)
        assert fit.q_inverse[k] == pytest.approx(ref[("q_inverse", str(age))], rel=1e-12)
        assert fit.rho_paid[k] == pytest.approx(ref[("rho_paid", str(age))], rel=1e-12)
        assert fit.rho_incurred[k] == pytest.approx(ref[("rho_incurred", str(age))], rel=1e-12)
    assert (fit.n_residuals_paid, fit.n_residuals_incurred) == (20, 20)


def test_matches_r_with_tails():
    ref = reference("reserving_munich_r.csv", "mack;tail=1.05/1.02")
    fit = MunichChainLadder(
        sigma_interpolation_paid="mack",
        sigma_interpolation_incurred="mack",
        tail_paid=1.05,
        tail_incurred=1.02,
    ).fit(mcl(), "paid", "incurred")
    assert fit.total_ultimate_paid == pytest.approx(
        sum(v for (q, _), v in ref.items() if q == "ultimate_paid"), rel=1e-12
    )
    assert fit.total_ultimate_incurred == pytest.approx(
        sum(v for (q, _), v in ref.items() if q == "ultimate_incurred"), rel=1e-12
    )


def test_default_matches_chainladder_python():
    # chainladder-python's MunichAdjustment on Development() (log-linear).
    ref = reference("reserving_munich_python.csv", "log-linear")
    fit = MunichChainLadder().fit(mcl(), "paid", "incurred")
    for o, year in enumerate(range(2001, 2008)):
        assert fit.ultimate_paid[o] == pytest.approx(ref[("ultimate_paid", str(year))], rel=1e-12)


def test_frames_and_separate_fits():
    pd = pytest.importorskip("pandas")
    fit = MunichChainLadder(
        sigma_interpolation_paid="mack", sigma_interpolation_incurred="mack"
    ).fit(mcl(), "paid", "incurred")
    df = fit.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "origin",
        "latest_paid",
        "latest_incurred",
        "latest_ratio",
        "ultimate_paid",
        "ultimate_incurred",
        "ultimate_ratio",
        "reserve_paid",
        "ibnr_incurred",
        "chain_ladder_paid",
        "chain_ladder_incurred",
    ]
    assert df["ultimate_ratio"].iloc[-1] == pytest.approx(0.980301200225115, rel=1e-12)
    totals = fit.totals_frame()
    assert totals["ultimate_paid"].iloc[0] == pytest.approx(32121.4969949898, rel=1e-12)
    assert totals["lambda_paid"].iloc[0] == pytest.approx(fit.lambda_paid)
    dev = fit.development_frame()
    assert list(dev["development"]) == [12, 24, 36, 48, 60, 72, 84]
    assert math.isnan(dev["ldf_paid"].iloc[-1])
    assert isinstance(fit.mack_paid, MackFit)
    cl = ChainLadder().fit(mcl(), "paid")
    assert fit.chain_ladder_paid == pytest.approx(cl.ultimate, rel=1e-12)
    assert "MunichFit(origins=7" in repr(fit)
    assert repr(MunichChainLadder(tail_paid=1.05)).startswith(
        'MunichChainLadder(sigma_interpolation_paid="log-linear"'
    )


def test_segments():
    rows = ROWS + ROWS
    tri = Triangle.from_long(
        origin=[int(r["origin"]) for r in rows],
        development=[int(r["development"]) for r in rows],
        values={
            "paid": [float(r["paid"]) for r in rows],
            "incurred": [float(r["incurred"]) for r in rows],
        },
        keys={"lob": ["A"] * len(ROWS) + ["B"] * len(ROWS)},
    )
    fit = MunichChainLadder().fit(tri, "paid", "incurred")
    assert len(fit.ultimate_paid) == 14
    with pytest.raises(ValueError, match="segment"):
        fit.lambda_paid
    one = fit.segment(lob="B")
    assert one.lambda_paid == pytest.approx(fit.segment(lob="A").lambda_paid)


def test_no_information_is_the_separate_chain_ladders():
    incurred = [1.25 * float(r["paid"]) for r in ROWS]
    fit = MunichChainLadder().fit(mcl(incurred), "paid", "incurred")
    assert (fit.lambda_paid, fit.lambda_incurred) == (0.0, 0.0)
    assert fit.ultimate_paid == pytest.approx(fit.chain_ladder_paid, rel=1e-12)


def test_a_zero_sigma_leaves_only_its_own_regression():
    # Incurred flat from 60 months in 2001 and 2002, so incurred's sigma is
    # 0 at 60 to 72 months and its residuals there are 0/0. R ChainLadder
    # 0.2.21, MunichChainLadder(MCLpaid, I, est.sigmaP = "Mack",
    # est.sigmaI = "Mack") with I[1, 6] <- I[1, 5], I[2, 6] <- I[2, 5],
    # I[1, 7] <- I[1, 6], drops them from the incurred regression only.
    flat = {("2001", "72"): 2174.0, ("2001", "84"): 2174.0, ("2002", "72"): 2508.0}
    incurred = [flat.get((r["origin"], r["development"]), float(r["incurred"])) for r in ROWS]
    fit = MunichChainLadder(
        sigma_interpolation_paid="mack", sigma_interpolation_incurred="mack"
    ).fit(mcl(incurred), "paid", "incurred")
    assert (fit.n_residuals_paid, fit.n_residuals_incurred) == (20, 18)
    assert fit.lambda_paid == pytest.approx(0.636021466425142, rel=1e-12)
    assert fit.lambda_incurred == pytest.approx(0.413992067837447, rel=1e-12)
    totals = fit.totals_frame()
    assert totals["n_residuals_incurred"].tolist() == [18.0]

def test_missing_incurred_cell_is_refused():
    incurred = [float(r["incurred"]) for r in ROWS]
    row = next(
        i for i, r in enumerate(ROWS) if r["origin"] == "2004" and r["development"] == "48"
    )
    incurred[row] = math.nan
    with pytest.raises(ValueError, match="origin 2004 at age 48"):
        MunichChainLadder().fit(mcl(incurred), "paid", "incurred")


def test_bad_sigma_interpolation():
    with pytest.raises(ValueError):
        MunichChainLadder(sigma_interpolation_paid="cubic")
