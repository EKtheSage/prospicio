"""Merz and Wuthrich's paid-incurred chain (``docs/design/reserving-v02.md``,
decision 32). The Rust core is checked against R ChainLadder's
``PaidIncurredChain`` in ``validation/tests/reserving_paid_incurred.rs``;
these tests check the binding's arguments, results, frames and errors
against the same R reference through Python."""

import csv
import math
from pathlib import Path

import pytest

from prospicio.reserving import PaidIncurredChain, PaidIncurredFit, Triangle

VALIDATION = Path(__file__).resolve().parents[2] / "validation"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def load(name):
    rows = read_csv(VALIDATION / "data" / f"{name}.csv")
    return Triangle.from_long(
        origin=[int(r["origin"]) for r in rows],
        development=[int(r["development"]) for r in rows],
        values={
            "paid": [float(r["paid"]) for r in rows],
            "incurred": [float(r["incurred"]) for r in rows],
        },
    )


R = {}
for row in read_csv(VALIDATION / "reference" / "reserving_paid_incurred_r.csv"):
    R[(row["dataset"], row["quantity"], row["arg"])] = float(row["expected"])


def close(a, b, rel=1e-9):
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)


@pytest.mark.parametrize("name", ["usaa", "dahms"])
def test_r_chainladder_convention_matches_r(name):
    fit = PaidIncurredChain(convention="r_chainladder").fit(load(name), "paid", "incurred")
    assert isinstance(fit, PaidIncurredFit)
    assert close(fit.total_reserve, R[(name, "total_reserve", "")])
    assert close(fit.total_std_err, R[(name, "total_std_err", "")])
    for o, origin in enumerate(fit.origins[1:], start=1):
        assert close(fit.ultimate[o], R[(name, "ultimate", origin)])
        assert close(fit.std_err[o], R[(name, "std_err", origin)])
    for j, s in enumerate(fit.sigma):
        assert close(s, R[(name, "sigma", str(j))])
    for j, t in enumerate(fit.tau):
        assert close(t, R[(name, "tau", str(j))])
        assert close(fit.psi[j], R[(name, "psi", str(j))])
        assert close(fit.psi_sd[j], R[(name, "psi_sd", str(j))])
    for j, p in enumerate(fit.phi):
        assert close(p, R[(name, "phi", str(j))])
        assert close(fit.phi_sd[j], R[(name, "phi_sd", str(j))])


def test_the_default_is_the_models_posterior():
    tri = load("usaa")
    fit = PaidIncurredChain().fit(tri, "paid", "incurred")
    r = PaidIncurredChain(convention="r_chainladder").fit(tri, "paid", "incurred")
    # About 0.5% above R's on USAA (reserving-v02.md decision 32).
    assert fit.total_reserve / r.total_reserve - 1 == pytest.approx(0.0050, abs=1e-4)
    assert fit.total_std_err / r.total_std_err - 1 == pytest.approx(0.0049, abs=1e-4)
    assert fit.sigma == r.sigma and fit.tau == r.tau
    assert fit.reserve[0] == 0.0 and fit.std_err[0] == 0.0
    assert fit.ultimate[0] == fit.latest[0]
    assert all(0 < b < 1 for b in fit.incurred_weight[1:])
    assert fit.total_ultimate == pytest.approx(sum(fit.ultimate))
    n = len(fit.development)
    cov = fit.posterior_covariance
    assert len(cov) == 2 * n - 1 and all(len(row) == 2 * n - 1 for row in cov)
    assert math.sqrt(cov[0][0]) == pytest.approx(fit.phi_sd[0])
    assert "total_reserve" in repr(fit)


def test_settings_reach_the_fit():
    tri = load("dahms")
    est = PaidIncurredChain().fit(tri, "paid", "incurred")
    given = PaidIncurredChain(sigma=est.sigma, tau=est.tau).fit(tri, "paid", "incurred")
    assert given.total_reserve == est.total_reserve
    mack = PaidIncurredChain(extrapolation="mack").fit(tri, "paid", "incurred")
    assert mack.sigma[9] == mack.sigma[7]
    vague = PaidIncurredChain(
        phi_mean=[0.0] * 10,
        phi_sd=[math.inf] * 10,
        psi_mean=[0.0] * 9,
        psi_sd=[math.inf] * 9,
    ).fit(tri, "paid", "incurred")
    assert vague.total_reserve == est.total_reserve
    tight = PaidIncurredChain(phi_mean=est.phi, phi_sd=[1e-9] * 10).fit(tri, "paid", "incurred")
    assert tight.phi == pytest.approx(est.phi, abs=1e-12)
    assert tight.total_std_err < est.total_std_err
    pic = PaidIncurredChain(extrapolation="mack", convention="r_chainladder")
    assert (pic.extrapolation, pic.convention, pic.sigma) == ("mack", "r_chainladder", None)
    assert repr(pic).startswith("PaidIncurredChain(sigma=None")


def test_frames():
    pd = pytest.importorskip("pandas")
    fit = PaidIncurredChain().fit(load("usaa"), "paid", "incurred")
    long = fit.to_frame()
    assert isinstance(long, pd.DataFrame)
    assert list(long.columns) == [
        "origin",
        "latest",
        "latest_incurred",
        "ultimate",
        "reserve",
        "incurred_weight",
        "std_err",
    ]
    assert long["reserve"].tolist() == fit.reserve
    totals = fit.totals_frame()
    assert totals["std_err"].tolist() == [fit.total_std_err]
    dev = fit.development_frame()
    assert dev["development"].tolist() == fit.development
    assert math.isnan(dev["tau"].iloc[-1])


def test_errors():
    tri = load("usaa")
    with pytest.raises(ValueError, match="extrapolation"):
        PaidIncurredChain(extrapolation="bondy")
    with pytest.raises(ValueError, match="convention"):
        PaidIncurredChain(convention="stan")
    with pytest.raises(ValueError, match="phi_mean and phi_sd"):
        PaidIncurredChain(phi_mean=[0.0] * 10)
    with pytest.raises(ValueError, match="sigma needs 10 values"):
        PaidIncurredChain(sigma=[0.1] * 9).fit(tri, "paid", "incurred")
    with pytest.raises(ValueError, match="paid-incurred chain"):
        PaidIncurredChain(tau=[-0.1] * 9).fit(tri, "paid", "incurred")
    with pytest.raises(ValueError):
        PaidIncurredChain().fit(tri, "paid", "missing")
