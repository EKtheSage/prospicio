"""invert_chf on the normal inverse Gaussian, variance gamma and CGMY
tempered stable laws (#780), against aggregate 2.1.0's AggregateCHF in
validation/reference/chf_levy_aggregate.csv, the rows the Rust parity test
reads."""

import pytest

from helpers import VALIDATION, read_csv
from prospicio.distributions import (
    NormalInverseGaussian,
    TemperedStable,
    VarianceGamma,
    invert_chf,
)


def _params(text):
    return {k: float(x) for k, x in (pair.split("=") for pair in text.split(";"))}


def _law(kind, v):
    if kind == "normal_inverse_gaussian":
        return NormalInverseGaussian(v["alpha"], v["beta"], v["delta"], v["mu"])
    if kind == "variance_gamma":
        return VarianceGamma(v["sigma"], v["nu"], v["theta"], v["mu"])
    return TemperedStable(v["C"], v["G"], v["M"], v["Y"], v["mu"])


def test_matches_aggregate():
    rows = read_csv(VALIDATION / "reference" / "chf_levy_aggregate.csv")
    fits = {}
    failures = []
    for r in rows:
        key = (r["distribution"], r["params"])
        if key not in fits:
            v = _params(r["params"])
            fits[key] = invert_chf(_law(r["distribution"], v), log2=int(v["log2"]))
        d = fits[key]
        q = r["quantity"]
        got = {
            "step": lambda: d.step,
            "x_min": lambda: d.x_min,
            "truncation_error": lambda: d.truncation_error,
            "outside_below": lambda: d.outside_below,
            "outside_above": lambda: d.outside_above,
            "sum": lambda: d.total_mass,
            "mean": lambda: d.mean(),
            "sd": lambda: d.std_dev(),
            "quantile": lambda: d.quantile(float(r["arg"])),
            "mass": lambda: d.masses[int(r["arg"])],
        }[q]()
        want = float(r["expected"])
        tol = float(r["abs_tol"]) + float(r["rel_tol"]) * abs(want)
        if abs(got - want) > tol:
            failures.append(f"{key} {q} {r['arg']}: {got!r} vs {want!r}")
    assert not failures, "\n".join(failures[:20])


def test_parameters_and_moments():
    nig = NormalInverseGaussian(2.0, 0.5, 1.5, 0.3)
    g = (4.0 - 0.25) ** 0.5
    assert abs(nig.mean - (0.3 + 1.5 * 0.5 / g)) < 1e-15
    assert abs(nig.variance - 1.5 * 4.0 / g**3) < 1e-15
    vg = VarianceGamma(0.2, 0.5, -0.1)
    assert abs(vg.variance - 0.045) < 1e-15 and vg.mu == 0.0
    ts = TemperedStable(1.0, 5.0, 10.0, 0.5)
    assert (ts.C, ts.G, ts.M, ts.Y) == (1.0, 5.0, 10.0, 0.5)
    with pytest.raises(ValueError, match="beta"):
        NormalInverseGaussian(1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="nu"):
        VarianceGamma(1.0, 0.0)
    with pytest.raises(ValueError, match="Y"):
        TemperedStable(1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="power tail"):
        invert_chf(vg, tilt=3.0)
