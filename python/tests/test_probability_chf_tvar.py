"""ChfDistribution.tvar (#779, #780) against aggregate 2.1.0's
AggregateCHF.tvar in validation/reference/chf_tvar_aggregate.csv, the rows
the Rust parity test reads."""

import pytest

from helpers import VALIDATION, read_csv
from prospicio.distributions import (
    Gamma,
    InverseGaussian,
    Normal,
    NormalInverseGaussian,
    Stable,
    TemperedStable,
    TweediePower,
    VarianceGamma,
    invert_chf,
)


def _params(text):
    return {k: float(x) for k, x in (pair.split("=") for pair in text.split(";"))}


def _law(kind, v):
    return {
        "gamma": lambda: Gamma(v["shape"], v["scale"]),
        "normal": lambda: Normal(v["mu"], v["sigma"]),
        "inverse_gaussian": lambda: InverseGaussian(v["mean"], v["shape"]),
        "tweedie_power": lambda: TweediePower(v["power"], v["mean"], v["dispersion"]),
        "normal_inverse_gaussian": lambda: NormalInverseGaussian(
            v["alpha"], v["beta"], v["delta"], v["mu"]),
        "variance_gamma": lambda: VarianceGamma(v["sigma"], v["nu"], v["theta"], v["mu"]),
        "tempered_stable": lambda: TemperedStable(v["C"], v["G"], v["M"], v["Y"], v["mu"]),
        "stable": lambda: Stable(v["alpha"], v["beta"], v["scale"], v["loc"],
                                 "S0" if v["parameterization"] == 1 else "S1"),
    }[kind]()


def test_matches_aggregate():
    rows = read_csv(VALIDATION / "reference" / "chf_tvar_aggregate.csv")
    fits = {}
    failures = []
    for r in rows:
        key = (r["distribution"], r["params"])
        if key not in fits:
            v = _params(r["params"])
            fits[key] = invert_chf(_law(r["distribution"], v), log2=int(v["log2"]))
        got = fits[key].tvar(float(r["arg"]))
        want = float(r["expected"])
        tol = max(float(r["abs_tol"]), float(r["rel_tol"]) * abs(want))
        if abs(got - want) > tol:
            failures.append(f"{key} {r['arg']}: {got!r} vs {want!r}")
    assert len(rows) == 84
    assert not failures, "\n".join(failures)


def test_tvar_is_at_least_the_var_and_refuses_bad_levels():
    d = invert_chf(Gamma(4.0, 25.0), log2=12)
    for p in (0.0, 0.5, 0.99):
        assert d.tvar(p) >= d.quantile(p)
    assert d.tvar(1.0) == max(x for x, m in zip(d.xs, d.masses) if m > 0)
    with pytest.raises(ValueError):
        d.tvar(1.5)
