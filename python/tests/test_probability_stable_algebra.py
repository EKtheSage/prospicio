"""Stable's algebra (#780), against aggregate 2.1.0's Stable in
validation/reference/stable_algebra_aggregate.csv, the rows the Rust parity
test reads."""

import pytest

from helpers import VALIDATION, read_csv
from prospicio.distributions import Stable


def _params(text):
    return {k: float(x) for k, x in (pair.split("=") for pair in text.split(";"))}


def _law(v, prefix):
    return Stable(v[f"{prefix}_alpha"], v[f"{prefix}_beta"], v[f"{prefix}_scale"],
                  v[f"{prefix}_loc"])


def _result(op, v):
    if op == "identity":
        return _law(v, "x")
    if op == "add":
        return _law(v, "x") + _law(v, "y")
    if op == "shift":
        return _law(v, "x") + v["c"]
    if op == "neg":
        return -_law(v, "x")
    if op == "scale":
        return v["c"] * _law(v, "x")
    if op == "copies":
        return v["c"] @ _law(v, "x")
    if op == "s0":
        return Stable(v["x_alpha"], v["x_beta"], v["x_scale"], v["x_loc"], "S0")
    return Stable.from_compound_pareto(v["frequency"], v["alpha"], v["theta"], v["sev_mean"])


def test_matches_aggregate():
    failures = []
    for r in read_csv(VALIDATION / "reference" / "stable_algebra_aggregate.csv"):
        st = _result(r["arg"], _params(r["params"]))
        got = getattr(st, r["quantity"])
        want = float(r["expected"])
        tol = float(r["abs_tol"]) + float(r["rel_tol"]) * abs(want)
        if not abs(got - want) <= tol:
            failures.append(f"{r['arg']} {r['params']} {r['quantity']}: {got!r} vs {want!r}")
    assert not failures, "\n".join(failures[:20])


def test_operators():
    x = Stable(1.5, 0.4, 2.0, 1.0)
    y = Stable(1.5, -0.8, 1.0, -3.0)
    d = x - y
    assert d.loc == pytest.approx(x.loc - y.loc)
    assert (1.0 - x).loc == pytest.approx(1.0 - x.loc)
    assert (x / 2.0).scale == pytest.approx(1.0)
    assert x.copies(2.0).scale == pytest.approx((x + x).scale)
    with pytest.raises(ValueError, match="alpha"):
        x + Stable(1.2, 0.0)
    with pytest.raises(ValueError):
        x / 0.0
