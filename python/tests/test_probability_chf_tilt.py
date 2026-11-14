"""invert_chf on stable laws with the tilt (#779, second PR), against
aggregate 2.1.0's AggregateCHF in validation/reference/chf_tilt_aggregate.csv,
the rows the Rust parity test reads."""

import math

import pytest

from helpers import VALIDATION, read_csv
from prospicio.distributions import Gamma, Stable, invert_chf


def _params(text):
    return {k: float(x) for k, x in (pair.split("=") for pair in text.split(";"))}


def _tilt(code):
    return True if code == -1 else False if code == -2 else code


def test_matches_aggregate():
    rows = read_csv(VALIDATION / "reference" / "chf_tilt_aggregate.csv")
    fits = {}
    failures = []
    for r in rows:
        if r["params"] not in fits:
            v = _params(r["params"])
            law = Stable(v["alpha"], v["beta"], v["scale"], v["loc"],
                         "S0" if v["parameterization"] == 1 else "S1")
            fits[r["params"]] = invert_chf(law, log2=int(v["log2"]), step=v["bs"] or None,
                                           tilt=_tilt(v["tilt"]))
        d = fits[r["params"]]
        q = r["quantity"]
        got = {
            "step": lambda: d.step,
            "x_min": lambda: d.x_min,
            "tilt_left": lambda: d.tilt[0],
            "tilt_right": lambda: d.tilt[1],
            "edge_error": lambda: d.edge_error,
            "truncation_error": lambda: d.truncation_error,
            "outside_below": lambda: d.outside_below,
            "outside_above": lambda: d.outside_above,
            "sum": lambda: d.total_mass,
            "quantile": lambda: d.quantile(float(r["arg"])),
            "mass": lambda: d.masses[int(r["arg"])],
        }[q]()
        want = float(r["expected"])
        tol = float(r["abs_tol"]) + float(r["rel_tol"]) * abs(want)
        if abs(got - want) > tol:
            failures.append(f"{r['params']} {q} {r['arg']}: {got!r} vs {want!r}")
    assert not failures, "\n".join(failures[:20])


def test_tilt_options():
    s = Stable(1.5, 1.0)
    auto = invert_chf(s, log2=12)
    off = invert_chf(s, log2=12, tilt=False)
    assert auto.tilt[1] > 0 and off.tilt == (0.0, 0.0)
    assert auto.edge_error < 1e-4 < off.edge_error
    assert invert_chf(s, log2=12, tilt=8).tilt == (0.0, 8.0)
    # tilt=1 is one e-fold, not True.
    assert invert_chf(s, log2=12, tilt=1).tilt == (0.0, 1.0)
    assert math.isnan(invert_chf(Gamma(2.0, 1.0), log2=10).edge_error)
    with pytest.raises(ValueError, match="power tail"):
        invert_chf(Gamma(2.0, 1.0), log2=10, tilt=4.0)
    # Two power tails: split, each half tilted toward its light side.
    split = invert_chf(Stable(1.5, 0.5), log2=10)
    assert split.tilt[0] > 0 and split.tilt[1] > 0
    with pytest.raises(TypeError):
        invert_chf(s, tilt="yes")


def test_stable_parameters():
    s = Stable(1.5, 1.0, 2.0)
    assert round(s.tail_constant, 6) == 1.128379 and s.left_tail_constant == 0.0
    assert abs(Stable(1.5, 1.0, 2.0, s.loc0, "s0").loc - s.loc) < 1e-12
    assert s.mean == 0.0 and Stable(0.7, 1.0).mean == math.inf
    with pytest.raises(ValueError, match="alpha"):
        Stable(2.5, 0.0)
    with pytest.raises(ValueError, match="parameterization"):
        Stable(1.5, 0.0, parameterization="S2")
