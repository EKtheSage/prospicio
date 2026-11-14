"""invert_chf on stable laws with two power tails, split and tilted (#780), against
aggregate 2.1.0's AggregateCHF in validation/reference/chf_split_aggregate.csv,
the rows the Rust parity test reads."""

from helpers import VALIDATION, read_csv
from prospicio.distributions import Stable, invert_chf


def _params(text):
    return {k: float(x) for k, x in (pair.split("=") for pair in text.split(";"))}


def _tilt(code):
    return True if code == -1 else False if code == -2 else code


def test_matches_aggregate():
    rows = read_csv(VALIDATION / "reference" / "chf_split_aggregate.csv")
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
