"""invert_chf on the Tweedie law at any power (#780), against aggregate
2.1.0's AggregateCHF in validation/reference/chf_tweedie_aggregate.csv, the
rows the Rust parity test reads."""

import pytest

from helpers import VALIDATION, read_csv
from prospicio.distributions import TweediePower, invert_chf


def _params(text):
    return {k: float(x) for k, x in (pair.split("=") for pair in text.split(";"))}


def test_matches_aggregate():
    rows = read_csv(VALIDATION / "reference" / "chf_tweedie_aggregate.csv")
    fits = {}
    failures = []
    for r in rows:
        if r["params"] not in fits:
            v = _params(r["params"])
            law = TweediePower(v["power"], v["mean"], v["dispersion"])
            fits[r["params"]] = invert_chf(law, log2=int(v["log2"]))
        d = fits[r["params"]]
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
            failures.append(f"{r['params']} {q} {r['arg']}: {got!r} vs {want!r}")
    assert not failures, "\n".join(failures[:20])


def test_parameters():
    t = TweediePower(3.0, 10.0, 0.5)
    assert t.variance == 500.0 and abs(t.skewness - 3.0 * 500.0**0.5 / 10.0) < 1e-12
    cv = TweediePower.from_mean_cv(2.5, 10.0, 0.3)
    assert abs(cv.variance**0.5 / 10.0 - 0.3) < 1e-12
    for p in (0.5, 1.0, float("inf")):
        with pytest.raises(ValueError, match="power"):
            TweediePower(p, 1.0, 1.0)
