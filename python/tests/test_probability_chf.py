"""invert_chf: a law's distribution from its characteristic function (#779),
against aggregate 2.1.0's AggregateCHF in validation/reference/chf_aggregate.csv,
the rows the Rust parity test reads."""

import cmath
import math

import pytest

from helpers import VALIDATION, read_csv
from prospicio.distributions import (
    Affine,
    Gamma,
    Grid,
    InverseGaussian,
    Normal,
    Tweedie,
    invert_chf,
)


def _law(kind, v):
    if kind == "gamma":
        return Gamma(v["shape"], v["scale"])
    if kind == "normal":
        return Normal(v["mu"], v["sigma"])
    if kind == "inverse_gaussian":
        return InverseGaussian(v["mean"], v["shape"])
    if kind == "tweedie":
        return Tweedie(v["mean"], v["dispersion"], v["power"])
    raise ValueError(kind)


def _params(text):
    return {k: float(x) for k, x in (pair.split("=") for pair in text.split(";"))}


def test_matches_aggregate():
    rows = read_csv(VALIDATION / "reference" / "chf_aggregate.csv")
    fits = {}
    failures = []
    for r in rows:
        key = (r["distribution"], r["params"])
        if key not in fits:
            v = _params(r["params"])
            kind = r["distribution"]
            if kind.startswith("affine_"):
                base = {k[5:]: x for k, x in v.items() if k.startswith("base_")}
                law = Affine(_law(kind[7:], base), v["scale"], v["shift"])
            else:
                law = _law(kind, v)
            fits[key] = invert_chf(law, log2=int(v["log2"]), step=v["bs"] or None)
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


def test_masses_are_bucket_probabilities():
    d = invert_chf(Gamma(4.0, 25.0), log2=12)
    g = Gamma(4.0, 25.0)
    h = d.step
    for k in (10, 400, 1600):
        x = d.xs[k]
        assert abs(d.masses[k] - (g.cdf(x + h / 2) - g.cdf(x - h / 2))) < 1e-12
    assert d.step == 0.25 and d.x_min == 0.0 and not d.is_truncated
    assert abs(d.law_mean - 100.0) < 1e-12 and abs(d.law_variance - 2500.0) < 1e-9


def test_own_characteristic_function_on_a_given_grid():
    def chf(ts):
        return [(1 - 25j * t) ** -4 for t in ts]

    d = invert_chf(chf, log2=12, step=0.25, x_min=0.0)
    ref = invert_chf(Gamma(4.0, 25.0), log2=12)
    assert max(abs(a - b) for a, b in zip(d.masses, ref.masses)) < 1e-14
    assert math.isnan(d.law_mean)
    with pytest.raises(ValueError, match="step and x_min"):
        invert_chf(chf, log2=12)


def test_errors_from_a_callable_reach_python():
    def bad(ts):
        raise RuntimeError("no chf here")

    with pytest.raises(ValueError, match="no chf here"):
        invert_chf(bad, log2=4, step=1.0, x_min=0.0)

    with pytest.raises(ValueError, match="not finite"):
        invert_chf(lambda ts: [cmath.nan] * len(ts), log2=4, step=1.0, x_min=0.0)


def test_normal_below_zero_and_to_grid():
    d = invert_chf(Normal(0.0, 1.0), log2=12)
    assert d.x_min < -7 and abs(d.quantile(0.5)) < 1e-12
    with pytest.raises(ValueError, match="start"):
        d.to_grid()
    g = invert_chf(Affine(Gamma(4.0, 25.0), 1.0, 1000.0), log2=12).to_grid()
    assert isinstance(g, Grid) and abs(g.mean() - 1100.0) < 1e-6


def test_rejects_other_laws_and_options():
    from prospicio.distributions import Lognormal

    with pytest.raises(ValueError, match="characteristic-function engine"):
        invert_chf(Lognormal(0.0, 1.0))
    with pytest.raises(TypeError):
        invert_chf(3.0)
    with pytest.raises(ValueError, match="log2"):
        invert_chf(Gamma(1.0, 1.0), log2=40)
    with pytest.raises(ValueError, match="give step too"):
        invert_chf(Gamma(1.0, 1.0), x_min=0.0)
