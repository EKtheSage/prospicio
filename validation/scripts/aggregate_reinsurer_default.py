# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy==2.5.4",
#     "scipy==1.18.1",
# ]
# [tool.uv]
# exclude-newer = "2026-11-24T00:00:00Z"
# ///
"""Regenerate validation/reference/reinsurer_default.csv.

    uv run validation/scripts/aggregate_reinsurer_default.py

Closed forms for reinsurer default on a tower (#980,
`docs/design/reinsurer-default.md`, Validation), computed from the formulas
with no simulation, so nothing here mirrors the Rust. The annual gross loss
is lognormal, `G = exp(meanlog + sdlog x)` with `x` standard normal, and the
tower has two annual excess layers, `A` (20 xs 15) and `B` (40 xs 35), each
ceding `c(G) = min((G - a)^+, l)`. Reinsurer `r` writes lines `s_rA` and
`s_rB`, so its exposure is `E_r = s_rA c_A + s_rB c_B`; it defaults with
probability `p_r(x) = Phi((Phi^{-1}(p_r) + sqrt(rho_r) x) / sqrt(1 - rho_r))`
given the year's rank `Phi(x)` by the gross loss (Vasicek's one factor, the
worst years likeliest; `rho_r = 0` is independent default), and then the
cedant loses `(1 - R_r) (E_r - C_r)^+`. Every expectation is a
one-dimensional integral over `x` by `scipy.integrate.quad`, split at the
kinks (each layer's attachment and exhaustion, and where an exposure meets
its collateral).

Each row's `params` holds the case's whole set-up (`name=value` pairs that
the Rust test reads to build the same tower and panel), `distribution`
names the case, `quantity` and `arg` the value:

- `gross`: `gross_mean`, and `ceded_mean` of each layer.
- `a_independent`: one reinsurer, independent default, no collateral:
  `credit_loss_mean` and `credit_loss_var` (Bernoulli moments:
  `E[CL^2] = p (1 - R)^2 E[Y^2]`), `net_after_default_minus_net_mean` and
  `default_frequency`.
- `b_collateral`: the same with collateral.
- `c_two_reinsurers`: two reinsurers sharing both layers with collateral:
  `credit_loss_mean` by reinsurer and in total, `credit_loss_var` in total
  (the two defaults independent but their exposures sharing `G`, so the
  cross term `2 p_1 p_2 (1 - R_1) (1 - R_2) E[Y_1 Y_2]` enters),
  `credit_loss_mean_by_layer`
  (a reinsurer's credit loss split over its layers in proportion to its
  exposure on each), `net_after_default_minus_net_mean`.
- `d_factor`: one reinsurer on the factor, `rho = 0.5`, with collateral, on
  layer `B`: `credit_loss_mean`, `default_frequency` (which is `p`), and
  `default_given_top_1pct`, the probability of default in the years whose
  gross loss is above its 99th percentile.
- `d_independent`: the same reinsurer with `rho = 0`, for contrast.
- `e_joint`: two reinsurers on the factor: `joint_default_frequency`,
  `Phi_2(Phi^{-1}(p_1), Phi^{-1}(p_2); sqrt(rho_1 rho_2))` by
  `scipy.stats.multivariate_normal`, and each one's `default_frequency` and
  `default_given_top_1pct`.

`abs_tol` and `rel_tol` are the accuracy of the reference value itself (the
quadrature, about 1e-12 relative; the self-checks below hold it to 1e-10),
which `rerun_references.py` compares a rerun against. They are not the Rust
test's tolerance: the test simulates the years, so its error is Monte Carlo
noise, and it accepts a simulated value within four of its own standard
errors of `expected` (`validation/tests/aggregate_reinsurer_default.rs`).
The probabilities of default are 10 % to 25 % (the joint default about
4.6 %), so a few hundred thousand simulated years resolve them.
"""

import csv
import math
from pathlib import Path

import numpy as np
import scipy
from scipy import integrate
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

OUT = Path(__file__).resolve().parents[1] / "reference" / "reinsurer_default.csv"
SOURCE = f"scipy {scipy.__version__} quad"
ABS_TOL, REL_TOL = 1e-12, 1e-9

MEANLOG, SDLOG = 3.0, 0.8
# name: (limit, attachment)
LAYERS = {"A": (20.0, 15.0), "B": (40.0, 35.0)}
X_MAX = 12.0


def ceded(layer: str, g: float) -> float:
    limit, attach = LAYERS[layer]
    return min(max(g - attach, 0.0), limit)


def gross(x: float) -> float:
    return math.exp(MEANLOG + SDLOG * x)


def x_of(g: float) -> float:
    return (math.log(g) - MEANLOG) / SDLOG


class Re:
    def __init__(self, name, p, rec, col, rho, lines):
        self.name, self.p, self.rec, self.col, self.rho = name, p, rec, col, rho
        self.lines = lines  # layer -> line

    def exposure(self, g: float) -> float:
        return sum(s * ceded(l, g) for l, s in self.lines.items())

    def after(self, g: float) -> float:
        return max(self.exposure(g) - self.col, 0.0)

    def pd(self, x: float) -> float:
        if self.rho == 0.0:
            return self.p
        q = norm.ppf(self.p)
        return float(norm.cdf((q + math.sqrt(self.rho) * x) / math.sqrt(1.0 - self.rho)))


def kinks(panel) -> list[float]:
    """The x at every layer's attachment and exhaustion, and where each
    exposure meets its collateral."""
    gs = set()
    for limit, attach in LAYERS.values():
        gs.update([attach, attach + limit])
    top = max(a + l for l, a in LAYERS.values())
    for r in panel:
        if 0.0 < r.col < r.exposure(top):
            gs.add(brentq(lambda g: r.exposure(g) - r.col, 1e-9, top, xtol=1e-14, rtol=1e-15))
    return sorted(x_of(g) for g in gs if g > 0.0)


def expect(f, points, lo=-X_MAX) -> float:
    """E[f(x)] over x > lo, x standard normal, split at `points`, the normal
    cut at +-X_MAX, where its mass (1e-33) is far below the tolerance."""
    edges = [lo] + [p for p in points if p > lo] + [X_MAX]
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        v, _ = integrate.quad(
            lambda x: f(x) * norm.pdf(x), a, b, epsabs=0.0, epsrel=1e-13, limit=500
        )
        total += v
    return total


def lev(u: float) -> float:
    """E[min(G, u)] for the lognormal."""
    m, s = MEANLOG, SDLOG
    return math.exp(m + s * s / 2) * norm.cdf((math.log(u) - m - s * s) / s) + u * norm.sf(
        (math.log(u) - m) / s
    )


def params(panel) -> str:
    out = [f"meanlog={MEANLOG:g}", f"sdlog={SDLOG:g}"]
    for l, (limit, attach) in LAYERS.items():
        out += [f"limit_{l}={limit:g}", f"attach_{l}={attach:g}"]
    out.append(f"reinsurers={len(panel)}")
    for r in panel:
        n = r.name
        out += [f"p_{n}={r.p:g}", f"rec_{n}={r.rec:g}", f"col_{n}={r.col:g}", f"rho_{n}={r.rho:g}"]
        out += [f"line_{n}_{l}={r.lines.get(l, 0.0):g}" for l in LAYERS]
    return ";".join(out)


def fmt(v: float) -> str:
    return f"{v:.12g}"


rows = []


def row(case, panel, quantity, arg, value, source=SOURCE):
    rows.append([case, params(panel), quantity, arg, fmt(value), f"{ABS_TOL:g}", f"{REL_TOL:g}", source])


def close(a, b, tol=1e-10):
    assert abs(a - b) <= tol * max(1.0, abs(b)), (a, b)


def credit_mean(r, pts) -> float:
    return (1 - r.rec) * expect(lambda x: r.after(gross(x)) * r.pd(x), pts)


def by_layer(panel, layer, pts) -> float:
    def f(x):
        g, total = gross(x), 0.0
        for r in panel:
            e = r.exposure(g)
            if e > 0.0 and layer in r.lines:
                total += (1 - r.rec) * r.after(g) * r.lines[layer] * ceded(layer, g) / e * r.pd(x)
        return total

    return expect(f, pts)


def default_frequency(r, pts) -> float:
    v = expect(r.pd, pts)
    close(v, r.p, 1e-11)
    return v


def top_1pct(r) -> float:
    """P(default | G above its 99th percentile)."""
    x99 = norm.ppf(0.99)
    v = expect(r.pd, [], lo=x99) / 0.01
    # Default when sqrt(rho) z + sqrt(1 - rho) e < Phi^{-1}(p), with z = -x.
    joint = multivariate_normal(mean=[0, 0], cov=[[1, math.sqrt(r.rho)], [math.sqrt(r.rho), 1]])
    close(v, joint.cdf([norm.ppf(r.p), -x99]) / 0.01)
    return v


# gross
empty = []
pts = kinks(empty)
row("gross", empty, "gross_mean", "", math.exp(MEANLOG + SDLOG**2 / 2), "closed form")
for l, (limit, attach) in LAYERS.items():
    v = expect(lambda x, l=l: ceded(l, gross(x)), pts)
    close(v, lev(attach + limit) - lev(attach))
    row("gross", empty, "ceded_mean", l, v)

# a and b: one reinsurer on A, independent default, without and with collateral
for case, col in [("a_independent", 0.0), ("b_collateral", 6.0)]:
    r = Re("R1", 0.1, 0.4, col, 0.0, {"A": 1.0})
    panel = [r]
    pts = kinks(panel)
    y1 = expect(lambda x: r.after(gross(x)), pts)
    y2 = expect(lambda x: r.after(gross(x)) ** 2, pts)
    if col == 0.0:
        close(y1, lev(35.0) - lev(15.0))
    mean = r.p * (1 - r.rec) * y1
    close(mean, credit_mean(r, pts))
    var = r.p * (1 - r.rec) ** 2 * y2 - mean**2
    row(case, panel, "credit_loss_mean", "R1", mean)
    row(case, panel, "credit_loss_var", "R1", var)
    row(case, panel, "net_after_default_minus_net_mean", "", mean)
    row(case, panel, "default_frequency", "R1", default_frequency(r, pts))

# c: two reinsurers sharing both layers, with collateral
panel = [
    Re("R1", 0.15, 0.3, 8.0, 0.0, {"A": 0.6, "B": 0.3}),
    Re("R2", 0.25, 0.5, 4.0, 0.0, {"A": 0.4, "B": 0.7}),
]
pts = kinks(panel)
means = [credit_mean(r, pts) for r in panel]
for r, m in zip(panel, means):
    row("c_two_reinsurers", panel, "credit_loss_mean", r.name, m)
row("c_two_reinsurers", panel, "credit_loss_mean", "total", sum(means))
# Var of the total: E[CL^2] = sum_r p_r (1 - R_r)^2 E[Y_r^2]
# + 2 p_1 p_2 (1 - R_1)(1 - R_2) E[Y_1 Y_2], the defaults independent.
r1, r2 = panel
k1, k2 = 1 - r1.rec, 1 - r2.rec
ey1 = expect(lambda x: r1.after(gross(x)), pts)
ey2 = expect(lambda x: r2.after(gross(x)), pts)
close(r1.p * k1 * ey1, means[0])
close(r2.p * k2 * ey2, means[1])
second = (
    r1.p * k1**2 * expect(lambda x: r1.after(gross(x)) ** 2, pts)
    + r2.p * k2**2 * expect(lambda x: r2.after(gross(x)) ** 2, pts)
    + 2 * r1.p * r2.p * k1 * k2 * expect(lambda x: r1.after(gross(x)) * r2.after(gross(x)), pts)
)
# The same second moment from the joint default probabilities by year.
close(
    second,
    expect(
        lambda x: (1 - r1.pd(x)) * r2.pd(x) * (k2 * r2.after(gross(x))) ** 2
        + r1.pd(x) * (1 - r2.pd(x)) * (k1 * r1.after(gross(x))) ** 2
        + r1.pd(x) * r2.pd(x) * (k1 * r1.after(gross(x)) + k2 * r2.after(gross(x))) ** 2,
        pts,
    ),
)
row("c_two_reinsurers", panel, "credit_loss_var", "total", second - sum(means) ** 2)
layers = [by_layer(panel, l, pts) for l in LAYERS]
close(sum(layers), sum(means))
for l, v in zip(LAYERS, layers):
    row("c_two_reinsurers", panel, "credit_loss_mean_by_layer", l, v)
row("c_two_reinsurers", panel, "net_after_default_minus_net_mean", "", sum(means))
for r in panel:
    row("c_two_reinsurers", panel, "default_frequency", r.name, default_frequency(r, pts))

# d: one reinsurer on B with the factor, and the same without it
for case, rho in [("d_factor", 0.5), ("d_independent", 0.0)]:
    r = Re("R1", 0.1, 0.4, 5.0, rho, {"B": 1.0})
    panel = [r]
    pts = kinks(panel)
    row(case, panel, "credit_loss_mean", "R1", credit_mean(r, pts))
    row(case, panel, "default_frequency", "R1", default_frequency(r, pts))
    if rho > 0.0:
        row(case, panel, "default_given_top_1pct", "R1", top_1pct(r))

# e: two reinsurers on the factor, defaulting together
panel = [
    Re("R1", 0.1, 0.4, 0.0, 0.3, {"A": 0.5}),
    Re("R2", 0.2, 0.4, 0.0, 0.6, {"A": 0.5}),
]
pts = kinks(panel)
r1, r2 = panel
c = math.sqrt(r1.rho * r2.rho)
joint = float(
    multivariate_normal(mean=[0, 0], cov=[[1, c], [c, 1]]).cdf([norm.ppf(r1.p), norm.ppf(r2.p)])
)
close(joint, expect(lambda x: r1.pd(x) * r2.pd(x), pts))
row(
    "e_joint", panel, "joint_default_frequency", "R1+R2", joint,
    f"scipy {scipy.__version__} multivariate_normal",
)
for r in panel:
    row("e_joint", panel, "default_frequency", r.name, default_frequency(r, pts))
    row("e_joint", panel, "default_given_top_1pct", r.name, top_1pct(r))

assert np.all(np.isfinite([float(r[4]) for r in rows]))
with open(OUT, "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["distribution", "params", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
    w.writerows(rows)
print(f"wrote {len(rows)} rows to {OUT}")
