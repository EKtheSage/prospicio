# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy==2.5.4",
#     "scipy==1.18.1",
# ]
# [tool.uv]
# exclude-newer = "2026-11-24T00:00:00Z"
# ///
"""Regenerate validation/reference/reinsurer_default.csv and
reinsurer_default_solvency2.csv.

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

`reinsurer_default_solvency2.csv` holds the Solvency II standard formula's
counterparty default module for type 1 exposures (Delegated Regulation (EU)
2015/35, Arts 192, 199, 200 and 201), computed at the end of this script
from the articles' text, not from the Rust: `probability_by_step` (every
credit quality step), `probability_by_ratio` (the table's points, points
between them and both clamps, `arg` the ratio as a fraction),
`reinsurance_lgd` (the 50 % and 90 % shares, the factors 100 %, 50 % and
18 %, and a loss given default floored at 0; `params` the inputs) and
`type1` (`params` the exposures `pd_i` and `lgd_i`, `arg` the portfolio,
`quantity` one of `v_inter`, `v_intra`, `sigma`, `total_lgd` and `scr`; the
portfolios take all three of Art 200's branches, with repeated and zero
probabilities of default). These are exact arithmetic, so the tests hold
them to 1e-12 relative (`validation/tests/aggregate_reinsurer_default_solvency2.rs`).
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


# ---------------------------------------------------------------------------
# Solvency II: validation/reference/reinsurer_default_solvency2.csv
#
# The standard formula's counterparty default module for type 1 exposures,
# written here from the text of Delegated Regulation (EU) 2015/35
# (knowledge/references/solvency2-counterparty-default.md), not from the
# Rust: Art 192(2) (loss given default of a reinsurance arrangement), Art
# 199(2) and (3) (probability of default by credit quality step and by
# solvency ratio), Art 200 (SCR_def,1) and Art 201 (V_inter and V_intra).
# Every value is plain arithmetic on the inputs, so the tolerance is 1e-12
# relative; the values are written with 17 significant digits (`repr`) and
# the inputs exactly as the tests parse them back.

SII_OUT = OUT.with_name("reinsurer_default_solvency2.csv")
SII_SOURCE = "Delegated Regulation (EU) 2015/35"
SII_ABS_TOL, SII_REL_TOL = 0.0, 1e-12

# Art 199(2): credit quality steps 0 to 6.
SII_STEPS = [0.00002, 0.0001, 0.0005, 0.0024, 0.012, 0.042, 0.042]
# Art 199(3): (solvency ratio, probability of default), the ratio as a fraction.
SII_RATIO_TABLE = [
    (1.96, 0.0001),
    (1.75, 0.0005),
    (1.50, 0.001),
    (1.25, 0.002),
    (1.22, 0.0024),
    (1.00, 0.005),
    (0.95, 0.012),
    (0.75, 0.042),
]


def sii_by_ratio(ratio: float) -> float:
    """Art 199(3): linear in the ratio between the points of the table,
    0.01 % at or above 196 %, 4.2 % at or below 75 %."""
    table = sorted(SII_RATIO_TABLE)  # ascending in the ratio
    if ratio <= table[0][0]:
        return table[0][1]
    if ratio >= table[-1][0]:
        return table[-1][1]
    for (r0, p0), (r1, p1) in zip(table[:-1], table[1:]):
        if r0 <= ratio <= r1:
            if ratio == r0:
                return p0
            if ratio == r1:
                return p1
            return p0 + (p1 - p0) * (ratio - r0) / (r1 - r0)
    raise AssertionError(ratio)


def sii_lgd(recoverables, rm, collateral, share, factor) -> float:
    """Art 192(2): max(share (Recoverables + 50 % RM_re) - F Collateral, 0)."""
    return max(share * (recoverables + 0.5 * rm) - factor * collateral, 0.0)


def sii_type1(exposures):
    """Arts 200 and 201 on a list of (PD, LGD): exposures grouped by their
    probability of default, V_inter over every ordered pair of groups (the
    diagonal included; a term whose two PDs are both 0 is 0, Q&A 2530),
    V_intra over the groups, and SCR_def,1 by the size of sigma against
    the sum of the LGDs."""
    groups: dict[float, list[float]] = {}
    for pd, lgd in exposures:
        groups.setdefault(pd, []).append(lgd)
    v_inter = 0.0
    for pj, lj in groups.items():
        for pk, lk in groups.items():
            if pj == 0.0 and pk == 0.0:
                continue
            u = pj * (1 - pj) * pk * (1 - pk) / (1.25 * (pj + pk) - pj * pk)
            v_inter += u * sum(lj) * sum(lk)
    v_intra = 0.0
    for pj, lj in groups.items():
        v_intra += 1.5 * pj * (1 - pj) / (2.5 - pj) * sum(x * x for x in lj)
    sigma = math.sqrt(v_inter + v_intra)
    total = sum(x for _, x in exposures)
    if sigma <= 0.07 * total:
        scr, branch = 3 * sigma, "3sigma"
    elif sigma <= 0.2 * total:
        scr, branch = 5 * sigma, "5sigma"
    else:
        scr, branch = total, "total"
    values = {"v_inter": v_inter, "v_intra": v_intra, "sigma": sigma, "total_lgd": total, "scr": scr}
    return values, branch, (sigma / total if total > 0 else 0.0)


def sii_v_inter_by_exposure(exposures) -> float:
    """V_inter over every ordered pair of single exposures, no grouping: the
    same as by group, since each term is bilinear in the two LGDs."""
    v = 0.0
    for pi, li in exposures:
        for pk, lk in exposures:
            if pi == 0.0 and pk == 0.0:
                continue
            v += pi * (1 - pi) * pk * (1 - pk) / (1.25 * (pi + pk) - pi * pk) * li * lk
    return v


sii_rows = []


def sii_row(case, params_, quantity, arg, value):
    sii_rows.append(
        [case, params_, quantity, arg, repr(float(value)), f"{SII_ABS_TOL:g}", f"{SII_REL_TOL:g}", SII_SOURCE]
    )


for step, p in enumerate(SII_STEPS):
    sii_row("probability_by_step", "", "probability", str(step), p)

for r, p in SII_RATIO_TABLE:
    assert sii_by_ratio(r) == p
SII_RATIOS = [
    3.0, 1.96, 1.9, 1.855, 1.75, 1.6, 1.5, 1.3, 1.25, 1.234, 1.22, 1.1, 1.0, 0.975,
    0.95, 0.85, 0.75, 0.6, 0.0,
]
for ratio in SII_RATIOS:
    sii_row("probability_by_ratio", "", "probability", repr(ratio), sii_by_ratio(ratio))

# name: (recoverables, RM_re, collateral, share, factor)
SII_LGD_CASES = {
    "plain": (100.0, 40.0, 30.0, 0.5, 1.0),
    "factor_half": (100.0, 40.0, 30.0, 0.5, 0.5),
    "no_mitigation_no_collateral": (250.0, 0.0, 0.0, 0.5, 1.0),
    "collateral_above": (100.0, 40.0, 200.0, 0.5, 1.0),
    "collateralised_90": (100.0, 40.0, 30.0, 0.9, 1.0),
    "collateralised_90_f18": (100.0, 40.0, 30.0, 0.9, 0.18),
    "zero_recoverables": (0.0, 20.0, 0.0, 0.5, 1.0),
}
close(sii_lgd(*SII_LGD_CASES["plain"]), 30.0, 0.0)
close(sii_lgd(*SII_LGD_CASES["collateral_above"]), 0.0, 0.0)
for name, (rec, rm, col, share, factor) in SII_LGD_CASES.items():
    p = (
        f"recoverables={rec!r};risk_mitigation={rm!r};collateral={col!r};"
        f"lgd_share={share!r};factor={factor!r}"
    )
    sii_row("reinsurance_lgd", p, "lgd", name, sii_lgd(rec, rm, col, share, factor))

# name: [(PD, LGD), ...]
SII_PORTFOLIOS = {
    "single_3sigma": [(0.002, 100.0)],
    "single_5sigma": [(0.012, 50.0)],
    "single_total": [(0.042, 10.0)],
    "mixed_3sigma": [
        (0.0005, 120.0), (0.0005, 80.0), (0.001, 200.0), (0.0024, 60.0),
        (0.0, 300.0), (0.00002, 500.0), (0.0001, 150.0), (0.0005, 40.0),
    ],
    "mixed_5sigma": [(0.012, 40.0), (0.042, 30.0), (0.0024, 10.0), (0.012, 5.0), (0.0, 2.0)],
    "mixed_total": [(0.3, 20.0), (0.3, 5.0), (0.0, 1.0), (0.0, 2.0), (0.1, 3.0)],
    "with_certain_default": [(1.0, 50.0), (0.005, 80.0), (0.0024, 70.0), (0.0001, 400.0)],
    "zeros_only": [(0.0, 10.0), (0.0, 20.0)],
    "empty": [],
}
sii_branches = set()
for name, exposures in SII_PORTFOLIOS.items():
    values, branch, ratio = sii_type1(exposures)
    if exposures:
        # Clear of the 7 % and 20 % thresholds, so that a last-bit
        # difference cannot flip the branch.
        assert all(abs(ratio - t) > 1e-4 for t in (0.07, 0.2)), (name, ratio)
    sii_branches.add(branch)
    close(values["v_inter"], sii_v_inter_by_exposure(exposures), 1e-13)
    if len(exposures) == 1:
        # One exposure: V_inter + V_intra = PD (1 - PD) LGD^2.
        ((pd, lgd),) = exposures
        close(values["sigma"] ** 2, pd * (1 - pd) * lgd * lgd, 1e-13)
    # The single_ and mixed_ portfolios name the branch they take.
    assert not name.startswith(("single_", "mixed_")) or name.endswith(branch), (name, branch)
    params_ = ";".join(
        [f"n={len(exposures)}"]
        + [f"pd_{i}={pd!r};lgd_{i}={lgd!r}" for i, (pd, lgd) in enumerate(exposures, 1)]
    )
    for quantity in ("v_inter", "v_intra", "sigma", "total_lgd", "scr"):
        sii_row("type1", params_, quantity, name, values[quantity])
assert sii_branches == {"3sigma", "5sigma", "total"}, sii_branches

with open(SII_OUT, "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["distribution", "params", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
    w.writerows(sii_rows)
print(f"wrote {len(sii_rows)} rows to {SII_OUT}")
