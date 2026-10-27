"""Design study for issue #646: reading an annual accident-year pattern at
monthly ages before `SelectedPattern::to_underwriting_basis` (decision 21).

An accident period of `s` months reports, at age `a`, the even-exposure
average of its months, `P_s(a) = (1/s) sum_{q<s} p(a - q)` with `p` the
accident-month pattern and `p(t) = 0` for `t <= 0` (decision 18 item 1).
Decision 21 converts a pattern of `s`-month periods as
`P_UY(a) = sum_j v(j) P_s(a - j)` and reads `P_s` between its given ages by
decision 18 item 2 (linear from 0 below the first age, ln-linear between
ages). Each candidate here expands the given annual values to `P_s` at
every month; the exact conversion then follows. Candidates:

  baseline   decision 18 item 2 on `P_s` (what the crate does today)
  pchip      shape-preserving cubic (PCHIP) through (0, 0) and the given
             points, on `P_s` itself
  linknot    exact: `p` piecewise linear on the given ages, `p(0) = 0`,
             the forward recursion on the averages
  lnknot     exact: `p` ln-linear between the given ages, the first year a
             geometric run-in at the second year's rate, roots by bisection
  monoqp     exact: monotone `p` on monthly ages, averages matched, least
             sum of squared second differences of `p` (a quadratic
             programme, SLSQP)
  llog, wbl  Clark's loglogistic and Weibull `p(t) = L G(t) / G(E)` with
             (omega, theta) fitted so that the 12-month averages of `G`
             match the given values in least squares (decision 23's scale)
  llog+exact, wbl+exact
             the curve's increments moved as little as possible (least
             squares) to a monotone `p` whose averages are the given values
             exactly; increments are free only within a span whose given
             value is below the one before it (downward development)
  wbl+scale  closed form: the Weibull's increments rescaled span by span so
             that each span's average is the given value, capped at the
             level (inexact where the cap or a scale below 0 binds)
  llog-mid   the loglogistic fitted to the given values directly at the
             mid-year age `a - 6` (`BenchmarkCurve` with `origin_width = 12`)

Run with the Python venv: it needs numpy and scipy. Prints the tables of
the finding `knowledge/findings/ay-pattern-monthly-reading.md`.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, minimize

ROOT = Path(__file__).resolve().parents[2]


# --- the crate's conventions -------------------------------------------


def percent_at(points, age):
    """Decision 18 item 2: 0 at or below 0, linear from 0 to the first
    point, ln-linear between points, held past the last."""
    ages = [a for a, _ in points]
    vals = [v for _, v in points]
    if age <= 0:
        return 0.0
    if age < ages[0]:
        return vals[0] * age / ages[0]
    if age >= ages[-1]:
        return vals[-1]
    i = int(np.searchsorted(ages, age, side="right")) - 1
    if ages[i] == age:
        return vals[i]
    b, c = ages[i], ages[i + 1]
    share = (age - b) / (c - b)
    return float(np.exp(np.log(vals[i]) + share * (np.log(vals[i + 1]) - np.log(vals[i]))))


def earning(m):
    k = np.full(m + 1, 1.0 / m)
    k[0] = k[-1] = 0.5 / m
    return k


def exposure(writing, m):
    """`UnderwritingBasis::exposure`: writing convolved with the earning."""
    w = np.asarray(writing, float)
    w = w / w.sum()
    return np.convolve(w, earning(m))


def weights_over(writing, m, s):
    """`UnderwritingBasis::weights_over` for a term that is a whole number
    of `s`-month periods."""
    w = np.asarray(writing, float)
    w = w / w.sum()
    if m % s == 0:
        r = m // s
        comb = np.zeros((r - 1) * s + 1)
        comb[::s] = 1.0 / r
        return np.convolve(np.convolve(w, [0.5, 0.5]), comb)
    # writing even within each period of s months: u at the period's first
    # month, convolved with a policy's earning
    blocks = -(-len(w) // s)
    u = np.zeros((blocks - 1) * s + 1)
    for b in range(blocks):
        block = [w[b * s + q] if b * s + q < len(w) else 0.0 for q in range(s)]
        assert max(abs(x - np.mean(block)) for x in block) < 1e-12
        u[b * s] = sum(block)
    return np.convolve(u, earning(m))


def convert(points, v):
    """`to_underwriting_basis` on a pattern of `s`-month periods given at
    `points`, `v` its weights: percents at every month from the first age
    to the end plus `len(v) - 1`, the last the level itself."""
    first, end = points[0][0], points[-1][0]
    last = end + len(v) - 1
    out = {}
    for a in range(first, last + 1):
        if a == last:
            out[a] = points[-1][1]
        else:
            out[a] = sum(vj * percent_at(points, a - j) for j, vj in enumerate(v))
    return out


def average(p, s, ages):
    """`P_s` at `ages` from a monthly `p` (a callable, 0 at or below 0)."""
    return [(a, sum(p(a - q) for q in range(s)) / s) for a in ages]


# --- candidates: each returns P_s at every month 1..E ------------------


def expand_baseline(points, s):
    E = points[-1][0]
    return [(a, percent_at(points, a)) for a in range(1, E + 1)]


def expand_pchip(points, s):
    E = points[-1][0]
    f = PchipInterpolator([0] + [a for a, _ in points], [0.0] + [v for _, v in points])
    return [(a, float(f(a))) for a in range(1, E + 1)]


def monthly_from_knots(knots, E, log):
    """`p` at every month from knot values at the given ages: linear (or
    ln-linear) between knots, the first span a run-in from 0 (linear) or at
    the next span's rate (ln-linear), held past the last knot."""
    ages = [0] + [a for a, _ in knots]
    vals = [0.0] + [v for _, v in knots]
    p = np.zeros(E + 1)
    for t in range(1, E + 1):
        if t >= ages[-1]:
            p[t] = vals[-1]
            continue
        i = int(np.searchsorted(ages, t, side="right")) - 1
        b, c, vb, vc = ages[i], ages[i + 1], vals[i], vals[i + 1]
        if not log:
            p[t] = vb + (t - b) / (c - b) * (vc - vb)
        elif i > 0:
            p[t] = np.exp(np.log(vb) + (t - b) / (c - b) * (np.log(vc) - np.log(vb)))
        else:
            # the first span: geometric at the second span's monthly rate
            # (flat if there is one knot or the next knot is equal)
            if len(vals) > 2 and vals[2] > vals[1]:
                g = (vals[2] / vals[1]) ** (1.0 / (ages[2] - ages[1]))
            else:
                g = 1.0
            p[t] = vals[1] * g ** (t - c)
    return p


def avg_of(p, s, a):
    return sum(p[a - q] if a - q > 0 else 0.0 for q in range(s)) / s


def expand_knots(points, s, log):
    """Exact on the given ages: knot values solved so that the `s`-month
    averages at the given ages equal the given values. Linear knots close
    forward one at a time; ln-linear knots need the first two together."""
    E = points[-1][0]
    ages = [a for a, _ in points]
    vals = [v for _, v in points]
    knots = []

    def avg_at(kn, a):
        return avg_of(monthly_from_knots(kn, E, log), s, a)

    for k, (a, target) in enumerate(points):
        if log and k == 0 and len(points) > 1:
            def y2_of(y1):
                def f(y2):
                    return avg_at([(ages[0], y1), (ages[1], y2)], ages[1]) - vals[1]
                return brentq(f, y1, max(y1 * 50, 50.0)) if f(y1) < 0 else y1

            def f1(y1):
                return avg_at([(ages[0], y1), (ages[1], y2_of(y1))], ages[0]) - target

            y1 = brentq(f1, 1e-9, 50.0)
            knots = [(ages[0], y1), (ages[1], y2_of(y1))]
            continue
        if log and k == 1:
            continue

        def f(y):
            return avg_at(knots + [(a, y)], a) - target

        y = brentq(f, -50.0, 50.0) if not log else brentq(f, 1e-12, 50.0)
        knots.append((a, y))
    p = monthly_from_knots(knots, E, log)
    return [(a, avg_of(p, s, a)) for a in range(1, E + 1)]


def constraint_matrix(points, s):
    """`A d = b` for the averages, with `d` the monthly increments of `p`
    on months 1..E and `p = C d`."""
    E = points[-1][0]
    C = np.tril(np.ones((E, E)))
    A = []
    for a, _ in points:
        row = np.zeros(E)
        for q in range(s):
            t = a - q
            if t >= 1:
                row += C[t - 1]
        A.append(row / s)
    return C, np.array(A), np.array([v for _, v in points])


def second_differences(n):
    D2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        D2[i, i:i + 3] = [1, -2, 1]
    return D2


def solve_qp(points, s, G, g, name):
    """min d'G d + g'd, d >= 0, sum d <= level, A d = b, by SLSQP."""
    E = points[-1][0]
    level = points[-1][1]
    C, A, b = constraint_matrix(points, s)
    H = G + G.T

    def obj(d):
        return float(d @ G @ d + g @ d)

    def jac(d):
        return H @ d + g

    cons = [
        {"type": "eq", "fun": lambda d: A @ d - b, "jac": lambda d: A},
        {"type": "ineq", "fun": lambda d: level - d.sum(), "jac": lambda d: -np.ones(E)},
    ]
    base = np.array([v for _, v in expand_baseline(points, s)])
    d0 = np.clip(np.diff(np.concatenate([[0.0], base])), 0, None)
    # increments are free (may be negative) only within a span whose given
    # value is below the one before it: downward development, as an
    # incurred pattern with factors below 1 has it
    bounds = [(0, None)] * E
    prev_age, prev_val = 0, 0.0
    for a, val in points:
        if val < prev_val:
            for t in range(prev_age, a):
                bounds[t] = (None, None)
        prev_age, prev_val = a, val
    res = minimize(obj, d0, jac=jac, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"maxiter": 1000, "ftol": 1e-14})
    if not res.success:
        print(f"  {name}:", res.message, file=sys.stderr)
    p = np.concatenate([[0.0], C @ res.x])
    return [(a, avg_of(p, s, a)) for a in range(1, E + 1)]


def expand_monoqp(points, s):
    """Exact and monotone: increments d >= 0 on months 1..E, p = cumsum d,
    p <= level, the averages equal to the given values, least sum of
    squared second differences of p."""
    E = points[-1][0]
    C = np.tril(np.ones((E, E)))
    Q = second_differences(E) @ C
    return solve_qp(points, s, Q.T @ Q, np.zeros(E), "monoqp")


def expand_logqp(points, s):
    """Exact and monotone: `u = ln d` on months 1..E, least sum of squared
    second differences of `u` (zero for geometrically decaying increments,
    the book's `1 - 0.9^t`), the averages matched and `p <= level`."""
    E = points[-1][0]
    level = points[-1][1]
    C, A, b = constraint_matrix(points, s)
    D2 = second_differences(E)
    G = D2.T @ D2

    def obj(u):
        return float(u @ G @ u)

    def jac(u):
        return 2 * G @ u

    cons = [
        {"type": "eq", "fun": lambda u: A @ np.exp(u) - b, "jac": lambda u: A * np.exp(u)},
        {"type": "ineq", "fun": lambda u: level - np.exp(u).sum(), "jac": lambda u: -np.exp(u)},
    ]
    _, p0 = scale_increments(points, s, curve_increments(points, s, "wbl"))
    d0 = np.clip(np.diff(p0), 1e-12, None)
    res = minimize(obj, np.log(d0), jac=jac, bounds=[(-40, 5)] * E, constraints=cons,
                   method="SLSQP", options={"maxiter": 3000, "ftol": 1e-12})
    if not res.success:
        print("  logqp:", res.message, file=sys.stderr)
    p = np.concatenate([[0.0], C @ np.exp(res.x)])
    return [(a, avg_of(p, s, a)) for a in range(1, E + 1)]


def expand_curve_exact(points, s, curve):
    """The curve fitted through the averages, then the nearest monotone
    `p` (least squared change of the increments) whose averages are the
    given values exactly."""
    E = points[-1][0]
    omega, theta, _ = expand_curve(points, s, curve)[1]
    level = points[-1][1]
    t = np.arange(1, E + 1, dtype=float)
    g = level * growth(curve, t, omega, theta) / growth(curve, E, omega, theta)
    d_curve = np.diff(np.concatenate([[0.0], g]))
    return solve_qp(points, s, np.eye(E), -2 * d_curve, f"{curve}+exact")


def growth(curve, t, omega, theta):
    t = np.maximum(np.asarray(t, float), 0.0)
    with np.errstate(divide="ignore", over="ignore"):
        if curve == "llog":
            return 1.0 / (1.0 + (t / theta) ** (-omega))
        return 1.0 - np.exp(-((t / theta) ** omega))


def expand_curve(points, s, curve, mid=False):
    """Clark's curve, `p(t) = L G(t)/G(E)`, (omega, theta) by least squares
    on the percents: through the `s`-month averages, or (`mid`) read at the
    mid-period age `a - s/2` as BenchmarkCurve with origin_width = s."""
    E = points[-1][0]
    level = points[-1][1]
    ages = np.array([a for a, _ in points], float)
    vals = np.array([v for _, v in points])
    t = np.arange(1, E + 1, dtype=float)

    def pattern(th):
        omega, theta = np.exp(th)
        if mid:
            return level * growth(curve, ages - s / 2, omega, theta) / growth(curve, E - s / 2, omega, theta)
        g = level * growth(curve, t, omega, theta) / growth(curve, E, omega, theta)
        p = np.concatenate([[0.0], g])
        return np.array([avg_of(p, s, int(a)) for a in ages])

    def sse(th):
        return float(np.sum((pattern(th) - vals) ** 2))

    best = None
    for w0 in (0.8, 1.5, 3.0):
        for th0 in (6.0, 18.0, 36.0):
            r = minimize(sse, np.log([w0, th0]), method="Nelder-Mead",
                         options={"xatol": 1e-12, "fatol": 1e-16, "maxiter": 4000})
            if best is None or r.fun < best.fun:
                best = r
    omega, theta = np.exp(best.x)
    g = level * growth(curve, t, omega, theta) / growth(curve, E, omega, theta)
    p = np.concatenate([[0.0], g])
    out = [(a, avg_of(p, s, a)) for a in range(1, E + 1)]
    return out, (omega, theta, best.fun)


def curve_increments(points, s, curve):
    E = points[-1][0]
    omega, theta, _ = expand_curve(points, s, curve)[1]
    t = np.arange(1, E + 1, dtype=float)
    g = points[-1][1] * growth(curve, t, omega, theta) / growth(curve, E, omega, theta)
    return np.diff(np.concatenate([[0.0], g]))


def scale_increments(points, s, d_curve):
    """Closed form, forward: within each span between given ages the
    increments keep the curve's shape, scaled by one constant so that the
    span's average is the given value; a scale below 0 (the given value
    below the pattern already reached) is set to 0, and `p` is capped at
    the level. Both flags make the result inexact there."""
    E = points[-1][0]
    level = points[-1][1]
    p = np.zeros(E + 1)
    prev_age = 0
    for a, target in points:
        S = np.cumsum(d_curve[prev_age:a])  # the curve's progress within the span
        # the span's s-month average at age a: p_prev + c * mean of S over
        # the s months ending at a that lie within the span, plus the
        # earlier months' p (already set) for s > span
        months = list(range(a - s + 1, a + 1))
        inside = [t for t in months if t > prev_age]
        before = sum(p[t] if t > 0 else 0.0 for t in months if t <= prev_age)
        mean_s = sum(S[t - prev_age - 1] for t in inside) / s
        c = (target - before / s - len(inside) * p[prev_age] / s) / mean_s if mean_s > 0 else 0.0
        c = max(c, 0.0)
        for t in range(prev_age + 1, a + 1):
            p[t] = min(p[prev_age] + c * S[t - prev_age - 1], level)
        prev_age = a
    return [(a, avg_of(p, s, a)) for a in range(1, E + 1)], p


def expand_curve_scaled(points, s, curve):
    return scale_increments(points, s, curve_increments(points, s, curve))[0]


CANDIDATES = {
    "baseline": lambda pts, s: expand_baseline(pts, s),
    "pchip": lambda pts, s: expand_pchip(pts, s),
    "linknot": lambda pts, s: expand_knots(pts, s, log=False),
    "lnknot": lambda pts, s: expand_knots(pts, s, log=True),
    "monoqp": lambda pts, s: expand_monoqp(pts, s),
    "llog": lambda pts, s: expand_curve(pts, s, "llog")[0],
    "wbl": lambda pts, s: expand_curve(pts, s, "wbl")[0],
    "llog+exact": lambda pts, s: expand_curve_exact(pts, s, "llog"),
    "wbl+exact": lambda pts, s: expand_curve_exact(pts, s, "wbl"),
    "wbl+scale": lambda pts, s: expand_curve_scaled(pts, s, "wbl"),
    "llog-mid": lambda pts, s: expand_curve(pts, s, "llog", mid=True)[0],
}
# `logqp` (least roughness of ln of the increments) is left out of the
# tables: SLSQP on 120 log increments hit its iteration limit in 15 of 17
# runs, so its figures are not evidence about the family. `expand_logqp`
# stays for anyone who wants to try a better solver.


# --- truths --------------------------------------------------------------


def book_p(t):
    return 0.0 if t <= 0 else 1.0 - 0.9 ** min(t, 60)


def curve_truth(curve, omega, theta, E0=108):
    def p(t):
        if t <= 0:
            return 0.0
        if t >= E0:
            return 1.0
        return float(growth(curve, t, omega, theta) / growth(curve, E0, omega, theta))
    return p


def uy_truth(p, e, ages):
    return {a: sum(e[t] * p(a - t) for t in range(len(e))) for a in ages}


def given_annual(p, s=12, E=120, step=12):
    return average(p, s, list(range(step, E + 1, step)))


def checks(expanded, given):
    vals = np.array([v for _, v in expanded])
    mono = bool(np.all(np.diff(vals) >= -1e-12))
    top = max(v for _, v in given)  # the level, or the peak of a pattern that develops downward
    inrange = bool(np.all(vals >= -1e-12) and np.all(vals <= top + 1e-12))
    d = dict(expanded)
    repro = max(abs(d[a] - v) for a, v in given)
    return mono, inrange, repro


def run_case(p, s=12, E=120, step=12, writing=None, m=12, ages=(12, 24, 36, 48, 60)):
    writing = writing if writing is not None else [1.0] * 12
    e = exposure(writing, m)
    truth = uy_truth(p, e, ages)
    given = given_annual(p, s, E, step)
    v = weights_over(writing, m, s)
    rows = {}
    for cname, fn in CANDIDATES.items():
        t0 = time.perf_counter()
        expanded = fn(given, s)
        dt = time.perf_counter() - t0
        conv = convert(expanded, v)
        level = given[-1][1]  # past the result's end the share is the level
        rows[cname] = ([conv.get(a, level) / truth[a] for a in ages], dt, checks(expanded, given))
    return rows


def fmt_row(name, ratios, dt, chk):
    mono, inrange, repro = chk
    return (f"| {name:9s} | " + " | ".join(f"{r:.4f}" for r in ratios)
            + f" | {'yes' if mono else 'NO':3s} | {'yes' if inrange else 'NO':3s} | {repro:.0e} | {dt * 1000:6.1f} |")


HEAD = ("| candidate | 12 | 24 | 36 | 48 | 60 | mono | in [0, L] | repro | ms |\n"
        "|---|---|---|---|---|---|---|---|---|---|")


def table(rows):
    print(HEAD)
    for c, (r, dt, chk) in rows.items():
        print(fmt_row(c, r, dt, chk))
    print()


def industry_patterns():
    out = []
    with open(ROOT / "validation/data/clrd_lines.csv") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    for lob in ("ppauto", "wkcomp", "medmal", "othliab"):
        tri = {}
        for r in rows:
            if r["lob"] == lob:
                tri.setdefault(int(r["origin"]), {})[int(r["development"])] = float(r["paid"])
        ldf = []
        for k in range(12, 120, 12):
            num = sum(v[k + 12] for v in tri.values() if k + 12 in v)
            den = sum(v[k] for v in tri.values() if k + 12 in v)
            ldf.append(num / den)
        cdf = np.cumprod(ldf[::-1])[::-1]
        pts = [(12 * (i + 1), 1.0 / cdf[i]) for i in range(len(cdf))] + [(120, 1.0)]
        out.append((f"CLRD {lob} paid, volume-weighted chain ladder", pts))
    fried = {
        "Friedland XYZ reported (selected, factors below 1)":
            [1.687, 1.265, 1.102, 1.020, 1.050, 1.010, 1.011, 1.000, 0.993, 0.999],
        "Friedland XYZ paid (selected)":
            [3.349, 2.079, 1.574, 1.316, 1.203, 1.136, 1.059, 1.022, 1.017, 1.004],
    }
    for name, ldf in fried.items():
        cdf = np.cumprod(ldf[::-1])[::-1]
        pts = [(12 * (i + 1), 1.0 / cdf[i]) for i in range(len(cdf))] + [(132, 1.0)]
        out.append((name, pts))
    return out


def main():
    print("## Synthetic truths (converted UY share / true UY share, default parallelogram)\n")
    cases = [
        ("book 1 - 0.9^t (held from 60)", book_p),
        ("slow loglogistic omega 1.5 theta 36", curve_truth("llog", 1.5, 36.0)),
        ("fast Weibull omega 1.2 theta 6", curve_truth("wbl", 1.2, 6.0)),
        ("very fast exponential theta 2 (property-like)", curve_truth("wbl", 1.0, 2.0)),
        ("medium Weibull omega 1.5 theta 18", curve_truth("wbl", 1.5, 18.0)),
    ]
    for name, p in cases:
        g = given_annual(p)
        print(f"### {name}: given P(12) = {g[0][1]:.4f}, P(24) = {g[1][1]:.4f}, P(36) = {g[2][1]:.4f}\n")
        table(run_case(p))

    print("## Robustness\n")
    p = book_p
    print("### Accident-year pattern given at every month (must be exact)\n")
    given = average(p, 12, list(range(1, 121)))
    e = exposure([1.0] * 12, 12)
    truth = uy_truth(p, e, (12, 24, 36, 48, 60))
    v = weights_over([1.0] * 12, 12, 12)
    for c in ("baseline", "pchip", "monoqp", "lnknot", "linknot"):
        conv = convert(CANDIDATES[c](given, 12), v)
        print(f"- {c}: max |ratio - 1| = {max(abs(conv[a] / truth[a] - 1) for a in truth):.1e}")
    print("\n### Accident-year pattern at quarterly ages (3, 6, ..., 120)\n")
    table(run_case(p, step=3))
    print("### Accident-quarter pattern (s = 3) at quarterly ages, even writing, 12-month policies\n")
    table(run_case(p, s=3, step=3))
    print("### Three given ages only (12, 24, 36), book truth held at 25 months\n")
    table(run_case(lambda t: 0.0 if t <= 0 else 1.0 - 0.9 ** min(t, 25), E=36, ages=(12, 24, 36, 48, 60)))
    print("### A level below 1 (every share 0.95 of the book's)\n")
    table(run_case(lambda t: 0.95 * book_p(t)))
    print("### Seasonal writing (three times the rate in Q4), 12-month policies, book truth\n")
    table(run_case(p, writing=[1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3]))
    print("### Six-month policies, even writing, book truth\n")
    table(run_case(p, m=6))

    print("## Industry patterns (no truth: the expanded P_12 at 6 and 18 months, the converted UY share at 12 to 60, the checks)\n")
    for name, pts in industry_patterns():
        print(f"### {name}: given " + ", ".join(f"{v:.3f}" for _, v in pts[:4]) + " ...\n")
        print("| candidate | P(6) | P(18) | UY 12 | UY 24 | UY 36 | UY 48 | UY 60 | mono | in [0, L] | repro |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|")
        for c, fn in CANDIDATES.items():
            try:
                ex = fn(pts, 12)
            except Exception as err:  # noqa: BLE001
                print(f"| {c:9s} | error: {type(err).__name__}: {err} |")
                continue
            d = dict(ex)
            conv = convert(ex, v)
            mono, inrange, repro = checks(ex, pts)
            print(f"| {c:9s} | {d[6]:.4f} | {d[18]:.4f} | "
                  + " | ".join(f"{conv[a]:.4f}" for a in (12, 24, 36, 48, 60))
                  + f" | {'yes' if mono else 'NO'} | {'yes' if inrange else 'NO'} | {repro:.0e} |")
        print()


if __name__ == "__main__":
    main()
