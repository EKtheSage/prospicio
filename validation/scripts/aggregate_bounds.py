"""Regenerate validation/reference/bounds_aggregate.csv.

    python3.12 -m venv agg && agg/bin/pip install aggregate==1.0.1
    agg/bin/python validation/scripts/aggregate_bounds.py

Pricing bounds against Mildenhall's `aggregate` 1.0.1
(`aggregate.bounds.PricingBounds` and `Bounds`). The reference risk X is
Monograph 15's InsCo total (22, 28, 36, 55, 65 and 100 with probability 0.1
each, 40 with 0.4).

- `price`: the price range of the risks Y given X priced at P, from
  `PricingBounds(X, Y, a=).bounds(P)`: Y the InsCo unit X3 on its own
  (0 with 0.3, then 7, 8, 9, 10, 11, 40 and 64 with 0.1 each) and a
  four-point risk (0, 30, 60, 90 with 0.4, 0.3, 0.2, 0.1), uncapped and
  capped at 65 (`params` `a=`), at several P. Quantity `<risk>.lower` or
  `.upper`, `arg` P.
- `envelope`: the envelope of the distortions pricing X at 53.565 (the
  CCoC price at 15%), as the price range of a Bernoulli risk with
  `P(1) = s`, from `PricingBounds` (`arg` s). The levels are at or above
  X's top atom (0.1): below it `aggregate` stops at the last breakpoint
  under 1 and misses `TVaR_1`, so its upper bound is too low there
  (prospicio includes the limit).
- `bounds_min` and `bounds_max`: `Bounds(X, 53.565).min_envelope.g(s)` and
  `max_envelope(s)`, `aggregate`'s grid search over 256 levels. Its cloud is
  nearly a subset of the distortions pricing X at P, so the exact envelope
  lies outside it: the test checks prospicio's lower at or below
  `bounds_min` and upper at or above `bounds_max`, and within 0.05 of them.
  "Nearly": its members price X at P to its root tolerance (`2^-17` in the
  level) and `max_envelope` interpolates linearly in s, so it pokes out by
  up to 2.4e-4 (at s = 0.1, the kink at X's top atom); the test allows
  1e-3.

The hull vertices are exact, so `price` and `envelope` match at `1e-12`
(relative for prices, absolute for the envelope).

Where X saturates at the cap before Y does (X capped at 65 has a top atom of
0.2, X3 one of 0.1 at 64), `PricingBounds` keeps the first vertex at the
saturated price and drops the limit `p → 1`, so its upper bound is too low:
at P = 65 only TVaRs at levels of 0.8 or more price X at 65, and `TVaR_1`
prices X3 at 64 where `PricingBounds` says 52. Every price row is also
computed by brute force over the BiTVaRs on the curve's vertices with
`p = 1` included (the extreme points of the set, by `aggregate`'s own
TVaR); where the two differ the row holds the brute force, and its
`source` says so.
"""

import csv
import sys
import warnings

import aggregate
import numpy as np
import pandas as pd
from aggregate._grid_distribution import GridDistribution
from aggregate.bounds import Bounds, PricingBounds

OUT = "validation/reference/bounds_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"
X = pd.Series([0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1], index=[22.0, 28, 36, 40, 55, 65, 100])
YS = {
    "X3": pd.Series([0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    index=[0.0, 7, 8, 9, 10, 11, 40, 64]),
    "Z": pd.Series([0.4, 0.3, 0.2, 0.1], index=[0.0, 30, 60, 90]),
}
PREMIUM = 53.565217391304344
LEVELS = [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9]


def capped(ser, a):
    """The pmf of min(Y, a) as (x, p)."""
    x = np.minimum(ser.index.to_numpy(float), a)
    u, inv = np.unique(x, return_inverse=True)
    p = np.zeros_like(u)
    np.add.at(p, inv, ser.to_numpy(float))
    return u, p


def brute_force(x_ser, y_ser, a, P):
    """The price range of Y over every BiTVaR on the curve's vertices,
    p = 1 included, that prices X at P: the extreme points of the set."""
    xs, ys = capped(x_ser, a), capped(y_ser, a)
    tx, ty = GridDistribution(*xs).tvar, GridDistribution(*ys).tvar
    ps = np.unique(np.concatenate([[0.0], np.cumsum(xs[1])[:-1], np.cumsum(ys[1])[:-1]]))
    t = [float(tx(p)) for p in ps] + [xs[0][-1]]
    y = [float(ty(p)) for p in ps] + [ys[0][-1]]
    lo, hi = np.inf, -np.inf
    for i in range(len(t)):
        for j in range(len(t)):
            if t[i] <= P <= t[j]:
                w = 0.0 if t[j] == t[i] else (P - t[i]) / (t[j] - t[i])
                v = (1 - w) * y[i] + w * y[j]
                lo, hi = min(lo, v), max(hi, v)
    return lo, hi


def main():
    warnings.filterwarnings("ignore")
    rows = []

    def add(dist, params, quantity, arg, value, abs_tol, rel_tol):
        rows.append([dist, params, quantity, repr(float(arg)), "", repr(float(value)),
                     abs_tol, rel_tol, SOURCE])

    for a, premiums in [(np.inf, [46.6, 50, PREMIUM, 70, 90, 100]), (65.0, [44.8, 47, 50, 60, 65])]:
        pb = PricingBounds(X, YS, a=a)
        b = pb.bounds(premiums)
        for (P, risk), row in b.iterrows():
            exact = dict(zip(["lower", "upper"], brute_force(X, YS[risk], a, P)))
            for side in ["lower", "upper"]:
                if abs(row[side] - exact[side]) <= 1e-9 * abs(exact[side]):
                    add("price", f"a={a}", f"{risk}.{side}", P, row[side], "0.0", "1e-12")
                else:
                    # aggregate drops the p -> 1 vertex where X saturates.
                    rows.append(["price", f"a={a}", f"{risk}.{side}", repr(float(P)), "",
                                 repr(float(exact[side])), "0.0", "1e-12",
                                 f"BiTVaRs on the curve's vertices with p = 1, by {SOURCE}'s "
                                 f"TVaR (PricingBounds gives {float(row[side])!r})"])
    bern = {f"{s}": pd.Series([1 - s, s], index=[0.0, 1.0]) for s in LEVELS}
    env = PricingBounds(X, bern).bounds(PREMIUM)
    for (_, s), row in env.iterrows():
        for side in ["lower", "upper"]:
            add("envelope", f"premium={PREMIUM!r}", side, float(s), row[side], "1e-12", "0.0")
    grid = Bounds(X, PREMIUM)
    for s in LEVELS + [0.05, 0.4, 0.6, 0.8]:
        add("bounds_min", f"premium={PREMIUM!r}", "lower", s, grid.min_envelope.g(s), "0.0", "0.0")
        add("bounds_max", f"premium={PREMIUM!r}", "upper", s, grid.max_envelope(s), "0.0", "0.0")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
