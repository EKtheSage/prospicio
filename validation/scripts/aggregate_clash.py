"""Regenerate validation/reference/clash_aggregate.csv.

    python3.12 -m venv agg && agg/bin/pip install aggregate==1.0.1
    agg/bin/python validation/scripts/aggregate_clash.py

Clash, two lines hit by one count of events, against Mildenhall's
`aggregate` 1.0.1 (`clash NAME na nb nc claims <A> <B> <freq>`, a
`BivariateAggregate` with independent triggers). Line A loses a gamma
(mean 10, cv 1.5) when hit and line B a lognormal (mean 5, cv 0.5);
2 events a year hit A only, 3 B only and 1 both, so 6 hit neither and 12
in all, with `p_A = 1/4` and `p_B = 1/3`. Two shared counts:

- `poisson`;
- `mixed`: a gamma-mixed Poisson (negative binomial) with mixing cv 0.5,
  so variance `12 + 0.25 * 12^2 = 48`.

Each on two grids (`update(bs=(h_A, h_B), log2=(l_A, l_B))`, padding 1, so
the FFT buffer is twice each axis):

- `U`: steps 1 and 0.5 on 2^11 and 2^10 points (the total on the larger
  step by `aggregate`'s mean-keeping split);
- `E`: step 1 on both, 2^11 and 2^9 points (the total exact on the
  anti-diagonals).

`aggregate` discretizes each severity by rounding and rescales it to sum
to 1; prospicio's `Grid::rounding` carries the tail on the last point.
These severities have no mass past their grids to 1e-12, so the inputs
agree to that. Quantities: the joint density at cells `(arg, arg2)`
(bucket indices), each marginal's density at `arg` (`marginal_a`,
`marginal_b`), the total's density at `arg` (`total`), the means and
variances of each axis and of the total, `E[AB]` and the correlation.
`aggregate` truncates the joint (`deficit`) where prospicio carries the
tail on the last row and column; the grids reach far enough that the
deficit is below 1e-13 (asserted). `aggregate` also zeroes joint cells
below 1e-15 as round-off; that drops about 1e-11 of real tail mass here
and moves the second moments by 3e-9, so the script turns it off.
Probabilities at `1e-12` absolute, moments at `1e-9` relative (observed
4e-11).
"""

import csv
import sys
import warnings

import numpy as np

import aggregate
import aggregate.bivariate
from aggregate import build

OUT = "validation/reference/clash_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"
SEV = "sev gamma 10 cv 1.5 sev lognorm 5 cv 0.5"


def main():
    warnings.filterwarnings("ignore")
    # aggregate zeroes every joint cell below 1e-15 (round-off fuzz). On
    # these grids that drops about 1e-11 of real mass far in the tail and
    # moves the second moments by 3e-9; prospicio keeps it. Turned off, the
    # deficit is round-off (asserted).
    aggregate.bivariate._clip_density_fuzz = lambda density, where: None
    rows = []
    for count, freq in [("poisson", "poisson"), ("mixed", "mixed gamma 0.5")]:
        for _grid, bs, log2 in [("U", (1.0, 0.5), (11, 10)), ("E", (1.0, 1.0), (11, 9))]:
            b = build(f"clash C 2 3 1 claims {SEV} {freq}", update=False)
            b.update(bs=bs, log2=log2)
            assert b.padding == 1 and list(b._nout) == [1 << l for l in log2]
            assert b.clash["pa"] == 0.25
            assert abs(b.deficit) < 1e-13, b.deficit
            params = f"step_a={bs[0]!r};step_b={bs[1]!r};" \
                     f"points_a={1 << log2[0]};points_b={1 << log2[1]}"

            def add(quantity, arg, arg2, value, abs_tol, rel_tol):
                rows.append([count, params, quantity, repr(float(arg)),
                             "" if arg2 is None else repr(float(arg2)),
                             repr(float(value)), abs_tol, rel_tol, SOURCE])

            d = b.density
            for i, j in [(0, 0), (0, 10), (10, 0), (5, 7), (30, 40), (60, 20), (100, 50),
                         (200, 100)]:
                add("density", i, j, d[i, j], "1e-12", "0.0")
            ma, mb = b.marginals
            for k in [0, 1, 10, 30, 60, 100, 200, 400]:
                add("marginal_a", k, None, ma[k], "1e-12", "0.0")
            for k in [0, 1, 10, 30, 60, 100, 200]:
                add("marginal_b", k, None, mb[k], "1e-12", "0.0")
            t = b.total
            tp = np.asarray(t.p, dtype=float)
            for k in [0, 1, 10, 50, 100, 200, 400]:
                add("total", k, None, tp[k], "1e-12", "0.0")
            m = b.moments(2).to_numpy()
            add("mean_a", 0, None, m[1, 0], "0.0", "1e-9")
            add("mean_b", 0, None, m[0, 1], "0.0", "1e-9")
            add("var_a", 0, None, m[2, 0] - m[1, 0] ** 2, "0.0", "1e-9")
            add("var_b", 0, None, m[0, 2] - m[0, 1] ** 2, "0.0", "1e-9")
            add("moment_ab", 0, None, m[1, 1], "0.0", "1e-9")
            add("correlation", 0, None, b.corr, "0.0", "1e-9")
            xs = np.asarray(t.x, dtype=float)
            add("total_mean", 0, None, float((xs * tp).sum()), "0.0", "1e-9")
            add("total_step", 0, None, xs[1] - xs[0], "0.0", "0.0")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
