"""Regenerate validation/reference/reinsurance_aggregate.csv.

    python3.12 -m venv agg && agg/bin/pip install aggregate==1.0.1
    agg/bin/python validation/scripts/aggregate_reinsurance.py

Reinsurance on the grid against Mildenhall's `aggregate` 1.0.1, three books
on a step of 1 with 2^14 points and no renormalization
(`bs=1, log2=14, normalize=False`):

- `C`: Poisson 2 claims, a lognormal severity (mean 100, cv 2) limited to
  1,000, ceding 500 xs 100 per occurrence;
- `R`: book `C` at 10 claims, with an aggregate 2,000 xs 1,500 on the net
  of the occurrence layer (an inuring tower);
- `S`: a negative binomial count (`mixed gamma 0.3`, prospicio's
  r = 1/0.09, β = 5 × 0.09), a gamma severity (mean 50, cv 1.5) limited to
  1,000, ceding 50% of 400 xs 100 and 500 xs 500.

The quantities:

- `density_<view>` at loss `arg`: the annual gross, ceded and net
  probabilities on the grid (book `R`), from `reins_density_df`'s `p_*` columns;
  the views are `gross`, `ceded_occ` (the occurrence layer), `ceded`
  (the aggregate layer) and `net`;
- `<layer>:<stat>`: `reins_stats_df`'s occurrence block. A per-layer
  column (`layer.1`, `layer.2`) has `pr_attach` (`P(X > a)`, exclusive),
  `pr_detach` (`P(X ≥ a + l)`, inclusive), `pr_loss` and `lol`, the
  excess count (`freq_*`), the layer severity given a loss to it
  (`sev_*`) and the annual ceded loss (`agg_*`), each as mean, cv and skew;
  `gross` has its severity and annual moments;
- `price:<view>:<L|P>`: `reins_price_df` under the proportional hazard
  distortion with ρ = 0.7, each view priced on its own.

Tolerances follow what each value is computed from:

- read from the severity grid (the attach and detach chances, the excess
  count, the layer's severity): rounding, `1e-12` relative; the gross
  severity's cv and skewness `1e-9` (`aggregate` reaches them by another
  route, 4e-10 off);
- `agg_*` and `lol`: `aggregate` takes the moments of its FFT density,
  prospicio the cumulants of the same severity grid; they differ by the
  FFT's round-off and aliasing, observed to 1e-9 on the mean and cv and to
  1.3e-7 on the skewness of a rare layer (skew 18);
- `P`: the proportional hazard raises the survival to the power 0.7, so
  round-off of 1e-16 in the far tail of an FFT density moves the premium
  by about 1e-8 relative (`aggregate`'s own `1 − cumsum(p)` and its sum
  from the top differ by as much). `L` is a mean, `1e-9`.
- the densities: `1e-13` absolute.

A layer placed at a share below 1 (book `S`'s `layer.1`) is compared on its
means and grid probabilities only: `aggregate` maps the shared losses onto
the grid, which splits a ceded 0.5 between 0 and 1 and so lowers `pr_loss`
(by 0.4%) and smears the shape; prospicio scales the full layer's
cumulants by the share exactly.
"""

import csv
import sys
import warnings

import aggregate
from aggregate import Distortion, build

OUT = "validation/reference/reinsurance_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"

BOOKS = [
    (
        "C",
        "lambda=2;severity=lognormal;mean=100;cv=2;cap=1000;"
        "occ=1:500:100",
        "agg C 2 claims 1000 xs 0 sev lognorm 100 cv 2 "
        "occurrence net of 500 xs 100 poisson",
    ),
    (
        "R",
        "lambda=10;severity=lognormal;mean=100;cv=2;cap=1000;"
        "occ=1:500:100;agg=2000:1500",
        "agg R 10 claims 1000 xs 0 sev lognorm 100 cv 2 "
        "occurrence net of 500 xs 100 poisson "
        "aggregate ceded to 2000 xs 1500",
    ),
    (
        "S",
        "lambda=5;mixing_cv=0.3;severity=gamma;mean=50;cv=1.5;cap=1000;"
        "occ=0.5:400:100+1:500:500",
        "agg S 5 claims 1000 xs 0 sev gamma 50 cv 1.5 "
        "occurrence net of 200 po 400 xs 100 and 500 xs 500 mixed gamma 0.3",
    ),
]

# (abs, rel) tolerances: read from the grid, or the FFT density's moments.
GRID = (1e-14, 1e-12)
FFT = {"mean": (0.0, 1e-9), "cv": (0.0, 1e-8), "skew": (0.0, 5e-7)}
PRICE = {"L": (0.0, 1e-9), "P": (0.0, 2e-8)}
# (key, quantity, tolerance, also for a share below 1)
LAYER_STATS = [
    (("meta", "pr_attach"), "pr_attach", GRID, True),
    (("meta", "pr_detach"), "pr_detach", GRID, True),
    (("meta", "pr_loss"), "pr_loss", (1e-13, 1e-11), False),
    (("meta", "lol"), "lol", FFT["mean"], True),
] + [
    ((block, m), f"{block}_{m}", FFT[m] if block == "agg" else GRID,
     block == "freq" or m == "mean")
    for block in ("freq", "sev", "agg")
    for m in ("mean", "cv", "skew")
]
DENSITY_POINTS = [0, 1, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 5000, 8000]


def main():
    rows = []
    ph = Distortion("ph", 0.7)
    for name, params, program in BOOKS:
        a = build(program, bs=1, log2=14, normalize=False)
        stats = a.reins_stats_df["occ"]

        def add(quantity, arg, value, tol):
            rows.append([name, params, quantity, arg, repr(float(value)),
                         tol[0], tol[1], SOURCE])

        for col in stats.columns:
            if col.startswith("layer."):
                shared = stats.loc[("meta", "share"), col] < 1
                for key, q, tol, any_share in LAYER_STATS:
                    if any_share or not shared:
                        add(f"{col}:{q}", "", stats.loc[key, col], tol)
        for block in ("sev", "agg"):
            for m in ("mean", "cv", "skew"):
                if block == "agg":
                    tol = FFT[m]
                else:
                    tol = GRID if m == "mean" else (0.0, 1e-9)
                add(f"gross:{block}_{m}", "", stats.loc[(block, m), "Gross"], tol)

        price = a.reins_price_df(ph).loc["ph"]
        views = {"gross": "gross", "net": "net"}
        if "ceded occ" in price.index:
            views |= {"ceded occ": "ceded_occ", "ceded": "ceded"}
        elif name == "C":
            views["ceded"] = "ceded_occ"
        for theirs, ours in views.items():
            for stat in ("L", "P"):
                add(f"price:{ours}:{stat}", "", price.loc[theirs, stat], PRICE[stat])

        if name == "R":
            df = a.reins_density_df
            cols = {
                "gross": "p_agg_gross",
                "ceded_occ": "p_agg_ceded_occ",
                "ceded": "p_agg_ceded",
                "net": "p_agg_net",
            }
            for view, col in cols.items():
                for x in DENSITY_POINTS:
                    add(f"density_{view}", x, df.loc[x, col], (1e-13, 0.0))

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
