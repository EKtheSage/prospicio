"""Regenerate validation/reference/contract_terms_aggregate.csv.

    python3.12 -m venv agg && agg/bin/pip install aggregate==1.0.1
    agg/bin/python validation/scripts/aggregate_contract_terms.py

Variable-rating contract terms against Mildenhall's `aggregate` 1.0.1
(`aggregate.contract_terms`), as currency amounts for one year's loss:

- `swing_premium`: a swing-rated layer's premium at ceded loss `arg`,
  `SwingTerms` built for the placed `share` as aggregate's underwriter
  builds it (`Underwriter._make_feature_terms`, which scales the basic
  premium, minimum and maximum by the share).
- `retro_premium`: `RetroTerms` at account loss `arg`.
- `sliding_commission`: `SlideTerms.phi(arg / premium) * premium`, the
  commission the analysis layer books on a ceded loss `arg`.
- `profit_commission`: `ProfitCommissionTerms.phi(arg / premium) * premium`.

Each set of terms is checked at losses across its range, through every
kink. Values match to rounding (`1e-12` relative).
"""

import csv
import sys

import aggregate
import numpy as np
from aggregate.contract_terms import ProfitCommissionTerms, RetroTerms, SlideTerms
from aggregate.underwriter import Underwriter

OUT = "validation/reference/contract_terms_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"

SWINGS = [  # (name, basic, lcm, minimum, maximum, share); None: aggregate's default
    ("swing_a", 0.0, 1.0, 100.0, 300.0, 1.0),
    ("swing_b", 20.0, 1.25, 50.0, 300.0, 0.4),
    ("swing_c", 100.0, 100 / 80, None, None, 0.75),
]
RETROS = [  # (name, basic, lcm, minimum, maximum)
    ("retro_a", 1000.0, 1.10, 1000.0, 2500.0),
    ("retro_b", 300.0, 1.2 * 1.05, 450.0, None),
]
SLIDES = [  # (name, premium, ((commission, loss_ratio), ...))
    ("slide_a", 100.0, ((0.45, 0.60), (0.25, 0.70), (0.19, 0.80))),
    ("slide_b", 6e6, ((0.35, 0.50), (0.25, 0.70))),
    ("slide_c", 250.0, ((0.30, 0.55),)),
]
PCS = [  # (name, premium, share, allowance)
    ("pc_a", 100.0, 0.25, 0.10),
    ("pc_b", 6e6, 0.20, 0.30),
    ("pc_c", 40.0, 0.50, 0.0),
]


def fmt(**kw):
    return ";".join(f"{k}={v!r}" for k, v in kw.items() if v is not None)


def main():
    rows = []
    for name, basic, lcm, lo, hi, share in SWINGS:
        params = {"basic": basic, "lcm": lcm}
        if lo is not None:
            params["minimum"] = lo
        if hi is not None:
            params["maximum"] = hi
        terms = Underwriter._make_feature_terms("swing", params, share=share)
        p = fmt(basic=basic, lcm=lcm, minimum=lo, maximum=hi, share=share)
        for x in np.linspace(0.0, 400.0, 33):
            rows.append((name, p, "swing_premium", x, float(terms.phi(x))))
    for name, basic, lcm, lo, hi in RETROS:
        terms = RetroTerms(basic=basic, lcm=lcm, minimum=lo, maximum=hi)
        p = fmt(basic=basic, lcm=lcm, minimum=lo, maximum=hi)
        for x in np.linspace(0.0, 2000.0, 41):
            rows.append((name, p, "retro_premium", x, float(terms.phi(x))))
    for name, premium, anchors in SLIDES:
        terms = SlideTerms(anchors=anchors)
        flat = {f"c{i}": c for i, (c, _) in enumerate(anchors)}
        flat.update({f"lr{i}": lr for i, (_, lr) in enumerate(anchors)})
        p = fmt(premium=premium, n=len(anchors), **flat)
        for lr in np.linspace(0.0, 1.2, 49):
            x = lr * premium
            rows.append((name, p, "sliding_commission", x, float(terms.phi(x / premium)) * premium))
    for name, premium, share, allowance in PCS:
        terms = ProfitCommissionTerms(share=share, allowance=allowance)
        p = fmt(premium=premium, share=share, allowance=allowance)
        for lr in np.linspace(0.0, 1.2, 49):
            x = lr * premium
            rows.append((name, p, "profit_commission", x, float(terms.phi(x / premium)) * premium))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "expected", "abs_tol", "rel_tol", "source"])
        for name, params, qty, arg, val in rows:
            w.writerow([name, params, qty, repr(float(arg)), repr(val), 1e-9, 1e-12, SOURCE])
    print(f"wrote {OUT}: {len(rows)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
