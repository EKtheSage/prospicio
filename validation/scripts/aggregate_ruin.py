"""Regenerate validation/reference/ruin_aggregate.csv.

    python3.12 -m venv agg && agg/bin/pip install aggregate==1.0.1
    agg/bin/python validation/scripts/aggregate_ruin.py

Eventual ruin by the Pollaczeck–Khinchine formula against Mildenhall's
`aggregate` 1.0.1 (`Aggregate.pollaczeck_khinchine` and the Lundberg
exponent `eventual_ruin` reports), on two Poisson books (the rate cancels):

- `G`: a gamma severity, mean 50 and cv 1.5, on a step of 1 with 2^16
  points (`bs=1, log2=16`), at margins ρ = 0.1 and 0.25; and at ρ = 0.1
  with the severity capped (`cap=200`, `X | X <= 200`), in excess
  (`excess=100`, `X | X > 100`) and stopped (`stop_loss=300`, `X ∧ 300`);
- `D`: claims of 1, 2 or 3 with probabilities 0.5, 0.3, 0.2, on a step of 1
  with 2^10 points, at ρ = 0.1.

Quantities: `psi` at initial surplus `arg`; `u_index` and `u_interp`, the
capital for ruin probability `arg` from `find_u` with `kind='index'` and
`'interp'`; `mean`, the severity's grid mean; `lundberg`, the exponent `R`
(on the whole severity only, as `eventual_ruin` computes it).

aggregate computes `ψ` by an FFT of twice the grid with wrapping; prospicio
by its compound FFT at four and eight times the grid. Both are exact to
round-off on these books, so `ψ` matches at `1e-12` absolute and the
capital on the grid exactly. The interpolated capital divides that
round-off by a slope of about 1e-6 at ψ = 0.001: `1e-10` relative
(observed 1.5e-11). `R` (brentq against bisection) at `1e-10` relative.
"""

import csv
import sys
import warnings

import aggregate
from aggregate import build
from aggregate._aggregate import _lundberg_exponent

OUT = "validation/reference/ruin_aggregate.csv"
SOURCE = f"aggregate {aggregate.__version__}"


def main():
    warnings.filterwarnings("ignore")
    rows = []

    def add(book, params, quantity, arg, value, abs_tol, rel_tol):
        rows.append([book, params, quantity, repr(float(arg)), "", repr(float(value)),
                     abs_tol, rel_tol, SOURCE])

    g = build("agg G 10 claims sev gamma 50 cv 1.5 poisson", bs=1, log2=16)
    d = build("agg D 10 claims dsev [1 2 3] [.5 .3 .2] poisson", bs=1, log2=10)
    cases = [("G", g, 0.1, {}), ("G", g, 0.25, {}), ("G", g, 0.1, {"cap": 200}),
             ("G", g, 0.1, {"excess": 100}), ("G", g, 0.1, {"stop_loss": 300}),
             ("D", d, 0.1, {})]
    for book, agg, rho, part in cases:
        params = f"rho={rho!r}" + "".join(f";{k}={v}" for k, v in part.items())
        ruin, find_u, mean, _ = agg.pollaczeck_khinchine(rho, **part)
        _, find_interp, _, _ = agg.pollaczeck_khinchine(rho, kind="interp", **part)
        add(book, params, "mean", 0, mean, "0.0", "1e-12")
        us = [0, 1, 2, 3, 5, 10, 20, 30] if book == "D" else [0, 1, 10, 50, 100, 500, 1000, 2000, 5000]
        for u in us:
            add(book, params, "psi", u, ruin.loc[u], "1e-12", "0.0")
        for p in [0.5, 0.1, 0.01, 0.001]:
            if p < ruin.iloc[0]:
                add(book, params, "u_index", p, find_u(p), "0.0", "0.0")
                add(book, params, "u_interp", p, find_interp(p), "0.0", "1e-10")
        if not part:
            bit = agg.sev_density_df.p_sev
            r = _lundberg_exponent(bit.to_numpy(), bit.index.to_numpy(), rho)
            if r is not None:
                add(book, params, "lundberg", 0, r, "0.0", "1e-10")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected",
                    "abs_tol", "rel_tol", "source"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
