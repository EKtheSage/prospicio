"""Summarises the arms of the sub-annual study.

Reads the CSVs that validation/tests/reserving_compartmental_subannual.rs
writes to target/compartmental-subannual/ (or the directory given as the first
argument) and prints one line per arm: the books, the fits that failed, the
coverage of the 90% interval of the total reserve with its binomial standard
error, the Kolmogorov-Smirnov D of the ultimate percentile against the uniform
with its 5% critical value (1.36 / sqrt(n), Meyers' test), the mean reserve
against the outcome, the sd, and the sampler's diagnostics. A second table has
each scored parameter's coverage of the central 90%. `grouped_*.csv` gets a
table of its own, each book's per-age fit against its grouped fit.

Run: uv run --no-project python validation/scripts/compartmental_subannual_summary.py
"""

import csv
import math
import statistics
import sys
from pathlib import Path


def ks(percentiles):
    """The Kolmogorov-Smirnov distance of percentiles (0 to 100) from the uniform."""
    x = sorted(p / 100.0 for p in percentiles)
    n = len(x)
    return max(max((i + 1) / n - v, v - i / n) for i, v in enumerate(x))


def read(path):
    """The header, the good rows of an arm's CSV and the fits that failed."""
    with path.open(newline="") as f:
        lines = list(csv.reader(f))
    head, good, errors = lines[0], [], 0
    for row in lines[1:]:
        if len(row) > 3 and row[3] == "error":
            errors += 1
        elif len(row) == len(head):
            good.append({k: float(v) for k, v in zip(head, row)})
    return head, good, errors


def med(rows, key):
    return statistics.median(r[key] for r in rows)


def summary(path):
    head, rows, errors = read(path)
    if not rows:
        return None
    n = len(rows)
    cover = sum(r["q05"] <= r["outcome"] <= r["q95"] for r in rows) / n
    d = ks([r["percentile"] for r in rows])
    bias = statistics.mean((r["mean"] - r["outcome"]) / r["outcome"] for r in rows)
    z = [(r["mean"] - r["outcome"]) / r["sd"] for r in rows]
    line = {
        "arm": path.stem,
        "n": n,
        "err": errors,
        "cells": int(med(rows, "cells")),
        "cover90": f"{100 * cover:.0f}",
        "se": f"{100 * math.sqrt(0.09 / n):.0f}",
        "D": f"{100 * d:.1f}",
        "Dcrit": f"{136 / math.sqrt(n):.1f}",
        "bias%": f"{100 * bias:.1f}",
        "z_mean": f"{statistics.mean(z):.2f}",
        "z_sd": f"{statistics.pstdev(z):.2f}",
        "rmse": f"{math.sqrt(statistics.mean((r['mean'] - r['outcome']) ** 2 for r in rows)):.0f}",
        "sd": f"{med(rows, 'sd'):.0f}",
        "secs": f"{med(rows, 'seconds'):.1f}",
        "rhat": f"{med(rows, 'max_rhat'):.3f}",
        "rhat>1.01": f"{100 * sum(r['max_rhat'] > 1.01 for r in rows) / n:.0f}",
        "ess": f"{med(rows, 'ess_bulk'):.0f}",
        "div_books": sum(r["divergences"] > 0 for r in rows),
        "div_med": f"{med(rows, 'divergences'):.0f}",
        "swaps": sum(r["rate_swaps"] > 0 for r in rows),
    }
    params = [
        (k[:-4], 100 * sum(5 <= r[k] <= 95 for r in rows) / n)
        for k in head
        if k.endswith("_pct") and all(not math.isnan(r[k]) for r in rows)
    ]
    means = [(k[:-5], statistics.mean(r[k] for r in rows)) for k in head if k.endswith("_mean")]
    return line, params, means


def grouped(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    forms = {}
    for r in rows:
        forms.setdefault(r["form"], {})[int(r["book"])] = r
    both = sorted(set(forms.get("per_age", {})) & set(forms.get("grouped", {})))
    if not both:
        return
    print(f"\n{path.stem}: {len(both)} books")
    for form in ("per_age", "grouped"):
        fs = [forms[form][b] for b in both]

        def num(key):
            return [float(r[key]) for r in fs]

        cover = sum(float(r["q05"]) <= float(r["outcome"]) <= float(r["q95"]) for r in fs) / len(fs)
        elpd = [v for v in num("elpd_loo") if not math.isnan(v)]
        print(
            f"  {form}: dim {fs[0]['draw_dim']} secs {statistics.median(num('seconds')):.1f}"
            f" rhat {statistics.median(num('max_rhat')):.3f}"
            f" rhat>1.01 {100 * sum(v > 1.01 for v in num('max_rhat')) / len(fs):.0f}%"
            f" ess {statistics.median(num('ess_bulk')):.0f}"
            f" div_books {sum(v > 0 for v in num('divergences'))}"
            f" cover90 {100 * cover:.0f} D {100 * ks(num('percentile')):.1f}"
            f" sd {statistics.median(num('sd')):.0f}"
            f" mean {statistics.mean(num('mean')):.0f}"
            f" elpd {statistics.mean(elpd):.1f}"
        )
    a, g = forms["per_age"], forms["grouped"]
    diff = [(float(g[b]["mean"]) - float(a[b]["mean"])) / float(a[b]["sd"]) for b in both]
    elpd = [float(g[b]["elpd_loo"]) - float(a[b]["elpd_loo"]) for b in both]
    elpd = [v for v in elpd if not math.isnan(v)]
    print(
        f"  grouped less per_age: reserve {statistics.mean(diff):.2f} sd on average"
        f" (mean absolute {statistics.mean(abs(v) for v in diff):.2f}),"
        f" elpd {statistics.mean(elpd):.1f}, better in {sum(v > 0 for v in elpd)} of {len(elpd)}"
    )


def paired(folder, seasonal, plain):
    """Book by book, a seasonal arm against the same book without the factor."""
    a, b = folder / f"{seasonal}.csv", folder / f"{plain}.csv"
    if not (a.exists() and b.exists()):
        return
    _, sr, _ = read(a)
    _, pr, _ = read(b)
    by_book = {r["book"]: r for r in pr}
    both = [(r, by_book[r["book"]]) for r in sr if r["book"] in by_book]
    if not both:
        return
    err = [(s["mean"] - s["outcome"]) / s["sd"] for s, _ in both]
    err0 = [(p["mean"] - p["outcome"]) / p["sd"] for _, p in both]
    shift = [(s["mean"] - p["mean"]) / p["mean"] for s, p in both]
    out = (
        f"{seasonal} against {plain}: {len(both)} books,"
        f" mean less outcome {statistics.mean(err):.2f} sd against {statistics.mean(err0):.2f},"
        f" fitted reserve {100 * statistics.mean(shift):.2f}% (mean absolute"
        f" {100 * statistics.mean(abs(v) for v in shift):.2f}%),"
        f" outcome {100 * statistics.mean((s['outcome'] - p['outcome']) / p['outcome'] for s, p in both):.2f}%,"
        f" sd ratio {statistics.mean(s['sd'] / p['sd'] for s, p in both):.3f}"
    )
    for key in ("sigma_pd_mean", "sigma_os_mean"):
        if key in both[0][0]:
            out += f", {key[:-5]} ratio {statistics.mean(s[key] / p[key] for s, p in both):.3f}"
    print(out)


def main():
    root = Path(__file__).resolve().parents[2] / "target" / "compartmental-subannual"
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else root
    paths = sorted(folder.glob("*.csv"))
    skip = ("prior_bands", "zero_cells")
    results = [summary(p) for p in paths if not p.stem.startswith("grouped_") and p.stem not in skip]
    results = [r for r in results if r]
    if results:
        keys = list(results[0][0])
        print("\t".join(keys))
        for line, _, _ in results:
            print("\t".join(str(line[k]) for k in keys))
        print("\nCoverage of the central 90% of each scored parameter, % of books")
        for line, params, _ in results:
            print(line["arm"], " ".join(f"{k} {v:.0f}" for k, v in params))
        print("\nMean posterior mean of each scored parameter")
        for line, _, means in results:
            print(line["arm"], " ".join(f"{k} {v:.3f}" for k, v in means))
    print()
    for g in ("aa", "aq"):
        paired(folder, f"season_{g}_m1_preset_study", f"iid_{g}_m1_preset_study")
    paired(folder, "season117_aq_m1_preset_study", "iid117_aq_m1_preset_study")
    paired(folder, "season114_aq_m1_preset_study", "iid114_aq_m1_preset_study")
    paired(folder, "seasonq_qq_m1_preset_study", "earned_qq_m1_preset_study")
    paired(folder, "m2season_aq_m2_preset_study", "m2y_aq_m2_preset_study")
    paired(folder, "m2season114_aq_m2_preset_study", "m2y114_aq_m2_preset_study")
    for p in paths:
        if p.stem.startswith("grouped_"):
            grouped(p)


if __name__ == "__main__":
    main()
