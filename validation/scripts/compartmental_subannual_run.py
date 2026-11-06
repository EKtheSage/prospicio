"""Lists and runs every arm of the sub-annual study.

The study's results are CSVs in target/compartmental-subannual/, which is
gitignored, so a clean checkout has none. This script holds the arms, the
number of books each ran on and the commands that produced them, as run on
2026-11-05 and 06. An arm is `world:grain:model:prior:sampler`; its CSV is the
arm with the colons turned into underscores. Every command appends to its CSV
and skips books already there, so a rerun resumes.

    uv run --no-project python validation/scripts/compartmental_subannual_run.py --list
    uv run --no-project python validation/scripts/compartmental_subannual_run.py --run --jobs 2
    uv run --no-project python validation/scripts/compartmental_subannual_run.py --run --only persist

A full run is 1,637 fits, about 12.4 hours of summed fit time in the original
run (4 threads a fit, release build), 6 to 7 hours of wall time at two fits at
a time (--jobs 2). Seeds are fixed by the harness (book `b` of a world is
`StreamRng::new(20261105, stream * 10,000 + b)`, its fit seed `1,000 + b`), and
NUTS draws differ by platform, so a rerun matches to the study's tolerances and
not to the last digit. The summary tables of the original run are in
validation/results/.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

S, D = "preset:study", "preset:default"
RW = "ratewide:study"

# (arm, books). Order is the original queues' order, cheapest first.
PANELS = [
    *[(f"iid:{g}:m1:{S}", 50) for g in ("aa", "aq", "am")],
    *[(f"persist:{g}:m1:{S}", 50) for g in ("aa", "aq", "am")],
    *[(f"persist:{g}:m1i:{S}", 50) for g in ("aa", "aq", "am")],
    *[(f"earned:{g}:m1:{S}", 50) for g in ("aa", "aq", "am", "qq")],
    ("earned:qm:m1:" + S, 20),
    *[(f"m2y:{g}:m2:{S}", 50) for g in ("aa", "aq")],
    ("m2y:am:m2:" + S, 20),
    *[(f"m2q:{g}:m2:{S}", 50) for g in ("aa", "aq")],
    ("m2qo:qq:m2:" + S, 20),
    ("m2seasonq:qq:m2:" + S, 20),
    *[(f"season:{g}:m1:{S}", 20) for g in ("aa", "aq")],
    ("seasonq:qq:m1:" + S, 20),
    ("season117:aq:m1:" + S, 20),
    ("iid117:aq:m1:" + S, 20),
    ("season114:aq:m1:" + S, 20),
    ("iid114:aq:m1:" + S, 20),
    ("m2season:aq:m2:" + S, 20),
    ("m2y114:aq:m2:" + S, 20),
    ("m2season114:aq:m2:" + S, 20),
    *[(f"{w}:{g}:m1:{S}", 20) for w in ("fast", "conflict") for g in ("aa", "aq", "am")],
    *[(f"{w}:{g}:m1:{RW}", 20) for w in ("fast", "conflict", "iid") for g in ("aa", "aq", "am")
      if (w, g) != ("iid", "am")],
    ("m2y:aq:m2:" + RW, 20),
    ("m2y:aq:m2:sigmawide:study", 20),
    # The default sampler, on a few books of some arms.
    *[(f"iid:{g}:m1:{D}", 5) for g in ("aa", "aq", "am")],
    ("earned:qq:m1:" + D, 3),
    ("m2y:aa:m2:" + D, 5),
    ("m2y:aq:m2:" + D, 3),
    ("m2y:am:m2:" + D, 1),
]

# Model 2's per-age levels against levels by development year (the test-only
# code in prospicio-reserving), with the number of books.
GROUPED = [("m2y_aq", 30), ("m2q_aq", 30), ("m2season_aq", 20), ("m2y_am", 10)]

# Tests that need no fit.
EXTRA = ["prior_bands", "zero_cells"]

PANEL_CMD = [
    "cargo", "test", "--release", "-p", "prospicio-validation", "--features", "slow-bayes",
    "--test", "reserving_compartmental_subannual",
]
GROUPED_CMD = [
    "cargo", "test", "--release", "-p", "prospicio-reserving", "--features", "bayes", "--lib",
    "grouped_against_per_age", "--", "--ignored", "--nocapture",
]


def jobs():
    for arm, books in PANELS:
        yield arm, {"SUBANNUAL_ARMS": arm, "SUBANNUAL_BOOKS": str(books)}, PANEL_CMD + [
            "panel", "--", "--ignored", "--nocapture"]
    for name, books in GROUPED:
        yield "grouped:" + name, {"SUBANNUAL_GROUPED": name, "SUBANNUAL_BOOKS": str(books)}, GROUPED_CMD
    for name in EXTRA:
        yield name, {}, PANEL_CMD + [name, "--", "--ignored", "--nocapture"]


def run(job):
    label, env, cmd = job
    full = dict(os.environ, RAYON_NUM_THREADS="4", **env)
    root = Path(__file__).resolve().parents[2]
    code = subprocess.call(cmd, env=full, cwd=root)
    print(f"DONE {label} exit {code}", flush=True)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print every arm and its books")
    ap.add_argument("--run", action="store_true", help="run them")
    ap.add_argument("--jobs", type=int, default=2, help="fits at a time (4 threads each)")
    ap.add_argument("--only", default="", help="only labels containing this text")
    args = ap.parse_args()
    chosen = [j for j in jobs() if args.only in j[0]]
    if args.list or not args.run:
        for label, env, _ in chosen:
            print(label, env.get("SUBANNUAL_BOOKS", ""))
        print(f"{len(chosen)} commands")
        return 0
    with ThreadPoolExecutor(args.jobs) as pool:
        codes = list(pool.map(run, chosen))
    return 1 if any(codes) else 0


if __name__ == "__main__":
    sys.exit(main())
