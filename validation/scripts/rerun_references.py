#!/usr/bin/env python3
"""Rerun reference generators and compare their output with the committed files (#844).

Each generator named here runs in the environment its own inline script
metadata pins (PEP 723, `uv run`), or with `--latest` in the newest release
of each package it names (the pins and the resolution date dropped). Every
file the manifest (`validation/reference/MANIFEST.toml`) lists for the
generator is compared with the committed copy, then put back, so the working
tree is the same afterwards:

- a CSV with `expected`, `abs_tol` and `rel_tol` columns: rows are matched on
  every other column but `source`, and a new `expected` passes when it is
  within `max(abs_tol, rel_tol * |committed|)` of the committed value; a row
  added or removed is a difference, and a changed `source` (a package that
  moved) is listed but is not one;
- another CSV: the same rows and columns, numbers within 1e-9 relative, text
  exact;
- any other file: the same bytes, line endings as LF.

Some generators also rewrite an input under `validation/data/` (the
stacking scripts write the log densities they then weight); every file
there must come back the same (a CSV's numbers within 1e-9 relative, as
another CSV above), and is put back too.

    python3 validation/scripts/rerun_references.py               # the weekly set, pinned
    python3 validation/scripts/rerun_references.py --latest      # the weekly set, newest releases
    python3 validation/scripts/rerun_references.py aggregate_oep.py scipy_special.py
    python3 validation/scripts/rerun_references.py --list

`--report FILE` appends a markdown summary (CI passes `$GITHUB_STEP_SUMMARY`).
The exit status is 1 when any generator fails or any file differs. The
weekly workflow `.github/workflows/references.yml` runs both modes; it is not
a required check, since a new release of a reference package is the signal
it exists to give, not a fault of a pull request.

R, Stan, chainladder-python and the slow or data-fetching generators are not
in the weekly set; rerun them by hand (`uv run validation/scripts/<name>.py`
for the Python ones) and record the result with
`check_reference_manifest.py --update`.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "validation" / "scripts"
REFERENCE = ROOT / "validation" / "reference"
DATA = ROOT / "validation" / "data"
MANIFEST = REFERENCE / "MANIFEST.toml"
TIMEOUT = 1800

# The cheap pure-Python generators the weekly job reruns: the aggregate,
# SciPy, statsmodels, NumPy and mpmath families (#844, part 3). Not here:
# statsmodels_fremtpl2.py (it needs the downloaded freMTPL2 data), the
# chainladder-python, Stan, ONNX, pyvinecopulib, scikit-learn, MAPIE and SHAP
# generators, and the R scripts.
WEEKLY = (
    sorted(p.name for p in SCRIPTS.glob("aggregate_*.py"))
    + sorted(p.name for p in SCRIPTS.glob("scipy_*.py"))
    + sorted(p.name for p in SCRIPTS.glob("mpmath_*.py"))
    + [
        "statsmodels_glm.py",
        "statsmodels_glm_robust.py",
        "statsmodels_tweedie_profile.py",
        "stacking_weights.py",
        "stacking_grid.py",
        "numpy_allocation.py",
        "bayes_glm_grid.py",
        "credibility_bayes_grid.py",
        "compound_convolution.py",
        "hawkes_scipy.py",
        "sobol_scipy.py",
        "bonus_malus_reference.py",
    ]
)

TOLERANCE_COLUMNS = {"expected", "abs_tol", "rel_tol", "source"}


def outputs() -> dict[str, list[str]]:
    """Each generator script's reference files, from the manifest."""
    by_script: dict[str, list[str]] = {}
    for entry in tomllib.loads(MANIFEST.read_text())["file"]:
        for g in entry["generators"]:
            by_script.setdefault(g, []).append(entry["path"])
    return by_script


def dependencies(script: Path) -> list[str]:
    """The package names in a script's inline metadata, without versions."""
    block = re.search(r"^# /// script\n(.*?)^# ///$", script.read_text(), re.S | re.M)
    if not block:
        return []
    meta = tomllib.loads("\n".join(line[2:] for line in block.group(1).splitlines()))
    return [re.split(r"[=<>!~;\[ ]", d, maxsplit=1)[0] for d in meta.get("dependencies", [])]


def command(script: Path, latest: bool) -> list[str]:
    if not latest:
        return ["uv", "run", "--quiet", "--script", str(script)]
    cmd = ["uv", "run", "--quiet", "--no-project", "--isolated"]
    for d in dependencies(script):
        cmd += ["--with", d]
    return cmd + ["python", str(script)]


def rows(text: str) -> tuple[list[str], list[list[str]]]:
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    table = list(csv.reader(io.StringIO("\n".join(lines))))
    return (table[0], table[1:]) if table else ([], [])


def number(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def same(a: float, b: float, abs_tol: float, rel_tol: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= max(abs_tol, rel_tol * abs(b))


def compare_tolerance_rows(header: list[str], old: list[list[str]], new: list[list[str]]):
    """Differences and moved sources of a CSV whose rows carry their tolerance."""
    col = {c: i for i, c in enumerate(header)}
    key_cols = [i for i, c in enumerate(header) if c not in TOLERANCE_COLUMNS]

    def index(table):
        out: dict[tuple, list[list[str]]] = {}
        for r in table:
            out.setdefault(tuple(r[i] for i in key_cols), []).append(r)
        return out

    old_i, new_i = index(old), index(new)
    diffs, moved = [], set()
    for k in old_i.keys() - new_i.keys():
        diffs.append(f"row removed: {','.join(k)}")
    for k in new_i.keys() - old_i.keys():
        diffs.append(f"row added: {','.join(k)}")
    for k in old_i.keys() & new_i.keys():
        if len(old_i[k]) != len(new_i[k]):
            diffs.append(f"{','.join(k)}: {len(old_i[k])} rows, now {len(new_i[k])}")
            continue
        for o, n in zip(old_i[k], new_i[k]):
            want, got = number(o[col["expected"]]), number(n[col["expected"]])
            if want is None or got is None:
                if o[col["expected"]] != n[col["expected"]]:
                    diffs.append(f"{','.join(k)}: {o[col['expected']]} -> {n[col['expected']]}")
            else:
                at = number(o[col["abs_tol"]]) or 0.0
                rt = number(o[col["rel_tol"]]) or 0.0
                if not same(got, want, at, rt):
                    diffs.append(
                        f"{','.join(k)}: {want!r} -> {got!r} "
                        f"(abs_tol {at:g}, rel_tol {rt:g})"
                    )
            if "source" in col and o[col["source"]] != n[col["source"]]:
                moved.add(f"{o[col['source']]} -> {n[col['source']]}")
    return diffs, sorted(moved)


def compare_plain_rows(old: list[list[str]], new: list[list[str]]) -> list[str]:
    if len(old) != len(new):
        return [f"{len(old)} rows, now {len(new)}"]
    diffs = []
    for line, (o, n) in enumerate(zip(old, new), start=2):
        if len(o) != len(n):
            diffs.append(f"line {line}: {len(o)} columns, now {len(n)}")
            continue
        for a, b in zip(o, n):
            x, y = number(a), number(b)
            ok = same(y, x, 1e-12, 1e-9) if x is not None and y is not None else a == b
            if not ok:
                diffs.append(f"line {line}: {a} -> {b}")
    return diffs


def compare_input(path: Path, old: bytes, new: bytes) -> list[str]:
    """An input under validation/data/ that a generator rewrote: a CSV's
    numbers within 1e-9 relative, any other file byte for byte."""
    old, new = old.replace(b"\r\n", b"\n"), new.replace(b"\r\n", b"\n")
    if old == new:
        return []
    if path.suffix != ".csv":
        return ["an input it rewrote differs"]
    (h0, r0), (h1, r1) = rows(old.decode()), rows(new.decode())
    if h0 != h1:
        return [f"header {h0} -> {h1}"]
    return compare_plain_rows(r0, r1)


def compare(path: str, old: bytes, new: bytes) -> tuple[list[str], list[str]]:
    old, new = old.replace(b"\r\n", b"\n"), new.replace(b"\r\n", b"\n")
    if old == new:
        return [], []
    if not path.endswith(".csv"):
        return ["bytes differ"], []
    (h0, r0), (h1, r1) = rows(old.decode()), rows(new.decode())
    if h0 != h1:
        return [f"header {h0} -> {h1}"], []
    if {"expected", "abs_tol", "rel_tol"} <= set(h0):
        return compare_tolerance_rows(h0, r0, r1)
    return compare_plain_rows(r0, r1), []


def run(script_name: str, files: list[str], latest: bool) -> dict:
    script = SCRIPTS / script_name
    saved = {p: (REFERENCE / p).read_bytes() for p in files}
    inputs = {p: p.read_bytes() for p in DATA.rglob("*") if p.is_file()}
    start = time.monotonic()
    result = {"script": script_name, "files": {}, "error": None, "moved": []}
    try:
        proc = subprocess.run(
            command(script, latest),
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        if proc.returncode != 0:
            result["error"] = (proc.stderr or proc.stdout).strip().splitlines()[-20:]
        else:
            moved = set()
            for p in files:
                diffs, m = compare(p, saved[p], (REFERENCE / p).read_bytes())
                result["files"][p] = diffs
                moved.update(m)
            result["moved"] = sorted(moved)
            for p, data in inputs.items():
                diffs = compare_input(p, data, p.read_bytes())
                if diffs:
                    result["files"][p.relative_to(ROOT).as_posix()] = diffs
    except subprocess.TimeoutExpired:
        result["error"] = [f"timed out after {TIMEOUT} s"]
    finally:
        for p, data in saved.items():
            (REFERENCE / p).write_bytes(data)
        for p, data in inputs.items():
            p.write_bytes(data)
    result["seconds"] = time.monotonic() - start
    return result


def report(results: list[dict], latest: bool) -> str:
    mode = "newest releases" if latest else "pinned environments"
    bad = [r for r in results if r["error"] or any(r["files"].values())]
    out = [
        f"## Reference generators rerun ({mode})",
        "",
        f"{len(results)} generators, {len(bad)} with a difference or an error.",
        "",
    ]
    for r in results:
        status = "error" if r["error"] else ("differs" if any(r["files"].values()) else "same")
        out.append(f"- `{r['script']}`: {status} ({r['seconds']:.0f} s)")
        for m in r["moved"]:
            out.append(f"  - source moved: {m}")
        if r["error"]:
            out += ["  ```"] + [f"  {line}" for line in r["error"]] + ["  ```"]
        for p, diffs in r["files"].items():
            if diffs:
                out.append(f"  - `{p}`: {len(diffs)} difference(s)")
                out += [f"    - {d}" for d in diffs[:10]]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scripts", nargs="*", help="generators to rerun (default: the weekly set)")
    ap.add_argument("--latest", action="store_true", help="newest releases, not the pins")
    ap.add_argument("--list", action="store_true", help="print the weekly set and exit")
    ap.add_argument("--report", help="append a markdown summary to this file")
    args = ap.parse_args()
    if args.list:
        print("\n".join(n for n in WEEKLY if n in outputs()))
        return 0
    by_script = outputs()
    # A weekly-family script that writes no reference file (it prints numbers a
    # Rust test copies, as aggregate_sharpen.py does) has nothing to compare.
    names = args.scripts or [n for n in WEEKLY if n in by_script]
    missing = [n for n in names if n not in by_script]
    if missing:
        print(f"no reference file names these generators: {', '.join(missing)}", file=sys.stderr)
        return 1
    results = []
    for n in names:
        r = run(n, by_script[n], args.latest)
        status = "error" if r["error"] else ("differs" if any(r["files"].values()) else "same")
        print(f"{n}: {status} ({r['seconds']:.0f} s)", flush=True)
        results.append(r)
    text = report(results, args.latest)
    if args.report:
        with open(args.report, "a") as f:
            f.write(text)
    bad = [r for r in results if r["error"] or any(r["files"].values())]
    if bad:
        print(text)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
