#!/usr/bin/env python3
"""Run the R test files, several at a time, and report every failure (#849, #845).

Each file of R/prospicio/tests/ is an independent `Rscript` process run
from the repository root, so files can run side by side. This runner starts
up to `--jobs` of them at once, writes each file's output to a log of its
own, prints the file, its time and PASS or FAIL as each one finishes, and at
the end lists the slowest files, the lines of any file that start with
`SKIP` (an optional package that was not installed), and the full log of
every failed file. It exits non-zero if any file failed, after running all
of them.

    python3 validation/scripts/run_r_tests.py                  # every file, one per core
    python3 validation/scripts/run_r_tests.py --jobs 1         # one at a time
    python3 validation/scripts/run_r_tests.py --exclude test-guides.R
    python3 validation/scripts/run_r_tests.py test-guides.R    # named files only
    python3 validation/scripts/run_r_tests.py --self-test

`--first` names files to start before the others: the slowest files, so
that the longest one does not start last and set the step's time alone.
CI's r job and `cargo xtask r` both run the tests through this script.
"""

import argparse
import concurrent.futures
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
TESTS = ROOT / "R" / "prospicio" / "tests"


def plan(names, tests=TESTS, exclude=(), first=()):
    """The test files to run, in the order they start: `first` (in the order
    given), then the rest by name. `names` empty means every file of
    `tests`. Unknown names in any list are an error."""
    every = sorted(p.name for p in tests.glob("*.R"))
    errors = [f"no test file {n} in {tests}" for n in [*names, *exclude, *first] if n not in every]
    if errors:
        raise SystemExit("\n".join(errors))
    chosen = [n for n in (names or every) if n not in exclude]
    head = [n for n in first if n in chosen]
    return head + [n for n in sorted(chosen) if n not in head]


def run_one(rscript, name, tests, logs):
    log = logs / f"{name}.log"
    start = time.monotonic()
    with open(log, "wb") as out:
        code = subprocess.call(
            [rscript, str(tests / name)], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT
        )
    return name, code, time.monotonic() - start, log


def run(order, jobs, tests=TESTS, rscript=None, stream=sys.stdout):
    """Run `order`, `jobs` at a time; return the names that failed."""
    rscript = rscript or shutil.which("Rscript")
    if rscript is None:
        raise SystemExit("Rscript is not on the PATH")
    say = lambda s: print(s, file=stream, flush=True)
    say(f"Running {len(order)} R test files, {jobs} at a time.")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        logs = pathlib.Path(tmp)
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(run_one, rscript, n, tests, logs) for n in order]
            for f in concurrent.futures.as_completed(futures):
                name, code, secs, log = f.result()
                text = log.read_text(encoding="utf-8", errors="replace")
                results.append((name, code, secs, text))
                say(f"{'PASS' if code == 0 else 'FAIL'} {name}: {secs:.0f} s")
    say("\nThe slowest R test files:")
    for name, _, secs, _ in sorted(results, key=lambda r: -r[2])[:10]:
        say(f"{secs:6.0f} s  {name}")
    skips = [
        f"{name}: {line.strip()}"
        for name, _, _, text in sorted(results)
        for line in text.splitlines()
        if line.startswith("SKIP")
    ]
    if skips:
        say(f"\n{len(skips)} skipped:")
        for s in skips:
            say(f"  {s}")
    failed = sorted(name for name, code, _, _ in results if code != 0)
    for name, code, _, text in sorted(results):
        if code != 0:
            say(f"\n===== {name} (exit {code}) =====\n{text.rstrip()}")
    if failed:
        say(f"\n{len(failed)} of {len(results)} R test files failed: {', '.join(failed)}")
    else:
        say(f"\nAll {len(results)} R test files passed.")
    return failed


def self_test():
    import io

    with tempfile.TemporaryDirectory() as tmp:
        tests = pathlib.Path(tmp)
        for name in ["test-a.R", "test-b.R", "test-c.R", "test-d.R"]:
            (tests / name).write_text("")
        assert plan([], tests) == ["test-a.R", "test-b.R", "test-c.R", "test-d.R"]
        assert plan([], tests, exclude=["test-b.R"], first=["test-d.R", "test-b.R"]) == [
            "test-d.R", "test-a.R", "test-c.R"]
        assert plan(["test-c.R", "test-a.R"], tests) == ["test-a.R", "test-c.R"]
        try:
            plan(["test-x.R"], tests)
            raise AssertionError("an unknown file must be an error")
        except SystemExit as e:
            assert "test-x.R" in str(e)
        # A fake Rscript: fails on files whose text says so, prints a skip.
        fake = tests / "fake_rscript.py"
        fake.write_text(
            "import sys\n"
            "text = open(sys.argv[1]).read()\n"
            "print('SKIP actuar not installed' if 'skip' in text else 'ok')\n"
            "sys.exit(1 if 'boom' in text else 0)\n"
        )
        (tests / "test-b.R").write_text("boom")
        (tests / "test-c.R").write_text("skip")
        (tests / "test-d.R").write_text("boom")
        if os.name == "nt":
            rscript = tests / "fake.bat"
            rscript.write_text(f'@"{sys.executable}" "{fake}" %*\n')
        else:
            rscript = tests / "fake.sh"
            rscript.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
            rscript.chmod(0o755)
        out = io.StringIO()
        failed = run(plan([], tests), 2, tests, str(rscript), out)
        text = out.getvalue()
        assert failed == ["test-b.R", "test-d.R"], failed
        assert "PASS test-a.R" in text and "FAIL test-b.R" in text, text
        assert "===== test-b.R (exit 1) =====" in text and "===== test-d.R (exit 1) =====" in text, text
        assert "test-c.R: SKIP actuar not installed" in text, text
        assert "2 of 4 R test files failed: test-b.R, test-d.R" in text, text
    print("run_r_tests self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="test files to run (names in R/prospicio/tests); all if none")
    ap.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--exclude", action="append", default=[], metavar="FILE")
    ap.add_argument("--first", action="append", default=[], metavar="FILE")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    failed = run(plan(args.files, exclude=args.exclude, first=args.first), max(1, args.jobs))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
