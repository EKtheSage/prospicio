#!/usr/bin/env python3
"""Run the R test files, several at a time, and report every failure (#849, #845).

Each test file of R/prospicio/tests/ (`test-*.R`; `helper.R` is sourced by
them) is an independent `Rscript` process run from the repository root, so
files can run side by side. This runner starts
up to `--jobs` of them at once, writes each file's output to a log of its
own, prints the file, its time and PASS or FAIL as each one finishes, and at
the end lists the slowest files, the lines of any file that start with
`SKIP` (an optional package that was not installed: `with_package()` in
`R/prospicio/tests/helper.R`), and the full log of every failed file. It exits non-zero if any file failed, after running all
of them.

    python3 validation/scripts/run_r_tests.py                  # every file, one per core
    python3 validation/scripts/run_r_tests.py --jobs 1         # one at a time
    python3 validation/scripts/run_r_tests.py --exclude test-guides.R
    python3 validation/scripts/run_r_tests.py test-guides.R    # named files only
    python3 validation/scripts/run_r_tests.py reserving        # names containing "reserving"
    python3 validation/scripts/run_r_tests.py --alone test-boosting.R
    python3 validation/scripts/run_r_tests.py --self-test

Files that run side by side get one rayon thread each
(RAYON_NUM_THREADS=1, unless it is set already), so that four files do not
start sixteen threads on four cores; no result depends on the thread count
(one RNG stream per simulation). `--alone` names files to run after the
others (ALONE below when not given), one at a time with the machine and the default thread count to
themselves: the Bayesian fits, whose NUTS chains run in parallel and which
took three times as long at one thread, and the boosting tests, whose
LightGBM and XGBoost (OpenMP) threads slowed down eightfold beside other
processes (#849). `--first` names files of the parallel batch to start
before the others.
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
# The files run alone when --alone is not given (see the module docstring).
ALONE = [
    "test-reserving-compartmental.R",
    "test-reserving-compartmental-segments.R",
    "test-reserving-pattern-draws.R",
    "test-reserving-compartmental-one-year-segments.R",
    "test-boosting.R",
]


def plan(names, tests=TESTS, exclude=(), first=(), alone=()):
    """The test files to run as (parallel, alone): the parallel batch in the
    order its files start, `first` (in the order given) then the rest by
    name, and the files of `alone` to run one at a time after it. `names`
    are file names or fragments of them; empty means every file of `tests`.
    A name that matches nothing is an error, as is an unknown name in the
    other lists."""
    every = sorted(p.name for p in tests.glob("test-*.R"))
    errors = [f"no test file {n} in {tests}" for n in [*exclude, *first, *alone] if n not in every]
    picked = []
    for n in names:
        hits = [n] if n in every else [f for f in every if n in f]
        if not hits:
            errors.append(f"no test file matches {n} in {tests}")
        picked += [f for f in hits if f not in picked]
    if errors:
        raise SystemExit("\n".join(errors))
    chosen = [n for n in (picked or every) if n not in exclude]
    tail = [n for n in alone if n in chosen]
    head = [n for n in first if n in chosen and n not in tail]
    return head + [n for n in sorted(chosen) if n not in head and n not in tail], tail


def run_one(rscript, name, tests, logs, env=None):
    log = logs / f"{name}.log"
    start = time.monotonic()
    with open(log, "wb") as out:
        code = subprocess.call(
            [rscript, str(tests / name)], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=env
        )
    return name, code, time.monotonic() - start, log


def run(order, jobs, tests=TESTS, rscript=None, stream=sys.stdout, alone=()):
    """Run `order`, `jobs` at a time, then `alone` one at a time; return the
    names that failed."""
    rscript = rscript or shutil.which("Rscript")
    if rscript is None:
        raise SystemExit("Rscript is not on the PATH")
    say = lambda s: print(s, file=stream, flush=True)
    say(f"Running {len(order)} R test files, {jobs} at a time"
        + (f", then {len(alone)} alone." if alone else "."))
    results = []

    def done(name, code, secs, log):
        text = log.read_text(encoding="utf-8", errors="replace")
        results.append((name, code, secs, text))
        say(f"{'PASS' if code == 0 else 'FAIL'} {name}: {secs:.0f} s")

    start = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        logs = pathlib.Path(tmp)
        env = dict(os.environ)
        if jobs > 1:
            env.setdefault("RAYON_NUM_THREADS", "1")
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(run_one, rscript, n, tests, logs, env) for n in order]
            for f in concurrent.futures.as_completed(futures):
                done(*f.result())
        if alone:
            say(f"Parallel batch: {time.monotonic() - start:.0f} s")
        for n in alone:
            done(*run_one(rscript, n, tests, logs))
    say(f"Total: {time.monotonic() - start:.0f} s")
    say("\nThe slowest R test files:")
    for name, _, secs, _ in sorted(results, key=lambda r: -r[2])[:10]:
        say(f"{secs:6.0f} s  {name}")
    # with_package() names its file on the SKIP line; a line printed
    # twice (a package checked in two blocks) is listed once.
    skips = list(dict.fromkeys(
        line.strip()
        for _, _, _, text in sorted(results)
        for line in text.splitlines()
        if line.startswith("SKIP")
    ))
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
    if not skips:
        say("No block was skipped.")
    return failed


def self_test():
    import io

    with tempfile.TemporaryDirectory() as tmp:
        tests = pathlib.Path(tmp)
        for name in ["test-a.R", "test-b.R", "test-c.R", "test-d.R", "helper.R"]:
            (tests / name).write_text("")
        assert plan([], tests) == (["test-a.R", "test-b.R", "test-c.R", "test-d.R"], [])
        assert plan([], tests, exclude=["test-b.R"], first=["test-d.R", "test-b.R"]) == (
            ["test-d.R", "test-a.R", "test-c.R"], [])
        assert plan(["test-c.R", "test-a.R"], tests) == (["test-a.R", "test-c.R"], [])
        assert plan(["-b", "c."], tests) == (["test-b.R", "test-c.R"], [])
        assert plan([], tests, first=["test-c.R"], alone=["test-c.R", "test-a.R"]) == (
            ["test-b.R", "test-d.R"], ["test-c.R", "test-a.R"])
        assert plan(["test-b.R"], tests, alone=["test-a.R"]) == (["test-b.R"], [])
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
            "import os\n"
            "print('SKIP test-c.R: package actuar is not installed\\n' * 2 if 'skip' in text else 'ok')\n"
            "print('threads', os.environ.get('RAYON_NUM_THREADS'))\n"
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
        os.environ.pop("RAYON_NUM_THREADS", None)
        order, alone = plan([], tests, alone=["test-d.R"])
        failed = run(order, 2, tests, str(rscript), out, alone)
        text = out.getvalue()
        # A failed file's log is printed: test-b.R ran in the batch, test-d.R alone.
        assert "===== test-b.R (exit 1) =====\nok\nthreads 1\n" in text, text
        assert "===== test-d.R (exit 1) =====\nok\nthreads None" in text, text
        assert failed == ["test-b.R", "test-d.R"], failed
        assert "PASS test-a.R" in text and "FAIL test-b.R" in text, text
        assert "===== test-b.R (exit 1) =====" in text and "===== test-d.R (exit 1) =====" in text, text
        assert "1 skipped:\n  SKIP test-c.R: package actuar is not installed\n" in text, text
        assert "2 of 4 R test files failed: test-b.R, test-d.R" in text, text
    print("run_r_tests self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="test files to run (names in R/prospicio/tests); all if none")
    ap.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--exclude", action="append", default=[], metavar="FILE")
    ap.add_argument("--first", action="append", default=[], metavar="FILE")
    ap.add_argument("--alone", action="append", default=None, metavar="FILE")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    order, alone = plan(args.files, exclude=args.exclude, first=args.first, alone=ALONE if args.alone is None else args.alone)
    failed = run(order, max(1, args.jobs), alone=alone)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
