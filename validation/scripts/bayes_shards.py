#!/usr/bin/env python3
"""The Bayesian parity tests of one leg of CI's `bayes` job (#824).

validation/Cargo.toml lists the slow targets as [[test]] entries that
require the `slow-bayes` feature, and names each one's leg in
[package.metadata.bayes-shards]. This script checks that every slow target
is in exactly one leg, every name in a leg is a slow target and every leg
is one that ci.yml's `bayes` matrix runs (and the other way round), then prints
the leg's lines for $GITHUB_OUTPUT:

    tests=--test a --test b ...
    unit=true|false        (prospicio-reserving's NUTS unit tests)

    python3 validation/scripts/bayes_shards.py a     # one leg
    python3 validation/scripts/bayes_shards.py       # check, print all legs
"""

import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def legs(manifest=ROOT / "validation" / "Cargo.toml"):
    t = tomllib.loads(manifest.read_text())
    slow = [
        x["name"]
        for x in t.get("test", [])
        if "slow-bayes" in x.get("required-features", [])
    ]
    shards = t["package"]["metadata"]["bayes-shards"]
    unit = shards.get("unit-tests")
    named = {k: v for k, v in shards.items() if k != "unit-tests"}
    errors = []
    seen = {}
    for leg, names in named.items():
        for n in names:
            if n not in slow:
                errors.append(f"leg {leg} names {n}, which is not a slow-bayes [[test]]")
            if n in seen:
                errors.append(f"{n} is in legs {seen[n]} and {leg}")
            seen[n] = leg
    for n in slow:
        if n not in seen:
            errors.append(
                f"{n} requires slow-bayes but no leg of [package.metadata.bayes-shards] names it"
            )
    if unit not in named:
        errors.append(f"unit-tests = {unit!r} is not a leg")
    # The legs CI runs: the `shard:` list of the bayes job's matrix. A leg
    # named here but not run there would never run its tests.
    run = ci_legs()
    if run is None:
        errors.append("cannot find the bayes job's `shard: [...]` matrix in .github/workflows/ci.yml")
    else:
        for leg in named:
            if leg not in run:
                errors.append(f"leg {leg} is in validation/Cargo.toml but ci.yml's bayes matrix runs only {run}")
        for leg in run:
            if leg not in named:
                errors.append(f"ci.yml's bayes matrix runs leg {leg}, which validation/Cargo.toml does not define")
    return named, unit, errors


def ci_legs(workflow=ROOT / ".github" / "workflows" / "ci.yml"):
    """The legs of the bayes job's matrix, from its `shard: [a, b, c]` line."""
    text = workflow.read_text()
    job = re.search(r"^  bayes:\n(.*?)(?=^  \S)", text, re.S | re.M)
    if not job:
        return None
    m = re.search(r"^\s+shard:\s*\[([^\]]*)\]", job.group(1), re.M)
    if not m:
        return None
    return [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]


def main():
    named, unit, errors = legs()
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    wanted = sys.argv[1:] or list(named)
    for leg in wanted:
        if leg not in named:
            print(f"error: no leg {leg!r}; the legs are {', '.join(named)}", file=sys.stderr)
            sys.exit(1)
        tests = " ".join(f"--test {n}" for n in named[leg])
        if sys.argv[1:]:
            print(f"tests={tests}")
            print(f"unit={'true' if leg == unit else 'false'}")
        else:
            extra = " and the reserving NUTS unit tests" if leg == unit else ""
            print(f"{leg}: {len(named[leg])} targets{extra}")


if __name__ == "__main__":
    main()
