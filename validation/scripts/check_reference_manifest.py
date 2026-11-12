#!/usr/bin/env python3
"""Tie each parity reference file to its generator and hash (#844).

`validation/reference/MANIFEST.toml` has one `[[file]]` entry per file under
`validation/reference/`: its `path`, the `generators` that write it
(scripts under `validation/scripts/`, or a test under `validation/tests/`),
the `environment` it records about the packages that produced it (its own
header comment, when it has one), and the `sha256` of its bytes with line
endings as LF.

    python3 validation/scripts/check_reference_manifest.py            # check (CI runs this)
    python3 validation/scripts/check_reference_manifest.py --update F # after regenerating F
    python3 validation/scripts/check_reference_manifest.py --seed     # rebuild every entry

The check fails when a file has no entry, an entry has no file, a hash
differs (a reference edited by hand, or regenerated without `--update`),
a generator does not exist, or a Python generator has no inline script
metadata (PEP 723: its dependencies at the versions the file records, the
rest resolved as of the day the file was committed by `exclude-newer`, so
`uv run validation/scripts/<name>.py` rebuilds it). `--update` rewrites
the hash and the environment of the files named (paths relative to `validation/reference/`)
and keeps their generators. `--seed` rebuilds the manifest from the files,
taking each one's generators from the scripts that name it; its result is
reviewed by hand once, then `--update` keeps it current.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "validation" / "reference"
MANIFEST = REFERENCE / "MANIFEST.toml"
SCRIPTS = ROOT / "validation" / "scripts"
HEADER = """\
# The parity reference files, each tied to its generator and its hash (#844).
# Written by validation/scripts/check_reference_manifest.py; CI fails when a
# file under validation/reference/ has no entry, an entry has no file, a
# hash differs or a generator is missing. After regenerating a file, run
#   python3 validation/scripts/check_reference_manifest.py --update <path>
# `environment` is what the file itself records about the packages that
# produced it (its header comment), not a pin.
"""


def files() -> list[str]:
    return sorted(
        p.relative_to(REFERENCE).as_posix()
        for p in REFERENCE.rglob("*")
        if p.is_file() and p != MANIFEST
    )


def digest(path: str) -> str:
    data = (REFERENCE / path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def environment(path: str) -> str:
    """The file's leading comment lines, joined, for a text file."""
    if not path.endswith((".csv", ".txt")):
        return ""
    lines = []
    with open(REFERENCE / path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("#"):
                break
            lines.append(line.lstrip("#").strip())
    return " ".join(lines)


def generator_exists(name: str) -> bool:
    if name.startswith("validation/"):
        return (ROOT / name.split(" ", 1)[0]).exists()
    return (SCRIPTS / name).exists()


def pinned(name: str) -> bool:
    """Whether a Python generator carries PEP 723 metadata, so `uv run` rebuilds it."""
    text = (SCRIPTS / name).read_text(encoding="utf-8")
    return re.search(r"^# /// script$.*?^# dependencies = .*?^# ///$", text, re.S | re.M) is not None


def quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write(entries: list[dict]) -> None:
    out = [HEADER]
    for e in sorted(entries, key=lambda e: e["path"]):
        out.append("[[file]]")
        out.append(f"path = {quote(e['path'])}")
        out.append("generators = [" + ", ".join(quote(g) for g in e["generators"]) + "]")
        if e.get("environment"):
            out.append(f"environment = {quote(e['environment'])}")
        out.append(f"sha256 = {quote(e['sha256'])}")
        out.append("")
    MANIFEST.write_text("\n".join(out), encoding="utf-8")


def read() -> list[dict]:
    if tomllib is None:
        sys.exit("check_reference_manifest.py needs Python 3.11 or later (tomllib)")
    with open(MANIFEST, "rb") as f:
        return tomllib.load(f).get("file", [])


def seed_generators(path: str, texts: dict[str, str]) -> list[str]:
    """The scripts that write `path`, by its name near a write in each."""
    name = Path(path).name
    if path.startswith("artifacts/"):
        return ["validation/tests/artifacts.rs (artifacts::write_missing_artifacts)"]
    if name.startswith("compartmental_subannual_summary"):
        return ["compartmental_subannual_summary.py"]
    pattern = re.compile(r"(?<![\w.-])" + re.escape(name))
    named = [s for s, t in texts.items() if pattern.search(t)]
    writers = []
    for s in named:
        t = texts[s]
        for m in pattern.finditer(t):
            line = t[t.rfind("\n", 0, m.start()) + 1 : t.find("\n", m.end())]
            if re.search(r"write|open\(.*[\"']w|file\(|to_csv|\bout\s*(<-|=)|OUT|dest|writeLines|Regenerate", line):
                writers.append(s)
                break
    return sorted(writers or named)


def seed() -> None:
    texts = {
        p.name: p.read_text(errors="ignore") for p in SCRIPTS.iterdir() if p.is_file()
    }
    entries = []
    for path in files():
        entries.append(
            {
                "path": path,
                "generators": seed_generators(path, texts),
                "environment": environment(path),
                "sha256": digest(path),
            }
        )
    write(entries)
    missing = [e["path"] for e in entries if not e["generators"]]
    print(f"seeded {len(entries)} entries; no generator found for {len(missing)}: {missing}")


def update(paths: list[str]) -> None:
    entries = read()
    by_path = {e["path"]: e for e in entries}
    for p in paths:
        p = p.removeprefix("validation/reference/")
        if not (REFERENCE / p).is_file():
            sys.exit(f"no file validation/reference/{p}")
        e = by_path.get(p)
        if e is None:
            sys.exit(f"{p} has no entry: add one with its generators first")
        e["sha256"] = digest(p)
        e["environment"] = environment(p)
    write(entries)
    print(f"updated {len(paths)} entr{'y' if len(paths) == 1 else 'ies'}")


def check() -> int:
    entries = read()
    problems = []
    listed = {}
    for e in entries:
        p = e.get("path", "")
        if p in listed:
            problems.append(f"{p}: listed twice")
        listed[p] = e
        if not e.get("generators"):
            problems.append(f"{p}: no generator")
        for g in e.get("generators", []):
            if not generator_exists(g):
                problems.append(f"{p}: generator {g} does not exist")
            elif g.endswith(".py") and "/" not in g and not pinned(g):
                problems.append(
                    f"{p}: generator {g} has no inline script metadata (a `# /// script` "
                    "block with its dependencies, pinned as the file records them)"
                )
    present = set(files())
    for p in sorted(present - listed.keys()):
        problems.append(f"{p}: no entry in MANIFEST.toml (add it with its generators)")
    for p in sorted(listed.keys() - present):
        problems.append(f"{p}: in MANIFEST.toml but not under validation/reference/")
    for p in sorted(present & listed.keys()):
        if digest(p) != listed[p].get("sha256"):
            problems.append(
                f"{p}: its hash differs from MANIFEST.toml; if it was regenerated, run "
                f"check_reference_manifest.py --update {p}"
            )
    for line in problems:
        print(line)
    if problems:
        print(f"{len(problems)} problem(s) in validation/reference/MANIFEST.toml")
        return 1
    print(f"ok: {len(present)} reference files, each with its generator and hash")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--seed"]:
        seed()
    elif args[:1] == ["--update"] and len(args) > 1:
        update(args[1:])
    elif not args:
        sys.exit(check())
    else:
        sys.exit(__doc__)
