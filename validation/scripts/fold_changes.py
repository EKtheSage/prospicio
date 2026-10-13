"""Validate and fold change fragments into knowledge/log.md and CHANGELOG.md.

    python validation/scripts/fold_changes.py --check      # validate, summarise
    python validation/scripts/fold_changes.py              # fold and delete
    python validation/scripts/fold_changes.py --dry-run    # show the diff only
    python validation/scripts/fold_changes.py --self-test  # test on a fixture
    python validation/scripts/fold_changes.py --root DIR   # another repo root

Every PR adds its own small files under `changes/` instead of editing the
two shared files, which conflicted whenever two PRs ran at once. See
`changes/README.md` for the formats. Standard library only.
"""

import argparse
import datetime
import difflib
import pathlib
import re
import sys
import tempfile

EM_DASH = chr(0x2014)
KINDS = ("Creation", "Update", "Correction", "Deprecation", "Note")
NAME = r"[A-Za-z0-9][A-Za-z0-9._-]*"
KNOWLEDGE_NAME = re.compile(rf"^(\d{{4}}-\d{{2}}-\d{{2}})-({NAME})\.md$")
CHANGELOG_NAME = re.compile(rf"^({NAME})\.md$")
KNOWLEDGE_BULLET = re.compile(r"^\* \*\*(\w+)\*\*: \S")
LINK = re.compile(r"\]\(([^)\s]*)\)")
EXTERNAL = re.compile(r"^https?://")
DATE_HEADING = re.compile(r"^## (\d{4}-\d{2}-\d{2})\s*$")
KEEP = {".gitkeep"}


class Fragment:
    def __init__(self, path, kind):
        self.path = path
        self.kind = kind  # "knowledge" or "changelog"
        self.date = None
        self.lines = []  # knowledge: bullet lines
        self.sections = []  # changelog: [(name, [lines])]


def read_text(path):
    data = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in data
    return data.replace("\r\n", "\n"), crlf


def write_text(path, text, crlf):
    if crlf:
        text = text.replace("\n", "\r\n")
    path.write_bytes(text.encode("utf-8"))


def trim_blank(lines):
    lines = list(lines)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def parse_knowledge(frag, text, root, errors):
    def err(msg):
        errors.append(f"{frag.path}: {msg}")

    lines = trim_blank(text.split("\n"))
    if not lines:
        err("is empty")
        return
    out = []
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("* "):
            m = KNOWLEDGE_BULLET.match(line)
            if not m or m.group(1) not in KINDS:
                err(f"bullet must start with `* **<Kind>**:` with Kind one of {', '.join(KINDS)}: {line[:60]!r}")
        elif not line.startswith("  "):
            err(f"line is neither a `* ` bullet nor an indented continuation: {line[:60]!r}")
        elif not out:
            err("first line must be a bullet")
        out.append(line)
    for target in LINK.findall(text):
        if EXTERNAL.match(target):
            continue
        if not target.startswith("/"):
            err(f"link {target!r} must be root-absolute within knowledge/ (`/dir/name.md`) or an http(s) URL")
            continue
        path = target.split("#", 1)[0].lstrip("/")
        if not path or not (root / "knowledge" / path).is_file():
            err(f"link {target} does not resolve to a file under knowledge/")
    frag.lines = out


def parse_changelog(frag, text, errors):
    def err(msg):
        errors.append(f"{frag.path}: {msg}")

    lines = trim_blank(text.split("\n"))
    if not lines:
        err("is empty")
        return
    sections = []
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("### ") and line[4:].strip():
            sections.append((line[4:].strip(), []))
        elif line.startswith("#"):
            err(f"only `### Section` headings are allowed: {line[:60]!r}")
        elif not sections:
            err("must start with a `### Section` heading")
            return
        elif line.startswith("- "):
            sections[-1][1].append(line)
        elif line.startswith("  ") and sections[-1][1]:
            sections[-1][1].append(line)
        else:
            err(f"line is neither a `- ` bullet nor an indented continuation: {line[:60]!r}")
    for name, bullets in sections:
        if not bullets:
            err(f"section {name!r} has no bullets")
    frag.sections = sections


def load(root, errors):
    """Read and validate every fragment; returns (knowledge, changelog)."""
    base = root / "changes"
    knowledge, changelog = [], []
    for sub, kind, pattern, out in (
        ("knowledge", "knowledge", KNOWLEDGE_NAME, knowledge),
        ("changelog", "changelog", CHANGELOG_NAME, changelog),
    ):
        folder = base / sub
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.name in KEEP:
                continue
            frag = Fragment(path, kind)
            m = pattern.match(path.name)
            if not path.is_file() or not m or path.name == "README.md":
                shape = "YYYY-MM-DD-<slug>.md" if kind == "knowledge" else "<slug>.md"
                errors.append(f"{path}: file name must be {shape}")
                continue
            if kind == "knowledge":
                try:
                    date = datetime.date.fromisoformat(m.group(1))
                except ValueError:
                    errors.append(f"{path}: {m.group(1)} is not a date")
                    continue
                # One day of slack for an author whose local date is ahead of UTC.
                if date > datetime.date.today() + datetime.timedelta(days=1):
                    errors.append(f"{path}: {m.group(1)} is in the future")
                    continue
                frag.date = m.group(1)
            try:
                text, _ = read_text(path)
            except UnicodeDecodeError:
                errors.append(f"{path}: is not valid UTF-8")
                continue
            if EM_DASH in text:
                errors.append(f"{path}: contains an em dash (U+2014)")
            if kind == "knowledge":
                parse_knowledge(frag, text, root, errors)
            else:
                parse_changelog(frag, text, errors)
            out.append(frag)
    knowledge.sort(key=lambda f: f.path.name)
    changelog.sort(key=lambda f: f.path.name)
    return knowledge, changelog


def fold_log(text, knowledge):
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        lines = ["# Bundle history"]
    by_date = {}
    for frag in knowledge:
        by_date.setdefault(frag.date, []).extend(frag.lines)
    for date, bullets in by_date.items():
        headings = [(i, DATE_HEADING.match(l).group(1)) for i, l in enumerate(lines) if DATE_HEADING.match(l)]
        existing = [i for i, d in headings if d == date]
        if existing:
            i = existing[0] + 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            empty = i >= len(lines) or lines[i].startswith("#")
            lines[i:i] = bullets + ([""] if empty and i < len(lines) else [])
            continue
        older = [i for i, d in headings if d < date]
        block = [f"## {date}", "", *bullets, ""]
        if older:
            lines[older[0]:older[0]] = block
        else:
            while lines and not lines[-1].strip():
                lines.pop()
            lines.extend(["", *block[:-1]])
    return "\n".join(lines) + "\n"


def fold_changelog(text, changelog):
    lines = text.split("\n")
    start = next((i for i, l in enumerate(lines) if l.strip() == "## Unreleased"), None)
    if start is None:
        raise SystemExit("CHANGELOG.md has no `## Unreleased` section")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    body = lines[start + 1:end]
    first = next((i for i, l in enumerate(body) if l.startswith("### ")), len(body))
    preamble = trim_blank(body[:first])
    sections = []  # [name, lines]
    for line in body[first:]:
        if line.startswith("### "):
            sections.append([line[4:].strip(), []])
        else:
            sections[-1][1].append(line)
    for sec in sections:
        sec[1] = trim_blank(sec[1])
    new_names = []
    for frag in changelog:
        for name, bullets in frag.sections:
            target = next((s for s in sections if s[0] == name), None)
            if target is None:
                target = [name, []]
                sections.append(target)
                new_names.append(name)
            target[1].extend(bullets)
    out = ["## Unreleased", ""]
    if preamble:
        out += preamble + [""]
    for name, content in sections:
        out += [f"### {name}", ""]
        if content:
            out += content + [""]
    tail = lines[end:]
    if not tail and out[-1] == "":
        out.pop()
    return "\n".join(lines[:start] + out + tail), new_names


def plan(root, errors):
    """Compute the folded texts without writing. Returns a dict of changes."""
    knowledge, changelog = load(root, errors)
    if errors:
        return None
    result = {"knowledge": knowledge, "changelog": changelog, "files": {}, "new_sections": []}
    if knowledge:
        path = root / "knowledge" / "log.md"
        old, crlf = read_text(path) if path.exists() else ("", False)
        result["files"][path] = (old, fold_log(old, knowledge), crlf)
    if changelog:
        path = root / "CHANGELOG.md"
        old, crlf = read_text(path)
        new, names = fold_changelog(old, changelog)
        result["files"][path] = (old, new, crlf)
        result["new_sections"] = names
    return result


def summary(result):
    parts = [
        f"{len(result['knowledge'])} knowledge fragment(s)",
        f"{len(result['changelog'])} changelog fragment(s)",
    ]
    text = ", ".join(parts)
    if result["new_sections"]:
        text += "; new CHANGELOG section(s) will be created: " + ", ".join(result["new_sections"])
    return text


def run(root, check, dry_run):
    errors = []
    result = plan(root, errors)
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(f"{len(errors)} problem(s) in {root / 'changes'}")
    print(summary(result))
    if check:
        return
    if dry_run:
        for path, (old, new, _) in result["files"].items():
            sys.stdout.writelines(
                difflib.unified_diff(
                    old.splitlines(True), new.splitlines(True), f"a/{path.name}", f"b/{path.name}"
                )
            )
        return
    for path, (_, new, crlf) in result["files"].items():
        write_text(path, new, crlf)
    for frag in result["knowledge"] + result["changelog"]:
        frag.path.unlink()
    print(f"folded into {', '.join(p.name for p in result['files']) or 'nothing'}")


FIXTURE_LOG = """# Bundle history

## 2026-10-13

* **Note**: [old](/findings/x.md), existing entry.

## 2026-10-11

* **Creation**: [older](/findings/x.md), existing entry.
"""

FIXTURE_CHANGELOG = """# Changelog

Intro.

## Unreleased

### Reserving

- Existing reserving item
  with a continuation (#1).

### Pricing

- Existing pricing item (#2).

## 0.0.1 (2026-10-07)

### Reserving

- Released item.
"""


def self_test():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "knowledge" / "findings").mkdir(parents=True)
        (root / "knowledge" / "findings" / "x.md").write_text("x\n")
        (root / "knowledge" / "log.md").write_text(FIXTURE_LOG)
        (root / "CHANGELOG.md").write_text(FIXTURE_CHANGELOG)
        k = root / "changes" / "knowledge"
        c = root / "changes" / "changelog"
        k.mkdir(parents=True)
        c.mkdir(parents=True)
        (k / ".gitkeep").write_text("")
        (root / "changes" / "README.md").write_text("readme\n")
        (k / "2026-10-13-b-branch.md").write_text("* **Update**: [x](/findings/x.md#a), [s](https://example.org), second.\n")
        (k / "2026-10-13-a-branch.md").write_text("* **Creation**: [x](/findings/x.md), first.\n  continued.\n")
        (k / "2026-10-12-mid.md").write_text("* **Note**: a new date between two.\n")
        (k / "2026-10-14-new.md").write_text("* **Note**: a newer date.\n")
        (c / "one.md").write_text("### Pricing\n\n- Pricing two (#3).\n\n### Brand new\n\n- New item (#3).\n")
        (c / "two.md").write_text("### Reserving\n- Reserving two (#4).\n")

        errors = []
        result = plan(root, errors)
        assert not errors, errors
        assert result["new_sections"] == ["Brand new"], result["new_sections"]
        run(root, check=False, dry_run=False)

        expected_log = """# Bundle history

## 2026-10-14

* **Note**: a newer date.

## 2026-10-13

* **Creation**: [x](/findings/x.md), first.
  continued.
* **Update**: [x](/findings/x.md#a), [s](https://example.org), second.
* **Note**: [old](/findings/x.md), existing entry.

## 2026-10-12

* **Note**: a new date between two.

## 2026-10-11

* **Creation**: [older](/findings/x.md), existing entry.
"""
        expected_changelog = """# Changelog

Intro.

## Unreleased

### Reserving

- Existing reserving item
  with a continuation (#1).
- Reserving two (#4).

### Pricing

- Existing pricing item (#2).
- Pricing two (#3).

### Brand new

- New item (#3).

## 0.0.1 (2026-10-07)

### Reserving

- Released item.
"""
        assert (root / "knowledge" / "log.md").read_text() == expected_log, (root / "knowledge" / "log.md").read_text()
        assert (root / "CHANGELOG.md").read_text() == expected_changelog, (root / "CHANGELOG.md").read_text()
        assert sorted(p.name for p in k.iterdir()) == [".gitkeep"]
        assert list(c.iterdir()) == []
        assert (root / "changes" / "README.md").exists()
        errors = []
        again = plan(root, errors)
        assert not errors and not again["files"], "second fold must change nothing"

        bad = {
            (k, "2026-10-13-bad.md", "* Creation: no bold\n"): "Kind",
            (k, "2026-10-13-kind.md", "* **Wish**: not a kind\n"): "Kind",
            (k, "2026-10-13-link.md", "* **Note**: [x](/findings/nope.md), gone\n"): "resolve",
            (k, "2026-10-13-rel.md", "* **Note**: [x](../findings/x.md), relative\n"): "root-absolute",
            (k, "2026-10-13-dir.md", "* **Note**: [x](/findings), a directory\n"): "resolve",
            (k, "2026-02-31-impossible.md", "* **Note**: x\n"): "not a date",
            (k, "2999-01-01-future.md", "* **Note**: x\n"): "future",
            (k, "2026-10-13-latin1.md", "* **Note**: caf\xe9\n".encode("latin-1")): "UTF-8",
            (k, "2026-10-13-dash.md", f"* **Note**: has {EM_DASH} dash\n"): "em dash",
            (k, "2026-10-13-empty.md", "\n"): "empty",
            (k, "no-date.md", "* **Note**: x\n"): "file name",
            (c, "Bad Name.md", "### A\n\n- x\n"): "file name",
            (c, "nohead.md", "- x\n"): "### Section",
            (c, "nobullet.md", "### A\n"): "no bullets",
            (c, "h2.md", "## A\n\n- x\n"): "###",
        }
        for (folder, name, text), needle in bad.items():
            path = folder / name
            path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
            errors = []
            plan(root, errors)
            assert any(needle in e and name in e for e in errors), (name, errors)
            path.unlink()
    print("fold_changes self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="validate the fragments and print a summary")
    ap.add_argument("--dry-run", action="store_true", help="print the diff a fold would make, write nothing")
    ap.add_argument("--self-test", action="store_true", help="fold a fixture tree and assert the result")
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2],
                    help="repository root (default: this script's repository)")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.root, args.check, args.dry_run)


if __name__ == "__main__":
    main()
