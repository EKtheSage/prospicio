"""Check the lane status files, docs/lanes/*.md and docs/lanes/done/*.md
(AGENTS.md, "Lane status files").

    python validation/scripts/check_lanes.py [docs/lanes]

Each lane has two files: `docs/lanes/<lane>.md`, imported into every agent
session, with the section `## Next`, and `docs/lanes/done/<lane>.md`, with
the section `## Done` (#846). Both merge with git's union driver, which
keeps both sides' lines, so two PRs that delete adjacent lines bring both
deleted lines back without a conflict. This check catches that and the
other ways a union merge goes wrong. For each lane:

- the lane file has `## Next` and no `## Done` or `## Now` (a Done bullet
  goes in the done file), and the done file has `## Done` only;
- every bullet is one physical line starting with "- " (no wrapped bullets);
- every Next bullet has an id: its first issue number (`#392`) or, when the
  bullet has no issue number, a leading `[slug]` tag;
- no bullet appears twice in a file.

A Next bullet whose id appears in a Done bullet is an item that is finished
(a Done bullet repeats the id of the Next item it finishes). A pull request
that finishes an item only appends its Done bullet and leaves the Next
bullet alone, so that no two pull requests running at once delete lines
that a union merge could bring back (#853). Such bullets are listed as
"finished, to prune at the next fold", not as errors; the fold
(`fold_changes.py`, one pull request at a time) deletes them, as does

    python validation/scripts/check_lanes.py --prune [docs/lanes]
"""

import pathlib
import re
import sys

ISSUE = re.compile(r"#(\d+)(?!\d)")
SLUG = re.compile(r"^- \[([a-z0-9]+(?:-[a-z0-9]+)*)\]")


def next_id(bullet):
    """Return ('issue', n) or ('slug', s) for a Next bullet, or None."""
    m = SLUG.match(bullet)
    if m:
        return ("slug", m.group(1))
    m = ISSUE.search(bullet)
    if m:
        return ("issue", m.group(1))
    return None


def done_mentions(bullet, ident):
    kind, value = ident
    if kind == "slug":
        return f"[{value}]" in bullet
    return re.search(rf"#{value}(?!\d)", bullet) is not None


def sections_of(path, allowed, errors):
    """The bullets of each section of `path`, checking the bullets' form."""
    name = path.as_posix()
    sections = {}
    current = None
    seen = {}
    for no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.rstrip()
        if line.startswith("## "):
            current = line[3:].strip()
            if current in sections:
                errors.append(f"{name}:{no}: section '## {current}' appears twice")
            sections.setdefault(current, [])
            if current == "Now":
                errors.append(f"{name}: has a '## Now' section; drop it (open PRs are in `gh pr list`)")
            elif current not in allowed:
                where = {"Done": "done/" + path.name, "Next": "../" + path.name}.get(current)
                hint = f" (it belongs in {where})" if where else ""
                errors.append(
                    f"{name}:{no}: section '## {current}' does not belong here{hint}; "
                    f"this file has only {', '.join('## ' + a for a in allowed)}"
                )
            continue
        if current is None or not line.strip() or current not in allowed:
            continue
        if not line.startswith("- "):
            errors.append(f"{name}:{no}: not a bullet: every bullet is one line starting with '- ' (never wrapped)")
            continue
        if line in seen:
            errors.append(
                f"{name}:{no}: duplicate of the bullet on line {seen[line]} "
                "(a union merge keeps both sides; delete one)"
            )
        else:
            seen[line] = no
        sections[current].append((no, line))
    for need in allowed:
        if need not in sections:
            errors.append(f"{name}: missing the section '## {need}'")
    return sections


def lane_bullets(path, errors):
    """The Next bullets of `path` and the Done bullets of its done file."""
    done_path = path.parent / "done" / path.name
    nxt = sections_of(path, ("Next",), errors).get("Next", [])
    if done_path.exists():
        done = sections_of(done_path, ("Done",), errors).get("Done", [])
    else:
        errors.append(f"{path.as_posix()}: no {done_path.as_posix()} for the lane's Done bullets")
        done = []
    return nxt, done


def check_lane(path):
    """Check one lane: `path` (Next) and `done/<same name>` (Done)."""
    errors = []
    nxt, _ = lane_bullets(path, errors)
    for no, line in nxt:
        if next_id(line) is None:
            errors.append(
                f"{path.as_posix()}:{no}: Next bullet has no id: start it with [slug] or name an issue (#N)"
            )
    return errors


def finished(path):
    """The Next bullets of the lane whose id is in a Done bullet:
    [(line number, label, Done line number)]."""
    nxt, done = lane_bullets(path, [])
    out = []
    for no, line in nxt:
        ident = next_id(line)
        if ident is None:
            continue
        label = f"#{ident[1]}" if ident[0] == "issue" else f"[{ident[1]}]"
        for dno, dline in done:
            if done_mentions(dline, ident):
                out.append((no, label, dno))
                break
    return out


def pruned(path):
    """The lane file's text without its finished Next bullets, and how many
    were removed. Newlines are kept as the file has them."""
    gone = {no for no, _, _ in finished(path)}
    lines = path.read_bytes().decode("utf-8").splitlines(keepends=True)
    kept = [line for no, line in enumerate(lines, 1) if no not in gone]
    return "".join(kept), len(gone)


def main(argv):
    args = [a for a in argv[1:] if a != "--prune"]
    prune = len(args) < len(argv) - 1
    root = pathlib.Path(args[0] if args else "docs/lanes")
    files = sorted(root.glob("*.md"))
    if not files:
        print(f"no lane files under {root}")
        return 1
    errors = []
    for f in files:
        errors.extend(check_lane(f))
    for e in errors:
        print(e)
    if errors:
        print(f"{len(errors)} problem(s) in the lane status files")
        return 1
    if prune:
        total = 0
        for f in files:
            text, n = pruned(f)
            if n:
                f.write_bytes(text.encode("utf-8"))
                total += n
        print(f"ok: {len(files)} lane files, {total} finished Next bullet(s) pruned")
        return 0
    waiting = 0
    for f in files:
        for no, label, dno in finished(f):
            print(
                f"{f.as_posix()}:{no}: {label} is finished (Done bullet on line {dno} of "
                f"done/{f.name}), to prune at the next fold; do not start it"
            )
            waiting += 1
    tail = f", {waiting} finished Next bullet(s) to prune at the next fold" if waiting else ""
    print(f"ok: {len(files)} lane files{tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
