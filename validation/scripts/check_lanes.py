"""Check the lane status files, docs/lanes/*.md (AGENTS.md, "Lane status files").

    python validation/scripts/check_lanes.py [docs/lanes]

The files merge with git's union driver, which keeps both sides' lines, so
two PRs that delete adjacent lines bring both deleted lines back without a
conflict. This check catches that and the other ways a union merge goes
wrong. For each file:

- it has the sections `## Next` and `## Done` and no `## Now`;
- every bullet is one physical line starting with "- " (no wrapped bullets);
- every Next bullet has an id: its first issue number (`#392`) or, when the
  bullet has no issue number, a leading `[slug]` tag;
- no Next bullet's id appears in a Done bullet (a Done bullet repeats the id
  of the Next item it finishes, so a hit is a finished item that is still
  listed in Next, or one that a union merge brought back);
- no bullet appears twice in the file.
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


def check_file(path):
    errors = []
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
            continue
        if current is None or not line.strip():
            continue
        if current == "Now":
            continue
        if current not in ("Next", "Done"):
            errors.append(f"{name}:{no}: unknown section '## {current}'; use only '## Next' and '## Done'")
            current = None
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
    if "Now" in sections:
        errors.append(f"{name}: has a '## Now' section; drop it (open PRs are in `gh pr list`)")
    for need in ("Next", "Done"):
        if need not in sections:
            errors.append(f"{name}: missing the section '## {need}'")
    done = sections.get("Done", [])
    for no, line in sections.get("Next", []):
        ident = next_id(line)
        if ident is None:
            errors.append(f"{name}:{no}: Next bullet has no id: start it with [slug] or name an issue (#N)")
            continue
        label = f"#{ident[1]}" if ident[0] == "issue" else f"[{ident[1]}]"
        for dno, dline in done:
            if done_mentions(dline, ident):
                errors.append(
                    f"{name}:{no}: finished item still in Next: {label} appears in the Done bullet "
                    f"on line {dno}; delete the Next bullet"
                )
                break
    return errors


def main(argv):
    root = pathlib.Path(argv[1] if len(argv) > 1 else "docs/lanes")
    files = sorted(root.glob("*.md"))
    if not files:
        print(f"no lane files under {root}")
        return 1
    errors = []
    for f in files:
        errors.extend(check_file(f))
    for e in errors:
        print(e)
    if errors:
        print(f"{len(errors)} problem(s) in the lane status files")
        return 1
    print(f"ok: {len(files)} lane files")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
