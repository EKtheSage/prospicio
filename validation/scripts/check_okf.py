"""Check the knowledge bundle against Open Knowledge Format v0.2.

    python validation/scripts/check_okf.py [knowledge]

Conformance (SPEC.md §11): every non-reserved .md file has a parseable
YAML frontmatter block with a non-empty `type`; `index.md` has no
frontmatter except `okf_version` at the bundle root; `log.md` dates are
ISO `YYYY-MM-DD` headings. On top of that, this bundle's conventions
(knowledge/index.md): every concept has `title`, `description` and
`tags`; `generated.by` and `verified[].by` follow the actor convention
(§7); timestamps are ISO 8601 with an offset; `status` is draft, stable or
deprecated; every footnote label in the body is a `sources[].id`; and every
concept is listed in its directory's `index.md`.
"""

import datetime
import pathlib
import re
import sys

import yaml

ACTOR = re.compile(r"^(human:|process:)\S+$|^[^/\s:]+/\S+$")
FOOTNOTE = re.compile(r"\[\^([^\]]+)\]")


def frontmatter(text):
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return None, text
    return yaml.safe_load(text[4:end]) or {}, text[end + 5:]


def timestamp(value):
    if isinstance(value, datetime.datetime):
        return value.tzinfo is not None
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).tzinfo is not None
    except ValueError:
        return False


def actors(entries):
    if isinstance(entries, dict):
        entries = [entries]
    return entries or []


def check_concept(path, meta, body, errors):
    def err(msg):
        errors.append(f"{path}: {msg}")

    if not isinstance(meta, dict) or not meta.get("type"):
        err("frontmatter needs a non-empty `type`")
        return
    for key in ("title", "description", "tags"):
        if not meta.get(key):
            err(f"missing `{key}`")
    if meta.get("status", "stable") not in ("draft", "stable", "deprecated"):
        err(f"status {meta['status']!r} is not draft, stable or deprecated")
    generated = meta.get("generated")
    if generated is not None:
        if not ACTOR.match(str(generated.get("by", ""))):
            err("generated.by is not an actor")
        if "at" in generated and not timestamp(generated["at"]):
            err("generated.at is not an ISO 8601 datetime with an offset")
    for v in actors(meta.get("verified")):
        if not ACTOR.match(str(v.get("by", ""))) or not timestamp(v.get("at")):
            err(f"verified entry {v} needs an actor `by` and an ISO 8601 `at`")
    if "stale_after" in meta and not timestamp(meta["stale_after"]):
        err("stale_after is not an ISO 8601 datetime with an offset")
    ids = set()
    for s in meta.get("sources") or []:
        if not s.get("resource"):
            err(f"source {s} has no `resource`")
        if s.get("id"):
            ids.add(s["id"])
    for label in set(FOOTNOTE.findall(body)):
        if label not in ids:
            err(f"footnote [^{label}] has no matching sources[].id")


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "knowledge")
    errors = []
    concepts = 0
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        meta, body = frontmatter(text)
        if path.name == "index.md":
            if meta is not None and (path.parent != root or set(meta) - {"okf_version"}):
                errors.append(f"{path}: index.md may carry only okf_version, at the bundle root")
            continue
        if path.name == "log.md":
            for heading in re.findall(r"^## (.+)$", body, re.M):
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", heading.strip()):
                    errors.append(f"{path}: log heading {heading!r} is not YYYY-MM-DD")
            continue
        concepts += 1
        if meta is None:
            errors.append(f"{path}: no YAML frontmatter")
            continue
        check_concept(path, meta, body, errors)
        index = path.parent / "index.md"
        if not index.exists() or f"({path.name})" not in index.read_text(encoding="utf-8"):
            errors.append(f"{path}: not listed in {index}")
    for e in errors:
        print(e, file=sys.stderr)
    if errors:
        sys.exit(f"{len(errors)} OKF problem(s) in {root}")
    print(f"{root}: {concepts} concepts conform to OKF v0.2")


if __name__ == "__main__":
    main()
