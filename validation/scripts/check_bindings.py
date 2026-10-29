"""Check that the public Rust API is bound in Python and R, or allowed not to be.

    python validation/scripts/check_bindings.py            # check, print the table
    python validation/scripts/check_bindings.py --seed     # rewrite the allow-list
    python validation/scripts/check_bindings.py --root DIR # another repo root

Every lane promises Python and R parity, and until this check that promise
lived only in prose ("Rust only for now"). The check lists the public items
of the library crates, finds which of them the binding crates reach, and
compares the unbound ones with `validation/bindings-allowlist.toml`
(issue #572). It fails on:

- a public item that Python or R does not reach and that has no allow-list
  entry (bind it, or add the printed TOML line with a real reason);
- an allow-list entry whose item is now bound (delete the entry);
- an allow-list entry whose item no longer exists (delete or rename it);
- an allow-list section for an unknown language or crate.

It always prints a coverage table per crate, for the PR description.
Standard library only (Python 3.11+ for `tomllib`), no nightly toolchain.

What counts as a public item
----------------------------

The library crates are `crates/prospicio-*` except the bindings
(`prospicio-python`, `prospicio-r`, `prospicio-xll`) and the umbrella
`prospicio`, which only re-exports. The scan reads source text, not
rustdoc JSON, so it is an approximation:

- Granularity: module-level `struct`, `enum`, `union`, `trait`, `type`
  and `fn`. Methods, associated items, fields, enum variants, consts and
  statics are not counted: method names (`fit`, `new`, `mean`) are too
  generic for the name matching below, and counting them would triple the
  list. A type's binding stands in for its methods.
- Reachability: walk from `src/lib.rs` through `mod` declarations (files
  `x.rs` or `x/mod.rs`, and inline `mod x { }` blocks). An item counts
  when it is declared plain `pub` (not `pub(crate)` and the like) and
  either every module on its path is `pub mod`, or a `pub use` in a
  reachable module re-exports it (by name, through a module, or by glob),
  following chains of re-exports within the crate.
- Skipped: anything under `#[cfg(test)]`, anything `#[doc(hidden)]`, items
  that macros generate, and re-exports of other crates' items (each item
  is counted once, in the crate that declares it). Feature-gated items
  (`#[cfg(feature = ...)]`) count.
- An item's path is the shortest public path, so a type re-exported at the
  crate root is listed as `Triangle`, not `triangle::Triangle`. Two items
  with one path (alternative `cfg`s) are one item.

What counts as bound
--------------------

Name matching on the binding crates' Rust source (`crates/prospicio-python/src`
for Python, `crates/prospicio-r/src` for R). `R/prospicio/R` is not read:
it calls the extendr wrappers by their own names and never names a Rust
item. A library item is bound in a language when its name appears in that
binding crate as

- a name in a `use` tree that starts at a `prospicio_*` crate (the
  original name of an `x as y` import counts),
- a segment of an inline path starting at a `prospicio_*` crate, as in
  `prospicio_prob::evt::fit_gpd(...)`, or
- the segment after a lower-case name that such a `use` imported, as in
  `use prospicio_reserving::view;` then `view::run(...)`.

Known limits: the match is by name, not by path, so two crates' items of
one name are bound together (a false "bound"), and an item reached only
through a glob import or a type alias of the binding crate is missed (a
false "unbound", which shows up as an allow-list entry to review). A trait
that a binding imports only to call its methods counts as bound. Calling a
method on a value of a bound type does not bind other items it returns.

Excel
-----

The Excel add-in wraps a chosen subset (its lane's rule), so its coverage,
from the same scan of `crates/prospicio-xll/src`, is a report-only column of
the table: it never fails and has no allow-list section.

The allow-list
--------------

`validation/bindings-allowlist.toml` has one table per language and crate,
`[python.prospicio_prob]`, with one entry per item path and a one-line
reason: `"grid::Discretization" = "internal: ..."`. `--seed` rewrites it
from the current tree: it keeps the entries (and reasons) still needed,
drops stale ones and adds every new unbound item with a dated reason to
review. Use it to start the list or after a large change, and replace the
added reasons by real ones before committing.
"""

import argparse
import datetime
import pathlib
import re
import sys
import tomllib

LANGUAGES = {"python": "crates/prospicio-python/src", "r": "crates/prospicio-r/src"}
REPORT_ONLY = {"excel": "crates/prospicio-xll/src"}
NOT_LIBRARIES = {"prospicio", "prospicio-python", "prospicio-r", "prospicio-xll"}
ALLOWLIST = "validation/bindings-allowlist.toml"
ITEM_KEYWORDS = {"struct", "enum", "union", "trait", "type", "fn"}

# ---------------------------------------------------------------- tokens

TOKEN = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<line>//[^\n]*)
  | (?P<block>/\*)
  | (?P<raw>(?:br|cr|r)(?P<hashes>\#*)")
  | (?P<str>(?:b|c)?"(?:\\.|[^"\\])*")
  | (?P<char>b?'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|.)|[^\\'\n])')
  | (?P<life>'[A-Za-z_][A-Za-z0-9_]*)
  | (?P<ident>r\#[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*)
  | (?P<num>[0-9][0-9A-Za-z_]*(?:\.[0-9][0-9A-Za-z_]*)?)
  | (?P<path>::)
  | (?P<punct>.)
    """,
    re.VERBOSE | re.DOTALL,
)


def tokenize(text):
    """Rust source to a list of (kind, text); comments and whitespace dropped."""
    out = []
    i, n = 0, len(text)
    while i < n:
        m = TOKEN.match(text, i)
        kind = m.lastgroup
        if kind == "hashes":
            kind = "raw"
        if kind == "block":
            depth, j = 1, m.end()
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            i = j
            continue
        if kind == "raw":
            close = '"' + m.group("hashes")
            j = text.find(close, m.end())
            j = n if j < 0 else j + len(close)
            out.append(("lit", text[i:j]))
            i = j
            continue
        i = m.end()
        if kind in ("ws", "line"):
            continue
        if kind in ("str", "char", "num"):
            out.append(("lit", m.group()))
        elif kind == "ident":
            out.append(("ident", m.group().removeprefix("r#")))
        elif kind == "life":
            out.append(("life", m.group()))
        else:
            out.append(("punct", m.group()))
    return out


OPEN = {"(": ")", "[": "]", "{": "}"}


def close_of(toks, i):
    """Index just past the group that opens at toks[i]."""
    depth = 0
    while i < len(toks):
        t = toks[i][1] if toks[i][0] == "punct" else None
        if t in OPEN:
            depth += 1
        elif t in (")", "]", "}"):
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def skip_item(toks, i, through_braces):
    """Index past an item: its `;`, or its first top-level `{ }` block.

    With through_braces, braces do not end the item (a const's initializer).
    """
    while i < len(toks):
        kind, t = toks[i]
        if kind == "punct" and t == ";":
            return i + 1
        if kind == "punct" and t == "{" and not through_braces:
            return close_of(toks, i)
        if kind == "punct" and t in OPEN:
            i = close_of(toks, i)
            continue
        i += 1
    return i


# ----------------------------------------------------------- use trees


def parse_use(toks):
    """A `use` tree's tokens (without `use` and `;`) to (path, leaf, alias) triples.

    `leaf` is a name, `*` or `self`; alias is the imported name.
    """
    out = []

    def tree(i, prefix):
        path = list(prefix)
        while i < len(toks):
            kind, t = toks[i]
            if kind == "punct" and t == "{":
                end = close_of(toks, i)
                j = i + 1
                while j < end - 1:
                    j = tree(j, path)
                    if j < end - 1 and toks[j][1] == ",":
                        j += 1
                return end
            if kind == "punct" and t == "*":
                out.append((path, "*", None))
                return i + 1
            if kind == "ident":
                name = t
                if i + 1 < len(toks) and toks[i + 1][1] == "::":
                    path.append(name)
                    i += 2
                    continue
                alias = name
                i += 1
                if i + 1 < len(toks) and toks[i][1] == "as":
                    alias = toks[i + 1][1]
                    i += 2
                out.append((path, name, alias))
                return i
            if kind == "path":  # leading `::`
                i += 1
                continue
            return i + 1
        return i

    i = 0
    while i < len(toks):
        i = tree(i, [])
        while i < len(toks) and toks[i][1] == ",":
            i += 1
    return out


# ------------------------------------------------------------- modules


class Module:
    def __init__(self, path, public):
        self.path = path  # tuple of names below the crate root
        self.public = public  # declared `pub mod` (the root is public)
        self.items = {}  # name -> (kind, declared pub)
        self.children = {}  # name -> Module
        self.uses = []  # (is pub, path, leaf, alias)


def attrs_say(attrs, *needles):
    text = " ".join(attrs)
    return any(n in text for n in needles)


def parse_module(toks, module, file_dir, errors):
    """Fill `module` from the tokens of its body; `file_dir` resolves `mod x;`."""
    i = 0
    while i < len(toks):
        attrs = []
        while i < len(toks) and toks[i][1] == "#":
            j = i + 1
            if j < len(toks) and toks[j][1] == "!":
                j += 1
            end = close_of(toks, j)
            attrs.append(" ".join(t for _, t in toks[j:end]).replace(" ", ""))
            i = end
        if i >= len(toks):
            break
        hidden = attrs_say(attrs, "cfg(test)", "doc(hidden)")
        pub = restricted = False
        if toks[i][1] == "pub":
            pub = True
            i += 1
            if i < len(toks) and toks[i][1] == "(":
                restricted = True
                i = close_of(toks, i)
        while i < len(toks) and toks[i][1] in ("unsafe", "async", "default", "extern") or (
            i < len(toks) and toks[i][0] == "lit" and i and toks[i - 1][1] == "extern"
        ):
            i += 1
        if i >= len(toks):
            break
        kind, word = toks[i]
        if word == "const" and i + 1 < len(toks) and toks[i + 1][1] in ("fn", "unsafe", "async", "extern"):
            i += 1
            while toks[i][1] != "fn":
                i += 1
            kind, word = toks[i]
        open_pub = pub and not restricted
        if kind == "ident" and word == "mod":
            name = toks[i + 1][1]
            after = toks[i + 2][1] if i + 2 < len(toks) else ";"
            if after == "{":
                end = close_of(toks, i + 2)
                if not hidden:
                    child = Module(module.path + (name,), open_pub)
                    module.children[name] = child
                    parse_module(toks[i + 3:end - 1], child, file_dir / name, errors)
                i = end
            else:
                i += 3
                if hidden:
                    continue
                for cand in (file_dir / f"{name}.rs", file_dir / name / "mod.rs"):
                    if cand.exists():
                        child = Module(module.path + (name,), open_pub)
                        module.children[name] = child
                        parse_module(tokenize(cand.read_text(encoding="utf-8")), child, file_dir / name, errors)
                        break
                else:
                    errors.append(f"{file_dir}: no file for `mod {name};`")
            continue
        if kind == "ident" and word in ITEM_KEYWORDS:
            name = toks[i + 1][1] if i + 1 < len(toks) else ""
            if not hidden and toks[i + 1][0] == "ident":
                module.items.setdefault(name, (word, open_pub))
            i = skip_item(toks, i + 2, through_braces=False)
            continue
        if kind == "ident" and word in ("const", "static"):
            i = skip_item(toks, i + 1, through_braces=True)
            continue
        if kind == "ident" and word == "use":
            end = skip_item(toks, i + 1, through_braces=True)
            if not hidden:
                for path, leaf, alias in parse_use(toks[i + 1:end - 1]):
                    module.uses.append((open_pub, path, leaf, alias))
            i = end
            continue
        if kind == "ident" and word in ("impl", "macro_rules", "crate"):
            i = skip_item(toks, i + 1, through_braces=False)
            continue
        if kind == "ident" and i + 1 < len(toks) and toks[i + 1][1] == "!":
            # an item-level macro call: name! { } or name!( ); or a macro_rules
            j = i + 2
            if j < len(toks) and toks[j][0] == "ident":
                j += 1
            if j < len(toks) and toks[j][1] in OPEN:
                j = close_of(toks, j)
            if j < len(toks) and toks[j][1] == ";":
                j += 1
            i = j
            continue
        if kind == "punct" and word in OPEN:
            i = close_of(toks, i)
            continue
        i += 1


def walk(module):
    yield module
    for child in module.children.values():
        yield from walk(child)


def public_items(crate_dir, errors):
    """{(decl module path, name): (kind, shortest public path)} for one crate."""
    lib = crate_dir / "src" / "lib.rs"
    root = Module((), True)
    parse_module(tokenize(lib.read_text(encoding="utf-8")), root, lib.parent, errors)
    modules = {m.path: m for m in walk(root)}

    def lookup(mod_path, name, seen):
        """Resolve `name` in a module to ('item', decl path) or ('mod', path)."""
        mod = modules.get(mod_path)
        if mod is None or (mod_path, name) in seen:
            return None
        seen = seen | {(mod_path, name)}
        if name in mod.children:
            return ("mod", mod.children[name].path)
        if name in mod.items:
            return ("item", mod_path) if mod.items[name][1] else None
        for _, path, leaf, alias in mod.uses:
            if alias == name and leaf not in ("*", "self"):
                hit = resolve(mod_path, path + [leaf], seen)
                if hit:
                    return hit
            if leaf == "self" and path and path[-1] == name:
                hit = resolve(mod_path, path, seen)
                if hit:
                    return hit
        for _, path, leaf, _ in mod.uses:
            if leaf == "*":
                target = resolve(mod_path, path, seen)
                if target and target[0] == "mod":
                    hit = lookup(target[1], name, seen)
                    if hit:
                        return hit
        return None

    def resolve(mod_path, segments, seen=frozenset()):
        if not segments:
            return ("mod", mod_path)
        first, rest = segments[0], segments[1:]
        if first == "crate":
            base = ()
        elif first == "self":
            base = mod_path
        elif first == "super":
            base = mod_path[:-1]
        else:
            hit = lookup(mod_path, first, seen)
            if hit is None:
                return None
            if not rest:
                return hit
            if hit[0] != "mod":
                return None
            base = hit[1]
        for k, seg in enumerate(rest):
            if seg == "super":
                base = base[:-1]
                continue
            hit = lookup(base, seg, seen)
            if hit is None:
                return None
            if k == len(rest) - 1:
                return hit
            if hit[0] != "mod":
                return None
            base = hit[1]
        return ("mod", base)

    found = {}  # (decl path, name) -> (kind, public path tuple)

    def offer(decl, name, public_path):
        kind = modules[decl].items[name][0]
        old = found.get((decl, name))
        if old is None or len(public_path) < len(old[1]):
            found[(decl, name)] = (kind, public_path)

    reachable = {(): ()}  # module decl path -> its public path
    queue = [()]
    while queue:
        mod_path = queue.pop()
        mod = modules[mod_path]
        here = reachable[mod_path]
        for name, child in mod.children.items():
            if child.public and child.path not in reachable:
                reachable[child.path] = here + (name,)
                queue.append(child.path)
        for name, (_, is_pub) in mod.items.items():
            if is_pub:
                offer(mod_path, name, here + (name,))
        for is_pub, path, leaf, alias in mod.uses:
            if not is_pub:
                continue
            if leaf == "*":
                target = resolve(mod_path, path)
                if target and target[0] == "mod":
                    tmod = modules[target[1]]
                    for name, (_, item_pub) in tmod.items.items():
                        if item_pub:
                            offer(target[1], name, here + (name,))
                    for name, child in tmod.children.items():
                        if child.public and child.path not in reachable:
                            reachable[child.path] = here + (name,)
                            queue.append(child.path)
                continue
            segs = path if leaf == "self" else path + [leaf]
            target = resolve(mod_path, segs)
            if target is None:
                continue
            if target[0] == "item":
                offer(target[1], segs[-1], here + (alias,))
            elif target[1] not in reachable:
                reachable[target[1]] = here + (alias,)
                queue.append(target[1])
    return found


def library_items(root, errors):
    """{crate name (snake): {public path: kind}}."""
    out = {}
    for crate_dir in sorted((root / "crates").glob("prospicio-*")):
        if crate_dir.name in NOT_LIBRARIES or not (crate_dir / "src" / "lib.rs").exists():
            continue
        crate = crate_dir.name.replace("-", "_")
        items = {}
        for (_, name), (kind, public_path) in public_items(crate_dir, errors).items():
            items["::".join(public_path)] = (kind, name)
        out[crate] = items
    return out


# ------------------------------------------------------ binding names


def binding_names(src_dir):
    """Names of library items a binding crate's source reaches (see the docstring)."""
    names = set()
    module_aliases = set()
    for f in sorted(src_dir.rglob("*.rs")):
        toks = tokenize(f.read_text(encoding="utf-8"))
        i = 0
        while i < len(toks):
            kind, t = toks[i]
            if kind == "ident" and t == "use":
                end = skip_item(toks, i + 1, through_braces=True)
                for path, leaf, alias in parse_use(toks[i + 1:end - 1]):
                    if not path or not path[0].startswith("prospicio_"):
                        continue
                    names.update(path[1:])
                    if leaf == "self":
                        module_aliases.add(path[-1])
                    elif leaf != "*":
                        names.add(leaf)
                        if alias[:1].islower():
                            module_aliases.add(alias)
                i = end
                continue
            if kind == "ident" and t.startswith("prospicio_") and i + 1 < len(toks) and toks[i + 1][1] == "::":
                j = i + 1
                while j + 1 < len(toks) and toks[j][1] == "::" and toks[j + 1][0] == "ident":
                    names.add(toks[j + 1][1])
                    j += 2
                i = j
                continue
            i += 1
        for i in range(len(toks) - 2):
            if toks[i][1] in module_aliases and toks[i + 1][1] == "::" and toks[i + 2][0] == "ident":
                # only a module alias at the start of a path, not `x.view::<T>`
                if i == 0 or toks[i - 1][1] not in ("::", "."):
                    names.add(toks[i + 2][1])
    return names


# ---------------------------------------------------------- allow-list


def seed_reason(crate, path, kind, name):
    if kind == "trait":
        return "trait: the bindings wrap its implementors, not the trait itself (seed); review"
    if name == "Error" or name.endswith("Error"):
        return "error type: the bindings turn it into an exception or an R error (seed); review"
    if name == "Result":
        return "Result alias of the crate's error type (seed)"
    if crate == "prospicio_math":
        return "numerical building block; the bindings do not depend on prospicio-math (seed); review"
    return f"unbound on {datetime.date.today().isoformat()} (seed); review"


def quote(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


HEADER = """\
# Public Rust items that a binding deliberately does not reach (issue #572).
#
# Checked by validation/scripts/check_bindings.py on every PR: an unbound
# public item needs an entry here, and an entry whose item is now bound or
# gone fails the check, so the list cannot rot. One table per language and
# crate; one line per item path (as the script prints it) with a reason.
# Excel wraps a chosen subset and is report-only, so it has no tables here.
#
# Entries marked "(seed); review" were unbound when the list was started;
# replace the reason with a real one (internal, Rust only by decision, ...)
# or bind the item and delete the entry.
"""


def render_allowlist(entries):
    lines = [HEADER]
    for lang in sorted(entries):
        for crate in sorted(entries[lang]):
            rows = entries[lang][crate]
            if not rows:
                continue
            lines.append(f"[{lang}.{crate}]")
            for path in sorted(rows):
                lines.append(f"{quote(path)} = {quote(rows[path])}")
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- check


def analyse(root):
    errors = []
    items = library_items(root, errors)
    bound = {lang: binding_names(root / src) for lang, src in {**LANGUAGES, **REPORT_ONLY}.items()}
    return items, bound, errors


def is_bound(names, path, kind_name):
    return kind_name[1] in names


def table(items, bound, allow):
    langs = list(LANGUAGES)
    head = ["crate", "items"] + [f"{lang} bound" for lang in langs] + [f"{lang} allowed" for lang in langs]
    head += [f"{lang} bound" for lang in REPORT_ONLY]
    rows = []
    totals = [0] * (len(head) - 1)
    for crate in sorted(items):
        n = len(items[crate])
        cells = [n]
        for lang in langs:
            cells.append(sum(is_bound(bound[lang], p, k) for p, k in items[crate].items()))
        for lang in langs:
            cells.append(len(allow.get(lang, {}).get(crate, {})))
        for lang in REPORT_ONLY:
            cells.append(sum(is_bound(bound[lang], p, k) for p, k in items[crate].items()))
        totals = [a + b for a, b in zip(totals, cells)]
        rows.append([crate] + cells)
    rows.append(["total"] + totals)

    def fmt(row):
        n = row[1]
        out = [row[0], str(n)]
        for k, v in enumerate(row[2:]):
            is_allowed = 2 + k in range(2 + len(langs), 2 + 2 * len(langs))
            out.append(str(v) if is_allowed or not n else f"{v} ({100 * v / n:.0f}%)")
        return out

    text = [fmt(r) for r in rows]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in text]
    return "\n".join(lines)


def load_allowlist(path):
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def check(root, allow):
    items, bound, errors = analyse(root)
    problems = list(errors)
    for lang in allow:
        if lang not in LANGUAGES:
            problems.append(f"{ALLOWLIST}: unknown language table [{lang}] (use {', '.join(LANGUAGES)})")
            continue
        for crate in allow[lang]:
            if crate not in items:
                problems.append(f"{ALLOWLIST}: [{lang}.{crate}] is not a library crate")
    for lang in LANGUAGES:
        for crate in sorted(items):
            listed = allow.get(lang, {}).get(crate, {})
            for path, kn in sorted(items[crate].items()):
                if is_bound(bound[lang], path, kn):
                    if path in listed:
                        problems.append(
                            f"{ALLOWLIST}: [{lang}.{crate}] {quote(path)} is now bound in {lang}; delete the entry"
                        )
                elif path not in listed:
                    problems.append(
                        f"{crate}::{path} ({kn[0]}) has no {lang} binding: bind it, or add under "
                        f"[{lang}.{crate}] in {ALLOWLIST}:\n    {quote(path)} = "
                        f"{quote(seed_reason(crate, path, kn[0], kn[1]))}"
                    )
            for path in sorted(set(listed) - set(items[crate])):
                problems.append(
                    f"{ALLOWLIST}: [{lang}.{crate}] {quote(path)} is not a public item; delete or rename the entry"
                )
    return items, bound, problems


def seed(root, allow):
    items, bound, errors = analyse(root)
    entries = {}
    for lang in LANGUAGES:
        for crate in items:
            old = allow.get(lang, {}).get(crate, {})
            rows = {}
            for path, kn in items[crate].items():
                if not is_bound(bound[lang], path, kn):
                    rows[path] = old.get(path) or seed_reason(crate, path, kn[0], kn[1])
            entries.setdefault(lang, {})[crate] = rows
    return render_allowlist(entries), errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2],
                    help="repository root (default: this script's repository)")
    ap.add_argument("--seed", action="store_true",
                    help="rewrite the allow-list from the tree, keeping existing reasons")
    args = ap.parse_args(argv)
    path = args.root / ALLOWLIST
    allow = load_allowlist(path)
    if args.seed:
        text, errors = seed(args.root, allow)
        for e in errors:
            print(e)
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path}")
        return 1 if errors else 0
    items, bound, problems = check(args.root, allow)
    print(table(items, bound, allow))
    print()
    for p in problems:
        print(p)
    if problems:
        print(f"{len(problems)} problem(s) in binding coverage")
        return 1
    print(f"ok: {sum(len(v) for v in items.values())} public items in {len(items)} crates, "
          "each bound in Python and R or on the allow-list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
