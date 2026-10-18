"""The guides in docs/guides/ run, and every number they print is checked.

Each guide's ```python blocks run in order in one namespace, from the guide's
own folder. A block followed by a ```text block must print exactly that text
(trailing spaces ignored), so a number in the prose of a guide cannot go stale
when the library changes. A block whose info string is ``python no-run`` is
shown but not run.
"""

import contextlib
import io
import re
import time
from pathlib import Path

import pytest

GUIDES = Path(__file__).resolve().parents[2] / "docs" / "guides"
GUIDE_FILES = sorted(p for p in GUIDES.glob("*.md") if p.name != "README.md")
EM_DASH = "\u2014"
SECONDS_PER_GUIDE = 20.0


def fenced_blocks(text):
    """The fenced code blocks of a Markdown text as (info, code, first line, last line)."""
    lines = text.splitlines()
    blocks, i = [], 0
    while i < len(lines):
        m = re.match(r"^```(.*)$", lines[i])
        if not m:
            i += 1
            continue
        start, body = i, []
        i += 1
        while i < len(lines) and not lines[i].startswith("```"):
            body.append(lines[i])
            i += 1
        blocks.append((m.group(1).strip(), "\n".join(body), start + 1, i + 1))
        i += 1
    return blocks, lines


def expected_output(blocks, lines, k):
    """The text block that follows block k with only blank lines between, or None."""
    if k + 1 >= len(blocks) or blocks[k + 1][0] != "text":
        return None
    between = lines[blocks[k][3] : blocks[k + 1][2] - 1]
    return blocks[k + 1][1] if all(not line.strip() for line in between) else None


def normalized(text):
    return [line.rstrip() for line in text.rstrip().splitlines()]


def run_guide(path):
    """Run a guide's python blocks; return how many ran and how many had their output checked."""
    blocks, lines = fenced_blocks(path.read_text(encoding="utf-8"))
    ns, ran, checked = {}, 0, 0
    for k, (info, code, first, _) in enumerate(blocks):
        if info != "python":
            continue
        where = f"{path.name}, python block {ran + 1} (line {first})"
        ran += 1
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                exec(compile(code, f"{path.name}:{first}", "exec"), ns)
        except Exception as e:  # report which block of which guide failed
            pytest.fail(f"{where} raised {type(e).__name__}: {e}")
        want = expected_output(blocks, lines, k)
        if want is not None:
            checked += 1
            got, want = normalized(out.getvalue()), normalized(want)
            assert got == want, (
                f"{where} prints something other than the text under it\n"
                f"--- guide\n" + "\n".join(want) + "\n--- printed\n" + "\n".join(got)
            )
    return ran, checked


def test_there_are_guides():
    assert GUIDE_FILES, f"no guides found in {GUIDES}"


@pytest.mark.parametrize("path", GUIDE_FILES, ids=lambda p: p.name)
def test_guide_runs_and_prints_what_it_shows(path, monkeypatch):
    monkeypatch.chdir(path.parent)
    start = time.perf_counter()
    ran, checked = run_guide(path)
    elapsed = time.perf_counter() - start
    assert ran > 0, f"{path.name} has no python block"
    assert checked > 0, f"{path.name} has no python block followed by the text it prints"
    assert elapsed < SECONDS_PER_GUIDE, f"{path.name} took {elapsed:.1f} s, over {SECONDS_PER_GUIDE:.0f} s"


def test_guides_have_no_em_dash_and_are_listed():
    readme = (GUIDES / "README.md").read_text(encoding="utf-8")
    for path in [GUIDES / "README.md", *GUIDE_FILES]:
        assert EM_DASH not in path.read_text(encoding="utf-8"), f"{path.name} has an em dash"
    for path in GUIDE_FILES:
        assert f"({path.name})" in readme, f"{path.name} is not listed in docs/guides/README.md"


def test_the_harness_compares_printed_output(tmp_path):
    guide = tmp_path / "g.md"
    guide.write_text(
        "```python\nx = 2\nprint(x * 3)\n```\n\n```text\n6\n```\n\n"
        "```python no-run\nraise SystemExit\n```\n\n```python\nprint(x)\n```\n\n```text\n3\n```\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match=r"python block 2 \(line 14\)"):
        run_guide(guide)


def test_chalk_portfolio_files_match_generator(tmp_path):
    # The R twins of the Monograph 16 guides read numpy's draws from examples/data/: the committed files are
    # what examples/data/make_chalk_portfolio.py writes, which is the Python guide's own simulation. Floats
    # are compared to a relative 1e-12, since numpy's normal draws go through libm, whose last digit may
    # differ between platforms; everything else must be equal.
    import importlib.util

    data = GUIDES.parents[1] / "examples" / "data"
    spec = importlib.util.spec_from_file_location("make_chalk_portfolio", data / "make_chalk_portfolio.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generator.main(tmp_path)
    for name in ["chalk_portfolio.csv", "chalk_toys.csv"]:
        made = (tmp_path / name).read_text(encoding="utf-8").splitlines()
        committed = (data / name).read_text(encoding="utf-8").splitlines()
        assert len(made) == len(committed), name
        for a, b in zip(made, committed):
            if a == b:
                continue
            fa, fb = a.split(","), b.split(",")
            assert len(fa) == len(fb) and all(
                x == y or ("." in x and float(x) == pytest.approx(float(y), rel=1e-12)) for x, y in zip(fa, fb)
            ), (name, a, b)
