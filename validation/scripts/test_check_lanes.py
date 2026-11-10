"""Self-test of check_lanes.py on lanes written to a temporary directory.

    python3 validation/scripts/test_check_lanes.py

CI runs it before check_lanes.py on the real files (#846).
"""

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import check_lanes  # noqa: E402

LANE = """# Test lane

**Owns:** nothing

## Next

{next}
"""

DONE = """# Test lane: done

## Done

{done}
"""


def run(next_bullets, done_bullets, *, lane_extra="", done_file=True, what="errors"):
    """What check_lanes reports for one lane: its errors, its finished Next
    bullets, or its text after pruning them."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "done").mkdir()
        lane = root / "test.md"
        lane.write_text(LANE.format(next="\n".join(next_bullets)) + lane_extra)
        if done_file:
            (root / "done" / "test.md").write_text(DONE.format(done="\n".join(done_bullets)))
        if what == "finished":
            return check_lanes.finished(lane)
        if what == "pruned":
            return check_lanes.pruned(lane)
        return check_lanes.check_lane(lane)


def expect(errors, fragment, case):
    if not any(fragment in e for e in errors):
        print(f"FAIL {case}: expected an error containing {fragment!r}, got {errors}")
        return 1
    return 0


def main():
    failures = 0
    ok = run(["- #12 an open item", "- [open-slug] another"], ["- #11 a finished item"])
    if ok:
        print(f"FAIL clean lane: {ok}")
        failures += 1
    # A finished item, left in Next until the fold (or brought back by a
    # union merge), is not an error: it is listed as finished, and pruned.
    for nxt, done, label in [
        (["- #12 open", "- #11 the finished item"], ["- #11 a finished item"], "#11"),
        (["- [done-slug] finished", "- #12 open"], ["- [done-slug] finished"], "[done-slug]"),
    ]:
        if run(nxt, done):
            print(f"FAIL finished {label}: reported as an error: {run(nxt, done)}")
            failures += 1
        got = [lab for _, lab, _ in run(nxt, done, what="finished")]
        if got != [label]:
            print(f"FAIL finished {label}: listed {got}")
            failures += 1
        text, n = run(nxt, done, what="pruned")
        want = LANE.format(next="- #12 open")
        if n != 1 or text != want:
            print(f"FAIL prune {label}: removed {n}, left {text!r}")
            failures += 1
    # #1 must not match #11.
    if run(["- #1 open"], ["- #11 a finished item"], what="finished"):
        print("FAIL #1 matched #11")
        failures += 1
    failures += expect(run(["- no id here"], []), "has no id", "Next bullet without id")
    failures += expect(
        run(["- #12 open", "  wrapped line"], []), "not a bullet", "wrapped bullet"
    )
    failures += expect(run(["- #12 open"], ["- #9 a", "- #9 a"]), "duplicate", "duplicate")
    # A Done section in the imported lane file belongs in done/.
    failures += expect(
        run(["- #12 open"], [], lane_extra="\n## Done\n\n- #10 put here by an old branch\n"),
        "belongs in done/test.md",
        "Done section in the lane file",
    )
    failures += expect(run(["- #12 open"], [], done_file=False), "no ", "missing done file")
    if failures:
        print(f"{failures} self-test case(s) failed")
        return 1
    print("ok: check_lanes self-test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
