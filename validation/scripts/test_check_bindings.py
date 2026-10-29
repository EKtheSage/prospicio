"""Tests of check_bindings.py on a small fixture tree.

    python3 validation/scripts/test_check_bindings.py   # what CI runs
    python3 -m pytest validation/scripts/test_check_bindings.py

Standard library only (unittest), so CI needs no pytest; pytest collects
the same tests.
"""

import contextlib
import io
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import check_bindings as cb  # noqa: E402

LIB = """\
//! A library crate.
pub mod shapes;
mod hidden;
pub use hidden::Reexported;

/// Bound in both languages.
pub struct Bound;

impl Bound {
    /// A method: not counted.
    pub fn area(&self) -> f64 { let s = "pub fn fake() {}"; 0.0 }
}

/// Unbound, and on the allow-list.
pub fn allowed_fn() {}

/// Unbound, and not on the allow-list.
pub enum Missing { A, B }

/// Bound, but still on the allow-list (a stale entry).
pub fn stale_fn() {}

pub(crate) struct CrateOnly;

#[doc(hidden)]
pub struct Hidden;

const X: [u8; 2] = { [1, 2] };

#[cfg(test)]
mod tests {
    pub struct OnlyInTests;
}
"""

SHAPES = """\
pub trait Shape {
    fn area(&self) -> f64;
}

pub mod inner {
    pub fn helper<'a>(c: char) -> char { if c == '{' { '}' } else { c } }
}
"""

HIDDEN = """\
pub struct Reexported;
pub struct Unreachable;
"""

PYTHON = """\
use prospicio_demo::{Bound, Reexported, shapes::{Shape, inner}};
use pyo3::prelude::*;

#[pyfunction]
fn wrap() { inner::helper('x'); prospicio_demo::stale_fn(); }
"""

R = """\
use prospicio_demo::{Bound, Reexported, stale_fn};
use prospicio_demo::shapes::Shape;
use prospicio_demo::shapes::inner::helper;

// `Missing` named in a comment is not a binding: Missing
fn wrap() { let s = "allowed_fn"; }
"""

ALLOW = """\
[python.prospicio_demo]
"allowed_fn" = "Rust only by decision"
"stale_fn" = "stale: Python binds it now"

[r.prospicio_demo]
"allowed_fn" = "Rust only by decision"
"Missing" = "Rust only by decision"
"gone_fn" = "no longer exists"
"""


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        write(self.root, "crates/prospicio-demo/src/lib.rs", LIB)
        write(self.root, "crates/prospicio-demo/src/shapes.rs", SHAPES)
        write(self.root, "crates/prospicio-demo/src/hidden/mod.rs", HIDDEN)
        write(self.root, "crates/prospicio-python/src/lib.rs", PYTHON)
        write(self.root, "crates/prospicio-r/src/lib.rs", R)
        write(self.root, "crates/prospicio-xll/src/lib.rs", "fn udf() {}\n")
        write(self.root, "crates/prospicio/src/lib.rs", "pub use prospicio_demo as demo;\n")
        write(self.root, cb.ALLOWLIST, ALLOW)

    def tearDown(self):
        self.tmp.cleanup()

    def run_check(self):
        allow = cb.load_allowlist(self.root / cb.ALLOWLIST)
        return cb.check(self.root, allow)

    def test_public_surface(self):
        items, _, _ = self.run_check()
        self.assertEqual(list(items), ["prospicio_demo"])  # bindings and umbrella skipped
        self.assertEqual(
            sorted(items["prospicio_demo"]),
            ["Bound", "Missing", "Reexported", "allowed_fn", "shapes::Shape", "shapes::inner::helper", "stale_fn"],
        )

    def test_bound_names(self):
        _, bound, _ = self.run_check()
        for lang in ("python", "r"):
            for name in ("Bound", "Reexported", "Shape", "helper", "stale_fn"):
                self.assertIn(name, bound[lang], (lang, name))
            self.assertNotIn("Missing", bound[lang])
            self.assertNotIn("allowed_fn", bound[lang])

    def test_problems(self):
        _, _, problems = self.run_check()
        text = "\n".join(problems)
        # the missing item fails in Python, where it is not listed, with a TOML line to paste
        self.assertIn("prospicio_demo::Missing (enum) has no python binding", text)
        self.assertIn('[python.prospicio_demo]', text)
        self.assertIn('"Missing" = "unbound on', text)
        # the allowed item and the item listed for R do not fail
        self.assertNotIn("allowed_fn", text)
        self.assertNotIn("Missing (enum) has no r binding", text)
        # the stale entries fail: one now bound, one gone
        self.assertIn('[python.prospicio_demo] "stale_fn" is now bound in python', text)
        self.assertIn('[r.prospicio_demo] "gone_fn" is not a public item', text)
        self.assertEqual(len(problems), 3, text)

    def test_unknown_tables(self):
        write(self.root, cb.ALLOWLIST, ALLOW + '\n[excel.prospicio_demo]\n"Missing" = "x"\n'
              '\n[python.prospicio_nope]\n"X" = "x"\n')
        _, _, problems = self.run_check()
        text = "\n".join(problems)
        self.assertIn("unknown language table [excel]", text)
        self.assertIn("[python.prospicio_nope] is not a library crate", text)

    def test_seed_then_clean(self):
        text, errors = cb.seed(self.root, cb.load_allowlist(self.root / cb.ALLOWLIST))
        self.assertEqual(errors, [])
        write(self.root, cb.ALLOWLIST, text)
        allow = cb.load_allowlist(self.root / cb.ALLOWLIST)
        self.assertEqual(allow["r"]["prospicio_demo"]["allowed_fn"], "Rust only by decision")
        self.assertIn("Missing", allow["python"]["prospicio_demo"])
        self.assertNotIn("stale_fn", allow["python"]["prospicio_demo"])
        _, _, problems = self.run_check()
        self.assertEqual(problems, [])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cb.main(["--root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("| prospicio_demo | 7 | 5 (71%) | 5 (71%) | 2 | 2 | 0 (0%) |", out.getvalue())


if __name__ == "__main__":
    unittest.main()
