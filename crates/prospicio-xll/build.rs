//! Discovers the worksheet test files for one test binary (#825).
//!
//! Every `tests/<name>.rs` is a module of the `worksheet` test binary
//! instead of a binary of its own, so `cargo test` links one binary here,
//! not one per file. A new test file needs no registration line: it is
//! found here, and it reaches the shared helpers as `use crate::common;`.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("set by cargo"));
    let tests = root.join("tests");
    println!("cargo:rerun-if-changed={}", tests.display());
    let mut files: Vec<(String, PathBuf)> = fs::read_dir(&tests)
        .expect("tests/")
        .map(|e| e.expect("a directory entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "rs"))
        .map(|p| {
            let name = p
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("a UTF-8 file name")
                .to_string();
            (name, p)
        })
        .filter(|(name, _)| name != "worksheet")
        .collect();
    files.sort();
    let text: String = files
        .iter()
        .map(|(name, file)| format!("#[path = {:?}]\nmod {name};\n", file.display().to_string()))
        .collect();
    let out = PathBuf::from(env::var("OUT_DIR").expect("set by cargo"));
    fs::write(out.join("worksheet_tests.rs"), text).expect("OUT_DIR is writable");
}
