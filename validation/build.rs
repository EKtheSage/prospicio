//! Discovers the validation test files for the two test binaries (#825).
//!
//! Every `tests/<name>.rs` is a module of one binary instead of a binary of
//! its own, so a plain `cargo test` links two binaries here, not one per
//! file: `validation` (the files a plain `cargo test` runs) and `bayes` (the
//! Bayesian parity files, behind the `slow-bayes` feature: the names listed
//! in `Cargo.toml`'s `[package.metadata.bayes-shards]`). A new test file
//! needs no registration line: it is found here. `fremtpl2.rs` keeps a
//! binary of its own for CI's `fremtpl2` job.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// The files that are test binaries themselves, not modules.
const ENTRIES: &[&str] = &["validation", "bayes", "fremtpl2"];

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("set by cargo"));
    let tests = root.join("tests");
    let manifest = root.join("Cargo.toml");
    println!("cargo:rerun-if-changed={}", tests.display());
    println!("cargo:rerun-if-changed={}", manifest.display());
    let slow = slow_names(&fs::read_to_string(&manifest).expect("Cargo.toml"));
    let mut fast = Vec::new();
    let mut bayes = Vec::new();
    for entry in fs::read_dir(&tests).expect("tests/") {
        let path = entry.expect("a directory entry").path();
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("a UTF-8 file name")
            .to_string();
        if ENTRIES.contains(&name.as_str()) {
            continue;
        }
        if slow.contains(&name) {
            bayes.push((name, path));
        } else {
            fast.push((name, path));
        }
    }
    let out = PathBuf::from(env::var("OUT_DIR").expect("set by cargo"));
    write_modules(&out.join("validation_tests.rs"), fast);
    write_modules(&out.join("bayes_tests.rs"), bayes);
}

/// The quoted names in the `[package.metadata.bayes-shards]` table.
fn slow_names(manifest: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut inside = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') && !line.starts_with("[\"") {
            inside = line == "[package.metadata.bayes-shards]";
            continue;
        }
        if inside && !line.starts_with('#') {
            names.extend(line.split('"').skip(1).step_by(2).map(String::from));
        }
    }
    names
}

/// One `mod` line per file, sorted, by absolute path.
fn write_modules(path: &Path, mut files: Vec<(String, PathBuf)>) {
    files.sort();
    let text: String = files
        .iter()
        .map(|(name, file)| format!("#[path = {:?}]\nmod {name};\n", file.display().to_string()))
        .collect();
    fs::write(path, text).expect("OUT_DIR is writable");
}
