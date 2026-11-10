//! The validation tests a plain `cargo test` runs, one module per file of
//! `tests/`, found by `build.rs` (#825): one binary links them all, where one
//! binary per file linked the dependency tree dozens of times.
//!
//! Run one file with a filter on its module: `cargo test -p
//! prospicio-validation --test validation reserving_tails::`.

include!(concat!(env!("OUT_DIR"), "/validation_tests.rs"));
