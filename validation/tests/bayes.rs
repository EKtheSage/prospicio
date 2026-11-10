//! The Bayesian parity tests (the `slow-bayes` feature), one module per
//! file of `tests/` named in `Cargo.toml`'s
//! `[package.metadata.bayes-shards]`, found by `build.rs` (#825). CI's
//! `bayes` job runs them in three legs, each with the filters of its
//! modules (`validation/scripts/bayes_shards.py`).
//!
//! Run one file with a filter on its module: `cargo test --release -p
//! prospicio-validation --features slow-bayes --test bayes reserving_meyers_cay::`.

include!(concat!(env!("OUT_DIR"), "/bayes_tests.rs"));
