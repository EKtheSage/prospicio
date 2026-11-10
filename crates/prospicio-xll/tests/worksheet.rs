//! The worksheet functions' tests, one module per file of `tests/`, found
//! by `build.rs` (#825): one binary links them all, where one binary per
//! file linked the dependency tree 49 times. Each file reaches the shared
//! helpers as `use crate::common;`.
//!
//! Run one file with a filter on its module: `cargo test -p prospicio-xll
//! --test worksheet triangles::`.

mod common;

include!(concat!(env!("OUT_DIR"), "/worksheet_tests.rs"));
