//! JavaScript bindings over the crates in prospicio's WASM scope
//! (`prospicio-core`, `-math`, `-prob`, `-aggregate`, `-pricing` and
//! `-reserving`), built with `wasm-bindgen` for a browser page, Node or an
//! Office.js Excel add-in, with no server.
//!
//! Thin wrappers only: every number comes from the Rust core, as in the
//! Python, R and Excel front ends. One module file per lane.
//!
//! Conventions of the JavaScript API:
//!
//! - Names are camelCase (`chainLadder`, `Triangle.fromLong`). The columns
//!   of a result table keep the core's names (`standard_error`), which are
//!   the column names of the Python and R data frames too.
//! - Numeric arrays come back as `Float64Array`; either a plain array or a
//!   typed array is accepted going in.
//! - A table is an object of columns, one array per column, each with one
//!   value per row.
//! - An error from the core is thrown as a JavaScript `Error` whose
//!   `message` is the core's message.
//! - Simulations run on one thread and are reproducible by `seed`, with
//!   the same draws as the other front ends.
//!
//! `cargo xtask wasm` builds the npm package and runs its tests under Node.

use wasm_bindgen::prelude::*;

mod aggregate;
mod convert;
mod predictive;
mod pricing;
mod prob;
mod reserving;
pub use aggregate::{Count, EventSet, Grid, Tower, fft, occurrence_pml, oep, simulate_events};
pub use predictive::Predictive;
pub use pricing::{Distortion, price, price_at_cost_of_capital};

pub use prob::Dist;
pub use reserving::{Triangle, bornhuetter_ferguson, chain_ladder, mack, odp_bootstrap};

/// The version of the prospicio crates this module was built from.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}
