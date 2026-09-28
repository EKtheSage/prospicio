//! Foundation shared by every `act-*` crate: the error type and reproducible
//! random-number streams.
//!
//! See `docs/architecture.md` (Core abstractions) and `docs/design/rng.md`.

pub mod error;
pub mod rng;

pub use error::{Error, Result};
pub use rng::StreamRng;
