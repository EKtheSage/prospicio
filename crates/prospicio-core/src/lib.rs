//! Foundation shared by every `prospicio-*` crate: the error type, reproducible
//! random-number streams, and calendar periods.
//!
//! See `docs/architecture.md` (Core abstractions) and `docs/design/rng.md`.

pub mod error;
pub mod period;
pub mod rng;

pub use error::{Error, Result};
pub use period::{Grain, Lag, Month, Period};
pub use rng::StreamRng;
