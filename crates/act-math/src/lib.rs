//! Numerical engine shared by every `act-*` crate.
//!
//! Phase 0 holds only the special functions the first distribution needs.
//! Linear algebra, optimization, integration, FFT and root finding join as
//! the phases that use them land (see `docs/architecture.md`).

pub mod special;
