//! Numerical engine shared by every `act-*` crate.
//!
//! Special functions, and the small dense linear algebra dependence models
//! need. Optimization, integration and root finding join as the phases
//! that use them land (see `docs/architecture.md`).

pub mod linalg;
pub mod special;
