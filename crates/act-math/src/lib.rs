//! Numerical engine shared by every `act-*` crate.
//!
//! Special functions, small dense linear algebra, B-splines, root finding,
//! quadrature and one-variable optimization. Domain crates call these
//! instead of writing their own (see `docs/architecture.md`).

pub mod integrate;
pub mod linalg;
pub mod optimize;
pub mod roots;
pub mod special;
pub mod spline;
