//! Numerical engine shared by every `prospicio-*` crate.
//!
//! Special functions, small dense linear algebra, B-splines, root finding,
//! quadrature, one-variable optimization, the Nelder–Mead simplex, Sobol
//! points and the multivariate normal distribution function. Domain crates call these
//! instead of writing their own (see `docs/architecture.md`).

pub mod integrate;
pub mod linalg;
pub mod mvn;
pub mod optimize;
pub mod qmc;
pub mod roots;
mod sobol_table;
pub mod special;
pub mod spline;
