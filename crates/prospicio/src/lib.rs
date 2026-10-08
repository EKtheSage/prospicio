//! Actuarial and risk modeling on a Rust core.
//!
//! This crate re-exports the `prospicio-*` crates under one name, so a
//! project can depend on `prospicio` alone. Each module is the whole of the
//! crate it names:
//!
//! | Module | Crate | What |
//! |---|---|---|
//! | [`core`] | `prospicio-core` | Error type, reproducible RNG streams, calendar periods |
//! | [`math`] | `prospicio-math` | Special functions, linear algebra, root finding, quadrature, optimization |
//! | [`prob`] | `prospicio-prob` | Distributions, `PredictiveDistribution`, risk measures, copulas |
//! | [`aggregate`] | `prospicio-aggregate` | Aggregate loss (Panjer, FFT, Monte Carlo) and reinsurance |
//! | [`pricing`] | `prospicio-pricing` | Layer rating, exposure curves, tower matching |
//! | [`reserving`] | `prospicio-reserving` | Triangles, Chain Ladder, Mack, bootstrap, the one-year view |
//! | [`models`] | `prospicio-models` | The model interface: families, links, designs, metrics |
//! | [`glm`] | `prospicio-glm` | GLM, GAM, elastic net |
//! | `bayes` | `prospicio-bayes` | NUTS sampling, MCMC diagnostics (feature `bayes`) |
//! | `nn` | `prospicio-nn` | Neural networks on Burn (feature `nn`) |
//!
//! The feature `arrow` reads and writes `PredictiveDistribution` as Arrow
//! IPC files; `full` turns on every feature.

pub use prospicio_aggregate as aggregate;
#[cfg(feature = "bayes")]
pub use prospicio_bayes as bayes;
pub use prospicio_core as core;
pub use prospicio_glm as glm;
pub use prospicio_math as math;
pub use prospicio_models as models;
#[cfg(feature = "nn")]
pub use prospicio_nn as nn;
pub use prospicio_pricing as pricing;
pub use prospicio_prob as prob;
pub use prospicio_reserving as reserving;
