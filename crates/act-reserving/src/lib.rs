//! Reserving: the Triangle, deterministic and stochastic methods, and the
//! one-year view.
//!
//! v0.1 so far: the [`Triangle`] (`docs/design/triangle.md`), development
//! factors, [`ChainLadder`], [`Mack`] and the [`OdpBootstrap`]. Results are checked against R
//! ChainLadder and chainladder-python in `validation/tests/reserving.rs`.
//!
//! ```
//! use act_reserving::{ChainLadder, DevelopmentColumn, Grain, Long, Mack, Month, Triangle};
//!
//! let origin = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023].map(Month::january);
//! let tri = Triangle::from_long(&Long {
//!     keys: &[],
//!     origin: &origin,
//!     development: DevelopmentColumn::Age(&[12, 24, 36, 48, 12, 24, 36, 12, 24, 12]),
//!     values: &[(
//!         "paid",
//!         &[100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
//!     )],
//!     origin_grain: Grain::Year,
//!     development_grain: Grain::Year,
//!     cumulative: true,
//! })?;
//! let cl = ChainLadder::default().fit(&tri, "paid")?;
//! let mack = Mack::default().fit(&tri, "paid")?;
//! assert!(cl.total_reserve() > 0.0 && mack.total_standard_error > 0.0);
//! # Ok::<(), act_reserving::Error>(())
//! ```

pub mod bootstrap;
pub mod chain_ladder;
pub mod development;
pub mod error;
pub mod mack;
pub mod triangle;

pub use act_core::{Grain, Lag, Month, Period};
pub use bootstrap::{OdpBootstrap, OdpBootstrapFit, ProcessDistribution};
pub use chain_ladder::{ChainLadder, ChainLadderFit};
pub use development::{Average, Development, DevelopmentFit, SigmaInterpolation};
pub use error::{Error, Result};
pub use mack::{Mack, MackFit};
pub use triangle::{CalendarView, DevelopmentColumn, Diagonal, Label, Long, LongTable, Triangle};
