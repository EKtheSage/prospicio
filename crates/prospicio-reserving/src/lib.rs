//! Reserving: the Triangle, deterministic and stochastic methods, and the
//! one-year view.
//!
//! v0.1 so far: the [`Triangle`] (`docs/design/triangle.md`), development
//! factors, [`ChainLadder`], [`Mack`] and the [`OdpBootstrap`]; v0.2 adds
//! [`Tail`] factors, the expected-loss family ([`ExpectedLoss`],
//! [`BornhuetterFerguson`], [`Benktander`], [`CapeCod`]) driven by an
//! exposure column, Merz and Wüthrich's one-year view,
//! [`MackFit::claims_development_result`], its simulated counterpart for
//! any method, [`OdpBootstrap::one_year`] (or [`MackBootstrap::one_year`]
//! with Mack's process), and Clark's growth curves
//! ([`ClarkLdf`], [`ClarkCapeCod`]) (`docs/design/reserving-v02.md`).
//! Results are checked against R ChainLadder and chainladder-python in
//! `validation/tests/reserving*.rs`.
//!
//! ```
//! use prospicio_reserving::{ChainLadder, DevelopmentColumn, Grain, Long, Mack, Month, Triangle};
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
//! # Ok::<(), prospicio_reserving::Error>(())
//! ```

pub mod backtest;
pub mod bootstrap;
pub mod chain_ladder;
pub mod clark;
pub mod development;
pub mod error;
pub mod expected_loss;
pub mod frame;
pub mod mack;
pub mod mack_bootstrap;
pub mod odp_glm;
pub mod one_year;
pub mod one_year_bootstrap;
pub mod segments;
pub mod tail;
pub mod triangle;
pub mod view;

pub use backtest::{
    Backtest, CellForecast, GlmCandidate, METRICS, TriangleModel, diagonal_backtest,
};
pub use bootstrap::{
    OdpBootstrap, OdpBootstrapFit, OdpBootstrapFits, OdpBootstrapSegment, ProcessDistribution,
};
pub use chain_ladder::{ChainLadder, ChainLadderFit};
pub use clark::{ClarkCapeCod, ClarkFit, ClarkLdf, GrowthCurve};
pub use development::{Average, Development, DevelopmentFit, SigmaInterpolation};
pub use error::{Error, Result};
pub use expected_loss::{
    Benktander, BornhuetterFerguson, CapeCod, CapeCodFit, ExpectedLoss, ExpectedLossFit,
};
pub use frame::TriangleFrame;
pub use mack::{Mack, MackFit};
pub use mack_bootstrap::{
    MackBootstrap, MackBootstrapFit, MackBootstrapFits, MackBootstrapSegment, MackProcess,
};
pub use odp_glm::{OdpGlm, OdpGlmFit};
pub use one_year::ClaimsDevelopmentResult;
pub use one_year_bootstrap::{OneYearFit, OneYearFits, OneYearMethod, OneYearSegment};
pub use prospicio_core::{Grain, Lag, Month, Period};
pub use segments::{FitTable, ReserveFit, SegmentFits};
pub use tail::{CurveShape, Tail, TailBondy, TailConstant, TailCurve, TailFit};
pub use triangle::{CalendarView, DevelopmentColumn, Diagonal, Label, Long, LongTable, Triangle};
pub use view::{SummaryRow, TriangleSummary, TriangleView};
