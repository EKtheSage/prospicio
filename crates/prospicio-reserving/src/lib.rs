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
//! with Mack's process), Mack's bootstrap of the lifetime view
//! ([`MackBootstrap::fit`]), and Clark's growth curves
//! ([`ClarkLdf`], [`ClarkCapeCod`]), and selected development patterns
//! ([`SelectedPattern`], decision 12), and a growth curve fitted to a
//! benchmark pattern ([`BenchmarkCurve`], decision 23)
//! (`docs/design/reserving-v02.md`).
//! Results are checked against R ChainLadder and chainladder-python in
//! `validation/tests/reserving*.rs`.
//!
//! With the `bayes` feature, the `bayes` module adds the Bayesian chain
//! ladder, `BayesOdp`, and Meyers' CRC, CSR and CAY models, `Meyers`,
//! sampled with NUTS (`docs/design/reserving-bayes.md`).
//!
//! With the `io` feature, the `io` module reads triangles from CSV files
//! and Excel workbooks (`Triangle::from_csv`, `read_csv`, `read_excel`,
//! `docs/design/triangle.md`, decision 6). It is off by default, so the
//! default build reads no files and builds for WASM.
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

pub mod assumptions;
pub mod backtest;
#[cfg(feature = "bayes")]
pub mod bayes;
pub mod benchmark_curve;
pub mod bootstrap;
pub mod chain_ladder;
pub mod clark;
pub mod dependence;
pub mod development;
pub mod diagnostics;
pub mod draws;
pub mod error;
pub mod excess;
pub mod expected_loss;
pub mod frame;
#[cfg(feature = "io")]
pub mod io;
pub mod latest_diagonal;
pub mod mack;
pub mod mack_bootstrap;
pub mod monthly_reading;
pub mod odp_glm;
pub mod one_year;
pub mod one_year_bootstrap;
pub mod pattern;
pub mod pattern_draws;
pub mod segments;
pub mod tail;
pub mod triangle;
pub mod view;

pub use assumptions::{
    CALENDAR_YEAR_CI, CalendarDiagonal, CalendarYearTest, DEVELOPMENT_CORRELATION_CI,
    DevelopmentCorrelationTest, DevelopmentPair, calendar_year_effect, development_correlation,
};
pub use backtest::{
    Backtest, CellForecast, GlmCandidate, METRICS, NamedModel, TriangleModel, diagonal_backtest,
};
#[cfg(feature = "bayes")]
pub use bayes::{
    BayesBf, BayesBfFit, BayesBfNormalised, BayesBfNormalisedFit, BayesOdp, BayesOdpFit,
    LayerSharePrior, Meyers, MeyersFit, MeyersModel, ShareDependence,
};
pub use benchmark_curve::{BenchmarkCurve, BenchmarkCurveFit};
pub use bootstrap::{
    OdpBootstrap, OdpBootstrapFit, OdpBootstrapFits, OdpBootstrapSegment, ProcessDistribution,
};
pub use chain_ladder::{ChainLadder, ChainLadderFit};
pub use clark::{ClarkCapeCod, ClarkFit, ClarkLdf, GrowthCurve};
pub use dependence::SegmentDependence;
pub use development::{Average, Development, DevelopmentFit, SigmaInterpolation};
pub use diagnostics::Diagnostics;
pub use draws::UltimateDraws;
pub use error::{Error, Result};
pub use excess::{Basis, ExcessPattern, Layer, SeverityByMaturity, layer_exposure, layer_share};
pub use expected_loss::{
    Apriori, Benktander, BornhuetterFerguson, CapeCod, CapeCodFit, ExpectedLoss, ExpectedLossFit,
};
pub use frame::TriangleFrame;
#[cfg(feature = "io")]
pub use io::FileLayout;
pub use latest_diagonal::{
    DiagonalAge, LatestDiagonal, LatestDiagonals, LongDiagonal, PastPatternEnd, SegmentMap,
};
pub use mack::{Mack, MackFit};
pub use mack_bootstrap::{
    MackBootstrap, MackBootstrapFit, MackBootstrapFits, MackBootstrapSegment, MackProcess,
};
pub use monthly_reading::AnnualReading;
pub use odp_glm::{
    OdpGlm, OdpGlmFit, OdpPredictionError, OdpResiduals, ResidualGroup, ResidualSummary,
    SumPredictionError,
};
pub use one_year::ClaimsDevelopmentResult;
pub use one_year_bootstrap::{OneYearFit, OneYearFits, OneYearMethod, OneYearSegment};
pub use pattern::{OneYearPattern, PatternKind, PatternWeight, SelectedPattern, UnderwritingBasis};
pub use pattern_draws::{DevelopmentPatterns, Monotone, MonthlyBasis, SegmentPatterns};
pub use prospicio_core::{Grain, Lag, Month, Period};
pub use segments::{FitTable, ReserveFit, SegmentFits};
pub use tail::{CurveShape, Tail, TailBondy, TailConstant, TailCurve, TailFit};
pub use triangle::{CalendarView, DevelopmentColumn, Diagonal, Label, Long, LongTable, Triangle};
pub use view::{SummaryRow, TriangleSummary, TriangleView};
