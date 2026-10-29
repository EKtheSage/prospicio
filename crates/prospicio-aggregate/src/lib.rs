//! Aggregate loss and reinsurance.
//!
//! The collective (frequency-severity) model with closed-form layer
//! moments, compound distributions by Panjer's recursion, FFT and Monte
//! Carlo, and reinsurance layers and towers applied to simulated events
//! or, exactly, to the aggregate grid. See
//! `docs/design/aggregate.md`.
//!
//! This crate holds no numerics of its own beyond the aggregation
//! algorithms: distributions, grids and risk measures come from `prospicio-prob`.
//!
//! # Independence
//!
//! Panjer's recursion and the FFT ([`panjer()`], [`fft()`]) assume the
//! collective model: the claim count independent of the severities and the
//! severities independent and identically distributed. Units combined on a
//! grid, a portfolio's total as the product of its units' transforms, are
//! independent of each other too. Dependence takes another tool:
//! [`fft_bivariate`] or the clash model ([`Clash`]) for two lines hit by the
//! same events or sharing a count, one mixed count on a severity mixture for
//! components that share a mixing variable, and
//! `PredictiveDistribution::reorder_groups` (or `reorder_groups_by_copula`)
//! for a rank correlation or a copula between units' simulated totals.
//! Monte Carlo on an [`EventSet`] keeps whatever dependence its events carry.

pub mod bivariate;
pub mod cat;
pub mod collective;
pub mod compound;
pub mod deal;
pub mod fft;
pub mod grid_reinsurance;
pub mod hawkes;
pub mod layer_statistics;
pub mod moments;
pub mod monte_carlo;
pub mod panjer;
pub mod per_risk;
pub mod picks;
pub mod pml;
pub mod pnl;
pub mod reinsurance;
pub mod renewal;
pub mod ruin;
pub mod serial;
pub mod sizing;
pub mod term;
pub mod timeline;
pub mod trials;
pub mod ylt;

pub use bivariate::{BivariateReport, Clash, JointGrid, fft_bivariate};
pub use cat::EventLossTable;
pub use collective::CollectiveModel;
pub use compound::{CompoundMethod, CompoundReport};
pub use deal::{DealColumn, DealColumns, DealScale, Deduction};
pub use fft::fft;
pub use grid_reinsurance::{AnnualGrids, TowerGrids, TowerPrice, ViewPrice};
pub use hawkes::{Hawkes, HawkesFit};
pub use layer_statistics::{
    AggregateLayerStatistics, CessionStatistics, LayerStatistics, aggregate_layer_statistics,
    occurrence_cession_statistics, occurrence_layer_statistics,
};
pub use moments::{Approximation, ApproximationKind, CompoundMoments, compound_moments};
pub use monte_carlo::{EventSet, GroupLabels, PartialYears, simulate_events};
pub use panjer::panjer;
pub use per_risk::{PerRiskInuring, per_risk_inuring};
pub use picks::{PickedLayer, Picks, layer_loss_picks};
pub use pml::{OccurrencePml, occurrence_exceedance, occurrence_pml};
pub use reinsurance::{
    Basis, CarryForward, Commission, Corridor, Layer, LimitAllocation, LossSensitivePremium,
    NoClaimsBonus, ProfitCommission, Tower,
};
pub use renewal::{RenewalCount, Wait, WaitGrid, renewal_count};
pub use ruin::{
    FiniteRuin, RenewalRuin, Ruin, SeverityPart, finite_horizon, finite_horizon_poisson,
    pollaczeck_khinchine, wiener_hopf,
};
pub use sizing::{
    AuditSettings, GridAudit, GridCell, GridSize, ScoreTerms, Sizing, SizingMethod, audit_grid,
    audit_portfolio_grid, fft_auto, recommend_grid, recommend_portfolio_grid, round_bucket,
};
pub use trials::TrialStatistics;
pub use ylt::YearLossTable;
