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

pub mod collective;
pub mod compound;
pub mod fft;
pub mod grid_reinsurance;
pub mod layer_statistics;
pub mod moments;
pub mod monte_carlo;
pub mod panjer;
pub mod picks;
pub mod pml;
pub mod reinsurance;
pub mod ruin;
pub mod serial;
pub mod sizing;

pub use collective::CollectiveModel;
pub use compound::{CompoundMethod, CompoundReport};
pub use fft::fft;
pub use grid_reinsurance::{TowerGrids, TowerPrice, ViewPrice};
pub use layer_statistics::{
    AggregateLayerStatistics, CessionStatistics, LayerStatistics, aggregate_layer_statistics,
    occurrence_cession_statistics, occurrence_layer_statistics,
};
pub use moments::{Approximation, ApproximationKind, CompoundMoments, compound_moments};
pub use monte_carlo::{EventSet, simulate_events};
pub use panjer::panjer;
pub use picks::{PickedLayer, Picks, layer_loss_picks};
pub use pml::{OccurrencePml, occurrence_exceedance, occurrence_pml};
pub use reinsurance::{
    Basis, Commission, Corridor, Layer, LossSensitivePremium, ProfitCommission, Tower,
};
pub use ruin::{Ruin, SeverityPart, pollaczeck_khinchine};
pub use sizing::{GridSize, Sizing, SizingMethod, fft_auto, recommend_grid, round_bucket};
