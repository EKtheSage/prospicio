//! The settings enums of a compartmental fit: outstanding source, likelihood, scale, paid response, fitted and the rate order.

#[cfg(doc)]
use super::priors::{CompartmentalPriors, MAX_INCURRED_ONLY_KP_SD};
#[cfg(doc)]
use super::results::CompartmentalFit;
#[cfg(doc)]
use crate::error::Error;
/// Where the outstanding amounts come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outstanding<'a> {
    /// A column of outstanding amounts (a case reserve, or one with bulk and
    /// IBNR: the user's choice of column).
    Column(&'a str),
    /// A column of incurred amounts: outstanding is incurred less paid,
    /// cell by cell, where both are observed. Gesmann and Morris' outstanding
    /// on company 337 is incurred with bulk and IBNR less paid.
    FromIncurred(&'a str),
}

/// The likelihood of the observed amounts (decision 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompartmentalLikelihood {
    /// Normal, each response type with its own sigma: Gesmann and Morris'
    /// model 1.
    #[default]
    Normal,
    /// Log-normal, each response type with its own sigma on the log scale,
    /// `log y ~ N(log μ, σ²)`: Gesmann and Morris' model 2. Every observed
    /// response must be positive.
    LogNormal,
}

/// The scale of the response (decision 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scale {
    /// Amounts, as model 1.
    #[default]
    Amount,
    /// Loss ratios to each origin's premium, as model 2: the response and
    /// a normal likelihood's sigma are per unit of premium (a log-normal
    /// sigma is the same on either scale). Predictions and reserves stay
    /// amounts.
    LossRatio,
}

/// The paid response (decision 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaidResponse {
    /// Cumulative paid, as model 1.
    #[default]
    Cumulative,
    /// Incremental paid, as model 2: the paid of each period, whose mean is
    /// the closed form at the cell's age less the same at the age before,
    /// both with the cell's own parameters (the paper's `claimsprocess`).
    /// Each observed paid cell after the first age needs the paid of the
    /// age before.
    Incremental,
}

/// Which responses are fitted (decision 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Fitted {
    /// Paid and outstanding, as both of the paper's models.
    #[default]
    PaidAndOutstanding,
    /// Paid and incurred (decision 12): incurred replaces outstanding as
    /// the second response. Each incurred cell's mean is `RLR_i w_i` times the sum of
    /// `OS(t)` and `RRF_i PD(t)`, with the cumulative `PD` (even under incremental
    /// paid), `w_i` being the premium on the amount scale and 1 on the
    /// ratio scale. The second `σ` is the incurred response's. The second
    /// column of the fit gives the incurred cells: from an incurred
    /// column ([`Outstanding::FromIncurred`]) directly, from an outstanding
    /// column ([`Outstanding::Column`]) as outstanding plus paid where both
    /// are observed. The fit's `outstanding` distribution then holds the
    /// future incurred cells.
    PaidAndIncurred,
    /// Paid alone (decision 12), cumulative or incremental, under either
    /// likelihood and scale. It needs [`RateOrder::Ordered`] (an
    /// [`Error::NotSupported`] under [`RateOrder::Free`]): from one response the
    /// means are invariant under the rate swap, and from paid alone only
    /// `RLR × RRF` and the pair of rates are identified, so the priors on
    /// `RLR` and `RRF` carry their split. The second column of the fit is
    /// not read, and the model has one `σ` (paid's, `sigma_pd`). The fit
    /// has no outstanding distribution
    /// ([`CompartmentalFit::outstanding`] is `None`): the outstanding
    /// depends on `RLR` alone, which only its prior pins.
    ///
    /// Fit it with [`CompartmentalPriors::diffuse`] unless you hold
    /// information about the rates that is not in the data. On company 337
    /// the paid pins `k_p` (the posterior sd of its log is 5% of the diffuse
    /// prior's) and the diffuse prior gives a total reserve 2.9% above the
    /// paper's preset.
    PaidOnly,
    /// Incurred alone (decision 12): each incurred cell's mean as under
    /// [`PaidAndIncurred`](Self::PaidAndIncurred), the second column giving
    /// the incurred as there. It needs [`RateOrder::Ordered`] as
    /// [`PaidOnly`](Self::PaidOnly) does. The model has one `σ` (the
    /// incurred's, `sigma_os`). The paid column is still read, for each
    /// origin's latest paid only: the reserve is the expected paid at the
    /// last age (from the latent paid path `RLR_i RRF_i PD(t)`, which the
    /// late incurred cells pin) less the latest paid, and the future paid
    /// cells are those expected values, without observation noise, since no
    /// paid cell is fitted. The fit's `outstanding` holds the future
    /// incurred cells, drawn with the incurred `σ`.
    ///
    /// The incurred alone does not identify the payment rate `k_p`, so the
    /// paid run-off and the reserve come from its prior, at every width of
    /// it. The user's decision of 2026-10-23 is that this fit requires a
    /// prior on `k_p` the user believes: it refuses one with a log-scale sd
    /// above [`MAX_INCURRED_ONLY_KP_SD`] (0.5) with an [`Error::InvalidSetting`],
    /// before any data are read, so [`CompartmentalPriors::diffuse`] (sd 1)
    /// is refused and the paper's preset (0.1) is accepted. Take the prior
    /// from the book's paid history or from a paid-and-incurred fit of a
    /// similar book. On company 337 the reserve runs from -51,000 (sd
    /// 152,000) at a log-scale sd of 1 to 203,000 at 0.3, against 219,051
    /// under the paper's preset and 173,390 with paid and outstanding, and
    /// the fit warns, under any accepted prior, when the posterior sd of
    /// `log k_p` is more than half its prior's. Prefer a fit with paid.
    IncurredOnly,
}

/// How the two rates are kept apart (decision 2, "Identification"; open
/// decision 4, settled as (c)). The means are exactly invariant under the
/// swap `(k_er, k_p, RLR_i, RRF_i) -> (k_p, k_er, RLR_i k_er / k_p,
/// RRF_i k_p / k_er)`, so only the priors tell the two modes apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RateOrder {
    /// The paper's independent rate priors; the fit warns (in its
    /// diagnostics) when any posterior draw has `k_er <= k_p`.
    #[default]
    Free,
    /// `k_er = k_p + e^delta`: the same rate priors restricted to
    /// `k_er > k_p` (the prior on `k_er`'s `o` is kept, with the Jacobian of
    /// the map), for a user who widens the rate priors.
    Ordered,
}

/// How wide a development level is in a model with development effects
/// (model 2; `docs/design/reserving-compartmental.md` decision 14 and open
/// decision 10). A level is one effect on each of the parameters that has a
/// development effect in the priors.
///
/// The width is a modelling choice to make with `loo` on the data, not a
/// default to leave alone. On simulated books (decision 14; the study's
/// regrouping of the levels, under a short sampler): where the effects move
/// by development year, levels by year agree with one level per age on the
/// reserve (within 0.10 sd on average) and win elpd in 30 of 30 quarterly
/// books, and on 10 monthly books they took 89 s against 126 s (rough times)
/// with a worst R-hat of 1.07 against 1.62, though a median bulk ESS of 101
/// against 114; where the effects move within a year, levels by year lose
/// 361 of elpd in 30 of 30 quarterly books and widen the sd by 59%. Neither
/// is right for every book, so [`PerAge`](Self::PerAge) is the default, as
/// built. With annual development the two are the same model.
///
/// A development year is counted from each origin's own start, not by
/// calendar: the development year of an age of `a` months is
/// `(a - 1) / 12` (0-based), so ages 1 to 12 are the first year, 13 to 24
/// the second, whatever month the origin starts in, and an age such as 15
/// months (quarterly development) is in the second year. A year is a level
/// only when some observed cell has an age in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DevelopmentLevels {
    /// One level for each age of the square: the model as built on every
    /// grain, and bit for bit the fit this crate gave before the setting
    /// existed.
    #[default]
    PerAge,
    /// One level for each development year: the ages of a year share their
    /// effect in the density (a fit and its `loo`) and in the predictive,
    /// where an age with no observed cell in an observed year takes that
    /// year's effect and a year with no observed cell takes one new effect
    /// for all its ages. At the annual grain this is [`PerAge`](Self::PerAge).
    /// An [`Error::NotSupported`] under priors with no development effects
    /// (model 1), which have no levels to group.
    ByYear,
}

/// Where the sampler's pilot run starts (decision 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompartmentalStart {
    /// A least-squares fit of the population curve (shared rates and
    /// ratios, no origin effects), mapped to the intercepts, with each
    /// `log sigma` at the log of its type's residual sd; it falls back to
    /// [`PriorCentre`](Self::PriorCentre) when that fit fails.
    #[default]
    Data,
    /// Every `o` at 0 (the prior medians), each `log sigma` at the log of
    /// its type's residual sd about the prior-median curve.
    PriorCentre,
}
