//! A layer's contract terms: commissions, loss-sensitive premium,
//! carry-forwards, corridors, the limit allocation and the basis.

use super::*;

/// An nth-event cover (#591): only the `n`-th and later events in a year
/// whose loss to the layer exceeds `threshold` count; the earlier ones
/// cede nothing. Events are counted in the order given, which is
/// chronological when the event set has times.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NthEvent {
    /// The first event covered, 1 for every event, 2 for a second-event
    /// cover.
    pub n: u32,
    /// The loss an event must exceed to count, at least 0.
    pub threshold: f64,
}

/// A top-and-drop cover (#591): the layer sits at its own attachment until
/// the annual limit of the layer named `underlying` (in the same stage,
/// earlier in the tower) is used up, and from the next event attaches at
/// `attachment` instead, with the same occurrence and annual limits.
#[derive(Debug, Clone, PartialEq)]
pub struct DropDown {
    /// The layer whose exhaustion drops this one.
    pub underlying: String,
    /// The attachment after the drop, below the layer's own.
    pub attachment: f64,
}

/// The events a layer covers, by one label's values (#584): a windstorm
/// treaty is scoped to `peril` in `["windstorm"]`. See [`Layer::scope`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scope {
    /// The label dimension, as in [`EventSet::with_labels`].
    pub label: String,
    /// The values the layer sees.
    pub values: Vec<String>,
}

/// A premium that depends on the year's loss `x`:
/// `clip(basic + lcm × x, minimum, maximum)`.
///
/// It is the premium of a swing-rated reinsurance layer (with `x` the
/// layer's ceded loss; see [`Layer::swing_rated`]) and of a
/// retrospectively rated policy (with `x` the account's loss, net of any
/// reinsurance that inures to it). `lcm` is the loss conversion factor,
/// often written as a loading such as `100/80` and, for a retro plan,
/// multiplied by the tax multiplier along with `basic`. As in `aggregate`,
/// the minimum defaults to `basic` and the maximum to no cap.
///
/// ```
/// use prospicio_aggregate::LossSensitivePremium;
///
/// // aggregate's retro example: basic 1000, factor 1.1, maximum 2500.
/// let retro = LossSensitivePremium::new(1000.0, 1.1, None, Some(2500.0)).unwrap();
/// assert_eq!(retro.premium(0.0), 1000.0);
/// assert!((retro.premium(500.0) - 1550.0).abs() < 1e-9);
/// assert_eq!(retro.premium(1500.0), 2500.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossSensitivePremium {
    pub basic: f64,
    /// Loss conversion factor applied to the loss.
    pub lcm: f64,
    pub minimum: f64,
    /// May be infinite (no cap).
    pub maximum: f64,
}

impl LossSensitivePremium {
    /// `minimum` defaults to `basic`, `maximum` to infinity. Fails unless
    /// `basic`, `lcm` and the minimum are finite and non-negative and the
    /// minimum is at most the maximum.
    pub fn new(basic: f64, lcm: f64, minimum: Option<f64>, maximum: Option<f64>) -> Result<Self> {
        if !(basic.is_finite() && basic >= 0.0) {
            return Err(invalid("basic", basic, "must be finite and non-negative"));
        }
        if !(lcm.is_finite() && lcm >= 0.0) {
            return Err(invalid("lcm", lcm, "must be finite and non-negative"));
        }
        let minimum = minimum.unwrap_or(basic);
        let maximum = maximum.unwrap_or(f64::INFINITY);
        if !(minimum.is_finite() && minimum >= 0.0) {
            return Err(invalid(
                "minimum",
                minimum,
                "must be finite and non-negative",
            ));
        }
        if maximum.is_nan() || maximum < minimum {
            return Err(invalid("maximum", maximum, "must be at least the minimum"));
        }
        Ok(Self {
            basic,
            lcm,
            minimum,
            maximum,
        })
    }

    /// The premium for a loss `x`.
    pub fn premium(&self, x: f64) -> f64 {
        (self.basic + self.lcm * x).clamp(self.minimum, self.maximum)
    }

    /// The terms for a placed share `s`: the currency amounts scaled by
    /// `s`, the factor kept (it applies to a loss that is already at the
    /// placed share).
    pub(super) fn placed(&self, s: f64) -> Self {
        Self {
            basic: s * self.basic,
            lcm: self.lcm,
            minimum: s * self.minimum,
            maximum: s * self.maximum,
        }
    }
}

/// A ceding commission, as a fraction of the year's premium.
#[derive(Debug, Clone, PartialEq)]
pub enum Commission {
    /// The same rate whatever the losses.
    Flat(f64),
    /// A sliding scale: `(commission, loss_ratio)` anchors in increasing
    /// loss ratio, the commission linear between them and flat beyond the
    /// first and last (the maximum and minimum commission).
    SlidingScale(Vec<(f64, f64)>),
}

impl Commission {
    /// The commission rate at ceded loss ratio `lr`.
    pub fn rate(&self, lr: f64) -> f64 {
        match self {
            Commission::Flat(c) => *c,
            Commission::SlidingScale(anchors) => {
                let (first, last) = (anchors[0], anchors[anchors.len() - 1]);
                if lr <= first.1 {
                    return first.0;
                }
                if lr >= last.1 {
                    return last.0;
                }
                let k = anchors.partition_point(|a| a.1 <= lr);
                let ((c0, l0), (c1, l1)) = (anchors[k - 1], anchors[k]);
                c0 + (c1 - c0) * (lr - l0) / (l1 - l0)
            }
        }
    }
}

/// A profit commission: `share` of the reinsurer's profit after an
/// expense allowance, `share × max(1 − LR − allowance, 0)` of the year's
/// premium, with `LR` the ceded loss ratio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfitCommission {
    pub share: f64,
    /// The reinsurer's allowance for its expenses and margin, as a
    /// fraction of premium; it includes any ceding commission the contract
    /// deducts before profit.
    pub allowance: f64,
}

/// A no-claims bonus (#592, `docs/design/aggregate.md`, "No-claims
/// bonus"): `share` of the year's premium returned to the cedant when the
/// layer's ceded loss over the last `years` contract years, the year
/// itself included, is at most `threshold`. Within a term the window is
/// cut at the term's start, so a claim-free first year pays; a single year
/// takes `years = 1` only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoClaimsBonus {
    /// The share of the year's premium returned, in `(0, 1]`.
    pub share: f64,
    /// The contract years looked back over, at least 1.
    pub years: u32,
    /// The ceded loss over those years that still counts as no claims, at
    /// least 0.
    pub threshold: f64,
}

/// What a profit commission or sliding scale carries from one year of a
/// contract term into the next ([`Tower::apply_term`]).
///
/// A *deficit* is the part of a year's result the term could not absorb:
/// for a profit commission the loss above the premium after the allowance,
/// `ceded − premium × (1 − allowance)` when positive; for a sliding scale
/// the ceded loss above the loss ratio of the last anchor (where the
/// commission reaches its minimum), `ceded − lr_max × premium`. A *credit*
/// (sliding scales only) is the ceded loss below the loss ratio of the
/// first anchor (where the commission reaches its maximum), carried as a
/// negative amount. The next year adds what was carried to its own ceded
/// loss before the commission is figured, and carries forward whatever is
/// then still beyond the band.
///
/// Amounts are used up oldest first, so what is carried out of a year is
/// the newest part. With `years = Some(n)`, an amount from year `t` can be
/// used in years `t + 1` to `t + n` and is then dropped; `None` carries it
/// until it is used up (to extinction).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CarryForward {
    /// Carry deficits forward.
    pub deficit: bool,
    /// Carry credits forward (sliding scales only).
    pub credit: bool,
    /// How many later years an amount may be carried into; `None` without
    /// limit.
    pub years: Option<u32>,
}

/// A loss corridor: of the annual layer loss at 100% after the annual
/// deductible, the cedant keeps `retained` of the part between `lower`
/// and `upper`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Corridor {
    pub lower: f64,
    pub upper: f64,
    /// Share of the band the cedant keeps, in `(0, 1]`.
    pub retained: f64,
}

impl Corridor {
    /// The part of an annual layer loss `x` the cedant keeps.
    pub(super) fn kept(&self, x: f64) -> f64 {
        self.retained * (x - self.lower).clamp(0.0, self.upper - self.lower)
    }
}

/// How a layer's annual terms (the deductible, a corridor, the annual
/// limit) are shared among the year's events when they bite. The annual
/// ceded loss is the same either way; only the split by event differs, and
/// with it what a later inuring stage sees of each event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LimitAllocation {
    /// The events in the order given, taken as chronological: the
    /// deductible absorbs the first recoveries and the annual limit cuts
    /// the last.
    #[default]
    EventOrder,
    /// Every event of the year scaled by the same factor: event `e` with
    /// recovery `u_e` (after the per-occurrence terms) cedes
    /// `share × u_e × A(U) / U`, where `U = Σ u_e` and `A` applies the
    /// annual terms. With no deductible or corridor the factor is
    /// `min(1, annual limit / U)`.
    ProRata,
}

/// What a layer's per-event recovery is figured on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Basis {
    /// The loss: `min(max(x − attachment, 0), limit)`.
    Loss,
    /// A surplus treaty with retention line `retention` and `lines` lines:
    /// a loss `x` on a risk with sum insured `SI` cedes the share
    /// `min(max(SI − retention, 0), lines × retention) / SI` of `x` (the
    /// layer's `attachment` and `limit`, an event limit, then apply to
    /// that). Needs each event's sum insured.
    Surplus { retention: f64, lines: f64 },
}

impl Basis {
    /// The amount a layer's per-event terms apply to, for a loss on a risk
    /// with sum insured `si` (NaN for a surplus without one).
    pub(super) fn amount(self, loss: f64, si: Option<f64>) -> f64 {
        match self {
            Basis::Loss => loss,
            Basis::Surplus { retention, lines } => match si {
                Some(si) if si > 0.0 => loss * (si - retention).clamp(0.0, lines * retention) / si,
                _ => f64::NAN,
            },
        }
    }
}
