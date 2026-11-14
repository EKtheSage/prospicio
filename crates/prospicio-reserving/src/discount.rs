//! Discounted reserves and a risk adjustment for non-financial risk, for
//! the IFRS 17 liability for incurred claims (`reserving-ifrs17.md`, issue
//! #553).
//!
//! [`discounted_reserves`] discounts every simulation of a stochastic fit's
//! reserves from the valuation date along its payment timing, by origin and
//! in total, and [`DiscountedReserves::risk_adjustment`] reads a risk
//! adjustment off the discounted total by a confidence level, TVaR or a
//! cost-of-capital method, with the confidence level it corresponds to
//! (IFRS 17 para. 119).
//!
//! # Each simulation's payments
//!
//! For simulation `s`, origin `w` at its latest age `a_w` with reserve
//! `R_{s,w}` and cumulative pattern `c_s` at the pattern's ages `d`
//! (`reserving-ifrs17.md` decision 1a):
//!
//! ```text
//! share(d)   = (c_s(d) - c_s(d - 1)) / (1 - c_s(a_w))   for d > a_w
//! payment(d) = R_{s,w} share(d)
//! ```
//!
//! paid `t = (d - a_w) / 12` years after the valuation date at the end of
//! each development period ([`PaymentTiming::EndOfPeriod`]), or half a
//! development period earlier ([`PaymentTiming::MidPeriod`], the default).
//! Ages are in months on the triangle's grain and each origin is timed from
//! its own latest age, so ragged triangles and quarterly or monthly grains
//! need nothing special. With a tail the pattern has one more age, one
//! development period past the oldest, where the tail's share is paid.
//!
//! * When `1 - c_s(a_w)` is at most `1e-12`, or a share is not finite, the
//!   whole reserve is paid at the next age (one development period on when
//!   the pattern has no later age).
//! * Negative increments and negative reserves are kept and discounted as
//!   they are.
//! * [`PaymentBasis::Paid`] spreads by each simulation's own pattern (the
//!   fit's `patterns`); [`PaymentBasis::Incurred`] by one paid pattern the
//!   caller brings, for a fit on incurred claims (decision 4b).
//! * `Compartmental` is not spread: its `paid` cells are each simulation's
//!   cumulative paid by origin and age, so each increment from the latest
//!   paid is paid at its own age, process timing included. Under
//!   `Fitted::IncurredOnly` its paid cells have no noise, which is an
//!   error.
//!
//! The timing varies between simulations only through the pattern, so
//! spreading carries the pattern's parameter uncertainty in the timing, not
//! each cell's process error (decision 1).
//!
//! # Discounting and the run-off
//!
//! A payment at `t` is discounted by [`DiscountCurve::factor`],
//! `(1 + r(t))^-t`. Each simulation is discounted on its own timing and
//! then summed, so the quantiles of the discounted total are read from the
//! discounted draws, never by scaling undiscounted quantiles.
//! [`DiscountedReserves::runoff`] is the best estimate outstanding at each
//! whole year `t`, `BE(t) = sum over payments after t of E[payment] v(time)
//! / v(t)`, the mean discounted outstanding valued at `t` (EIOPA's method 2
//! proxy for the cost of capital's run-off).
//!
//! # What is supported
//!
//! [`Discountable`] is implemented by the fits with reserves and patterns
//! per simulation: [`OdpBootstrapFit`], [`OdpBootstrapFits`],
//! [`MackBootstrapFit`], [`MackBootstrapFits`] and, with the `bayes`
//! feature, `BayesOdpFit`, `BayesOdpFits`, `BayesBfFit`,
//! `BayesBfNormalisedFit`, `MeyersFit`, `MeyersFits`, `CompartmentalFit`
//! and `CompartmentalFits`. Analytic [`MackFit`] (no simulations) and
//! `HierarchicalBfFit` (no pattern per simulation) implement it too, and
//! return [`Error::NotSupported`] saying why.
//!
//! A `*Fits` over segments discounts each segment by its own pattern and
//! ages and keeps the fit's simulations together, so the discounted total
//! keeps the dependence between segments. Under
//! [`PaymentBasis::Incurred`] the one paid pattern is applied to every
//! segment, so every segment must have the same development ages; give the
//! segments their own paid patterns by discounting each segment's fit.
//!
//! ```
//! use prospicio_reserving::{
//!     DevelopmentColumn, DiscountCurve, Discounting, Grain, Long, MackBootstrap, Month,
//!     PaymentBasis, PaymentTiming, RiskAdjustment, Triangle, discounted_reserves,
//! };
//!
//! let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
//! let tri = Triangle::from_long(&Long {
//!     keys: &[],
//!     origin: &origin,
//!     development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
//!     values: &[("paid", &[100.0, 150.0, 165.0, 110.0, 170.0, 120.0])],
//!     origin_grain: Grain::Year,
//!     development_grain: Grain::Year,
//!     cumulative: true,
//! })?;
//! let fit = MackBootstrap { n_sims: 1_000, seed: 1, ..Default::default() }.fit(&tri, "paid")?;
//! let discounting = Discounting {
//!     curve: DiscountCurve::flat(0.03)?,
//!     timing: PaymentTiming::MidPeriod,
//! };
//! let d = discounted_reserves(&fit, &discounting, &PaymentBasis::Paid)?;
//! assert!(d.best_estimate < fit.reserves.mean());
//! let ra = d.risk_adjustment(&RiskAdjustment::ConfidenceLevel(0.75))?;
//! assert!(ra.implied_confidence >= 0.75);
//! # Ok::<(), prospicio_reserving::Error>(())
//! ```

use prospicio_core::Lag;
use prospicio_prob::PredictiveDistribution;
use prospicio_prob::risk::{tvar_sorted, var_sorted};

use crate::bootstrap::{OdpBootstrapFit, OdpBootstrapFits};
use crate::chain_ladder::ChainLadderFit;
use crate::error::{Error, Result};
use crate::mack::MackFit;
use crate::mack_bootstrap::{MackBootstrapFit, MackBootstrapFits};
use crate::pattern_draws::{DevelopmentPatterns, SegmentPatterns};
use crate::segments::{ReserveFit, SegmentFits};
use crate::triangle::Label;

use sealed::{CashFlows, Component, How, Pattern};

/// The share of the ultimate still to develop at or below which an
/// origin's whole reserve is paid at its next age.
const FULLY_DEVELOPED: f64 = 1e-12;

/// Annual effective spot rates by maturity in years: linear in the rate
/// between maturities and flat beyond the first and the last.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscountCurve {
    maturities: Vec<f64>,
    rates: Vec<f64>,
}

impl DiscountCurve {
    /// One annual effective rate at every maturity.
    ///
    /// Fails unless `rate` is finite and above -1.
    pub fn flat(rate: f64) -> Result<Self> {
        Self::spot(&[1.0], &[rate])
    }

    /// Spot rates `rates[k]` at `maturities[k]` years, annually compounded.
    ///
    /// Fails unless both are non-empty and of equal length, the maturities
    /// are finite, positive and strictly increasing, and every rate is
    /// finite and above -1.
    pub fn spot(maturities: &[f64], rates: &[f64]) -> Result<Self> {
        if maturities.is_empty() {
            return Err(Error::InvalidSetting {
                name: "maturities",
                value: 0.0,
                expected: "at least one maturity",
            });
        }
        if rates.len() != maturities.len() {
            return Err(Error::LengthMismatch {
                column: "rates".into(),
                expected: maturities.len(),
                found: rates.len(),
            });
        }
        let mut previous = 0.0;
        for &m in maturities {
            if !(m.is_finite() && m > previous) {
                return Err(Error::InvalidSetting {
                    name: "maturity",
                    value: m,
                    expected: "finite, positive and strictly increasing",
                });
            }
            previous = m;
        }
        for &r in rates {
            if !(r.is_finite() && r > -1.0) {
                return Err(Error::InvalidSetting {
                    name: "rate",
                    value: r,
                    expected: "finite and above -1",
                });
            }
        }
        Ok(Self {
            maturities: maturities.to_vec(),
            rates: rates.to_vec(),
        })
    }

    /// The maturities in years (`[1.0]` for a flat curve).
    pub fn maturities(&self) -> &[f64] {
        &self.maturities
    }

    /// The spot rate at each maturity.
    pub fn rates(&self) -> &[f64] {
        &self.rates
    }

    /// The spot rate at `t` years: the first rate up to the first maturity,
    /// the last from the last maturity on, linear between.
    pub fn rate(&self, t: f64) -> f64 {
        let m = &self.maturities;
        let r = &self.rates;
        let last = m.len() - 1;
        if t <= m[0] {
            return r[0];
        }
        if t >= m[last] {
            return r[last];
        }
        let k = m.partition_point(|&x| x <= t);
        let w = (t - m[k - 1]) / (m[k] - m[k - 1]);
        r[k - 1] + w * (r[k] - r[k - 1])
    }

    /// The discount factor of a payment at `t` years, `(1 + r(t))^-t`.
    pub fn factor(&self, t: f64) -> f64 {
        if t == 0.0 {
            return 1.0;
        }
        (1.0 + self.rate(t)).powf(-t)
    }
}

/// When in a development period its payments fall.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PaymentTiming {
    /// At the middle of the period (decision 3a): claims are paid through
    /// the period.
    #[default]
    MidPeriod,
    /// At the end of the period, as `PaymentPattern` in `prospicio-pricing`.
    EndOfPeriod,
}

/// The curve and timing of [`discounted_reserves`].
#[derive(Debug, Clone, PartialEq)]
pub struct Discounting {
    pub curve: DiscountCurve,
    pub timing: PaymentTiming,
}

/// What the fit's column is, stated by the caller (decision 4b), so that a
/// fit on incurred claims is not discounted by its reporting pattern by
/// mistake.
#[derive(Debug, Clone, PartialEq)]
pub enum PaymentBasis {
    /// The fit is on paid claims: each simulation's reserve is paid along
    /// that simulation's own pattern (`Compartmental`: its paid cells).
    Paid,
    /// The fit is on incurred claims: its reserves are paid along this
    /// cumulative paid pattern, the share of the ultimate paid by each of
    /// the fit's pattern ages (the triangle's development ages, and the
    /// tail's age one period past the oldest when the fit has a tail), the
    /// same in every simulation and every segment. It must end at 1. Not
    /// for `Compartmental`, whose reserves are paid.
    Incurred(Vec<f64>),
}

/// The capital a cost-of-capital risk adjustment charges for, at the
/// valuation date. There is no default (decision 6b).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Capital {
    /// `Q_q(X) - BE`, the inverted-cdf quantile of the discounted total.
    Var(f64),
    /// `TVaR_q(X) - BE`.
    Tvar(f64),
    /// A capital the caller brings, such as the one-year view's 99.5% VaR
    /// of the claims development result.
    Amount(f64),
}

/// When in each year the cost of capital is charged (decision 2a).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CapitalCharge {
    /// At the end of each year, `RA = rate Σ_t K_t v(t + 1)` (Solvency II's
    /// risk margin).
    #[default]
    EndOfYear,
    /// At the start of each year, `RA = rate Σ_t K_t v(t)`, as
    /// `price_over_payout` in `prospicio-pricing` (`time-value.md` decision
    /// 4).
    StartOfYear,
}

/// How [`DiscountedReserves::risk_adjustment`] reads the risk adjustment
/// off the discounted total `X`, whose mean is the best estimate `BE`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RiskAdjustment {
    /// `Q_p(X) - BE`, the inverted-cdf quantile
    /// ([`var_sorted`](prospicio_prob::risk::var_sorted)).
    ConfidenceLevel(f64),
    /// `TVaR_p(X) - BE`, the exact TVaR of the draws
    /// ([`tvar_sorted`](prospicio_prob::risk::tvar_sorted): the draw at the
    /// VaR counted with the share of its mass above `p`).
    Tvar(f64),
    /// `rate` times the capital held in each future year, discounted:
    /// `K_t = K_0 BE(t) / BE(0)` from the [run-off](DiscountedReserves::runoff),
    /// `K_0` from `capital`, charged as `charge` says.
    CostOfCapital {
        rate: f64,
        capital: Capital,
        charge: CapitalCharge,
    },
}

/// A risk adjustment and the confidence level it corresponds to.
#[derive(Debug, Clone, PartialEq)]
pub struct RiskAdjustmentResult {
    /// The risk adjustment.
    pub amount: f64,
    /// The mean discounted total.
    pub best_estimate: f64,
    /// `#{X_s <= BE + RA} / n` (IFRS 17 para. 119). For
    /// [`ConfidenceLevel`](RiskAdjustment::ConfidenceLevel) and
    /// [`Tvar`](RiskAdjustment::Tvar) the draws are counted against the
    /// quantile or TVaR itself, so rounding in `BE + RA` cannot drop a draw
    /// tied with it: under `ConfidenceLevel(p)` it is at least `p`.
    pub implied_confidence: f64,
    /// The capital at the valuation date, `K_0`, of a cost-of-capital
    /// method; `None` otherwise.
    pub capital: Option<f64>,
    /// Each segment's stand-alone result, for a fit over segments (no
    /// allocation of the total, decision 8a); empty for a fit of one
    /// segment, and under [`Capital::Amount`], which is the total's capital.
    pub segments: Vec<SegmentRiskAdjustment>,
}

/// One segment's stand-alone risk adjustment.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentRiskAdjustment {
    /// The segment's label, one part per key.
    pub label: Vec<String>,
    /// Its risk adjustment on its own discounted total and run-off.
    pub result: RiskAdjustmentResult,
}

/// A fit's reserves discounted simulation by simulation
/// ([`discounted_reserves`]).
#[derive(Debug, Clone)]
pub struct DiscountedReserves {
    /// The fit's reserves, undiscounted.
    pub reserves: PredictiveDistribution,
    /// The present value of each reserve draw: the same dimensions,
    /// components and simulations as [`reserves`](Self::reserves).
    pub discounted: PredictiveDistribution,
    /// The mean discounted total.
    pub best_estimate: f64,
    /// The best estimate outstanding at `t = 0, 1, 2, ...` whole years from
    /// the valuation date, the mean discounted payments after `t` valued at
    /// `t`, ending with the first year after the last payment, where it is
    /// 0. `runoff[0]` is the best estimate up to rounding.
    pub runoff: Vec<f64>,
    /// Each component's mean present value over its mean reserve, in the
    /// order of the components; 1 where the mean reserve is 0.
    pub discount_factors: Vec<f64>,
    discounting: Discounting,
    /// Each segment's label (one empty label for a fit of one segment).
    segments: Vec<Vec<String>>,
    /// Whether the fit is over segments.
    over_segments: bool,
    /// Each component's segment.
    component_segment: Vec<usize>,
    /// Each segment's run-off, as [`runoff`](Self::runoff).
    segment_runoff: Vec<Vec<f64>>,
}

/// A fit whose reserves [`discounted_reserves`] can discount (see the
/// [module docs](self) for the list). Sealed: the cash flows it gives are
/// internal.
pub trait Discountable: sealed::Sealed {}

/// Each simulation of `fit`'s reserves discounted from the valuation date
/// along its payment timing (see the [module docs](self)).
///
/// Fails for a fit with no pattern per simulation or no simulations
/// ([`Error::NotSupported`]), for a `Compartmental` fit under
/// `Fitted::IncurredOnly` or [`PaymentBasis::Incurred`], for a paid pattern
/// of the wrong length or not ending at 1, or when the fit's patterns do
/// not match its reserves.
pub fn discounted_reserves<F: Discountable + ?Sized>(
    fit: &F,
    discounting: &Discounting,
    basis: &PaymentBasis,
) -> Result<DiscountedReserves> {
    if let PaymentBasis::Incurred(pattern) = basis {
        check_paid_pattern(pattern)?;
    }
    let flows = fit.cash_flows(basis)?;
    discount(flows, discounting, basis)
}

fn check_paid_pattern(pattern: &[f64]) -> Result<()> {
    let last = pattern.last().copied().unwrap_or(f64::NAN);
    if pattern.iter().any(|v| !v.is_finite()) || (last - 1.0).abs() > 1e-9 {
        return Err(Error::InvalidSetting {
            name: "paid pattern",
            value: last,
            expected: "finite cumulative shares ending at 1",
        });
    }
    Ok(())
}

mod sealed {
    use super::*;

    pub trait Sealed {
        /// The fit's reserves and how each component is paid.
        fn cash_flows<'a>(&'a self, basis: &'a PaymentBasis) -> Result<CashFlows<'a>>;
    }

    /// A fit's reserves, its segments and how each component is paid.
    pub struct CashFlows<'a> {
        pub(super) reserves: &'a PredictiveDistribution,
        /// Each segment's label; one empty label for a fit of one segment.
        pub(super) segments: Vec<Vec<String>>,
        pub(super) over_segments: bool,
        pub(super) components: Vec<Component<'a>>,
    }

    pub struct Component<'a> {
        pub(super) segment: usize,
        pub(super) how: How<'a>,
    }

    pub enum How<'a> {
        /// The reserve spread over a cumulative pattern at `ages` from the
        /// age at index `latest`.
        Spread {
            pattern: Pattern<'a>,
            ages: &'a [Lag],
            latest: usize,
        },
        /// Cumulative paid cells: each cell's component in `paid` and age,
        /// ascending, from `latest_value` at `latest_age` (`Compartmental`,
        /// with the `bayes` feature).
        #[cfg_attr(not(feature = "bayes"), allow(dead_code))]
        Cells {
            paid: &'a PredictiveDistribution,
            cells: Vec<(usize, Lag)>,
            latest_value: f64,
            latest_age: Lag,
            period: Lag,
        },
    }

    #[derive(Clone, Copy)]
    pub enum Pattern<'a> {
        /// Each simulation's own pattern.
        Draws(&'a SegmentPatterns),
        /// One pattern for every simulation.
        Fixed(&'a [f64]),
    }
}

/// The spread components of one segment's origins.
fn spread_segment<'a>(
    segment: usize,
    chain_ladder: &ChainLadderFit,
    patterns: &'a SegmentPatterns,
    basis: &'a PaymentBasis,
    out: &mut Vec<Component<'a>>,
) -> Result<()> {
    let ages = patterns.ages();
    let triangle = &chain_ladder.development.development;
    if !(ages.len() >= triangle.len()
        && ages.len() <= triangle.len() + 1
        && ages[..triangle.len()] == triangle[..])
    {
        return Err(Error::PatternDraws(format!(
            "the patterns' ages {ages:?} are not the fit's development ages {triangle:?} \
             (and at most one tail age)"
        )));
    }
    let pattern = match basis {
        PaymentBasis::Paid => Pattern::Draws(patterns),
        PaymentBasis::Incurred(paid) => {
            if paid.len() != ages.len() {
                return Err(Error::LengthMismatch {
                    column: "paid pattern".into(),
                    expected: ages.len(),
                    found: paid.len(),
                });
            }
            Pattern::Fixed(paid)
        }
    };
    for &latest in &chain_ladder.latest_position {
        if latest >= ages.len() {
            return Err(Error::PatternDraws(format!(
                "an origin's latest position {latest} is past the patterns' {} ages",
                ages.len()
            )));
        }
        out.push(Component {
            segment,
            how: How::Spread {
                pattern,
                ages,
                latest,
            },
        });
    }
    Ok(())
}

fn label_parts(label: &Label) -> Vec<String> {
    label.parts().to_vec()
}

fn segment_patterns<'a>(
    patterns: &'a DevelopmentPatterns,
    label: &Label,
) -> Result<&'a SegmentPatterns> {
    let parts: Vec<&str> = label.parts().iter().map(String::as_str).collect();
    patterns.segment(&parts)
}

fn check_sims(reserves: &PredictiveDistribution, patterns: &DevelopmentPatterns) -> Result<()> {
    if patterns.n_sims() != reserves.n_sims() {
        return Err(Error::PatternDraws(format!(
            "{} patterns for {} simulations of the reserves",
            patterns.n_sims(),
            reserves.n_sims()
        )));
    }
    Ok(())
}

/// The cash flows of a fit of one segment spread by its patterns.
fn one_segment<'a>(
    reserves: &'a PredictiveDistribution,
    chain_ladder: &ChainLadderFit,
    patterns: &'a DevelopmentPatterns,
    basis: &'a PaymentBasis,
) -> Result<CashFlows<'a>> {
    check_sims(reserves, patterns)?;
    let segment = match patterns.segments() {
        [only] => only,
        other => {
            return Err(Error::PatternDraws(format!(
                "a fit of one segment with patterns of {} segments",
                other.len()
            )));
        }
    };
    let mut components = Vec::new();
    spread_segment(0, chain_ladder, segment, basis, &mut components)?;
    finish(reserves, vec![Vec::new()], false, components)
}

/// The cash flows of a fit over segments, each spread by its own patterns.
fn over_segments<'a, T: ReserveFit>(
    reserves: &'a PredictiveDistribution,
    segments: &SegmentFits<T>,
    patterns: &'a DevelopmentPatterns,
    basis: &'a PaymentBasis,
) -> Result<CashFlows<'a>> {
    check_sims(reserves, patterns)?;
    let mut components = Vec::new();
    let mut labels = Vec::new();
    for (s, (label, fit)) in segments.iter().enumerate() {
        let p = segment_patterns(patterns, label)?;
        spread_segment(s, fit.chain_ladder(), p, basis, &mut components)?;
        labels.push(label_parts(label));
    }
    finish(reserves, labels, true, components)
}

fn finish<'a>(
    reserves: &'a PredictiveDistribution,
    segments: Vec<Vec<String>>,
    over_segments: bool,
    components: Vec<Component<'a>>,
) -> Result<CashFlows<'a>> {
    if components.len() != reserves.n_components() {
        return Err(Error::LengthMismatch {
            column: "reserves".into(),
            expected: components.len(),
            found: reserves.n_components(),
        });
    }
    Ok(CashFlows {
        reserves,
        segments,
        over_segments,
        components,
    })
}

impl sealed::Sealed for OdpBootstrapFit {
    fn cash_flows<'a>(&'a self, basis: &'a PaymentBasis) -> Result<CashFlows<'a>> {
        one_segment(&self.reserves, &self.chain_ladder, &self.patterns, basis)
    }
}
impl Discountable for OdpBootstrapFit {}

impl sealed::Sealed for OdpBootstrapFits {
    fn cash_flows<'a>(&'a self, basis: &'a PaymentBasis) -> Result<CashFlows<'a>> {
        over_segments(&self.reserves, &self.segments, &self.patterns, basis)
    }
}
impl Discountable for OdpBootstrapFits {}

impl sealed::Sealed for MackBootstrapFit {
    fn cash_flows<'a>(&'a self, basis: &'a PaymentBasis) -> Result<CashFlows<'a>> {
        one_segment(
            &self.reserves,
            &self.mack.chain_ladder,
            &self.patterns,
            basis,
        )
    }
}
impl Discountable for MackBootstrapFit {}

impl sealed::Sealed for MackBootstrapFits {
    fn cash_flows<'a>(&'a self, basis: &'a PaymentBasis) -> Result<CashFlows<'a>> {
        over_segments(&self.reserves, &self.segments, &self.patterns, basis)
    }
}
impl Discountable for MackBootstrapFits {}

/// Analytic Mack has no simulations: an error saying so.
impl sealed::Sealed for MackFit {
    fn cash_flows<'a>(&'a self, _: &'a PaymentBasis) -> Result<CashFlows<'a>> {
        Err(Error::NotSupported {
            setting: "MackFit",
            method: "discounted_reserves",
            reason: "analytic Mack has no simulations to discount: use MackBootstrap, whose \
                     reserves and patterns are per simulation",
        })
    }
}
impl Discountable for MackFit {}

#[cfg(feature = "bayes")]
mod bayes_fits {
    use super::sealed::{CashFlows, Component, How, Sealed};
    use super::{Discountable, PaymentBasis, finish, one_segment, over_segments};
    use crate::bayes::{
        BayesBfFit, BayesBfNormalisedFit, BayesOdpFit, BayesOdpFits, CompartmentalFit,
        CompartmentalFits, HierarchicalBfFit, MeyersFit, MeyersFits,
    };
    use crate::error::{Error, Result};
    use prospicio_core::Lag;
    use prospicio_prob::{KeyValue, PredictiveDistribution};

    impl Sealed for BayesOdpFit {
        fn cash_flows<'a>(&'a self, basis: &'a PaymentBasis) -> Result<CashFlows<'a>> {
            one_segment(&self.reserves, &self.chain_ladder, &self.patterns, basis)
        }
    }
    impl Discountable for BayesOdpFit {}

    impl Sealed for BayesOdpFits {
        fn cash_flows<'a>(&'a self, basis: &'a PaymentBasis) -> Result<CashFlows<'a>> {
            over_segments(&self.reserves, &self.segments, &self.patterns, basis)
        }
    }
    impl Discountable for BayesOdpFits {}

    impl Sealed for BayesBfFit {
        fn cash_flows<'a>(&'a self, basis: &'a PaymentBasis) -> Result<CashFlows<'a>> {
            one_segment(&self.reserves, &self.chain_ladder, &self.patterns, basis)
        }
    }
    impl Discountable for BayesBfFit {}

    impl Sealed for BayesBfNormalisedFit {
        fn cash_flows<'a>(&'a self, basis: &'a PaymentBasis) -> Result<CashFlows<'a>> {
            one_segment(&self.reserves, &self.chain_ladder, &self.patterns, basis)
        }
    }
    impl Discountable for BayesBfNormalisedFit {}

    /// Meyers' models simulate no path: their reserves are spread by each
    /// simulation's pattern.
    impl Sealed for MeyersFit {
        fn cash_flows<'a>(&'a self, basis: &'a PaymentBasis) -> Result<CashFlows<'a>> {
            one_segment(&self.reserves, &self.chain_ladder, &self.patterns, basis)
        }
    }
    impl Discountable for MeyersFit {}

    impl Sealed for MeyersFits {
        fn cash_flows<'a>(&'a self, basis: &'a PaymentBasis) -> Result<CashFlows<'a>> {
            over_segments(&self.reserves, &self.segments, &self.patterns, basis)
        }
    }
    impl Discountable for MeyersFits {}

    /// The hierarchical BF keeps no pattern per simulation: an error
    /// saying so.
    impl Sealed for HierarchicalBfFit {
        fn cash_flows<'a>(&'a self, _: &'a PaymentBasis) -> Result<CashFlows<'a>> {
            Err(Error::NotSupported {
                setting: "HierarchicalBfFit",
                method: "discounted_reserves",
                reason: "the hierarchical Bornhuetter-Ferguson keeps no development pattern \
                         per simulation to time its reserves by",
            })
        }
    }
    impl Discountable for HierarchicalBfFit {}

    fn compartmental_checks(fit: &CompartmentalFit, basis: &PaymentBasis) -> Result<()> {
        if matches!(basis, PaymentBasis::Incurred(_)) {
            return Err(Error::NotSupported {
                setting: "PaymentBasis::Incurred",
                method: "Compartmental",
                reason: "its reserves are paid claims: discount it with PaymentBasis::Paid, \
                         which times them by its paid cells",
            });
        }
        if !fit.paid_timing().0 {
            return Err(Error::NotSupported {
                setting: "Fitted::IncurredOnly",
                method: "discounted_reserves",
                reason: "the paid cells of a fit on incurred alone are the expected paid \
                         without observation noise, so they cannot time each simulation's \
                         payments",
            });
        }
        Ok(())
    }

    /// The components of `paid` (keys `[..., origin, age]`) grouped by the
    /// component of `reserves` (keys `[..., origin]`) they belong to, each
    /// group by age.
    fn cells_by_component(
        reserves: &PredictiveDistribution,
        paid: &PredictiveDistribution,
    ) -> Result<Vec<Vec<(usize, Lag)>>> {
        let index: std::collections::HashMap<&[KeyValue], usize> = reserves
            .components()
            .iter()
            .enumerate()
            .map(|(j, k)| (k.as_slice(), j))
            .collect();
        let mut out = vec![Vec::new(); reserves.n_components()];
        for (c, key) in paid.components().iter().enumerate() {
            let (age, prefix) = key.split_last().ok_or(Error::Empty)?;
            let age = match age {
                KeyValue::Int(a) => Lag::try_from(*a).ok(),
                _ => None,
            };
            let (Some(age), Some(&j)) = (age, index.get(prefix)) else {
                return Err(Error::PatternDraws(format!(
                    "the paid cell {key:?} matches no reserve component"
                )));
            };
            out[j].push((c, age));
        }
        for cells in &mut out {
            cells.sort_by_key(|&(_, age)| age);
        }
        Ok(out)
    }

    /// One segment's components from its paid cells, `cells` that
    /// segment's part of `by_component`.
    fn compartmental_components<'a>(
        segment: usize,
        fit: &CompartmentalFit,
        paid: &'a PredictiveDistribution,
        cells: impl Iterator<Item = Vec<(usize, Lag)>>,
        out: &mut Vec<Component<'a>>,
    ) {
        let (_, period, latest_age) = fit.paid_timing();
        for ((cells, &latest_value), &latest_age) in cells.zip(&fit.latest_paid).zip(&latest_age) {
            out.push(Component {
                segment,
                how: How::Cells {
                    paid,
                    cells,
                    latest_value,
                    latest_age,
                    period,
                },
            });
        }
    }

    /// From the paid cells, each increment at its own age.
    impl Sealed for CompartmentalFit {
        fn cash_flows<'a>(&'a self, basis: &'a PaymentBasis) -> Result<CashFlows<'a>> {
            compartmental_checks(self, basis)?;
            let cells = cells_by_component(&self.reserves, &self.paid)?;
            let mut components = Vec::new();
            compartmental_components(0, self, &self.paid, cells.into_iter(), &mut components);
            finish(&self.reserves, vec![Vec::new()], false, components)
        }
    }
    impl Discountable for CompartmentalFit {}

    impl Sealed for CompartmentalFits {
        fn cash_flows<'a>(&'a self, basis: &'a PaymentBasis) -> Result<CashFlows<'a>> {
            let mut cells = cells_by_component(&self.reserves, &self.paid)?.into_iter();
            let mut components = Vec::new();
            let mut labels = Vec::new();
            for (s, (label, segment)) in self.segments.iter().enumerate() {
                compartmental_checks(&segment.fit, basis)?;
                let n = segment.fit.latest_paid.len();
                compartmental_components(
                    s,
                    &segment.fit,
                    &self.paid,
                    cells.by_ref().take(n),
                    &mut components,
                );
                labels.push(label.parts().to_vec());
            }
            finish(&self.reserves, labels, true, components)
        }
    }
    impl Discountable for CompartmentalFits {}
}

/// A component's payments in half-months from the valuation date, each a
/// slot of the sorted distinct times.
enum Plan<'a> {
    Spread {
        pattern: Pattern<'a>,
        latest: usize,
        /// The slot of each age after `latest`.
        slots: Vec<usize>,
        /// Where the whole reserve goes when the pattern has fully
        /// developed: the next age.
        next: usize,
    },
    Cells {
        paid: &'a PredictiveDistribution,
        /// Each cell's component in `paid` and slot.
        cells: Vec<(usize, usize)>,
        latest_value: f64,
    },
}

/// The time of a payment at `age` from `latest`, `period` months after the
/// previous age, in half-months.
fn half_months(age: Lag, latest: Lag, period: Lag, timing: PaymentTiming) -> i64 {
    let end = 2 * (i64::from(age) - i64::from(latest));
    match timing {
        PaymentTiming::EndOfPeriod => end,
        PaymentTiming::MidPeriod => end - i64::from(period),
    }
}

/// The half-month times of a component: each age after `latest` and the
/// next age (the first after `latest`, or one period on past the last).
fn spread_times(ages: &[Lag], latest: usize, timing: PaymentTiming) -> (Vec<i64>, i64) {
    let a = ages[latest];
    let times: Vec<i64> = (latest + 1..ages.len())
        .map(|d| half_months(ages[d], a, ages[d] - ages[d - 1], timing))
        .collect();
    let next = times.first().copied().unwrap_or_else(|| {
        let period = if latest > 0 { a - ages[latest - 1] } else { a };
        half_months(a + period, a, period, timing)
    });
    (times, next)
}

fn discount(
    flows: CashFlows<'_>,
    discounting: &Discounting,
    basis: &PaymentBasis,
) -> Result<DiscountedReserves> {
    let timing = discounting.timing;
    // Every payment time, in half-months.
    let mut keys = Vec::new();
    for c in &flows.components {
        match &c.how {
            How::Spread { ages, latest, .. } => {
                let (times, next) = spread_times(ages, *latest, timing);
                keys.extend(times);
                keys.push(next);
            }
            How::Cells {
                cells,
                latest_age,
                period,
                ..
            } => keys.extend(
                cells
                    .iter()
                    .map(|&(_, age)| half_months(age, *latest_age, *period, timing)),
            ),
        }
    }
    keys.sort_unstable();
    keys.dedup();
    let slot = |k: i64| keys.binary_search(&k).expect("every time is a key");
    let years: Vec<f64> = keys.iter().map(|&k| k as f64 / 24.0).collect();
    let factors: Vec<f64> = years.iter().map(|&t| discounting.curve.factor(t)).collect();
    let component_segment: Vec<usize> = flows.components.iter().map(|c| c.segment).collect();
    let plans: Vec<Plan<'_>> = flows
        .components
        .into_iter()
        .map(|c| match c.how {
            How::Spread {
                pattern,
                ages,
                latest,
            } => {
                let (times, next) = spread_times(ages, latest, timing);
                Plan::Spread {
                    pattern,
                    latest,
                    slots: times.into_iter().map(slot).collect(),
                    next: slot(next),
                }
            }
            How::Cells {
                paid,
                cells,
                latest_value,
                latest_age,
                period,
            } => Plan::Cells {
                paid,
                cells: cells
                    .into_iter()
                    .map(|(c, age)| (c, slot(half_months(age, latest_age, period, timing))))
                    .collect(),
                latest_value,
            },
        })
        .collect();
    let reserves = flows.reserves;
    let n_comp = reserves.n_components();
    let n_sims = reserves.n_sims();
    let n_segments = flows.segments.len();
    let mut pv = Vec::with_capacity(n_sims * n_comp);
    // The sum over the simulations of each segment's payments in each slot.
    let mut paid_in = vec![vec![0.0; keys.len()]; n_segments];
    let mut payments: Vec<(usize, f64)> = Vec::new();
    for (s, row) in reserves.draw_matrix().chunks_exact(n_comp).enumerate() {
        for (j, (plan, &reserve)) in plans.iter().zip(row).enumerate() {
            payments.clear();
            pay(plan, s, reserve, &mut payments);
            let mut value = 0.0;
            for &(k, amount) in &payments {
                value += amount * factors[k];
                paid_in[component_segment[j]][k] += amount;
            }
            pv.push(value);
        }
    }
    let mean_paid: Vec<Vec<f64>> = paid_in
        .into_iter()
        .map(|p| p.into_iter().map(|x| x / n_sims as f64).collect())
        .collect();
    let horizon = years.last().map_or(0, |t| t.ceil() as usize);
    let runoff_of = |mean: &[f64]| -> Vec<f64> {
        (0..=horizon)
            .map(|t| {
                let t = t as f64;
                let v = discounting.curve.factor(t);
                years
                    .iter()
                    .zip(mean)
                    .zip(&factors)
                    .filter(|((y, _), _)| **y > t)
                    .map(|((_, m), f)| m * f / v)
                    .sum()
            })
            .collect()
    };
    let segment_runoff: Vec<Vec<f64>> = mean_paid.iter().map(|m| runoff_of(m)).collect();
    let total_mean: Vec<f64> = (0..keys.len())
        .map(|k| mean_paid.iter().map(|m| m[k]).sum())
        .collect();
    let runoff = runoff_of(&total_mean);

    let mut provenance = reserves.provenance().clone();
    provenance.parameters.retain(|(k, _)| k != "output");
    let provenance = provenance
        .param("output", "discounted reserves")
        .param("timing", format!("{timing:?}"))
        .param(
            "basis",
            match basis {
                PaymentBasis::Paid => "paid",
                PaymentBasis::Incurred(_) => "incurred",
            },
        );
    let discounted = PredictiveDistribution::from_draws(
        reserves.dims().to_vec(),
        reserves.components().to_vec(),
        pv,
        provenance,
    )?;
    let discount_factors = column_means(reserves)
        .into_iter()
        .zip(column_means(&discounted))
        .map(|(r, d)| if r == 0.0 { 1.0 } else { d / r })
        .collect();
    let best_estimate = mean(&row_sums(&discounted, |_| true));
    Ok(DiscountedReserves {
        reserves: reserves.clone(),
        discounted,
        best_estimate,
        runoff,
        discount_factors,
        discounting: discounting.clone(),
        segments: flows.segments,
        over_segments: flows.over_segments,
        component_segment,
        segment_runoff,
    })
}

/// Simulation `s`'s payments of `reserve` as `(slot, amount)`.
fn pay(plan: &Plan<'_>, s: usize, reserve: f64, out: &mut Vec<(usize, f64)>) {
    match plan {
        Plan::Spread {
            pattern,
            latest,
            slots,
            next,
        } => {
            let c = match pattern {
                Pattern::Draws(p) => p.draw(s),
                Pattern::Fixed(p) => p,
            };
            let base = c[*latest];
            let rest = 1.0 - base;
            let start = out.len();
            if rest > FULLY_DEVELOPED && base.is_finite() {
                let mut previous = base;
                for (&k, &cum) in slots.iter().zip(&c[latest + 1..]) {
                    out.push((k, reserve * (cum - previous) / rest));
                    previous = cum;
                }
            }
            if out.len() == start || out[start..].iter().any(|(_, a)| !a.is_finite()) {
                out.truncate(start);
                out.push((*next, reserve));
            }
        }
        Plan::Cells {
            paid,
            cells,
            latest_value,
        } => {
            let n = paid.n_components();
            let row = &paid.draw_matrix()[s * n..(s + 1) * n];
            let mut previous = *latest_value;
            for &(c, k) in cells {
                out.push((k, row[c] - previous));
                previous = row[c];
            }
        }
    }
}

fn column_means(pd: &PredictiveDistribution) -> Vec<f64> {
    let n = pd.n_components();
    let mut sums = vec![0.0; n];
    for row in pd.draw_matrix().chunks_exact(n) {
        for (s, v) in sums.iter_mut().zip(row) {
            *s += v;
        }
    }
    sums.iter().map(|s| s / pd.n_sims() as f64).collect()
}

/// Each simulation's sum over the components `keep` selects.
fn row_sums(pd: &PredictiveDistribution, keep: impl Fn(usize) -> bool) -> Vec<f64> {
    let n = pd.n_components();
    pd.draw_matrix()
        .chunks_exact(n)
        .map(|row| {
            row.iter()
                .enumerate()
                .filter(|(j, _)| keep(*j))
                .map(|(_, v)| v)
                .sum()
        })
        .collect()
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

impl DiscountedReserves {
    /// The curve and timing the reserves were discounted with.
    pub fn discounting(&self) -> &Discounting {
        &self.discounting
    }

    /// Each segment's label, for a fit over segments; empty for a fit of
    /// one segment.
    pub fn segment_labels(&self) -> Vec<Vec<String>> {
        if self.over_segments {
            self.segments.clone()
        } else {
            Vec::new()
        }
    }

    /// Segment `i`'s run-off, as [`runoff`](Self::runoff), in the order of
    /// [`segment_labels`](Self::segment_labels); `None` past the last.
    pub fn segment_runoff(&self, i: usize) -> Option<&[f64]> {
        self.segment_runoff.get(i).map(Vec::as_slice)
    }

    /// The risk adjustment on the discounted total (see [`RiskAdjustment`])
    /// and, for a fit over segments, each segment's stand-alone one.
    ///
    /// Fails for a probability outside `[0, 1]`, a cost-of-capital rate
    /// that is negative or not finite, a capital amount that is not finite,
    /// or a cost of capital on a best estimate of 0 with a capital that is
    /// not 0.
    pub fn risk_adjustment(&self, method: &RiskAdjustment) -> Result<RiskAdjustmentResult> {
        let total = row_sums(&self.discounted, |_| true);
        let mut result = risk_adjustment_on(total, &self.runoff, &self.discounting.curve, method)?;
        let amount_capital = matches!(
            method,
            RiskAdjustment::CostOfCapital {
                capital: Capital::Amount(_),
                ..
            }
        );
        if self.over_segments && !amount_capital {
            for (i, label) in self.segments.iter().enumerate() {
                let x = row_sums(&self.discounted, |j| self.component_segment[j] == i);
                let r = risk_adjustment_on(
                    x,
                    &self.segment_runoff[i],
                    &self.discounting.curve,
                    method,
                )?;
                result.segments.push(SegmentRiskAdjustment {
                    label: label.clone(),
                    result: r,
                });
            }
        }
        Ok(result)
    }
}

fn risk_adjustment_on(
    mut x: Vec<f64>,
    runoff: &[f64],
    curve: &DiscountCurve,
    method: &RiskAdjustment,
) -> Result<RiskAdjustmentResult> {
    let be = mean(&x);
    x.sort_by(f64::total_cmp);
    let (amount, level, capital) = match *method {
        RiskAdjustment::ConfidenceLevel(p) => {
            let q = var_sorted(&x, p)?;
            (q - be, q, None)
        }
        RiskAdjustment::Tvar(p) => {
            let t = tvar_sorted(&x, p)?;
            (t - be, t, None)
        }
        RiskAdjustment::CostOfCapital {
            rate,
            capital,
            charge,
        } => {
            if !(rate.is_finite() && rate >= 0.0) {
                return Err(Error::InvalidSetting {
                    name: "cost of capital rate",
                    value: rate,
                    expected: "finite and not negative",
                });
            }
            let k0 = match capital {
                Capital::Var(q) => var_sorted(&x, q)? - be,
                Capital::Tvar(q) => tvar_sorted(&x, q)? - be,
                Capital::Amount(k) => {
                    if !k.is_finite() {
                        return Err(Error::InvalidSetting {
                            name: "capital",
                            value: k,
                            expected: "finite",
                        });
                    }
                    k
                }
            };
            let ra = cost_of_capital(rate, k0, runoff, curve, charge)?;
            (ra, be + ra, Some(k0))
        }
    };
    let implied_confidence = x.partition_point(|&v| v <= level) as f64 / x.len() as f64;
    Ok(RiskAdjustmentResult {
        amount,
        best_estimate: be,
        implied_confidence,
        capital,
        segments: Vec::new(),
    })
}

/// `rate Σ_t K_t v(t + 1)` (end of year) or `rate Σ_t K_t v(t)` (start of
/// year), `K_t = k0 BE(t) / BE(0)`.
fn cost_of_capital(
    rate: f64,
    k0: f64,
    runoff: &[f64],
    curve: &DiscountCurve,
    charge: CapitalCharge,
) -> Result<f64> {
    let be0 = runoff.first().copied().unwrap_or(0.0);
    if be0 == 0.0 {
        if k0 == 0.0 {
            return Ok(0.0);
        }
        return Err(Error::InvalidSetting {
            name: "best estimate",
            value: be0,
            expected: "not 0, to run the capital off with it",
        });
    }
    let shift = match charge {
        CapitalCharge::EndOfYear => 1.0,
        CapitalCharge::StartOfYear => 0.0,
    };
    Ok(rate
        * runoff
            .iter()
            .enumerate()
            .map(|(t, be)| k0 * be / be0 * curve.factor(t as f64 + shift))
            .sum::<f64>())
}

#[cfg(test)]
mod tests {
    use prospicio_core::Grain;
    use prospicio_prob::risk::var_sorted;
    use prospicio_prob::{Distribution, KeyValue, PredictiveDistribution, Provenance};

    use super::sealed::{CashFlows, Component, How, Pattern};
    use super::*;
    use crate::triangle::tests::raa;
    use crate::{
        ChainLadder, DevelopmentColumn, Long, Mack, MackBootstrap, OdpBootstrap, Tail, Triangle,
    };

    fn at(rate: f64, timing: PaymentTiming) -> Discounting {
        Discounting {
            curve: DiscountCurve::flat(rate).unwrap(),
            timing,
        }
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    fn mack_boot(n_sims: usize) -> MackBootstrapFit {
        MackBootstrap {
            n_sims,
            seed: 11,
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap()
    }

    #[test]
    fn a_zero_rate_leaves_every_draw_as_it_was() {
        let fit = mack_boot(300);
        let zero = DiscountCurve::spot(&[1.0, 10.0], &[0.0, 0.0]).unwrap();
        for timing in [PaymentTiming::MidPeriod, PaymentTiming::EndOfPeriod] {
            let d = discounted_reserves(
                &fit,
                &Discounting {
                    curve: zero.clone(),
                    timing,
                },
                &PaymentBasis::Paid,
            )
            .unwrap();
            assert_eq!(d.discounted.components(), fit.reserves.components());
            for (a, b) in d
                .discounted
                .draw_matrix()
                .iter()
                .zip(fit.reserves.draw_matrix())
            {
                assert!(close(*a, *b, 1e-9), "{a} vs {b}");
            }
            assert!(close(d.runoff[0], d.best_estimate, 1e-9));
            assert_eq!(*d.runoff.last().unwrap(), 0.0);
            assert!(d.discount_factors.iter().all(|f| close(*f, 1.0, 1e-9)));
        }
    }

    /// RAA's chain-ladder pattern, cumulative at each age, built from its
    /// age-to-age factors by hand.
    fn raa_pattern(ldf: &[f64]) -> Vec<f64> {
        let n = ldf.len() + 1;
        (0..n)
            .map(|d| 1.0 / ldf[d..].iter().product::<f64>())
            .collect()
    }

    /// Each origin's present value of one unit of reserve under `pattern`,
    /// mid-period payments, flat `rate`, by hand.
    fn unit_values(pattern: &[f64], latest: &[usize], rate: f64) -> Vec<f64> {
        latest
            .iter()
            .map(|&a| {
                let rest = 1.0 - pattern[a];
                if rest <= 1e-12 {
                    return (1.0 + rate).powf(-0.5);
                }
                (a + 1..pattern.len())
                    .map(|d| {
                        let share = (pattern[d] - pattern[d - 1]) / rest;
                        share * (1.0 + rate).powf(-((d - a) as f64 - 0.5))
                    })
                    .sum()
            })
            .collect()
    }

    /// Mack's bootstrap with RAA's own factors fully selected: every
    /// simulation has the chain ladder's pattern.
    fn selected_fit() -> (MackBootstrapFit, Vec<f64>) {
        let cl = ChainLadder::default().fit(&raa(), "values").unwrap();
        let fit = MackBootstrap {
            n_sims: 200,
            seed: 5,
            development: cl.to_chain_ladder().development,
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap();
        (fit, raa_pattern(cl.ldf()))
    }

    #[test]
    fn a_flat_rate_on_raa_is_the_hand_built_cash_flow() {
        let (fit, pattern) = selected_fit();
        let seg = fit.patterns.segment(&[]).unwrap();
        for s in 0..fit.reserves.n_sims() {
            for (a, b) in seg.draw(s).iter().zip(&pattern) {
                assert!(close(*a, *b, 1e-12));
            }
        }
        let latest = &fit.mack.chain_ladder.latest_position;
        let unit = unit_values(&pattern, latest, 0.03);
        let mid = at(0.03, PaymentTiming::MidPeriod);
        let d = discounted_reserves(&fit, &mid, &PaymentBasis::Paid).unwrap();
        let n = unit.len();
        for (pv, r) in d
            .discounted
            .draw_matrix()
            .chunks_exact(n)
            .zip(fit.reserves.draw_matrix().chunks_exact(n))
        {
            for w in 0..n {
                assert!(
                    close(pv[w], r[w] * unit[w], 1e-9),
                    "{w}: {} vs {}",
                    pv[w],
                    r[w]
                );
            }
        }
        for (w, f) in d.discount_factors.iter().enumerate() {
            if latest[w] + 1 < pattern.len() {
                assert!(close(*f, unit[w], 1e-9));
            } else {
                // The oldest origin has no reserve.
                assert_eq!(*f, 1.0);
            }
        }
        assert!(d.best_estimate < fit.reserves.total().mean());

        // The same pattern given as a paid pattern reproduces it.
        let given =
            discounted_reserves(&fit, &mid, &PaymentBasis::Incurred(pattern.clone())).unwrap();
        for (a, b) in given
            .discounted
            .draw_matrix()
            .iter()
            .zip(d.discounted.draw_matrix())
        {
            assert!(close(*a, *b, 1e-12));
        }
        // A paid pattern of the wrong length or not ending at 1.
        let short = PaymentBasis::Incurred(pattern[1..].to_vec());
        assert!(matches!(
            discounted_reserves(&fit, &mid, &short),
            Err(Error::LengthMismatch { .. })
        ));
        let mut low = pattern.clone();
        *low.last_mut().unwrap() = 0.9;
        assert!(matches!(
            discounted_reserves(&fit, &mid, &PaymentBasis::Incurred(low)),
            Err(Error::InvalidSetting { .. })
        ));
    }

    /// Three origins at ages 12, 24 and 36 (a pattern to 36, no tail) in two
    /// simulations, with patterns `[0.5, 1.1, 1.0]` (a falling last step)
    /// and `[0.4, 1.0, 1.0]` (fully developed at 24), and a reserve on the
    /// oldest origin with no later age.
    #[test]
    fn a_hand_built_case_times_each_payment() {
        let ages: Vec<Lag> = vec![12, 24, 36];
        let label = Label::new(Vec::<String>::new());
        let patterns = SegmentPatterns::new(
            &label,
            Grain::Year,
            ages.clone(),
            vec![0.5, 1.1, 1.0, 0.4, 1.0, 1.0],
        );
        let reserves = PredictiveDistribution::from_draws(
            vec!["origin".into()],
            (1..=3).map(|o| vec![KeyValue::Int(o)]).collect(),
            vec![100.0, 50.0, 5.0, 80.0, 40.0, 0.0],
            Provenance::new("test"),
        )
        .unwrap();
        let flows = || CashFlows {
            reserves: &reserves,
            segments: vec![Vec::new()],
            over_segments: false,
            components: (0..3)
                .map(|latest| Component {
                    segment: 0,
                    how: How::Spread {
                        pattern: Pattern::Draws(&patterns),
                        ages: &ages,
                        latest,
                    },
                })
                .collect(),
        };
        let v = |t: f64| 1.1f64.powf(-t);
        for (timing, m) in [
            (PaymentTiming::EndOfPeriod, 0.0),
            (PaymentTiming::MidPeriod, 0.5),
        ] {
            let d = discount(flows(), &at(0.10, timing), &PaymentBasis::Paid).unwrap();
            let pv = d.discounted.draw_matrix();
            let want = [
                // Simulation 0: origin 1 pays 120 at the next age and -20
                // after it; origin 2's pattern has passed 1, so it pays its
                // 50 at the next age; origin 3 has no later age and pays one
                // period on.
                120.0 * v(1.0 - m) - 20.0 * v(2.0 - m),
                50.0 * v(1.0 - m),
                5.0 * v(1.0 - m),
                // Simulation 1: origin 1 pays everything at 24, nothing at
                // 36; origin 2 is fully developed (1 - c = 0 <= 1e-12).
                80.0 * v(1.0 - m),
                40.0 * v(1.0 - m),
                0.0,
            ];
            for (a, b) in pv.iter().zip(want) {
                assert!(close(*a, b, 1e-12), "{timing:?}: {a} vs {b}");
            }
            // Mean payments: 147.5 in the first year, -10 in the second.
            let be0 = 147.5 * v(1.0 - m) - 10.0 * v(2.0 - m);
            assert!(close(d.runoff[0], be0, 1e-12));
            assert!(close(d.runoff[1], -10.0 * v(2.0 - m) / v(1.0), 1e-12));
            assert_eq!(d.runoff.len(), 3);
            assert_eq!(d.runoff[2], 0.0);
            assert!(close(d.best_estimate, be0, 1e-12));
        }
    }

    #[test]
    fn the_tail_is_paid_one_period_past_the_oldest_age() {
        let fit = MackBootstrap {
            n_sims: 200,
            seed: 3,
            tail: Tail::from(1.05),
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap();
        let ages = fit.patterns.segment(&[]).unwrap().ages().to_vec();
        assert_eq!(ages.len(), 11);
        assert_eq!(ages[10], 132);
        let oldest = 0;
        assert_eq!(fit.mack.chain_ladder.latest_position[oldest], 9);
        for (timing, t) in [
            (PaymentTiming::EndOfPeriod, 1.0),
            (PaymentTiming::MidPeriod, 0.5),
        ] {
            let d = discounted_reserves(&fit, &at(0.03, timing), &PaymentBasis::Paid).unwrap();
            let n = d.discounted.n_components();
            let mut some = false;
            for (pv, r) in d
                .discounted
                .draw_matrix()
                .chunks_exact(n)
                .zip(fit.reserves.draw_matrix().chunks_exact(n))
            {
                some |= r[oldest] != 0.0;
                assert!(close(pv[oldest], r[oldest] * 1.03f64.powf(-t), 1e-9));
            }
            assert!(some);
        }
    }

    #[test]
    fn the_curve_interpolates_the_spot_rate_and_is_flat_outside() {
        let flat = DiscountCurve::flat(0.03).unwrap();
        let spot = DiscountCurve::spot(&[1.0, 5.0], &[0.03, 0.03]).unwrap();
        for t in [0.0, 0.25, 1.0, 2.5, 7.0, 40.0] {
            assert_eq!(flat.factor(t), spot.factor(t));
            assert!(close(flat.factor(t), 1.03f64.powf(-t), 1e-15));
        }
        let c = DiscountCurve::spot(&[1.0, 3.0], &[0.02, 0.04]).unwrap();
        assert_eq!(c.rate(0.5), 0.02);
        assert!(close(c.rate(2.0), 0.03, 1e-15));
        assert!(close(c.rate(1.5), 0.025, 1e-15));
        assert_eq!(c.rate(5.0), 0.04);
        assert!(close(c.factor(2.0), 1.03f64.powf(-2.0), 1e-15));
        assert!(close(c.factor(5.0), 1.04f64.powf(-5.0), 1e-15));
        assert_eq!(c.factor(0.0), 1.0);
        for bad in [
            DiscountCurve::spot(&[], &[]),
            DiscountCurve::spot(&[1.0], &[0.01, 0.02]),
            DiscountCurve::spot(&[2.0, 1.0], &[0.01, 0.02]),
            DiscountCurve::spot(&[0.0], &[0.01]),
            DiscountCurve::flat(-1.0),
            DiscountCurve::flat(f64::NAN),
        ] {
            assert!(bad.is_err());
        }
    }

    #[test]
    fn the_cost_of_capital_charges_each_years_capital() {
        let fit = mack_boot(500);
        let d = discounted_reserves(
            &fit,
            &at(0.03, PaymentTiming::MidPeriod),
            &PaymentBasis::Paid,
        )
        .unwrap();
        let k0 = 4_000.0;
        let ra = |charge| {
            d.risk_adjustment(&RiskAdjustment::CostOfCapital {
                rate: 0.06,
                capital: Capital::Amount(k0),
                charge,
            })
            .unwrap()
        };
        let be = &d.runoff;
        // time-value.md decision 4's capital_years: year 0 in full.
        let start = 0.06
            * k0
            * be.iter()
                .enumerate()
                .map(|(k, b)| b / be[0] * 1.03f64.powi(-(k as i32)))
                .sum::<f64>();
        // Solvency II: each year's capital charged at its end.
        let end: f64 = be
            .iter()
            .enumerate()
            .map(|(k, b)| 0.06 * k0 * b / be[0] / 1.03f64.powi(k as i32 + 1))
            .sum();
        let s = ra(CapitalCharge::StartOfYear);
        let e = ra(CapitalCharge::EndOfYear);
        assert!(close(s.amount, start, 1e-12));
        assert!(close(e.amount, end, 1e-12));
        assert!(close(e.amount, s.amount / 1.03, 1e-12));
        assert_eq!(e.capital, Some(k0));
        assert!(e.amount > 0.06 * k0 / 1.03 && e.amount < 0.06 * k0 * be.len() as f64);
        // A capital from the draws: the 99.5% VaR less the best estimate.
        let v = d
            .risk_adjustment(&RiskAdjustment::CostOfCapital {
                rate: 0.06,
                capital: Capital::Var(0.995),
                charge: CapitalCharge::EndOfYear,
            })
            .unwrap();
        let mut x = row_sums(&d.discounted, |_| true);
        x.sort_by(f64::total_cmp);
        let q = var_sorted(&x, 0.995).unwrap();
        assert!(close(v.capital.unwrap(), q - d.best_estimate, 1e-12));
        assert!(close(v.amount, e.amount / k0 * (q - d.best_estimate), 1e-9));
        assert!(
            d.risk_adjustment(&RiskAdjustment::CostOfCapital {
                rate: -0.01,
                capital: Capital::Amount(k0),
                charge: CapitalCharge::EndOfYear,
            })
            .is_err()
        );
    }

    #[test]
    fn the_implied_confidence_of_a_confidence_level_is_that_level() {
        let fit = mack_boot(997);
        let d = discounted_reserves(
            &fit,
            &at(0.03, PaymentTiming::MidPeriod),
            &PaymentBasis::Paid,
        )
        .unwrap();
        let n = 997.0;
        for p in [0.5, 0.6, 0.75, 0.9, 0.995] {
            let r = d
                .risk_adjustment(&RiskAdjustment::ConfidenceLevel(p))
                .unwrap();
            assert!(r.implied_confidence >= p && r.implied_confidence < p + 1.0 / n);
            // The total is skewed to the right: above its mean from 0.75.
            assert_eq!(r.amount > 0.0, p >= 0.75);
            assert!(r.segments.is_empty() && r.capital.is_none());
            let t = d.risk_adjustment(&RiskAdjustment::Tvar(p)).unwrap();
            assert!(t.amount > r.amount);
            assert!(t.implied_confidence > p);
        }
        assert!(
            d.risk_adjustment(&RiskAdjustment::ConfidenceLevel(1.5))
                .is_err()
        );
    }

    /// RAA twice, under the `lob` values `A` and `B` (B scaled by 2).
    fn two_lines() -> Triangle {
        let segment = raa().segment("values").unwrap();
        let (mut keys, mut origin, mut ages, mut values) = (vec![], vec![], vec![], vec![]);
        for (name, scale) in [("A", 1.0), ("B", 2.0)] {
            for (o, period) in segment.origins.iter().enumerate() {
                for (d, &age) in segment.ages.iter().enumerate() {
                    if let Some(v) = segment.get(o, d) {
                        keys.push(name);
                        origin.push(period.start());
                        ages.push(age);
                        values.push(scale * v);
                    }
                }
            }
        }
        Triangle::from_long(&Long {
            keys: &[("lob", &keys)],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &values)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    #[test]
    fn segments_are_discounted_jointly_with_each_standalone_result_beside() {
        let fits = OdpBootstrap {
            n_sims: 400,
            seed: 2,
            ..Default::default()
        }
        .fit_segments(&two_lines(), "paid")
        .unwrap();
        let d = discounted_reserves(
            &fits,
            &at(0.03, PaymentTiming::MidPeriod),
            &PaymentBasis::Paid,
        )
        .unwrap();
        assert_eq!(
            d.segment_labels(),
            [vec!["A".to_string()], vec!["B".to_string()]]
        );
        let n = d.discounted.n_components();
        let by_rows = |keep: &dyn Fn(usize) -> bool| -> Vec<f64> {
            d.discounted
                .draw_matrix()
                .chunks_exact(n)
                .map(|row| (0..n).filter(|j| keep(*j)).map(|j| row[j]).sum())
                .collect()
        };
        let ra_of = |mut x: Vec<f64>, p: f64| {
            let be = mean(&x);
            x.sort_by(f64::total_cmp);
            var_sorted(&x, p).unwrap() - be
        };
        let r = d
            .risk_adjustment(&RiskAdjustment::ConfidenceLevel(0.75))
            .unwrap();
        assert!(close(r.amount, ra_of(by_rows(&|_| true), 0.75), 1e-12));
        assert_eq!(r.segments.len(), 2);
        for (i, seg) in r.segments.iter().enumerate() {
            let x = by_rows(&|j| (j < n / 2) == (i == 0));
            assert!(close(seg.result.amount, ra_of(x, 0.75), 1e-12));
            assert!(seg.result.segments.is_empty());
        }
        // B is A scaled by 2, so B's stand-alone RA is the larger, and the
        // run-offs add up to the total's.
        assert!(r.segments[1].result.amount > r.segments[0].result.amount);
        let (ra, rb) = (d.segment_runoff(0).unwrap(), d.segment_runoff(1).unwrap());
        for (t, total) in d.runoff.iter().enumerate() {
            assert!(close(ra[t] + rb[t], *total, 1e-9));
        }
        // A cost of capital on each segment's own capital, and none for a
        // capital amount, which is the total's.
        let coc = |capital| {
            d.risk_adjustment(&RiskAdjustment::CostOfCapital {
                rate: 0.06,
                capital,
                charge: CapitalCharge::EndOfYear,
            })
            .unwrap()
        };
        assert_eq!(coc(Capital::Var(0.995)).segments.len(), 2);
        assert!(coc(Capital::Amount(1e4)).segments.is_empty());
    }

    #[test]
    fn analytic_mack_has_no_simulations_to_discount() {
        let fit = Mack::default().fit(&raa(), "values").unwrap();
        assert!(matches!(
            discounted_reserves(
                &fit,
                &at(0.03, PaymentTiming::MidPeriod),
                &PaymentBasis::Paid
            ),
            Err(Error::NotSupported { .. })
        ));
    }

    #[cfg(feature = "bayes")]
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_bayesian_chain_ladder_is_spread_by_its_posterior_patterns() {
        use prospicio_bayes::nuts::Sampler;

        let fit = crate::BayesOdp {
            sampler: Sampler {
                tune: 200,
                draws: 200,
                ..Sampler::default()
            },
            n_sims: 400,
            seed: 4,
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap();
        let zero = discounted_reserves(
            &fit,
            &at(0.0, PaymentTiming::MidPeriod),
            &PaymentBasis::Paid,
        )
        .unwrap();
        for (a, b) in zero
            .discounted
            .draw_matrix()
            .iter()
            .zip(fit.reserves.draw_matrix())
        {
            assert!(close(*a, *b, 1e-9));
        }
        let d = discounted_reserves(
            &fit,
            &at(0.03, PaymentTiming::MidPeriod),
            &PaymentBasis::Paid,
        )
        .unwrap();
        assert!(d.best_estimate < zero.best_estimate);
    }
}
