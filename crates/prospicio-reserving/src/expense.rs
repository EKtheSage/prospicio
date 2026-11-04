//! Recoveries and claim adjustment expenses: the ratio methods that finish an
//! unpaid estimate (`docs/design/reserving-v02.md`, decision 37).
//!
//! An unpaid estimate covers more than losses. [`RatioDevelopment`] develops
//! the ratio of two columns of a triangle (received salvage and subrogation
//! to paid claims, paid ALAE to paid claims) and applies the selected
//! ultimate ratio to ultimate claims from any method, as Friedland does in
//! chapters 14 and 16 of *Estimating Unpaid Claims Using Basic Techniques*.
//! [`Ulae`] estimates unpaid unallocated expense from calendar-year history,
//! as her chapter 17 does: the classical paid-to-paid technique, Kittel's
//! refinement and Conger and Nolibos' generalized approach.
//!
//! Triangles of recoveries need nothing of their own: [`Triangle`] takes
//! negative increments, and the chain ladder takes link ratios and factors
//! to ultimate below 1.

use crate::chain_ladder::ChainLadder;
use crate::development::{Average, Development};
use crate::error::{Error, Result};
use crate::tail::Tail;
use crate::triangle::{Segment, Triangle};
use prospicio_core::{Lag, Period};

const RATIO_METHOD: &str = "additive ratio development";

/// How a ratio develops from one age to the next.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum RatioFactors {
    /// Age-to-age factors, multiplied to ultimate: the chain ladder of the
    /// ratio.
    #[default]
    Multiplicative,
    /// Age-to-age differences, added to ultimate: Friedland's chapter 16,
    /// Exhibit I, Sheets 9 and 10, after Wiser, who finds it more stable
    /// when the ratios are very small at early ages.
    Additive {
        /// Differences chosen by the user, each keyed by the age it starts
        /// from; the others are averaged. Empty selects none.
        selected: Vec<(Lag, f64)>,
        /// The addition past the oldest age; 0 for none.
        tail: f64,
    },
}

/// The ratio of two columns of one triangle, developed to ultimate and
/// applied to ultimate claims: Friedland's ratio approach.
///
/// Each cell's ratio is taken from the cumulative columns themselves, so the
/// numerator may be received recoveries or paid ALAE and the denominator paid
/// claims. A cell whose denominator is zero or missing, or whose numerator is
/// missing, has no ratio. The ratio's own age-to-age factors then give each
/// origin's projected ultimate ratio, [`selected`](Self::selected) replaces
/// it where the actuary judges otherwise, and the ultimate of the numerator
/// is the ratio times the ultimate claims, which come from any method.
///
/// The default averages the ratio's link ratios simply, as Friedland does: a
/// weight by the size of the ratio means little. Any [`Development`] works
/// for the multiplicative factors, with its [`pattern`](Development::pattern)
/// for selected factors.
///
/// ```
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, RatioDevelopment, Triangle};
///
/// let origin = [2020, 2020, 2021].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 12]),
///     values: &[
///         ("alae", &[10.0, 30.0, 20.0]),
///         ("paid", &[100.0, 150.0, 200.0]),
///     ],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let fit = RatioDevelopment::default().fit(&tri, "alae", "paid", &[160.0, 320.0])?;
/// assert_eq!(fit.factors, vec![2.0]);
/// assert_eq!(fit.projected_ratio, vec![0.2, 0.2]);
/// assert_eq!(fit.ultimate, vec![32.0, 64.0]);
/// assert_eq!(fit.unpaid(), vec![2.0, 44.0]);
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RatioDevelopment {
    /// How the ratio's age-to-age factors are estimated. For additive
    /// factors it must be the simple average, and its `n_periods`,
    /// `drop_high` and `drop_low` apply to the differences.
    pub development: Development,
    /// Development of the ratio past the oldest age, for multiplicative
    /// factors. Friedland starts from no tail on the ratio of paid ALAE to
    /// paid claims and then tests 1.005.
    pub tail: Tail,
    /// Multiplicative or additive factors.
    pub factors: RatioFactors,
    /// An ultimate ratio chosen by the user for an origin, which replaces
    /// the projected one: Friedland's 0.345 for accident year 2008. Each
    /// origin at most once.
    pub selected: Vec<(Period, f64)>,
}

impl Default for RatioDevelopment {
    fn default() -> Self {
        Self {
            development: Development {
                average: Average::Simple,
                ..Development::default()
            },
            tail: Tail::default(),
            factors: RatioFactors::default(),
            selected: Vec::new(),
        }
    }
}

/// A fitted [`RatioDevelopment`], per origin.
#[derive(Debug, Clone, PartialEq)]
pub struct RatioDevelopmentFit {
    /// Origin periods, oldest first; every per-origin vector follows them.
    pub origins: Vec<Period>,
    /// Age of each development position.
    pub ages: Vec<Lag>,
    /// The ratio triangle, one row per origin; `None` where a cell has no
    /// ratio.
    pub ratios: Vec<Vec<Option<f64>>>,
    /// The averages of the ratio's age-to-age factors or differences, one
    /// fewer than the ages; NaN where a fully selected age has none.
    pub averages: Vec<f64>,
    /// The age-to-age factors or differences in use, after any selection.
    pub factors: Vec<f64>,
    /// Age-to-ultimate factors or additions, one per age; element `k`
    /// develops a ratio at age `k` to ultimate.
    pub cdf: Vec<f64>,
    /// Development position of each origin's latest ratio.
    pub latest_position: Vec<usize>,
    /// The latest ratio per origin.
    pub latest_ratio: Vec<f64>,
    /// The latest ratio developed to ultimate.
    pub projected_ratio: Vec<f64>,
    /// The ultimate ratio in use: the projected one, or the user's.
    pub selected_ratio: Vec<f64>,
    /// The ultimate claims the ratio applies to, as given.
    pub ultimate_claims: Vec<f64>,
    /// Latest cumulative numerator per origin (paid ALAE, received
    /// recoveries).
    pub latest: Vec<f64>,
    /// The numerator's ultimate per origin: the selected ratio times the
    /// ultimate claims.
    pub ultimate: Vec<f64>,
}

impl RatioDevelopment {
    /// Fits the ratio of `numerator` to `denominator`, two columns of a
    /// single-segment triangle, and applies it to `ultimate_claims`, one
    /// value per origin of `denominator`, oldest first.
    pub fn fit(
        &self,
        triangle: &Triangle,
        numerator: &str,
        denominator: &str,
        ultimate_claims: &[f64],
    ) -> Result<RatioDevelopmentFit> {
        let den = triangle.segment(denominator)?;
        check_values("ultimate_claims", ultimate_claims, den.n_origins)?;
        let c = triangle.column_position(numerator)?;
        let cum = triangle.to_cumulative();
        let cell = |o: usize, d: usize| cum.get(0, c, o + den.origin_offset, d + den.dev_offset);
        let mut cells = Vec::with_capacity(den.n_origins * den.n_dev);
        let mut latest = Vec::with_capacity(den.n_origins);
        for o in 0..den.n_origins {
            latest.push(
                (0..den.n_dev)
                    .rev()
                    .find_map(|d| cell(o, d))
                    .ok_or_else(|| Error::EmptyOrigin(den.origins[o].to_string()))?,
            );
            for d in 0..den.n_dev {
                cells.push(match (cell(o, d), den.get(o, d)) {
                    (Some(n), Some(x)) if x != 0.0 => Some(n / x),
                    _ => None,
                });
            }
        }
        let ratios = Segment::from_cells(den.origins.clone(), den.ages.clone(), cells);
        let developed = match &self.factors {
            RatioFactors::Multiplicative => {
                let cl = ChainLadder {
                    development: self.development.clone(),
                    tail: self.tail,
                }
                .fit_segment(&ratios, &ratios.ages)?;
                Developed {
                    averages: cl.development.ldf.clone(),
                    factors: cl.ldf().to_vec(),
                    cdf: cl.cdf.clone(),
                    latest_position: cl.latest_position.clone(),
                    latest_ratio: cl.latest.clone(),
                    projected_ratio: cl.ultimate,
                }
            }
            RatioFactors::Additive { selected, tail } => self.additive(&ratios, selected, *tail)?,
        };
        let selected_ratio = self.selected_ratios(&ratios.origins, &developed.projected_ratio)?;
        let ultimate = selected_ratio
            .iter()
            .zip(ultimate_claims)
            .map(|(r, u)| r * u)
            .collect();
        Ok(RatioDevelopmentFit {
            ratios: (0..ratios.n_origins)
                .map(|o| (0..ratios.n_dev).map(|d| ratios.get(o, d)).collect())
                .collect(),
            origins: ratios.origins,
            ages: ratios.ages,
            averages: developed.averages,
            factors: developed.factors,
            cdf: developed.cdf,
            latest_position: developed.latest_position,
            latest_ratio: developed.latest_ratio,
            projected_ratio: developed.projected_ratio,
            selected_ratio,
            ultimate_claims: ultimate_claims.to_vec(),
            latest,
            ultimate,
        })
    }

    /// The additive development of `ratios`.
    fn additive(&self, ratios: &Segment, selected: &[(Lag, f64)], tail: f64) -> Result<Developed> {
        if self.development.average != Average::Simple {
            return Err(Error::NotSupported {
                setting: "an average other than the simple one",
                method: RATIO_METHOD,
                reason: "differences of ratios have no regression behind them: use the simple \
                         average",
            });
        }
        if self.development.pattern.is_some() {
            return Err(Error::NotSupported {
                setting: "a selected pattern",
                method: RATIO_METHOD,
                reason: "a pattern holds factors: give the selected differences in \
                         RatioFactors::Additive",
            });
        }
        if !self.tail.is_none() {
            return Err(Error::NotSupported {
                setting: "a tail factor",
                method: RATIO_METHOD,
                reason: "a tail factor multiplies: give the addition past the oldest age in \
                         RatioFactors::Additive",
            });
        }
        if !tail.is_finite() {
            return Err(Error::InvalidTail(tail));
        }
        let n_links = ratios.n_dev.saturating_sub(1);
        let mut picks = vec![None; n_links];
        for (k, &(age, value)) in selected.iter().enumerate() {
            let position = ratios.ages[..n_links].iter().position(|a| *a == age);
            let reason = if position.is_none() {
                Some("no link of the triangle starts at this age")
            } else if !value.is_finite() {
                Some("a selected difference must be finite")
            } else if selected[..k].iter().any(|(a, _)| *a == age) {
                Some("this age is selected twice")
            } else {
                None
            };
            if let Some(reason) = reason {
                return Err(Error::Pattern {
                    age: Some(age),
                    reason,
                });
            }
            picks[position.expect("checked above")] = Some(value);
        }
        // `n_periods` keeps the links as it does for factors; the drops rank
        // the differences here, not the link ratios.
        let kept = Development {
            drop_high: 0,
            drop_low: 0,
            ..self.development.clone()
        }
        .kept_mask(ratios)?;
        let mut averages = Vec::with_capacity(n_links);
        for age in 0..n_links {
            let mut differences: Vec<f64> = ratios
                .link_pairs_indexed(age)
                .into_iter()
                .filter(|&(o, _, _)| kept[o * n_links + age])
                .map(|(_, from, to)| to - from)
                .collect();
            if differences.is_empty() && picks[age].is_none() {
                return Err(Error::Factor {
                    age,
                    reason: "no origin is observed at both ages",
                });
            }
            differences.sort_by(f64::total_cmp);
            let (low, high) = (self.development.drop_low, self.development.drop_high);
            let kept = if low.saturating_add(high) < differences.len() {
                &differences[low..differences.len() - high]
            } else {
                &differences[..]
            };
            averages.push(kept.iter().sum::<f64>() / kept.len() as f64);
        }
        let factors: Vec<f64> = averages
            .iter()
            .zip(&picks)
            .map(|(a, p)| p.unwrap_or(*a))
            .collect();
        let mut cdf = vec![tail; ratios.n_dev];
        for k in (0..n_links).rev() {
            cdf[k] = cdf[k + 1] + factors[k];
        }
        let (latest_position, latest_ratio): (Vec<usize>, Vec<f64>) = (0..ratios.n_origins)
            .map(|o| ratios.latest(o))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip();
        let projected_ratio = latest_position
            .iter()
            .zip(&latest_ratio)
            .map(|(&p, r)| r + cdf[p])
            .collect();
        Ok(Developed {
            averages,
            factors,
            cdf,
            latest_position,
            latest_ratio,
            projected_ratio,
        })
    }

    /// Each origin's ultimate ratio: its entry in `selected`, or `projected`.
    fn selected_ratios(&self, origins: &[Period], projected: &[f64]) -> Result<Vec<f64>> {
        let mut out = projected.to_vec();
        for (k, &(origin, ratio)) in self.selected.iter().enumerate() {
            if !ratio.is_finite() {
                return Err(Error::InvalidSetting {
                    name: "selected",
                    value: ratio,
                    expected: "a finite ultimate ratio",
                });
            }
            if self.selected[..k].iter().any(|(o, _)| *o == origin) {
                return Err(Error::Apriori(format!("origin {origin} is given twice")));
            }
            let Some(position) = origins.iter().position(|o| *o == origin) else {
                return Err(Error::Apriori(format!(
                    "{origin} is not among the origins fitted"
                )));
            };
            out[position] = ratio;
        }
        Ok(out)
    }
}

/// What the multiplicative and additive developments have in common.
struct Developed {
    averages: Vec<f64>,
    factors: Vec<f64>,
    cdf: Vec<f64>,
    latest_position: Vec<usize>,
    latest_ratio: Vec<f64>,
    projected_ratio: Vec<f64>,
}

impl RatioDevelopmentFit {
    /// The numerator still to come per origin: ultimate less latest. For
    /// recoveries it is the recoverable, a reduction of the unpaid estimate.
    /// A selected ratio can make it negative, as in Friedland's 2007 ALAE.
    pub fn unpaid(&self) -> Vec<f64> {
        self.ultimate
            .iter()
            .zip(&self.latest)
            .map(|(u, l)| u - l)
            .collect()
    }

    /// Total ultimate across origins.
    pub fn total_ultimate(&self) -> f64 {
        self.ultimate.iter().sum()
    }

    /// Total still to come across origins.
    pub fn total_unpaid(&self) -> f64 {
        self.total_ultimate() - self.latest.iter().sum::<f64>()
    }
}

/// Conger and Nolibos' shares of ULAE spent opening, maintaining and closing
/// claims, `U1`, `U2` and `U3`. They sum to 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UlaeWeights {
    /// `U1`, the share spent opening claims, in proportion to the ultimate
    /// cost of claims reported.
    pub opening: f64,
    /// `U2`, the share spent maintaining claims, in proportion to payments.
    pub maintaining: f64,
    /// `U3`, the share spent closing claims, in proportion to the ultimate
    /// cost of claims closed.
    pub closing: f64,
}

/// How far the weights may be from summing to 1.
const WEIGHT_TOLERANCE: f64 = 1e-9;

impl UlaeWeights {
    /// Friedland's simplification, `U3 = 0`: no particular effort goes into
    /// closing a claim.
    pub fn new(opening: f64, maintaining: f64) -> Self {
        Self {
            opening,
            maintaining,
            closing: 0.0,
        }
    }

    fn check(&self) -> Result<()> {
        for (name, value) in [
            ("opening", self.opening),
            ("maintaining", self.maintaining),
            ("closing", self.closing),
        ] {
            if !(0.0..=1.0).contains(&value) {
                return Err(Error::InvalidSetting {
                    name,
                    value,
                    expected: "a share from 0 to 1",
                });
            }
        }
        let sum = self.opening + self.maintaining + self.closing;
        if (sum - 1.0).abs() > WEIGHT_TOLERANCE {
            return Err(Error::InvalidSetting {
                name: "weights",
                value: sum,
                expected: "opening, maintaining and closing summing to 1",
            });
        }
        Ok(())
    }
}

/// A technique for unpaid unallocated loss adjustment expense (ULAE), from
/// Friedland's chapter 17.
///
/// Each one relates calendar-year paid ULAE to a claims basis, takes a
/// selected ratio and applies it to what remains of the claims. `ratio` is
/// that selection, a judgment in every one of Friedland's exhibits; `None`
/// uses the ratio of the totals.
///
/// The techniques differ in the basis, the weighted sum of
/// [`reported`](UlaeInputs::reported), [`paid`](UlaeInputs::paid) and
/// [`closed`](UlaeInputs::closed): 0, 1 and 0 for the classical technique,
/// one half each of reported and paid for Kittel, and any
/// [`UlaeWeights`] for Conger and Nolibos.
///
/// Mango and Allen's refinement is the classical technique with expected
/// paid claims in place of the actual ones.
///
/// ```
/// use prospicio_reserving::{Ulae, UlaeInputs};
///
/// // Friedland, chapter 17, Exhibit I, with the selected ratio of 0.045.
/// let fit = Ulae::classical(Some(0.045)).estimate(&UlaeInputs {
///     paid_ulae: &[14_352_000.0, 15_321_000.0],
///     paid: &[333_000_000.0, 358_000_000.0],
///     case_outstanding: &[603_000_000.0],
///     ibnr: &[316_000_000.0],
///     ..Default::default()
/// })?;
/// assert!((fit.total() - 27_787_500.0).abs() < 1e-6);
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ulae {
    /// The classical, or paid-to-paid, technique: calendar paid ULAE over
    /// calendar paid claims, applied to `1 - opening` of the unpaid known
    /// claims and all of the claims not yet reported.
    Classical {
        /// The selected ratio; `None` uses the ratio of the totals.
        ratio: Option<f64>,
        /// The share of a claim's ULAE sustained when it opens, 0.5 by the
        /// classical assumption. Friedland's erratum to page 388: an insurer
        /// spending 25% opening and 75% closing applies 75% of the ratio to
        /// case outstanding and 100% to IBNR.
        opening: f64,
    },
    /// Kittel's refinement: calendar paid ULAE over the average of calendar
    /// paid and calendar incurred claims, applied as the classical technique
    /// applies it under the 50/50 assumption.
    Kittel {
        /// The selected ratio; `None` uses the ratio of the totals.
        ratio: Option<f64>,
    },
    /// Conger and Nolibos' generalized approach: calendar paid ULAE over the
    /// claims basis `B = U1 R + U2 P + U3 C`, applied to each share's
    /// remaining claims.
    CongerNolibos {
        /// `U1`, `U2` and `U3`.
        weights: UlaeWeights,
        /// The selected ratio `W*`; `None` uses the ratio of the totals.
        ratio: Option<f64>,
    },
}

/// The inputs of [`Ulae::estimate`]; a technique reads what it needs and
/// says so if something it needs is missing.
///
/// The first group is calendar-year history, one value per year, of one
/// length. The second is what remains of the claims at the valuation date,
/// one value per row, of one length: Friedland's exhibits use a single row
/// of totals, and rows by accident year give the unpaid ULAE by accident
/// year, since every formula is linear in them.
#[derive(Debug, Clone, Copy, Default)]
pub struct UlaeInputs<'a> {
    /// Paid ULAE by calendar year, `M`.
    pub paid_ulae: &'a [f64],
    /// Paid claims by calendar year, `P`. Claims include ALAE and exclude
    /// ULAE. For Mango and Allen's refinement, the expected paid claims.
    pub paid: &'a [f64],
    /// The reporting side of the basis by calendar year. For Kittel, the
    /// calendar year's incurred claims: paid claims plus the change in case
    /// outstanding and IBNR. For Conger and Nolibos, `R`, the ultimate cost
    /// of the claims reported in the year, or in their simplification the
    /// accident year's ultimate claims. Unused by the classical technique.
    pub reported: &'a [f64],
    /// `C`, the ultimate cost of the claims closed in each calendar year,
    /// for Conger and Nolibos when `U3` is not 0.
    pub closed: &'a [f64],
    /// Case outstanding at the valuation date.
    pub case_outstanding: &'a [f64],
    /// Total IBNR at the valuation date, in its broad sense: claims not yet
    /// reported and future development on known claims.
    pub ibnr: &'a [f64],
    /// Pure IBNR, the part of [`ibnr`](Self::ibnr) on claims not yet
    /// reported. With it the classical technique and Kittel apply the full
    /// ratio to it alone, and half of it to the rest of the IBNR. Conger and
    /// Nolibos need it when `U1` is not 0: it is `L - R(t)`, their claims
    /// not yet reported.
    pub pure_ibnr: Option<&'a [f64]>,
    /// The ultimate cost of the claims not yet closed at the valuation date,
    /// `L - C(t)`, for Conger and Nolibos when `U3` is not 0.
    pub unclosed: &'a [f64],
    /// Ultimate claims `L` by accident year, for Conger and Nolibos' totals
    /// ([`UlaeEstimate::expected_claims`] and the two after it). They
    /// compare `L` with the whole of the calendar history, so the years in
    /// both must cover the same claims.
    pub ultimate: &'a [f64],
}

/// The result of [`Ulae::estimate`].
#[derive(Debug, Clone, PartialEq)]
pub struct UlaeEstimate {
    /// The claims basis by calendar year.
    pub basis: Vec<f64>,
    /// Paid ULAE over the basis by calendar year; NaN at a zero basis.
    pub ratios: Vec<f64>,
    /// Total paid ULAE over the total basis.
    pub total_ratio: f64,
    /// The ratio applied: the selected one, or [`total_ratio`](Self::total_ratio).
    pub ratio: f64,
    /// Unpaid ULAE by row of the valuation inputs.
    pub unpaid: Vec<f64>,
    /// Conger and Nolibos' expected claims form, `W* L - M`, which they do
    /// not prefer. `None` without ultimate claims, or for the other
    /// techniques.
    pub expected_claims: Option<f64>,
    /// Conger and Nolibos' preferred form, `W* (L - B)`. Equal to the total
    /// of [`unpaid`](Self::unpaid) when `R` and `C` are the exact calendar
    /// amounts. Under their simplification `R` is the accident year's
    /// ultimate, the opening share of the basis sums to that of `L`, and
    /// this form loses the pure IBNR: the unpaid rows are then the estimate.
    pub bornhuetter_ferguson: Option<f64>,
    /// Conger and Nolibos' development form, `M (L / B - 1)`.
    pub development: Option<f64>,
}

impl UlaeEstimate {
    /// Total unpaid ULAE.
    pub fn total(&self) -> f64 {
        self.unpaid.iter().sum()
    }
}

impl Ulae {
    /// The classical technique under the 50/50 assumption.
    pub fn classical(ratio: Option<f64>) -> Self {
        Self::Classical {
            ratio,
            opening: 0.5,
        }
    }

    /// Kittel's refinement.
    pub fn kittel(ratio: Option<f64>) -> Self {
        Self::Kittel { ratio }
    }

    /// Conger and Nolibos' generalized approach.
    pub fn conger_nolibos(weights: UlaeWeights, ratio: Option<f64>) -> Self {
        Self::CongerNolibos { weights, ratio }
    }

    /// The weights of the basis, on reported, paid and closed claims.
    fn basis_weights(&self) -> UlaeWeights {
        match *self {
            Self::Classical { .. } => UlaeWeights::new(0.0, 1.0),
            Self::Kittel { .. } => UlaeWeights::new(0.5, 0.5),
            Self::CongerNolibos { weights, .. } => weights,
        }
    }

    /// The selected ratio, if any.
    fn selected(&self) -> Option<f64> {
        match *self {
            Self::Classical { ratio, .. }
            | Self::Kittel { ratio }
            | Self::CongerNolibos { ratio, .. } => ratio,
        }
    }

    /// Estimates unpaid ULAE: the ratios by calendar year, the ratio applied
    /// and the unpaid amount by row.
    pub fn estimate(&self, inputs: &UlaeInputs<'_>) -> Result<UlaeEstimate> {
        let weights = self.basis_weights();
        weights.check()?;
        if let Self::Classical { opening, .. } = *self
            && !(0.0..=1.0).contains(&opening)
        {
            return Err(Error::InvalidSetting {
                name: "opening",
                value: opening,
                expected: "a share from 0 to 1",
            });
        }
        let years = inputs.paid_ulae.len();
        if years == 0 {
            return Err(Error::Empty);
        }
        check_values("paid_ulae", inputs.paid_ulae, years)?;
        check_values("paid", inputs.paid, years)?;
        if weights.opening != 0.0 {
            check_values("reported", inputs.reported, years)?;
        }
        if weights.closing != 0.0 {
            check_values("closed", inputs.closed, years)?;
        }
        let basis: Vec<f64> = (0..years)
            .map(|y| {
                let reported = inputs.reported.get(y).unwrap_or(&0.0);
                let closed = inputs.closed.get(y).unwrap_or(&0.0);
                weights.opening * reported
                    + weights.maintaining * inputs.paid[y]
                    + weights.closing * closed
            })
            .collect();
        let ratios = inputs
            .paid_ulae
            .iter()
            .zip(&basis)
            .map(|(m, b)| if *b == 0.0 { f64::NAN } else { m / b })
            .collect();
        let total_paid: f64 = inputs.paid_ulae.iter().sum();
        let total_basis: f64 = basis.iter().sum();
        let total_ratio = total_paid / total_basis;
        let ratio = self.selected().unwrap_or(total_ratio);
        if !ratio.is_finite() {
            return Err(Error::InvalidSetting {
                name: "ratio",
                value: ratio,
                expected: "a finite ratio; with none selected, a basis that does not sum to 0",
            });
        }
        let unpaid = match *self {
            Self::Classical { opening, .. } => ratio_unpaid(inputs, ratio, 1.0 - opening)?,
            Self::Kittel { .. } => ratio_unpaid(inputs, ratio, 0.5)?,
            Self::CongerNolibos { weights, .. } => conger_nolibos_unpaid(inputs, ratio, weights)?,
        };
        let conger_nolibos = matches!(self, Self::CongerNolibos { .. });
        let claims = if conger_nolibos && !inputs.ultimate.is_empty() {
            check_values("ultimate", inputs.ultimate, inputs.ultimate.len())?;
            Some(inputs.ultimate.iter().sum::<f64>())
        } else {
            None
        };
        Ok(UlaeEstimate {
            basis,
            ratios,
            total_ratio,
            ratio,
            unpaid,
            expected_claims: claims.map(|l| ratio * l - total_paid),
            bornhuetter_ferguson: claims.map(|l| ratio * (l - total_basis)),
            development: claims.map(|l| total_paid * (l / total_basis - 1.0)),
        })
    }
}

/// `ratio x (known x (case + ibnr - pure) + pure)` by row, where `known` is
/// the share of the ratio on claims already reported. Without a pure IBNR
/// the whole IBNR takes the full ratio, Friedland's first form.
fn ratio_unpaid(inputs: &UlaeInputs<'_>, ratio: f64, known: f64) -> Result<Vec<f64>> {
    let rows = inputs.case_outstanding.len();
    if rows == 0 {
        return Err(Error::Empty);
    }
    check_values("case_outstanding", inputs.case_outstanding, rows)?;
    check_values("ibnr", inputs.ibnr, rows)?;
    if let Some(pure) = inputs.pure_ibnr {
        check_values("pure_ibnr", pure, rows)?;
    }
    Ok((0..rows)
        .map(|i| {
            let (case, ibnr) = (inputs.case_outstanding[i], inputs.ibnr[i]);
            let pure = inputs.pure_ibnr.map_or(ibnr, |p| p[i]);
            ratio * (known * (case + ibnr - pure) + pure)
        })
        .collect())
}

/// `W* x (U1 (L - R(t)) + U2 (L - P(t)) + U3 (L - C(t)))` by row, with `L -
/// R(t)` the pure IBNR, `L - P(t)` the case outstanding and IBNR and `L -
/// C(t)` the claims not yet closed.
fn conger_nolibos_unpaid(
    inputs: &UlaeInputs<'_>,
    ratio: f64,
    weights: UlaeWeights,
) -> Result<Vec<f64>> {
    let rows = inputs.case_outstanding.len();
    if rows == 0 {
        return Err(Error::Empty);
    }
    check_values("case_outstanding", inputs.case_outstanding, rows)?;
    check_values("ibnr", inputs.ibnr, rows)?;
    let pure = inputs.pure_ibnr.unwrap_or(&[]);
    if weights.opening != 0.0 || inputs.pure_ibnr.is_some() {
        check_values("pure_ibnr", pure, rows)?;
    }
    if weights.closing != 0.0 {
        check_values("unclosed", inputs.unclosed, rows)?;
    }
    Ok((0..rows)
        .map(|i| {
            let opening = pure.get(i).unwrap_or(&0.0);
            let closing = inputs.unclosed.get(i).unwrap_or(&0.0);
            let unpaid_claims = inputs.case_outstanding[i] + inputs.ibnr[i];
            ratio
                * (weights.opening * opening
                    + weights.maintaining * unpaid_claims
                    + weights.closing * closing)
        })
        .collect())
}

/// Errors unless `values` has `expected` entries, all finite.
fn check_values(column: &str, values: &[f64], expected: usize) -> Result<()> {
    if values.len() != expected {
        return Err(Error::LengthMismatch {
            column: column.into(),
            expected,
            found: values.len(),
        });
    }
    match values.iter().position(|v| !v.is_finite()) {
        Some(row) => Err(Error::NonFinite {
            column: column.into(),
            row,
        }),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangle::{DevelopmentColumn, Long};
    use prospicio_core::{Grain, Month};

    /// Two columns on the origins 2020 to 2022 and ages 12, 24 and 36.
    fn triangle(num: &[f64], den: &[f64]) -> Triangle {
        let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
            values: &[("num", num), ("den", den)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    fn book() -> Triangle {
        triangle(
            &[1.0, 3.0, 4.0, 2.0, 5.0, 3.0],
            &[10.0, 20.0, 20.0, 20.0, 25.0, 30.0],
        )
    }

    fn close(got: f64, want: f64) {
        assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
    }

    #[test]
    fn the_ratio_is_cellwise_and_its_factors_are_simple_averages() {
        let ultimate = [20.0, 25.0, 40.0];
        let fit = RatioDevelopment::default()
            .fit(&book(), "num", "den", &ultimate)
            .unwrap();
        assert_eq!(fit.ratios[0], vec![Some(0.1), Some(0.15), Some(0.2)]);
        assert_eq!(fit.ratios[2], vec![Some(0.1), None, None]);
        close(fit.factors[0], (1.5 + 2.0) / 2.0);
        close(fit.factors[1], 0.2 / 0.15);
        close(fit.cdf[0], 1.75 * 0.2 / 0.15);
        assert_eq!(fit.latest, vec![4.0, 5.0, 3.0]);
        for (o, u) in ultimate.iter().enumerate() {
            close(fit.ultimate[o], fit.projected_ratio[o] * u);
        }
        close(fit.total_unpaid(), fit.unpaid().iter().sum());
    }

    #[test]
    fn a_tail_scales_the_ratio() {
        let ultimate = [20.0, 25.0, 40.0];
        let base = RatioDevelopment::default()
            .fit(&book(), "num", "den", &ultimate)
            .unwrap();
        let tailed = RatioDevelopment {
            tail: 1.1.into(),
            ..Default::default()
        }
        .fit(&book(), "num", "den", &ultimate)
        .unwrap();
        close(tailed.projected_ratio[2], base.projected_ratio[2] * 1.1);
    }

    #[test]
    fn a_selected_ratio_replaces_the_projection() {
        let fit = RatioDevelopment {
            selected: vec![(Period::year(2022), 0.3)],
            ..Default::default()
        }
        .fit(&book(), "num", "den", &[20.0, 25.0, 40.0])
        .unwrap();
        assert_eq!(fit.selected_ratio[2], 0.3);
        assert_eq!(fit.selected_ratio[1], fit.projected_ratio[1]);
        close(fit.ultimate[2], 12.0);
    }

    #[test]
    fn a_selected_ratio_must_name_one_origin_once() {
        let unknown = RatioDevelopment {
            selected: vec![(Period::year(2019), 0.3)],
            ..Default::default()
        };
        assert!(matches!(
            unknown.fit(&book(), "num", "den", &[1.0, 1.0, 1.0]),
            Err(Error::Apriori(_))
        ));
        let twice = RatioDevelopment {
            selected: vec![(Period::year(2021), 0.3), (Period::year(2021), 0.4)],
            ..Default::default()
        };
        assert!(matches!(
            twice.fit(&book(), "num", "den", &[1.0, 1.0, 1.0]),
            Err(Error::Apriori(_))
        ));
    }

    #[test]
    fn a_zero_denominator_has_no_ratio() {
        let tri = triangle(
            &[1.0, 3.0, 4.0, 2.0, 5.0, 3.0],
            &[0.0, 20.0, 20.0, 20.0, 25.0, 30.0],
        );
        let fit = RatioDevelopment::default()
            .fit(&tri, "num", "den", &[20.0, 25.0, 40.0])
            .unwrap();
        assert_eq!(fit.ratios[0][0], None);
        close(fit.factors[0], 2.0);
    }

    #[test]
    fn an_origin_with_no_ratio_is_an_error_naming_it() {
        let tri = triangle(
            &[1.0, 3.0, 4.0, 2.0, 5.0, 3.0],
            &[10.0, 20.0, 20.0, 20.0, 25.0, 0.0],
        );
        assert_eq!(
            RatioDevelopment::default().fit(&tri, "num", "den", &[20.0, 25.0, 40.0]),
            Err(Error::EmptyOrigin("2022".into()))
        );
    }

    #[test]
    fn the_ultimates_line_up_with_the_origins() {
        assert!(matches!(
            RatioDevelopment::default().fit(&book(), "num", "den", &[1.0]),
            Err(Error::LengthMismatch { .. })
        ));
        assert!(matches!(
            RatioDevelopment::default().fit(&book(), "num", "den", &[1.0, f64::NAN, 1.0]),
            Err(Error::NonFinite { .. })
        ));
    }

    #[test]
    fn additive_factors_add_to_ultimate() {
        let additive = RatioFactors::Additive {
            selected: vec![],
            tail: 0.01,
        };
        let fit = RatioDevelopment {
            factors: additive,
            ..Default::default()
        }
        .fit(&book(), "num", "den", &[20.0, 25.0, 40.0])
        .unwrap();
        close(fit.factors[0], (0.05 + 0.1) / 2.0);
        close(fit.factors[1], 0.05);
        close(fit.cdf[2], 0.01);
        close(fit.cdf[0], 0.075 + 0.05 + 0.01);
        close(fit.projected_ratio[2], 0.1 + 0.135);
    }

    #[test]
    fn additive_differences_can_be_selected_and_dropped() {
        let fit = RatioDevelopment {
            factors: RatioFactors::Additive {
                selected: vec![(24, 0.02)],
                tail: 0.0,
            },
            development: Development {
                average: Average::Simple,
                drop_high: 1,
                ..Default::default()
            },
            ..Default::default()
        }
        .fit(&book(), "num", "den", &[20.0, 25.0, 40.0])
        .unwrap();
        close(fit.averages[0], 0.05);
        close(fit.averages[1], 0.05);
        close(fit.factors[1], 0.02);
    }

    #[test]
    fn additive_factors_refuse_what_only_multiplies() {
        let additive = RatioFactors::Additive {
            selected: vec![],
            tail: 0.0,
        };
        let volume = RatioDevelopment {
            factors: additive.clone(),
            development: Development::default(),
            ..Default::default()
        };
        let tail = RatioDevelopment {
            factors: additive,
            tail: 1.1.into(),
            ..Default::default()
        };
        for model in [volume, tail] {
            assert!(matches!(
                model.fit(&book(), "num", "den", &[1.0, 1.0, 1.0]),
                Err(Error::NotSupported { .. })
            ));
        }
        let off_grid = RatioDevelopment {
            factors: RatioFactors::Additive {
                selected: vec![(36, 0.1)],
                tail: 0.0,
            },
            ..Default::default()
        };
        assert!(matches!(
            off_grid.fit(&book(), "num", "den", &[1.0, 1.0, 1.0]),
            Err(Error::Pattern { .. })
        ));
    }

    /// Calendar history of two years and unpaid claims of two accident years.
    fn inputs<'a>() -> UlaeInputs<'a> {
        UlaeInputs {
            paid_ulae: &[10.0, 20.0],
            paid: &[100.0, 300.0],
            reported: &[200.0, 400.0],
            case_outstanding: &[50.0, 30.0],
            ibnr: &[40.0, 20.0],
            ..Default::default()
        }
    }

    #[test]
    fn the_classical_technique_takes_half_of_case_and_all_of_ibnr() {
        let fit = Ulae::classical(Some(0.1)).estimate(&inputs()).unwrap();
        assert_eq!(fit.basis, vec![100.0, 300.0]);
        close(fit.ratios[1], 20.0 / 300.0);
        close(fit.total_ratio, 30.0 / 400.0);
        close(fit.unpaid[0], 0.1 * (25.0 + 40.0));
        close(fit.total(), 0.1 * (25.0 + 40.0 + 15.0 + 20.0));
        assert_eq!(fit.expected_claims, None);
    }

    #[test]
    fn with_no_selection_the_ratio_of_the_totals_applies() {
        let fit = Ulae::kittel(None).estimate(&inputs()).unwrap();
        assert_eq!(fit.basis, vec![150.0, 350.0]);
        close(fit.ratio, 30.0 / 500.0);
    }

    #[test]
    fn pure_ibnr_takes_the_full_ratio_alone() {
        let pure = [4.0, 2.0];
        let fit = Ulae::classical(Some(0.1))
            .estimate(&UlaeInputs {
                pure_ibnr: Some(&pure),
                ..inputs()
            })
            .unwrap();
        close(fit.unpaid[0], 0.1 * (0.5 * (50.0 + 36.0) + 4.0));
    }

    #[test]
    fn the_erratum_opening_share_puts_more_on_case_outstanding() {
        let fit = Ulae::Classical {
            ratio: Some(0.1),
            opening: 0.25,
        }
        .estimate(&inputs())
        .unwrap();
        close(fit.unpaid[0], 0.1 * (0.75 * 50.0 + 40.0));
    }

    #[test]
    fn kittel_is_conger_and_nolibos_at_equal_weights_on_the_basis() {
        let kittel = Ulae::kittel(None).estimate(&inputs()).unwrap();
        let general = Ulae::conger_nolibos(UlaeWeights::new(0.5, 0.5), None)
            .estimate(&UlaeInputs {
                pure_ibnr: Some(&[0.0, 0.0]),
                ..inputs()
            })
            .unwrap();
        assert_eq!(kittel.basis, general.basis);
    }

    #[test]
    fn conger_and_nolibos_close_on_their_third_share() {
        let weights = UlaeWeights {
            opening: 0.5,
            maintaining: 0.25,
            closing: 0.25,
        };
        let model = Ulae::conger_nolibos(weights, Some(0.1));
        let closed = [50.0, 150.0];
        let (pure, unclosed) = ([5.0, 3.0], [70.0, 40.0]);
        let fit = model
            .estimate(&UlaeInputs {
                closed: &closed,
                pure_ibnr: Some(&pure),
                unclosed: &unclosed,
                ultimate: &[700.0, 800.0],
                ..inputs()
            })
            .unwrap();
        assert_eq!(fit.basis[0], 0.5 * 200.0 + 0.25 * 100.0 + 0.25 * 50.0);
        close(fit.unpaid[0], 0.1 * (2.5 + 0.25 * 90.0 + 0.25 * 70.0));
        assert_eq!(fit.expected_claims, Some(0.1 * 1500.0 - 30.0));
        let basis: f64 = fit.basis.iter().sum();
        assert_eq!(fit.bornhuetter_ferguson, Some(0.1 * (1500.0 - basis)));
        assert_eq!(fit.development, Some(30.0 * (1500.0 / basis - 1.0)));
        let missing = model.estimate(&UlaeInputs {
            closed: &closed,
            pure_ibnr: Some(&pure),
            ..inputs()
        });
        assert!(matches!(missing, Err(Error::LengthMismatch { .. })));
    }

    #[test]
    fn conger_and_nolibos_need_the_pure_ibnr_for_their_opening_share() {
        let model = Ulae::conger_nolibos(UlaeWeights::new(0.6, 0.4), Some(0.1));
        assert!(matches!(
            model.estimate(&inputs()),
            Err(Error::LengthMismatch { .. })
        ));
    }

    #[test]
    fn weights_are_shares_summing_to_one() {
        for weights in [UlaeWeights::new(0.6, 0.6), UlaeWeights::new(1.5, -0.5)] {
            assert!(matches!(
                Ulae::conger_nolibos(weights, None).estimate(&inputs()),
                Err(Error::InvalidSetting { .. })
            ));
        }
    }

    #[test]
    fn a_ratio_needs_a_basis_or_a_selection() {
        let zero = UlaeInputs {
            paid: &[0.0, 0.0],
            ..inputs()
        };
        assert!(matches!(
            Ulae::classical(None).estimate(&zero),
            Err(Error::InvalidSetting { .. })
        ));
        let fit = Ulae::classical(Some(0.1)).estimate(&zero).unwrap();
        assert!(fit.ratios[0].is_nan());
    }

    #[test]
    fn inputs_are_checked() {
        let model = Ulae::classical(None);
        let short = UlaeInputs {
            paid: &[1.0],
            ..inputs()
        };
        assert!(matches!(
            model.estimate(&short),
            Err(Error::LengthMismatch { .. })
        ));
        let nan = UlaeInputs {
            ibnr: &[f64::NAN, 1.0],
            ..inputs()
        };
        assert!(matches!(model.estimate(&nan), Err(Error::NonFinite { .. })));
        assert!(matches!(
            model.estimate(&UlaeInputs::default()),
            Err(Error::Empty)
        ));
        let opening = Ulae::Classical {
            ratio: None,
            opening: 1.5,
        };
        assert!(matches!(
            opening.estimate(&inputs()),
            Err(Error::InvalidSetting { .. })
        ));
    }
}
