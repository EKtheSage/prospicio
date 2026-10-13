//! Experience rating from individual large losses: the record a
//! reinsurance submission brings, trended to the prospective period, the
//! as-if loss to each layer by accident year, and the burning cost. See
//! `docs/design/experience-rating.md`.
//!
//! The steps follow Clark's *Basics of Reinsurance Pricing* (CAS study
//! note, 2014), section on per-risk experience rating:
//!
//! 0. Convert each loss to the pricing currency, at a stated rate or a rate
//!    per loss ([`Experience::with_fx`]).
//! 1. Develop each known loss to its ultimate value (IBNER), by a factor
//!    per loss or a [`DevelopmentPattern`] at its age at the valuation
//!    date, if the record is not developed by year instead
//!    ([`Experience::prepare_developed`]).
//! 2. Trend each loss from its accident date to the average accident date
//!    of the prospective period ([`Experience::prepare`]); dates are
//!    decimal years, as in [`ratemaking`](crate::ratemaking).
//! 3. Apply each year's treaty to the trended losses of that year
//!    ([`Prepared::burning_cost`]), with any annual terms.
//! 4. Develop each year's layer total: with Clark's excess development
//!    factor, or, when the losses were developed one by one, with a factor
//!    for the claims not yet reported (count IBNR,
//!    [`YearBasis::from_count_pattern`]).
//! 5. Divide by the year's on-level subject premium: the loss cost by
//!    year, and in total the developed losses over the premium, summed.
//!
//! A treaty with an index (stability) clause moves its retention and limit
//! with an index; [`Prepared::burning_cost_indexed`] applies each loss to
//! the terms its own index ratio sets ([`IndexClause`]).
//!
//! Losses reported above a threshold in nominal terms leave the older
//! years short of losses that trend above it. With a threshold `T`, the
//! prepared record is complete above `T` times the largest trend factor,
//! and losses trended below that are dropped, so every year is compared
//! on the same basis ([`Prepared::complete_above`]). The test is on the
//! reported amount trended, not the developed one: the record is complete
//! for losses reported above the threshold, whatever they develop to, and
//! in the currency they were reported in.

use prospicio_aggregate::Tower;
use prospicio_core::{Error, Result};

/// One large loss: an identifier, its accident date as a decimal year, and
/// its amount (incurred at the latest valuation).
#[derive(Debug, Clone, PartialEq)]
pub struct LargeLoss {
    pub id: String,
    pub accident: f64,
    pub amount: f64,
}

/// How losses are trended to the prospective period.
#[derive(Debug, Clone, PartialEq)]
pub enum Trend {
    /// A constant annual severity trend `rate`, from each accident date to
    /// the date `to`: `(1 + rate)^(to − accident)`.
    Rate { rate: f64, to: f64 },
    /// A factor per loss, in the record's order (from an index, or as a
    /// source exhibit prints them).
    Factors(Vec<f64>),
}

/// A record of large losses; see the [module](self).
#[derive(Debug, Clone, PartialEq)]
pub struct Experience {
    losses: Vec<LargeLoss>,
    threshold: f64,
    valuation: Option<f64>,
    fx: Vec<f64>,
}

impl Experience {
    /// From losses with finite accident dates and non-negative amounts.
    pub fn new(losses: Vec<LargeLoss>) -> Result<Self> {
        if losses.is_empty() {
            return Err(Error::Data("an experience needs at least one loss".into()));
        }
        for l in &losses {
            if !l.accident.is_finite() {
                return Err(Error::InvalidParameter {
                    name: "accident",
                    value: l.accident,
                    reason: "must be finite",
                });
            }
            if !(l.amount.is_finite() && l.amount >= 0.0) {
                return Err(Error::InvalidParameter {
                    name: "amount",
                    value: l.amount,
                    reason: "must be finite and non-negative",
                });
            }
        }
        Ok(Self {
            losses,
            threshold: 0.0,
            valuation: None,
            fx: Vec::new(),
        })
    }

    /// The nominal reporting threshold: losses were reported when at least
    /// `threshold` at their own date.
    pub fn with_threshold(mut self, threshold: f64) -> Result<Self> {
        if !(threshold.is_finite() && threshold >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "threshold",
                value: threshold,
                reason: "must be finite and non-negative",
            });
        }
        self.threshold = threshold;
        Ok(self)
    }

    /// The date the amounts were valued at, a decimal year; a loss's age
    /// for a [`DevelopmentPattern`] is the valuation date less its accident
    /// date.
    pub fn with_valuation(mut self, valuation: f64) -> Result<Self> {
        if !valuation.is_finite() {
            return Err(Error::InvalidParameter {
                name: "valuation",
                value: valuation,
                reason: "must be finite",
            });
        }
        if let Some(l) = self.losses.iter().find(|l| l.accident > valuation) {
            return Err(Error::Data(format!(
                "loss {} has an accident date after the valuation date",
                l.id
            )));
        }
        self.valuation = Some(valuation);
        Ok(self)
    }

    /// The valuation date, if set.
    pub fn valuation(&self) -> Option<f64> {
        self.valuation
    }

    /// Converts each loss to the pricing currency: amounts are multiplied
    /// by the rate (units of the pricing currency per unit of the loss's),
    /// before development and trend, so a trend `rate` is then the pricing
    /// currency's. The threshold and [`Prepared::complete_above`] stay in
    /// the currency the losses were reported in.
    pub fn with_fx(mut self, fx: Fx) -> Result<Self> {
        let rates = match fx {
            Fx::Rate(r) => vec![r; self.losses.len()],
            Fx::PerLoss(r) => {
                if r.len() != self.losses.len() {
                    return Err(Error::Data(format!(
                        "{} exchange rates for {} losses",
                        r.len(),
                        self.losses.len()
                    )));
                }
                r
            }
        };
        if let Some(&bad) = rates.iter().find(|r| !(r.is_finite() && **r > 0.0)) {
            return Err(Error::InvalidParameter {
                name: "fx",
                value: bad,
                reason: "must be finite and positive",
            });
        }
        self.fx = rates;
        Ok(self)
    }

    /// The losses, in the order given.
    pub fn losses(&self) -> &[LargeLoss] {
        &self.losses
    }

    /// The reporting threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Each loss trended to the prospective period.
    ///
    /// ```
    /// use prospicio_pricing::experience::{Experience, LargeLoss, Trend};
    ///
    /// let losses = vec![
    ///     LargeLoss { id: "a".into(), accident: 1995.0, amount: 200.0 },
    ///     LargeLoss { id: "b".into(), accident: 1996.0, amount: 300.0 },
    /// ];
    /// let p = Experience::new(losses).unwrap()
    ///     .prepare(&Trend::Rate { rate: 0.05, to: 1997.0 }).unwrap();
    /// assert!((p.losses()[0].trended - 200.0 * 1.05f64.powi(2)).abs() < 1e-9);
    /// assert_eq!(p.losses()[1].year, 1996);
    /// ```
    pub fn prepare(&self, trend: &Trend) -> Result<Prepared> {
        self.prepare_developed(trend, &Development::Factors(vec![1.0; self.losses.len()]))
    }

    /// Each loss developed to ultimate (IBNER), then trended to the
    /// prospective period.
    ///
    /// ```
    /// use prospicio_pricing::experience::{
    ///     Development, DevelopmentPattern, Experience, LargeLoss, Trend,
    /// };
    ///
    /// let losses = vec![
    ///     LargeLoss { id: "a".into(), accident: 2020.5, amount: 200.0 },
    ///     LargeLoss { id: "b".into(), accident: 2022.5, amount: 300.0 },
    /// ];
    /// // To ultimate: 1.5 at age 2, 1.1 at age 4 (ages in years).
    /// let pattern = DevelopmentPattern::new(vec![2.0, 4.0], vec![1.5, 1.1]).unwrap();
    /// let p = Experience::new(losses).unwrap()
    ///     .with_valuation(2024.5).unwrap()
    ///     .prepare_developed(&Trend::Factors(vec![1.0, 1.0]), &Development::Pattern(pattern))
    ///     .unwrap();
    /// assert!((p.losses()[0].ultimate - 220.0).abs() < 1e-12); // age 4
    /// assert!((p.losses()[1].ultimate - 450.0).abs() < 1e-12); // age 2
    /// ```
    pub fn prepare_developed(&self, trend: &Trend, development: &Development) -> Result<Prepared> {
        let developments: Vec<f64> = match development {
            Development::Factors(f) => {
                if f.len() != self.losses.len() {
                    return Err(Error::Data(format!(
                        "{} development factors for {} losses",
                        f.len(),
                        self.losses.len()
                    )));
                }
                if let Some(&bad) = f.iter().find(|x| !(x.is_finite() && **x > 0.0)) {
                    return Err(Error::InvalidParameter {
                        name: "development",
                        value: bad,
                        reason: "must be finite and positive",
                    });
                }
                f.clone()
            }
            Development::Pattern(pattern) => {
                let valuation = self.valuation.ok_or_else(|| {
                    Error::Data("development by age needs a valuation date".into())
                })?;
                self.losses
                    .iter()
                    .map(|l| pattern.at(valuation - l.accident))
                    .collect()
            }
        };
        let factors: Vec<f64> = match trend {
            Trend::Rate { rate, to } => {
                if !(rate.is_finite() && *rate > -1.0 && to.is_finite()) {
                    return Err(Error::InvalidParameter {
                        name: "rate",
                        value: *rate,
                        reason: "must be finite and above -1, with a finite date",
                    });
                }
                self.losses
                    .iter()
                    .map(|l| (1.0 + rate).powf(to - l.accident))
                    .collect()
            }
            Trend::Factors(f) => {
                if f.len() != self.losses.len() {
                    return Err(Error::Data(format!(
                        "{} trend factors for {} losses",
                        f.len(),
                        self.losses.len()
                    )));
                }
                if let Some(&bad) = f.iter().find(|x| !(x.is_finite() && **x > 0.0)) {
                    return Err(Error::InvalidParameter {
                        name: "factors",
                        value: bad,
                        reason: "must be finite and positive",
                    });
                }
                f.clone()
            }
        };
        let complete_above = self.threshold * factors.iter().copied().fold(0.0, f64::max);
        let losses = self
            .losses
            .iter()
            .enumerate()
            .zip(factors.iter().zip(&developments))
            .map(|((k, l), (&factor, &development))| {
                let fx = self.fx.get(k).copied().unwrap_or(1.0);
                PreparedLoss {
                    id: l.id.clone(),
                    year: l.accident.floor() as i32,
                    accident: l.accident,
                    amount: l.amount,
                    fx,
                    development,
                    ultimate: l.amount * fx * development,
                    factor,
                    trended: l.amount * fx * development * factor,
                }
            })
            .collect();
        Ok(Prepared {
            losses,
            complete_above,
            valuation: self.valuation,
        })
    }
}

/// How losses are converted to the pricing currency, for
/// [`Experience::with_fx`].
#[derive(Debug, Clone, PartialEq)]
pub enum Fx {
    /// One stated rate for every loss.
    Rate(f64),
    /// A rate per loss, in the record's order (historical rates).
    PerLoss(Vec<f64>),
}

/// How an index clause moves a layer's retention and limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    /// Within the franchise nothing moves; beyond it, the terms move by the
    /// full index ratio, up or down.
    Full,
    /// Severe inflation clause: the terms move only by the rise beyond the
    /// franchise, `r / (1 + franchise)`, and never down.
    Severe,
}

/// An index (stability) clause: the treaty's retention and limit move with
/// an index, by the ratio `r` of the index at a loss's payment to the
/// index at the treaty's base date, within a franchise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexClause {
    franchise: f64,
    kind: IndexKind,
}

impl IndexClause {
    /// A clause with a franchise, a fraction such as 0.1 for 10%.
    pub fn new(franchise: f64, kind: IndexKind) -> Result<Self> {
        if !(franchise.is_finite() && franchise >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "franchise",
                value: franchise,
                reason: "must be finite and non-negative",
            });
        }
        Ok(Self { franchise, kind })
    }

    /// The franchise.
    pub fn franchise(&self) -> f64 {
        self.franchise
    }

    /// The kind of clause.
    pub fn kind(&self) -> IndexKind {
        self.kind
    }

    /// The factor the terms move by at index ratio `ratio`.
    ///
    /// ```
    /// use prospicio_pricing::experience::{IndexClause, IndexKind};
    ///
    /// let full = IndexClause::new(0.1, IndexKind::Full).unwrap();
    /// assert_eq!(full.adjustment(1.08), 1.0); // within the franchise
    /// assert_eq!(full.adjustment(1.25), 1.25); // all of it beyond
    /// let severe = IndexClause::new(0.1, IndexKind::Severe).unwrap();
    /// assert!((severe.adjustment(1.21) - 1.1).abs() < 1e-15); // the excess
    /// assert_eq!(severe.adjustment(0.8), 1.0);
    /// ```
    pub fn adjustment(&self, ratio: f64) -> f64 {
        let f = self.franchise;
        match self.kind {
            IndexKind::Full if (ratio - 1.0).abs() <= f => 1.0,
            IndexKind::Full => ratio,
            IndexKind::Severe if ratio > 1.0 + f => ratio / (1.0 + f),
            IndexKind::Severe => 1.0,
        }
    }
}

/// How known losses are developed to ultimate (IBNER), for
/// [`Experience::prepare_developed`].
#[derive(Debug, Clone, PartialEq)]
pub enum Development {
    /// A factor to ultimate per loss, in the record's order.
    Factors(Vec<f64>),
    /// Factors to ultimate by age, each loss at its age at the valuation
    /// date ([`Experience::with_valuation`]).
    Pattern(DevelopmentPattern),
}

/// Development factors to ultimate by age, in years since the accident
/// date (or in any unit, used consistently).
///
/// [`at`](Self::at) interpolates linearly in age between the given ages,
/// takes the first factor at younger ages and the last at older ones; give
/// a pattern that starts at the youngest age in the record and ends where
/// development stops (a last factor of 1, or a tail).
#[derive(Debug, Clone, PartialEq)]
pub struct DevelopmentPattern {
    ages: Vec<f64>,
    factors: Vec<f64>,
}

/// One valuation of one claim: its amount at an age, for
/// [`DevelopmentPattern::from_history`].
#[derive(Debug, Clone, PartialEq)]
pub struct ClaimValuation {
    pub id: String,
    pub age: f64,
    pub amount: f64,
}

impl DevelopmentPattern {
    /// From strictly increasing finite ages and finite positive factors to
    /// ultimate at each.
    pub fn new(ages: Vec<f64>, factors: Vec<f64>) -> Result<Self> {
        if ages.is_empty() || ages.len() != factors.len() {
            return Err(Error::Data(format!(
                "{} ages and {} factors; a pattern needs at least one of each, as many ages as factors",
                ages.len(),
                factors.len()
            )));
        }
        if ages.iter().any(|a| !a.is_finite()) || ages.windows(2).any(|w| w[1] <= w[0]) {
            return Err(Error::Data(
                "ages must be finite and strictly increasing".into(),
            ));
        }
        if let Some(&bad) = factors.iter().find(|f| !(f.is_finite() && **f > 0.0)) {
            return Err(Error::InvalidParameter {
                name: "factors",
                value: bad,
                reason: "must be finite and positive",
            });
        }
        Ok(Self { ages, factors })
    }

    /// Estimated from each claim's valuations at the given ages: the
    /// age-to-age factor from `ages[k]` to `ages[k + 1]` is the sum of the
    /// amounts at `ages[k + 1]` over the sum at `ages[k]`, over the claims
    /// valued at both with a positive amount at `ages[k]` (a claim first
    /// seen later was not yet reported, which is IBNR, not IBNER). The
    /// factors to ultimate are their products, 1 at the last age (no
    /// tail). With every claim valued at every age up to its latest, this
    /// is the volume-weighted chain ladder on the summed triangle.
    ///
    /// Valuations at an age not in `ages` are ignored.
    ///
    /// ```
    /// use prospicio_pricing::experience::{ClaimValuation, DevelopmentPattern};
    ///
    /// let v = |id: &str, age: f64, amount: f64| ClaimValuation { id: id.into(), age, amount };
    /// let history = [v("a", 1.0, 100.0), v("a", 2.0, 150.0), v("b", 1.0, 50.0), v("b", 2.0, 50.0)];
    /// let p = DevelopmentPattern::from_history(&history, &[1.0, 2.0]).unwrap();
    /// assert!((p.factors()[0] - 200.0 / 150.0).abs() < 1e-15);
    /// assert_eq!(p.factors()[1], 1.0);
    /// ```
    pub fn from_history(valuations: &[ClaimValuation], ages: &[f64]) -> Result<Self> {
        use std::collections::HashMap;
        if ages.len() < 2 {
            return Err(Error::Data(
                "estimating a pattern needs at least two ages".into(),
            ));
        }
        if ages.iter().any(|a| !a.is_finite()) || ages.windows(2).any(|w| w[1] <= w[0]) {
            return Err(Error::Data(
                "ages must be finite and strictly increasing".into(),
            ));
        }
        let slot = |age: f64| {
            ages.iter()
                .position(|&a| (a - age).abs() <= 1e-9 * a.abs().max(1.0))
        };
        let mut by_claim: HashMap<&str, Vec<Option<f64>>> = HashMap::new();
        for v in valuations {
            if !(v.amount.is_finite() && v.amount >= 0.0) {
                return Err(Error::InvalidParameter {
                    name: "amount",
                    value: v.amount,
                    reason: "must be finite and non-negative",
                });
            }
            let Some(k) = slot(v.age) else { continue };
            let row = by_claim
                .entry(v.id.as_str())
                .or_insert_with(|| vec![None; ages.len()]);
            if row[k].is_some() {
                return Err(Error::Data(format!(
                    "claim {} is valued twice at age {}",
                    v.id, ages[k]
                )));
            }
            row[k] = Some(v.amount);
        }
        let mut links = Vec::with_capacity(ages.len() - 1);
        for k in 0..ages.len() - 1 {
            let (mut from, mut to) = (0.0, 0.0);
            for row in by_claim.values() {
                if let (Some(a), Some(b)) = (row[k], row[k + 1])
                    && a > 0.0
                {
                    from += a;
                    to += b;
                }
            }
            if from <= 0.0 {
                return Err(Error::Data(format!(
                    "no claim is valued at both ages {} and {}",
                    ages[k],
                    ages[k + 1]
                )));
            }
            links.push(to / from);
        }
        let mut factors = vec![1.0; ages.len()];
        for k in (0..links.len()).rev() {
            factors[k] = factors[k + 1] * links[k];
        }
        Self::new(ages.to_vec(), factors)
    }

    /// The ages.
    pub fn ages(&self) -> &[f64] {
        &self.ages
    }

    /// The factors to ultimate at each age.
    pub fn factors(&self) -> &[f64] {
        &self.factors
    }

    /// The age-to-age factors, one fewer than the ages.
    pub fn age_to_age(&self) -> Vec<f64> {
        self.factors.windows(2).map(|w| w[0] / w[1]).collect()
    }

    /// The factor to ultimate at `age`: linear between the given ages, the
    /// first factor before the first age and the last after the last.
    pub fn at(&self, age: f64) -> f64 {
        let (a, f) = (&self.ages, &self.factors);
        if age <= a[0] {
            return f[0];
        }
        if age >= a[a.len() - 1] {
            return f[f.len() - 1];
        }
        let k = a.partition_point(|&x| x <= age) - 1;
        let w = (age - a[k]) / (a[k + 1] - a[k]);
        f[k] + w * (f[k + 1] - f[k])
    }
}

/// One loss trended to the prospective period.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedLoss {
    pub id: String,
    /// The accident year, the whole part of the accident date.
    pub year: i32,
    pub accident: f64,
    /// The loss as reported, in its own currency.
    pub amount: f64,
    /// The exchange rate to the pricing currency; 1 without conversion.
    pub fx: f64,
    /// The development factor to ultimate applied (IBNER); 1 when the
    /// record is developed by year instead.
    pub development: f64,
    /// `amount × fx × development`.
    pub ultimate: f64,
    /// The trend factor applied.
    pub factor: f64,
    /// `ultimate × factor`.
    pub trended: f64,
}

/// The trended record, from [`Experience::prepare`].
#[derive(Debug, Clone, PartialEq)]
pub struct Prepared {
    losses: Vec<PreparedLoss>,
    complete_above: f64,
    valuation: Option<f64>,
}

/// An accident year's exposure base and excess development factor, for
/// [`Prepared::burning_cost`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YearBasis {
    pub year: i32,
    /// On-level subject premium (or another exposure base).
    pub premium: f64,
    /// The development factor for the year's layer losses: Clark's excess
    /// development factor when the losses were not developed one by one,
    /// or, after [`Development`] per loss, only the factor for claims not
    /// yet reported (count IBNR), so that known claims' development is not
    /// counted twice.
    pub development: f64,
}

impl YearBasis {
    /// A year whose development factor is a count pattern's factor at the
    /// year's average age, `valuation − (year + ½)`: the factor for the
    /// claims not yet reported, after each known claim was developed.
    ///
    /// ```
    /// use prospicio_pricing::experience::{DevelopmentPattern, YearBasis};
    ///
    /// let counts = DevelopmentPattern::new(vec![1.0, 3.0], vec![1.4, 1.0]).unwrap();
    /// let y = YearBasis::from_count_pattern(2022, 1000.0, &counts, 2024.5).unwrap();
    /// assert!((y.development - 1.2).abs() < 1e-12); // age 2, halfway
    /// ```
    pub fn from_count_pattern(
        year: i32,
        premium: f64,
        counts: &DevelopmentPattern,
        valuation: f64,
    ) -> Result<Self> {
        let age = valuation - (f64::from(year) + 0.5);
        if !(age.is_finite() && age >= 0.0) {
            return Err(Error::Data(format!(
                "year {year} is after the valuation date {valuation}"
            )));
        }
        Ok(Self {
            year,
            premium,
            development: counts.at(age),
        })
    }
}

impl Prepared {
    /// Every loss, trended, in the record's order.
    pub fn losses(&self) -> &[PreparedLoss] {
        &self.losses
    }

    /// The valuation date of the record, if it had one.
    pub fn valuation(&self) -> Option<f64> {
        self.valuation
    }

    /// The trended size above which every year's record is complete: the
    /// reporting threshold times the largest trend factor. Losses trended
    /// below it, as reported, are left out of
    /// [`burning_cost`](Self::burning_cost).
    pub fn complete_above(&self) -> f64 {
        self.complete_above
    }

    /// The as-if loss to each layer by accident year, developed and
    /// divided by the premium: `towers[k]` is the treaty in force in
    /// `years[k]` (pass the same tower for every year to price the
    /// proposed one). Each year's trended losses at or above
    /// [`complete_above`](Self::complete_above) go through its tower,
    /// annual terms included; the layer totals are then developed.
    ///
    /// ```
    /// use prospicio_aggregate::{Layer, Tower};
    /// use prospicio_pricing::experience::{Experience, LargeLoss, Trend, YearBasis};
    ///
    /// let losses = vec![
    ///     LargeLoss { id: "a".into(), accident: 1995.5, amount: 150.0 },
    ///     LargeLoss { id: "b".into(), accident: 1996.5, amount: 400.0 },
    /// ];
    /// let p = Experience::new(losses).unwrap()
    ///     .prepare(&Trend::Factors(vec![1.0, 1.0])).unwrap();
    /// let tower = Tower::new(vec![Layer::xol("200 xs 100", 200.0, 100.0).unwrap()]).unwrap();
    /// let years = [
    ///     YearBasis { year: 1995, premium: 1000.0, development: 1.0 },
    ///     YearBasis { year: 1996, premium: 1000.0, development: 1.5 },
    /// ];
    /// let b = p.burning_cost(&[tower.clone(), tower], &years).unwrap();
    /// assert_eq!(b.years[0].layer_loss, vec![50.0]);
    /// assert_eq!(b.years[1].developed, vec![300.0]); // 200 × 1.5
    /// assert!((b.loss_cost[0] - 350.0 / 2000.0).abs() < 1e-15);
    /// ```
    pub fn burning_cost(&self, towers: &[Tower], years: &[YearBasis]) -> Result<BurningCost> {
        self.cost(towers, years, None)
    }

    /// [`burning_cost`](Self::burning_cost) under an index clause: each
    /// loss meets the retention and limit moved by the clause's adjustment
    /// at its own index ratio, `ratios[k]` for loss `k` in the record's
    /// order (the index at its payment over the index at the treaty's base
    /// date). With every term per occurrence, terms moved by `a` cede
    /// `a × ceded(x / a)`, which is how each loss is applied.
    ///
    /// Annual terms (an aggregate deductible or limit, a corridor) span
    /// losses with different ratios and are not indexed: a tower with them
    /// is an error, as is a surplus treaty.
    ///
    /// ```
    /// use prospicio_aggregate::{Layer, Tower};
    /// use prospicio_pricing::experience::{
    ///     Experience, IndexClause, IndexKind, LargeLoss, Trend, YearBasis,
    /// };
    ///
    /// let p = Experience::new(vec![LargeLoss { id: "a".into(), accident: 2020.5, amount: 700.0 }])
    ///     .unwrap()
    ///     .prepare(&Trend::Factors(vec![1.0]))
    ///     .unwrap();
    /// let tower = Tower::new(vec![Layer::xol("400 xs 100", 400.0, 100.0).unwrap()]).unwrap();
    /// let year = [YearBasis { year: 2020, premium: 1000.0, development: 1.0 }];
    /// let clause = IndexClause::new(0.1, IndexKind::Full).unwrap();
    /// // Index up 25%: 500 xs 125, so the 700 loss cedes 500.
    /// let b = p.burning_cost_indexed(&[tower], &year, &clause, &[1.25]).unwrap();
    /// assert!((b.years[0].layer_loss[0] - 500.0).abs() < 1e-12);
    /// ```
    pub fn burning_cost_indexed(
        &self,
        towers: &[Tower],
        years: &[YearBasis],
        clause: &IndexClause,
        ratios: &[f64],
    ) -> Result<BurningCost> {
        if ratios.len() != self.losses.len() {
            return Err(Error::Data(format!(
                "{} index ratios for {} losses",
                ratios.len(),
                self.losses.len()
            )));
        }
        if let Some(&bad) = ratios.iter().find(|r| !(r.is_finite() && **r > 0.0)) {
            return Err(Error::InvalidParameter {
                name: "ratios",
                value: bad,
                reason: "must be finite and positive",
            });
        }
        for tower in towers {
            if tower.needs_sums_insured() || tower.layers.iter().any(|l| l.has_annual_terms()) {
                return Err(Error::Data(
                    "an index clause applies to per-occurrence covers; annual terms and \
                     surplus treaties are not indexed"
                        .into(),
                ));
            }
        }
        let adjustments: Vec<f64> = ratios.iter().map(|&r| clause.adjustment(r)).collect();
        self.cost(towers, years, Some(&adjustments))
    }

    fn cost(
        &self,
        towers: &[Tower],
        years: &[YearBasis],
        adjustments: Option<&[f64]>,
    ) -> Result<BurningCost> {
        if towers.len() != years.len() {
            return Err(Error::Data(format!(
                "{} towers for {} years",
                towers.len(),
                years.len()
            )));
        }
        let layers = towers.first().map_or(0, |t| t.layers.len());
        if towers.iter().any(|t| t.layers.len() != layers) {
            return Err(Error::Data(
                "every year's tower needs the same number of layers".into(),
            ));
        }
        for y in years {
            for (name, v) in [("premium", y.premium), ("development", y.development)] {
                if !(v.is_finite() && v > 0.0) {
                    return Err(Error::InvalidParameter {
                        name,
                        value: v,
                        reason: "must be finite and positive",
                    });
                }
            }
        }
        let mut rows = Vec::with_capacity(years.len());
        for (tower, basis) in towers.iter().zip(years) {
            let kept: Vec<usize> = (0..self.losses.len())
                .filter(|&k| {
                    let l = &self.losses[k];
                    l.year == basis.year && l.amount * l.factor >= self.complete_above
                })
                .collect();
            let losses: Vec<f64> = kept.iter().map(|&k| self.losses[k].trended).collect();
            let layer_loss = match adjustments {
                None => tower.ceded(&losses),
                Some(a) => {
                    let mut total = vec![0.0; layers];
                    for &k in &kept {
                        let r = a[k];
                        for (t, c) in total
                            .iter_mut()
                            .zip(tower.ceded(&[self.losses[k].trended / r]))
                        {
                            *t += r * c;
                        }
                    }
                    total
                }
            };
            let developed: Vec<f64> = layer_loss.iter().map(|c| c * basis.development).collect();
            let loss_cost = developed.iter().map(|d| d / basis.premium).collect();
            rows.push(YearCost {
                year: basis.year,
                premium: basis.premium,
                losses: losses.len(),
                layer_loss,
                development: basis.development,
                developed,
                loss_cost,
            });
        }
        let premium: f64 = rows.iter().map(|r| r.premium).sum();
        let loss_cost = (0..layers)
            .map(|i| rows.iter().map(|r| r.developed[i]).sum::<f64>() / premium)
            .collect();
        let mean_loss_cost = (0..layers)
            .map(|i| rows.iter().map(|r| r.loss_cost[i]).sum::<f64>() / rows.len() as f64)
            .collect();
        Ok(BurningCost {
            years: rows,
            loss_cost,
            mean_loss_cost,
        })
    }
}

/// One accident year of a [`BurningCost`]; the vectors are per layer.
#[derive(Debug, Clone, PartialEq)]
pub struct YearCost {
    pub year: i32,
    pub premium: f64,
    /// The number of trended losses that went through the tower.
    pub losses: usize,
    /// The as-if loss to each layer, trended, before development.
    pub layer_loss: Vec<f64>,
    pub development: f64,
    /// `layer_loss × development`.
    pub developed: Vec<f64>,
    /// `developed / premium`.
    pub loss_cost: Vec<f64>,
}

/// The burning cost by year and overall, from
/// [`Prepared::burning_cost`].
#[derive(Debug, Clone, PartialEq)]
pub struct BurningCost {
    pub years: Vec<YearCost>,
    /// Each layer's developed losses over the premium, summed over the
    /// years: the premium-weighted average loss cost.
    pub loss_cost: Vec<f64>,
    /// Each layer's loss cost averaged over the years with equal weight.
    pub mean_loss_cost: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_aggregate::Layer;

    fn loss(id: &str, accident: f64, amount: f64) -> LargeLoss {
        LargeLoss {
            id: id.into(),
            accident,
            amount,
        }
    }

    #[test]
    fn the_threshold_drops_losses_old_years_cannot_have() {
        // Reported above 100: the 1990 record is complete above 100 × 1.1^2.
        let e = Experience::new(vec![loss("a", 1990.0, 105.0), loss("b", 1992.0, 115.0)])
            .unwrap()
            .with_threshold(100.0)
            .unwrap();
        let p = e
            .prepare(&Trend::Rate {
                rate: 0.1,
                to: 1992.0,
            })
            .unwrap();
        assert!((p.complete_above() - 121.0).abs() < 1e-9);
        let t = Tower::new(vec![Layer::xol("all", f64::INFINITY, 0.0).unwrap()]).unwrap();
        let years = [
            YearBasis {
                year: 1990,
                premium: 1.0,
                development: 1.0,
            },
            YearBasis {
                year: 1992,
                premium: 1.0,
                development: 1.0,
            },
        ];
        let b = p.burning_cost(&[t.clone(), t], &years).unwrap();
        // 105 × 1.21 = 127.05 stays; 115 is below 121 and goes.
        assert_eq!((b.years[0].losses, b.years[1].losses), (1, 0));
    }

    #[test]
    fn annual_terms_and_per_year_towers_apply() {
        let e = Experience::new(vec![
            loss("a", 2001.2, 300.0),
            loss("b", 2001.8, 300.0),
            loss("c", 2002.5, 300.0),
        ])
        .unwrap();
        let p = e.prepare(&Trend::Factors(vec![1.0; 3])).unwrap();
        // 2001's treaty had an annual limit of 250; 2002's attached higher.
        let old = Tower::new(vec![
            Layer::xol("L", 200.0, 100.0)
                .unwrap()
                .aggregate_limit(250.0)
                .unwrap(),
        ])
        .unwrap();
        let new = Tower::new(vec![Layer::xol("L", 200.0, 150.0).unwrap()]).unwrap();
        let years = [
            YearBasis {
                year: 2001,
                premium: 500.0,
                development: 1.0,
            },
            YearBasis {
                year: 2002,
                premium: 500.0,
                development: 2.0,
            },
        ];
        let b = p.burning_cost(&[old, new], &years).unwrap();
        assert_eq!(b.years[0].layer_loss, vec![250.0]);
        assert_eq!(b.years[1].developed, vec![300.0]);
        assert!((b.loss_cost[0] - 0.55).abs() < 1e-15);
        assert!((b.mean_loss_cost[0] - 0.55).abs() < 1e-15);
        assert!(p.burning_cost(&[], &years).is_err());
    }

    fn valued(id: &str, age: f64, amount: f64) -> ClaimValuation {
        ClaimValuation {
            id: id.into(),
            age,
            amount,
        }
    }

    #[test]
    fn a_pattern_interpolates_in_age_and_holds_at_the_ends() {
        let p = DevelopmentPattern::new(vec![1.0, 2.0, 4.0], vec![2.0, 1.5, 1.0]).unwrap();
        assert_eq!(p.at(0.25), 2.0);
        assert_eq!(p.at(1.0), 2.0);
        assert!((p.at(1.5) - 1.75).abs() < 1e-15);
        assert!((p.at(3.0) - 1.25).abs() < 1e-15);
        assert_eq!(p.at(9.0), 1.0);
        assert_eq!(p.age_to_age(), vec![2.0 / 1.5, 1.5]);
        assert!(DevelopmentPattern::new(vec![2.0, 1.0], vec![1.0, 1.0]).is_err());
        assert!(DevelopmentPattern::new(vec![1.0], vec![0.0]).is_err());
        assert!(DevelopmentPattern::new(vec![], vec![]).is_err());
    }

    #[test]
    fn history_pairs_only_claims_seen_at_both_ages() {
        let history = [
            valued("a", 1.0, 100.0),
            valued("a", 2.0, 180.0),
            valued("a", 3.0, 200.0),
            valued("b", 1.0, 100.0),
            valued("b", 2.0, 120.0),
            // Reported at age 2: IBNR, left out of the first link.
            valued("c", 2.0, 500.0),
            valued("c", 3.0, 550.0),
            // A valuation off the grid is ignored.
            valued("a", 2.5, 1e9),
        ];
        let p = DevelopmentPattern::from_history(&history, &[1.0, 2.0, 3.0]).unwrap();
        let first = 300.0 / 200.0;
        let second = 750.0 / 680.0;
        assert!((p.age_to_age()[0] - first).abs() < 1e-14);
        assert!((p.age_to_age()[1] - second).abs() < 1e-14);
        assert!((p.factors()[0] - first * second).abs() < 1e-14);
        assert_eq!(p.factors()[2], 1.0);

        let twice = [
            valued("a", 1.0, 1.0),
            valued("a", 1.0, 2.0),
            valued("a", 2.0, 2.0),
        ];
        assert!(DevelopmentPattern::from_history(&twice, &[1.0, 2.0]).is_err());
        let unpaired = [valued("a", 1.0, 1.0), valued("b", 2.0, 2.0)];
        assert!(DevelopmentPattern::from_history(&unpaired, &[1.0, 2.0]).is_err());
        assert!(DevelopmentPattern::from_history(&history, &[1.0]).is_err());
    }

    #[test]
    fn developed_losses_reach_the_layer_but_completeness_stays_on_reported() {
        let e = Experience::new(vec![loss("a", 2020.5, 90.0), loss("b", 2022.5, 150.0)])
            .unwrap()
            .with_threshold(100.0)
            .unwrap()
            .with_valuation(2024.5)
            .unwrap();
        let pattern = DevelopmentPattern::new(vec![2.0, 4.0], vec![2.0, 1.5]).unwrap();
        let p = e
            .prepare_developed(
                &Trend::Factors(vec![1.0, 1.0]),
                &Development::Pattern(pattern),
            )
            .unwrap();
        assert_eq!(p.valuation(), Some(2024.5));
        assert_eq!(p.losses()[0].development, 1.5);
        assert_eq!(p.losses()[0].trended, 135.0);
        assert_eq!(p.losses()[1].ultimate, 300.0);
        // "a" develops above the threshold but was reported below it: out.
        let tower = Tower::new(vec![Layer::xol("x", 1000.0, 100.0).unwrap()]).unwrap();
        let years = [
            YearBasis {
                year: 2020,
                premium: 1000.0,
                development: 1.0,
            },
            YearBasis {
                year: 2022,
                premium: 1000.0,
                development: 1.0,
            },
        ];
        let b = p.burning_cost(&[tower.clone(), tower], &years).unwrap();
        assert_eq!(b.years[0].losses, 0);
        assert_eq!(b.years[1].layer_loss, vec![200.0]);

        let no_date = Experience::new(vec![loss("a", 2020.5, 90.0)]).unwrap();
        let pattern = DevelopmentPattern::new(vec![1.0], vec![1.0]).unwrap();
        assert!(
            no_date
                .prepare_developed(&Trend::Factors(vec![1.0]), &Development::Pattern(pattern))
                .is_err()
        );
        assert!(
            no_date
                .prepare_developed(&Trend::Factors(vec![1.0]), &Development::Factors(vec![]))
                .is_err()
        );
        assert!(no_date.clone().with_valuation(2019.0).is_err());
    }

    #[test]
    fn a_count_pattern_gives_the_year_its_ibnr_factor() {
        let counts = DevelopmentPattern::new(vec![0.5, 1.5, 2.5], vec![2.0, 1.2, 1.0]).unwrap();
        let y = YearBasis::from_count_pattern(2023, 500.0, &counts, 2024.5).unwrap();
        assert_eq!((y.year, y.premium), (2023, 500.0));
        // Age 1: halfway from 2.0 at 0.5 to 1.2 at 1.5.
        assert!((y.development - 1.6).abs() < 1e-15);
        assert!(YearBasis::from_count_pattern(2025, 500.0, &counts, 2024.5).is_err());
    }

    fn one_year(year: i32) -> [YearBasis; 1] {
        [YearBasis {
            year,
            premium: 1000.0,
            development: 1.0,
        }]
    }

    #[test]
    fn an_index_clause_moves_each_loss_terms_as_a_scaled_tower_would() {
        let p = Experience::new(vec![
            loss("a", 2020.2, 300.0),
            loss("b", 2020.4, 700.0),
            loss("c", 2020.6, 150.0),
        ])
        .unwrap()
        .prepare(&Trend::Factors(vec![1.0; 3]))
        .unwrap();
        let tower = |a: f64| {
            Tower::inuring(vec![
                vec![Layer::quota_share("QS", 0.2).unwrap()],
                vec![
                    Layer::xol("400 xs 100", 400.0 * a, 100.0 * a).unwrap(),
                    Layer::xol("500 xs 500", 500.0 * a, 500.0 * a).unwrap(),
                ],
            ])
            .unwrap()
        };
        let full = IndexClause::new(0.1, IndexKind::Full).unwrap();
        let ratios = [1.05, 1.3, 0.8];
        let b = p
            .burning_cost_indexed(&[tower(1.0)], &one_year(2020), &full, &ratios)
            .unwrap();
        // Each loss through the tower its own adjustment scales.
        let mut want = vec![0.0; 3];
        for (x, a) in [(300.0, 1.0), (700.0, 1.3), (150.0, 0.8)] {
            for (w, c) in want.iter_mut().zip(tower(a).ceded(&[x])) {
                *w += c;
            }
        }
        for (got, want) in b.years[0].layer_loss.iter().zip(&want) {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
        // Ratios of 1 are the plain burning cost.
        let plain = p.burning_cost(&[tower(1.0)], &one_year(2020)).unwrap();
        let unit = p
            .burning_cost_indexed(&[tower(1.0)], &one_year(2020), &full, &[1.0; 3])
            .unwrap();
        assert_eq!(plain, unit);
    }

    #[test]
    fn index_clause_kinds_and_errors() {
        let severe = IndexClause::new(0.1, IndexKind::Severe).unwrap();
        assert_eq!(severe.adjustment(1.1), 1.0);
        assert!((severe.adjustment(1.32) - 1.2).abs() < 1e-15);
        assert_eq!(severe.adjustment(0.5), 1.0);
        let full = IndexClause::new(0.0, IndexKind::Full).unwrap();
        assert_eq!(full.adjustment(0.9), 0.9);
        assert!(IndexClause::new(-0.1, IndexKind::Full).is_err());

        let p = Experience::new(vec![loss("a", 2020.5, 700.0)])
            .unwrap()
            .prepare(&Trend::Factors(vec![1.0]))
            .unwrap();
        let annual = Tower::new(vec![
            Layer::xol("x", 400.0, 100.0)
                .unwrap()
                .aggregate_limit(800.0)
                .unwrap(),
        ])
        .unwrap();
        assert!(
            p.burning_cost_indexed(&[annual], &one_year(2020), &full, &[1.0])
                .is_err()
        );
        let plain = Tower::new(vec![Layer::xol("x", 400.0, 100.0).unwrap()]).unwrap();
        assert!(
            p.burning_cost_indexed(&[plain.clone()], &one_year(2020), &full, &[])
                .is_err()
        );
        assert!(
            p.burning_cost_indexed(&[plain], &one_year(2020), &full, &[0.0])
                .is_err()
        );
    }

    #[test]
    fn exchange_rates_apply_before_the_layer_but_not_to_completeness() {
        let e = Experience::new(vec![loss("a", 2020.5, 90.0), loss("b", 2020.5, 120.0)])
            .unwrap()
            .with_threshold(100.0)
            .unwrap()
            .with_fx(Fx::PerLoss(vec![2.0, 0.5]))
            .unwrap();
        let p = e.prepare(&Trend::Factors(vec![1.0, 1.0])).unwrap();
        assert_eq!(p.losses()[0].fx, 2.0);
        assert_eq!(p.losses()[0].trended, 180.0);
        assert_eq!(p.losses()[1].trended, 60.0);
        let tower = Tower::new(vec![Layer::xol("x", 1000.0, 50.0).unwrap()]).unwrap();
        // "a" was reported below the threshold: out, though 180 converted.
        let b = p.burning_cost(&[tower], &one_year(2020)).unwrap();
        assert_eq!(b.years[0].losses, 1);
        assert_eq!(b.years[0].layer_loss, vec![10.0]);

        let stated = Experience::new(vec![loss("a", 2020.5, 90.0)])
            .unwrap()
            .with_fx(Fx::Rate(1.5))
            .unwrap()
            .prepare(&Trend::Factors(vec![1.0]))
            .unwrap();
        assert_eq!(stated.losses()[0].ultimate, 135.0);
        assert!(e.clone().with_fx(Fx::PerLoss(vec![1.0])).is_err());
        assert!(e.with_fx(Fx::Rate(0.0)).is_err());
    }
}
