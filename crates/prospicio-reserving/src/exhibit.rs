//! Friedland-style exhibits and a weighted selection of the methods'
//! ultimates (`docs/design/reserving-v02.md`, decision 34; issue #557).
//!
//! A reserving review ends in exhibits: the development triangle with its
//! link ratios, averages, selected factors, factors to ultimate and
//! percents developed ([`DevelopmentExhibit`], Friedland's chapter 7
//! Exhibit I, sheets 1 and 2), a summary of each method's ultimate, IBNR
//! and unpaid by origin with the claims to date ([`UltimateExhibit`],
//! chapter 7 Exhibit I, sheet 4, and chapter 15 Exhibit I), and the
//! ultimate the actuary selects per origin, often a weighting of methods
//! ([`UltimateExhibit::select`] with [`UltimateSelection`], chapter 15
//! Exhibit I, sheet 2, column (11), and chainladder-python's
//! `VotingChainladder`).
//!
//! ```
//! use prospicio_reserving::{
//!     ChainLadder, DevelopmentColumn, Grain, Long, MethodWeights, Month, Triangle,
//!     UltimateExhibit, UltimateSelection,
//! };
//!
//! let origin = [2020, 2020, 2021].map(Month::january);
//! let tri = Triangle::from_long(&Long {
//!     keys: &[],
//!     origin: &origin,
//!     development: DevelopmentColumn::Age(&[12, 24, 12]),
//!     values: &[("reported", &[100.0, 150.0, 120.0]), ("paid", &[60.0, 140.0, 50.0])],
//!     origin_grain: Grain::Year,
//!     development_grain: Grain::Year,
//!     cumulative: true,
//! })?;
//! let reported = ChainLadder::default().fit(&tri, "reported")?;
//! let paid = ChainLadder::default().fit(&tri, "paid")?;
//! let exhibit = UltimateExhibit::from_triangle(&tri, Some("reported"), Some("paid"))?
//!     .with_fit("reported", &reported)?
//!     .with_fit("paid", &paid)?;
//! assert_eq!(exhibit.ultimate("reported").unwrap(), [150.0, 180.0]);
//! assert_eq!(exhibit.ibnr("reported")?, [0.0, 60.0]);
//!
//! // The reported projection for the mature year, half of each for the green one.
//! let selection = UltimateSelection::new(MethodWeights::ByOrigin(vec![
//!     ("reported".into(), vec![1.0, 0.5]),
//!     ("paid".into(), vec![0.0, 0.5]),
//! ]));
//! let selected = exhibit.select(&selection)?;
//! let green = 0.5 * 180.0 + 0.5 * 50.0 * 140.0 / 60.0;
//! assert_eq!(selected.ultimate[0], 150.0);
//! assert!((selected.ultimate[1] - green).abs() < 1e-9);
//! # Ok::<(), prospicio_reserving::Error>(())
//! ```

use prospicio_core::{Lag, Period};

use crate::chain_ladder::ChainLadder;
use crate::development::{Average, Development};
use crate::error::{Error, Result};
use crate::segments::{FitTable, ReserveFit};
use crate::triangle::Triangle;

/// How far each origin's weights may sum from 1 before a selection is
/// refused.
pub const WEIGHT_TOLERANCE: f64 = 1e-9;

/// Friedland's averages of the link ratios (chapter 7, Exhibit I, sheets 1
/// and 2, part 3), in her order: simple averages of the latest 5 and 3
/// years, the medial average of the latest 5 without the highest and
/// lowest, volume-weighted averages of the latest 5 and 3, and the
/// geometric average of the latest 4. Each is a [`Development`] with
/// `n_periods` and the drops; at an age with fewer link ratios, all of
/// them are used, and a drop that would leave none is skipped, as the book
/// does.
pub fn friedland_averages() -> Vec<(String, Development)> {
    let dev = |average, n, drop| Development {
        average,
        n_periods: Some(n),
        drop_high: drop,
        drop_low: drop,
        ..Default::default()
    };
    vec![
        ("simple latest 5".into(), dev(Average::Simple, 5, 0)),
        ("simple latest 3".into(), dev(Average::Simple, 3, 0)),
        ("medial latest 5x1".into(), dev(Average::Simple, 5, 1)),
        ("volume latest 5".into(), dev(Average::Volume, 5, 0)),
        ("volume latest 3".into(), dev(Average::Volume, 3, 0)),
        ("geometric latest 4".into(), dev(Average::Geometric, 4, 0)),
    ]
}

/// The development exhibit of one column of a single-segment triangle:
/// the cumulative values, the link ratios, each named average of them,
/// the selected factors, the tail, the factors to ultimate and the
/// percents developed (Friedland's chapter 7, Exhibit I, sheets 1 and 2).
///
/// The selected factors and the factors to ultimate are those of a
/// [`ChainLadder`] fit, so a selected pattern (`Development::pattern`),
/// the exclusions and the tail all apply as in the chain ladder; the
/// exhibit rounds nothing. Friedland multiplies by factors to ultimate
/// rounded to three decimals: give those as a pattern of factors to
/// ultimate to reproduce her projections.
#[derive(Debug, Clone, PartialEq)]
pub struct DevelopmentExhibit {
    /// Origin periods, oldest first.
    pub origins: Vec<Period>,
    /// Development ages; link `k` runs from age `k` to age `k + 1`.
    pub ages: Vec<Lag>,
    /// Cumulative values by origin, then age; NaN where not observed.
    pub values: Vec<Vec<f64>>,
    /// Link ratios by origin, then link (one fewer than the ages); NaN
    /// where either end is not observed or the earlier value is 0.
    pub link_ratios: Vec<Vec<f64>>,
    /// Each named average's factor per link, in the order given.
    pub averages: Vec<(String, Vec<f64>)>,
    /// The age-to-age factors the projection uses, per link: estimated,
    /// selected or blended, and an estimated tail's own factors from its
    /// attachment age on ([`ChainLadderFit::ldf`](crate::ChainLadderFit::ldf)).
    pub selected: Vec<f64>,
    /// The factor from the oldest age to ultimate.
    pub tail: f64,
    /// The factor to ultimate at each age; the last is the tail.
    pub cdf: Vec<f64>,
    /// The share of the ultimate developed at each age, `1 / cdf`.
    pub percent_developed: Vec<f64>,
}

impl DevelopmentExhibit {
    /// The exhibit with [`friedland_averages`] and `development` as the
    /// selection, without a tail.
    ///
    /// Errors: as [`with_averages`](Self::with_averages).
    pub fn new(triangle: &Triangle, column: &str, development: &Development) -> Result<Self> {
        let selection = ChainLadder {
            development: development.clone(),
            ..Default::default()
        };
        Self::with_averages(triangle, column, &selection, &friedland_averages())
    }

    /// The exhibit with the given averages and the chain ladder `selection`
    /// (its development and tail) for the selected factors.
    ///
    /// Errors: a triangle with several segments or without the column, an
    /// empty or repeated average name, an average that cannot be estimated
    /// (named in the message) and a selection the chain ladder cannot fit.
    pub fn with_averages(
        triangle: &Triangle,
        column: &str,
        selection: &ChainLadder,
        averages: &[(String, Development)],
    ) -> Result<Self> {
        let segment = triangle.segment(column)?;
        let fit = selection.fit(triangle, column)?;
        let (no, nd) = (segment.n_origins, segment.n_dev);
        let values: Vec<Vec<f64>> = (0..no)
            .map(|o| {
                (0..nd)
                    .map(|d| segment.get(o, d).unwrap_or(f64::NAN))
                    .collect()
            })
            .collect();
        let link_ratios = values
            .iter()
            .map(|row| {
                row.windows(2)
                    .map(|w| {
                        if w[0].is_nan() || w[1].is_nan() || w[0] == 0.0 {
                            f64::NAN
                        } else {
                            w[1] / w[0]
                        }
                    })
                    .collect()
            })
            .collect();
        check_names(averages.iter().map(|(n, _)| n.as_str()), "average")?;
        let averages = averages
            .iter()
            .map(|(name, development)| {
                let fit = development
                    .fit(triangle, column)
                    .map_err(|e| Error::Exhibit(format!("the average {name:?}: {e}")))?;
                Ok((name.clone(), fit.blended_ldf()))
            })
            .collect::<Result<Vec<_>>>()?;
        let percent_developed = fit.cdf.iter().map(|c| 1.0 / c).collect();
        Ok(Self {
            origins: segment.origins.clone(),
            ages: segment.ages.clone(),
            values,
            link_ratios,
            averages,
            selected: fit.ldf().to_vec(),
            tail: fit.tail.factor,
            cdf: fit.cdf.clone(),
            percent_developed,
        })
    }

    /// The label of each link, `"12-24"` and so on, then `"120-ult"` for
    /// the tail: the columns of Friedland's parts 2 to 4.
    pub fn link_labels(&self) -> Vec<String> {
        let mut labels: Vec<String> = self
            .ages
            .windows(2)
            .map(|w| format!("{}-{}", w[0], w[1]))
            .collect();
        if let Some(last) = self.ages.last() {
            labels.push(format!("{last}-ult"));
        }
        labels
    }

    /// The named average `name`'s factors per link.
    pub fn average(&self, name: &str) -> Option<&[f64]> {
        self.averages
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_slice())
    }
}

/// A summary of each method's ultimate per origin beside the claims to
/// date, from which IBNR (ultimate less reported), the total unpaid
/// (ultimate less paid) and the case outstanding (reported less paid)
/// follow: Friedland's chapter 7, Exhibit I, sheet 4, and the summaries of
/// chapter 15, Exhibit I. A method's ultimate may be NaN at an origin it
/// does not project (a blank cell of the book's summaries).
#[derive(Debug, Clone, PartialEq)]
pub struct UltimateExhibit {
    /// Origin periods, oldest first; every per-origin vector follows them.
    pub origins: Vec<Period>,
    /// Each origin's age at the valuation date: its latest development age.
    pub ages: Vec<Lag>,
    /// Reported claims to date per origin, when given.
    pub reported: Option<Vec<f64>>,
    /// Paid claims to date per origin, when given.
    pub paid: Option<Vec<f64>>,
    /// Each method's ultimate per origin, in the order added.
    pub methods: Vec<(String, Vec<f64>)>,
}

impl UltimateExhibit {
    /// An exhibit with no method yet, from the claims to date.
    ///
    /// Errors: vectors of different lengths, no origin, a repeated origin
    /// and a reported or paid value that is not finite.
    pub fn new(
        origins: Vec<Period>,
        ages: Vec<Lag>,
        reported: Option<Vec<f64>>,
        paid: Option<Vec<f64>>,
    ) -> Result<Self> {
        let n = origins.len();
        if n == 0 {
            return Err(Error::Exhibit("no origin".into()));
        }
        for (i, o) in origins.iter().enumerate() {
            if origins[..i].contains(o) {
                return Err(Error::Exhibit(format!("the origin {o} is given twice")));
            }
        }
        check_len("ages", ages.len(), n)?;
        for (name, values) in [("reported", &reported), ("paid", &paid)] {
            if let Some(values) = values {
                check_len(name, values.len(), n)?;
                if let Some(i) = values.iter().position(|v| !v.is_finite()) {
                    return Err(Error::Exhibit(format!(
                        "the {name} value of {} is not finite",
                        origins[i]
                    )));
                }
            }
        }
        Ok(Self {
            origins,
            ages,
            reported,
            paid,
            methods: Vec::new(),
        })
    }

    /// An exhibit with no method yet, from the latest diagonal of a
    /// single-segment triangle's reported and paid columns (either may be
    /// left out). Each origin's age is its latest age in the columns given
    /// (the later where they differ).
    ///
    /// Errors: neither column given, a triangle with several segments or
    /// without a column, columns whose observed origins differ and an
    /// origin with no observed value.
    pub fn from_triangle(
        triangle: &Triangle,
        reported: Option<&str>,
        paid: Option<&str>,
    ) -> Result<Self> {
        let columns: Vec<&str> = reported.iter().chain(paid.iter()).copied().collect();
        if columns.is_empty() {
            return Err(Error::Exhibit(
                "give the reported column, the paid column or both".into(),
            ));
        }
        let mut origins: Option<Vec<Period>> = None;
        let mut ages: Vec<Lag> = Vec::new();
        let mut latest = Vec::new();
        for column in &columns {
            let segment = triangle.segment(column)?;
            match &origins {
                None => {
                    origins = Some(segment.origins.clone());
                    ages = vec![0; segment.n_origins];
                }
                Some(o) if *o != segment.origins => {
                    return Err(Error::Exhibit(format!(
                        "the columns {:?} observe different origins",
                        columns
                    )));
                }
                Some(_) => {}
            }
            let mut values = Vec::with_capacity(segment.n_origins);
            for (o, age) in ages.iter_mut().enumerate() {
                let (d, v) = segment.latest(o)?;
                *age = (*age).max(segment.ages[d]);
                values.push(v);
            }
            latest.push(values);
        }
        let mut latest = latest.into_iter();
        let reported = reported.and_then(|_| latest.next());
        let paid = paid.and_then(|_| latest.next());
        Self::new(origins.expect("a column"), ages, reported, paid)
    }

    /// The exhibit with one more method, `name`, whose ultimate per origin
    /// is `ultimate` (NaN where it gives none).
    ///
    /// Errors: an empty or repeated name, the wrong number of values and
    /// an infinite value.
    pub fn with_method(mut self, name: impl Into<String>, ultimate: Vec<f64>) -> Result<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(Error::Exhibit("a method has an empty name".into()));
        }
        if self.methods.iter().any(|(n, _)| *n == name) {
            return Err(Error::Exhibit(format!(
                "the method {name:?} is given twice"
            )));
        }
        check_len(
            &format!("the method {name:?}"),
            ultimate.len(),
            self.n_origins(),
        )?;
        if let Some(i) = ultimate.iter().position(|v| v.is_infinite()) {
            return Err(Error::Exhibit(format!(
                "the ultimate of {name:?} at {} is infinite",
                self.origins[i]
            )));
        }
        self.methods.push((name, ultimate));
        Ok(self)
    }

    /// The exhibit with one more method, `name`, whose ultimates are those
    /// of a fit of one segment, [`ReserveFit::ultimate`]: the method's own
    /// for the chain ladder, Mack, the expected-loss methods and Clark. For
    /// a bootstrap or a Bayesian model that is the chain ladder's point
    /// projection, not the mean of the simulations or of the posterior:
    /// give those means with [`with_method`](Self::with_method).
    ///
    /// Errors: a fit whose origins are not the exhibit's, and those of
    /// [`with_method`](Self::with_method).
    pub fn with_fit<F: ReserveFit + ?Sized>(
        self,
        name: impl Into<String>,
        fit: &F,
    ) -> Result<Self> {
        let name = name.into();
        if fit.chain_ladder().origins != self.origins {
            return Err(Error::Exhibit(format!(
                "the fit {name:?} has other origins than the exhibit"
            )));
        }
        let ultimate = fit.ultimate().to_vec();
        self.with_method(name, ultimate)
    }

    /// Number of origins.
    pub fn n_origins(&self) -> usize {
        self.origins.len()
    }

    /// The names of the methods, in the order added.
    pub fn method_names(&self) -> Vec<&str> {
        self.methods.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// The method `name`'s ultimate per origin.
    pub fn ultimate(&self, name: &str) -> Option<&[f64]> {
        self.methods
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_slice())
    }

    /// The method `name`'s IBNR per origin, its ultimate less reported.
    ///
    /// Errors: an unknown method, or an exhibit without reported claims.
    pub fn ibnr(&self, name: &str) -> Result<Vec<f64>> {
        let reported = self.to_date("reported", self.reported.as_deref())?;
        Ok(less(self.method(name)?, reported))
    }

    /// The method `name`'s total unpaid per origin, its ultimate less paid
    /// (IBNR plus case outstanding).
    ///
    /// Errors: an unknown method, or an exhibit without paid claims.
    pub fn unpaid(&self, name: &str) -> Result<Vec<f64>> {
        let paid = self.to_date("paid", self.paid.as_deref())?;
        Ok(less(self.method(name)?, paid))
    }

    /// The case outstanding per origin, reported less paid, when both are
    /// given.
    pub fn case_outstanding(&self) -> Option<Vec<f64>> {
        Some(less(self.reported.as_deref()?, self.paid.as_deref()?))
    }

    /// Friedland's summary of ultimate claims: one row per origin, the
    /// claims to date given (`reported`, `paid`), then a column per method.
    pub fn ultimates_table(&self) -> FitTable {
        let mut values = self.to_date_columns();
        values.extend(self.methods.iter().cloned());
        self.table(values)
    }

    /// Friedland's summary of IBNR: one row per origin, a column per
    /// method's IBNR.
    ///
    /// Errors: an exhibit without reported claims.
    pub fn ibnr_table(&self) -> Result<FitTable> {
        let values = self
            .methods
            .iter()
            .map(|(n, _)| Ok((n.clone(), self.ibnr(n)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(self.table(values))
    }

    /// The summary of the total unpaid: one row per origin, a column per
    /// method's ultimate less paid.
    ///
    /// Errors: an exhibit without paid claims.
    pub fn unpaid_table(&self) -> Result<FitTable> {
        let values = self
            .methods
            .iter()
            .map(|(n, _)| Ok((n.clone(), self.unpaid(n)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(self.table(values))
    }

    /// The selected ultimate per origin, each a weighting of the methods'
    /// ultimates by `selection`'s weights, with its IBNR and unpaid.
    ///
    /// Each origin's weights must be finite, not negative and sum to 1
    /// within [`WEIGHT_TOLERANCE`]; they are used as given, not rescaled. A
    /// method left out of the weights has weight 0, and a method with
    /// weight 0 does not enter, so its ultimate may be NaN there.
    ///
    /// Errors: a method in the weights that the exhibit does not have or
    /// that is named twice, weights of the wrong length, an origin whose
    /// weights do not sum to 1 or include a negative or non-finite weight,
    /// a positive weight on a NaN ultimate, and, with
    /// [`MethodWeights::ByAge`], ages not increasing or an origin younger
    /// than the first age.
    pub fn select(&self, selection: &UltimateSelection) -> Result<SelectedUltimates> {
        let n = self.n_origins();
        let mut weights: Vec<(String, Vec<f64>)> = self
            .methods
            .iter()
            .map(|(name, _)| (name.clone(), vec![0.0; n]))
            .collect();
        let position = |name: &str| -> Result<usize> {
            self.methods
                .iter()
                .position(|(m, _)| m == name)
                .ok_or_else(|| Error::Exhibit(format!("the exhibit has no method {name:?}")))
        };
        match &selection.weights {
            MethodWeights::Single(name) => {
                weights[position(name)?].1 = vec![1.0; n];
            }
            MethodWeights::ByOrigin(by_method) => {
                check_names(by_method.iter().map(|(m, _)| m.as_str()), "method")?;
                for (name, w) in by_method {
                    check_len(&format!("the weights of {name:?}"), w.len(), n)?;
                    weights[position(name)?].1.clone_from(w);
                }
            }
            MethodWeights::ByAge(rows) => {
                if rows.is_empty() {
                    return Err(Error::Exhibit("the weights by age have no age".into()));
                }
                if rows.windows(2).any(|w| w[0].0 >= w[1].0) {
                    return Err(Error::Exhibit(
                        "the ages of the weights by age must increase".into(),
                    ));
                }
                for (age, row) in rows {
                    check_names(row.iter().map(|(m, _)| m.as_str()), "method")?;
                    for (name, _) in row {
                        position(name).map_err(|e| {
                            Error::Exhibit(format!("the weights at age {age}: {e}"))
                        })?;
                    }
                }
                for (o, &age) in self.ages.iter().enumerate() {
                    let Some((_, row)) = rows.iter().rev().find(|(a, _)| *a <= age) else {
                        return Err(Error::Exhibit(format!(
                            "the origin {} at age {age} is younger than the first age of the weights, {}",
                            self.origins[o], rows[0].0
                        )));
                    };
                    for (name, w) in row {
                        weights[position(name)?].1[o] = *w;
                    }
                }
            }
        }
        let mut ultimate = vec![0.0; n];
        for o in 0..n {
            let origin = &self.origins[o];
            let mut sum = 0.0;
            for ((name, w), (_, u)) in weights.iter().zip(&self.methods) {
                let w = w[o];
                if !w.is_finite() || w < 0.0 {
                    return Err(Error::Exhibit(format!(
                        "the weight of {name:?} at {origin} is {w}, not a finite number of at least 0"
                    )));
                }
                sum += w;
                if w > 0.0 {
                    if u[o].is_nan() {
                        return Err(Error::Exhibit(format!(
                            "{name:?} has weight {w} at {origin} but no ultimate there"
                        )));
                    }
                    ultimate[o] += w * u[o];
                }
            }
            if (sum - 1.0).abs() > WEIGHT_TOLERANCE {
                return Err(Error::Exhibit(format!(
                    "the weights at {origin} sum to {sum}, not 1"
                )));
            }
        }
        Ok(SelectedUltimates {
            origins: self.origins.clone(),
            ages: self.ages.clone(),
            reported: self.reported.clone(),
            paid: self.paid.clone(),
            weights,
            ultimate,
        })
    }

    fn method(&self, name: &str) -> Result<&[f64]> {
        self.ultimate(name)
            .ok_or_else(|| Error::Exhibit(format!("the exhibit has no method {name:?}")))
    }

    fn to_date<'a>(&self, what: &str, values: Option<&'a [f64]>) -> Result<&'a [f64]> {
        values.ok_or_else(|| Error::Exhibit(format!("the exhibit has no {what} claims")))
    }

    fn to_date_columns(&self) -> Vec<(String, Vec<f64>)> {
        [("reported", &self.reported), ("paid", &self.paid)]
            .into_iter()
            .filter_map(|(n, v)| Some((n.to_string(), v.clone()?)))
            .collect()
    }

    fn table(&self, values: Vec<(String, Vec<f64>)>) -> FitTable {
        FitTable {
            keys: Vec::new(),
            origin: Some(self.origins.clone()),
            age: None,
            values,
        }
    }
}

/// The weights of the methods in a selection of ultimates.
#[derive(Debug, Clone, PartialEq)]
pub enum MethodWeights {
    /// Each named method's weight per origin, in the exhibit's origin
    /// order (chainladder-python's `VotingChainladder` weights, one column
    /// per method). A method not named has weight 0.
    ByOrigin(Vec<(String, Vec<f64>)>),
    /// Weights by the origin's age at the valuation date: rows of
    /// `(age, [(method, weight)])` with increasing ages, each origin taking
    /// the row of the oldest age not past its own, so a row holds from its
    /// age until the next row's (the chain ladder from 48 months, say, and
    /// Bornhuetter-Ferguson before). A method not named in a row has
    /// weight 0 there.
    ByAge(Vec<(Lag, Vec<(String, f64)>)>),
    /// One method for every origin.
    Single(String),
}

/// A selection of ultimates: the weights of the methods by origin.
#[derive(Debug, Clone, PartialEq)]
pub struct UltimateSelection {
    /// The methods' weights.
    pub weights: MethodWeights,
}

impl UltimateSelection {
    /// A selection with these weights.
    pub fn new(weights: MethodWeights) -> Self {
        Self { weights }
    }
}

/// The selected ultimate per origin with its IBNR and unpaid, and the
/// weights that gave it.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedUltimates {
    /// Origin periods, oldest first.
    pub origins: Vec<Period>,
    /// Each origin's age at the valuation date.
    pub ages: Vec<Lag>,
    /// Reported claims to date, when the exhibit has them.
    pub reported: Option<Vec<f64>>,
    /// Paid claims to date, when the exhibit has them.
    pub paid: Option<Vec<f64>>,
    /// Every method's weight per origin, in the exhibit's method order.
    pub weights: Vec<(String, Vec<f64>)>,
    /// The selected ultimate per origin.
    pub ultimate: Vec<f64>,
}

impl SelectedUltimates {
    /// IBNR per origin, the selected ultimate less reported.
    pub fn ibnr(&self) -> Option<Vec<f64>> {
        Some(less(&self.ultimate, self.reported.as_deref()?))
    }

    /// Total unpaid per origin, the selected ultimate less paid.
    pub fn unpaid(&self) -> Option<Vec<f64>> {
        Some(less(&self.ultimate, self.paid.as_deref()?))
    }

    /// The total selected ultimate.
    pub fn total_ultimate(&self) -> f64 {
        self.ultimate.iter().sum()
    }

    /// The total IBNR, when reported claims are given.
    pub fn total_ibnr(&self) -> Option<f64> {
        Some(self.ibnr()?.iter().sum())
    }

    /// The total unpaid, when paid claims are given.
    pub fn total_unpaid(&self) -> Option<f64> {
        Some(self.unpaid()?.iter().sum())
    }

    /// One row per origin: `reported` and `paid` when given, `ultimate`,
    /// `ibnr` and `unpaid` when they are defined, then each method's
    /// weight as `weight_<method>`.
    pub fn to_table(&self) -> FitTable {
        let mut values: Vec<(String, Vec<f64>)> = Vec::new();
        for (name, v) in [("reported", &self.reported), ("paid", &self.paid)] {
            if let Some(v) = v {
                values.push((name.into(), v.clone()));
            }
        }
        values.push(("ultimate".into(), self.ultimate.clone()));
        if let Some(v) = self.ibnr() {
            values.push(("ibnr".into(), v));
        }
        if let Some(v) = self.unpaid() {
            values.push(("unpaid".into(), v));
        }
        for (name, w) in &self.weights {
            values.push((format!("weight_{name}"), w.clone()));
        }
        FitTable {
            keys: Vec::new(),
            origin: Some(self.origins.clone()),
            age: None,
            values,
        }
    }
}

fn less(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn check_len(what: &str, found: usize, expected: usize) -> Result<()> {
    if found == expected {
        Ok(())
    } else {
        Err(Error::Exhibit(format!(
            "{what} has {found} values for {expected} origins"
        )))
    }
}

fn check_names<'a>(names: impl Iterator<Item = &'a str>, what: &str) -> Result<()> {
    let mut seen: Vec<&str> = Vec::new();
    for name in names {
        if name.is_empty() {
            return Err(Error::Exhibit(format!("an {what} has an empty name")));
        }
        if seen.contains(&name) {
            return Err(Error::Exhibit(format!(
                "the {what} {name:?} is given twice"
            )));
        }
        seen.push(name);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DevelopmentColumn, Grain, Long, Month};

    fn triangle() -> Triangle {
        let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
            values: &[
                ("reported", &[100.0, 130.0, 140.0, 110.0, 150.0, 120.0]),
                ("paid", &[50.0, 100.0, 130.0, 60.0, 120.0, 40.0]),
            ],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    fn exhibit() -> UltimateExhibit {
        let tri = triangle();
        let reported = ChainLadder::default().fit(&tri, "reported").unwrap();
        let paid = ChainLadder::default().fit(&tri, "paid").unwrap();
        UltimateExhibit::from_triangle(&tri, Some("reported"), Some("paid"))
            .unwrap()
            .with_fit("cl reported", &reported)
            .unwrap()
            .with_fit("cl paid", &paid)
            .unwrap()
    }

    #[test]
    fn development_exhibit_lays_out_the_triangle() {
        let tri = triangle();
        let ex = DevelopmentExhibit::new(&tri, "reported", &Development::default()).unwrap();
        assert_eq!(ex.ages, [12, 24, 36]);
        assert_eq!(ex.link_labels(), ["12-24", "24-36", "36-ult"]);
        assert_eq!(ex.link_ratios[0], [1.3, 140.0 / 130.0]);
        assert!(ex.link_ratios[1][1].is_nan() && ex.values[2][1].is_nan());
        let simple = ex.average("simple latest 3").unwrap();
        assert!((simple[0] - (1.3 + 150.0 / 110.0) / 2.0).abs() < 1e-15);
        assert_eq!(ex.averages.len(), 6);
        assert_eq!(ex.selected, [280.0 / 210.0, 140.0 / 130.0]);
        assert_eq!(ex.tail, 1.0);
        assert_eq!(ex.cdf.len(), 3);
        assert_eq!(ex.percent_developed[2], 1.0);
        assert!((ex.cdf[0] * ex.percent_developed[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn development_exhibit_refuses_repeated_averages() {
        let tri = triangle();
        let avg = vec![
            ("a".to_string(), Development::default()),
            ("a".to_string(), Development::default()),
        ];
        let err = DevelopmentExhibit::with_averages(&tri, "paid", &ChainLadder::default(), &avg)
            .unwrap_err();
        assert!(err.to_string().contains("given twice"), "{err}");
    }

    #[test]
    fn ibnr_unpaid_and_case() {
        let ex = exhibit();
        assert_eq!(ex.ages, [36, 24, 12]);
        assert_eq!(ex.case_outstanding().unwrap(), [10.0, 30.0, 80.0]);
        let ult = ex.ultimate("cl reported").unwrap().to_vec();
        let ibnr = ex.ibnr("cl reported").unwrap();
        let unpaid = ex.unpaid("cl reported").unwrap();
        for o in 0..3 {
            assert_eq!(ibnr[o], ult[o] - ex.reported.as_ref().unwrap()[o]);
            assert!((unpaid[o] - ibnr[o] - ex.case_outstanding().unwrap()[o]).abs() < 1e-12);
        }
        let t = ex.ultimates_table();
        let names: Vec<_> = t.values.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["reported", "paid", "cl reported", "cl paid"]);
        assert_eq!(ex.ibnr_table().unwrap().column("cl paid").unwrap().len(), 3);
    }

    #[test]
    fn single_and_by_origin_selections() {
        let ex = exhibit();
        let single = ex
            .select(&UltimateSelection::new(MethodWeights::Single(
                "cl paid".into(),
            )))
            .unwrap();
        assert_eq!(single.ultimate, ex.ultimate("cl paid").unwrap());
        assert_eq!(single.weights[0].1, [0.0; 3]);
        let by = ex
            .select(&UltimateSelection::new(MethodWeights::ByOrigin(vec![
                ("cl reported".into(), vec![1.0, 0.25, 0.0]),
                ("cl paid".into(), vec![0.0, 0.75, 1.0]),
            ])))
            .unwrap();
        let (r, p) = (
            ex.ultimate("cl reported").unwrap(),
            ex.ultimate("cl paid").unwrap(),
        );
        assert_eq!(by.ultimate, [r[0], 0.25 * r[1] + 0.75 * p[1], p[2]]);
        let total_ibnr: f64 = by.ibnr().unwrap().iter().sum();
        assert_eq!(by.total_ibnr(), Some(total_ibnr));
        assert!(by.to_table().column("weight_cl paid").is_some());
    }

    #[test]
    fn by_age_steps_from_each_row() {
        let ex = exhibit();
        let rows = vec![
            (12, vec![("cl paid".to_string(), 1.0)]),
            (24, vec![("cl reported".to_string(), 1.0)]),
        ];
        let sel = ex
            .select(&UltimateSelection::new(MethodWeights::ByAge(rows)))
            .unwrap();
        let (r, p) = (
            ex.ultimate("cl reported").unwrap(),
            ex.ultimate("cl paid").unwrap(),
        );
        assert_eq!(sel.ultimate, [r[0], r[1], p[2]]);
        let young = vec![(24, vec![("cl paid".to_string(), 1.0)])];
        let err = ex
            .select(&UltimateSelection::new(MethodWeights::ByAge(young)))
            .unwrap_err();
        assert!(err.to_string().contains("younger"), "{err}");
    }

    #[test]
    fn weights_not_summing_to_one_are_refused() {
        let ex = exhibit();
        let sel = |a: Vec<f64>, b: Vec<f64>| {
            ex.select(&UltimateSelection::new(MethodWeights::ByOrigin(vec![
                ("cl reported".into(), a),
                ("cl paid".into(), b),
            ])))
        };
        let err = sel(vec![1.0, 0.5, 0.5], vec![0.0, 0.6, 0.5]).unwrap_err();
        assert!(err.to_string().contains("sum to 1.1"), "{err}");
        let err = sel(vec![1.5, 0.5, 0.5], vec![-0.5, 0.5, 0.5]).unwrap_err();
        assert!(err.to_string().contains("at least 0"), "{err}");
        assert!(sel(vec![0.0; 3], vec![0.0; 3]).is_err());
        let err = sel(vec![1.0; 2], vec![0.0; 2]).unwrap_err();
        assert!(err.to_string().contains("2 values for 3 origins"), "{err}");
        let err = ex
            .select(&UltimateSelection::new(MethodWeights::Single("bf".into())))
            .unwrap_err();
        assert!(err.to_string().contains("no method \"bf\""), "{err}");
    }

    #[test]
    fn a_nan_ultimate_enters_only_at_weight_zero() {
        let ex = exhibit()
            .with_method("freq-sev", vec![200.0, f64::NAN, f64::NAN])
            .unwrap();
        let ok = ex.select(&UltimateSelection::new(MethodWeights::ByOrigin(vec![
            ("freq-sev".into(), vec![0.5, 0.0, 0.0]),
            ("cl reported".into(), vec![0.5, 1.0, 1.0]),
        ])));
        assert!(ok.is_ok());
        let err = ex
            .select(&UltimateSelection::new(MethodWeights::Single(
                "freq-sev".into(),
            )))
            .unwrap_err();
        assert!(err.to_string().contains("no ultimate there"), "{err}");
    }

    #[test]
    fn construction_errors() {
        let tri = triangle();
        assert!(UltimateExhibit::from_triangle(&tri, None, None).is_err());
        let ex = UltimateExhibit::from_triangle(&tri, None, Some("paid")).unwrap();
        assert!(ex.reported.is_none() && ex.case_outstanding().is_none());
        assert!(ex.clone().with_method("", vec![1.0; 3]).is_err());
        assert!(ex.clone().with_method("a", vec![f64::INFINITY; 3]).is_err());
        let ex = ex.with_method("a", vec![1.0; 3]).unwrap();
        assert!(ex.ibnr("a").is_err());
        assert!(ex.unpaid_table().is_ok() && ex.ibnr_table().is_err());
        assert!(ex.clone().with_method("a", vec![1.0; 3]).is_err());
        let other = triangle().max_age(24).unwrap();
        let fit = ChainLadder::default().fit(&other, "paid").unwrap();
        assert!(ex.clone().with_fit("b", &fit).is_ok());
        let two = Triangle::from_long(&Long {
            keys: &[],
            origin: &[Month::january(2020), Month::january(2021)],
            development: DevelopmentColumn::Age(&[12, 12]),
            values: &[("paid", &[1.0, 2.0])],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let fit = ChainLadder::default().fit(&two, "paid");
        if let Ok(fit) = fit {
            assert!(ex.with_fit("c", &fit).is_err());
        }
        assert!(
            UltimateExhibit::new(
                vec![Period::year(2020), Period::year(2020)],
                vec![12, 12],
                None,
                None
            )
            .is_err()
        );
    }
}
