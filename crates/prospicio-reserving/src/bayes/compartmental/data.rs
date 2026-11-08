//! The data of a fit: the square grain, observed cells and the response and hierarchy of the priors.

use prospicio_core::{Grain, Lag, Period};

use super::fit::outstanding_name;
use super::priors::{CompartmentalPriors, HalfPrior, PositivePrior};
use super::settings::{CompartmentalLikelihood, DevelopmentLevels, Fitted, Outstanding};
use crate::error::{Error, Result};
use crate::triangle::{Segment, Triangle};

/// The grains of a fit's square (decision 13): the origin grain must be a
/// whole multiple of the development grain (annual origins with quarterly
/// or monthly development, quarterly origins with monthly development, or
/// equal grains).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SquareGrain {
    pub(super) origin: Grain,
    pub(super) development: Grain,
}

impl SquareGrain {
    pub(super) fn new(origin: Grain, development: Grain) -> Result<Self> {
        if !origin.is_multiple_of(development) {
            return Err(Error::Bayes(format!(
                "origin grain {origin} is not a whole multiple of development grain \
                 {development}; the compartmental model needs each origin period to be a \
                 whole number of development periods"
            )));
        }
        Ok(Self {
            origin,
            development,
        })
    }

    /// Development periods per origin period.
    pub(super) fn per_origin(self) -> usize {
        (self.origin.months() / self.development.months()) as usize
    }

    /// The square's ages for `n_origins` origins: `n_origins` origin
    /// periods, in development periods (decision 5's horizon).
    pub(super) fn square_ages(self, n_origins: usize) -> usize {
        n_origins * self.per_origin()
    }

    /// The age in months of age index `k`.
    pub(super) fn age(self, k: usize) -> Lag {
        (k as Lag + 1) * self.development.months()
    }

    /// The age index of an age in months, which must be a whole positive
    /// number of development periods.
    pub(super) fn age_index(self, age: Lag) -> Result<usize> {
        let months = self.development.months();
        if age == 0 || age % months != 0 {
            return Err(Error::Bayes(format!(
                "development age {age} months is not a whole number of {} periods",
                self.development
            )));
        }
        Ok((age / months - 1) as usize)
    }

    /// The development time of age index `k` in years (decision 13): the
    /// age in months over 12, on every grain.
    pub(super) fn time(self, k: usize) -> f64 {
        f64::from(self.age(k)) / 12.0
    }
}

/// Each origin of the premium column, its premium (the latest value, which
/// must be finite and positive) and the grains of the square.
pub(super) fn premium_of(
    triangle: &Triangle,
    premium: &str,
) -> Result<(Vec<Period>, Vec<f64>, SquareGrain)> {
    let grain = SquareGrain::new(triangle.origin_grain(), triangle.development_grain())?;
    let segment = triangle.segment(premium)?;
    let premiums = (0..segment.n_origins)
        .map(|o| {
            segment
                .latest(o)
                .ok()
                .map(|(_, v)| v)
                .filter(|v| v.is_finite() && *v > 0.0)
                .ok_or_else(|| {
                    Error::Bayes(format!(
                        "origin {} has no positive premium in column {premium}",
                        segment.origins[o]
                    ))
                })
        })
        .collect::<Result<Vec<f64>>>()?;
    Ok((segment.origins.clone(), premiums, grain))
}

/// The observed cells of a fit, by the premium column's origins.
#[derive(Debug, Clone)]
pub(super) struct Data {
    pub(super) origins: Vec<Period>,
    pub(super) premium: Vec<f64>,
    pub(super) grain: SquareGrain,
    /// The square's number of ages, in development periods.
    pub(super) n_ages: usize,
    /// `(origin, age index, value)` of each observed outstanding cell.
    pub(super) outstanding: Vec<(usize, usize, f64)>,
    /// The same for paid.
    pub(super) paid: Vec<(usize, usize, f64)>,
    /// Each origin's latest observed paid: age index and value.
    pub(super) latest_paid: Vec<Option<(usize, f64)>>,
    /// Each origin's latest observed outstanding age index.
    pub(super) latest_outstanding: Vec<Option<usize>>,
    /// Whether the second response is fitted (all but
    /// [`Fitted::PaidOnly`]); without it the square has no future
    /// outstanding cells.
    pub(super) second: bool,
}

/// The future cells of the square: `(origin, age index)` of each paid and
/// outstanding cell, and the position in `paid` of each origin's cell at
/// the last age, if it is in the future.
#[derive(Debug, Clone)]
pub(super) struct Future {
    pub(super) paid: Vec<(usize, usize)>,
    pub(super) outstanding: Vec<(usize, usize)>,
    pub(super) last_paid: Vec<Option<usize>>,
}

impl Data {
    pub(super) fn new(
        triangle: &Triangle,
        paid: &str,
        outstanding: Outstanding<'_>,
        premium: &str,
        fitted: Fitted,
    ) -> Result<Self> {
        let (origins, premiums, grain) = premium_of(triangle, premium)?;
        let premium_segment = triangle.segment(premium)?;
        let paid_segment = triangle.segment(paid)?;
        // The fit's origin of a segment's origin `o`.
        let origin_of = |segment: &Segment, o: usize, column: &str| -> Result<usize> {
            (segment.origin_offset + o)
                .checked_sub(premium_segment.origin_offset)
                .filter(|&i| i < origins.len())
                .ok_or_else(|| {
                    Error::Bayes(format!(
                        "origin {} has {column} but no premium in column {premium}",
                        segment.origins[o]
                    ))
                })
        };
        let age_index = |age: Lag| grain.age_index(age);
        let cells_of = |segment: &Segment, column: &str| -> Result<Vec<(usize, usize, f64)>> {
            let mut out = Vec::new();
            for o in 0..segment.n_origins {
                for d in 0..segment.n_dev {
                    if let Some(v) = segment.get(o, d) {
                        if !v.is_finite() {
                            return Err(Error::Bayes(format!(
                                "origin {} at development age {} months has {column} {v}",
                                segment.origins[o], segment.ages[d]
                            )));
                        }
                        out.push((
                            origin_of(segment, o, column)?,
                            age_index(segment.ages[d])?,
                            v,
                        ));
                    }
                }
            }
            Ok(out)
        };
        let paid_cells = cells_of(&paid_segment, paid)?;
        let incurred = matches!(fitted, Fitted::PaidAndIncurred | Fitted::IncurredOnly);
        let second = fitted != Fitted::PaidOnly;
        let paid_at = |i: usize, k: usize| paid_cells.iter().find(|c| c.0 == i && c.1 == k);
        let os_cells = match outstanding {
            // Paid alone: the second column is not read.
            _ if !second => Vec::new(),
            // Incurred is outstanding plus paid where both are observed.
            Outstanding::Column(column) if incurred => {
                cells_of(&triangle.segment(column)?, column)?
                    .into_iter()
                    .filter_map(|(i, k, v)| paid_at(i, k).map(|c| (i, k, v + c.2)))
                    .collect()
            }
            Outstanding::Column(column) => cells_of(&triangle.segment(column)?, column)?,
            // The incurred cells as they are.
            Outstanding::FromIncurred(column) if incurred => {
                cells_of(&triangle.segment(column)?, column)?
            }
            Outstanding::FromIncurred(column) => {
                let incurred = cells_of(&triangle.segment(column)?, column)?;
                incurred
                    .into_iter()
                    .filter_map(|(i, k, v)| paid_at(i, k).map(|c| (i, k, v - c.2)))
                    .collect()
            }
        };
        if second && os_cells.is_empty() {
            return Err(Error::Bayes(format!(
                "column {} gives no {} cell; the model needs the second response's column \
                 (or Fitted::PaidOnly, with ordered rates)",
                outstanding_name(outstanding),
                if incurred { "incurred" } else { "outstanding" }
            )));
        }
        let mut data = Self {
            origins,
            premium: premiums,
            grain,
            n_ages: 0,
            outstanding: os_cells,
            paid: paid_cells,
            latest_paid: Vec::new(),
            latest_outstanding: Vec::new(),
            second,
        };
        data.n_ages = data.square_ages();
        data.set_latest();
        Ok(data)
    }

    /// Each origin's latest paid and outstanding, from the cells.
    pub(super) fn set_latest(&mut self) {
        let n = self.origins.len();
        self.latest_paid = vec![None; n];
        for &(i, k, v) in &self.paid {
            if self.latest_paid[i].is_none_or(|(l, _)| k > l) {
                self.latest_paid[i] = Some((k, v));
            }
        }
        self.latest_outstanding = vec![None; n];
        for &(i, k, _) in &self.outstanding {
            if self.latest_outstanding[i].is_none_or(|l| k > l) {
                self.latest_outstanding[i] = Some(k);
            }
        }
    }

    /// The same data with the cells `paid` (cumulative) and `outstanding`
    /// added, each `(origin, age index, value)` within the square.
    pub(super) fn extended(
        &self,
        paid: impl IntoIterator<Item = (usize, usize, f64)>,
        outstanding: impl IntoIterator<Item = (usize, usize, f64)>,
    ) -> Self {
        let mut data = self.clone();
        data.paid.extend(paid);
        data.outstanding.extend(outstanding);
        data.set_latest();
        data
    }

    /// The age in months of age index `k`.
    pub(super) fn age(&self, k: usize) -> Lag {
        self.grain.age(k)
    }

    /// The square's number of ages with the cells as they are now: the
    /// premium column's origin periods in development periods (decision
    /// 13), or the latest observed age if later.
    pub(super) fn square_ages(&self) -> usize {
        self.grain.square_ages(self.origins.len()).max(
            self.paid
                .iter()
                .chain(&self.outstanding)
                .map(|c| c.1 + 1)
                .max()
                .unwrap_or(0),
        )
    }

    pub(super) fn future_cells(&self) -> Future {
        let n = self.origins.len();
        let mut paid = Vec::new();
        let mut last_paid = vec![None; n];
        for (i, last) in last_paid.iter_mut().enumerate() {
            let from = self.latest_paid[i].map_or(0, |(k, _)| k + 1);
            for k in from..self.n_ages {
                if k + 1 == self.n_ages {
                    *last = Some(paid.len());
                }
                paid.push((i, k));
            }
        }
        let mut outstanding = Vec::new();
        for i in (0..n).filter(|_| self.second) {
            let from = self.latest_outstanding[i].map_or(0, |k| k + 1);
            for k in from..self.n_ages {
                outstanding.push((i, k));
            }
        }
        Future {
            paid,
            outstanding,
            last_paid,
        }
    }
}

/// The response settings of a fit (decision 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Response {
    pub(super) likelihood: CompartmentalLikelihood,
    /// Loss ratios to premium (else amounts).
    pub(super) loss_ratio: bool,
    /// Incremental paid (else cumulative).
    pub(super) incremental: bool,
    /// The second response is incurred (else outstanding).
    pub(super) incurred: bool,
    /// The paid cells are fitted (all but [`Fitted::IncurredOnly`]).
    pub(super) paid: bool,
    /// The second response's cells are fitted (all but
    /// [`Fitted::PaidOnly`]).
    pub(super) second: bool,
    /// How wide a development level is.
    pub(super) levels: DevelopmentLevels,
}

impl DevelopmentLevels {
    /// The group of the age index `k`, of `age` months: the index itself, or
    /// the 0-based development year `(age - 1) / 12`.
    pub(super) fn group(self, age: Lag, k: usize) -> usize {
        match self {
            Self::PerAge => k,
            Self::ByYear => ((age - 1) / 12) as usize,
        }
    }
}

impl Response {
    /// Gesmann and Morris' model 1: normal on amounts, cumulative paid.
    pub(super) const MODEL_1: Self = Self {
        likelihood: CompartmentalLikelihood::Normal,
        loss_ratio: false,
        incremental: false,
        incurred: false,
        paid: true,
        second: true,
        levels: DevelopmentLevels::PerAge,
    };
    /// Their model 2: log-normal on loss ratios, incremental paid.
    pub(super) const MODEL_2: Self = Self {
        likelihood: CompartmentalLikelihood::LogNormal,
        loss_ratio: true,
        incremental: true,
        incurred: false,
        paid: true,
        second: true,
        levels: DevelopmentLevels::PerAge,
    };

    pub(super) fn log_normal(self) -> bool {
        self.likelihood == CompartmentalLikelihood::LogNormal
    }

    /// Whether the cells of a type (`paid`, else the second response) are
    /// fitted.
    fn fits(self, paid: bool) -> bool {
        if paid { self.paid } else { self.second }
    }

    /// The fitted types in the order of their `log σ` in `theta` and in a
    /// natural-scale draw: the second response's (`false`), then paid's
    /// (`true`), each only when fitted.
    pub(super) fn sigma_types(self) -> impl Iterator<Item = bool> {
        [false, true].into_iter().filter(move |&p| self.fits(p))
    }

    /// The number of `log σ`: one per fitted type.
    pub(super) fn n_sigma(self) -> usize {
        usize::from(self.paid) + usize::from(self.second)
    }

    /// The position among the `log σ` of a fitted type's, `None` for a type
    /// not fitted.
    pub(super) fn sigma_slot(self, paid: bool) -> Option<usize> {
        self.fits(paid).then(|| usize::from(paid && self.second))
    }
}

/// Which of the paper's hierarchies a prior set has, for the provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Hierarchy {
    /// Origin effects on `RLR` and `RRF` only.
    Model1,
    /// Origin effects on all four parameters and development effects on
    /// all four.
    Model2,
    Other,
}

impl CompartmentalPriors {
    pub(super) fn hierarchy(&self) -> Hierarchy {
        let extra = [
            self.ker_effect_sd.is_some(),
            self.kp_effect_sd.is_some(),
            self.development_effect_sd[0].is_some(),
            self.development_effect_sd[1].is_some(),
            self.development_effect_sd[2].is_some(),
            self.development_effect_sd[3].is_some(),
        ];
        if extra.iter().all(|&e| !e) {
            Hierarchy::Model1
        } else if extra.iter().all(|&e| e) {
            Hierarchy::Model2
        } else {
            Hierarchy::Other
        }
    }

    /// The four positive priors, in the order `k_er`, `k_p`, `RLR`, `RRF`.
    pub(super) fn positive(&self) -> [PositivePrior; 4] {
        [self.k_er, self.k_p, self.rlr, self.rrf]
    }

    /// The origin effect sd priors on the rates, `k_er` then `k_p`.
    pub(super) fn rate_effect_sd(&self) -> [Option<HalfPrior>; 2] {
        [self.ker_effect_sd, self.kp_effect_sd]
    }
}
