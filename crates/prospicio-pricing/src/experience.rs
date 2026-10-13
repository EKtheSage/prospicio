//! Experience rating from individual large losses: the record a
//! reinsurance submission brings, trended to the prospective period, the
//! as-if loss to each layer by accident year, and the burning cost. See
//! `docs/design/experience-rating.md`.
//!
//! The steps follow Clark's *Basics of Reinsurance Pricing* (CAS study
//! note, 2014), section on per-risk experience rating:
//!
//! 1. Trend each loss from its accident date to the average accident date
//!    of the prospective period ([`Experience::prepare`]); dates are
//!    decimal years, as in [`ratemaking`](crate::ratemaking).
//! 2. Apply each year's treaty to the trended losses of that year
//!    ([`Prepared::burning_cost`]), with any annual terms.
//! 3. Develop each year's layer total with an excess development factor.
//! 4. Divide by the year's on-level subject premium: the loss cost by
//!    year, and in total the developed losses over the premium, summed.
//!
//! Losses reported above a threshold in nominal terms leave the older
//! years short of losses that trend above it. With a threshold `T`, the
//! prepared record is complete above `T` times the largest trend factor,
//! and losses trended below that are dropped, so every year is compared
//! on the same basis ([`Prepared::complete_above`]).

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
            .zip(&factors)
            .map(|(l, &factor)| PreparedLoss {
                id: l.id.clone(),
                year: l.accident.floor() as i32,
                accident: l.accident,
                amount: l.amount,
                factor,
                trended: l.amount * factor,
            })
            .collect();
        Ok(Prepared {
            losses,
            complete_above,
        })
    }
}

/// One loss trended to the prospective period.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedLoss {
    pub id: String,
    /// The accident year, the whole part of the accident date.
    pub year: i32,
    pub accident: f64,
    /// The loss as reported.
    pub amount: f64,
    /// The trend factor applied.
    pub factor: f64,
    /// `amount × factor`.
    pub trended: f64,
}

/// The trended record, from [`Experience::prepare`].
#[derive(Debug, Clone, PartialEq)]
pub struct Prepared {
    losses: Vec<PreparedLoss>,
    complete_above: f64,
}

/// An accident year's exposure base and excess development factor, for
/// [`Prepared::burning_cost`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YearBasis {
    pub year: i32,
    /// On-level subject premium (or another exposure base).
    pub premium: f64,
    /// The excess development factor for the year's layer losses.
    pub development: f64,
}

impl Prepared {
    /// Every loss, trended, in the record's order.
    pub fn losses(&self) -> &[PreparedLoss] {
        &self.losses
    }

    /// The trended size above which every year's record is complete: the
    /// reporting threshold times the largest trend factor. Losses trended
    /// below it are left out of [`burning_cost`](Self::burning_cost).
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
            let losses: Vec<f64> = self
                .losses
                .iter()
                .filter(|l| l.year == basis.year && l.trended >= self.complete_above)
                .map(|l| l.trended)
                .collect();
            let layer_loss = tower.ceded(&losses);
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
}
