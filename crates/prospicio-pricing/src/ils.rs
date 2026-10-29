//! Cat bonds and other insurance-linked securities (#570): a tranche of
//! principal lost on a trigger that need not be the cedant's own loss, its
//! expected loss and probabilities of attachment and exhaustion, its
//! spread under a distortion, the investor's return over the term and the
//! basis risk between the trigger and the cedant's indemnity loss.
//!
//! A trigger turns each simulated event into one index value:
//!
//! - [`Trigger::Indemnity`] and [`Trigger::ModelledLoss`]: the event's
//!   loss itself, the cedant's or a catastrophe model's;
//! - [`Trigger::IndustryIndex`]: `Σ w_c x_c` over the event's industry
//!   losses `x_c` by region or line, with the cedant's weights `w_c`
//!   (market shares, say);
//! - [`Trigger::Parametric`]: `Σ w_s f(x_s)` over a physical measure `x_s`
//!   at each site (wind speed, ground motion), with `f` a payout table read
//!   linearly between its points and flat beyond them, and `w_s` each
//!   site's amount.
//!
//! The [`Tranche`] attaches and exhausts on the index, per occurrence (each
//! event's index through the layer) or on the year's aggregate index. A
//! year's principal loss is the tranche's recovery over its width, at most
//! 1, and over a term of several years the principal is lost once: the
//! term's loss is the sum of its years' at most 1, and each year's coupon
//! is paid on the principal outstanding at its start (losses taken at the
//! year's end).

use prospicio_aggregate::EventSet;
use prospicio_core::{Error, Result};
use prospicio_prob::Distortion;
use prospicio_prob::distortion::{Family, calibrate};

use crate::returns::{Irr, IrrOptions, irr};

/// What turns an event into the index the tranche reads.
#[derive(Debug, Clone, PartialEq)]
pub enum Trigger {
    /// The cedant's own loss on the event (one value an event).
    Indemnity,
    /// A catastrophe model's loss on the event (one value an event).
    ModelledLoss,
    /// An industry index weighted by the cedant's shares: `Σ w_c x_c` over
    /// the event's industry losses, one weight each.
    IndustryIndex {
        /// One weight a component of the event.
        weights: Vec<f64>,
    },
    /// A parametric index: `Σ w_s f(x_s)` over a measure at each site,
    /// with `f` the payout table.
    Parametric {
        /// Each site's amount, one a component of the event.
        weights: Vec<f64>,
        /// The payout `(measure, share)` points, measures ascending,
        /// shares in `[0, 1]`; read linearly between the points and flat
        /// beyond them.
        payout: Vec<(f64, f64)>,
    },
}

impl Trigger {
    /// The number of values an event needs.
    fn components(&self) -> usize {
        match self {
            Trigger::Indemnity | Trigger::ModelledLoss => 1,
            Trigger::IndustryIndex { weights } | Trigger::Parametric { weights, .. } => {
                weights.len()
            }
        }
    }

    /// The index of an event with `values`, one a component.
    pub fn index(&self, values: &[f64]) -> Result<f64> {
        if values.len() != self.components() {
            return Err(Error::Data(format!(
                "an event has {} values; this trigger reads {}",
                values.len(),
                self.components()
            )));
        }
        if let Some(&x) = values.iter().find(|x| !x.is_finite()) {
            return Err(Error::InvalidParameter {
                name: "values",
                value: x,
                reason: "must be finite",
            });
        }
        Ok(match self {
            Trigger::Indemnity | Trigger::ModelledLoss => values[0],
            Trigger::IndustryIndex { weights } => {
                weights.iter().zip(values).map(|(w, x)| w * x).sum()
            }
            Trigger::Parametric { weights, payout } => weights
                .iter()
                .zip(values)
                .map(|(w, &x)| w * read_payout(payout, x))
                .sum(),
        })
    }

    fn check(&self) -> Result<()> {
        let weights = match self {
            Trigger::Indemnity | Trigger::ModelledLoss => return Ok(()),
            Trigger::IndustryIndex { weights } | Trigger::Parametric { weights, .. } => weights,
        };
        if weights.is_empty() {
            return Err(Error::Data("an index needs at least one weight".into()));
        }
        if let Some(&w) = weights.iter().find(|w| !(w.is_finite() && **w >= 0.0)) {
            return Err(Error::InvalidParameter {
                name: "weights",
                value: w,
                reason: "must be finite and non-negative",
            });
        }
        if let Trigger::Parametric { payout, .. } = self {
            if payout.is_empty() {
                return Err(Error::Data(
                    "a payout table needs at least one point".into(),
                ));
            }
            if let Some(&(x, _)) = payout.iter().find(|(x, _)| !x.is_finite()) {
                return Err(Error::InvalidParameter {
                    name: "payout",
                    value: x,
                    reason: "measures must be finite",
                });
            }
            if payout.windows(2).any(|w| w[0].0 >= w[1].0) {
                return Err(Error::Data(
                    "a payout table's measures must be strictly ascending".into(),
                ));
            }
            if let Some(&(_, s)) = payout.iter().find(|(_, s)| !(0.0..=1.0).contains(s)) {
                return Err(Error::InvalidParameter {
                    name: "payout",
                    value: s,
                    reason: "shares must be in [0, 1]",
                });
            }
        }
        Ok(())
    }
}

fn read_payout(table: &[(f64, f64)], x: f64) -> f64 {
    let (first, last) = (table[0], table[table.len() - 1]);
    if x <= first.0 {
        return first.1;
    }
    if x >= last.0 {
        return last.1;
    }
    let k = table.partition_point(|p| p.0 <= x);
    let ((x0, y0), (x1, y1)) = (table[k - 1], table[k]);
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// Whether the tranche reads each event or the year's total.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrancheBasis {
    /// Each event's index through the tranche, the recoveries summed over
    /// the year.
    PerOccurrence,
    /// The year's total index through the tranche.
    AnnualAggregate,
}

/// The tranche: the principal is lost linearly as the index runs from the
/// attachment to the exhaustion point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tranche {
    /// The index value at which the first principal is lost.
    pub attachment: f64,
    /// The index value at which all of it is lost.
    pub exhaustion: f64,
    /// Per occurrence or annual aggregate.
    pub basis: TrancheBasis,
}

/// A cat bond: a trigger, a tranche, its principal and its term.
///
/// ```
/// use prospicio_pricing::ils::{CatBond, Tranche, TrancheBasis, Trigger};
///
/// // 100 of principal on an industry index of two regions at shares 10%
/// // and 5%, attaching at 50 and exhausting at 150 per occurrence.
/// let tranche = Tranche { attachment: 50.0, exhaustion: 150.0, basis: TrancheBasis::PerOccurrence };
/// let bond = CatBond::new(Trigger::IndustryIndex { weights: vec![0.10, 0.05] }, tranche)
///     .unwrap()
///     .principal(100.0)
///     .unwrap();
/// // Year 1: one event of industry losses 800 and 400, an index of 100: half
/// // the principal. Year 2: no event.
/// let losses = bond.apply(&[vec![vec![800.0, 400.0]], vec![]]).unwrap();
/// assert_eq!(losses.annual(), [0.5, 0.0]);
/// assert_eq!(losses.expected_loss(), 0.25);
/// assert_eq!(losses.recoveries(), [50.0, 0.0]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CatBond {
    trigger: Trigger,
    tranche: Tranche,
    principal: f64,
    term_years: usize,
}

impl CatBond {
    /// A bond on `trigger` and `tranche`, with a principal of 1 and a term
    /// of one year.
    pub fn new(trigger: Trigger, tranche: Tranche) -> Result<Self> {
        trigger.check()?;
        let (a, e) = (tranche.attachment, tranche.exhaustion);
        if !(a.is_finite() && e.is_finite() && a >= 0.0 && e > a) {
            return Err(Error::Data(format!(
                "a tranche needs 0 <= attachment < exhaustion, both finite; got {a} and {e}"
            )));
        }
        Ok(Self {
            trigger,
            tranche,
            principal: 1.0,
            term_years: 1,
        })
    }

    /// The principal, positive.
    pub fn principal(mut self, principal: f64) -> Result<Self> {
        if !(principal.is_finite() && principal > 0.0) {
            return Err(Error::InvalidParameter {
                name: "principal",
                value: principal,
                reason: "must be positive and finite",
            });
        }
        self.principal = principal;
        Ok(self)
    }

    /// The term in years, at least 1: consecutive runs of that many
    /// simulated years make one term.
    pub fn term_years(mut self, years: usize) -> Result<Self> {
        if years == 0 {
            return Err(Error::InvalidParameter {
                name: "term_years",
                value: 0.0,
                reason: "must be at least 1",
            });
        }
        self.term_years = years;
        Ok(self)
    }

    /// The trigger.
    pub fn trigger(&self) -> &Trigger {
        &self.trigger
    }

    /// The tranche.
    pub fn tranche(&self) -> Tranche {
        self.tranche
    }

    /// The share of principal one year loses, from its events' index
    /// values, in `[0, 1]`.
    pub fn annual_loss(&self, index: &[f64]) -> f64 {
        let Tranche {
            attachment: a,
            exhaustion: e,
            basis,
        } = self.tranche;
        let layer = |x: f64| (x - a).clamp(0.0, e - a);
        let recovered = match basis {
            TrancheBasis::PerOccurrence => index.iter().map(|&x| layer(x)).sum(),
            TrancheBasis::AnnualAggregate => layer(index.iter().sum()),
        };
        (recovered / (e - a)).min(1.0)
    }

    /// The bond on simulated years: `years[i][e]` the values of event `e` of
    /// year `i`, as many as the trigger reads. The number of years must be a
    /// whole number of terms.
    pub fn apply(&self, years: &[Vec<Vec<f64>>]) -> Result<CatBondLosses> {
        let annual = years
            .iter()
            .map(|events| {
                let index = events
                    .iter()
                    .map(|v| self.trigger.index(v))
                    .collect::<Result<Vec<_>>>()?;
                Ok(self.annual_loss(&index))
            })
            .collect::<Result<Vec<_>>>()?;
        self.losses(annual)
    }

    /// The bond on an event set's years, for a trigger that reads one value
    /// an event (indemnity or a modelled loss).
    pub fn apply_events(&self, events: &EventSet) -> Result<CatBondLosses> {
        if self.trigger.components() != 1 {
            return Err(Error::Data(
                "an event set holds one value an event; give an index's components to `apply`"
                    .into(),
            ));
        }
        let annual = (0..events.n_sims())
            .map(|i| self.annual_loss(events.events(i)))
            .collect();
        self.losses(annual)
    }

    fn losses(&self, annual: Vec<f64>) -> Result<CatBondLosses> {
        let t = self.term_years;
        if annual.is_empty() || annual.len() % t != 0 {
            return Err(Error::Data(format!(
                "{} simulated years are not a whole number of {t}-year terms",
                annual.len()
            )));
        }
        Ok(CatBondLosses {
            annual,
            principal: self.principal,
            term_years: t,
        })
    }
}

/// A cat bond's principal losses on simulated years.
#[derive(Debug, Clone, PartialEq)]
pub struct CatBondLosses {
    annual: Vec<f64>,
    principal: f64,
    term_years: usize,
}

/// A cat bond's price under a distortion, per unit of principal a year.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CatBondPrice {
    /// The expected annual loss.
    pub expected_loss: f64,
    /// The risk-adjusted annual loss, the spread over the collateral
    /// yield: `ρ_g(L)`.
    pub spread: f64,
    /// `spread − expected_loss`.
    pub risk_load: f64,
    /// `spread / expected_loss`, NaN when the expected loss is 0.
    pub multiple: f64,
}

/// The gap between what a trigger pays and what the cedant needs.
#[derive(Debug, Clone, PartialEq)]
pub struct BasisRisk {
    /// Each year's recovery less the need.
    pub difference: Vec<f64>,
    /// The mean of `max(need − recovery, 0)`.
    pub mean_shortfall: f64,
    /// The mean of `max(recovery − need, 0)`.
    pub mean_excess: f64,
    /// The standard deviation of the difference.
    pub std_dev: f64,
    /// The correlation of recovery and need, NaN when either is constant.
    pub correlation: f64,
    /// The share of years with a need and no recovery.
    pub missed: f64,
}

impl CatBondLosses {
    /// Losses from each simulated year's share of principal lost (in
    /// `[0, 1]`), the principal and the term, as [`CatBond::apply`] gives
    /// them.
    pub fn from_annual(annual: Vec<f64>, principal: f64, term_years: usize) -> Result<Self> {
        if let Some(&x) = annual.iter().find(|x| !(0.0..=1.0).contains(*x)) {
            return Err(Error::InvalidParameter {
                name: "annual",
                value: x,
                reason: "a share of principal lost is in [0, 1]",
            });
        }
        let bond = CatBond::new(
            Trigger::Indemnity,
            Tranche {
                attachment: 0.0,
                exhaustion: 1.0,
                basis: TrancheBasis::PerOccurrence,
            },
        )?
        .principal(principal)?
        .term_years(term_years)?;
        bond.losses(annual)
    }

    /// Each year's share of principal lost, in `[0, 1]`.
    pub fn annual(&self) -> &[f64] {
        &self.annual
    }

    /// Each year's recovery, the share lost times the principal.
    pub fn recoveries(&self) -> Vec<f64> {
        self.annual.iter().map(|l| l * self.principal).collect()
    }

    /// The mean annual share lost.
    pub fn expected_loss(&self) -> f64 {
        self.annual.iter().sum::<f64>() / self.annual.len() as f64
    }

    /// The share of years with any loss: the probability of attachment.
    pub fn attachment_probability(&self) -> f64 {
        share(&self.annual, |l| l > 0.0)
    }

    /// The share of years that lose all the principal: the probability of
    /// exhaustion.
    pub fn exhaustion_probability(&self) -> f64 {
        share(&self.annual, |l| l >= 1.0)
    }

    /// The share of principal lost over each term, `min(Σ years, 1)`.
    pub fn term_losses(&self) -> Vec<f64> {
        self.annual
            .chunks(self.term_years)
            .map(|c| c.iter().sum::<f64>().min(1.0))
            .collect()
    }

    /// The price under `distortion`: the spread `ρ_g(L)` of the annual
    /// share lost `L` over its simulated years, each weighing the same.
    pub fn price(&self, distortion: &Distortion) -> CatBondPrice {
        let mut sorted = self.annual.clone();
        sorted.sort_by(f64::total_cmp);
        let spread = distortion.apply_sorted(&sorted);
        let el = self.expected_loss();
        CatBondPrice {
            expected_loss: el,
            spread,
            risk_load: spread - el,
            multiple: if el > 0.0 { spread / el } else { f64::NAN },
        }
    }

    /// The distortion of `family` that prices the bond at `spread`, as
    /// [`calibrate`] finds it on the annual shares lost.
    pub fn calibrate(&self, family: Family, spread: f64) -> Result<Distortion> {
        let mut sorted = self.annual.clone();
        sorted.sort_by(f64::total_cmp);
        let p = vec![1.0 / sorted.len() as f64; sorted.len()];
        calibrate(family, &sorted, &p, spread)
    }

    /// Lane's (2000) spread `γ PFL^α CEL^β` from the probability of first
    /// loss `PFL` (the attachment probability) and the conditional expected
    /// loss `CEL = EL / PFL`, with the coefficients given (fitted to a
    /// market's bonds by the user); 0 when the bond never attaches.
    pub fn lane_spread(&self, gamma: f64, alpha: f64, beta: f64) -> f64 {
        let pfl = self.attachment_probability();
        if pfl == 0.0 {
            return 0.0;
        }
        gamma * pfl.powf(alpha) * (self.expected_loss() / pfl).powf(beta)
    }

    /// The investor's annual cash flows in each term per unit of principal:
    /// `−1` at time 0, at the end of year `y` the coupon on the principal
    /// outstanding at its start, and the principal left returned at the
    /// end of the term.
    pub fn investor_flows(&self, coupon: f64) -> Vec<Vec<f64>> {
        self.annual
            .chunks(self.term_years)
            .map(|c| {
                let mut flows = vec![-1.0];
                let mut lost = 0.0f64;
                for &l in c {
                    flows.push(coupon * (1.0 - lost));
                    lost = (lost + l).min(1.0);
                }
                let last = flows.len() - 1;
                flows[last] += 1.0 - lost;
                flows
            })
            .collect()
    }

    /// The investor's internal rate of return in each term at `coupon` (the
    /// collateral yield plus the spread), by [`irr`].
    pub fn investor_returns(&self, coupon: f64, options: &IrrOptions) -> Result<Vec<Irr>> {
        let times: Vec<f64> = (0..=self.term_years).map(|t| t as f64).collect();
        self.investor_flows(coupon)
            .iter()
            .map(|f| irr(f, &times, options))
            .collect()
    }

    /// The basis risk against `need`, the cedant's recovery from an
    /// indemnity cover in each simulated year (in currency, as
    /// [`recoveries`](Self::recoveries)).
    pub fn basis_risk(&self, need: &[f64]) -> Result<BasisRisk> {
        if need.len() != self.annual.len() {
            return Err(Error::Data(format!(
                "{} needs for {} simulated years",
                need.len(),
                self.annual.len()
            )));
        }
        let r = self.recoveries();
        let n = r.len() as f64;
        let difference: Vec<f64> = r.iter().zip(need).map(|(a, b)| a - b).collect();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / n;
        let md = mean(&difference);
        let std_dev = (difference.iter().map(|d| (d - md).powi(2)).sum::<f64>() / n).sqrt();
        let (mr, mn) = (mean(&r), mean(need));
        let cov: f64 = r
            .iter()
            .zip(need)
            .map(|(a, b)| (a - mr) * (b - mn))
            .sum::<f64>()
            / n;
        let sr = (r.iter().map(|a| (a - mr).powi(2)).sum::<f64>() / n).sqrt();
        let sn = (need.iter().map(|b| (b - mn).powi(2)).sum::<f64>() / n).sqrt();
        Ok(BasisRisk {
            mean_shortfall: difference.iter().map(|d| (-d).max(0.0)).sum::<f64>() / n,
            mean_excess: difference.iter().map(|d| d.max(0.0)).sum::<f64>() / n,
            std_dev,
            correlation: if sr > 0.0 && sn > 0.0 {
                cov / (sr * sn)
            } else {
                f64::NAN
            },
            missed: r
                .iter()
                .zip(need)
                .filter(|(a, b)| **b > 0.0 && **a == 0.0)
                .count() as f64
                / n,
            difference,
        })
    }
}

fn share(v: &[f64], f: impl Fn(f64) -> bool) -> f64 {
    v.iter().filter(|&&x| f(x)).count() as f64 / v.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_aggregate::Layer;
    use prospicio_core::StreamRng;

    /// Years of events, each event's values `components` exponentials.
    fn years(n: usize, components: usize, seed: u64) -> Vec<Vec<Vec<f64>>> {
        (0..n as u64)
            .map(|i| {
                let mut rng = StreamRng::new(seed, i);
                let k = (rng.next_open01() * 4.0) as usize;
                (0..k)
                    .map(|_| {
                        (0..components)
                            .map(|_| -30.0 * rng.next_open01().ln())
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    fn per_occ(a: f64, e: f64) -> Tranche {
        Tranche {
            attachment: a,
            exhaustion: e,
            basis: TrancheBasis::PerOccurrence,
        }
    }

    #[test]
    fn an_indemnity_tranche_is_the_layer_with_its_limit_once_a_year() {
        let ys = years(3_000, 1, 2);
        for basis in [TrancheBasis::PerOccurrence, TrancheBasis::AnnualAggregate] {
            let t = Tranche {
                attachment: 40.0,
                exhaustion: 100.0,
                basis,
            };
            let bond = CatBond::new(Trigger::Indemnity, t)
                .unwrap()
                .principal(60.0)
                .unwrap();
            let layer = match basis {
                TrancheBasis::PerOccurrence => Layer::xol("T", 60.0, 40.0)
                    .unwrap()
                    .aggregate_limit(60.0)
                    .unwrap(),
                TrancheBasis::AnnualAggregate => Layer::stop_loss("T", 60.0, 40.0).unwrap(),
            };
            let r = bond.apply(&ys).unwrap().recoveries();
            for (i, y) in ys.iter().enumerate() {
                let flat: Vec<f64> = y.iter().map(|e| e[0]).collect();
                assert!((r[i] - layer.ceded(&flat)).abs() < 1e-9, "{basis:?} {i}");
            }
            // Through an event set too.
            let flat: Vec<Vec<f64>> = ys
                .iter()
                .map(|y| y.iter().map(|e| e[0]).collect())
                .collect();
            let set = EventSet::from_years(flat.clone(), 0).unwrap();
            assert_eq!(bond.apply_events(&set).unwrap().recoveries(), r);
            // Attachment and exhaustion are the OEP or the AEP at the points.
            let l = bond.apply(&ys).unwrap();
            let n = ys.len() as f64;
            let (pa, pe) = match basis {
                TrancheBasis::PerOccurrence => (
                    flat.iter().filter(|y| y.iter().any(|&x| x > 40.0)).count() as f64 / n,
                    flat.iter()
                        .filter(|y| {
                            y.iter().map(|&x| (x - 40.0).clamp(0.0, 60.0)).sum::<f64>() >= 60.0
                        })
                        .count() as f64
                        / n,
                ),
                TrancheBasis::AnnualAggregate => (
                    flat.iter().filter(|y| y.iter().sum::<f64>() > 40.0).count() as f64 / n,
                    flat.iter()
                        .filter(|y| y.iter().sum::<f64>() >= 100.0)
                        .count() as f64
                        / n,
                ),
            };
            assert_eq!(
                (l.attachment_probability(), l.exhaustion_probability()),
                (pa, pe)
            );
            assert!(pa > 0.0 && pe > 0.0 && pe < pa);
            // Zero basis risk against the same indemnity cover.
            let b = l.basis_risk(&r).unwrap();
            assert_eq!(
                (b.mean_shortfall, b.mean_excess, b.std_dev, b.missed),
                (0.0, 0.0, 0.0, 0.0)
            );
            assert!((b.correlation - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn indices_read_their_components() {
        let index = Trigger::IndustryIndex {
            weights: vec![0.1, 0.2],
        };
        assert!((index.index(&[100.0, 50.0]).unwrap() - 20.0).abs() < 1e-12);
        assert!(index.index(&[100.0]).is_err());
        let p = Trigger::Parametric {
            weights: vec![10.0, 20.0],
            payout: vec![(30.0, 0.0), (50.0, 0.5), (70.0, 1.0)],
        };
        // 40 m/s pays a quarter, 80 m/s all: 10 * 0.25 + 20 * 1.
        assert!((p.index(&[40.0, 80.0]).unwrap() - 22.5).abs() < 1e-12);
        assert_eq!(p.index(&[10.0, 30.0]).unwrap(), 0.0);
        let bad = Trigger::Parametric {
            weights: vec![1.0],
            payout: vec![(50.0, 0.5), (30.0, 1.0)],
        };
        assert!(CatBond::new(bad, per_occ(0.0, 1.0)).is_err());
        assert!(CatBond::new(Trigger::Indemnity, per_occ(5.0, 5.0)).is_err());
        let two = CatBond::new(index, per_occ(0.0, 1.0)).unwrap();
        assert!(
            two.apply_events(&EventSet::from_years(vec![vec![1.0]], 0).unwrap())
                .is_err()
        );
    }

    #[test]
    fn the_wang_spread_of_a_binary_tranche_is_its_closed_form() {
        use prospicio_math::special::{norm_cdf, norm_quantile};
        // A tranche that is all or nothing: the index jumps past it.
        let n = 10_000;
        let ys: Vec<Vec<Vec<f64>>> = (0..n)
            .map(|i| {
                if i % 50 == 0 {
                    vec![vec![500.0]]
                } else {
                    vec![]
                }
            })
            .collect();
        let l = CatBond::new(Trigger::ModelledLoss, per_occ(100.0, 200.0))
            .unwrap()
            .apply(&ys)
            .unwrap();
        let p = 0.02;
        assert!((l.expected_loss() - p).abs() < 1e-15);
        let wang = Distortion::wang(0.45).unwrap();
        let price = l.price(&wang);
        let want = norm_cdf(norm_quantile(p) + 0.45);
        assert!(
            (price.spread - want).abs() < 1e-12,
            "{} {want}",
            price.spread
        );
        assert!((price.multiple - want / p).abs() < 1e-9);
        // Calibration finds λ back.
        let back = l.calibrate(Family::Wang, price.spread).unwrap();
        assert!((back.parameter().unwrap() - 0.45).abs() < 1e-6, "{back:?}");
        // Lane's form at chosen coefficients: PFL 0.02, CEL 1.
        assert!((l.lane_spread(0.55, 0.5, 0.5) - 0.55 * 0.02f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn principal_is_lost_once_over_a_term() {
        let bond = CatBond::new(Trigger::Indemnity, per_occ(0.0, 10.0))
            .unwrap()
            .term_years(3)
            .unwrap();
        // Term 1: 40%, then 80% (only 60% left), then nothing. Term 2: none.
        let ys = vec![
            vec![vec![4.0]],
            vec![vec![8.0]],
            vec![],
            vec![],
            vec![],
            vec![],
        ];
        let l = bond.apply(&ys).unwrap();
        assert_eq!(l.term_losses(), [1.0, 0.0]);
        let flows = l.investor_flows(0.1);
        let want = [-1.0, 0.1, 0.06, 0.0];
        for (g, w) in flows[0].iter().zip(want) {
            assert!((g - w).abs() < 1e-12, "{:?}", flows[0]);
        }
        assert_eq!(flows[1], [-1.0, 0.1, 0.1, 1.1]);
        let r = l.investor_returns(0.1, &IrrOptions::default()).unwrap();
        // No loss: the coupon.
        assert!((r[1].rate - 0.1).abs() < 1e-9);
        assert!(r[0].rate < -0.5);
        assert!(bond.apply(&ys[..4]).is_err());
    }
}
