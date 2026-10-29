//! An efficient frontier of reinsurance structures (#569).
//!
//! For each retention and limit of a layer, the cost of the cover (the
//! margin paid away: ceded premium less expected ceded loss) against the
//! risk kept (a VaR or TVaR of the net loss), and which structures no
//! other beats on both. A cedant chooses among the efficient ones by its
//! budget ([`Frontier::best`]) or by the total of retained risk and
//! premium ([`Frontier::least_total`], Cai and Tan's objective).
//!
//! The losses come from simulated years with their events
//! ([`frontier_events`], a per-occurrence or an aggregate layer, each
//! structure the tower API's `Layer::xol` or `Layer::stop_loss`),
//! from annual losses ([`frontier_annual`], an aggregate layer), or from an
//! annual loss on a grid ([`frontier_grid`], exact, one gross distribution
//! for every structure, with retentions and limits on the grid's step).
//! The ceded premium is a [`PremiumRule`] on the assets of a capital
//! standard, as [`price`](crate::risk_load::price) prices one cover, or
//! the expected value principle `(1 + ρ) E[C]`.

use prospicio_aggregate::{EventSet, Layer};
use prospicio_core::{Error, Result};
use prospicio_prob::{Distortion, Distribution, Empirical, Grid, Sampled};

use crate::risk_load::{PremiumRule, priced};

/// How the layer reads the year's losses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Each occurrence: `min(max(x − r, 0), l)` summed over the year.
    PerOccurrence,
    /// The year's total: `min(max(S − r, 0), l)`.
    Aggregate,
}

/// The ceded premium.
#[derive(Debug, Clone, PartialEq)]
pub enum CededPrice {
    /// A premium rule on the assets of a capital standard, as
    /// [`price`](crate::risk_load::price) prices one cover.
    Rule {
        /// The premium rule.
        rule: PremiumRule,
        /// The capital standard for the assets.
        assets: Distortion,
    },
    /// The expected value principle `(1 + loading) E[C]`.
    ExpectedValue {
        /// The loading, at least 0.
        loading: f64,
    },
}

/// The risk kept: a measure of the net loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RetainedRisk {
    /// The value at risk of the net at a level.
    Var(f64),
    /// The tail value at risk of the net at a level.
    Tvar(f64),
}

/// One structure on the frontier.
#[derive(Debug, Clone, PartialEq)]
pub struct FrontierPoint {
    /// The layer's retention.
    pub retention: f64,
    /// The layer's limit (`inf` for none).
    pub limit: f64,
    /// The ceded premium.
    pub premium: f64,
    /// The expected ceded loss.
    pub expected_ceded: f64,
    /// `premium − expected_ceded`, the cost of the cover.
    pub margin: f64,
    /// The risk measure of the net loss.
    pub net_risk: f64,
    /// Whether no other structure has both a margin and a net risk at most
    /// this one's, with one of them smaller.
    pub efficient: bool,
}

/// Every structure evaluated, in the order of the retentions, then the
/// limits.
#[derive(Debug, Clone, PartialEq)]
pub struct Frontier {
    /// The structures.
    pub points: Vec<FrontierPoint>,
}

impl Frontier {
    fn new(mut points: Vec<FrontierPoint>) -> Self {
        // Sorted by margin (then risk), a point is efficient when its risk
        // is below every cheaper point's.
        let mut order: Vec<usize> = (0..points.len()).collect();
        order.sort_by(|&a, &b| {
            points[a]
                .margin
                .total_cmp(&points[b].margin)
                .then(points[a].net_risk.total_cmp(&points[b].net_risk))
        });
        // A point equal to the last efficient one in both is not dominated
        // by it either.
        let mut best: Option<(f64, f64)> = None;
        for &i in &order {
            let (m, r) = (points[i].margin, points[i].net_risk);
            let efficient = match best {
                None => true,
                Some((bm, br)) => r < br || (r == br && m == bm),
            };
            if efficient {
                points[i].efficient = true;
                best = Some((m, r));
            }
        }
        Frontier { points }
    }

    /// The efficient structures, by margin.
    pub fn efficient(&self) -> Vec<&FrontierPoint> {
        let mut e: Vec<&FrontierPoint> = self.points.iter().filter(|p| p.efficient).collect();
        e.sort_by(|a, b| a.margin.total_cmp(&b.margin));
        e
    }

    /// The structure with the least net risk whose margin is at most
    /// `budget`; `None` when none is within it.
    pub fn best(&self, budget: f64) -> Option<&FrontierPoint> {
        self.points
            .iter()
            .filter(|p| p.margin <= budget)
            .min_by(|a, b| a.net_risk.total_cmp(&b.net_risk))
    }

    /// The structure with the least `net_risk + premium`: the risk of the
    /// cedant's total cost, retained loss plus premium, as Cai and Tan
    /// (2007) minimise it.
    pub fn least_total(&self) -> &FrontierPoint {
        self.points
            .iter()
            .min_by(|a, b| (a.net_risk + a.premium).total_cmp(&(b.net_risk + b.premium)))
            .expect("a frontier has a point")
    }
}

fn check_inputs(
    retentions: &[f64],
    limits: &[f64],
    price: &CededPrice,
    risk: RetainedRisk,
) -> Result<()> {
    if retentions.is_empty() || limits.is_empty() {
        return Err(Error::Data(
            "give at least one retention and one limit".into(),
        ));
    }
    if let Some(&r) = retentions.iter().find(|r| !(r.is_finite() && **r >= 0.0)) {
        return Err(Error::InvalidParameter {
            name: "retentions",
            value: r,
            reason: "must be finite and at least 0",
        });
    }
    if let Some(&l) = limits.iter().find(|l| l.is_nan() || **l <= 0.0) {
        return Err(Error::InvalidParameter {
            name: "limits",
            value: l,
            reason: "must be positive (inf for none)",
        });
    }
    if let CededPrice::ExpectedValue { loading } = price
        && !(loading.is_finite() && *loading >= 0.0)
    {
        return Err(Error::InvalidParameter {
            name: "loading",
            value: *loading,
            reason: "must be finite and at least 0",
        });
    }
    let p = match risk {
        RetainedRisk::Var(p) | RetainedRisk::Tvar(p) => p,
    };
    if !(0.0..1.0).contains(&p) {
        return Err(Error::InvalidParameter {
            name: "level",
            value: p,
            reason: "must be in [0, 1)",
        });
    }
    Ok(())
}

/// The ceded premium from the expected ceded loss and a distortion measure
/// of the ceded loss.
fn premium(expected: f64, measure: impl Fn(&Distortion) -> f64, price: &CededPrice) -> f64 {
    match price {
        CededPrice::Rule { rule, assets } => priced(expected, measure, rule, assets).premium,
        CededPrice::ExpectedValue { loading } => (1.0 + loading) * expected,
    }
}

/// The frontier from per-year ceded and net losses for each structure.
fn from_samples(
    gross: &[f64],
    retentions: &[f64],
    limits: &[f64],
    price: &CededPrice,
    risk: RetainedRisk,
    ceded_of: impl Fn(f64, f64) -> Result<Vec<f64>>,
) -> Result<Frontier> {
    check_inputs(retentions, limits, price, risk)?;
    let mut points = Vec::with_capacity(retentions.len() * limits.len());
    for &r in retentions {
        for &l in limits {
            let ceded = ceded_of(r, l)?;
            let net: Vec<f64> = gross.iter().zip(&ceded).map(|(g, c)| g - c).collect();
            let ceded = Sampled::new(ceded)?;
            let net = Sampled::new(net)?;
            let expected = ceded.mean();
            let p = premium(expected, |d| ceded.distortion(d), price);
            let net_risk = match risk {
                RetainedRisk::Var(q) => net.var(q)?,
                RetainedRisk::Tvar(q) => net.tvar(q)?,
            };
            points.push(FrontierPoint {
                retention: r,
                limit: l,
                premium: p,
                expected_ceded: expected,
                margin: p - expected,
                net_risk,
                efficient: false,
            });
        }
    }
    Ok(Frontier::new(points))
}

/// The frontier on simulated years with their events: each structure a
/// layer of `retention` and `limit`, per occurrence
/// ([`Layer::xol`]) or on the year's total ([`Layer::stop_loss`]).
///
/// ```
/// use prospicio_aggregate::simulate_events;
/// use prospicio_pricing::frontier::{CededPrice, LayerKind, RetainedRisk, frontier_events};
/// use prospicio_prob::{Lognormal, Poisson};
///
/// let ev = simulate_events(&Poisson::new(3.0).unwrap(),
///     &Lognormal::from_mean_cv(10.0, 1.5).unwrap(), 5_000, 1).unwrap();
/// let f = frontier_events(&ev, LayerKind::PerOccurrence, &[10.0, 20.0, 40.0],
///     &[20.0, 50.0, f64::INFINITY], &CededPrice::ExpectedValue { loading: 0.3 },
///     RetainedRisk::Tvar(0.99)).unwrap();
/// assert_eq!(f.points.len(), 9);
/// // An unlimited cover from the lowest retention keeps the least risk.
/// let least = f.points.iter().min_by(|a, b| a.net_risk.total_cmp(&b.net_risk)).unwrap();
/// assert_eq!((least.retention, least.limit), (10.0, f64::INFINITY));
/// assert!(least.efficient);
/// ```
pub fn frontier_events(
    events: &EventSet,
    kind: LayerKind,
    retentions: &[f64],
    limits: &[f64],
    price: &CededPrice,
    risk: RetainedRisk,
) -> Result<Frontier> {
    let n = events.n_sims();
    let gross: Vec<f64> = (0..n).map(|i| events.events(i).iter().sum()).collect();
    from_samples(&gross, retentions, limits, price, risk, |r, l| {
        let layer = match kind {
            LayerKind::PerOccurrence => Layer::xol("frontier", l, r)?,
            LayerKind::Aggregate => Layer::stop_loss("frontier", l, r)?,
        };
        Ok(match kind {
            LayerKind::PerOccurrence => (0..n).map(|i| layer.ceded(events.events(i))).collect(),
            LayerKind::Aggregate => gross.iter().map(|&s| layer.ceded(&[s])).collect(),
        })
    })
}

/// The frontier of aggregate layers on annual losses, one a simulated
/// year.
pub fn frontier_annual(
    losses: &[f64],
    retentions: &[f64],
    limits: &[f64],
    price: &CededPrice,
    risk: RetainedRisk,
) -> Result<Frontier> {
    from_samples(losses, retentions, limits, price, risk, |r, l| {
        Ok(losses.iter().map(|&s| (s - r).clamp(0.0, l)).collect())
    })
}

/// The frontier of aggregate layers on an annual loss on a grid, exact:
/// the ceded and net losses of each structure are the gross grid's points
/// moved, so every retention and finite limit must be a whole number of
/// the grid's steps.
///
/// ```
/// use prospicio_pricing::frontier::{CededPrice, RetainedRisk, frontier_grid};
/// use prospicio_prob::Grid;
///
/// // Annual losses 0, 1, 2, 3 equally likely.
/// let g = Grid::new(1.0, vec![0.25; 4]).unwrap();
/// let f = frontier_grid(&g, &[1.0, 2.0], &[1.0, f64::INFINITY],
///     &CededPrice::ExpectedValue { loading: 0.5 }, RetainedRisk::Var(0.9)).unwrap();
/// // 2 xs 1: ceded 0, 0, 1, 2 (mean 0.75); net 0, 1, 1, 1.
/// let p = &f.points[1];
/// assert_eq!((p.expected_ceded, p.premium, p.net_risk), (0.75, 1.125, 1.0));
/// ```
pub fn frontier_grid(
    annual: &Grid,
    retentions: &[f64],
    limits: &[f64],
    price: &CededPrice,
    risk: RetainedRisk,
) -> Result<Frontier> {
    check_inputs(retentions, limits, price, risk)?;
    let h = annual.step();
    let steps = |x: f64, name: &'static str| -> Result<usize> {
        let k = (x / h).round();
        if (k * h - x).abs() > 1e-9 * h.max(x.abs()) {
            return Err(Error::InvalidParameter {
                name,
                value: x,
                reason: "must be a whole number of the grid's steps",
            });
        }
        Ok(k as usize)
    };
    let probs = annual.probs();
    let mut points = Vec::with_capacity(retentions.len() * limits.len());
    for &r in retentions {
        let kr = steps(r, "retentions")?;
        for &l in limits {
            let kl = if l.is_finite() {
                steps(l, "limits")?
            } else {
                usize::MAX
            };
            let mut ceded = vec![0.0; probs.len()];
            let mut net = vec![0.0; probs.len()];
            for (k, &p) in probs.iter().enumerate() {
                let c = k.saturating_sub(kr).min(kl);
                ceded[c] += p;
                net[k - c] += p;
            }
            let ceded = Grid::new(h, ceded)?;
            let net = Grid::new(h, net)?;
            let expected = ceded.mean();
            let p = premium(expected, |d| ceded.distortion(d), price);
            let net_risk = match risk {
                RetainedRisk::Var(q) => net.quantile(q)?,
                RetainedRisk::Tvar(q) => net.tvar(q)?,
            };
            points.push(FrontierPoint {
                retention: r,
                limit: l,
                premium: p,
                expected_ceded: expected,
                margin: p - expected,
                net_risk,
                efficient: false,
            });
        }
    }
    Ok(Frontier::new(points))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_aggregate::{Tower, simulate_events};
    use prospicio_prob::{Lognormal, Poisson};

    #[test]
    fn the_stop_loss_retention_is_cai_and_tans() {
        // Exponential annual losses (mean 1) on a fine grid; the expected
        // value principle at loading 0.25 puts the retention that minimises
        // VaR(net) + premium at S(d) = 1 / 1.25, d = ln 1.25.
        let h = 0.001;
        let mut probs: Vec<f64> = (0..20_000)
            .map(|k| (-(k as f64) * h).exp() * (1.0 - (-h).exp()))
            .collect();
        // The tail past 20 on the last point.
        let rest = 1.0 - probs.iter().sum::<f64>();
        *probs.last_mut().unwrap() += rest;
        let g = Grid::new(h, probs).unwrap();
        let retentions: Vec<f64> = (0..600).map(|k| k as f64 * h).collect();
        let price = CededPrice::ExpectedValue { loading: 0.25 };
        for risk in [RetainedRisk::Var(0.99), RetainedRisk::Tvar(0.99)] {
            let f = frontier_grid(&g, &retentions, &[f64::INFINITY], &price, risk).unwrap();
            let d = f.least_total().retention;
            assert!((d - 1.25f64.ln()).abs() <= 2.0 * h, "{risk:?}: {d}");
        }
    }

    #[test]
    fn efficient_points_are_not_dominated() {
        let ev = simulate_events(
            &Poisson::new(3.0).unwrap(),
            &Lognormal::from_mean_cv(10.0, 1.5).unwrap(),
            4_000,
            2,
        )
        .unwrap();
        let price = CededPrice::Rule {
            rule: PremiumRule::cost_of_capital(0.1).unwrap(),
            assets: Distortion::tvar(0.99).unwrap(),
        };
        let retentions: Vec<f64> = (1..=12).map(|k| 5.0 * k as f64).collect();
        let limits = [10.0, 25.0, 50.0, 100.0, f64::INFINITY];
        let f = frontier_events(
            &ev,
            LayerKind::PerOccurrence,
            &retentions,
            &limits,
            &price,
            RetainedRisk::Tvar(0.99),
        )
        .unwrap();
        for e in f.efficient() {
            assert!(!f.points.iter().any(|q| q.margin <= e.margin
                && q.net_risk <= e.net_risk
                && (q.margin < e.margin || q.net_risk < e.net_risk)));
        }
        // Every inefficient point is dominated by some point.
        for p in f.points.iter().filter(|p| !p.efficient) {
            assert!(f.points.iter().any(|q| q.margin <= p.margin
                && q.net_risk <= p.net_risk
                && (q.margin < p.margin || q.net_risk < p.net_risk)));
        }
        // Each structure is the tower API's on the same years.
        let p = &f.points[7];
        let tower = Tower::new(vec![Layer::xol("L", p.limit, p.retention).unwrap()]).unwrap();
        let ceded: Vec<f64> = (0..ev.n_sims())
            .map(|i| tower.ceded(ev.events(i))[0])
            .collect();
        let mean = ceded.iter().sum::<f64>() / ceded.len() as f64;
        assert!((mean - p.expected_ceded).abs() < 1e-9);
        // A budget picks the least risk within it.
        let budget = f.efficient()[2].margin;
        let b = f.best(budget).unwrap();
        assert!(b.margin <= budget && b.efficient);
        assert!(f.best(-1.0).is_none());
    }

    #[test]
    fn the_annual_and_grid_frontiers_agree_on_a_discrete_loss() {
        let g = Grid::new(1.0, vec![0.1, 0.2, 0.3, 0.2, 0.1, 0.1]).unwrap();
        // The same distribution as 1,000 equally likely years.
        let mut years = Vec::new();
        for (k, p) in g.probs().iter().enumerate() {
            years.extend(std::iter::repeat_n(k as f64, (p * 1000.0).round() as usize));
        }
        let price = CededPrice::Rule {
            rule: PremiumRule::Distortion(Distortion::wang(0.2).unwrap()),
            assets: Distortion::tvar(0.95).unwrap(),
        };
        let (rs, ls) = ([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, f64::INFINITY]);
        let a = frontier_annual(&years, &rs, &ls, &price, RetainedRisk::Tvar(0.9)).unwrap();
        let b = frontier_grid(&g, &rs, &ls, &price, RetainedRisk::Tvar(0.9)).unwrap();
        for (x, y) in a.points.iter().zip(&b.points) {
            assert!((x.premium - y.premium).abs() < 1e-9, "{x:?} {y:?}");
            assert!((x.net_risk - y.net_risk).abs() < 1e-9);
            // The flags agree except where rounding splits a tie in one
            // coordinate (net risks of 3 and 3 - 4e-15, say).
            if x.efficient != y.efficient {
                assert!(a.points.iter().any(|q| !std::ptr::eq(q, x)
                    && ((q.margin - x.margin).abs() < 1e-9
                        || (q.net_risk - x.net_risk).abs() < 1e-9)));
            }
        }
        assert!(frontier_grid(&g, &[0.5], &ls, &price, RetainedRisk::Var(0.9)).is_err());
        assert!(frontier_annual(&years, &[], &ls, &price, RetainedRisk::Var(0.9)).is_err());
        assert!(frontier_annual(&years, &rs, &ls, &price, RetainedRisk::Var(1.0)).is_err());
    }
}
