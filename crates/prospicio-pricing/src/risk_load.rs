//! Risk-loaded prices from simulated losses: the premium, margin and
//! capital of a cover, and of each component of a portfolio.
//!
//! The losses are draws: a ceded result from
//! `prospicio_aggregate::reinsurance::Tower`, a reserve bootstrap, or any
//! [`PredictiveDistribution`]. Two inputs set the price:
//!
//! - the **assets** that back the risk, a distortion risk measure `a = ρ(X)`
//!   (for example `Distortion::tvar(0.99)`);
//! - the **premium rule** ([`PremiumRule`]): a pricing distortion
//!   `P = ρ_g(X)`, or a constant cost of capital `r` on the capital
//!   `Q = a - P`, which gives `P = (E[X] + r a) / (1 + r)`.
//!
//! The margin is `P - E[X]`, the capital `a - P`, and the return on
//! capital their ratio. On a portfolio, premium and assets are each
//! allocated by co-measure ([`PredictiveDistribution::allocate`]), the
//! natural allocation of Mildenhall and Major (*Pricing Insurance Risk*,
//! 2022): component prices add up to the portfolio price, and a component
//! that diversifies the portfolio gets a lower price than it would alone.
//! See `docs/design/pareto.md`.

use prospicio_core::{Error, Result};
use prospicio_prob::{ComponentKey, Distortion, Empirical, PredictiveDistribution};

/// How the premium is set from the losses and the assets.
#[derive(Debug, Clone, PartialEq)]
pub enum PremiumRule {
    /// The premium is the distortion risk measure of the losses. The
    /// distortion should load less than the asset measure, or the
    /// premium exceeds the assets.
    Distortion(Distortion),
    /// A constant cost of capital `r > 0`: the margin is `r` times the
    /// capital `a - P`, so `P = (E[X] + r a) / (1 + r)`.
    CostOfCapital(f64),
}

impl PremiumRule {
    /// A constant cost of capital `rate`; fails unless `rate` is positive
    /// and finite.
    pub fn cost_of_capital(rate: f64) -> Result<Self> {
        if !(rate > 0.0 && rate.is_finite()) {
            return Err(Error::InvalidParameter {
                name: "rate",
                value: rate,
                reason: "must be positive and finite",
            });
        }
        Ok(Self::CostOfCapital(rate))
    }
}

/// The price of one cover, or of one component's share of a portfolio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Price {
    /// Expected loss `E[X]`.
    pub expected_loss: f64,
    /// Premium `P`.
    pub premium: f64,
    /// Assets `a` backing the loss.
    pub assets: f64,
}

impl Price {
    /// Margin `P - E[X]`.
    pub fn margin(&self) -> f64 {
        self.premium - self.expected_loss
    }

    /// Capital `a - P`: the assets the premium does not fund.
    pub fn capital(&self) -> f64 {
        self.assets - self.premium
    }

    /// Loss ratio `E[X] / P`.
    pub fn loss_ratio(&self) -> f64 {
        self.expected_loss / self.premium
    }

    /// Return on capital, margin over capital. For a component that
    /// hedges the portfolio both can be negative.
    pub fn return_on_capital(&self) -> f64 {
        self.margin() / self.capital()
    }
}

/// Prices one cover from its loss draws.
///
/// ```
/// use prospicio_prob::{Distortion, Sampled};
/// use prospicio_pricing::risk_load::{price, PremiumRule};
///
/// // Ceded losses in four equally likely years.
/// let ceded = Sampled::new(vec![0.0, 0.0, 2.0, 6.0]).unwrap();
/// let assets = Distortion::tvar(0.5).unwrap(); // a = 4
/// let p = price(&ceded, &PremiumRule::cost_of_capital(0.25).unwrap(), &assets).unwrap();
/// assert_eq!(p.premium, (2.0 + 0.25 * 4.0) / 1.25); // 2.4
/// assert!((p.return_on_capital() - 0.25).abs() < 1e-15);
/// ```
pub fn price<E: Empirical + ?Sized>(
    losses: &E,
    rule: &PremiumRule,
    assets: &Distortion,
) -> Result<Price> {
    let p = price_unchecked(losses, rule, assets);
    check_funded(p.premium, p.assets)?;
    Ok(p)
}

/// Prices of a portfolio's components and of the portfolio as a whole.
#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioPrice {
    /// Component keys, in the order of the distribution's components.
    pub components: Vec<ComponentKey>,
    /// Each component's share of the portfolio price; these add up to
    /// [`total`](Self::total).
    pub allocated: Vec<Price>,
    /// Each component priced on its own, with the same rule and asset
    /// measure.
    pub standalone: Vec<Price>,
    /// The portfolio, priced on the total of its components.
    pub total: Price,
}

impl PortfolioPrice {
    /// Premium saved by writing the components together: the sum of the
    /// standalone premiums less the portfolio premium. Non-negative when
    /// the premium rule is subadditive, as every [`Distortion`] is.
    pub fn diversification(&self) -> f64 {
        self.standalone.iter().map(|p| p.premium).sum::<f64>() - self.total.premium
    }
}

/// Prices a portfolio and allocates the price to its components.
///
/// The components must add up to the portfolio: pass segments, or
/// contracts, not a tower result that holds gross, ceded and net side by
/// side (aggregate it first, or keep the ceded part).
///
/// ```
/// use prospicio_prob::{Distortion, KeyValue, PredictiveDistribution, Provenance};
/// use prospicio_pricing::risk_load::{price_portfolio, PremiumRule};
///
/// // Two covers over four simulations; the second pays in the
/// // portfolio's best years, so it hedges the first.
/// let pd = PredictiveDistribution::from_draws(
///     vec!["cover".into()],
///     vec![vec![KeyValue::from("a")], vec![KeyValue::from("b")]],
///     vec![0.0, 2.0, 1.0, 1.0, 4.0, 0.0, 8.0, 0.0],
///     Provenance::new("example"),
/// )
/// .unwrap();
/// let rule = PremiumRule::cost_of_capital(0.1).unwrap();
/// let p = price_portfolio(&pd, &rule, &Distortion::tvar(0.5).unwrap()).unwrap();
/// let sum: f64 = p.allocated.iter().map(|c| c.premium).sum();
/// assert!((sum - p.total.premium).abs() < 1e-12);
/// assert!(p.allocated[1].margin() < 0.0); // the hedge earns a negative margin
/// assert!(p.diversification() > 0.0);
/// ```
pub fn price_portfolio(
    pd: &PredictiveDistribution,
    rule: &PremiumRule,
    assets: &Distortion,
) -> Result<PortfolioPrice> {
    let m = pd.n_components();
    let n = pd.n_sims() as f64;
    let mut expected = vec![0.0; m];
    for row in pd.draw_matrix().chunks_exact(m) {
        for (e, x) in expected.iter_mut().zip(row) {
            *e += x;
        }
    }
    expected.iter_mut().for_each(|e| *e /= n);
    let asset_shares = pd.allocate(assets);
    let premium_shares: Vec<f64> = match rule {
        PremiumRule::Distortion(g) => pd.allocate(g),
        PremiumRule::CostOfCapital(r) => expected
            .iter()
            .zip(&asset_shares)
            .map(|(e, a)| (e + r * a) / (1.0 + r))
            .collect(),
    };
    let total = price(pd.total(), rule, assets)?;
    let allocated = expected
        .iter()
        .zip(&premium_shares)
        .zip(&asset_shares)
        .map(|((&expected_loss, &premium), &assets)| Price {
            expected_loss,
            premium,
            assets,
        })
        .collect();
    let standalone = pd
        .components()
        .iter()
        .map(|key| {
            let marginal = pd.marginal(key).expect("key comes from the distribution");
            price_unchecked(&marginal, rule, assets)
        })
        .collect();
    Ok(PortfolioPrice {
        components: pd.components().to_vec(),
        allocated,
        standalone,
        total,
    })
}

/// [`price`] without the funding check: a component on its own may be
/// priced above the assets it would need alone (a hedge priced by a
/// distortion), which is not an error for the portfolio.
fn price_unchecked<E: Empirical + ?Sized>(
    losses: &E,
    rule: &PremiumRule,
    assets: &Distortion,
) -> Price {
    let expected_loss = losses.mean();
    let a = losses.distortion(assets);
    let premium = match rule {
        PremiumRule::Distortion(g) => losses.distortion(g),
        PremiumRule::CostOfCapital(r) => (expected_loss + r * a) / (1.0 + r),
    };
    Price {
        expected_loss,
        premium,
        assets: a,
    }
}

fn check_funded(premium: f64, assets: f64) -> Result<()> {
    // Allow for rounding when the two measures coincide.
    if premium > assets + 1e-12 * assets.abs().max(1.0) {
        return Err(Error::InvalidParameter {
            name: "rule",
            value: premium,
            reason: "premium exceeds the assets; use a pricing distortion that loads less than the asset measure",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_prob::{KeyValue, Provenance, Sampled};

    fn two_covers() -> PredictiveDistribution {
        PredictiveDistribution::from_draws(
            vec!["cover".into()],
            vec![vec![KeyValue::from("a")], vec![KeyValue::from("b")]],
            vec![0.0, 2.0, 1.0, 1.0, 4.0, 0.0, 8.0, 0.0],
            Provenance::new("test"),
        )
        .unwrap()
    }

    #[test]
    fn cost_of_capital_earns_the_rate_on_capital() {
        let s = Sampled::new(vec![1.0, 3.0, 5.0, 11.0]).unwrap();
        let assets = Distortion::tvar(0.75).unwrap();
        let p = price(&s, &PremiumRule::cost_of_capital(0.15).unwrap(), &assets).unwrap();
        assert_eq!(p.expected_loss, 5.0);
        assert_eq!(p.assets, 11.0);
        assert!((p.margin() - 0.15 * p.capital()).abs() < 1e-12);
        assert!((p.loss_ratio() - 5.0 / p.premium).abs() < 1e-15);
    }

    #[test]
    fn distortion_rule_is_the_distortion_measure() {
        let s = Sampled::new(vec![1.0, 3.0, 5.0, 11.0]).unwrap();
        let wang = Distortion::wang(0.3).unwrap();
        let p = price(
            &s,
            &PremiumRule::Distortion(wang.clone()),
            &Distortion::tvar(0.9).unwrap(),
        )
        .unwrap();
        assert_eq!(p.premium, s.distortion(&wang));
        assert!(p.premium > p.expected_loss && p.premium < p.assets);
    }

    #[test]
    fn rejects_unfunded_premium_and_bad_rates() {
        let s = Sampled::new(vec![1.0, 3.0, 5.0, 11.0]).unwrap();
        let heavy = PremiumRule::Distortion(Distortion::tvar(0.9).unwrap());
        assert!(price(&s, &heavy, &Distortion::tvar(0.5).unwrap()).is_err());
        assert!(PremiumRule::cost_of_capital(0.0).is_err());
        assert!(PremiumRule::cost_of_capital(f64::NAN).is_err());
    }

    #[test]
    fn allocation_adds_up_under_both_rules() {
        let pd = two_covers();
        let assets = Distortion::tvar(0.5).unwrap();
        for rule in [
            PremiumRule::cost_of_capital(0.1).unwrap(),
            PremiumRule::Distortion(Distortion::proportional_hazard(0.7).unwrap()),
        ] {
            let p = price_portfolio(&pd, &rule, &assets).unwrap();
            let sum = |f: fn(&Price) -> f64| p.allocated.iter().map(f).sum::<f64>();
            assert!((sum(|c| c.expected_loss) - p.total.expected_loss).abs() < 1e-12);
            assert!((sum(|c| c.premium) - p.total.premium).abs() < 1e-12);
            assert!((sum(|c| c.assets) - p.total.assets).abs() < 1e-12);
            assert!(p.diversification() >= -1e-12);
        }
    }

    #[test]
    fn allocated_assets_are_the_co_tvars() {
        // Totals 2, 2, 4, 8: TVaR at 50% is the mean of the years with
        // totals 4 and 8, where cover a pays 4 and 8 and cover b nothing.
        let p = price_portfolio(
            &two_covers(),
            &PremiumRule::cost_of_capital(0.1).unwrap(),
            &Distortion::tvar(0.5).unwrap(),
        )
        .unwrap();
        assert_eq!(p.allocated[0].assets, 6.0);
        assert_eq!(p.allocated[1].assets, 0.0);
        assert_eq!(p.total.assets, 6.0);
        assert_eq!(p.standalone[1].assets, 1.5);
        assert_eq!(p.components[1], vec![KeyValue::from("b")]);
    }
}
