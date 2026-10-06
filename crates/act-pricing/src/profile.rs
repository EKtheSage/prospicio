//! Risk profiles for property per-risk business: bands of sum insured,
//! each with an expected loss (given, or premium times a loss ratio) and
//! its own exposure curve. A profile gives exposure-rated expected losses
//! for surplus treaties and per-risk layers, and simulates events that
//! carry each risk's sum insured, so a surplus treaty and the per-risk
//! excess of loss it inures to can be applied by `act_aggregate::Tower`
//! (`docs/design/aggregate.md`, "Risk profiles").
//!
//! In band `b` with sum insured `SI_b` (the band's representative risk:
//! its total sum insured over its number of risks, say), expected loss
//! `EL_b` and curve `G_b` with mean destruction rate `m_b`, one loss is on
//! average `SI_b m_b`, so the band expects `λ_b = EL_b / (SI_b m_b)`
//! losses a year. Each simulated year draws a Poisson number of losses
//! with mean `Σ λ_b`; each loss falls in band `b` with probability
//! `λ_b / Σ λ`, and is `SI_b` times a destruction rate drawn from `G_b`.

use act_aggregate::EventSet;
use act_core::{Error, Result, StreamRng};
use act_prob::{Counting, Poisson};

use crate::exposure::{ExposureCurve, Mbbefd, TabulatedCurve};

/// An exposure curve a profile band can own.
#[derive(Debug, Clone, PartialEq)]
pub enum BandCurve {
    Mbbefd(Mbbefd),
    Tabulated(TabulatedCurve),
}

impl ExposureCurve for BandCurve {
    fn g(&self, x: f64) -> f64 {
        match self {
            Self::Mbbefd(c) => c.g(x),
            Self::Tabulated(c) => c.g(x),
        }
    }

    fn rate_quantile(&self, u: f64) -> f64 {
        match self {
            Self::Mbbefd(c) => c.rate_quantile(u),
            Self::Tabulated(c) => c.rate_quantile(u),
        }
    }

    fn mean_rate(&self) -> f64 {
        match self {
            Self::Mbbefd(c) => c.mean_rate(),
            Self::Tabulated(c) => c.mean_rate(),
        }
    }
}

impl From<Mbbefd> for BandCurve {
    fn from(c: Mbbefd) -> Self {
        Self::Mbbefd(c)
    }
}

impl From<TabulatedCurve> for BandCurve {
    fn from(c: TabulatedCurve) -> Self {
        Self::Tabulated(c)
    }
}

/// One band of a risk profile.
#[derive(Debug, Clone, PartialEq)]
pub struct Band {
    /// Sum insured of the band's representative risk, taken as its MPL.
    pub sum_insured: f64,
    /// Number of risks in the band (for reference; the loss count comes
    /// from the expected loss).
    pub risks: f64,
    /// Expected annual loss of the whole band.
    pub expected_loss: f64,
    /// The band's exposure curve.
    pub curve: BandCurve,
}

impl Band {
    /// A band with its expected annual loss.
    pub fn from_expected_loss(
        sum_insured: f64,
        risks: f64,
        expected_loss: f64,
        curve: impl Into<BandCurve>,
    ) -> Result<Self> {
        if !(sum_insured.is_finite() && sum_insured > 0.0) {
            return Err(invalid(
                "sum_insured",
                sum_insured,
                "must be positive and finite",
            ));
        }
        if !(risks.is_finite() && risks > 0.0) {
            return Err(invalid("risks", risks, "must be positive and finite"));
        }
        if !(expected_loss.is_finite() && expected_loss >= 0.0) {
            return Err(invalid(
                "expected_loss",
                expected_loss,
                "must be non-negative and finite",
            ));
        }
        Ok(Self {
            sum_insured,
            risks,
            expected_loss,
            curve: curve.into(),
        })
    }

    /// A band with its premium and an expected loss ratio: expected loss
    /// `premium × loss_ratio`.
    pub fn from_premium(
        sum_insured: f64,
        risks: f64,
        premium: f64,
        loss_ratio: f64,
        curve: impl Into<BandCurve>,
    ) -> Result<Self> {
        if !(loss_ratio.is_finite() && loss_ratio >= 0.0) {
            return Err(invalid(
                "loss_ratio",
                loss_ratio,
                "must be non-negative and finite",
            ));
        }
        if !(premium.is_finite() && premium >= 0.0) {
            return Err(invalid(
                "premium",
                premium,
                "must be non-negative and finite",
            ));
        }
        Self::from_expected_loss(sum_insured, risks, premium * loss_ratio, curve)
    }

    /// Expected number of losses a year, `EL / (SI × mean rate)`.
    pub fn expected_claims(&self) -> f64 {
        self.expected_loss / (self.sum_insured * self.curve.mean_rate())
    }
}

/// Bands of sum insured with their expected losses and exposure curves.
///
/// ```
/// use act_pricing::exposure::Mbbefd;
/// use act_pricing::profile::{Band, RiskProfile};
///
/// let c3 = Mbbefd::swiss_re(3.0).unwrap();
/// let profile = RiskProfile::new(vec![
///     Band::from_premium(1e6, 800.0, 2.0e6, 0.6, c3).unwrap(),
///     Band::from_expected_loss(10e6, 50.0, 0.8e6, Mbbefd::swiss_re(4.0).unwrap()).unwrap(),
/// ])
/// .unwrap();
/// // The whole expected loss sits in a layer from 0 to the top band's SI.
/// let all = profile.expected_layer_loss(f64::INFINITY, 0.0, None).unwrap();
/// assert!((all - 2.0e6).abs() < 1e-6);
/// // Simulated years carry each loss's sum insured.
/// let events = profile.simulate(1_000, 7).unwrap();
/// assert!(events.has_sums_insured());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RiskProfile {
    bands: Vec<Band>,
}

impl RiskProfile {
    /// A profile of at least one band.
    pub fn new(bands: Vec<Band>) -> Result<Self> {
        if bands.is_empty() {
            return Err(Error::Data("a risk profile needs at least one band".into()));
        }
        Ok(Self { bands })
    }

    /// The bands.
    pub fn bands(&self) -> &[Band] {
        &self.bands
    }

    /// Expected losses a year, all bands.
    pub fn expected_claims(&self) -> f64 {
        self.bands.iter().map(Band::expected_claims).sum()
    }

    /// Expected annual loss, all bands.
    pub fn expected_loss(&self) -> f64 {
        self.bands.iter().map(|b| b.expected_loss).sum()
    }

    /// Exposure-rated expected loss to a per-risk layer `limit` xs
    /// `attachment`, `Σ EL_b × layer share_b`. With
    /// `surplus = Some((retention, lines))`, the layer sees each risk net
    /// of that surplus treaty: a risk ceding `c` keeps `(1 − c)` of each
    /// loss, so its retained curve is the same at sum insured `(1 − c) SI`.
    /// `limit = inf` and `attachment = 0` give the whole (retained) loss.
    pub fn expected_layer_loss(
        &self,
        limit: f64,
        attachment: f64,
        surplus: Option<(f64, f64)>,
    ) -> Result<f64> {
        let mut total = 0.0;
        for b in &self.bands {
            let keep = 1.0 - surplus.map_or(0.0, |(r, k)| cession(b.sum_insured, r, k));
            if keep <= 0.0 {
                continue;
            }
            let share = b
                .curve
                .layer_share(limit, attachment, keep * b.sum_insured)?;
            total += b.expected_loss * keep * share;
        }
        Ok(total)
    }

    /// Expected annual loss ceded to a surplus treaty with retention line
    /// `retention` and `lines` lines, `Σ cession(SI_b) × EL_b`.
    pub fn expected_surplus_loss(&self, retention: f64, lines: f64) -> f64 {
        self.bands
            .iter()
            .map(|b| cession(b.sum_insured, retention, lines) * b.expected_loss)
            .sum()
    }

    /// `n_sims` years of losses, each with its risk's sum insured.
    ///
    /// Year `i` uses stream `i` of `seed`: first the loss count (Poisson
    /// with mean [`expected_claims`](Self::expected_claims)), then for each
    /// loss a band and a destruction rate, all by inverse transform.
    pub fn simulate(&self, n_sims: usize, seed: u64) -> Result<EventSet> {
        if n_sims == 0 {
            return Err(invalid("n_sims", 0.0, "must be positive"));
        }
        let lambdas: Vec<f64> = self.bands.iter().map(Band::expected_claims).collect();
        let total: f64 = lambdas.iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return Err(Error::Data(
                "the profile expects no losses: every band's expected loss is 0".into(),
            ));
        }
        let mut cumulative = Vec::with_capacity(lambdas.len());
        let mut acc = 0.0;
        for l in &lambdas {
            acc += l / total;
            cumulative.push(acc);
        }
        let count = Poisson::new(total)?;
        let mut years = Vec::with_capacity(n_sims);
        let mut sums_insured = Vec::new();
        for i in 0..n_sims as u64 {
            let mut rng = StreamRng::new(seed, i);
            let n = count
                .quantile(rng.next_open01())
                .expect("next_open01 is always in (0, 1)");
            let mut year = Vec::with_capacity(n as usize);
            for _ in 0..n {
                let u = rng.next_open01();
                let b = cumulative
                    .partition_point(|&c| c < u)
                    .min(self.bands.len() - 1);
                let band = &self.bands[b];
                let rate = band.curve.rate_quantile(rng.next_open01());
                year.push(band.sum_insured * rate);
                sums_insured.push(band.sum_insured);
            }
            years.push(year);
        }
        EventSet::from_years(years, seed)?.with_sums_insured(sums_insured)
    }
}

/// A surplus treaty's cession on a risk with sum insured `si`.
fn cession(si: f64, retention: f64, lines: f64) -> f64 {
    (si - retention).clamp(0.0, lines * retention) / si
}

fn invalid(name: &'static str, value: f64, reason: &'static str) -> Error {
    Error::InvalidParameter {
        name,
        value,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use act_aggregate::{Layer, Tower};
    use act_prob::{Distribution, KeyValue};

    fn profile() -> RiskProfile {
        RiskProfile::new(vec![
            Band::from_premium(0.5e6, 2000.0, 1.0e6, 0.6, Mbbefd::swiss_re(2.0).unwrap()).unwrap(),
            Band::from_expected_loss(3e6, 300.0, 0.5e6, Mbbefd::swiss_re(3.0).unwrap()).unwrap(),
            Band::from_expected_loss(
                20e6,
                20.0,
                0.4e6,
                TabulatedCurve::new(&[0.0, 0.02, 0.2, 1.0], &[0.0, 0.3, 0.8, 1.0]).unwrap(),
            )
            .unwrap(),
        ])
        .unwrap()
    }

    fn ceded_mean(pd: &act_prob::PredictiveDistribution, layer: &str) -> (f64, f64) {
        let m = pd
            .marginal(&vec![KeyValue::from("ceded"), KeyValue::from(layer)])
            .unwrap();
        (m.mean(), m.std_dev() / (pd.n_sims() as f64).sqrt())
    }

    #[test]
    fn simulation_matches_exposure_rating() {
        let p = profile();
        assert!((p.expected_loss() - 1.5e6).abs() < 1e-6);
        let events = p.simulate(100_000, 11).unwrap();
        let gross = events.totals().unwrap().mean();
        let se = events.totals().unwrap().std_dev() / (100_000f64).sqrt();
        assert!((gross - p.expected_loss()).abs() < 4.0 * se, "{gross}");

        // A surplus (retention 1m, 5 lines) inuring to a 1m xs 0.5m per-risk XL.
        let tower = Tower::inuring(vec![
            vec![Layer::surplus("surplus", 1e6, 5.0).unwrap()],
            vec![Layer::xol("xl", 1e6, 0.5e6).unwrap()],
        ])
        .unwrap();
        let pd = tower.apply(&events).unwrap();
        let (s, s_se) = ceded_mean(&pd, "surplus");
        let want = p.expected_surplus_loss(1e6, 5.0);
        assert!((s - want).abs() < 4.0 * s_se, "surplus {s} vs {want}");
        let (x, x_se) = ceded_mean(&pd, "xl");
        let want = p.expected_layer_loss(1e6, 0.5e6, Some((1e6, 5.0))).unwrap();
        assert!((x - want).abs() < 4.0 * x_se, "xl {x} vs {want}");
    }

    #[test]
    fn checks_and_replays() {
        let p = profile();
        assert_eq!(p.simulate(50, 3).unwrap(), p.simulate(50, 3).unwrap());
        assert!(p.simulate(0, 3).is_err());
        assert!(RiskProfile::new(vec![]).is_err());
        let c = Mbbefd::swiss_re(2.0).unwrap();
        assert!(Band::from_expected_loss(0.0, 1.0, 1.0, c).is_err());
        assert!(Band::from_premium(1.0, 1.0, 1.0, -0.1, c).is_err());
        let zero =
            RiskProfile::new(vec![Band::from_expected_loss(1.0, 1.0, 0.0, c).unwrap()]).unwrap();
        assert!(zero.simulate(10, 1).is_err());
        // Without a surplus, the unlimited layer from 0 is the whole loss.
        let all = p.expected_layer_loss(f64::INFINITY, 0.0, None).unwrap();
        assert!((all - p.expected_loss()).abs() < 1e-6);
    }
}
