//! Risk profiles for property per-risk business: bands of sum insured,
//! each with an expected loss (given, or premium times a loss ratio) and
//! its own exposure curve. A profile gives exposure-rated expected losses
//! for surplus treaties and per-risk layers, and simulates events that
//! carry each risk's sum insured, so a surplus treaty and the per-risk
//! excess of loss it inures to can be applied by `prospicio_aggregate::Tower`
//! (`docs/design/aggregate.md`, "Risk profiles").
//!
//! In band `b` with sum insured `SI_b` (the band's representative risk:
//! its total sum insured over its number of risks, say), expected loss
//! `EL_b` and curve `G_b` with mean destruction rate `m_b`, one loss is on
//! average `SI_b m_b`, so the band expects `λ_b = EL_b / (SI_b m_b)`
//! losses a year. Each simulated year draws a Poisson number of losses
//! with mean `Σ λ_b`; each loss falls in band `b` with probability
//! `λ_b / Σ λ`, and is `SI_b` times a destruction rate drawn from `G_b`.
//!
//! A band may instead spread its sums insured between bounds `[L, U]`
//! ([`Band::with_bounds`]): its risks' sums insured are uniform on the
//! bounds. Every risk then has the same claim frequency (a loss is its SI
//! times a destruction rate from the band's curve, so a bigger risk has
//! bigger losses, not more of them), so each loss draws its SI uniformly
//! from `[L, U]` and the band expects `EL / (((L + U)/2) m_b)` losses. The
//! exposure-rated expectations average over the band, each SI weighted by
//! its share of the band's loss (`∝ SI`).
//!
//! A uniform spread ignores the band's given SI, whose mean `(L + U)/2`
//! may not match the band's total sum insured over its risks. A tilted
//! spread ([`Band::with_tilted_bounds`]) keeps both: the density
//! `∝ exp(θ s)` on `[L, U]` (the most even spread with a given mean, by
//! entropy) with `θ` solved so its mean is the band's SI.

use prospicio_aggregate::EventSet;
use prospicio_core::{Error, Result, StreamRng};
use prospicio_math::integrate::gauss_legendre;
use prospicio_prob::{Counting, Poisson};

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
    /// Bounds `[lower, upper]` between which the band's sums insured are
    /// spread; `None` for one representative risk at `sum_insured`.
    pub bounds: Option<(f64, f64)>,
    /// The spread's tilt `θ`: its density over the bounds is
    /// `∝ exp(θ s)`, so 0 is uniform.
    pub tilt: f64,
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
            bounds: None,
            tilt: 0.0,
        })
    }

    /// The same band with its sums insured spread uniformly between
    /// `lower` and `upper` (`0 < lower ≤ upper`, finite) instead of one
    /// representative risk. The band's mean sum insured becomes
    /// `(lower + upper)/2`; its expected loss is unchanged.
    pub fn with_bounds(mut self, lower: f64, upper: f64) -> Result<Self> {
        if !(lower.is_finite() && lower > 0.0) {
            return Err(invalid("lower", lower, "must be positive and finite"));
        }
        if !(upper.is_finite() && upper >= lower) {
            return Err(invalid("upper", upper, "must be finite and at least lower"));
        }
        self.bounds = Some((lower, upper));
        self.tilt = 0.0;
        Ok(self)
    }

    /// The same band with its sums insured spread between `lower` and
    /// `upper` with mean `sum_insured`, so the spread matches both the
    /// bounds and the band's total sum insured (`risks × sum_insured`):
    /// the density `∝ exp(θ s)` on the bounds, with `θ` solved for that
    /// mean (`θ = 0`, uniform, when `sum_insured` is the midpoint).
    /// `sum_insured` must lie strictly between the bounds.
    ///
    /// ```
    /// use prospicio_pricing::exposure::Mbbefd;
    /// use prospicio_pricing::profile::Band;
    ///
    /// // 400 risks from 1m to 5m whose sums insured total 800m: mean 2m,
    /// // below the midpoint, so the spread leans to small risks.
    /// let b = Band::from_expected_loss(2e6, 400.0, 1.2e6, Mbbefd::swiss_re(3.0).unwrap())
    ///     .unwrap()
    ///     .with_tilted_bounds(1e6, 5e6)
    ///     .unwrap();
    /// assert!(b.tilt < 0.0);
    /// assert!((b.mean_sum_insured() - 2e6).abs() < 1e-3);
    /// ```
    pub fn with_tilted_bounds(self, lower: f64, upper: f64) -> Result<Self> {
        let mut b = self.with_bounds(lower, upper)?;
        let si = b.sum_insured;
        if !(lower < si && si < upper) {
            return Err(invalid(
                "sum_insured",
                si,
                "must lie strictly between the bounds for a tilted spread",
            ));
        }
        let w = upper - lower;
        b.tilt = solve_tilt((si - lower) / w) / w;
        Ok(b)
    }

    /// The band's mean sum insured over its risks: with bounds, the
    /// spread's mean (`(lower + upper)/2` uniform, `sum_insured` tilted);
    /// otherwise `sum_insured`.
    pub fn mean_sum_insured(&self) -> f64 {
        self.bounds.map_or(self.sum_insured, |(l, u)| {
            l + (u - l) * tilt_mean((u - l) * self.tilt)
        })
    }

    /// A sum insured from the band's spread at probability `p` (inverse
    /// transform); `sum_insured` without bounds.
    fn sum_insured_quantile(&self, p: f64) -> f64 {
        let Some((l, u)) = self.bounds else {
            return self.sum_insured;
        };
        let t = (u - l) * self.tilt;
        let x = if t.abs() < 1e-12 {
            p
        } else if t > 0.0 {
            // From the top, so exp never overflows.
            1.0 + ((1.0 - p) * (-t).exp_m1()).ln_1p() / t
        } else {
            (p * t.exp_m1()).ln_1p() / t
        };
        (l + (u - l) * x.clamp(0.0, 1.0)).clamp(l, u)
    }

    /// The loss-weighted average of `f(SI)` over the band: `f(sum_insured)`
    /// for one risk, `E[S f(S)] / E[S]` over the spread `S` otherwise (a
    /// risk's expected loss is proportional to its SI). The expectations
    /// are integrals over probability, `∫₀¹ g(Q(p)) dp` with `Q` the
    /// spread's quantile, so a steep tilt needs no finer grid.
    fn loss_weighted(&self, f: impl Fn(f64) -> Result<f64>) -> Result<f64> {
        let Some((l, u)) = self.bounds else {
            return f(self.sum_insured);
        };
        if u - l <= 1e-12 * u {
            return f(l);
        }
        let pieces = 256;
        let step = 1.0 / pieces as f64;
        let mut num = 0.0;
        let mut den = 0.0;
        for k in 0..pieces {
            let a = step * k as f64;
            let b = if k + 1 == pieces { 1.0 } else { a + step };
            num += gauss_legendre(
                |p| {
                    let s = self.sum_insured_quantile(p);
                    f(s).map(|v| s * v)
                },
                a,
                b,
            )?;
            den += gauss_legendre(|p| Ok(self.sum_insured_quantile(p)), a, b)?;
        }
        Ok(num / den)
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

    /// Expected number of losses a year, `EL / (mean SI × mean rate)`.
    pub fn expected_claims(&self) -> f64 {
        self.expected_loss / (self.mean_sum_insured() * self.curve.mean_rate())
    }
}

/// Bands of sum insured with their expected losses and exposure curves.
///
/// ```
/// use prospicio_pricing::exposure::Mbbefd;
/// use prospicio_pricing::profile::{Band, RiskProfile};
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
            let per_si = |si: f64| -> Result<f64> {
                let keep = 1.0 - surplus.map_or(0.0, |(r, k)| cession(si, r, k));
                if keep <= 0.0 {
                    return Ok(0.0);
                }
                Ok(keep * b.curve.layer_share(limit, attachment, keep * si)?)
            };
            total += b.expected_loss * b.loss_weighted(per_si)?;
        }
        Ok(total)
    }

    /// Expected annual loss ceded to a surplus treaty with retention line
    /// `retention` and `lines` lines, `Σ cession(SI_b) × EL_b` (averaged
    /// over a band's bounds when it has them).
    pub fn expected_surplus_loss(&self, retention: f64, lines: f64) -> f64 {
        self.bands
            .iter()
            .map(|b| {
                let c = b
                    .loss_weighted(|si| Ok(cession(si, retention, lines)))
                    .expect("the cession never fails");
                c * b.expected_loss
            })
            .sum()
    }

    /// `n_sims` years of losses, each with its risk's sum insured.
    ///
    /// Year `i` uses stream `i` of `seed`: first the loss count (Poisson
    /// with mean [`expected_claims`](Self::expected_claims)), then for each
    /// loss a band, its sum insured (only for a band with bounds) and a
    /// destruction rate, all by inverse transform.
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
                let si = match band.bounds {
                    Some(_) => band.sum_insured_quantile(rng.next_open01()),
                    None => band.sum_insured,
                };
                let rate = band.curve.rate_quantile(rng.next_open01());
                year.push(si * rate);
                sums_insured.push(si);
            }
            years.push(year);
        }
        EventSet::from_years(years, seed)?.with_sums_insured(sums_insured)
    }
}

/// The mean of the density `∝ exp(t x)` on `[0, 1]`:
/// `1 / (1 − e^{−t}) − 1/t`, increasing from 0 to 1, with `1/2` at 0.
fn tilt_mean(t: f64) -> f64 {
    if t.abs() < 1e-4 {
        return 0.5 + t / 12.0 - t.powi(3) / 720.0;
    }
    if t < 0.0 {
        return 1.0 - tilt_mean(-t);
    }
    -1.0 / (-t).exp_m1() - 1.0 / t
}

/// The `t` whose [`tilt_mean`] is `p`, in `(0, 1)`, by bisection.
fn solve_tilt(p: f64) -> f64 {
    // tilt_mean(t) ≈ 1 − 1/t for large t, so ±2/min(p, 1 − p) brackets it.
    let reach = 2.0 / p.min(1.0 - p);
    let (mut lo, mut hi) = (-reach, reach);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if tilt_mean(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
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
    use prospicio_aggregate::{Layer, Tower};
    use prospicio_prob::{Distribution, KeyValue};

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

    fn ceded_mean(pd: &prospicio_prob::PredictiveDistribution, layer: &str) -> (f64, f64) {
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
    fn spread_sums_insured_match_exposure_rating() {
        let c = Mbbefd::swiss_re(3.0).unwrap();
        let spread = RiskProfile::new(vec![
            Band::from_expected_loss(3e6, 400.0, 1.2e6, c)
                .unwrap()
                .with_bounds(1e6, 5e6)
                .unwrap(),
        ])
        .unwrap();
        let point = RiskProfile::new(vec![
            Band::from_expected_loss(3e6, 400.0, 1.2e6, c).unwrap(),
        ])
        .unwrap();
        // Same mean SI, so the same claim count.
        assert!((spread.expected_claims() - point.expected_claims()).abs() < 1e-9);
        // Surplus cession by quadrature against the closed form:
        // ∫ s c(s) ds = ∫ clamp(s − R, 0, kR) ds over [1m, 5m], R = 2m, k = 4.
        let (r, k, l, u) = (2e6, 4.0, 1e6, 5e6);
        let closed = 0.5 * (u - r) * (u - r) / (0.5 * (u * u - l * l));
        let want = closed * 1.2e6;
        assert!((spread.expected_surplus_loss(r, k) / want - 1.0).abs() < 1e-9);
        // The representative risk cedes 1/3; the spread band cedes 3/8,
        // because its large risks carry more of the loss.
        assert!((closed - 0.375).abs() < 1e-12);
        assert!((point.expected_surplus_loss(r, k) / 1.2e6 - 1.0 / 3.0).abs() < 1e-12);

        let events = spread.simulate(80_000, 5).unwrap();
        let tower = Tower::inuring(vec![
            vec![Layer::surplus("surplus", r, k).unwrap()],
            vec![Layer::xol("xl", 0.5e6, 0.5e6).unwrap()],
        ])
        .unwrap();
        let pd = tower.apply(&events).unwrap();
        let (s, s_se) = ceded_mean(&pd, "surplus");
        let want = spread.expected_surplus_loss(r, k);
        assert!((s - want).abs() < 4.0 * s_se, "surplus {s} vs {want}");
        let (x, x_se) = ceded_mean(&pd, "xl");
        let want = spread
            .expected_layer_loss(0.5e6, 0.5e6, Some((r, k)))
            .unwrap();
        assert!((x - want).abs() < 4.0 * x_se, "xl {x} vs {want}");
        let si = events.sums_insured(0).unwrap_or(&[]);
        assert!(si.iter().all(|&v| (l..=u).contains(&v)));
        assert!(
            Band::from_expected_loss(1.0, 1.0, 1.0, c)
                .unwrap()
                .with_bounds(2.0, 1.0)
                .is_err()
        );
        assert!(
            Band::from_expected_loss(1.0, 1.0, 1.0, c)
                .unwrap()
                .with_bounds(0.0, 1.0)
                .is_err()
        );
    }

    #[test]
    fn tilt_solves_for_the_mean() {
        for p in [1e-3, 0.01, 0.2, 0.4999, 0.5, 0.7, 0.99, 0.999] {
            let t = solve_tilt(p);
            assert!((tilt_mean(t) - p).abs() < 1e-12, "{p}: {t}");
        }
        assert!(solve_tilt(0.5).abs() < 1e-12);
        // The series and the closed form agree where they meet.
        let (a, b) = (tilt_mean(0.99e-4), tilt_mean(1.01e-4));
        assert!(b > a && b - a < 1e-6);
    }

    #[test]
    fn tilted_spread_matches_bounds_and_total_sum_insured() {
        let c = Mbbefd::swiss_re(3.0).unwrap();
        let (l, u, si) = (1e6, 5e6, 2e6);
        let band = Band::from_expected_loss(si, 400.0, 1.2e6, c)
            .unwrap()
            .with_tilted_bounds(l, u)
            .unwrap();
        assert!(band.tilt < 0.0);
        assert!((band.mean_sum_insured() - si).abs() < 1e-6);
        // The same claim count as one representative risk at the given SI.
        let point = Band::from_expected_loss(si, 400.0, 1.2e6, c).unwrap();
        assert!((band.expected_claims() / point.expected_claims() - 1.0).abs() < 1e-12);
        // At the midpoint the tilt is 0: the uniform spread.
        let mid = Band::from_expected_loss(3e6, 400.0, 1.2e6, c).unwrap();
        let tilted = mid.clone().with_tilted_bounds(l, u).unwrap();
        assert!(tilted.tilt.abs() < 1e-18);
        let uniform = RiskProfile::new(vec![mid.with_bounds(l, u).unwrap()]).unwrap();
        let tilted = RiskProfile::new(vec![tilted]).unwrap();
        assert!(
            (tilted.expected_surplus_loss(2e6, 4.0) / uniform.expected_surplus_loss(2e6, 4.0)
                - 1.0)
                .abs()
                < 1e-9
        );

        // Surplus cession against Simpson's rule in s over the density
        // exp(θ s), independent of the quadrature over probability.
        let (r, k) = (2e6, 4.0);
        let n = 200_000;
        let h = (u - l) / n as f64;
        let (mut num, mut den) = (0.0, 0.0);
        for j in 0..=n {
            let s = l + h * j as f64;
            let w = if j == 0 || j == n {
                1.0
            } else if j % 2 == 1 {
                4.0
            } else {
                2.0
            };
            let d = (band.tilt * (s - l)).exp();
            num += w * s * d * cession(s, r, k);
            den += w * s * d;
        }
        let want = num / den * 1.2e6;
        let profile = RiskProfile::new(vec![band.clone()]).unwrap();
        let got = profile.expected_surplus_loss(r, k);
        // The cession's kink at the retention falls inside a quadrature
        // piece over probability, which costs about 3e-8.
        assert!((got / want - 1.0).abs() < 1e-7, "{got} vs {want}");
        // Leaning to small risks, it cedes less than the uniform band's 3/8.
        assert!(got / 1.2e6 < 0.375);

        // Simulated sums insured follow the spread: mean and KS.
        let events = profile.simulate(80_000, 9).unwrap();
        let mut drawn: Vec<f64> = (0..events.n_sims())
            .flat_map(|i| events.sums_insured(i).unwrap().to_vec())
            .collect();
        let m = drawn.len() as f64;
        let mean = drawn.iter().sum::<f64>() / m;
        let sd = (drawn.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / m).sqrt();
        assert!((mean - si).abs() < 4.0 * sd / m.sqrt(), "{mean}");
        drawn.sort_by(f64::total_cmp);
        let t = band.tilt;
        let cdf = |x: f64| (t * (x - l)).exp_m1() / (t * (u - l)).exp_m1();
        let ks = drawn
            .iter()
            .enumerate()
            .map(|(j, &x)| {
                let f = cdf(x);
                (f - j as f64 / m).abs().max(((j + 1) as f64 / m - f).abs())
            })
            .fold(0.0, f64::max);
        assert!(ks < 1.95 / m.sqrt(), "KS {ks}");

        // A surplus inuring to a per-risk XL, against exposure rating.
        let tower = Tower::inuring(vec![
            vec![Layer::surplus("surplus", r, k).unwrap()],
            vec![Layer::xol("xl", 0.5e6, 0.5e6).unwrap()],
        ])
        .unwrap();
        let pd = tower.apply(&events).unwrap();
        let (s, s_se) = ceded_mean(&pd, "surplus");
        assert!((s - got).abs() < 4.0 * s_se, "surplus {s} vs {got}");
        let (x, x_se) = ceded_mean(&pd, "xl");
        let want = profile
            .expected_layer_loss(0.5e6, 0.5e6, Some((r, k)))
            .unwrap();
        assert!((x - want).abs() < 4.0 * x_se, "xl {x} vs {want}");

        // A steep tilt: nearly every risk at the bottom of the band.
        let steep = Band::from_expected_loss(1.001e6, 1.0, 1.0, c)
            .unwrap()
            .with_tilted_bounds(l, u)
            .unwrap();
        assert!((steep.mean_sum_insured() / 1.001e6 - 1.0).abs() < 1e-9);
        let steep = RiskProfile::new(vec![steep]).unwrap();
        let at_bottom = RiskProfile::new(vec![
            Band::from_expected_loss(1.001e6, 1.0, 1.0, c).unwrap(),
        ])
        .unwrap();
        let (a, b) = (
            steep.expected_layer_loss(0.5e6, 0.5e6, None).unwrap(),
            at_bottom.expected_layer_loss(0.5e6, 0.5e6, None).unwrap(),
        );
        assert!((a / b - 1.0).abs() < 1e-3, "{a} vs {b}");

        for bad in [l, u, 0.5e6, 6e6] {
            assert!(
                Band::from_expected_loss(bad, 1.0, 1.0, c)
                    .unwrap()
                    .with_tilted_bounds(l, u)
                    .is_err()
            );
        }
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
