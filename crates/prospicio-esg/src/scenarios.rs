//! Simulated paths of the short rate, its integral and the curve.

use prospicio_core::{Error, Result, StreamRng};
use prospicio_prob::{ComponentKey, PredictiveDistribution, Provenance};

use crate::sampling::{noncentral_chi_square, normal};
use crate::short_rate::{HullWhite, ShortRate, loading};

/// The time step of the simulation. The transitions are exact, so the
/// step only sets how often the paths are reported (decision 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Step {
    /// Twelve steps a year (the default).
    #[default]
    Monthly,
    /// Four steps a year.
    Quarterly,
    /// One step a year.
    Annual,
}

impl Step {
    /// Steps in a year.
    pub fn per_year(self) -> usize {
        match self {
            Self::Monthly => 12,
            Self::Quarterly => 4,
            Self::Annual => 1,
        }
    }
}

/// An economic scenario generator: a short-rate model, the step and the
/// maturities of the curve reported at each step.
///
/// ```
/// use prospicio_esg::{Esg, Step, Vasicek};
///
/// let model = Vasicek::new(0.3, 0.04, 0.01, 0.02)?;
/// let esg = Esg::new(model).with_step(Step::Quarterly);
/// let paths = esg.simulate(5, 2_000, 7)?;
/// assert_eq!(paths.n_steps(), 20);
/// // The deflators' mean is the model's bond price today, within
/// // simulation error.
/// let mean = (0..2_000).map(|s| paths.deflator(s, 20).unwrap()).sum::<f64>() / 2_000.0;
/// assert!((mean - model.bond_price(0.02, 5.0)).abs() < 2e-3);
/// # Ok::<(), prospicio_core::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Esg {
    /// The short-rate model.
    pub rate: ShortRate,
    /// The time step.
    pub step: Step,
    /// The maturities in years of the curve reported at each step.
    pub maturities: Vec<f64>,
}

impl Esg {
    /// A generator on `rate`, monthly, reporting the curve at 1, 2, 3, 5,
    /// 7, 10, 20 and 30 years.
    pub fn new(rate: impl Into<ShortRate>) -> Self {
        Self {
            rate: rate.into(),
            step: Step::Monthly,
            maturities: vec![1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0],
        }
    }

    /// With this step.
    pub fn with_step(mut self, step: Step) -> Self {
        self.step = step;
        self
    }

    /// With the curve reported at these maturities (years).
    pub fn with_maturities(mut self, maturities: Vec<f64>) -> Self {
        self.maturities = maturities;
        self
    }

    /// `sims` paths over `years` years, path `i` from
    /// `StreamRng::new(seed, i)`, so any path can be replayed alone.
    ///
    /// Vasicek and Hull-White draw the factor and its integral over each
    /// step jointly from their exact Gaussian transition, so the
    /// deflators are exact; CIR draws the rate from its exact noncentral
    /// chi-square transition and integrates it by the trapezoid rule over
    /// the step, a bias of order `Δ²` in the deflators.
    ///
    /// Fails unless `years` and `sims` are positive and the maturities
    /// finite, positive and increasing.
    pub fn simulate(&self, years: usize, sims: usize, seed: u64) -> Result<Scenarios> {
        if years == 0 || sims == 0 {
            return Err(Error::Data(format!(
                "a simulation needs years and paths, got {years} and {sims}"
            )));
        }
        let mut previous = 0.0;
        for &m in &self.maturities {
            if !(m.is_finite() && m > previous) {
                return Err(Error::InvalidParameter {
                    name: "maturity",
                    value: m,
                    reason: "must be finite, positive and increasing",
                });
            }
            previous = m;
        }
        let n_steps = years * self.step.per_year();
        let dt = 1.0 / self.step.per_year() as f64;
        let points = n_steps + 1;
        let n_mat = self.maturities.len();
        let mut short = Vec::with_capacity(sims * points);
        let mut integral = Vec::with_capacity(sims * points);
        let mut prices = Vec::with_capacity(sims * points * n_mat);
        for sim in 0..sims {
            let mut rng = StreamRng::new(seed, sim as u64);
            self.path(
                &mut rng,
                n_steps,
                dt,
                &mut short,
                &mut integral,
                &mut prices,
            );
        }
        Ok(Scenarios {
            n_sims: sims,
            n_steps,
            step: self.step,
            maturities: self.maturities.clone(),
            short,
            integral,
            prices,
            seed,
        })
    }

    /// One path: the short rate, `∫ r` and the bond prices at each point.
    fn path(
        &self,
        rng: &mut StreamRng,
        n_steps: usize,
        dt: f64,
        short: &mut Vec<f64>,
        integral: &mut Vec<f64>,
        prices: &mut Vec<f64>,
    ) {
        let mats = &self.maturities;
        match &self.rate {
            ShortRate::Vasicek(v) => {
                let (mut x, mut y) = (v.r0 - v.b, 0.0);
                for k in 0..=n_steps {
                    if k > 0 {
                        (x, y) = ou_step(x, y, v.a, v.sigma, dt, rng);
                    }
                    let r = v.b + x;
                    short.push(r);
                    integral.push(v.b * k as f64 * dt + y);
                    prices.extend(mats.iter().map(|&m| v.bond_price(r, m)));
                }
            }
            ShortRate::HullWhite(hw) => {
                let (mut x, mut y) = (0.0, 0.0);
                for k in 0..=n_steps {
                    if k > 0 {
                        (x, y) = ou_step(x, y, hw.a, hw.sigma, dt, rng);
                    }
                    let t = k as f64 * dt;
                    short.push(x + hw.shift(t));
                    integral.push(y + shift_integral(hw, t));
                    prices.extend(mats.iter().map(|&m| hw.bond_price(t, x, m)));
                }
            }
            ShortRate::Cir(c) => {
                let e = (-c.a * dt).exp();
                let scale = c.sigma * c.sigma * (1.0 - e) / (4.0 * c.a);
                let dof = 4.0 * c.a * c.b / (c.sigma * c.sigma);
                let (mut r, mut y) = (c.r0, 0.0);
                for k in 0..=n_steps {
                    if k > 0 {
                        let next = scale * noncentral_chi_square(rng, dof, r * e / scale);
                        y += 0.5 * dt * (r + next);
                        r = next;
                    }
                    short.push(r);
                    integral.push(y);
                    prices.extend(mats.iter().map(|&m| c.bond_price(r, m)));
                }
            }
        }
    }
}

/// `∫_0^t φ = -ln P(0, t) + V(t) / 2`, Hull-White's deterministic part of
/// `∫ r` (Brigo and Mercurio, section 3.3).
fn shift_integral(hw: &HullWhite, t: f64) -> f64 {
    -hw.initial.discount(t).ln() + 0.5 * hw.integral_variance(t)
}

/// One exact step of `dx = -a x dt + σ dW` with its integral `y = ∫ x`:
/// `(x, y)` moves by a bivariate normal with
/// `Var x' = σ² (1 - e^(-2aΔ)) / (2a)`,
/// `Var Δy = σ²/a² [Δ - 2B + (1 - e^(-2aΔ)) / (2a)]` and
/// `Cov = σ² (1 - e^(-aΔ))² / (2a²)`, `B = (1 - e^(-aΔ)) / a`.
fn ou_step(x: f64, y: f64, a: f64, sigma: f64, dt: f64, rng: &mut StreamRng) -> (f64, f64) {
    let s2 = sigma * sigma;
    let e = (-a * dt).exp();
    let b = loading(a, dt);
    let one_minus_e2 = -(-2.0 * a * dt).exp_m1();
    let v1 = s2 * one_minus_e2 / (2.0 * a);
    let v2 = s2 / (a * a) * (dt - 2.0 * b + one_minus_e2 / (2.0 * a));
    let c12 = s2 / (2.0 * a * a) * (1.0 - e).powi(2);
    let (z1, z2) = (normal(rng), normal(rng));
    let s1 = v1.sqrt();
    let rho_part = c12 / s1;
    let rest = (v2 - rho_part * rho_part).max(0.0).sqrt();
    (x * e + s1 * z1, y + x * b + rho_part * z1 + rest * z2)
}

/// A curve at one step of one path: annually compounded spot rates by
/// maturity, for `prospicio_reserving::DiscountCurve::spot`.
#[derive(Debug, Clone, PartialEq)]
pub struct Curve {
    /// Maturities in years from the step.
    pub maturities: Vec<f64>,
    /// The annually compounded spot rate at each, `P^(-1/m) - 1`.
    pub rates: Vec<f64>,
}

/// Simulated paths, simulation-major: for each path, each step `k` from 0
/// (today) to `n_steps`, the short rate, `∫ r` since today and the bond
/// prices at the generator's maturities.
#[derive(Debug, Clone, PartialEq)]
pub struct Scenarios {
    n_sims: usize,
    n_steps: usize,
    step: Step,
    maturities: Vec<f64>,
    short: Vec<f64>,
    integral: Vec<f64>,
    prices: Vec<f64>,
    seed: u64,
}

impl Scenarios {
    /// The number of paths.
    pub fn n_sims(&self) -> usize {
        self.n_sims
    }

    /// The number of steps after today.
    pub fn n_steps(&self) -> usize {
        self.n_steps
    }

    /// The step.
    pub fn step(&self) -> Step {
        self.step
    }

    /// The time in years of step `k`.
    pub fn time(&self, k: usize) -> f64 {
        k as f64 / self.step.per_year() as f64
    }

    /// The maturities of the reported curve.
    pub fn maturities(&self) -> &[f64] {
        &self.maturities
    }

    fn index(&self, sim: usize, k: usize) -> Result<usize> {
        if sim >= self.n_sims || k > self.n_steps {
            return Err(Error::Data(format!(
                "path {sim}, step {k} is outside {} paths of {} steps",
                self.n_sims, self.n_steps
            )));
        }
        Ok(sim * (self.n_steps + 1) + k)
    }

    /// The short rate (continuously compounded) of path `sim` at step `k`.
    pub fn short_rate(&self, sim: usize, k: usize) -> Result<f64> {
        Ok(self.short[self.index(sim, k)?])
    }

    /// The stochastic discount factor `exp(-∫_0^t r)` of path `sim` at step
    /// `k`: the value today of 1 paid then on that path.
    pub fn deflator(&self, sim: usize, k: usize) -> Result<f64> {
        Ok((-self.integral[self.index(sim, k)?]).exp())
    }

    /// The price at step `k` of path `sim` of a bond paying 1 after the
    /// generator's maturity number `j`.
    pub fn bond_price(&self, sim: usize, k: usize, j: usize) -> Result<f64> {
        let i = self.index(sim, k)?;
        if j >= self.maturities.len() {
            return Err(Error::Data(format!(
                "maturity {j} is outside the {} reported",
                self.maturities.len()
            )));
        }
        Ok(self.prices[i * self.maturities.len() + j])
    }

    /// The curve at step `k` of path `sim`, annually compounded.
    pub fn curve(&self, sim: usize, k: usize) -> Result<Curve> {
        let i = self.index(sim, k)?;
        let n = self.maturities.len();
        let rates = self.prices[i * n..(i + 1) * n]
            .iter()
            .zip(&self.maturities)
            .map(|(p, m)| p.powf(-1.0 / m) - 1.0)
            .collect();
        Ok(Curve {
            maturities: self.maturities.clone(),
            rates,
        })
    }

    /// The short rates as a distribution with one component per step
    /// (dimension `step`, 0 to `n_steps`).
    pub fn short_rates(&self) -> Result<PredictiveDistribution> {
        self.by_step(self.short.clone(), "esg_short_rate")
    }

    /// The deflators as a distribution with one component per step.
    pub fn deflators(&self) -> Result<PredictiveDistribution> {
        let draws = self.integral.iter().map(|y| (-y).exp()).collect();
        self.by_step(draws, "esg_deflator")
    }

    fn by_step(&self, draws: Vec<f64>, model: &str) -> Result<PredictiveDistribution> {
        let components: Vec<ComponentKey> = (0..=self.n_steps)
            .map(|k| vec![(k as i64).into()])
            .collect();
        let mut provenance = Provenance::new(model)
            .version("prospicio-esg", env!("CARGO_PKG_VERSION"))
            .param("steps_per_year", self.step.per_year());
        provenance.seed = Some(self.seed);
        PredictiveDistribution::from_draws(vec!["step".into()], components, draws, provenance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cir, InitialCurve, Vasicek};

    #[test]
    fn a_path_replays_alone_and_steps_are_reported() {
        let esg = Esg::new(Cir::new(0.4, 0.03, 0.08, 0.02).unwrap()).with_step(Step::Annual);
        let all = esg.simulate(3, 5, 11).unwrap();
        assert_eq!((all.n_sims(), all.n_steps()), (5, 3));
        assert_eq!(all.time(3), 3.0);
        let first = esg.simulate(3, 1, 11).unwrap();
        for k in 0..=3 {
            assert_eq!(
                all.short_rate(0, k).unwrap(),
                first.short_rate(0, k).unwrap()
            );
        }
        assert_eq!(all.deflator(2, 0).unwrap(), 1.0);
        assert!(all.short_rate(5, 0).is_err() && all.short_rate(0, 4).is_err());
        assert!(all.bond_price(0, 0, 8).is_err());
        let curve = all.curve(0, 0).unwrap();
        assert_eq!(curve.maturities.len(), 8);
        let pd = all.short_rates().unwrap();
        assert_eq!((pd.n_sims(), pd.n_components()), (5, 4));
        assert_eq!(all.deflators().unwrap().n_components(), 4);
    }

    #[test]
    fn the_inputs_are_checked() {
        let esg = Esg::new(Vasicek::new(0.3, 0.04, 0.01, 0.02).unwrap());
        assert!(esg.simulate(0, 10, 1).is_err());
        assert!(esg.simulate(1, 0, 1).is_err());
        assert!(
            esg.clone()
                .with_maturities(vec![2.0, 1.0])
                .simulate(1, 1, 1)
                .is_err()
        );
        let flat = InitialCurve::spot(vec![1.0], vec![0.03]).unwrap();
        let hw = HullWhite::new(0.1, 0.01, flat).unwrap();
        let paths = Esg::new(hw).simulate(1, 3, 1).unwrap();
        // Today's curve is the initial one.
        let today = paths.curve(1, 0).unwrap();
        assert!(today.rates.iter().all(|r| (r - 0.03).abs() < 1e-12));
    }
}
