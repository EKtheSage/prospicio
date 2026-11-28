//! Simulated paths of the short rate, its integral and the curve.

use prospicio_core::{Error, Result, StreamRng};
use prospicio_prob::{ComponentKey, PredictiveDistribution, Provenance};

use prospicio_math::linalg::{cholesky, cholesky_solve};

use crate::inflation::{Inflation, Superimposed};
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

/// An economic scenario generator: a short-rate model, economic and
/// superimposed claims inflation if wanted, the correlation of their
/// shocks, the step and the maturities of the curve reported at each step.
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
///
/// With inflation correlated with the rate, and claims inflation above it:
///
/// ```
/// use prospicio_esg::{Esg, Index, Inflation, Step, Superimposed, Vasicek};
///
/// let esg = Esg::new(Vasicek::new(0.3, 0.04, 0.01, 0.02)?)
///     .with_inflation(Inflation::new(0.5, 0.025, 0.01, 0.03)?)
///     .with_superimposed(Superimposed::new(0.01, 0.5, 0.005)?)
///     .with_correlation(vec![
///         vec![1.0, 0.5, 0.0],
///         vec![0.5, 1.0, 0.3],
///         vec![0.0, 0.3, 1.0],
///     ]);
/// let paths = esg.simulate(3, 100, 1)?;
/// // One claims index level a year for each path, 1 today, for
/// // `prospicio_reserving::IndexPaths`.
/// let levels = paths.index_levels(Index::Claims, Step::Annual)?;
/// assert_eq!(levels.len(), 300);
/// # Ok::<(), prospicio_core::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Esg {
    /// The short-rate model.
    pub rate: ShortRate,
    /// Economic inflation, if simulated.
    pub inflation: Option<Inflation>,
    /// Superimposed claims inflation, if simulated.
    pub superimposed: Option<Superimposed>,
    /// The correlation matrix of the shocks, in the order rate, inflation,
    /// superimposed, over those simulated; empty for independent shocks.
    pub correlation: Vec<Vec<f64>>,
    /// The time step.
    pub step: Step,
    /// The maturities in years of the curve reported at each step.
    pub maturities: Vec<f64>,
}

impl Esg {
    /// A generator on `rate` alone, monthly, reporting the curve at 1, 2,
    /// 3, 5, 7, 10, 20 and 30 years.
    pub fn new(rate: impl Into<ShortRate>) -> Self {
        Self {
            rate: rate.into(),
            inflation: None,
            superimposed: None,
            correlation: Vec::new(),
            step: Step::Monthly,
            maturities: vec![1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0],
        }
    }

    /// With economic inflation.
    pub fn with_inflation(mut self, inflation: Inflation) -> Self {
        self.inflation = Some(inflation);
        self
    }

    /// With superimposed claims inflation.
    pub fn with_superimposed(mut self, superimposed: Superimposed) -> Self {
        self.superimposed = Some(superimposed);
        self
    }

    /// With this correlation matrix of the shocks: rows and columns in the
    /// order rate, inflation, superimposed, over those simulated.
    pub fn with_correlation(mut self, correlation: Vec<Vec<f64>>) -> Self {
        self.correlation = correlation;
        self
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
    /// Vasicek, Hull-White and inflation are Gaussian: the factors and
    /// their integrals over each step are drawn jointly from their exact
    /// transition, correlated through the shocks' correlation `ρ` (two
    /// factors' noises over a step have covariance `ρ ∫ f g`, `f` and `g`
    /// their kernels on the step), so the deflators and the price index
    /// are exact. CIR draws the rate from its exact noncentral chi-square
    /// transition and integrates it by the trapezoid rule over the step, a
    /// bias of order `Δ²` in the deflators; its shock cannot be correlated.
    ///
    /// Year `y`'s superimposed shock `ε_y` is drawn at the year's end,
    /// conditional on that year's Brownian increments of the rate and
    /// inflation, so that `Corr(ε_y, W(y) - W(y - 1))` is the given
    /// correlation; its excess is spread over the year, so the claims
    /// index within a year uses a shock of the whole year.
    ///
    /// Fails unless `years` and `sims` are positive, the maturities
    /// finite, positive and increasing, and the correlation empty or a
    /// positive definite correlation matrix of the shocks simulated, with
    /// no correlation of a CIR rate.
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
        let plan = Plan::new(self, dt)?;
        let points = n_steps + 1;
        let n_mat = self.maturities.len();
        let mut out = Paths {
            short: Vec::with_capacity(sims * points),
            integral: Vec::with_capacity(sims * points),
            prices: Vec::with_capacity(sims * points * n_mat),
            ..Paths::default()
        };
        for sim in 0..sims {
            let mut rng = StreamRng::new(seed, sim as u64);
            self.path(&plan, &mut rng, n_steps, dt, &mut out);
        }
        Ok(Scenarios {
            n_sims: sims,
            n_steps,
            step: self.step,
            maturities: self.maturities.clone(),
            short: out.short,
            integral: out.integral,
            prices: out.prices,
            inflation: out.inflation,
            price_log: out.price_log,
            claims_log: out.claims_log,
            superimposed: out.superimposed,
            seed,
        })
    }

    /// One path: the short rate, `∫ r`, the bond prices and the inflation
    /// at each point, and the superimposed excess of each year.
    fn path(&self, plan: &Plan, rng: &mut StreamRng, n_steps: usize, dt: f64, out: &mut Paths) {
        let mats = &self.maturities;
        let per_year = self.step.per_year();
        let base = out.price_log.len();
        // The Gaussian rate's factor (x, ∫x), CIR's rate and ∫r.
        let (mut x, mut y) = match &self.rate {
            ShortRate::Vasicek(v) => (v.r0 - v.b, 0.0),
            ShortRate::HullWhite(_) => (0.0, 0.0),
            ShortRate::Cir(c) => (c.r0, 0.0),
        };
        let (mut u, mut yu) = (self.inflation.map_or(0.0, |i| i.i0 - i.mean), 0.0);
        let mut s_prev = self.superimposed.map_or(0.0, |s| s.mean);
        let mut cum_s = 0.0;
        let mut wsum = vec![0.0; plan.w.len()];
        let mut z = vec![0.0; plan.n];
        let mut noise = vec![0.0; plan.n];
        for k in 0..=n_steps {
            if k > 0 {
                if let ShortRate::Cir(_) = &self.rate {
                    let next = plan.cir_step(x, rng);
                    y += 0.5 * dt * (x + next);
                    x = next;
                }
                z.iter_mut().for_each(|zi| *zi = normal(rng));
                plan.correlate(&z, &mut noise);
                if let Some((px, py, a)) = plan.rate {
                    (x, y) = (
                        x * (-a * dt).exp() + noise[px],
                        y + x * loading(a, dt) + noise[py],
                    );
                }
                if let Some((px, py, a)) = plan.inflation {
                    (u, yu) = (
                        u * (-a * dt).exp() + noise[px],
                        yu + u * loading(a, dt) + noise[py],
                    );
                }
                for (sum, &j) in wsum.iter_mut().zip(&plan.w) {
                    *sum += noise[j];
                }
            }
            let t = k as f64 * dt;
            match &self.rate {
                ShortRate::Vasicek(v) => {
                    let r = v.b + x;
                    out.short.push(r);
                    out.integral.push(v.b * t + y);
                    out.prices.extend(mats.iter().map(|&m| v.bond_price(r, m)));
                }
                ShortRate::HullWhite(hw) => {
                    out.short.push(x + hw.shift(t));
                    out.integral.push(y + shift_integral(hw, t));
                    out.prices
                        .extend(mats.iter().map(|&m| hw.bond_price(t, x, m)));
                }
                ShortRate::Cir(c) => {
                    out.short.push(x);
                    out.integral.push(y);
                    out.prices.extend(mats.iter().map(|&m| c.bond_price(x, m)));
                }
            }
            if let Some(i) = self.inflation {
                out.inflation.push(i.mean + u);
                out.price_log.push(i.mean * t + yu);
            }
            if let Some(sup) = self.superimposed {
                if k == 0 {
                    out.claims_log.push(0.0);
                } else if k % per_year == 0 {
                    let dot: f64 = plan.coeff.iter().zip(&wsum).map(|(c, w)| c * w).sum();
                    let eps = dot + plan.sd * normal(rng);
                    let s = sup.mean + sup.persistence * (s_prev - sup.mean) + sup.sigma * eps;
                    out.superimposed.push(s);
                    for j in k - per_year + 1..=k {
                        let price = out.price_log.get(base + j).copied().unwrap_or(0.0);
                        let within = (j + per_year - k) as f64 * dt;
                        out.claims_log.push(price + cum_s + s * within);
                    }
                    cum_s += s;
                    s_prev = s;
                    wsum.iter_mut().for_each(|w| *w = 0.0);
                }
            }
        }
    }
}

/// The paths of all simulations, simulation-major, as they are drawn.
#[derive(Default)]
struct Paths {
    short: Vec<f64>,
    integral: Vec<f64>,
    prices: Vec<f64>,
    inflation: Vec<f64>,
    price_log: Vec<f64>,
    claims_log: Vec<f64>,
    superimposed: Vec<f64>,
}

/// The kernel of a Gaussian noise over a step of length `Δ`, at `v` before
/// the step's end: a factor's `σ e^(-a v)`, its integral's `σ B_a(v)`, or
/// the Brownian increment's 1.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Kernel {
    Factor,
    Integral,
    Increment,
}

#[derive(Debug, Clone, Copy)]
struct Noise {
    /// The shock's row in the correlation matrix.
    shock: usize,
    a: f64,
    sigma: f64,
    kernel: Kernel,
}

/// `∫_0^Δ f g dv` of two noises' kernels, before the correlation.
fn kernel_product(f: Noise, g: Noise, dt: f64) -> f64 {
    use Kernel::*;
    let e = |c: f64| loading(c, dt);
    let (sf, sg) = (f.sigma, g.sigma);
    match (f.kernel, g.kernel) {
        (Factor, Factor) => sf * sg * e(f.a + g.a),
        (Factor, Integral) => sf * sg * (e(f.a) - e(f.a + g.a)) / g.a,
        (Integral, Factor) => kernel_product(g, f, dt),
        (Integral, Integral) => sf * sg / (f.a * g.a) * (dt - e(f.a) - e(g.a) + e(f.a + g.a)),
        (Increment, Factor) => sg * e(g.a),
        (Increment, Integral) => sg * (dt - e(g.a)) / g.a,
        (Factor | Integral, Increment) => kernel_product(g, f, dt),
        (Increment, Increment) => dt,
    }
}

/// What a simulation needs once: the noises of a step, their Cholesky
/// factor, and the superimposed shock's regression on the year's
/// increments.
#[derive(Debug, Clone)]
struct Plan {
    n: usize,
    chol: Vec<f64>,
    /// The rate's factor and integral noises and its speed, if Gaussian.
    rate: Option<(usize, usize, f64)>,
    /// Inflation's, if simulated.
    inflation: Option<(usize, usize, f64)>,
    /// The Brownian increments' noises, when there is a superimposed shock.
    w: Vec<usize>,
    coeff: Vec<f64>,
    sd: f64,
    /// CIR's transition: `e^(-aΔ)`, the scale and the degrees of freedom.
    cir: (f64, f64, f64),
}

impl Plan {
    fn new(esg: &Esg, dt: f64) -> Result<Self> {
        let n_shocks =
            1 + usize::from(esg.inflation.is_some()) + usize::from(esg.superimposed.is_some());
        let corr = correlation(&esg.correlation, n_shocks)?;
        let rho = |i: usize, j: usize| corr[i * n_shocks + j];
        let mut cir = (0.0, 0.0, 0.0);
        let rate = match &esg.rate {
            ShortRate::Vasicek(v) => Some((v.a, v.sigma)),
            ShortRate::HullWhite(hw) => Some((hw.a, hw.sigma)),
            ShortRate::Cir(c) => {
                if (1..n_shocks).any(|j| rho(0, j) != 0.0) {
                    return Err(Error::Data(
                        "a CIR rate's shock cannot be correlated: use Vasicek or Hull-White".into(),
                    ));
                }
                let e = (-c.a * dt).exp();
                cir = (
                    e,
                    c.sigma * c.sigma * (1.0 - e) / (4.0 * c.a),
                    4.0 * c.a * c.b / (c.sigma * c.sigma),
                );
                None
            }
        };
        let infl = esg.inflation.map(|i| (i.kappa, i.sigma));
        // The Gaussian factors: (shock row, speed, volatility).
        let factors: Vec<(usize, f64, f64)> = rate
            .map(|(a, s)| (0, a, s))
            .into_iter()
            .chain(infl.map(|(a, s)| (1, a, s)))
            .collect();
        let mut noises = Vec::new();
        for &(shock, a, sigma) in &factors {
            for kernel in [Kernel::Factor, Kernel::Integral] {
                noises.push(Noise {
                    shock,
                    a,
                    sigma,
                    kernel,
                });
            }
        }
        let mut w = Vec::new();
        let (mut coeff, mut sd) = (Vec::new(), 0.0);
        if esg.superimposed.is_some() {
            let sup = n_shocks - 1;
            for &(shock, a, _) in &factors {
                w.push(noises.len());
                noises.push(Noise {
                    shock,
                    a,
                    sigma: 1.0,
                    kernel: Kernel::Increment,
                });
            }
            // ε on the year's increments, which have unit variance and the
            // shocks' correlations: coeff = C_ww⁻¹ c, the rest independent.
            let g = factors.len();
            let cww: Vec<f64> = (0..g * g)
                .map(|ij| rho(factors[ij / g].0, factors[ij % g].0))
                .collect();
            let c: Vec<f64> = factors.iter().map(|f| rho(sup, f.0)).collect();
            coeff = if g == 0 {
                Vec::new()
            } else {
                let l = cholesky(&cww, g).expect("a principal minor of a positive definite matrix");
                cholesky_solve(&l, &c)
            };
            let explained: f64 = coeff.iter().zip(&c).map(|(a, b)| a * b).sum();
            sd = (1.0 - explained).max(0.0).sqrt();
        }
        let n = noises.len();
        let cov: Vec<f64> = (0..n * n)
            .map(|ij| {
                let (f, g) = (noises[ij / n], noises[ij % n]);
                rho(f.shock, g.shock) * kernel_product(f, g, dt)
            })
            .collect();
        let speed = |shock| factors.iter().find(|f| f.0 == shock).map(|f| f.1);
        let pos = |shock: usize| {
            let i = factors.iter().position(|f| f.0 == shock)?;
            Some((2 * i, 2 * i + 1, speed(shock)?))
        };
        Ok(Self {
            n,
            chol: semidefinite_cholesky(&cov, n),
            rate: rate.and_then(|_| pos(0)),
            inflation: infl.and_then(|_| pos(1)),
            w,
            coeff,
            sd,
            cir,
        })
    }

    /// `noise = L z`.
    fn correlate(&self, z: &[f64], noise: &mut [f64]) {
        let n = self.n;
        for (i, out) in noise.iter_mut().enumerate() {
            *out = (0..=i).map(|j| self.chol[i * n + j] * z[j]).sum();
        }
    }

    fn cir_step(&self, r: f64, rng: &mut StreamRng) -> f64 {
        let (e, scale, dof) = self.cir;
        scale * noncentral_chi_square(rng, dof, r * e / scale)
    }
}

/// The correlation matrix of `n` shocks, row-major: the identity when
/// `given` is empty, otherwise `given` checked.
fn correlation(given: &[Vec<f64>], n: usize) -> Result<Vec<f64>> {
    if given.is_empty() {
        return Ok((0..n * n)
            .map(|ij| f64::from(u8::from(ij / n == ij % n)))
            .collect());
    }
    if given.len() != n || given.iter().any(|row| row.len() != n) {
        return Err(Error::Data(format!(
            "the correlation must be {n} by {n}, one row for each shock simulated (rate, inflation, superimposed)"
        )));
    }
    let flat: Vec<f64> = given.iter().flatten().copied().collect();
    for i in 0..n {
        for j in 0..n {
            let (v, t) = (flat[i * n + j], flat[j * n + i]);
            let ok = v.is_finite()
                && (v - t).abs() < 1e-12
                && if i == j { v == 1.0 } else { v.abs() <= 1.0 };
            if !ok {
                return Err(Error::InvalidParameter {
                    name: "correlation",
                    value: v,
                    reason: "must be symmetric with a unit diagonal and entries in [-1, 1]",
                });
            }
        }
    }
    if cholesky(&flat, n).is_none() {
        return Err(Error::Data(
            "the correlation matrix is not positive definite".into(),
        ));
    }
    Ok(flat)
}

/// The lower Cholesky factor of a positive semidefinite `n × n` matrix,
/// row-major, a pivot that rounding takes to zero or below giving a zero
/// column: the noises of a short step are nearly collinear.
fn semidefinite_cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let dot: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            if i == j {
                l[i * n + i] = (a[i * n + i] - dot).max(0.0).sqrt();
            } else if l[j * n + j] > 0.0 {
                l[i * n + j] = (a[i * n + j] - dot) / l[j * n + j];
            }
        }
    }
    l
}

/// `∫_0^t φ = -ln P(0, t) + V(t) / 2`, Hull-White's deterministic part of
/// `∫ r` (Brigo and Mercurio, section 3.3).
fn shift_integral(hw: &HullWhite, t: f64) -> f64 {
    -hw.initial.discount(t).ln() + 0.5 * hw.integral_variance(t)
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
/// (today) to `n_steps`, the short rate, `∫ r` since today, the bond
/// prices at the generator's maturities and, when simulated, the inflation
/// rate and the price and claims indices; and each year's superimposed
/// excess.
#[derive(Debug, Clone, PartialEq)]
pub struct Scenarios {
    n_sims: usize,
    n_steps: usize,
    step: Step,
    maturities: Vec<f64>,
    short: Vec<f64>,
    integral: Vec<f64>,
    prices: Vec<f64>,
    inflation: Vec<f64>,
    price_log: Vec<f64>,
    claims_log: Vec<f64>,
    superimposed: Vec<f64>,
    seed: u64,
}

/// Which index [`Scenarios::index_levels`] reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Index {
    /// The price index, `exp(∫ i)`.
    Price,
    /// The claims index: the price index times `exp(∫ s)`, the
    /// superimposed excess; the price index without one, and the excess
    /// alone without economic inflation.
    Claims,
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

    /// The economic inflation rate (continuously compounded) of path `sim`
    /// at step `k`. Fails without an inflation model.
    pub fn inflation_rate(&self, sim: usize, k: usize) -> Result<f64> {
        let i = self.index(sim, k)?;
        self.inflation
            .get(i)
            .copied()
            .ok_or_else(|| Error::Data("the generator has no inflation model".into()))
    }

    /// The level of `index` on path `sim` at step `k`, 1 today. Fails for
    /// the price index without an inflation model, and for the claims
    /// index with neither inflation nor a superimposed excess.
    pub fn index_level(&self, index: Index, sim: usize, k: usize) -> Result<f64> {
        let i = self.index(sim, k)?;
        Ok(self.log_index(index)?[i].exp())
    }

    fn log_index(&self, index: Index) -> Result<&[f64]> {
        let logs = match index {
            Index::Claims if !self.claims_log.is_empty() => &self.claims_log,
            _ => &self.price_log,
        };
        if logs.is_empty() {
            return Err(Error::Data(match index {
                Index::Price => "the generator has no inflation model".into(),
                Index::Claims => {
                    "the generator has neither inflation nor a superimposed excess".into()
                }
            }));
        }
        Ok(logs)
    }

    /// The superimposed excess of path `sim` in year `year`, 1 to the
    /// horizon. Fails without a superimposed model.
    pub fn superimposed(&self, sim: usize, year: usize) -> Result<f64> {
        let years = self.n_steps / self.step.per_year();
        if self.superimposed.is_empty() {
            return Err(Error::Data(
                "the generator has no superimposed excess".into(),
            ));
        }
        if sim >= self.n_sims || year == 0 || year > years {
            return Err(Error::Data(format!(
                "path {sim}, year {year} is outside {} paths of years 1 to {years}",
                self.n_sims
            )));
        }
        Ok(self.superimposed[sim * years + year - 1])
    }

    /// The levels of `index` at the end of each period of `grain`,
    /// simulation-major, 1 today: `values[i * n + p]` ends period `p` of
    /// path `i`, `n` the horizon's periods. Scaled by a
    /// `prospicio_reserving::CalendarIndex`'s last level, they are the
    /// `IndexPaths` from the period after it at that grain, one path a
    /// simulation, for `Indexation::reinflate_draws`.
    ///
    /// Fails when `grain` is finer than the simulation's step, or the
    /// index is not simulated ([`index_level`](Self::index_level)).
    pub fn index_levels(&self, index: Index, grain: Step) -> Result<Vec<f64>> {
        let (per, step) = (grain.per_year(), self.step.per_year());
        if per > step {
            return Err(Error::Data(format!(
                "levels {per} a year need a step at least as fine, the simulation has {step}"
            )));
        }
        let logs = self.log_index(index)?;
        let stride = step / per;
        let periods = self.n_steps / stride;
        let mut out = Vec::with_capacity(self.n_sims * periods);
        for sim in 0..self.n_sims {
            let row = &logs[sim * (self.n_steps + 1)..(sim + 1) * (self.n_steps + 1)];
            out.extend((1..=periods).map(|p| row[p * stride].exp()));
        }
        Ok(out)
    }

    /// The levels of `index` as a distribution with one component per
    /// step (dimension `step`, 0 to `n_steps`).
    pub fn index_levels_by_step(&self, index: Index) -> Result<PredictiveDistribution> {
        let draws = self.log_index(index)?.iter().map(|l| l.exp()).collect();
        let model = match index {
            Index::Price => "esg_price_index",
            Index::Claims => "esg_claims_index",
        };
        self.by_step(draws, model)
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
