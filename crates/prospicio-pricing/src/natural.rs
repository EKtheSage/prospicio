//! Pricing and natural allocation of a portfolio held as the conditional
//! expectations of its units given the total: Mildenhall and Major,
//! *Pricing Insurance Risk* (2022), and CAS Monograph 15 (Major and
//! Mildenhall, 2026).
//!
//! A [`Portfolio`] is the distribution of the total `X = Σ Xᵢ` on its
//! distinct values `x_k`, with probabilities `p_k`, and each unit's
//! conditional expectation `κᵢ(x) = E[Xᵢ | X = x]`. Every price and
//! allocation below depends only on these. With assets `a` the insurer
//! pays `X ∧ a`, and each unit's recovery has equal priority:
//! `Xᵢ min(1, a / X)`.
//!
//! - [`Portfolio::price`]: the premium of `X ∧ a` under a distortion `g`,
//!   `P = Σ_{x_k ≤ a} x_k Δg_k + a g(S(a))`, and its allocation to units,
//!   with the loss, margin, capital and assets of each ([`Pentagon`]).
//!   The [`Allocation::Linear`] allocation gives unit `i`
//!   `Σ_{x_k ≤ a} κᵢ(x_k) Δg_k` plus its expected share `αᵢ(a)` of the
//!   assets above `a`; [`Allocation::Lifted`] uses the distorted share
//!   `βᵢ(a)` instead. They agree when `a` is the largest loss.
//! - Capital by layer: in the layer from `x` to `x + dx` the premium is
//!   `g(S) dx`, of which `S dx` is loss and `(g - S) dx` margin, and the
//!   capital is `(1 - g) dx`. Each unit gets the layer's capital in
//!   proportion to its margin there, so every layer earns one return,
//!   `(g - S) / (1 - g)`, and units' returns differ by where their margin
//!   sits.
//! - [`Portfolio::bodoff`]: Bodoff's percentile layer of capital, the
//!   assets of each layer shared by each unit's expected share of the
//!   losses that reach it.
//! - [`Portfolio::epd`]: the expected policyholder deficit ratio, in total
//!   and by unit under equal priority.
//! - [`Portfolio::calibrate`]: the distortion of a family that gives a
//!   target premium, return or loss ratio.
//!
//! The formulas follow the Python `aggregate` package (1.0.1), which this
//! module matches on Monograph 15's InsCo example to `1e-12`
//! (`validation/tests/pricing.rs`). One difference: tied totals' `κ` is
//! the probability-weighted mean of the units, where `aggregate` takes the
//! unweighted mean (the same when the rows are equally likely).

use prospicio_core::{Error, Result};
use prospicio_prob::distortion::{Family, calibrate};
use prospicio_prob::{Distortion, Grid, PredictiveDistribution};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

/// Which share of the assets above the asset level a unit gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Allocation {
    /// The expected share, `αᵢ(a) = E[Xᵢ / X | X > a]`.
    #[default]
    Linear,
    /// The distorted share, `βᵢ(a)`: the same average under the distorted
    /// probabilities. Lifts the units that drive the tail.
    Lifted,
}

/// Loss, margin, premium, capital and assets of a cover or a unit, with
/// `P = L + M` and `a = P + Q`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pentagon {
    /// Expected loss paid, `L`.
    pub loss: f64,
    /// Margin `M = P - L`.
    pub margin: f64,
    /// Premium `P`.
    pub premium: f64,
    /// Capital `Q = a - P`.
    pub capital: f64,
    /// Assets `a`.
    pub assets: f64,
}

/// A quantity of a [`Pentagon`], for [`Pentagon::solve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantity {
    /// `L`.
    Loss,
    /// `M`.
    Margin,
    /// `P`.
    Premium,
    /// `Q`.
    Capital,
    /// `a`.
    Assets,
    /// `L / P`.
    LossRatio,
    /// `P / Q`, the premium leverage.
    PremiumToCapital,
    /// `M / Q`.
    ReturnOnCapital,
}

impl Pentagon {
    /// From the loss, premium and assets.
    pub fn new(loss: f64, premium: f64, assets: f64) -> Self {
        Self {
            loss,
            margin: premium - loss,
            premium,
            capital: assets - premium,
            assets,
        }
    }

    /// `L / P`.
    pub fn loss_ratio(&self) -> f64 {
        self.loss / self.premium
    }

    /// `P / Q`.
    pub fn premium_to_capital(&self) -> f64 {
        self.premium / self.capital
    }

    /// `M / Q`.
    pub fn return_on_capital(&self) -> f64 {
        self.margin / self.capital
    }

    /// `δ = M / (a - L)`, the discount on the assets at risk: with return
    /// `ι` it is `ι / (1 + ι)`, and `P = (1 - δ) L + δ a`.
    pub fn discount(&self) -> f64 {
        self.margin / (self.assets - self.loss)
    }

    /// The pentagon fixed by three known quantities: amounts or ratios.
    /// The relations `P = L + M`, `a = P + Q`, `L = LR · P`, `P = PQ · Q`
    /// and `M = ι Q` are linear in the amounts once the ratios are known,
    /// so three knowns give a square linear system. Fails if they do not
    /// determine the pentagon (two ratios and no amount, or a dependent
    /// set such as `L`, `M` and `P`).
    ///
    /// ```
    /// use prospicio_pricing::natural::{Pentagon, Quantity};
    ///
    /// // Monograph 15's InsCo: L = 46.6, a = 100, a 15% cost of capital.
    /// let p = Pentagon::solve([
    ///     (Quantity::Loss, 46.6),
    ///     (Quantity::Assets, 100.0),
    ///     (Quantity::ReturnOnCapital, 0.15),
    /// ])
    /// .unwrap();
    /// assert!((p.premium - 53.565217391304344).abs() < 1e-12);
    /// ```
    pub fn solve(known: [(Quantity, f64); 3]) -> Result<Self> {
        use Quantity::*;
        // Unknowns in the order L, M, P, Q, a.
        let idx = |q: Quantity| match q {
            Loss => 0,
            Margin => 1,
            Premium => 2,
            Capital => 3,
            _ => 4,
        };
        let mut a = Vec::with_capacity(25);
        let mut b = Vec::with_capacity(5);
        // P - L - M = 0 and a - P - Q = 0.
        a.extend([-1.0, -1.0, 1.0, 0.0, 0.0]);
        b.push(0.0);
        a.extend([0.0, 0.0, -1.0, -1.0, 1.0]);
        b.push(0.0);
        for (i, &(q, v)) in known.iter().enumerate() {
            if !v.is_finite() {
                return Err(Error::Data(format!("{q:?} must be finite")));
            }
            if known[..i].iter().any(|k| k.0 == q) {
                return Err(Error::Data(format!("{q:?} is given twice")));
            }
            let row = match q {
                Loss | Margin | Premium | Capital | Assets => {
                    let mut r = [0.0; 5];
                    r[idx(q)] = 1.0;
                    b.push(v);
                    r
                }
                LossRatio => {
                    b.push(0.0);
                    [1.0, 0.0, -v, 0.0, 0.0]
                }
                PremiumToCapital => {
                    b.push(0.0);
                    [0.0, 0.0, 1.0, -v, 0.0]
                }
                ReturnOnCapital => {
                    b.push(0.0);
                    [0.0, 1.0, 0.0, -v, 0.0]
                }
            };
            a.extend(row);
        }
        let x = prospicio_math::linalg::solve(a, b, 5)
            .ok_or_else(|| Error::Data(format!("{known:?} do not determine the pentagon")))?;
        Ok(Self {
            loss: x[0],
            margin: x[1],
            premium: x[2],
            capital: x[3],
            assets: x[4],
        })
    }
}

/// The price of a portfolio and its natural allocation.
#[derive(Debug, Clone, PartialEq)]
pub struct NaturalPrice {
    /// Unit names, in the portfolio's order.
    pub units: Vec<String>,
    /// Each unit's share; the amounts add up to [`total`](Self::total).
    pub allocated: Vec<Pentagon>,
    /// The portfolio.
    pub total: Pentagon,
}

/// A portfolio as the distribution of its total and each unit's
/// conditional expectation given the total. See the [module](self).
#[derive(Debug, Clone, PartialEq)]
pub struct Portfolio {
    units: Vec<String>,
    /// Distinct totals, ascending, each with positive probability.
    x: Vec<f64>,
    p: Vec<f64>,
    /// `κ`, total-major: `kappa[k * m + i]` is unit `i` at total `x[k]`.
    kappa: Vec<f64>,
    /// `S_k = P(X > x_k)`, summed from the top, exactly 0 at the top.
    s: Vec<f64>,
}

impl Portfolio {
    /// From scenarios: `rows[j]` holds each unit's loss in scenario `j`,
    /// with probability `probs[j]` (all equal when `None`). Losses must be
    /// finite and non-negative. Scenarios with the same total are merged,
    /// and `κ` there is their probability-weighted mean.
    ///
    /// ```
    /// use prospicio_pricing::natural::Portfolio;
    ///
    /// let port = Portfolio::from_rows(
    ///     vec!["a".into(), "b".into()],
    ///     &[vec![1.0, 3.0], vec![2.0, 2.0], vec![5.0, 1.0]],
    ///     None,
    /// )
    /// .unwrap();
    /// assert_eq!(port.totals(), [4.0, 6.0]);
    /// assert_eq!(port.kappa(0), [1.5, 5.0]); // E[a | X = 4] = (1 + 2) / 2
    /// ```
    pub fn from_rows(units: Vec<String>, rows: &[Vec<f64>], probs: Option<&[f64]>) -> Result<Self> {
        let m = units.len();
        if m == 0 || rows.is_empty() {
            return Err(Error::Data(
                "a portfolio needs at least one unit and one scenario".into(),
            ));
        }
        let n = rows.len();
        let w: Vec<f64> = match probs {
            Some(p) if p.len() == n => p.to_vec(),
            Some(_) => return Err(Error::Data("give one probability per scenario".into())),
            None => vec![1.0 / n as f64; n],
        };
        if w.iter().any(|w| !(w.is_finite() && *w >= 0.0)) {
            return Err(Error::Data(
                "scenario probabilities must be non-negative".into(),
            ));
        }
        let sum: f64 = w.iter().sum();
        if (sum - 1.0).abs() > 1e-9 {
            return Err(Error::Data(format!(
                "scenario probabilities sum to {sum}, not 1"
            )));
        }
        let mut scen = Vec::with_capacity(n);
        for (row, &w) in rows.iter().zip(&w) {
            if row.len() != m {
                return Err(Error::Data(format!("each scenario needs {m} unit losses")));
            }
            if row.iter().any(|x| !(x.is_finite() && *x >= 0.0)) {
                return Err(Error::Data(
                    "unit losses must be finite and non-negative".into(),
                ));
            }
            if w > 0.0 {
                scen.push((row.iter().sum::<f64>(), w, row.as_slice()));
            }
        }
        scen.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (mut x, mut p, mut kappa) = (Vec::new(), Vec::new(), Vec::new());
        for (total, w, row) in scen {
            if x.last() == Some(&total) {
                let k = x.len() - 1;
                p[k] += w;
                for (i, v) in row.iter().enumerate() {
                    kappa[k * m + i] += w * v;
                }
            } else {
                x.push(total);
                p.push(w);
                kappa.extend(row.iter().map(|v| w * v));
            }
        }
        for (k, &pk) in p.iter().enumerate() {
            kappa[k * m..(k + 1) * m].iter_mut().for_each(|v| *v /= pk);
        }
        Ok(Self::build(units, x, p, kappa))
    }

    /// From a joint simulation: each component is a unit, named by its key
    /// (values joined by `/`), and each simulation an equally likely
    /// scenario.
    pub fn from_predictive(pd: &PredictiveDistribution) -> Result<Self> {
        let m = pd.n_components();
        let units = pd
            .components()
            .iter()
            .map(|key| {
                key.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join("/")
            })
            .collect();
        let rows: Vec<Vec<f64>> = pd
            .draw_matrix()
            .chunks_exact(m)
            .map(<[f64]>::to_vec)
            .collect();
        Self::from_rows(units, &rows, None)
    }

    /// From independent units, each a [`Grid`] with the same step: the
    /// total's distribution is their convolution, and
    /// `κᵢ(x) = Σ_y y pᵢ(y) p₋ᵢ(x - y) / p(x)`, where `p₋ᵢ` is the
    /// convolution of the other units. Both convolutions are done by FFT,
    /// on a buffer long enough that nothing wraps. Totals with probability
    /// below `1e-14` are dropped: their `κ` would be FFT rounding.
    ///
    /// ```
    /// use prospicio_pricing::natural::Portfolio;
    /// use prospicio_prob::Grid;
    ///
    /// // Two fair coins worth 1 and 2.
    /// let a = Grid::new(1.0, vec![0.5, 0.5]).unwrap();
    /// let b = Grid::new(1.0, vec![0.5, 0.0, 0.5]).unwrap();
    /// let port = Portfolio::from_independent(vec!["a".into(), "b".into()], &[a, b]).unwrap();
    /// assert_eq!(port.totals(), [0.0, 1.0, 2.0, 3.0]);
    /// assert!((port.kappa(0)[1] - 1.0).abs() < 1e-12); // X = 1 only as a = 1
    /// ```
    pub fn from_independent(units: Vec<String>, grids: &[Grid]) -> Result<Self> {
        let m = units.len();
        if m == 0 || grids.len() != m {
            return Err(Error::Data("give one grid per unit".into()));
        }
        let step = grids[0].step();
        if grids.iter().any(|g| g.step() != step) {
            return Err(Error::Data("the units' grids must share one step".into()));
        }
        let span: usize = grids.iter().map(|g| g.len() - 1).sum::<usize>() + 1;
        let len = span.next_power_of_two();
        let mut planner = FftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(len);
        let inverse = planner.plan_fft_inverse(len);
        let transform = |v: &mut Vec<Complex64>| forward.process(v);
        let pad = |values: &mut dyn Iterator<Item = f64>| {
            let mut v: Vec<Complex64> = values.map(|x| Complex64::new(x, 0.0)).collect();
            v.resize(len, Complex64::new(0.0, 0.0));
            v
        };
        let mut ft: Vec<Vec<Complex64>> = grids
            .iter()
            .map(|g| {
                let mut v = pad(&mut g.probs().iter().copied());
                transform(&mut v);
                v
            })
            .collect();
        // Products of the other units' transforms, from prefix and suffix
        // products (no division, so zeros in a transform are harmless).
        let one = vec![Complex64::new(1.0, 0.0); len];
        let mut prefix = vec![one.clone()];
        for f in &ft {
            let last = prefix.last().expect("starts with one");
            prefix.push(last.iter().zip(f).map(|(a, b)| a * b).collect());
        }
        let mut suffix = vec![one; m + 1];
        for i in (0..m).rev() {
            suffix[i] = suffix[i + 1]
                .iter()
                .zip(&ft[i])
                .map(|(a, b)| a * b)
                .collect();
        }
        let scale = 1.0 / len as f64;
        let back = |mut v: Vec<Complex64>| -> Vec<f64> {
            inverse.process(&mut v);
            v.iter().map(|c| c.re * scale).collect()
        };
        let total = back(prefix[m].clone());
        let mut weighted = Vec::with_capacity(m);
        for (i, g) in grids.iter().enumerate() {
            let mut xp = pad(&mut (0..g.len()).map(|j| g.x(j) * g.probs()[j]));
            transform(&mut xp);
            let others: Vec<Complex64> = prefix[i]
                .iter()
                .zip(&suffix[i + 1])
                .map(|(a, b)| a * b)
                .collect();
            weighted.push(back(xp.iter().zip(&others).map(|(a, b)| a * b).collect()));
        }
        ft.clear();
        let (mut x, mut p, mut kappa) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..span {
            if total[k] >= 1e-14 {
                x.push(k as f64 * step);
                p.push(total[k]);
                kappa.extend(weighted.iter().map(|w| w[k] / total[k]));
            }
        }
        Ok(Self::build(units, x, p, kappa))
    }

    fn build(units: Vec<String>, x: Vec<f64>, p: Vec<f64>, kappa: Vec<f64>) -> Self {
        let mut s = vec![0.0; x.len()];
        let mut above = 0.0;
        for k in (0..x.len()).rev() {
            s[k] = above;
            above += p[k];
        }
        Self {
            units,
            x,
            p,
            kappa,
            s,
        }
    }

    /// Unit names.
    pub fn units(&self) -> &[String] {
        &self.units
    }

    /// The distinct totals, ascending.
    pub fn totals(&self) -> &[f64] {
        &self.x
    }

    /// The probability of each total.
    pub fn probs(&self) -> &[f64] {
        &self.p
    }

    /// Unit `i`'s conditional expectation at each total, `κᵢ(x_k)`.
    pub fn kappa(&self, i: usize) -> Vec<f64> {
        let m = self.units.len();
        (0..self.x.len()).map(|k| self.kappa[k * m + i]).collect()
    }

    /// Expected loss of each unit, `E[Xᵢ]`.
    pub fn expected(&self) -> Vec<f64> {
        let m = self.units.len();
        (0..m)
            .map(|i| {
                (0..self.x.len())
                    .map(|k| self.kappa[k * m + i] * self.p[k])
                    .sum()
            })
            .collect()
    }

    /// The largest total.
    pub fn max(&self) -> f64 {
        self.x[self.x.len() - 1]
    }

    /// Assets at the capital standard `p`: the lower `p` quantile of the
    /// total, the smallest `x` with `P(X <= x) >= p`. `p = 1` gives the
    /// largest total.
    pub fn assets(&self, p: f64) -> Result<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(Error::InvalidParameter {
                name: "p",
                value: p,
                reason: "must be in [0, 1]",
            });
        }
        let mut cum = 0.0;
        for (k, &pk) in self.p.iter().enumerate() {
            cum += pk;
            if cum >= p {
                return Ok(self.x[k]);
            }
        }
        Ok(self.max())
    }

    /// The number of totals at or below `a`.
    fn count_at_or_below(&self, a: f64) -> usize {
        self.x.partition_point(|&x| x <= a)
    }

    /// `P(X > a)`.
    fn survival(&self, a: f64) -> f64 {
        match self.count_at_or_below(a) {
            0 => 1.0,
            c => self.s[c - 1],
        }
    }

    /// Suffix sums `Σ_{k >= c} (κᵢ / x)_k w_k` for `c = 0..=n`, unit-minor:
    /// the share of the totals from `x_c` up, weighted by `w`.
    fn suffix_shares(&self, w: &[f64]) -> Vec<f64> {
        let (n, m) = (self.x.len(), self.units.len());
        let mut out = vec![0.0; (n + 1) * m];
        for k in (0..n).rev() {
            for i in 0..m {
                let share = if self.x[k] > 0.0 {
                    self.kappa[k * m + i] / self.x[k]
                } else {
                    0.0
                };
                out[k * m + i] = out[(k + 1) * m + i] + share * w[k];
            }
        }
        out
    }

    /// Prefix sums `Σ_{k < c} κᵢ(x_k) w_k` for `c = 0..=n`, unit-minor.
    fn prefix_kappa(&self, w: &[f64]) -> Vec<f64> {
        let (n, m) = (self.x.len(), self.units.len());
        let mut out = vec![0.0; (n + 1) * m];
        for k in 0..n {
            for i in 0..m {
                out[(k + 1) * m + i] = out[k * m + i] + self.kappa[k * m + i] * w[k];
            }
        }
        out
    }

    /// Expected loss paid with assets `a`, `E[X ∧ a]`, and each unit's
    /// equal-priority recovery `E[Xᵢ min(1, a / X)]`.
    pub fn limited_expected(&self, a: f64) -> (f64, Vec<f64>) {
        let m = self.units.len();
        let c = self.count_at_or_below(a);
        let total = a * self.survival(a) + (0..c).map(|k| self.x[k] * self.p[k]).sum::<f64>();
        let tail = self.suffix_shares(&self.p);
        let head = self.prefix_kappa(&self.p);
        let units = (0..m)
            .map(|i| head[c * m + i] + a * tail[c * m + i])
            .collect();
        (total, units)
    }

    /// The expected policyholder deficit ratio with assets `a`:
    /// `(E[X] - E[X ∧ a]) / E[X]` in total, and by unit under equal
    /// priority, `(E[Xᵢ] - E[Xᵢ min(1, a / X)]) / E[Xᵢ]` (0 for a unit with
    /// no expected loss).
    pub fn epd(&self, a: f64) -> (f64, Vec<f64>) {
        let full = self.expected();
        let total_full: f64 = full.iter().sum();
        let (paid, units) = self.limited_expected(a);
        let ratio = |e: f64, l: f64| if e > 0.0 { (e - l) / e } else { 0.0 };
        (
            ratio(total_full, paid),
            full.iter()
                .zip(&units)
                .map(|(&e, &l)| ratio(e, l))
                .collect(),
        )
    }

    /// The smallest assets whose total EPD ratio is at most `epd`.
    pub fn assets_for_epd(&self, epd: f64) -> Result<f64> {
        if !(epd > 0.0 && epd < 1.0) {
            return Err(Error::InvalidParameter {
                name: "epd",
                value: epd,
                reason: "must be in (0, 1)",
            });
        }
        let max = self.max();
        Ok(prospicio_math::roots::bisect(0.0, max, |a| {
            self.epd(a).0 > epd
        }))
    }

    /// The premium of `X ∧ a` under `g`, and its allocation to units; see
    /// the [module](self). `a` must be positive.
    ///
    /// ```
    /// use prospicio_pricing::natural::{Allocation, Portfolio};
    /// use prospicio_prob::Distortion;
    ///
    /// // Monograph 15's InsCo, priced at a 15% cost of capital.
    /// let rows = [
    ///     [15.0, 7.0, 0.0], [15.0, 13.0, 0.0], [5.0, 20.0, 11.0], [7.0, 33.0, 0.0],
    ///     [13.0, 20.0, 7.0], [5.0, 27.0, 8.0], [15.0, 16.0, 9.0], [26.0, 19.0, 10.0],
    ///     [17.0, 8.0, 40.0], [16.0, 20.0, 64.0],
    /// ];
    /// let rows: Vec<Vec<f64>> = rows.iter().map(|r| r.to_vec()).collect();
    /// let port = Portfolio::from_rows(vec!["A".into(), "B".into(), "C".into()], &rows, None).unwrap();
    /// let price = port.price(&Distortion::ccoc(0.15).unwrap(), 100.0, Allocation::Linear).unwrap();
    /// assert!((price.total.premium - 53.565217391304344).abs() < 1e-12);
    /// // Unit C carries most of the margin: it drives the worst years.
    /// assert!((price.allocated[2].premium - 21.304347826086957).abs() < 1e-12);
    /// ```
    pub fn price(&self, g: &Distortion, a: f64, method: Allocation) -> Result<NaturalPrice> {
        if !(a.is_finite() && a > 0.0) {
            return Err(Error::InvalidParameter {
                name: "assets",
                value: a,
                reason: "must be finite and positive",
            });
        }
        let n = self.x.len();
        let m = self.units.len();
        let c = self.count_at_or_below(a);
        let gs: Vec<f64> = self.s.iter().map(|&s| g.g(s)).collect();
        let gp: Vec<f64> = (0..n)
            .map(|k| {
                let above = if k == 0 { 1.0 } else { gs[k - 1] };
                (above - gs[k]).max(0.0)
            })
            .collect();
        let gs_a = if c == 0 { 1.0 } else { gs[c - 1] };
        // Premiums.
        let premium = a * gs_a + self.x[..c].iter().zip(&gp).map(|(x, g)| x * g).sum::<f64>();
        let tail_p = self.suffix_shares(&self.p);
        let tail_g = self.suffix_shares(&gp);
        let head_p = self.prefix_kappa(&self.p);
        let head_g = self.prefix_kappa(&gp);
        // Unit i's premium and expected loss with assets y, where `c` totals
        // lie at or below y: `exag` and `exa` in aggregate's terms.
        let unit_premium = |y: f64, c: usize, i: usize| {
            let tail = match method {
                Allocation::Linear => {
                    let s = if c == 0 { 1.0 } else { self.s[c - 1] };
                    let g = if c == 0 { 1.0 } else { gs[c - 1] };
                    if s > 0.0 {
                        g * tail_p[c * m + i] / s
                    } else {
                        0.0
                    }
                }
                Allocation::Lifted => tail_g[c * m + i],
            };
            head_g[c * m + i] + y * tail
        };
        let unit_loss = |y: f64, c: usize, i: usize| head_p[c * m + i] + y * tail_p[c * m + i];
        let unit_p: Vec<f64> = (0..m).map(|i| unit_premium(a, c, i)).collect();
        let (loss, unit_l) = self.limited_expected(a);
        // Capital by layer: [0, x_0) with S = 1, then [x_k, x_{k+1}), up to
        // a. Each unit's margin over the layer, m(top) - m(bottom), takes
        // the layer's capital per unit of margin, (1 - g) / (g - S); the
        // margin's jump at a total falls in the layer below it.
        let fill = {
            let slope = g.slope_at_one();
            if slope >= 1.0 {
                f64::INFINITY
            } else {
                slope / (1.0 - slope)
            }
        };
        let margin = |y: f64, c: usize, i: usize| unit_premium(y, c, i) - unit_loss(y, c, i);
        let mut unit_q = vec![0.0; m];
        let mut lower = 0.0;
        for layer in 0..=n {
            let upper = if layer < n {
                self.x[layer]
            } else {
                f64::INFINITY
            };
            let top = upper.min(a);
            if top > lower {
                let (s, gsl) = if layer == 0 {
                    (1.0, 1.0)
                } else {
                    (self.s[layer - 1], gs[layer - 1])
                };
                let ratio = if gsl == 1.0 {
                    if s == 1.0 { fill } else { 0.0 }
                } else if gsl == s {
                    0.0
                } else {
                    (1.0 - gsl) / (gsl - s)
                };
                let c_top = self.count_at_or_below(top);
                for (i, q) in unit_q.iter_mut().enumerate() {
                    let dm = margin(top, c_top, i) - margin(lower, layer, i);
                    if dm != 0.0 {
                        *q += dm * ratio;
                    }
                }
            }
            if upper >= a {
                break;
            }
            lower = upper;
        }
        let allocated = (0..m)
            .map(|i| Pentagon {
                loss: unit_l[i],
                margin: unit_p[i] - unit_l[i],
                premium: unit_p[i],
                capital: unit_q[i],
                assets: unit_p[i] + unit_q[i],
            })
            .collect();
        Ok(NaturalPrice {
            units: self.units.clone(),
            allocated,
            total: Pentagon::new(loss, premium, a),
        })
    }

    /// Bodoff's percentile layer of capital with assets `a`: each unit's
    /// share of the assets, `∫₀ᵃ E[Xᵢ / X | X > x] dx`. The shares add up
    /// to `a` when every layer below `a` has a chance of loss.
    pub fn bodoff(&self, a: f64) -> Vec<f64> {
        let m = self.units.len();
        let n = self.x.len();
        let tail = self.suffix_shares(&self.p);
        let mut out = vec![0.0; m];
        let mut lower = 0.0;
        for layer in 0..=n {
            let upper = if layer < n {
                self.x[layer]
            } else {
                f64::INFINITY
            };
            let dx = upper.min(a) - lower;
            if dx > 0.0 {
                let s = if layer == 0 { 1.0 } else { self.s[layer - 1] };
                if s > 0.0 {
                    for (i, o) in out.iter_mut().enumerate() {
                        *o += dx * tail[layer * m + i] / s;
                    }
                }
            }
            if upper >= a {
                break;
            }
            lower = upper;
        }
        out
    }

    /// The member of `family` that prices `X ∧ a` at the target.
    pub fn calibrate(&self, family: Family, a: f64, target: Target) -> Result<Distortion> {
        let (loss, _) = self.limited_expected(a);
        let premium = match target {
            Target::Premium(p) => p,
            Target::ReturnOnCapital(r) => (loss + r * a) / (1.0 + r),
            Target::LossRatio(lr) => loss / lr,
        };
        let capped: Vec<f64> = self.x.iter().map(|x| x.min(a)).collect();
        calibrate(family, &capped, &self.p, premium)
    }
}

/// What a calibrated distortion must reproduce.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Target {
    /// A premium.
    Premium(f64),
    /// A return on capital `r`: `P = (L + r a) / (1 + r)`.
    ReturnOnCapital(f64),
    /// A loss ratio: `P = L / LR`.
    LossRatio(f64),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insco() -> Portfolio {
        let rows = [
            [15.0, 7.0, 0.0],
            [15.0, 13.0, 0.0],
            [5.0, 20.0, 11.0],
            [7.0, 33.0, 0.0],
            [13.0, 20.0, 7.0],
            [5.0, 27.0, 8.0],
            [15.0, 16.0, 9.0],
            [26.0, 19.0, 10.0],
            [17.0, 8.0, 40.0],
            [16.0, 20.0, 64.0],
        ];
        let rows: Vec<Vec<f64>> = rows.iter().map(|r| r.to_vec()).collect();
        Portfolio::from_rows(vec!["A".into(), "B".into(), "C".into()], &rows, None).unwrap()
    }

    #[test]
    fn allocations_add_up() {
        let port = insco();
        for g in [
            Distortion::ccoc(0.15).unwrap(),
            Distortion::proportional_hazard(0.72).unwrap(),
            Distortion::wang(0.34).unwrap(),
            Distortion::tvar(0.27).unwrap(),
        ] {
            for a in [100.0, 65.0, 50.0, 40.0, 30.0, 10.0] {
                for method in [Allocation::Linear, Allocation::Lifted] {
                    let p = port.price(&g, a, method).unwrap();
                    let sum = |f: fn(&Pentagon) -> f64| p.allocated.iter().map(f).sum::<f64>();
                    let t = p.total;
                    assert!(
                        (sum(|u| u.loss) - t.loss).abs() < 1e-10,
                        "{g:?} {a} {method:?}"
                    );
                    assert!(
                        (sum(|u| u.premium) - t.premium).abs() < 1e-10,
                        "{g:?} {a} {method:?}"
                    );
                    assert!(
                        (sum(|u| u.capital) - t.capital).abs() < 1e-9,
                        "{g:?} {a} {method:?}: {} vs {}",
                        sum(|u| u.capital),
                        t.capital
                    );
                }
            }
        }
    }

    #[test]
    fn independent_units_match_brute_force() {
        let a = Grid::new(
            1.0,
            vec![0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.25],
        )
        .unwrap();
        let mut bp = vec![0.0; 91];
        bp[0] = 0.5;
        bp[1] = 0.25;
        bp[90] = 0.25;
        let b = Grid::new(1.0, bp).unwrap();
        let c = Grid::new(1.0, vec![0.2, 0.3, 0.5]).unwrap();
        let port = Portfolio::from_independent(
            vec!["a".into(), "b".into(), "c".into()],
            &[a.clone(), b.clone(), c.clone()],
        )
        .unwrap();
        // Every combination, by hand.
        let mut rows = Vec::new();
        let mut probs = Vec::new();
        for (i, pa) in a.probs().iter().enumerate() {
            for (j, pb) in b.probs().iter().enumerate() {
                for (k, pc) in c.probs().iter().enumerate() {
                    if pa * pb * pc > 0.0 {
                        rows.push(vec![i as f64, j as f64, k as f64]);
                        probs.push(pa * pb * pc);
                    }
                }
            }
        }
        let brute = Portfolio::from_rows(
            vec!["a".into(), "b".into(), "c".into()],
            &rows,
            Some(&probs),
        )
        .unwrap();
        assert_eq!(port.totals(), brute.totals());
        for (x, y) in port.probs().iter().zip(brute.probs()) {
            assert!((x - y).abs() < 1e-15);
        }
        for i in 0..3 {
            for (x, y) in port.kappa(i).iter().zip(brute.kappa(i)) {
                assert!((x - y).abs() < 1e-12, "{x} vs {y}");
            }
        }
    }

    #[test]
    fn epd_and_bodoff() {
        let port = insco();
        let (total, units) = port.epd(100.0);
        assert_eq!(total, 0.0);
        assert!(units.iter().all(|u| u.abs() < 1e-15));
        // With assets 65 only the 100 year defaults, by 35 with p 0.1.
        let (total, _) = port.epd(65.0);
        assert!((total - 3.5 / 46.6).abs() < 1e-15);
        let a = port.assets_for_epd(3.5 / 46.6).unwrap();
        assert!((a - 65.0).abs() < 1e-9);
        let b = port.bodoff(100.0);
        assert!((b.iter().sum::<f64>() - 100.0).abs() < 1e-10);
        assert_eq!(port.assets(1.0).unwrap(), 100.0);
        assert_eq!(port.assets(0.85).unwrap(), 65.0);
        assert_eq!(port.assets(0.7).unwrap(), 40.0);
    }

    #[test]
    fn pentagon_solves_any_determined_triple() {
        use Quantity::*;
        let want = Pentagon::new(46.6, 53.565217391304344, 100.0);
        let all = [
            (Loss, want.loss),
            (Margin, want.margin),
            (Premium, want.premium),
            (Capital, want.capital),
            (Assets, want.assets),
            (LossRatio, want.loss_ratio()),
            (PremiumToCapital, want.premium_to_capital()),
            (ReturnOnCapital, want.return_on_capital()),
        ];
        let mut solved = 0;
        for i in 0..8 {
            for j in i + 1..8 {
                for k in j + 1..8 {
                    if let Ok(p) = Pentagon::solve([all[i], all[j], all[k]]) {
                        solved += 1;
                        for (got, w) in [
                            (p.loss, want.loss),
                            (p.premium, want.premium),
                            (p.assets, want.assets),
                        ] {
                            assert!((got - w).abs() < 1e-9 * w, "{:?}", [all[i], all[j], all[k]]);
                        }
                    }
                }
            }
        }
        // 46 of the 56 triples determine the pentagon. The rest are three
        // ratios, or a relation given twice: {L, M, P}, {P, Q, a},
        // {L, P, LR}, {P, Q, PQ}, {M, Q, ι}, and the like.
        assert_eq!(solved, 46);
        assert!(Pentagon::solve([(Loss, 1.0), (Margin, 1.0), (Premium, 2.0)]).is_err());
    }

    #[test]
    fn calibration_targets() {
        let port = insco();
        let ccoc = port
            .calibrate(Family::Ccoc, 100.0, Target::ReturnOnCapital(0.15))
            .unwrap();
        assert!(matches!(ccoc, Distortion::Ccoc(r) if (r - 0.15).abs() < 1e-12));
        let ph = port
            .calibrate(Family::ProportionalHazard, 65.0, Target::LossRatio(0.9))
            .unwrap();
        let p = port.price(&ph, 65.0, Allocation::Linear).unwrap();
        assert!((p.total.loss_ratio() - 0.9).abs() < 1e-9);
    }
}
