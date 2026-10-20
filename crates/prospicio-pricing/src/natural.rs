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
//! - [`Portfolio::priority`]: the priority ladder, each unit's recovery
//!   when it ranks ahead of the others, pro rata or behind them, and
//!   [`Portfolio::priority_kappa`], where the shortfall falls given the
//!   total. These need more than `κ`: the portfolio keeps the scenarios or
//!   grids it was built from.
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

/// Capital in the bottom layer of the assets, from 0 to the smallest total,
/// where `S = g(S) = 1` and the layer's return is undefined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BottomLayer {
    /// The units' margins there take capital at the limit of the layer
    /// above, `g'(1) / (1 − g'(1))`, as `aggregate` 1.0.1 and 1.2.0 do.
    #[default]
    Limit,
    /// No capital, Monograph 15's equation 8.5 as printed.
    Zero,
}

/// How the linear allocation treats a total exactly at the assets, when it
/// has positive probability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AtomAtAssets {
    /// The outcome keeps its own weight, `κᵢ(a) Δg`, and `a g(S(a))` is
    /// shared over the totals strictly above `a`, as `aggregate` does.
    #[default]
    Separate,
    /// The outcome is lumped with the default outcomes, which share
    /// `a g(S(a⁻))` by their expected shares: *Pricing Insurance Risk*'s
    /// linear allocation, which conditions on `X ∧ a`.
    Lumped,
}

/// The settings of [`Portfolio::price_with`]; the default is
/// [`Portfolio::price`] with the linear allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PriceSettings {
    pub allocation: Allocation,
    pub bottom_layer: BottomLayer,
    /// Used by the linear allocation only: the lifted one weights the
    /// outcome at the assets the same either way.
    pub atom: AtomAtAssets,
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
    /// What `κ` forgets and the priority ladder needs.
    joint: Joint,
}

/// The units' joint law, kept for [`Portfolio::priority`]: the senior and
/// junior recoveries are not functions of `κ`.
#[derive(Debug, Clone, PartialEq)]
enum Joint {
    /// Unit losses scenario-major, and each scenario's probability.
    Rows { losses: Vec<f64>, w: Vec<f64> },
    /// Each unit's grid, and the density of the other units' total on the
    /// same step.
    Independent {
        grids: Vec<Grid>,
        others: Vec<Vec<f64>>,
    },
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
        let (mut losses, mut kept) = (Vec::new(), Vec::new());
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
                losses.extend_from_slice(row);
                kept.push(w);
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
        let joint = Joint::Rows { losses, w: kept };
        Ok(Self::build(units, x, p, kappa, joint))
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
        let mut leave_out = Vec::with_capacity(m);
        for (i, g) in grids.iter().enumerate() {
            let mut xp = pad(&mut (0..g.len()).map(|j| g.x(j) * g.probs()[j]));
            transform(&mut xp);
            let others: Vec<Complex64> = prefix[i]
                .iter()
                .zip(&suffix[i + 1])
                .map(|(a, b)| a * b)
                .collect();
            weighted.push(back(xp.iter().zip(&others).map(|(a, b)| a * b).collect()));
            let mut density = back(others);
            density.truncate(span);
            density.iter_mut().for_each(|v| *v = v.max(0.0));
            leave_out.push(density);
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
        let joint = Joint::Independent {
            grids: grids.to_vec(),
            others: leave_out,
        };
        Ok(Self::build(units, x, p, kappa, joint))
    }

    fn build(units: Vec<String>, x: Vec<f64>, p: Vec<f64>, kappa: Vec<f64>, joint: Joint) -> Self {
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
            joint,
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

    /// The level whose tail value at risk of the total is its value at risk
    /// at `p`, `aggregate`'s `Portfolio.tvar_threshold(p, 'lower')`; see
    /// [`prospicio_prob::risk::tvar_threshold`].
    pub fn tvar_threshold(&self, p: f64) -> Result<f64> {
        prospicio_prob::risk::tvar_threshold(&self.x, &self.p, p)
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
        // Clamped at 0: the paid loss can round a last bit above the full.
        let ratio = |e: f64, l: f64| if e > 0.0 { ((e - l) / e).max(0.0) } else { 0.0 };
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

    /// Each unit's recovery from assets `a` under each priority rule: the
    /// priority ladder of `aggregate`'s `priority_epd_df`. The insurer pays
    /// `X ∧ a` under every rule; the rules only split it.
    ///
    /// - Senior: unit `i` ranks ahead of the others pooled, and recovers
    ///   `E[Xᵢ ∧ a]`.
    /// - Equal: `E[Xᵢ min(1, a / X)]`, as in [`limited_expected`](Self::limited_expected).
    /// - Junior: the others pooled rank ahead of unit `i`, which recovers
    ///   `E[min(Xᵢ, (a - X₋ᵢ)⁺)]`.
    ///
    /// Each is a two-tier reading: unit `i` against the rest pooled. On a
    /// book of three or more units the senior (or junior) recoveries are
    /// different hypotheticals and do not add up to `E[X ∧ a]`; the equal
    /// ones do. On two units, `senior₁ + junior₂ = E[X ∧ a]` exactly.
    ///
    /// From scenarios the ladder is exact under any dependence between the
    /// units. From independent grids it is `E[Xᵢ ∧ a]` on unit `i`'s grid
    /// and `Σ_y p₋ᵢ(y) E[Xᵢ ∧ (a - y)]` for the junior leg, with `p₋ᵢ` the
    /// other units' total; `E[Xᵢ ∧ t]` is linear in `t` between grid points,
    /// so `a` need not be one. `aggregate` needs the units independent.
    ///
    /// ```
    /// use prospicio_pricing::natural::{Portfolio, Rule};
    /// use prospicio_prob::Grid;
    ///
    /// // Pricing Insurance Risk's Discrete case: X1 0, 8 or 10 and X2 0, 1
    /// // or 90, each with probabilities 1/2, 1/4, 1/4, independent.
    /// let x1 = Grid::new(1.0, [vec![0.5], vec![0.0; 7], vec![0.25, 0.0, 0.25]].concat()).unwrap();
    /// let mut p2 = vec![0.0; 91];
    /// (p2[0], p2[1], p2[90]) = (0.5, 0.25, 0.25);
    /// let x2 = Grid::new(1.0, p2).unwrap();
    /// let port = Portfolio::from_independent(vec!["X1".into(), "X2".into()], &[x1, x2]).unwrap();
    /// let ladder = port.priority(90.0).unwrap();
    /// // Senior, X1 loses nothing; junior, it loses 1.125 of its 4.5.
    /// assert!(ladder.epd(Rule::Senior)[0].abs() < 1e-12);
    /// assert!((ladder.epd(Rule::Junior)[0] - 0.25).abs() < 1e-12);
    /// ```
    pub fn priority(&self, a: f64) -> Result<Priority> {
        if !(a.is_finite() && a >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "assets",
                value: a,
                reason: "must be finite and non-negative",
            });
        }
        let m = self.units.len();
        let (total, equal) = self.limited_expected(a);
        let (mut senior, mut junior) = (vec![0.0; m], vec![0.0; m]);
        match &self.joint {
            Joint::Rows { losses, w } => {
                for (row, &w) in losses.chunks_exact(m).zip(w) {
                    let sum: f64 = row.iter().sum();
                    for (i, &xi) in row.iter().enumerate() {
                        senior[i] += w * xi.min(a);
                        junior[i] += w * xi.min((a - (sum - xi)).max(0.0));
                    }
                }
            }
            Joint::Independent { grids, others } => {
                for (i, (g, other)) in grids.iter().zip(others).enumerate() {
                    let lev = LimitedOnGrid::new(g);
                    senior[i] = lev.at(a);
                    let h = g.step();
                    junior[i] = other
                        .iter()
                        .enumerate()
                        .take_while(|&(k, _)| (k as f64) * h < a)
                        .map(|(k, &p)| p * lev.at(a - k as f64 * h))
                        .sum();
                }
            }
        }
        Ok(Priority {
            units: self.units.clone(),
            assets: a,
            mean: self.expected(),
            senior,
            equal,
            junior,
            total,
        })
    }

    /// Each unit's comonotonic allocation, a non-decreasing function of the
    /// total `f̃ᵢ(x)` at each of [`totals`](Self::totals), by Denuit et
    /// al.'s convex-order improvement of the conditional mean risk sharing
    /// `κᵢ` (Denuit, Dhaene, Ghossoub and Robert, "Comonotonicity and
    /// Pareto optimality, with application to collaborative insurance",
    /// *Insurance: Mathematics and Economics* 120, 2025, Theorem 3.1).
    /// `aggregate`'s `make_comonotonic_allocations`, ported step for step:
    /// sweeping up the totals, each unit whose allocation falls is
    /// flattened back to the probability-weighted mean of the run it must
    /// pool with, and what it gives up goes to the units that rose, in
    /// proportion to their rise. The result depends on that order, so it
    /// is kept. The allocations still add up to the total; each is
    /// non-decreasing in it.
    ///
    /// ```
    /// use prospicio_pricing::natural::Portfolio;
    ///
    /// // B's share falls as the total rises from 4 to 6.
    /// let rows = [[0.0, 0.0], [1.0, 3.0], [6.0, 0.0], [7.0, 3.0]];
    /// let rows: Vec<Vec<f64>> = rows.iter().map(|r| r.to_vec()).collect();
    /// let port = Portfolio::from_rows(vec!["A".into(), "B".into()], &rows, None).unwrap();
    /// let f = port.comonotonic_kappa();
    /// for k in 0..port.totals().len() {
    ///     assert!((f[0][k] + f[1][k] - port.totals()[k]).abs() < 1e-12);
    ///     if k > 0 {
    ///         assert!(f[0][k] >= f[0][k - 1] && f[1][k] >= f[1][k - 1]);
    ///     }
    /// }
    /// ```
    pub fn comonotonic_kappa(&self) -> Vec<Vec<f64>> {
        let (n, m) = (self.x.len(), self.units.len());
        let mut kt: Vec<Vec<f64>> = (0..m).map(|i| self.kappa(i)).collect();
        let pdf = &self.p;
        for k in 1..n {
            let diffs: Vec<f64> = (0..m).map(|i| kt[i][k] - kt[i][k - 1]).collect();
            let violators: Vec<usize> = (0..m).filter(|&i| diffs[i] < 0.0).collect();
            if violators.is_empty() {
                continue;
            }
            let rising: Vec<usize> = (0..m).filter(|&i| diffs[i] >= 0.0).collect();
            for &i in &violators {
                // Scan back for the run to pool with: the integral average
                // must not lie below the value before it.
                let mut p = k as isize - 1;
                let mut mass = pdf[k];
                let mut weighted = kt[i][k] * mass;
                while p >= 0 && {
                    let avg = if mass > 0.0 {
                        weighted / mass
                    } else {
                        kt[i][k]
                    };
                    kt[i][p as usize] > avg
                } {
                    weighted += kt[i][p as usize] * pdf[p as usize];
                    mass += pdf[p as usize];
                    p -= 1;
                }
                let start = (p + 1) as usize;
                let lambda = if mass > 0.0 {
                    weighted / mass
                } else {
                    kt[i][k]
                };
                let delta: Vec<f64> = (start..=k).map(|j| kt[i][j] - lambda).collect();
                kt[i][start..=k].fill(lambda);
                if !rising.is_empty() {
                    let total: f64 = rising.iter().map(|&j| diffs[j]).sum();
                    for &j in &rising {
                        let alpha = if total > 0.0 {
                            diffs[j] / total
                        } else {
                            1.0 / rising.len() as f64
                        };
                        for (d, v) in delta.iter().zip(&mut kt[j][start..=k]) {
                            *v += d * alpha;
                        }
                    }
                }
            }
        }
        kt
    }

    /// Each unit's loss distribution on its own: its distinct losses,
    /// ascending, and their probabilities.
    fn marginal(&self, i: usize) -> (Vec<f64>, Vec<f64>) {
        let mut pts: Vec<(f64, f64)> = match &self.joint {
            Joint::Rows { losses, w } => {
                let m = self.units.len();
                losses
                    .chunks_exact(m)
                    .map(|r| r[i])
                    .zip(w.iter().copied())
                    .collect()
            }
            Joint::Independent { grids, .. } => {
                let g = &grids[i];
                (0..g.len())
                    .map(|j| (g.x(j), g.probs()[j]))
                    .filter(|&(_, w)| w > 0.0)
                    .collect()
            }
        };
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (mut x, mut w): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        for (xi, wi) in pts {
            if x.last() == Some(&xi) {
                *w.last_mut().unwrap() += wi;
            } else {
                x.push(xi);
                w.push(wi);
            }
        }
        (x, w)
    }

    /// Each unit priced as if it were the whole book, against the book
    /// priced together, `aggregate`'s `Portfolio.price_stand_alone`. A unit
    /// alone holds assets at its own lower `p` quantile, `aᵢ = VaR_p(Xᵢ)`,
    /// and its premium is `ρ_g(Xᵢ ∧ aᵢ)` with expected loss `E[Xᵢ ∧ aᵢ]`;
    /// the book holds `a = VaR_p(X)` and is priced as
    /// [`price`](Self::price) prices it. `sum` adds the stand-alone loss,
    /// premium and assets; its excess over `total` is the diversification
    /// benefit.
    ///
    /// ```
    /// use prospicio_pricing::natural::Portfolio;
    /// use prospicio_prob::Distortion;
    ///
    /// let rows = [[0.0, 0.0], [1.0, 4.0], [4.0, 1.0], [5.0, 5.0]];
    /// let rows: Vec<Vec<f64>> = rows.iter().map(|r| r.to_vec()).collect();
    /// let port = Portfolio::from_rows(vec!["a".into(), "b".into()], &rows, None).unwrap();
    /// let s = port.price_stand_alone(&Distortion::proportional_hazard(0.5).unwrap(), 0.75).unwrap();
    /// // Alone, each unit needs its own 75% VaR, 4; together the book needs 5.
    /// assert_eq!((s.stand_alone[0].assets, s.stand_alone[1].assets), (4.0, 4.0));
    /// assert_eq!(s.total.assets, 5.0);
    /// assert!(s.sum.assets >= s.total.assets);
    /// ```
    pub fn price_stand_alone(&self, g: &Distortion, p: f64) -> Result<StandAlone> {
        if !(p > 0.0 && p < 1.0) {
            return Err(Error::InvalidParameter {
                name: "p",
                value: p,
                reason: "must be in (0, 1)",
            });
        }
        let at = |port: &Portfolio| -> Result<Pentagon> {
            let a = port.assets(p)?;
            if a > 0.0 {
                Ok(port.price(g, a, Allocation::Linear)?.total)
            } else {
                Ok(Pentagon::new(0.0, 0.0, 0.0))
            }
        };
        let stand_alone = (0..self.units.len())
            .map(|i| {
                let (x, w) = self.marginal(i);
                let rows: Vec<Vec<f64>> = x.iter().map(|&v| vec![v]).collect();
                at(&Portfolio::from_rows(
                    vec![self.units[i].clone()],
                    &rows,
                    Some(&w),
                )?)
            })
            .collect::<Result<Vec<_>>>()?;
        let total = at(self)?;
        let add = |f: fn(&Pentagon) -> f64| stand_alone.iter().map(f).sum::<f64>();
        let sum = Pentagon::new(add(|t| t.loss), add(|t| t.premium), add(|t| t.assets));
        Ok(StandAlone {
            units: self.units.clone(),
            stand_alone,
            total,
            sum,
        })
    }

    /// Where in the total the shortfall `(X - a)⁺` falls when unit `i` is
    /// junior to the rest pooled: at each total `x` in
    /// [`totals`](Self::totals), the junior unit's expected part
    /// `E[min((X - a)⁺, Xᵢ) | X = x]` and the senior pool's,
    /// `E[((X - a)⁺ - Xᵢ)⁺ | X = x]`. They add up to `(x - a)⁺`: the junior
    /// unit absorbs the default first, up to its own loss. `aggregate`'s
    /// `priority_kappa`. From independent grids, two convolutions by FFT:
    /// with `y` the others' total, `min(x - a, x - y)` is `x - a` when
    /// `y <= a` and `x - y` otherwise.
    pub fn priority_kappa(&self, i: usize, a: f64) -> Result<(Vec<f64>, Vec<f64>)> {
        let m = self.units.len();
        if i >= m {
            return Err(Error::Data(format!("no unit {i}: the portfolio has {m}")));
        }
        if !(a.is_finite() && a >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "assets",
                value: a,
                reason: "must be finite and non-negative",
            });
        }
        let n = self.x.len();
        let mut junior = vec![0.0; n];
        match &self.joint {
            Joint::Rows { losses, w } => {
                for (row, &w) in losses.chunks_exact(m).zip(w) {
                    let sum: f64 = row.iter().sum();
                    let k = self.x.partition_point(|&x| x < sum);
                    junior[k] += w * (sum - a).max(0.0).min(row[i]);
                }
                for (j, p) in junior.iter_mut().zip(&self.p) {
                    *j /= p;
                }
            }
            Joint::Independent { grids, others } => {
                let (g, other) = (&grids[i], &others[i]);
                let h = g.step();
                let below: Vec<f64> = other
                    .iter()
                    .enumerate()
                    .map(|(k, &p)| if k as f64 * h <= a { p } else { 0.0 })
                    .collect();
                let above: Vec<f64> = other.iter().zip(&below).map(|(p, b)| p - b).collect();
                let weighted: Vec<f64> = (0..g.len()).map(|j| g.x(j) * g.probs()[j]).collect();
                let at_a = convolve(g.probs(), &below);
                let past_a = convolve(&weighted, &above);
                for (k, (&x, &p)) in self.x.iter().zip(&self.p).enumerate() {
                    if x > a {
                        let at = (x / h).round() as usize;
                        let v = ((x - a) * at_a[at] + past_a[at]) / p;
                        junior[k] = v.clamp(0.0, x - a);
                    }
                }
            }
        }
        let senior = self
            .x
            .iter()
            .zip(&junior)
            .map(|(&x, &j)| ((x - a).max(0.0) - j).max(0.0))
            .collect();
        Ok((junior, senior))
    }

    /// The premium of `X ∧ a` under `g`, and its allocation to units; see
    /// the [module](self). `a` must be positive.
    ///
    /// Two conventions follow `aggregate` (1.0.1 and 1.2.0 alike) and differ
    /// from the printed formulas of the books on small discrete examples;
    /// [`price_with`](Self::price_with) gives the books' versions:
    ///
    /// - **A total exactly at the assets** (linear allocation). An outcome
    ///   with `X = a` keeps its own weight, `κᵢ(a) Δg`, and `a g(S(a))` is
    ///   shared over the totals strictly above `a`. *Pricing Insurance
    ///   Risk*'s linear allocation conditions on `X ∧ a`, so it lumps that
    ///   outcome with the default ones and they share `g(S(a⁻))`; its
    ///   Example 235 gives 4.3 where this gives 4.16. Assets just below the
    ///   atom (`a − 1e-9`) give nearly the book's numbers, and
    ///   [`AtomAtAssets::Lumped`] gives them exactly. `assets(p)` is always
    ///   a total, so the case arises whenever a capital standard is used on
    ///   a discrete book.
    /// - **The bottom layer of capital**, from 0 to the smallest total,
    ///   where `S = g(S) = 1` and the return `(g − S)/(1 − g)` is
    ///   undefined. The units' margins there are not zero, though they sum
    ///   to zero, and they take capital at the limit of the layer above,
    ///   `g'(1) / (1 − g'(1))`: `1/r` for the constant cost of capital `r`,
    ///   `ρ/(1 − ρ)` for the proportional hazard, and 0 (no capital) for a
    ///   distortion with no slope at 1 (Wang, dual power, TVaR).
    ///   Monograph 15's equation 8.5 gives that layer no capital for every
    ///   distortion, so its CCoC and PH capital by unit differs (InsCo at
    ///   15%: 2.261, 1.478, 42.696 here against 5.419, 6.082, 34.934 by
    ///   equation 8.5, which [`BottomLayer::Zero`] gives); the total
    ///   capital, and every unit's premium, are the same.
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
        self.price_with(
            g,
            a,
            PriceSettings {
                allocation: method,
                ..PriceSettings::default()
            },
        )
    }

    /// [`price`](Self::price) with the books' conventions available: the
    /// bottom layer's capital and, for the linear allocation, the outcome
    /// at the assets lumped with the defaults.
    ///
    /// ```
    /// use prospicio_pricing::natural::{AtomAtAssets, Portfolio, PriceSettings};
    /// use prospicio_prob::Distortion;
    ///
    /// // Pricing Insurance Risk, Example 235: TVaR at 80% with assets 10.
    /// let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 12.0, 10.0];
    /// let x2 = [1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 8.0, 12.0, 13.0, 20.0];
    /// let rows: Vec<Vec<f64>> = x1.iter().zip(&x2).map(|(a, b)| vec![*a, *b]).collect();
    /// let port = Portfolio::from_rows(vec!["X1".into(), "X2".into()], &rows, None).unwrap();
    /// let book = PriceSettings { atom: AtomAtAssets::Lumped, ..PriceSettings::default() };
    /// let price = port.price_with(&Distortion::tvar(0.8).unwrap(), 10.0, book).unwrap();
    /// assert!((price.allocated[0].premium - 4.3).abs() < 1e-12);
    /// ```
    pub fn price_with(
        &self,
        g: &Distortion,
        a: f64,
        settings: PriceSettings,
    ) -> Result<NaturalPrice> {
        let method = settings.allocation;
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
        // The number of totals the linear allocation pays in full at the
        // assets: one fewer when the outcome at `a` is lumped with defaults.
        let lumped = method == Allocation::Linear
            && settings.atom == AtomAtAssets::Lumped
            && c > 0
            && self.x[c - 1] == a;
        let c_paid = if lumped { c - 1 } else { c };
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
        let unit_p: Vec<f64> = (0..m).map(|i| unit_premium(a, c_paid, i)).collect();
        let (loss, unit_l) = self.limited_expected(a);
        // Capital by layer: [0, x_0) with S = 1, then [x_k, x_{k+1}), up to
        // a. Each unit's margin over the layer, m(top) - m(bottom), takes
        // the layer's capital per unit of margin, (1 - g) / (g - S); the
        // margin's jump at a total falls in the layer below it.
        let fill = {
            let slope = g.slope_at_one();
            if settings.bottom_layer == BottomLayer::Zero {
                0.0
            } else if slope >= 1.0 {
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
                // (1 - g) / (g - S) as (1 - g) / ((1 - S) - (1 - g)), with
                // 1 - g from 1 - S: near S = 1 the plain quotient is a ratio
                // of two cancelled differences, and a last-bit change of the
                // distortion's parameter moved the capital by unit (#330).
                let s = if layer == 0 { 1.0 } else { self.s[layer - 1] };
                let (t, gc) = (1.0 - s, g.complement(s));
                let ratio = if t == 0.0 {
                    fill
                } else if gc == 0.0 || gc >= t {
                    0.0
                } else {
                    gc / (t - gc)
                };
                let c_top = if top == a {
                    c_paid
                } else {
                    self.count_at_or_below(top)
                };
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

    /// Bodoff's percentile layer of capital priced by a distortion: each
    /// layer of assets `[x, x + dx)` below `a` is shared among the units in
    /// Bodoff's proportions `αᵢ(x) = E[Xᵢ / X | X > x]`, and its cost
    /// `g(S(x)) dx` and capital `(1 - g(S(x))) dx` in the same proportions,
    /// so that every layer earns its own return `(g - S) / (1 - g)` (*Pricing
    /// Insurance Risk*, section 15.3). Unit `i` has loss `∫ αᵢ S dx` (its
    /// equal-priority recovery), premium `∫ αᵢ g(S) dx` and assets
    /// [`Portfolio::bodoff`]; the premiums add up to the price of `X ∧ a`.
    /// Under the CCoC the premium is the loss plus `d = r / (1 + r)` of the
    /// unit's shared liability, assets less loss.
    ///
    /// ```
    /// use prospicio_pricing::natural::Portfolio;
    /// use prospicio_prob::Distortion;
    ///
    /// // Bodoff's Thought Experiment 1 at a 10% return (the book's Table 15.24).
    /// let rows = [vec![0.0, 0.0], vec![99.0, 0.0], vec![0.0, 100.0], vec![99.0, 100.0]];
    /// let units = vec!["Hu".to_string(), "EQ".to_string()];
    /// let port = Portfolio::from_rows(units, &rows, Some(&[0.76, 0.19, 0.04, 0.01])).unwrap();
    /// let plc = port.bodoff_price(&Distortion::ccoc(0.1).unwrap(), 100.0);
    /// assert!((plc.allocated[0].premium - 24.8729).abs() < 5e-5);
    /// assert!((plc.allocated[1].premium - 5.8635).abs() < 5e-5);
    /// ```
    pub fn bodoff_price(&self, g: &Distortion, a: f64) -> NaturalPrice {
        let m = self.units.len();
        let n = self.x.len();
        let tail = self.suffix_shares(&self.p);
        let (mut loss, mut premium, mut assets) = (vec![0.0; m], vec![0.0; m], vec![0.0; m]);
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
                    let gs = g.g(s);
                    for i in 0..m {
                        let share = dx * tail[layer * m + i] / s;
                        assets[i] += share;
                        premium[i] += share * gs;
                        loss[i] += share * s;
                    }
                }
            }
            if upper >= a {
                break;
            }
            lower = upper;
        }
        let pentagon = |loss: f64, premium: f64, assets: f64| Pentagon {
            loss,
            margin: premium - loss,
            premium,
            capital: assets - premium,
            assets,
        };
        let allocated: Vec<Pentagon> = (0..m)
            .map(|i| pentagon(loss[i], premium[i], assets[i]))
            .collect();
        let sum = |v: &[f64]| v.iter().sum::<f64>();
        NaturalPrice {
            units: self.units.clone(),
            allocated,
            total: pentagon(sum(&loss), sum(&premium), sum(&assets)),
        }
    }

    /// The total's and each unit's premium under `TVaR_p` with assets `a`
    /// and the linear allocation, as [`Portfolio::price`] gives them, from
    /// the prefix sums `head` (`Σ κᵢ p`), `tail` (shares above) and `head_x`
    /// (`Σ x p`); `c` totals lie at or below `a`, and `s_a = P(X > a)`.
    #[allow(clippy::too_many_arguments)]
    fn tvar_linear(
        &self,
        p: f64,
        a: f64,
        c: usize,
        s_a: f64,
        head: &[f64],
        tail: &[f64],
        head_x: &[f64],
    ) -> (f64, Vec<f64>) {
        let m = self.units.len();
        let tvar = Distortion::Tvar(p);
        let g = |s: f64| tvar.g(s);
        // g(S_k) = 1 before the first total j with g(S_j) < 1, so Δg is 0
        // there; at j it is 1 - g(S_j), and above j it is p_k / (1 - p).
        let j = self.s.partition_point(|&s| g(s) >= 1.0);
        let gs_a = if c == 0 { 1.0 } else { g(s_a) };
        let tail_weight = if s_a > 0.0 { a * gs_a / s_a } else { 0.0 };
        let mut by_unit: Vec<f64> = (0..m).map(|i| tail_weight * tail[c * m + i]).collect();
        let mut total = a * gs_a;
        if j < c {
            let first = 1.0 - g(self.s[j]);
            let rest = 1.0 / (1.0 - p);
            for (i, u) in by_unit.iter_mut().enumerate() {
                *u += self.kappa[j * m + i] * first
                    + (head[c * m + i] - head[(j + 1) * m + i]) * rest;
            }
            total += self.x[j] * first + (head_x[c] - head_x[j + 1]) * rest;
        }
        (total, by_unit)
    }

    /// The range of each unit's premium over every distortion that prices
    /// `X ∧ a` at `premium`, under the linear allocation: Mildenhall and
    /// Major's pricing bounds (*Pricing Insurance Risk*, chapter 11).
    ///
    /// The distortions with a given price form a convex set whose extreme
    /// points are BiTVaRs: weight `1 - w` on `TVaR_p0` and `w` on
    /// `TVaR_p1`, with `TVaR_p0 <= premium <= TVaR_p1`. A unit's premium is
    /// linear in the distortion, so its range is attained at one of them.
    /// On a discrete total, `TVaR_p` and each unit's share are linear in
    /// `p` between the cumulative probabilities of `X ∧ a`, so the levels
    /// `p0` and `p1` need only range over those knots, 0, and the
    /// level `p*` with `TVaR_p* = premium`: the search is exact. (The top
    /// knot already gives the maximum, so 1 is not needed.) As in
    /// `aggregate`, a level inside the atom at `a` is not a knot: it prices
    /// `X ∧ a` the same as the atom's lower end, though the linear
    /// allocation, which uses `P(X > a)`, can split it differently.
    pub fn premium_bounds(&self, premium: f64, a: f64) -> Result<Vec<PremiumBound>> {
        let capped: Vec<f64> = self.x.iter().map(|x| x.min(a)).collect();
        let (mean, _) = self.limited_expected(a);
        let top = capped[capped.len() - 1];
        if !(premium >= mean && premium <= top) {
            return Err(Error::Data(format!(
                "the premium must lie between E[X ∧ a] = {mean} and {top}, not {premium}"
            )));
        }
        let tvar =
            |p: f64| -> Result<f64> { Ok(Distortion::tvar(p)?.apply_discrete(&capped, &self.p)) };
        let p_star =
            prospicio_math::roots::bisect(0.0, 1.0, |p| tvar(p).is_ok_and(|t| t < premium));
        let mut knots = vec![0.0, p_star];
        // The knots of X ∧ a: cumulative probabilities below the assets.
        let mut cum = 0.0;
        for (&x, &pk) in self.x.iter().zip(&self.p) {
            cum += pk;
            if x < a && cum < 1.0 {
                knots.push(cum);
            }
        }
        knots.sort_by(f64::total_cmp);
        knots.dedup();
        let m = self.units.len();
        // Each unit's premium, and the total's, under TVaR at each knot,
        // from prefix sums: O(m) a knot, not a pricing of the whole book.
        let c = self.count_at_or_below(a);
        let head = self.prefix_kappa(&self.p);
        let tail = self.suffix_shares(&self.p);
        let mut head_x = vec![0.0; self.x.len() + 1];
        for k in 0..self.x.len() {
            head_x[k + 1] = head_x[k] + self.x[k] * self.p[k];
        }
        let s_a = if c == 0 { 1.0 } else { self.s[c - 1] };
        let mut t = Vec::with_capacity(knots.len());
        let mut units = Vec::with_capacity(knots.len());
        for &p in &knots {
            let (total, by_unit) = self.tvar_linear(p, a, c, s_a, &head, &tail, &head_x);
            t.push(total);
            units.push(by_unit);
        }
        let mut best: Vec<PremiumBound> = (0..m)
            .map(|_| PremiumBound {
                lower: f64::INFINITY,
                upper: f64::NEG_INFINITY,
                lower_distortion: Distortion::Tvar(p_star),
                upper_distortion: Distortion::Tvar(p_star),
            })
            .collect();
        let tol = 1e-12 * premium.abs().max(1.0);
        // A unit's premium under the BiTVaR on knots i and j that prices the
        // total at `premium` is the chord from (t_i, u_i) to (t_j, u_j) at
        // `premium`. Its extremes over every pair lie on the upper and lower
        // convex hulls of those points; a knot whose TVaR is the premium
        // (to `tol`) is a candidate on its own.
        let bitvar = |i: usize, j: usize, w: f64| Distortion::BiTvar {
            p0: knots[i],
            p1: knots[j],
            w,
        };
        for (u, b) in best.iter_mut().enumerate() {
            let v: Vec<f64> = units.iter().map(|row| row[u]).collect();
            let mut candidates: Vec<(f64, Distortion)> = (0..knots.len())
                .filter(|&k| (t[k] - premium).abs() <= tol)
                .map(|k| (v[k], bitvar(k, k, 0.0)))
                .collect();
            for upper in [true, false] {
                if let Some((i, j)) = hull_edge(&t, &v, premium, upper) {
                    let w = if t[j] - t[i] > tol {
                        (premium - t[i]) / (t[j] - t[i])
                    } else {
                        0.0
                    };
                    candidates.push(((1.0 - w) * v[i] + w * v[j], bitvar(i, j, w)));
                }
            }
            for (value, g) in candidates {
                if value < b.lower {
                    b.lower = value;
                    b.lower_distortion = g.clone();
                }
                if value > b.upper {
                    b.upper = value;
                    b.upper_distortion = g;
                }
            }
        }
        Ok(best)
    }

    /// The natural allocation set's range for each unit: the smallest and
    /// largest premium of the unit under `g` with assets `a` over every
    /// order of the outcomes whose capped totals `X ∧ a` tie (*Pricing
    /// Insurance Risk*, section 15.1.3). The distortion sees only the
    /// capped total, so tied outcomes may take its weight in any order;
    /// each order is a natural allocation of the same total premium, and
    /// the linear and lifted allocations lie in the range. Since `g` is
    /// concave, a unit's premium is largest when, in every tie, the
    /// outcomes that pay it most take the highest ranks (by the
    /// rearrangement inequality), and smallest in the reverse order.
    ///
    /// Needs the scenarios, so only for a portfolio built from rows.
    ///
    /// ```
    /// use prospicio_pricing::natural::Portfolio;
    /// use prospicio_prob::Distortion;
    ///
    /// // The book's Example 259: the adjusted Simple Discrete Example, where
    /// // the total 10 is 9 + 1 or 10 + 0, under the PH at 0.5.
    /// let (x1, x2, w) = ([0.0, 9.0, 10.0], [0.0, 1.0, 90.0], [0.5, 0.25, 0.25]);
    /// let rows: Vec<Vec<f64>> = x1.iter().flat_map(|&a| x2.map(|b| vec![a, b])).collect();
    /// let probs: Vec<f64> = w.iter().flat_map(|&p| w.map(|q| p * q)).collect();
    /// let port = Portfolio::from_rows(vec!["X1".into(), "X2".into()], &rows, Some(&probs)).unwrap();
    /// let set = port.allocation_set(&Distortion::proportional_hazard(0.5).unwrap(), 100.0).unwrap();
    /// assert!((set[0].lower - 6.2009).abs() < 5e-5 && (set[0].upper - 6.2085).abs() < 5e-5);
    /// ```
    pub fn allocation_set(&self, g: &Distortion, a: f64) -> Result<Vec<AllocationRange>> {
        let Joint::Rows { losses, w } = &self.joint else {
            return Err(Error::Data(
                "the natural allocation set needs the scenarios: build the portfolio from rows"
                    .into(),
            ));
        };
        let m = self.units.len();
        // Each scenario's capped total and what each unit is paid in it.
        let mut scen: Vec<(f64, usize)> = (0..w.len())
            .map(|r| (losses[r * m..(r + 1) * m].iter().sum::<f64>().min(a), r))
            .collect();
        scen.sort_by(|u, v| v.0.total_cmp(&u.0));
        let paid = |r: usize, i: usize| {
            let row = &losses[r * m..(r + 1) * m];
            let total: f64 = row.iter().sum();
            if total > a {
                row[i] * a / total
            } else {
                row[i]
            }
        };
        let mut out = vec![
            AllocationRange {
                lower: 0.0,
                upper: 0.0
            };
            m
        ];
        let mut start = 0;
        let mut above = 0.0;
        while start < scen.len() {
            let mut end = start + 1;
            while end < scen.len() && scen[end].0 == scen[start].0 {
                end += 1;
            }
            let tie = &scen[start..end];
            for (i, range) in out.iter_mut().enumerate() {
                for upper in [true, false] {
                    let mut order: Vec<usize> = tie.iter().map(|&(_, r)| r).collect();
                    order.sort_by(|&u, &v| {
                        let by = paid(v, i).total_cmp(&paid(u, i));
                        if upper { by } else { by.reverse() }
                    });
                    let (mut s, mut value) = (above, 0.0);
                    for r in order {
                        let next = s + w[r];
                        value += paid(r, i) * (g.g(next) - g.g(s));
                        s = next;
                    }
                    if upper {
                        range.upper += value;
                    } else {
                        range.lower += value;
                    }
                }
            }
            above += tie.iter().map(|&(_, r)| w[r]).sum::<f64>();
            start = end;
        }
        Ok(out)
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

/// The edge of the upper (or lower) convex hull of the points `(t_k, v_k)`
/// that spans `x`, as the indices of its ends with `t_i < t_j`; `None` when
/// `x` lies outside the points' range or every point has the same `t`.
fn hull_edge(t: &[f64], v: &[f64], x: f64, upper: bool) -> Option<(usize, usize)> {
    let sign = if upper { 1.0 } else { -1.0 };
    let mut order: Vec<usize> = (0..t.len()).collect();
    // By t, and among equal t the extreme one last, so it is the one kept.
    order.sort_by(|&i, &j| {
        t[i].total_cmp(&t[j])
            .then((sign * v[i]).total_cmp(&(sign * v[j])))
    });
    let mut hull: Vec<usize> = Vec::new();
    for &k in &order {
        if let Some(&last) = hull.last()
            && t[last] == t[k]
        {
            hull.pop();
        }
        while hull.len() >= 2 {
            let (o, q) = (hull[hull.len() - 2], hull[hull.len() - 1]);
            let cross = (t[q] - t[o]) * (v[k] - v[o]) - (v[q] - v[o]) * (t[k] - t[o]);
            if sign * cross >= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(k);
    }
    hull.windows(2)
        .find(|e| t[e[0]] <= x && x <= t[e[1]])
        .map(|e| (e[0], e[1]))
}

/// `E[X ∧ t]` of a grid at any `t`: exact, since it is linear in `t`
/// between grid points.
struct LimitedOnGrid {
    step: f64,
    /// `E[X ∧ jh]` for `j = 0..len`.
    at_points: Vec<f64>,
}

impl LimitedOnGrid {
    fn new(g: &Grid) -> Self {
        let h = g.step();
        let mut at_points = Vec::with_capacity(g.len());
        // E[X ∧ jh] = E[X ∧ (j-1)h] + h P(X > (j-1)h).
        let (mut lev, mut above) = (0.0, 1.0);
        for &p in g.probs() {
            at_points.push(lev);
            above -= p;
            lev += h * above.max(0.0);
        }
        Self { step: h, at_points }
    }

    fn at(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let last = self.at_points.len() - 1;
        let pos = t / self.step;
        if pos >= last as f64 {
            return self.at_points[last];
        }
        let j = pos.floor() as usize;
        let f = pos - j as f64;
        self.at_points[j] + f * (self.at_points[j + 1] - self.at_points[j])
    }
}

/// The linear convolution of `a` and `b` by FFT.
fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let out = a.len() + b.len() - 1;
    let len = out.next_power_of_two();
    let mut planner = FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(len);
    let pad = |v: &[f64]| {
        let mut c: Vec<Complex64> = v.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        c.resize(len, Complex64::new(0.0, 0.0));
        forward.process(&mut c);
        c
    };
    let mut prod: Vec<Complex64> = pad(a).iter().zip(&pad(b)).map(|(x, y)| x * y).collect();
    planner.plan_fft_inverse(len).process(&mut prod);
    prod.iter().take(out).map(|c| c.re / len as f64).collect()
}

/// Each unit priced alone, the book priced together and the sum of the
/// parts, from [`Portfolio::price_stand_alone`].
#[derive(Debug, Clone, PartialEq)]
pub struct StandAlone {
    /// Unit names.
    pub units: Vec<String>,
    /// Each unit as if it were the whole book, at its own VaR.
    pub stand_alone: Vec<Pentagon>,
    /// The book, at the VaR of its total.
    pub total: Pentagon,
    /// The stand-alone loss, premium and assets added up.
    pub sum: Pentagon,
}

/// A priority rule for [`Priority`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// The unit ranks ahead of the others pooled.
    Senior,
    /// Pro rata to the losses.
    Equal,
    /// The others pooled rank ahead of the unit.
    Junior,
}

/// Each unit's recovery from the assets under each [`Rule`], from
/// [`Portfolio::priority`].
#[derive(Debug, Clone, PartialEq)]
pub struct Priority {
    /// Unit names.
    pub units: Vec<String>,
    /// The assets `a`.
    pub assets: f64,
    /// Each unit's expected loss, `E[Xᵢ]`.
    pub mean: Vec<f64>,
    /// `E[Xᵢ ∧ a]`.
    pub senior: Vec<f64>,
    /// `E[Xᵢ min(1, a / X)]`.
    pub equal: Vec<f64>,
    /// `E[min(Xᵢ, (a - X₋ᵢ)⁺)]`.
    pub junior: Vec<f64>,
    /// What the insurer pays under every rule, `E[X ∧ a]`.
    pub total: f64,
}

impl Priority {
    /// Each unit's recovery under `rule`.
    pub fn recovery(&self, rule: Rule) -> &[f64] {
        match rule {
            Rule::Senior => &self.senior,
            Rule::Equal => &self.equal,
            Rule::Junior => &self.junior,
        }
    }

    /// Each unit's shortfall under `rule`, `E[Xᵢ]` less its recovery.
    pub fn shortfall(&self, rule: Rule) -> Vec<f64> {
        self.mean
            .iter()
            .zip(self.recovery(rule))
            .map(|(e, r)| e - r)
            .collect()
    }

    /// Each unit's expected policyholder deficit ratio under `rule`, its
    /// shortfall over `E[Xᵢ]` (0 for a unit with no expected loss). For
    /// every unit, senior <= equal <= junior.
    pub fn epd(&self, rule: Rule) -> Vec<f64> {
        self.mean
            .iter()
            .zip(self.shortfall(rule))
            .map(|(&e, s)| if e > 0.0 { s / e } else { 0.0 })
            .collect()
    }
}

/// The range of one unit's premium over the distortions with a given
/// total price, from [`Portfolio::premium_bounds`].
#[derive(Debug, Clone, PartialEq)]
pub struct PremiumBound {
    /// The smallest premium.
    pub lower: f64,
    /// The largest premium.
    pub upper: f64,
    /// A BiTVaR that gives the smallest.
    pub lower_distortion: Distortion,
    /// A BiTVaR that gives the largest.
    pub upper_distortion: Distortion,
}

/// The range of a unit's premium over the natural allocation set, from
/// [`Portfolio::allocation_set`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AllocationRange {
    /// The smallest premium.
    pub lower: f64,
    /// The largest premium.
    pub upper: f64,
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
    fn premium_bounds_match_aggregate() {
        // aggregate 1.0.1 AllocationBounds on InsCo, to its printed digits.
        let port = insco();
        let b = port.premium_bounds(53.565217391304344, 100.0).unwrap();
        let want = [
            (13.097826, 15.032744),
            (17.465726, 20.411685),
            (19.254738, 22.098038),
        ];
        for (got, (lo, hi)) in b.iter().zip(want) {
            assert!(
                (got.lower - lo).abs() < 1e-6 && (got.upper - hi).abs() < 1e-6,
                "{got:?}"
            );
        }
        let b = port.premium_bounds(47.0, 65.0).unwrap();
        let want = [
            (12.374586, 13.881867),
            (16.335616, 19.292994),
            (15.057851, 17.671233),
        ];
        for (got, (lo, hi)) in b.iter().zip(want) {
            assert!(
                (got.lower - lo).abs() < 1e-6 && (got.upper - hi).abs() < 1e-6,
                "{got:?}"
            );
        }
        // Each bound's distortion prices the total at the premium.
        let capped: Vec<f64> = port.totals().iter().map(|x| x.min(65.0)).collect();
        for u in &b {
            for d in [&u.lower_distortion, &u.upper_distortion] {
                assert!((d.apply_discrete(&capped, port.probs()) - 47.0).abs() < 1e-9);
            }
        }
    }

    /// The bounds by pricing the book at every knot and trying every pair,
    /// as `premium_bounds` did before its prefix sums and hulls.
    fn bounds_by_every_pair(port: &Portfolio, premium: f64, a: f64) -> Vec<(f64, f64)> {
        let capped: Vec<f64> = port.x.iter().map(|x| x.min(a)).collect();
        let tvar = |p: f64| {
            Distortion::tvar(p)
                .unwrap()
                .apply_discrete(&capped, &port.p)
        };
        let p_star = prospicio_math::roots::bisect(0.0, 1.0, |p| tvar(p) < premium);
        let mut knots = vec![0.0, p_star];
        let mut cum = 0.0;
        for (&x, &pk) in port.x.iter().zip(&port.p) {
            cum += pk;
            if x < a && cum < 1.0 {
                knots.push(cum);
            }
        }
        knots.sort_by(f64::total_cmp);
        knots.dedup();
        let t: Vec<f64> = knots.iter().map(|&p| tvar(p)).collect();
        let units: Vec<Vec<f64>> = knots
            .iter()
            .map(|&p| {
                let price = port
                    .price(&Distortion::tvar(p).unwrap(), a, Allocation::Linear)
                    .unwrap();
                price.allocated.iter().map(|u| u.premium).collect()
            })
            .collect();
        let tol = 1e-12 * premium.abs().max(1.0);
        let mut out = vec![(f64::INFINITY, f64::NEG_INFINITY); port.units.len()];
        for i in 0..knots.len() {
            for j in i..knots.len() {
                if t[i] > premium + tol || t[j] < premium - tol {
                    continue;
                }
                let w = if t[j] - t[i] > tol {
                    (premium - t[i]) / (t[j] - t[i])
                } else {
                    0.0
                };
                if !(0.0..=1.0).contains(&w) {
                    continue;
                }
                for (u, o) in out.iter_mut().enumerate() {
                    let v = (1.0 - w) * units[i][u] + w * units[j][u];
                    *o = (o.0.min(v), o.1.max(v));
                }
            }
        }
        out
    }

    #[test]
    fn premium_bounds_match_every_pair() {
        // A book with ties and an atom at the assets, by a fixed LCG.
        let mut state = 12345_u64;
        let mut next = |k: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % k) as f64
        };
        let rows: Vec<Vec<f64>> = (0..80)
            .map(|_| vec![next(20), next(7) * next(5), next(3) * 10.0])
            .collect();
        let units = vec!["A".into(), "B".into(), "C".into()];
        let book = Portfolio::from_rows(units, &rows, None).unwrap();
        for port in [insco(), book] {
            let x = port.totals().to_vec();
            for a in [x[x.len() / 2], x[x.len() * 3 / 4], x[x.len() - 1]] {
                let (loss, _) = port.limited_expected(a);
                for f in [0.1, 0.4, 0.8] {
                    let premium = loss + f * (a - loss);
                    let got = port.premium_bounds(premium, a).unwrap();
                    let want = bounds_by_every_pair(&port, premium, a);
                    let capped: Vec<f64> = x.iter().map(|v| v.min(a)).collect();
                    for (g, (lo, hi)) in got.iter().zip(want) {
                        let near = |u: f64, v: f64| (u - v).abs() < 1e-9 * v.abs().max(1.0);
                        assert!(
                            near(g.lower, lo) && near(g.upper, hi),
                            "{g:?} vs {lo}, {hi}"
                        );
                        for d in [&g.lower_distortion, &g.upper_distortion] {
                            assert!(near(d.apply_discrete(&capped, port.probs()), premium));
                        }
                    }
                }
            }
        }
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
