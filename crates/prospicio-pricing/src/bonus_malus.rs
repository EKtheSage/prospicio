//! Bonus-malus and no-claim-discount scales as Markov chains
//! (`docs/design/credibility.md`, "Bonus-malus scales"; issue #563).
//!
//! A policyholder sits on one of the levels `0..levels` of a scale, 0 the
//! best, and moves each year by the number of claims reported: the level
//! next year is a function of this year's level and claim count. Under a
//! Poisson count with mean `λ` the levels are a Markov chain, and a mixed
//! Poisson portfolio, `λ Θ` with `Θ` of mean 1, is a mixture of such
//! chains. [`BonusMalus`] gives, for one `λ`, the transition matrix, the
//! stationary distribution, the distribution after a number of years, the
//! years until it is close to stationary and Loimaranta's efficiency of
//! given relativities; and for a [`Portfolio`], the stationary
//! distribution of the levels and Norberg's optimal relativities.

use prospicio_core::{Error, Result};
use prospicio_math::integrate::gauss_legendre;
use prospicio_math::linalg::solve;
use prospicio_math::special::ln_gamma;
use prospicio_prob::count_families::Mixing;
use prospicio_prob::{Distribution, Gamma, InverseGaussian};

/// The scale's moves: down `down` levels after a claim-free year, up
/// `up_per_claim` levels for each claim, both capped at the scale's ends.
///
/// The `-1/+top` scale, where any claim sends the policyholder to the top,
/// is `down = 1` and `up_per_claim = levels - 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BonusRule {
    /// Levels down after a claim-free year.
    pub down: usize,
    /// Levels up for each claim, at least 1.
    pub up_per_claim: usize,
}

/// A bonus-malus scale: its levels, the level a newcomer starts on and the
/// level after each claim count from each level.
///
/// ```
/// use prospicio_pricing::bonus_malus::{BonusMalus, BonusRule};
///
/// // Six levels, -1 a claim-free year, +3 a claim, newcomers on level 5.
/// let scale = BonusMalus::rule(6, 5, BonusRule { down: 1, up_per_claim: 3 }).unwrap();
/// assert_eq!(scale.next_level(2, 1), 5);
/// let pi = scale.stationary_at(0.1).unwrap();
/// assert!((pi.iter().sum::<f64>() - 1.0).abs() < 1e-14);
/// // A good driver spends most of the time on the bottom level.
/// assert!(pi[0] > 0.6);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BonusMalus {
    start: usize,
    /// `table[l][k]`: the level after `k` claims from level `l`; the last
    /// column is for that many claims or more.
    table: Vec<Vec<usize>>,
}

/// The portfolio the scale runs on: a priori classes, each with a weight
/// and an expected claim frequency `λ_k`, and the residual heterogeneity
/// `Θ` (mean 1) that the scale is meant to reveal, so that a policyholder
/// of class `k` has Poisson claims with mean `λ_k Θ`.
#[derive(Debug, Clone, PartialEq)]
pub struct Portfolio {
    weights: Vec<f64>,
    lambdas: Vec<f64>,
    mixing: Mixing,
}

impl Portfolio {
    /// One class with frequency `lambda`, the mixed Poisson portfolio of
    /// Lemaire's and Denuit et al.'s examples. `mixing` is
    /// [`Mixing::Gamma`] (a negative binomial portfolio) or
    /// [`Mixing::InverseGaussian`] (a Poisson-inverse Gaussian one).
    pub fn new(lambda: f64, mixing: Mixing) -> Result<Self> {
        Self::classes(vec![1.0], vec![lambda], mixing)
    }

    /// A priori classes with their weights (normalised to 1) and
    /// frequencies, and the residual heterogeneity `mixing`.
    pub fn classes(weights: Vec<f64>, lambdas: Vec<f64>, mixing: Mixing) -> Result<Self> {
        if weights.is_empty() || weights.len() != lambdas.len() {
            return Err(Error::Data(format!(
                "{} weights for {} frequencies; give one weight per class",
                weights.len(),
                lambdas.len()
            )));
        }
        for &w in &weights {
            if !(w.is_finite() && w >= 0.0) {
                return Err(Error::InvalidParameter {
                    name: "weights",
                    value: w,
                    reason: "must be finite and non-negative",
                });
            }
        }
        for &l in &lambdas {
            check_lambda(l)?;
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(Error::Data("the weights sum to zero".into()));
        }
        match mixing {
            Mixing::Gamma { cv } | Mixing::InverseGaussian { cv } => {
                if !(cv.is_finite() && cv > 0.0) {
                    return Err(Error::InvalidParameter {
                        name: "cv",
                        value: cv,
                        reason: "must be finite and positive",
                    });
                }
            }
            _ => {
                return Err(Error::Data(
                    "the bonus-malus portfolio's mixing must be gamma or inverse Gaussian".into(),
                ));
            }
        }
        Ok(Self {
            weights: weights.iter().map(|w| w / total).collect(),
            lambdas,
            mixing,
        })
    }

    /// The class weights, summing to 1.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The class frequencies.
    pub fn lambdas(&self) -> &[f64] {
        &self.lambdas
    }

    /// The residual heterogeneity.
    pub fn mixing(&self) -> Mixing {
        self.mixing
    }

    /// `∫ g(θ) dF(θ)` for each output of `g`, by Gauss–Legendre panels on
    /// `ln θ` between the quantiles `1e-15` and `1 - 1e-15`, the panels
    /// doubled until no output moves by more than `1e-13` of the largest.
    fn integrate(
        &self,
        n: usize,
        mut g: impl FnMut(f64, &mut [f64]) -> Result<()>,
    ) -> Result<Vec<f64>> {
        let (ln_pdf, lo, hi): (Box<dyn Fn(f64) -> f64>, f64, f64) = match self.mixing {
            Mixing::Gamma { cv } => {
                let a = 1.0 / (cv * cv);
                let d = Gamma::new(a, 1.0 / a)?;
                let c = a * a.ln() - ln_gamma(a);
                (
                    Box::new(move |t: f64| c + (a - 1.0) * t.ln() - a * t),
                    d.quantile(1e-15)?,
                    d.quantile(1.0 - 1e-15)?,
                )
            }
            Mixing::InverseGaussian { cv } => {
                let phi = 1.0 / (cv * cv);
                let d = InverseGaussian::new(1.0, phi)?;
                (
                    Box::new(move |t: f64| {
                        0.5 * (phi / (2.0 * std::f64::consts::PI * t * t * t)).ln()
                            - phi * (t - 1.0) * (t - 1.0) / (2.0 * t)
                    }),
                    d.quantile(1e-15)?,
                    d.quantile(1.0 - 1e-15)?,
                )
            }
            _ => unreachable!("checked in Portfolio::classes"),
        };
        let (a, b) = (lo.ln(), hi.ln());
        let mut buf = vec![0.0; n];
        let mut run = |panels: usize| -> Result<Vec<f64>> {
            let mut out = vec![0.0; n];
            let h = (b - a) / panels as f64;
            for j in 0..panels {
                let (x0, x1) = (a + j as f64 * h, a + (j + 1) as f64 * h);
                // gauss_legendre integrates one scalar; integrate each
                // output by the same nodes through a recorded pass.
                let mut nodes = Vec::with_capacity(8);
                gauss_legendre(
                    |x| {
                        nodes.push(x);
                        Ok::<_, Error>(0.0)
                    },
                    x0,
                    x1,
                )?;
                let mut values = Vec::with_capacity(nodes.len());
                for &x in &nodes {
                    let t = x.exp();
                    g(t, &mut buf)?;
                    values.push((ln_pdf(t) + x).exp());
                    let w = values.last().copied().unwrap_or(0.0);
                    values.extend(buf.iter().map(|v| v * w));
                }
                let stride = n + 1;
                for (i, o) in out.iter_mut().enumerate() {
                    let mut k = 0;
                    *o += gauss_legendre(
                        |_| {
                            let v = values[k * stride + 1 + i];
                            k += 1;
                            Ok::<_, Error>(v)
                        },
                        x0,
                        x1,
                    )?;
                }
            }
            Ok(out)
        };
        let mut panels = 32;
        let mut prev = run(panels)?;
        loop {
            panels *= 2;
            let next = run(panels)?;
            let scale = next.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1e-300);
            let moved = next
                .iter()
                .zip(&prev)
                .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
            if moved <= 1e-13 * scale || panels >= 4096 {
                return Ok(next);
            }
            prev = next;
        }
    }
}

fn check_lambda(lambda: f64) -> Result<()> {
    if lambda.is_finite() && lambda > 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name: "lambda",
            value: lambda,
            reason: "must be finite and positive",
        })
    }
}

/// Poisson probabilities `p_0..p_{K-1}` and `P(N >= K)` at mean `λ`.
fn poisson_lumped(lambda: f64, k_max: usize) -> Vec<f64> {
    let mut p = Vec::with_capacity(k_max + 1);
    let mut pk = (-lambda).exp();
    let mut below = 0.0;
    for k in 0..k_max {
        p.push(pk);
        below += pk;
        pk *= lambda / (k + 1) as f64;
    }
    p.push((1.0 - below).max(0.0));
    p
}

impl BonusMalus {
    /// A scale of `levels` levels moving by `rule`, newcomers on `start`.
    pub fn rule(levels: usize, start: usize, rule: BonusRule) -> Result<Self> {
        if levels < 2 {
            return Err(Error::InvalidParameter {
                name: "levels",
                value: levels as f64,
                reason: "a scale needs at least two levels",
            });
        }
        if rule.up_per_claim == 0 {
            return Err(Error::InvalidParameter {
                name: "up_per_claim",
                value: 0.0,
                reason: "a claim must move the policyholder up at least one level",
            });
        }
        let top = levels - 1;
        // From K claims on every level lands on the top.
        let k_max = top.div_ceil(rule.up_per_claim);
        let table = (0..levels)
            .map(|l| {
                (0..=k_max)
                    .map(|k| {
                        if k == 0 {
                            l.saturating_sub(rule.down)
                        } else {
                            (l + k * rule.up_per_claim).min(top)
                        }
                    })
                    .collect()
            })
            .collect();
        Self::table(start, table)
    }

    /// A scale from its table: `table[l][k]` is the level after `k` claims
    /// from level `l`, one row per level, every row with the same number of
    /// columns; the last column is for that many claims or more.
    pub fn table(start: usize, table: Vec<Vec<usize>>) -> Result<Self> {
        let levels = table.len();
        if levels < 2 {
            return Err(Error::Data("a scale needs at least two levels".into()));
        }
        let cols = table[0].len();
        if cols < 2 {
            return Err(Error::Data(
                "each row needs a column for no claim and one for a claim or more".into(),
            ));
        }
        for (l, row) in table.iter().enumerate() {
            if row.len() != cols {
                return Err(Error::Data(format!(
                    "row {l} has {} columns, row 0 has {cols}",
                    row.len()
                )));
            }
            if let Some(&bad) = row.iter().find(|&&j| j >= levels) {
                return Err(Error::Data(format!(
                    "row {l} moves to level {bad}, past the top level {}",
                    levels - 1
                )));
            }
        }
        if start >= levels {
            return Err(Error::InvalidParameter {
                name: "start",
                value: start as f64,
                reason: "must be one of the scale's levels",
            });
        }
        Ok(Self { start, table })
    }

    /// The number of levels.
    pub fn levels(&self) -> usize {
        self.table.len()
    }

    /// The level a newcomer starts on.
    pub fn start(&self) -> usize {
        self.start
    }

    /// The table: the level after `k` claims from each level, the last
    /// column for that many or more.
    pub fn transitions(&self) -> &[Vec<usize>] {
        &self.table
    }

    /// The level after `claims` claims in a year from `level` (which must
    /// be one of the scale's levels).
    pub fn next_level(&self, level: usize, claims: usize) -> usize {
        let row = &self.table[level];
        row[claims.min(row.len() - 1)]
    }

    /// The one-year transition matrix under Poisson claims with mean
    /// `lambda`, row-major: entry `(l, j)` is the probability of moving
    /// from level `l` to level `j`.
    pub fn transition_matrix(&self, lambda: f64) -> Result<Vec<f64>> {
        check_lambda(lambda)?;
        Ok(self.matrix(&poisson_lumped(lambda, self.table[0].len() - 1)))
    }

    fn matrix(&self, p: &[f64]) -> Vec<f64> {
        let s = self.levels();
        let mut m = vec![0.0; s * s];
        for (l, row) in self.table.iter().enumerate() {
            for (k, &j) in row.iter().enumerate() {
                m[l * s + j] += p[k];
            }
        }
        m
    }

    /// The derivative of the transition matrix in `λ`: `p_k' = p_{k-1} -
    /// p_k`, and `P(N >= K)' = p_{K-1}`.
    fn matrix_derivative(&self, lambda: f64) -> Vec<f64> {
        let k_max = self.table[0].len() - 1;
        let p = poisson_lumped(lambda, k_max + 1);
        let dp: Vec<f64> = (0..=k_max)
            .map(|k| {
                let prev = if k == 0 { 0.0 } else { p[k - 1] };
                if k == k_max { prev } else { prev - p[k] }
            })
            .collect();
        self.matrix(&dp)
    }

    /// Solve `x (P - I) = -rhs` with `Σ x = total`: the stationary
    /// distribution for `rhs = 0`, `total = 1`, and its derivative for
    /// `rhs = π P'`, `total = 0`.
    fn solve_left(&self, m: &[f64], rhs: &[f64], total: f64) -> Result<Vec<f64>> {
        let s = self.levels();
        // (P - I)^T, its last row replaced by ones.
        let mut a = vec![0.0; s * s];
        for i in 0..s {
            for j in 0..s {
                a[i * s + j] = m[j * s + i] - if i == j { 1.0 } else { 0.0 };
            }
        }
        for j in 0..s {
            a[(s - 1) * s + j] = 1.0;
        }
        let mut b: Vec<f64> = rhs.iter().map(|v| -v).collect();
        b[s - 1] = total;
        solve(a, b, s).ok_or_else(|| {
            Error::Data(
                "the scale has no unique stationary distribution: some levels are unreachable \
                 or absorbing"
                    .into(),
            )
        })
    }

    /// The stationary distribution of the levels for a policyholder with
    /// Poisson claims of mean `lambda`, by a linear solve of `π P = π`,
    /// `Σ π = 1`. An error when it is not unique.
    pub fn stationary_at(&self, lambda: f64) -> Result<Vec<f64>> {
        let m = self.transition_matrix(lambda)?;
        let mut pi = self.solve_left(&m, &vec![0.0; self.levels()], 1.0)?;
        for v in &mut pi {
            *v = v.max(0.0);
        }
        Ok(pi)
    }

    /// The distribution of the levels after `years` years from the start
    /// level.
    pub fn distribution_after(&self, lambda: f64, years: usize) -> Result<Vec<f64>> {
        let m = self.transition_matrix(lambda)?;
        let mut p = vec![0.0; self.levels()];
        p[self.start] = 1.0;
        for _ in 0..years {
            p = self.step(&m, &p);
        }
        Ok(p)
    }

    fn step(&self, m: &[f64], p: &[f64]) -> Vec<f64> {
        let s = self.levels();
        let mut next = vec![0.0; s];
        for (l, &pl) in p.iter().enumerate() {
            if pl != 0.0 {
                for (j, v) in next.iter_mut().enumerate() {
                    *v += pl * m[l * s + j];
                }
            }
        }
        next
    }

    /// The years from the start level until the distribution of the
    /// levels is within `tolerance` of the stationary one in total
    /// variation, `½ Σ |p_n - π|`; `None` past 100,000 years.
    pub fn years_to_stationarity(&self, lambda: f64, tolerance: f64) -> Result<Option<usize>> {
        if !(tolerance > 0.0 && tolerance < 1.0) {
            return Err(Error::InvalidParameter {
                name: "tolerance",
                value: tolerance,
                reason: "must be in (0, 1)",
            });
        }
        let m = self.transition_matrix(lambda)?;
        let pi = self.stationary_at(lambda)?;
        let mut p = vec![0.0; self.levels()];
        p[self.start] = 1.0;
        for n in 0..=100_000 {
            let tv = 0.5 * p.iter().zip(&pi).map(|(a, b)| (a - b).abs()).sum::<f64>();
            if tv < tolerance {
                return Ok(Some(n));
            }
            p = self.step(&m, &p);
        }
        Ok(None)
    }

    fn check_relativities(&self, relativities: &[f64]) -> Result<()> {
        if relativities.len() != self.levels() {
            return Err(Error::Data(format!(
                "{} relativities for {} levels",
                relativities.len(),
                self.levels()
            )));
        }
        Ok(())
    }

    /// The stationary mean relativity `r(λ) = Σ_l π_l(λ) r_l` of a
    /// policyholder with frequency `lambda`.
    pub fn mean_relativity(&self, lambda: f64, relativities: &[f64]) -> Result<f64> {
        self.check_relativities(relativities)?;
        let pi = self.stationary_at(lambda)?;
        Ok(pi.iter().zip(relativities).map(|(p, r)| p * r).sum())
    }

    /// Loimaranta's efficiency `η(λ) = d ln r(λ) / d ln λ` of the scale
    /// with `relativities`: how far the stationary premium follows a
    /// change in the claim frequency (1 would follow it fully). The
    /// derivative of `π` is exact, from `π' (P - I) = -π P'`, `Σ π' = 0`.
    pub fn efficiency(&self, lambda: f64, relativities: &[f64]) -> Result<f64> {
        self.check_relativities(relativities)?;
        let m = self.transition_matrix(lambda)?;
        let pi = self.solve_left(&m, &vec![0.0; self.levels()], 1.0)?;
        let dm = self.matrix_derivative(lambda);
        let s = self.levels();
        let rhs: Vec<f64> = (0..s)
            .map(|j| (0..s).map(|l| pi[l] * dm[l * s + j]).sum())
            .collect();
        let dpi = self.solve_left(&m, &rhs, 0.0)?;
        let r: f64 = pi.iter().zip(relativities).map(|(p, r)| p * r).sum();
        let dr: f64 = dpi.iter().zip(relativities).map(|(p, r)| p * r).sum();
        Ok(lambda * dr / r)
    }

    /// `Σ_k w_k ∫ π_l(λ_k θ) dF(θ)` and `Σ_k w_k ∫ θ π_l(λ_k θ) dF(θ)`,
    /// one after the other.
    fn portfolio_integrals(&self, portfolio: &Portfolio) -> Result<Vec<f64>> {
        let s = self.levels();
        portfolio.integrate(2 * s, |theta, out| {
            out.fill(0.0);
            for (&w, &l) in portfolio.weights.iter().zip(&portfolio.lambdas) {
                let pi = self.stationary_at(l * theta)?;
                for (j, p) in pi.iter().enumerate() {
                    out[j] += w * p;
                    out[s + j] += w * theta * p;
                }
            }
            Ok(())
        })
    }

    /// The stationary distribution of the levels over the portfolio,
    /// `Σ_k w_k ∫ π(λ_k θ) dF(θ)`.
    pub fn stationary(&self, portfolio: &Portfolio) -> Result<Vec<f64>> {
        let mut v = self.portfolio_integrals(portfolio)?;
        v.truncate(self.levels());
        Ok(v)
    }

    /// Norberg's optimal relativities under quadratic loss: the `r_l`
    /// minimising `E[(Θ - r_L)²]` over the stationary portfolio, the a
    /// priori premium `λ_k` charged as a factor, which is
    /// `r_l = E[Θ | L = l] = Σ_k w_k ∫ θ π_l(λ_k θ) dF / Σ_k w_k ∫ π_l(λ_k θ) dF`.
    /// They are financially balanced, `Σ_l π_l r_l = E[Θ] = 1`. A level
    /// the stationary portfolio never reaches gets NaN.
    pub fn optimal_relativities(&self, portfolio: &Portfolio) -> Result<Vec<f64>> {
        let s = self.levels();
        let v = self.portfolio_integrals(portfolio)?;
        Ok((0..s)
            .map(|l| {
                if v[l] > 0.0 {
                    v[s + l] / v[l]
                } else {
                    f64::NAN
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gamma(cv: f64) -> Mixing {
        Mixing::Gamma { cv }
    }

    #[test]
    fn the_rule_and_its_table() {
        let s = BonusMalus::rule(
            6,
            5,
            BonusRule {
                down: 1,
                up_per_claim: 3,
            },
        )
        .unwrap();
        assert_eq!(s.transitions()[0], vec![0, 3, 5]);
        assert_eq!(s.transitions()[4], vec![3, 5, 5]);
        assert_eq!(s.next_level(0, 7), 5);
        let top = BonusMalus::rule(
            6,
            5,
            BonusRule {
                down: 1,
                up_per_claim: 5,
            },
        )
        .unwrap();
        assert!(top.transitions().iter().all(|r| r.len() == 2 && r[1] == 5));
        let m = s.transition_matrix(0.3).unwrap();
        for l in 0..6 {
            let row: f64 = m[l * 6..(l + 1) * 6].iter().sum();
            assert!((row - 1.0).abs() < 1e-15);
        }
        assert!(
            BonusMalus::rule(
                1,
                0,
                BonusRule {
                    down: 1,
                    up_per_claim: 1
                }
            )
            .is_err()
        );
        assert!(
            BonusMalus::rule(
                4,
                4,
                BonusRule {
                    down: 1,
                    up_per_claim: 1
                }
            )
            .is_err()
        );
        assert!(
            BonusMalus::rule(
                4,
                0,
                BonusRule {
                    down: 1,
                    up_per_claim: 0
                }
            )
            .is_err()
        );
        assert!(BonusMalus::table(0, vec![vec![0, 1], vec![0, 2]]).is_err());
        assert!(BonusMalus::table(0, vec![vec![0, 1], vec![0]]).is_err());
    }

    #[test]
    fn stationary_is_the_limit_of_the_powers() {
        let s = BonusMalus::rule(
            10,
            6,
            BonusRule {
                down: 1,
                up_per_claim: 2,
            },
        )
        .unwrap();
        let pi = s.stationary_at(0.15).unwrap();
        let p = s.distribution_after(0.15, 2000).unwrap();
        for (a, b) in pi.iter().zip(&p) {
            assert!((a - b).abs() < 1e-13);
        }
        let n = s.years_to_stationarity(0.15, 1e-3).unwrap().unwrap();
        let tv = |q: &[f64]| 0.5 * q.iter().zip(&pi).map(|(a, b)| (a - b).abs()).sum::<f64>();
        assert!(tv(&s.distribution_after(0.15, n).unwrap()) < 1e-3);
        assert!(tv(&s.distribution_after(0.15, n - 1).unwrap()) >= 1e-3);
        // A scale whose levels never move has no unique stationary
        // distribution.
        let stuck = BonusMalus::table(0, vec![vec![0, 0], vec![1, 1]]).unwrap();
        assert!(stuck.stationary_at(0.1).is_err());
    }

    #[test]
    fn two_levels_under_gamma_mixing_are_exact() {
        // -1/+1 on two levels: level 0 after a claim-free year, so
        // π_0(λ) = e^{-λ}; with Θ ~ Gamma(a, a), ∫ e^{-λθ} dF = (a/(a+λ))^a
        // and E[Θ | L = 0] = a / (a + λ).
        let s = BonusMalus::rule(
            2,
            1,
            BonusRule {
                down: 1,
                up_per_claim: 1,
            },
        )
        .unwrap();
        for cv in [0.5, 1.0, 1.6] {
            let a = 1.0 / (cv * cv);
            let lambda = 0.12;
            let port = Portfolio::new(lambda, gamma(cv)).unwrap();
            let pi = s.stationary(&port).unwrap();
            let p0 = (a / (a + lambda)).powf(a);
            assert!((pi[0] - p0).abs() < 1e-12, "cv {cv}: {} vs {p0}", pi[0]);
            assert!((pi[1] - (1.0 - p0)).abs() < 1e-12);
            let r = s.optimal_relativities(&port).unwrap();
            assert!((r[0] - a / (a + lambda)).abs() < 1e-12);
            assert!((r[1] - (1.0 - p0 * a / (a + lambda)) / (1.0 - p0)).abs() < 1e-11);
            assert!((pi[0] * r[0] + pi[1] * r[1] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn minus_one_top_bottom_level_is_exact() {
        // On -1/+top the bottom level means no claim in the last `top`
        // years, so π_0 = (a/(a + top λ))^a and r_0 = a / (a + top λ).
        let s = BonusMalus::rule(
            6,
            5,
            BonusRule {
                down: 1,
                up_per_claim: 5,
            },
        )
        .unwrap();
        let (cv, lambda) = (0.7_f64, 0.1);
        let a = 1.0 / (cv * cv);
        let port = Portfolio::new(lambda, gamma(cv)).unwrap();
        let pi = s.stationary(&port).unwrap();
        let r = s.optimal_relativities(&port).unwrap();
        assert!((pi[0] - (a / (a + 5.0 * lambda)).powf(a)).abs() < 1e-12);
        assert!((r[0] - a / (a + 5.0 * lambda)).abs() < 1e-12);
    }

    #[test]
    fn the_quadrature_integrates_the_mixing() {
        for mixing in [
            gamma(0.3),
            gamma(1.0),
            gamma(2.0),
            Mixing::InverseGaussian { cv: 0.8 },
        ] {
            let port = Portfolio::new(0.1, mixing).unwrap();
            let m = port
                .integrate(3, |t, out| {
                    out[0] = 1.0;
                    out[1] = t;
                    out[2] = t * t;
                    Ok(())
                })
                .unwrap();
            let cv = mixing.cv();
            assert!((m[0] - 1.0).abs() < 1e-12, "{mixing:?}: {}", m[0]);
            assert!((m[1] - 1.0).abs() < 1e-12, "{mixing:?}: {}", m[1]);
            assert!(
                (m[2] - (1.0 + cv * cv)).abs() < 1e-10,
                "{mixing:?}: {}",
                m[2]
            );
        }
        assert!(Portfolio::new(0.1, Mixing::Beta { cv: 0.5, max: 1.0 }).is_err());
        assert!(Portfolio::classes(vec![1.0], vec![0.1, 0.2], gamma(0.5)).is_err());
    }

    #[test]
    fn relativities_balance_with_a_priori_classes() {
        let s = BonusMalus::rule(
            8,
            5,
            BonusRule {
                down: 1,
                up_per_claim: 3,
            },
        )
        .unwrap();
        let port =
            Portfolio::classes(vec![2.0, 1.0, 1.0], vec![0.05, 0.1, 0.2], gamma(1.1)).unwrap();
        let pi = s.stationary(&port).unwrap();
        let r = s.optimal_relativities(&port).unwrap();
        assert!((pi.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let balance: f64 = pi.iter().zip(&r).map(|(p, r)| p * r).sum();
        assert!((balance - 1.0).abs() < 1e-12);
        // Higher levels carry higher relativities.
        assert!(r.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn efficiency_matches_a_central_difference() {
        let s = BonusMalus::rule(
            8,
            5,
            BonusRule {
                down: 1,
                up_per_claim: 2,
            },
        )
        .unwrap();
        let r: Vec<f64> = (0..8).map(|l| 0.6 + 0.15 * l as f64).collect();
        let lambda = 0.15;
        let eta = s.efficiency(lambda, &r).unwrap();
        let h = 1e-5;
        let up = s.mean_relativity(lambda * (1.0 + h), &r).unwrap().ln();
        let down = s.mean_relativity(lambda * (1.0 - h), &r).unwrap().ln();
        let fd = (up - down) / ((1.0 + h).ln() - (1.0 - h).ln());
        assert!((eta - fd).abs() < 1e-8, "{eta} vs {fd}");
        assert!(eta > 0.0 && eta < 1.0);
        assert!(s.efficiency(lambda, &r[..7]).is_err());
    }
}
