//! Capital allocation and diversification for a joint
//! [`PredictiveDistribution`].
//!
//! The portfolio is the [`total`](PredictiveDistribution::total) of the
//! components, and its capital is a distortion risk measure `ρ` of that
//! total. An allocation splits `ρ(S)` back to the components. Methods
//! differ in what they reward; see [`AllocationMethod`] and
//! `docs/design/risk.md`.

use act_core::{Error, Result};

use crate::distortion::Distortion;
use crate::predictive::PredictiveDistribution;
use crate::sampled::Empirical;

/// How [`PredictiveDistribution::capital`] splits the portfolio's risk
/// measure `ρ(S)`, `S = Σ X_j`, between components `X_j`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationMethod {
    /// Euler (co-measure) allocation, `∂ρ(S + h X_j)/∂h` at `h = 0`: the
    /// component's own draws weighted by the distortion weights of the
    /// total's ranks. For TVaR these are the CoTVaRs,
    /// `E[X_j | S in its top 1 − p]`. The only method that is
    /// consistent with marginal changes to the portfolio. Sums to `ρ(S)`.
    Euler,
    /// `ρ(S) Cov(X_j, S) / Var(S)`: the variance's Euler allocation scaled
    /// to `ρ(S)`. Looks at the whole distribution, not the tail. Sums to
    /// `ρ(S)`.
    Covariance,
    /// `ρ(S) ρ(X_j) / Σ_k ρ(X_k)`, the stand-alone measures scaled to the
    /// total (the "haircut" allocation). Ignores dependence beyond the
    /// total. Sums to `ρ(S)`.
    Proportional,
    /// `ρ(S) − ρ(S − X_j)`, what the portfolio's measure falls by without
    /// the component (Merton–Perold). Does *not* sum to `ρ(S)`; the
    /// shortfall is the capital no single component is responsible for.
    Marginal,
    /// The Shapley value of the game `v(T) = ρ(Σ_{j∈T} X_j)`: each
    /// component's marginal contribution averaged over every order of
    /// joining. Sums to `ρ(S)`. Needs `ρ` of all `2^m` sub-portfolios, so
    /// it is limited to 12 components.
    Shapley,
}

/// The result of [`PredictiveDistribution::capital`].
#[derive(Debug, Clone, PartialEq)]
pub struct Allocation {
    pub method: AllocationMethod,
    /// `ρ(S)`, the portfolio's measure.
    pub total: f64,
    /// `ρ(X_j)` per component, in
    /// [`components`](PredictiveDistribution::components) order.
    pub standalone: Vec<f64>,
    /// Allocated capital per component, in the same order.
    pub allocated: Vec<f64>,
}

impl Allocation {
    /// `Σ_j ρ(X_j) − ρ(S)`: the capital saved by holding the components
    /// together. Non-negative for a subadditive measure (every
    /// [`Distortion`] is).
    pub fn diversification_benefit(&self) -> f64 {
        self.standalone.iter().sum::<f64>() - self.total
    }

    /// `ρ(X_j)` less its allocation, per component: each component's share
    /// of the diversification benefit.
    pub fn diversification(&self) -> Vec<f64> {
        self.standalone
            .iter()
            .zip(&self.allocated)
            .map(|(s, a)| s - a)
            .collect()
    }
}

/// Most components [`AllocationMethod::Shapley`] accepts.
pub const SHAPLEY_MAX_COMPONENTS: usize = 12;

impl PredictiveDistribution {
    /// The risk measure `ρ = d` of the total, each component's stand-alone
    /// measure, and the total's allocation to the components by `method`.
    ///
    /// Components must add up to the portfolio being allocated (segments,
    /// not a tower result holding gross, ceded and net side by side).
    ///
    /// ```
    /// use act_prob::capital::AllocationMethod;
    /// use act_prob::{Distortion, KeyValue, PredictiveDistribution, Provenance};
    ///
    /// // Two lines over four simulations; the totals are 3, 5, 7, 9.
    /// let pd = PredictiveDistribution::from_draws(
    ///     vec!["lob".into()],
    ///     vec![vec![KeyValue::from("motor")], vec![KeyValue::from("property")]],
    ///     vec![1.0, 2.0, 4.0, 1.0, 2.0, 5.0, 3.0, 6.0],
    ///     Provenance::new("example"),
    /// )
    /// .unwrap();
    /// let tvar = Distortion::tvar(0.5).unwrap();
    /// let a = pd.capital(&tvar, AllocationMethod::Euler).unwrap();
    /// assert_eq!(a.total, 8.0);
    /// assert_eq!(a.standalone, [3.5, 5.5]);
    /// assert_eq!(a.allocated, [2.5, 5.5]);
    /// assert_eq!(a.diversification_benefit(), 1.0);
    /// ```
    ///
    /// Fails for [`AllocationMethod::Covariance`] when the total does not
    /// vary, for [`AllocationMethod::Proportional`] when the stand-alone
    /// measures sum to 0, and for [`AllocationMethod::Shapley`] with more
    /// than [`SHAPLEY_MAX_COMPONENTS`] components.
    pub fn capital(&self, d: &Distortion, method: AllocationMethod) -> Result<Allocation> {
        let m = self.n_components();
        let draws = self.draw_matrix();
        let column = |j: usize| -> Vec<f64> { draws.iter().skip(j).step_by(m).copied().collect() };
        let total = self.total().distortion(d);
        let standalone: Vec<f64> = (0..m).map(|j| measure(d, column(j))).collect();
        let allocated = match method {
            AllocationMethod::Euler => self.allocate(d),
            AllocationMethod::Covariance => {
                let s = self.total().draws();
                let n = s.len() as f64;
                let mean_s = s.iter().sum::<f64>() / n;
                let var_s = s.iter().map(|x| (x - mean_s).powi(2)).sum::<f64>() / n;
                if var_s.is_nan() || var_s <= 0.0 {
                    return Err(invalid("draws", var_s, "the total must vary"));
                }
                (0..m)
                    .map(|j| {
                        let x = column(j);
                        let mean_x = x.iter().sum::<f64>() / n;
                        let cov = x
                            .iter()
                            .zip(s)
                            .map(|(a, b)| (a - mean_x) * (b - mean_s))
                            .sum::<f64>()
                            / n;
                        total * cov / var_s
                    })
                    .collect()
            }
            AllocationMethod::Proportional => {
                let sum: f64 = standalone.iter().sum();
                if sum == 0.0 || !sum.is_finite() {
                    return Err(invalid(
                        "standalone",
                        sum,
                        "must have a finite, non-zero sum",
                    ));
                }
                standalone.iter().map(|r| total * r / sum).collect()
            }
            AllocationMethod::Marginal => {
                let s = self.total().draws();
                (0..m)
                    .map(|j| {
                        let without: Vec<f64> = s
                            .iter()
                            .zip(column(j))
                            .map(|(total, x)| total - x)
                            .collect();
                        total - measure(d, without)
                    })
                    .collect()
            }
            AllocationMethod::Shapley => {
                if m > SHAPLEY_MAX_COMPONENTS {
                    return Err(invalid(
                        "components",
                        m as f64,
                        "Shapley allocation takes at most 12 components",
                    ));
                }
                shapley(d, draws, m)
            }
        };
        Ok(Allocation {
            method,
            total,
            standalone,
            allocated,
        })
    }
}

/// `ρ` of equally likely draws, in any order.
fn measure(d: &Distortion, mut draws: Vec<f64>) -> f64 {
    draws.sort_by(f64::total_cmp);
    d.apply_sorted(&draws)
}

/// Shapley values of `v(T) = ρ(Σ_{j∈T} X_j)` from all `2^m` coalitions;
/// `draws` is row-major, `m` per simulation.
fn shapley(d: &Distortion, draws: &[f64], m: usize) -> Vec<f64> {
    let n = draws.len() / m;
    let coalitions = 1usize << m;
    let mut value = vec![0.0; coalitions];
    for (mask, v) in value.iter_mut().enumerate().skip(1) {
        let sums = (0..n)
            .map(|i| {
                let row = &draws[i * m..(i + 1) * m];
                (0..m).filter(|j| mask >> j & 1 == 1).map(|j| row[j]).sum()
            })
            .collect();
        *v = measure(d, sums);
    }
    // weight(|T|) = |T|! (m − |T| − 1)! / m!
    let factorial = |k: usize| (1..=k).map(|i| i as f64).product::<f64>();
    let weight: Vec<f64> = (0..m)
        .map(|t| factorial(t) * factorial(m - t - 1) / factorial(m))
        .collect();
    (0..m)
        .map(|j| {
            (0..coalitions)
                .filter(|mask| mask >> j & 1 == 0)
                .map(|mask| {
                    let size = mask.count_ones() as usize;
                    weight[size] * (value[mask | 1 << j] - value[mask])
                })
                .sum()
        })
        .collect()
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
    use crate::{GaussianCopula, KeyValue, Lognormal, Provenance, copula};

    fn three_lines() -> PredictiveDistribution {
        let marginals = [
            Lognormal::from_mean_cv(10.0, 0.3).unwrap(),
            Lognormal::from_mean_cv(20.0, 0.8).unwrap(),
            Lognormal::from_mean_cv(5.0, 2.0).unwrap(),
        ];
        let corr = [1.0, 0.5, 0.2, 0.5, 1.0, 0.4, 0.2, 0.4, 1.0];
        let c = GaussianCopula::new(&corr, 3).unwrap();
        copula::simulate(
            &c,
            &[&marginals[0], &marginals[1], &marginals[2]],
            vec!["line".into()],
            ["a", "b", "c"]
                .iter()
                .map(|&l| vec![KeyValue::from(l)])
                .collect(),
            50_000,
            11,
            Provenance::new("test"),
        )
        .unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * b.abs().max(1.0)
    }

    #[test]
    fn full_allocations_sum_to_the_total() {
        let pd = three_lines();
        let d = Distortion::tvar(0.99).unwrap();
        for method in [
            AllocationMethod::Euler,
            AllocationMethod::Covariance,
            AllocationMethod::Proportional,
            AllocationMethod::Shapley,
        ] {
            let a = pd.capital(&d, method).unwrap();
            assert!(close(a.allocated.iter().sum(), a.total), "{method:?}");
            assert!(a.diversification_benefit() > 0.0);
        }
    }

    #[test]
    fn euler_matches_allocate_and_marginal_falls_short() {
        let pd = three_lines();
        let d = Distortion::wang(0.5).unwrap();
        let euler = pd.capital(&d, AllocationMethod::Euler).unwrap();
        assert_eq!(euler.allocated, pd.allocate(&d));
        let marginal = pd.capital(&d, AllocationMethod::Marginal).unwrap();
        // Subadditivity: each marginal is at most the stand-alone measure,
        // and together they leave capital unallocated.
        for (m, s) in marginal.allocated.iter().zip(&marginal.standalone) {
            assert!(m <= s);
        }
        assert!(marginal.allocated.iter().sum::<f64>() < marginal.total);
    }

    #[test]
    fn shapley_by_hand_for_two_components() {
        // Two components: φ_1 = (ρ(X1) + ρ(S) − ρ(X2)) / 2.
        let pd = three_lines();
        let d = Distortion::tvar(0.95).unwrap();
        let pd2 = {
            let m = pd.n_components();
            let draws: Vec<f64> = pd
                .draw_matrix()
                .chunks_exact(m)
                .flat_map(|r| [r[0], r[1] + r[2]])
                .collect();
            PredictiveDistribution::from_draws(
                vec!["line".into()],
                vec![vec![KeyValue::from("a")], vec![KeyValue::from("bc")]],
                draws,
                Provenance::new("test"),
            )
            .unwrap()
        };
        let a = pd2.capital(&d, AllocationMethod::Shapley).unwrap();
        let want = 0.5 * (a.standalone[0] + a.total - a.standalone[1]);
        assert!(close(a.allocated[0], want));
    }

    #[test]
    fn comonotonic_components_have_no_benefit() {
        // X2 = 2 X1 in every simulation: TVaR is additive.
        let x: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.37 % 11.0).collect();
        let draws: Vec<f64> = x.iter().flat_map(|&v| [v, 2.0 * v]).collect();
        let pd = PredictiveDistribution::from_draws(
            vec!["line".into()],
            vec![vec![KeyValue::from("a")], vec![KeyValue::from("b")]],
            draws,
            Provenance::new("test"),
        )
        .unwrap();
        let d = Distortion::tvar(0.9).unwrap();
        for method in [
            AllocationMethod::Euler,
            AllocationMethod::Covariance,
            AllocationMethod::Shapley,
        ] {
            let a = pd.capital(&d, method).unwrap();
            assert!(a.diversification_benefit().abs() < 1e-9);
            assert!(close(a.allocated[0], a.standalone[0]), "{method:?}");
        }
    }

    #[test]
    fn rejects_degenerate_inputs() {
        let pd = PredictiveDistribution::from_draws(
            vec!["line".into()],
            vec![vec![KeyValue::from("a")], vec![KeyValue::from("b")]],
            vec![1.0, -1.0, 2.0, -2.0],
            Provenance::new("test"),
        )
        .unwrap();
        let d = Distortion::tvar(0.5).unwrap();
        assert!(pd.capital(&d, AllocationMethod::Covariance).is_err());
    }
}
