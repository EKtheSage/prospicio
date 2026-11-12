//! The log posterior of the compartmental model, written once over `Scalar`, with its layout of parameters.

use prospicio_bayes::nuts::PosteriorDraws;
use prospicio_core::StreamRng;
use prospicio_math::special::norm_quantile;

use super::super::dual::{CHUNK, Dual, gradient};
use super::super::laplace::{Posterior, mode};
use super::super::scalar::Scalar;
use super::data::{Data, Future, Hierarchy, Response};
use super::one_os;
use super::priors::{
    CompartmentalPriors, PositivePrior, half_median, half_on_log, lognormal, real_prior,
};
use super::settings::CompartmentalStart;
use crate::error::{Error, Result};

/// The four parameters, in the order of [`CompartmentalPriors::positive`].
pub(super) const KER: usize = 0;
pub(super) const KP: usize = 1;
pub(super) const RLR: usize = 2;
pub(super) const RRF: usize = 3;
/// Their names in the posterior's origin values (`RLR[1990]`) and, short,
/// in the names of their effects (`tau_dev_ker`, `dev_ker[24]`).
const NATURAL: [&str; 4] = ["k_er", "k_p", "RLR", "RRF"];
const SHORT: [&str; 4] = ["ker", "kp", "RLR", "RRF"];

/// An observed cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Obs {
    pub(super) origin: usize,
    /// The age index, into [`Model::times`].
    pub(super) k: usize,
    pub(super) t: f64,
    /// For an incremental paid cell after the first age, the time of the
    /// age before.
    pub(super) t_before: Option<f64>,
    /// The response: outstanding, cumulative or incremental paid, as an
    /// amount or a loss ratio.
    pub(super) y: f64,
    /// `ln y` under the log-normal likelihood, 0 otherwise.
    pub(super) ln_y: f64,
    /// The premium under the amount scale, 1 under the loss-ratio scale.
    pub(super) w: f64,
    /// Paid (true) or outstanding.
    pub(super) paid: bool,
}

/// Positions in `theta` past the fixed ones, and in a natural-scale draw.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Layout {
    /// `log τ` of the origin effects on `k_er` and `k_p`.
    pub(super) tau_rate: [Option<usize>; 2],
    /// `log τ` of the development effects on each parameter.
    pub(super) tau_dev: [Option<usize>; 4],
    /// The first `z`: the origin pairs of `RLR` and `RRF`.
    pub(super) z0: usize,
    /// The first `z` of the origin effects on `k_er` and `k_p`.
    pub(super) z_rate: [Option<usize>; 2],
    /// The first `z` of the development effects on each parameter.
    pub(super) z_dev: [Option<usize>; 4],
    /// The first `log σ`, one per fitted type ([`Response::sigma_types`]):
    /// `log σ_OS` then `log σ_PD` with both fitted.
    log_sigma0: usize,
    /// The number of `log σ` (and of `σ` after the four population values
    /// of a natural-scale draw).
    pub(super) n_sigma: usize,
    /// In a natural-scale draw: the first origin value of each parameter.
    nat_origin: [Option<usize>; 4],
    /// The first development effect (on the `o` scale) of each parameter.
    pub(super) nat_dev: [Option<usize>; 4],
    /// `b_ker`, then `b_kp`, `b_RLR`, `b_RRF`, `tau_RLR` and `tau_RRF`.
    nat_b0: usize,
    nat_tau_rate: [Option<usize>; 2],
    nat_rho: usize,
    pub(super) nat_tau_dev: [Option<usize>; 4],
    /// The first `log_sigma`, as in `theta`.
    nat_log_sigma: usize,
}

impl Layout {
    fn new(
        priors: &CompartmentalPriors,
        n_levels: usize,
        n_dev_levels: usize,
        n_sigma: usize,
    ) -> Self {
        let rate = priors.rate_effect_sd();
        let dev = priors.development_effect_sd;
        let mut at = ATANH_RHO + 1;
        let next = |on: bool, size: usize, at: &mut usize| {
            on.then(|| {
                *at += size;
                *at - size
            })
        };
        let tau_rate = rate.map(|p| next(p.is_some(), 1, &mut at));
        let tau_dev = dev.map(|p| next(p.is_some(), 1, &mut at));
        let z0 = at;
        at += 2 * n_levels;
        let z_rate = rate.map(|p| next(p.is_some(), n_levels, &mut at));
        let z_dev = dev.map(|p| next(p.is_some(), n_dev_levels, &mut at));
        let log_sigma0 = at;
        let head = 4 + n_sigma;
        let mut n = head + 2 * n_levels;
        let nat_origin = [
            next(rate[0].is_some(), n_levels, &mut n),
            next(rate[1].is_some(), n_levels, &mut n),
            Some(head),
            Some(head + n_levels),
        ];
        let nat_dev = dev.map(|p| next(p.is_some(), n_dev_levels, &mut n));
        let nat_b0 = n;
        n += 6;
        let nat_tau_rate = rate.map(|p| next(p.is_some(), 1, &mut n));
        let nat_rho = n;
        n += 1;
        let nat_tau_dev = dev.map(|p| next(p.is_some(), 1, &mut n));
        Self {
            tau_rate,
            tau_dev,
            z0,
            z_rate,
            z_dev,
            log_sigma0,
            n_sigma,
            nat_origin,
            nat_dev,
            nat_b0,
            nat_tau_rate,
            nat_rho,
            nat_tau_dev,
            nat_log_sigma: n,
        }
    }
}

/// The log posterior on the unconstrained `theta = (b_ker | delta, b_kp,
/// b_RLR, b_RRF, log τ_RLR, log τ_RRF, atanh ρ, [log τ_ker, log τ_kp,
/// log τ_dev_ker, .., log τ_dev_RRF], z_11, z_12, .., z_m1, z_m2, [z_ker,
/// z_kp by origin, z_dev_ker, .., z_dev_RRF by development level],
/// log σ_OS, log σ_PD)` over the `m` origins with an observed cell (with one
/// response fitted, its `log σ` alone), where
/// the bracketed blocks are there only for the effects the priors have and
/// under [`RateOrder::Ordered`] `k_er = k_p + e^delta` (the population
/// rates). A cell's parameters are `θ = m e^(s (b + u_origin + u_dev))`.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Model {
    /// Outstanding cells first, then paid, each by origin and age.
    pub(super) cells: Vec<Obs>,
    pub(super) n_outstanding: usize,
    pub(super) premium: Vec<f64>,
    /// Each origin's latest observed cumulative paid (0 with none).
    pub(super) latest_paid: Vec<f64>,
    /// Each origin's level (position among the origins with data), if any.
    pub(super) level: Vec<Option<usize>>,
    /// The origin of each level.
    pub(super) level_origin: Vec<usize>,
    /// The origin label of each level, for the names.
    labels: Vec<String>,
    /// Each age index's development level (position among the ages with an
    /// observed cell), if any.
    pub(super) dev_level: Vec<Option<usize>>,
    /// The group of each age index, observed or not: the age itself with a
    /// level for each age, the development year with levels by year
    /// ([`DevelopmentLevels`]); ages of a group share one effect.
    dev_group: Vec<usize>,
    /// The first age in months of each development level, for the names.
    pub(super) dev_labels: Vec<String>,
    /// Development time of each age index of the square, in years.
    pub(super) times: Vec<f64>,
    /// The age indices that hold an observed cell, `0..n_observed`.
    pub(super) n_observed: usize,
    pub(super) priors: CompartmentalPriors,
    pub(super) ordered: bool,
    pub(super) response: Response,
    /// Whether any effect varies by cell beyond `RLR` and `RRF` by origin
    /// (model 1 has none: its rates are shared, one curve per age).
    pub(super) cellwise: bool,
    pub(super) layout: Layout,
}

/// The most entries of `theta` without a `z` (model 2: 13 and the two
/// `log σ`), the dual width of [`Model::prior_globals`].
const GLOBALS: usize = 16;

/// Positions in `theta`.
pub(super) const B_KER: usize = 0;
pub(super) const B_KP: usize = 1;
pub(super) const B_RLR: usize = 2;
pub(super) const B_RRF: usize = 3;
pub(super) const LOG_TAU: usize = 4;
pub(super) const ATANH_RHO: usize = 6;

impl Model {
    /// The model of `data`'s observed cells: a level (with its `z`) for
    /// each origin and each age with an observed cell.
    pub(super) fn new(
        data: &Data,
        priors: CompartmentalPriors,
        ordered: bool,
        response: Response,
    ) -> Result<Self> {
        Self::build(data, priors, ordered, response, false)
    }

    /// The model of `data`'s observed cells with a level for every origin
    /// of the premium column and every age of the square, observed or not
    /// (`docs/design/reserving-bayes.md`, decision 20): an effect no cell
    /// informs has only its prior, so this posterior is [`new`](Self::new)'s
    /// with independent standard normal `z` for the unobserved levels.
    pub(super) fn full(
        data: &Data,
        priors: CompartmentalPriors,
        ordered: bool,
        response: Response,
    ) -> Result<Self> {
        Self::build(data, priors, ordered, response, true)
    }

    fn build(
        data: &Data,
        priors: CompartmentalPriors,
        ordered: bool,
        response: Response,
        all_levels: bool,
    ) -> Result<Self> {
        let n = data.origins.len();
        let mut has = vec![all_levels; n];
        let mut has_age = vec![all_levels; data.n_ages];
        // Only the fitted cells: under incurred alone the paid cells give
        // each origin's latest paid and nothing else.
        let none: &[(usize, usize, f64)] = &[];
        let fitted_os = if response.second {
            &data.outstanding[..]
        } else {
            none
        };
        let fitted_paid = if response.paid { &data.paid[..] } else { none };
        for c in fitted_os.iter().chain(fitted_paid) {
            has[c.0] = true;
            has_age[c.1] = true;
        }
        let mut level = vec![None; n];
        let mut level_origin = Vec::new();
        for i in 0..n {
            if has[i] {
                level[i] = Some(level_origin.len());
                level_origin.push(i);
            }
        }
        let labels = level_origin
            .iter()
            .map(|&i| data.origins[i].to_string())
            .collect();
        // The development levels: one for each age with an observed cell,
        // or for each development year with one (decision 14). An age with
        // no cell of its own in an observed year takes the year's level.
        let dev_group: Vec<usize> = (0..data.n_ages)
            .map(|k| response.levels.group(data.age(k), k))
            .collect();
        let n_groups = dev_group.iter().max().map_or(0, |g| g + 1);
        let mut group_has = vec![false; n_groups];
        for (k, &g) in dev_group.iter().enumerate() {
            group_has[g] |= has_age[k];
        }
        let mut dev_level = vec![None; data.n_ages];
        let mut dev_labels = Vec::new();
        let mut level_of_group: Vec<Option<usize>> = vec![None; n_groups];
        for (k, &g) in dev_group.iter().enumerate() {
            if group_has[g] {
                let level = *level_of_group[g].get_or_insert_with(|| {
                    dev_labels.push(data.age(k).to_string());
                    dev_labels.len() - 1
                });
                dev_level[k] = Some(level);
            }
        }
        let times: Vec<f64> = (0..data.n_ages).map(|k| data.grain.time(k)).collect();
        let log_normal = response.log_normal();
        let obs =
            |origin: usize, k: usize, value: f64, before: Option<f64>, paid: bool| -> Result<Obs> {
                let amount = before.map_or(value, |b| value - b);
                let w = if response.loss_ratio {
                    1.0
                } else {
                    data.premium[origin]
                };
                let y = if response.loss_ratio {
                    amount / data.premium[origin]
                } else {
                    amount
                };
                if log_normal && !(y > 0.0 && y.is_finite()) {
                    let what = match (paid, before.is_some()) {
                        (false, _) if response.incurred => "incurred",
                        (false, _) => "outstanding",
                        (true, true) => "incremental paid",
                        (true, false) => "paid",
                    };
                    return Err(Error::Bayes(format!(
                        "origin {} at development age {} months has {what} {amount}; the \
                     log-normal likelihood needs every observed response positive",
                        data.origins[origin],
                        data.age(k)
                    )));
                }
                Ok(Obs {
                    origin,
                    k,
                    t: times[k],
                    t_before: before.map(|_| times[k - 1]),
                    y,
                    ln_y: if log_normal { y.ln() } else { 0.0 },
                    w,
                    paid,
                })
            };
        let mut os = fitted_os
            .iter()
            .map(|&(i, k, v)| obs(i, k, v, None, false))
            .collect::<Result<Vec<Obs>>>()?;
        let mut pd = fitted_paid
            .iter()
            .map(|&(i, k, v)| {
                let before = if response.incremental && k > 0 {
                    let b = data
                        .paid
                        .iter()
                        .find(|c| c.0 == i && c.1 == k - 1)
                        .ok_or_else(|| {
                            Error::Bayes(format!(
                                "origin {} has paid at development age {} months but none at \
                                 {} months; incremental paid needs the paid of the age before \
                                 each paid cell",
                                data.origins[i],
                                data.age(k),
                                data.age(k - 1)
                            ))
                        })?;
                    Some(b.2)
                } else {
                    None
                };
                obs(i, k, v, before, true)
            })
            .collect::<Result<Vec<Obs>>>()?;
        os.sort_by_key(|c| (c.origin, c.k));
        pd.sort_by_key(|c| (c.origin, c.k));
        let n_outstanding = os.len();
        let mut cells = os;
        cells.extend(pd);
        let n_observed = cells.iter().map(|c| c.k + 1).max().unwrap_or(0);
        let layout = Layout::new(
            &priors,
            level_origin.len(),
            dev_labels.len(),
            response.n_sigma(),
        );
        let cellwise = priors.hierarchy() != Hierarchy::Model1;
        debug_assert!(layout.z0 + 2 <= GLOBALS);
        Ok(Self {
            cells,
            n_outstanding,
            premium: data.premium.clone(),
            latest_paid: data
                .latest_paid
                .iter()
                .map(|l| l.map_or(0.0, |(_, v)| v))
                .collect(),
            level,
            level_origin,
            labels,
            dev_level,
            dev_group,
            dev_labels,
            times,
            n_observed,
            priors,
            ordered,
            response,
            cellwise,
            layout,
        })
    }

    /// The prior terms of [`log_density`](Self::log_density) that do not
    /// involve a `z`, at `g`: `theta[..z0]` followed by `log σ_OS` and
    /// `log σ_PD`.
    fn prior_globals<S: Scalar>(&self, g: &[S]) -> S {
        let pr = &self.priors;
        let lay = &self.layout;
        let mut f = S::from_f64(0.0);
        for &b in &g[B_KP..=B_RRF] {
            f -= b * b * 0.5;
        }
        let (k_er, _, b_ker) = self.rates(g);
        f -= b_ker * b_ker * 0.5;
        if self.ordered {
            f += g[B_KER] - k_er.ln();
        }
        f += half_on_log(g[LOG_TAU], pr.rlr_effect_sd)
            + half_on_log(g[LOG_TAU + 1], pr.rrf_effect_sd);
        let rho = g[ATANH_RHO].tanh();
        f += (-(rho * rho) + 1.0).ln() * pr.lkj_eta;
        for (at, prior) in lay.tau_rate.iter().zip(pr.rate_effect_sd()) {
            if let (Some(at), Some(prior)) = (at, prior) {
                f += half_on_log(g[*at], prior);
            }
        }
        for (at, prior) in lay.tau_dev.iter().zip(pr.development_effect_sd) {
            if let (Some(at), Some(prior)) = (at, prior) {
                f += half_on_log(g[*at], prior);
            }
        }
        self.sigma_prior(f, &g[lay.z0..])
    }

    /// `f` plus the prior terms of the `log σ` (`log_sigma`, in the order
    /// of [`Response::sigma_types`]), added one by one: `log_sigma[0]` of
    /// the priors on the second response's, `log_sigma[1]` on paid's.
    fn sigma_prior<S: Scalar>(&self, f: S, log_sigma: &[S]) -> S {
        let pr = &self.priors;
        self.response
            .sigma_types()
            .zip(log_sigma)
            .fold(f, |f, (paid, &l)| {
                f + real_prior(l, pr.log_sigma[usize::from(paid)])
            })
    }

    /// The `log σ` slot of a fitted cell type.
    fn slot(&self, paid: bool) -> usize {
        self.response
            .sigma_slot(paid)
            .expect("only fitted cells have a sigma")
    }

    /// The `σ` of each type, the second response's then paid's, from a
    /// natural-scale draw; 0 for a type not fitted (its cells, if any are
    /// predicted, are their expected values).
    pub(super) fn sigmas(&self, nat: &[f64]) -> [f64; 2] {
        [false, true].map(|paid| self.response.sigma_slot(paid).map_or(0.0, |s| nat[4 + s]))
    }

    /// One cell's term of the log likelihood, `-r² / 2 - log σ`, from the
    /// `o` values of its four parameters and its type's `log σ`, with the
    /// same operations as [`log_density`](Self::log_density)'s cell by cell
    /// path.
    fn cell_term<S: Scalar>(&self, c: &Obs, o: [S; 4], ls: S) -> S {
        let ms = self.priors.positive().map(lognormal);
        let th: [S; 4] = std::array::from_fn(|p| (o[p] * ms[p].1).exp() * ms[p].0);
        let r = self.residual(c, self.cell_mean(c, th)) * (-ls).exp();
        -(r * r * 0.5) - ls
    }

    /// The log posterior and its gradient with effects by cell: the prior
    /// terms without a `z` in one dual pass over their at most `GLOBALS`
    /// inputs, the `z` terms by hand, each cell's term in a dual pass over
    /// its four `o` values and its `log σ`, and the chain rule from the `o`
    /// values (`o = b + u_origin + u_dev`, each effect `τ z`, and `RRF`'s
    /// origin effect `τ₂ (ρ z₁ + sqrt(1 - ρ²) z₂)`) to `theta` by hand,
    /// summed by level. The same value and gradient as the chunked dual pass
    /// over [`log_density`](Self::log_density), to rounding (a unit test
    /// checks it).
    pub(super) fn cellwise_gradient(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
        let lay = &self.layout;
        let (z0, ls, ns) = (lay.z0, self.log_sigma0(), lay.n_sigma);
        grad.fill(0.0);
        let mut globals = theta[..z0].to_vec();
        globals.extend_from_slice(&theta[ls..ls + ns]);
        let mut g = vec![0.0; globals.len()];
        let mut f = gradient::<GLOBALS>(&globals, &mut g, |x| self.prior_globals(x));
        grad[..z0].copy_from_slice(&g[..z0]);
        grad[ls..ls + ns].copy_from_slice(&g[z0..z0 + ns]);
        for j in z0..ls {
            f -= theta[j] * theta[j] * 0.5;
            grad[j] -= theta[j];
        }
        // The cells, with the derivatives by `o` summed by level.
        let (u, v) = self.effects(theta);
        let (_, _, b_ker) = self.rates(theta);
        let b = [b_ker, theta[B_KP], theta[B_RLR], theta[B_RRF]];
        let mut by_origin = vec![[0.0; 4]; self.n_levels()];
        let mut by_dev = vec![[0.0; 4]; self.n_dev_levels()];
        let mut by_b = [0.0; 4];
        for c in &self.cells {
            let j = self.level[c.origin].expect("an observed origin has a level");
            let d = self.dev_level[c.k].expect("an observed age has a level");
            let o: [Dual<5>; 4] =
                std::array::from_fn(|p| Dual::variable(b[p] + u[p][j] + v[p][d], p));
            let at = ls + self.slot(c.paid);
            let t = self.cell_term(c, o, Dual::variable(theta[at], 4));
            f += t.v;
            grad[at] += t.d[4];
            for p in 0..4 {
                by_origin[j][p] += t.d[p];
                by_dev[d][p] += t.d[p];
                by_b[p] += t.d[p];
            }
        }
        // The intercepts: b_ker through the rate map (under the ordered
        // rates it depends on delta and b_kp).
        let rates = [
            Dual::<2>::variable(theta[B_KER], 0),
            Dual::variable(theta[B_KP], 1),
        ];
        let (_, _, b_ker) = self.rates(&rates);
        grad[B_KER] += by_b[KER] * b_ker.d[0];
        grad[B_KP] += by_b[KER] * b_ker.d[1] + by_b[KP];
        grad[B_RLR] += by_b[RLR];
        grad[B_RRF] += by_b[RRF];
        // The origin effects.
        let rho = theta[ATANH_RHO].tanh();
        let rho_c = (1.0 - rho * rho).sqrt();
        let (tau1, tau2) = (theta[LOG_TAU].exp(), theta[LOG_TAU + 1].exp());
        for (j, gj) in by_origin.iter().enumerate() {
            let (a, bz) = (lay.z0 + 2 * j, lay.z0 + 2 * j + 1);
            let (z1, z2) = (theta[a], theta[bz]);
            grad[LOG_TAU] += gj[RLR] * u[RLR][j];
            grad[LOG_TAU + 1] += gj[RRF] * u[RRF][j];
            grad[a] += gj[RLR] * tau1 + gj[RRF] * tau2 * rho;
            grad[bz] += gj[RRF] * tau2 * rho_c;
            // dρ/dv = 1 - ρ², d sqrt(1 - ρ²)/dv = -ρ sqrt(1 - ρ²).
            grad[ATANH_RHO] += gj[RRF] * tau2 * (z1 * rho_c * rho_c - z2 * rho * rho_c);
            for r in [KER, KP] {
                if let (Some(l), Some(z)) = (lay.tau_rate[r], lay.z_rate[r]) {
                    grad[l] += gj[r] * u[r][j];
                    grad[z + j] += gj[r] * theta[l].exp();
                }
            }
        }
        // The development effects.
        for (d, gd) in by_dev.iter().enumerate() {
            for p in 0..4 {
                if let (Some(l), Some(z)) = (lay.tau_dev[p], lay.z_dev[p]) {
                    grad[l] += gd[p] * v[p][d];
                    grad[z + d] += gd[p] * theta[l].exp();
                }
            }
        }
        f
    }

    pub(super) fn n_levels(&self) -> usize {
        self.level_origin.len()
    }

    pub(super) fn n_dev_levels(&self) -> usize {
        self.dev_labels.len()
    }

    /// Position of the first `log σ` ([`Layout::log_sigma0`]).
    pub(super) fn log_sigma0(&self) -> usize {
        self.layout.log_sigma0
    }

    /// Whether any development effect is in the priors.
    pub(super) fn has_dev(&self) -> bool {
        self.layout.z_dev.iter().any(Option::is_some)
    }

    /// The names of `theta`, for the pilot draws.
    pub(super) fn sampled_names(&self) -> Vec<String> {
        let lay = &self.layout;
        let mut names: Vec<String> = [
            if self.ordered { "delta" } else { "b_ker" },
            "b_kp",
            "b_RLR",
            "b_RRF",
            "log_tau_RLR",
            "log_tau_RRF",
            "atanh_rho",
        ]
        .map(String::from)
        .to_vec();
        for r in [KER, KP] {
            if lay.tau_rate[r].is_some() {
                names.push(format!("log_tau_{}", SHORT[r]));
            }
        }
        for (on, short) in lay.tau_dev.iter().zip(SHORT) {
            if on.is_some() {
                names.push(format!("log_tau_dev_{}", short));
            }
        }
        for j in 0..self.n_levels() {
            names.push(format!("z[{j},1]"));
            names.push(format!("z[{j},2]"));
        }
        for r in [KER, KP] {
            if lay.z_rate[r].is_some() {
                names.extend((0..self.n_levels()).map(|j| format!("z_{}[{j}]", SHORT[r])));
            }
        }
        for (on, short) in lay.z_dev.iter().zip(SHORT) {
            if on.is_some() {
                names.extend((0..self.n_dev_levels()).map(|d| format!("z_dev_{}[{d}]", short)));
            }
        }
        names.extend(self.sigma_names("log_sigma"));
        names
    }

    /// `{prefix}_os` and `{prefix}_pd`, each when its type is fitted.
    fn sigma_names(&self, prefix: &str) -> Vec<String> {
        self.response
            .sigma_types()
            .map(|paid| format!("{prefix}_{}", if paid { "pd" } else { "os" }))
            .collect()
    }

    /// The names of the natural-scale draws ([`CompartmentalFit::posterior`]).
    pub(super) fn names(&self) -> Vec<String> {
        let lay = &self.layout;
        let mut names: Vec<String> = ["k_er", "k_p", "RLR", "RRF"].map(String::from).to_vec();
        names.extend(self.sigma_names("sigma"));
        for p in [RLR, RRF, KER, KP] {
            if lay.nat_origin[p].is_some() {
                for label in self.labels.iter().take(self.n_levels()) {
                    names.push(format!("{}[{}]", NATURAL[p], label));
                }
            }
        }
        for (on, short) in lay.nat_dev.iter().zip(SHORT) {
            if on.is_some() {
                for label in &self.dev_labels {
                    names.push(format!("dev_{}[{}]", short, label));
                }
            }
        }
        names.extend(["b_ker", "b_kp", "b_RLR", "b_RRF", "tau_RLR", "tau_RRF"].map(String::from));
        for r in [KER, KP] {
            if lay.nat_tau_rate[r].is_some() {
                names.push(format!("tau_{}", SHORT[r]));
            }
        }
        names.push("rho".into());
        for (on, short) in lay.nat_tau_dev.iter().zip(SHORT) {
            if on.is_some() {
                names.push(format!("tau_dev_{}", short));
            }
        }
        names.extend(self.sigma_names("log_sigma"));
        names
    }

    /// The rates `(k_er, k_p)` and `b_ker`, from `theta`.
    pub(super) fn rates<S: Scalar>(&self, theta: &[S]) -> (S, S, S) {
        let (m_er, s_er) = lognormal(self.priors.k_er);
        let (m_p, s_p) = lognormal(self.priors.k_p);
        let k_p = (theta[B_KP] * s_p).exp() * m_p;
        if self.ordered {
            let k_er = k_p + theta[B_KER].exp();
            let b_ker = (k_er / m_er).ln() / s_er;
            (k_er, k_p, b_ker)
        } else {
            let b_ker = theta[B_KER];
            ((b_ker * s_er).exp() * m_er, k_p, b_ker)
        }
    }

    /// The effects on the `o` scale from `theta`: by origin level and by
    /// development level, for each of the four parameters (0 where the
    /// priors have none).
    #[allow(clippy::type_complexity)]
    pub(super) fn effects<S: Scalar>(&self, theta: &[S]) -> ([Vec<S>; 4], [Vec<S>; 4]) {
        let lay = &self.layout;
        let (m, nd) = (self.n_levels(), self.n_dev_levels());
        let zero = S::from_f64(0.0);
        let rho = theta[ATANH_RHO].tanh();
        let rho_c = (-(rho * rho) + 1.0).sqrt();
        let (tau1, tau2) = (theta[LOG_TAU].exp(), theta[LOG_TAU + 1].exp());
        let mut u: [Vec<S>; 4] = std::array::from_fn(|_| vec![zero; m]);
        for j in 0..m {
            let (z1, z2) = (theta[lay.z0 + 2 * j], theta[lay.z0 + 2 * j + 1]);
            u[RLR][j] = tau1 * z1;
            u[RRF][j] = tau2 * (rho * z1 + rho_c * z2);
        }
        for r in [KER, KP] {
            if let (Some(l), Some(z)) = (lay.tau_rate[r], lay.z_rate[r]) {
                let tau = theta[l].exp();
                for (j, u) in u[r].iter_mut().enumerate() {
                    *u = tau * theta[z + j];
                }
            }
        }
        let mut v: [Vec<S>; 4] = std::array::from_fn(|_| vec![zero; nd]);
        for p in 0..4 {
            if let (Some(l), Some(z)) = (lay.tau_dev[p], lay.z_dev[p]) {
                let tau = theta[l].exp();
                for (d, v) in v[p].iter_mut().enumerate() {
                    *v = tau * theta[z + d];
                }
            }
        }
        (u, v)
    }

    /// The mean of a cell's response given its parameters `(k_er, k_p, RLR,
    /// RRF)`: outstanding, cumulative paid or incremental paid (the closed
    /// form at the cell's age less the same at the age before, both with
    /// these parameters), times the premium on the amount scale.
    pub(super) fn cell_mean<S: Scalar>(&self, c: &Obs, [k_er, k_p, rlr, rrf]: [S; 4]) -> S {
        let base = rlr * c.w;
        if c.paid {
            let (_, pd) = one_os(k_er, k_p, c.t);
            let pd = match c.t_before {
                Some(t) => pd - one_os(k_er, k_p, t).1,
                None => pd,
            };
            base * rrf * pd
        } else if self.response.incurred {
            let (os, pd) = one_os(k_er, k_p, c.t);
            base * (os + rrf * pd)
        } else {
            base * one_os(k_er, k_p, c.t).0
        }
    }

    /// The residual of a cell on the likelihood's scale: `y - μ`, or
    /// `ln y - ln μ`.
    pub(super) fn residual<S: Scalar>(&self, c: &Obs, mean: S) -> S {
        if self.response.log_normal() {
            -(mean.ln()) + c.ln_y
        } else {
            -mean + c.y
        }
    }

    /// The log posterior, written once over [`Scalar`].
    pub(super) fn log_density<S: Scalar>(&self, theta: &[S]) -> S {
        let pr = &self.priors;
        let lay = &self.layout;
        let mut f = S::from_f64(0.0);
        for &b in &theta[B_KP..=B_RRF] {
            f -= b * b * 0.5;
        }
        let (k_er, k_p, b_ker) = self.rates(theta);
        f -= b_ker * b_ker * 0.5;
        if self.ordered {
            // The Jacobian of delta -> b_ker: e^delta / (s_ker k_er), less
            // its constant.
            f += theta[B_KER] - k_er.ln();
        }
        let (l1, l2) = (theta[LOG_TAU], theta[LOG_TAU + 1]);
        f += half_on_log(l1, pr.rlr_effect_sd) + half_on_log(l2, pr.rrf_effect_sd);
        // LKJ(eta) on a 2x2 correlation is (1 - ρ²)^(eta - 1); with the
        // Jacobian 1 - ρ² of ρ = tanh(v), eta log(1 - ρ²).
        let rho = theta[ATANH_RHO].tanh();
        let rho_c2 = -(rho * rho) + 1.0;
        f += rho_c2.ln() * pr.lkj_eta;
        for (at, prior) in lay.tau_rate.iter().zip(pr.rate_effect_sd()) {
            if let (Some(at), Some(prior)) = (at, prior) {
                f += half_on_log(theta[*at], prior);
            }
        }
        for (at, prior) in lay.tau_dev.iter().zip(pr.development_effect_sd) {
            if let (Some(at), Some(prior)) = (at, prior) {
                f += half_on_log(theta[*at], prior);
            }
        }
        let ls = self.log_sigma0();
        for &z in &theta[lay.z0..ls] {
            f -= z * z * 0.5;
        }
        let log_sigma = &theta[ls..ls + lay.n_sigma];
        f += self.sigma_prior(S::from_f64(0.0), log_sigma);
        // 1 / σ of each fitted type, by slot.
        let inv: Vec<S> = log_sigma.iter().map(|&l| (-l).exp()).collect();
        let (u, v) = self.effects(theta);
        // The number of cells of each slot.
        let mut n_cells = [0.0; 2];
        if self.cellwise {
            // Each cell its own parameters.
            let b = [b_ker, theta[B_KP], theta[B_RLR], theta[B_RRF]];
            let ms = pr.positive().map(lognormal);
            for c in &self.cells {
                let j = self.level[c.origin].expect("an observed origin has a level");
                let d = self.dev_level[c.k].expect("an observed age has a level");
                let th: [S; 4] =
                    std::array::from_fn(|p| ((b[p] + u[p][j] + v[p][d]) * ms[p].1).exp() * ms[p].0);
                let s = self.slot(c.paid);
                let r = self.residual(c, self.cell_mean(c, th)) * inv[s];
                n_cells[s] += 1.0;
                f -= r * r * 0.5;
            }
        } else {
            let (m_rlr, s_rlr) = lognormal(pr.rlr);
            let (m_rrf, s_rrf) = lognormal(pr.rrf);
            // The rates are shared, so the closed form is one per age.
            let curves: Vec<(S, S)> = self.times[..self.n_observed]
                .iter()
                .map(|&t| one_os(k_er, k_p, t))
                .collect();
            // Π RLR and Π RLR RRF of each level (RLR and RLR RRF on the
            // loss-ratio scale).
            let scale: Vec<(S, S)> = self
                .level_origin
                .iter()
                .enumerate()
                .map(|(j, &i)| {
                    let w = if self.response.loss_ratio {
                        m_rlr
                    } else {
                        m_rlr * self.premium[i]
                    };
                    let rlr = ((theta[B_RLR] + u[RLR][j]) * s_rlr).exp() * w;
                    let rrf = ((theta[B_RRF] + u[RRF][j]) * s_rrf).exp() * m_rrf;
                    (rlr, rlr * rrf)
                })
                .collect();
            for c in &self.cells {
                let (os, pd) = curves[c.k];
                let (a, b) = scale[self.level[c.origin].expect("an observed origin has a level")];
                let s = self.slot(c.paid);
                n_cells[s] += 1.0;
                let mean = if !c.paid {
                    if self.response.incurred {
                        a * os + b * pd
                    } else {
                        a * os
                    }
                } else if c.t_before.is_some() {
                    b * (pd - curves[c.k - 1].1)
                } else {
                    b * pd
                };
                let r = self.residual(c, mean) * inv[s];
                f -= r * r * 0.5;
            }
        }
        log_sigma
            .iter()
            .zip(n_cells)
            .fold(f, |f, (&l, n)| f - l * n)
    }

    /// `theta` on the natural scale ([`names`](Self::names)).
    pub(super) fn natural(&self, theta: &[f64]) -> Vec<f64> {
        let lay = &self.layout;
        let ms = self.priors.positive().map(lognormal);
        let (m_rlr, s_rlr) = ms[RLR];
        let (m_rrf, s_rrf) = ms[RRF];
        let (k_er, k_p, b_ker) = self.rates(theta);
        let ls = self.log_sigma0();
        let (u, v) = self.effects(theta);
        let log_sigma = &theta[ls..ls + lay.n_sigma];
        let mut out = vec![
            k_er,
            k_p,
            m_rlr * (s_rlr * theta[B_RLR]).exp(),
            m_rrf * (s_rrf * theta[B_RRF]).exp(),
        ];
        out.extend(log_sigma.iter().map(|l| l.exp()));
        let b = [b_ker, theta[B_KP], theta[B_RLR], theta[B_RRF]];
        for p in [RLR, RRF, KER, KP] {
            if lay.nat_origin[p].is_some() {
                out.extend(u[p].iter().map(|u| ms[p].0 * (ms[p].1 * (b[p] + u)).exp()));
            }
        }
        for (on, v) in lay.nat_dev.iter().zip(&v) {
            if on.is_some() {
                out.extend_from_slice(v);
            }
        }
        out.extend([
            b_ker,
            theta[B_KP],
            theta[B_RLR],
            theta[B_RRF],
            theta[LOG_TAU].exp(),
            theta[LOG_TAU + 1].exp(),
        ]);
        for l in lay.tau_rate.into_iter().flatten() {
            out.push(theta[l].exp());
        }
        out.push(theta[ATANH_RHO].tanh());
        for l in lay.tau_dev.into_iter().flatten() {
            out.push(theta[l].exp());
        }
        out.extend_from_slice(log_sigma);
        out
    }

    /// The origin values `(k_er, k_p, RLR, RRF)` of origin `i` (at a zero
    /// development effect) from a natural-scale draw: its own for an origin
    /// with data, else new effects from the population drawn from `rng`
    /// (two normals for `RLR` and `RRF`, then one for each rate with an
    /// origin effect).
    pub(super) fn origin_values(&self, nat: &[f64], i: usize, rng: &mut StreamRng) -> [f64; 4] {
        let lay = &self.layout;
        if let Some(j) = self.level[i] {
            return std::array::from_fn(|p| lay.nat_origin[p].map_or(nat[p], |at| nat[at + j]));
        }
        let b = lay.nat_b0;
        let (b_rlr, b_rrf, tau1, tau2, rho) = (
            nat[b + 2],
            nat[b + 3],
            nat[b + 4],
            nat[b + 5],
            nat[lay.nat_rho],
        );
        let ms = self.priors.positive().map(lognormal);
        let (m_rlr, s_rlr) = ms[RLR];
        let (m_rrf, s_rrf) = ms[RRF];
        let z1 = norm_quantile(rng.next_open01());
        let z2 = norm_quantile(rng.next_open01());
        let rho_c = (1.0 - rho * rho).max(0.0).sqrt();
        let mut out = [
            nat[0],
            nat[1],
            m_rlr * (s_rlr * (b_rlr + tau1 * z1)).exp(),
            m_rrf * (s_rrf * (b_rrf + tau2 * (rho * z1 + rho_c * z2))).exp(),
        ];
        for r in [KER, KP] {
            if let Some(at) = lay.nat_tau_rate[r] {
                let z = norm_quantile(rng.next_open01());
                out[r] = ms[r].0 * (ms[r].1 * (nat[b + r] + nat[at] * z)).exp();
            }
        }
        out
    }

    /// The development multipliers `e^(s v)` of each age of the square and
    /// each parameter from a natural-scale draw: its own effect for an age
    /// with data, else a new one from the population drawn from `rng` (one
    /// normal per parameter with a development effect, age by age), shared
    /// by every origin; 1 for a parameter with none.
    fn dev_multipliers(&self, nat: &[f64], rng: &mut StreamRng) -> Vec<[f64; 4]> {
        self.dev_multipliers_with(nat, |tau| tau * norm_quantile(rng.next_open01()))
    }

    /// The development multipliers of [`dev_multipliers`](Self::dev_multipliers),
    /// with `unobserved(tau)` the effect of an age with no observed cell,
    /// given its sd, age by age and parameter by parameter.
    pub(super) fn dev_multipliers_with(
        &self,
        nat: &[f64],
        mut unobserved: impl FnMut(f64) -> f64,
    ) -> Vec<[f64; 4]> {
        let lay = &self.layout;
        let ms = self.priors.positive().map(lognormal);
        // The ages of one group with no level share one new effect (a group
        // is one age under `PerAge`, so each is drawn once, in order).
        let mut new_effect: Vec<[Option<f64>; 4]> = vec![[None; 4]; self.dev_group.len()];
        self.dev_level
            .iter()
            .zip(&self.dev_group)
            .map(|(level, &group)| {
                let mut out = [1.0; 4];
                for p in 0..4 {
                    if let (Some(first), Some(tau)) = (lay.nat_dev[p], lay.nat_tau_dev[p]) {
                        let v = match level {
                            Some(d) => nat[first + d],
                            None => {
                                *new_effect[group][p].get_or_insert_with(|| unobserved(nat[tau]))
                            }
                        };
                        out[p] = (ms[p].1 * v).exp();
                    }
                }
                out
            })
            .collect()
    }

    /// The parameters of every observed cell from a natural-scale draw (its
    /// origin's values times its age's development multipliers).
    fn observed_parameters(&self, nat: &[f64]) -> Vec<[f64; 4]> {
        let lay = &self.layout;
        let ms = self.priors.positive().map(lognormal);
        // No new level is drawn for an observed cell, so the stream is
        // never read.
        let mut unused = StreamRng::new(0, 0);
        self.cells
            .iter()
            .map(|c| {
                let mut th = self.origin_values(nat, c.origin, &mut unused);
                let d = self.dev_level[c.k].expect("an observed age has a level");
                for (p, th) in th.iter_mut().enumerate() {
                    if let Some(first) = lay.nat_dev[p] {
                        *th *= (ms[p].1 * nat[first + d]).exp();
                    }
                }
                th
            })
            .collect()
    }

    /// A draw of a cell given its mean amount, the type's sigma and the
    /// origin `i`: normal (with sigma times the premium on the loss-ratio
    /// scale) or log-normal.
    fn draw_cell(&self, mean: f64, sigma: f64, i: usize, rng: &mut StreamRng) -> f64 {
        let e = norm_quantile(rng.next_open01());
        if self.response.log_normal() {
            mean * (sigma * e).exp()
        } else if self.response.loss_ratio {
            mean + sigma * self.premium[i] * e
        } else {
            mean + sigma * e
        }
    }

    /// One simulation of the future cells, paid then outstanding, into
    /// `row`, given a natural-scale draw: each cell (each increment under
    /// incremental paid, summed onto the latest paid) normal or log-normal
    /// about its mean with its type's σ, independent given the draw. An
    /// origin with no observed cell takes one new set of effects for all its
    /// cells, and an age with none one new set for all its cells.
    pub(super) fn simulate(
        &self,
        nat: &[f64],
        future: &Future,
        rng: &mut StreamRng,
        row: &mut [f64],
    ) {
        let (k_er, k_p) = (nat[0], nat[1]);
        // Under incurred alone paid has no σ: its cells are their means.
        let [sigma_os, sigma_pd] = self.sigmas(nat);
        let dev = if self.has_dev() {
            self.dev_multipliers(nat, rng)
        } else {
            vec![[1.0; 4]; self.times.len()]
        };
        let curves: Vec<(f64, f64)> = self.times.iter().map(|&t| one_os(k_er, k_p, t)).collect();
        let n = self.premium.len();
        let mut values: Vec<Option<[f64; 4]>> = vec![None; n];
        let mut get = |i: usize, rng: &mut StreamRng| -> [f64; 4] {
            *values[i].get_or_insert_with(|| self.origin_values(nat, i, rng))
        };
        // The cell's parameters and its closed form `(OS, PD)` at age `k`
        // and its PD at the age before (0 at the first age).
        let cell = |th: [f64; 4], k: usize| -> ([f64; 4], (f64, f64), f64) {
            if !self.cellwise {
                let before = if k == 0 { 0.0 } else { curves[k - 1].1 };
                return (th, curves[k], before);
            }
            let th: [f64; 4] = std::array::from_fn(|p| th[p] * dev[k][p]);
            let now = one_os(th[KER], th[KP], self.times[k]);
            let before = if k == 0 {
                0.0
            } else {
                one_os(th[KER], th[KP], self.times[k - 1]).1
            };
            (th, now, before)
        };
        let n_paid = future.paid.len();
        let mut current = (usize::MAX, 0.0);
        for (x, &(i, k)) in row[..n_paid].iter_mut().zip(&future.paid) {
            let (th, (_, pd), before) = cell(get(i, rng), k);
            let scale = self.premium[i] * th[RLR] * th[RRF];
            *x = if self.response.incremental {
                if current.0 != i {
                    current = (i, self.latest_paid[i]);
                }
                current.1 += self.draw_cell(scale * (pd - before), sigma_pd, i, rng);
                current.1
            } else {
                self.draw_cell(scale * pd, sigma_pd, i, rng)
            };
        }
        for (x, &(i, k)) in row[n_paid..].iter_mut().zip(&future.outstanding) {
            let (th, (os, pd), _) = cell(get(i, rng), k);
            let mean = if self.response.incurred {
                self.premium[i] * th[RLR] * (os + th[RRF] * pd)
            } else {
                self.premium[i] * th[RLR] * os
            };
            *x = self.draw_cell(mean, sigma_os, i, rng);
        }
    }

    /// The paid development pattern of a natural-scale draw into `shares`,
    /// one per age of the square (`reserving-v02.md` decision 28 item 7):
    /// the expected cumulative paid at each age over that at the square's
    /// last age, where the reserves end, so the last share is exactly 1, of
    /// an origin with nothing observed at the median of the population (its
    /// origin effects 0, so its rates are the population's `k_er` and
    /// `k_p`), with each observed age's development effects and 0 for an
    /// age with none. The expected paid is the cell means' (cumulative, or
    /// the running sum of the incremental means, each age's increment with
    /// that age's parameters); `RLR` and `RRF` enter only through their
    /// development effects, the premium and the origin's values cancelling.
    /// No random number is drawn. Without development effects the share at
    /// age `t` is `PD(t) / PD(t_last)` of the ODE's paid curve, which never
    /// falls.
    pub(super) fn pattern(&self, nat: &[f64], shares: &mut [f64]) {
        let n = self.times.len();
        let dev = if self.has_dev() {
            self.dev_multipliers_with(nat, |_| 0.0)
        } else {
            vec![[1.0; 4]; n]
        };
        let origin = [nat[KER], nat[KP], 1.0, 1.0];
        let mut cumulative = 0.0;
        for (k, share) in shares.iter_mut().enumerate() {
            let th: [f64; 4] = std::array::from_fn(|p| origin[p] * dev[k][p]);
            let pd = one_os(th[KER], th[KP], self.times[k]).1;
            let scale = th[RLR] * th[RRF];
            *share = if self.response.incremental {
                let before = if k == 0 {
                    0.0
                } else {
                    one_os(th[KER], th[KP], self.times[k - 1]).1
                };
                cumulative += scale * (pd - before);
                cumulative
            } else {
                scale * pd
            };
        }
        let last = shares[n - 1];
        for share in &mut shares[..n - 1] {
            *share /= last;
        }
        shares[n - 1] = 1.0;
    }

    /// The full log density of each cell's response from a natural-scale
    /// draw, appended to `out`: normal, or log-normal (with its `-ln y`).
    pub(super) fn log_likelihoods(&self, nat: &[f64], out: &mut Vec<f64>) {
        let [sigma_os, sigma_pd] = self.sigmas(nat);
        let half_log_tau = 0.5 * std::f64::consts::TAU.ln();
        for (c, th) in self.cells.iter().zip(self.observed_parameters(nat)) {
            let mean = self.cell_mean(c, th);
            let sigma = if c.paid { sigma_pd } else { sigma_os };
            out.push(if self.response.log_normal() {
                let z = (c.ln_y - mean.ln()) / sigma;
                -half_log_tau - sigma.ln() - c.ln_y - 0.5 * z * z
            } else {
                let z = (c.y - mean) / sigma;
                -half_log_tau - sigma.ln() - 0.5 * z * z
            });
        }
    }

    /// With effects on the rates, the share of draws and observed cells
    /// with `k_er <= k_p`.
    pub(super) fn cell_rate_swaps(&self, posterior: &PosteriorDraws) -> Option<f64> {
        if !self.priors.rates_vary() {
            return None;
        }
        let mut swaps = 0usize;
        for nat in posterior.values().chunks_exact(posterior.dim()) {
            swaps += self
                .observed_parameters(nat)
                .iter()
                .filter(|th| th[KER] <= th[KP])
                .count();
        }
        Some(swaps as f64 / (posterior.n_draws() * self.cells.len()) as f64)
    }

    /// Each origin's posterior mean `RLR_i RRF_i` (at a zero development
    /// effect); for an origin with no data, `E[RLR RRF | draw] = m e^(s_RLR
    /// b_RLR + s_RRF b_RRF + v / 2)` with `v` the variance of `s_RLR u_1 +
    /// s_RRF u_2`, averaged over the draws.
    pub(super) fn mean_ultimate_loss_ratios(&self, posterior: &PosteriorDraws) -> Vec<f64> {
        let lay = &self.layout;
        let (m_rlr, s_rlr) = lognormal(self.priors.rlr);
        let (m_rrf, s_rrf) = lognormal(self.priors.rrf);
        let (at_rlr, at_rrf) = (
            lay.nat_origin[RLR].expect("RLR by origin"),
            lay.nat_origin[RRF].expect("RRF by origin"),
        );
        let n = self.premium.len();
        let mut sums = vec![0.0; n];
        for nat in posterior.values().chunks_exact(posterior.dim()) {
            let b = lay.nat_b0;
            let (b_rlr, b_rrf, tau1, tau2, rho) = (
                nat[b + 2],
                nat[b + 3],
                nat[b + 4],
                nat[b + 5],
                nat[lay.nat_rho],
            );
            for (i, sum) in sums.iter_mut().enumerate() {
                *sum += match self.level[i] {
                    Some(j) => nat[at_rlr + j] * nat[at_rrf + j],
                    None => {
                        let (a, c) = (s_rlr * tau1, s_rrf * tau2);
                        let v = a * a + c * c + 2.0 * rho * a * c;
                        m_rlr * m_rrf * (s_rlr * b_rlr + s_rrf * b_rrf + 0.5 * v).exp()
                    }
                };
            }
        }
        let draws = posterior.n_draws() as f64;
        sums.iter().map(|s| s / draws).collect()
    }

    /// The start of the pilot run (decision 5). Under one response
    /// ([`Fitted::PaidOnly`], [`Fitted::IncurredOnly`]) the least squares
    /// of the population curve is not identified (from paid alone only
    /// `RLR x RRF`) and runs off along the free direction (on company 337 to
    /// `RLR` near 1e-136), so the data start is taken only when its log
    /// posterior is above the prior centre's (decision 12).
    pub(super) fn start(&self, start: CompartmentalStart) -> Vec<f64> {
        if start == CompartmentalStart::Data
            && let Some(theta) = self.data_start()
        {
            if self.response.paid && self.response.second {
                return theta;
            }
            let centre = self.prior_centre_start();
            let mut grad = vec![0.0; self.dim()];
            let at_data = self.log_posterior(&theta, &mut grad);
            let at_centre = self.log_posterior(&centre, &mut grad);
            return if at_data >= at_centre || at_centre.is_nan() {
                theta
            } else {
                centre
            };
        }
        self.prior_centre_start()
    }

    /// `theta` with the population values `(k_er, k_p, RLR, RRF)`, every `z`
    /// and `atanh ρ` at 0, each `log τ` at about its prior median and the
    /// `log σ` at the log of each type's residual sd about the population
    /// curve.
    fn theta_at(&self, population: [f64; 4]) -> Option<Vec<f64>> {
        let [k_er, k_p, rlr, rrf] = population;
        let pr = &self.priors;
        let lay = &self.layout;
        let o = |prior: PositivePrior, v: f64| {
            let (m, s) = lognormal(prior);
            (v / m).ln() / s
        };
        let mut theta = vec![0.0; self.dim()];
        theta[B_KER] = if self.ordered {
            if k_er <= k_p {
                return None;
            }
            (k_er - k_p).ln()
        } else {
            o(pr.k_er, k_er)
        };
        theta[B_KP] = o(pr.k_p, k_p);
        theta[B_RLR] = o(pr.rlr, rlr);
        theta[B_RRF] = o(pr.rrf, rrf);
        theta[LOG_TAU] = half_median(pr.rlr_effect_sd).ln();
        theta[LOG_TAU + 1] = half_median(pr.rrf_effect_sd).ln();
        for (at, prior) in lay.tau_rate.iter().zip(pr.rate_effect_sd()) {
            if let (Some(at), Some(prior)) = (at, prior) {
                theta[*at] = half_median(prior).ln();
            }
        }
        for (at, prior) in lay.tau_dev.iter().zip(pr.development_effect_sd) {
            if let (Some(at), Some(prior)) = (at, prior) {
                theta[*at] = half_median(prior).ln();
            }
        }
        self.set_log_sigma(&mut theta, self.residual_rms(population));
        let mut grad = vec![0.0; theta.len()];
        let f = self.log_posterior(&theta, &mut grad);
        (f.is_finite() && theta.iter().all(|v| v.is_finite())).then_some(theta)
    }

    /// Each fitted type's `log σ` in `theta` at the log of its `rms` (the
    /// second response's, then paid's).
    fn set_log_sigma(&self, theta: &mut [f64], rms: [f64; 2]) {
        let ls = self.log_sigma0();
        for (s, paid) in self.response.sigma_types().enumerate() {
            theta[ls + s] = rms[usize::from(paid)].ln();
        }
    }

    /// The root mean square residual (on the likelihood's scale) of the
    /// outstanding and the paid cells about the population curve, each at
    /// least `1e-6` of its cells' mean absolute value (of 1 on the log
    /// scale); 1 for a type with no cell (not fitted, so never used).
    fn residual_rms(&self, population: [f64; 4]) -> [f64; 2] {
        let mut sum = [0.0; 2];
        let mut size = [0.0; 2];
        let mut count = [0.0; 2];
        for c in &self.cells {
            let k = usize::from(c.paid);
            let r = self.residual(c, self.cell_mean(c, population));
            sum[k] += r * r;
            size[k] += if self.response.log_normal() {
                1.0
            } else {
                c.y.abs()
            };
            count[k] += 1.0;
        }
        [0, 1].map(|k| {
            if count[k] == 0.0 {
                return 1.0;
            }
            let floor = 1e-6 * (size[k] / count[k]).max(1.0);
            (sum[k] / count[k]).sqrt().max(floor)
        })
    }

    /// The data-based start: least squares of the population curve (on the
    /// likelihood's scale) of `log (k_er, k_p, RLR, RRF)`, each type
    /// weighted by the inverse of its residual variance (twice, from the
    /// prior medians), then the representative of the rate swap the priors
    /// prefer.
    fn data_start(&self) -> Option<Vec<f64>> {
        let pr = &self.priors;
        let medians = pr.positive().map(|p| lognormal(p).0);
        let mut phi = medians.map(f64::ln).to_vec();
        for _ in 0..2 {
            let population = [0, 1, 2, 3].map(|j| phi[j].exp());
            let rms = self.residual_rms(population);
            let ls = LeastSquares {
                model: self,
                weight: rms.map(|r| 1.0 / (r * r)),
            };
            phi = mode(&ls, phi).ok()?;
        }
        let [k_er, k_p, rlr, rrf] = [0, 1, 2, 3].map(|j| phi[j].exp());
        let swapped = [k_p, k_er, rlr * k_er / k_p, rrf * k_p / k_er];
        let candidates = [self.theta_at([k_er, k_p, rlr, rrf]), self.theta_at(swapped)];
        let mut grad = vec![0.0; self.dim()];
        candidates
            .into_iter()
            .flatten()
            .map(|theta| (self.log_posterior(&theta, &mut grad), theta))
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, theta)| theta)
    }

    /// The fallback start at the prior medians. Under
    /// [`RateOrder::Ordered`], with medians that break the order, `k_er` is
    /// raised to `1.1 k_p`.
    fn prior_centre_start(&self) -> Vec<f64> {
        let pr = &self.priors;
        let [k_er, k_p, rlr, rrf] = pr.positive().map(|p| lognormal(p).0);
        let k_er = if self.ordered {
            k_er.max(1.1 * k_p)
        } else {
            k_er
        };
        self.theta_at([k_er, k_p, rlr, rrf]).unwrap_or_else(|| {
            // A curve too far from the data to give a finite density: the
            // log σ at the scale of the data.
            let mut theta = vec![0.0; self.dim()];
            self.set_log_sigma(&mut theta, self.residual_rms([k_er, k_p, rlr, rrf]));
            theta
        })
    }
}

impl Posterior for Model {
    fn dim(&self) -> usize {
        self.log_sigma0() + self.layout.n_sigma
    }

    fn log_posterior(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
        let f = if self.cellwise {
            self.cellwise_gradient(theta, grad)
        } else {
            gradient::<CHUNK>(theta, grad, |x| self.log_density(x))
        };
        if f.is_finite() { f } else { f64::NAN }
    }
}

/// The population curve's weighted least squares on
/// `phi = log (k_er, k_p, RLR, RRF)`, as a [`Posterior`] for [`mode`].
struct LeastSquares<'a> {
    model: &'a Model,
    weight: [f64; 2],
}

impl LeastSquares<'_> {
    fn value<S: Scalar>(&self, phi: &[S]) -> S {
        let population = [phi[0].exp(), phi[1].exp(), phi[2].exp(), phi[3].exp()];
        let mut f = S::from_f64(0.0);
        for c in &self.model.cells {
            let r = self.model.residual(c, self.model.cell_mean(c, population));
            f -= r * r * (0.5 * self.weight[usize::from(c.paid)]);
        }
        f
    }
}

impl Posterior for LeastSquares<'_> {
    fn dim(&self) -> usize {
        4
    }

    fn log_posterior(&self, phi: &[f64], grad: &mut [f64]) -> f64 {
        let f = gradient::<4>(phi, grad, |x| self.value(x));
        if f.is_finite() { f } else { f64::NAN }
    }
}
