//! The compartmental model's coming year, for the one-year view
//! (`docs/design/reserving-bayes.md`, decision 20): [`CompartmentalYear`]
//! implements [`YearModel`] on the unconstrained scale NUTS samples.
//!
//! The year's values are the amounts of the cells valued in the twelve
//! months after the valuation: the paid cells first (cumulative paid),
//! origin by origin and then age by age, then the outstanding cells in the
//! same order. Each type runs from the origin's latest observed cell of
//! that type to the last cell whose valuation falls in the year, so an
//! origin that lags the valuation catches up. Given a posterior draw the
//! response of each cell (the amount or loss ratio, the paid cumulative or
//! incremental, as the fit's settings say) is normal or log-normal about
//! its mean, independently, as in the fit.
//!
//! The draws are those of the model with a level for every origin of the
//! premium column and every age of the square ([`Model::full`]): an
//! origin with no observed cell (the papers' 1997) and, with development
//! effects (model 2), an age with none take independent standard normal
//! `z` added to each stored draw (their posterior is their prior), so that
//! reweighting the draws by the year's cells updates them, where the
//! lifetime fit draws them afresh in each simulation. Nothing else the
//! year informs is drawn afresh.
//!
//! `theta` holds every effect centred, `u = τ z` (for an origin's `RLR`
//! and `RRF` pair `u₁ = τ₁ z₁`, `u₂ = τ₂ (ρ z₁ + sqrt(1 - ρ²) z₂)`), not
//! the `z` NUTS samples: the year's cells pin an origin's effect, which on
//! the `z` scale is a curved ridge `z = u / τ` that the affine maps of
//! moment matching cannot follow, and on the centred scale a shift and a
//! scaling. On company 337 matching rescues about 1 in 10 of model 1's
//! flagged simulations on the `z` scale and about 19 in 20 on the centred
//! one. The map is smooth and one to one; its log Jacobian enters
//! [`YearModel::log_posterior`] (conformance check 5 checks it).

use std::cell::Cell;

use prospicio_bayes::nuts::PosteriorDraws;
use prospicio_core::StreamRng;
use prospicio_math::special::norm_quantile;

use super::{Compartmental, Data, KER, KP, Model, Obs, lognormal};
use crate::bayes::laplace::Posterior;
use crate::bayes::one_year::YearModel;
use crate::error::Result;

/// What the paid amount of an incremental year cell is measured from.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Before {
    /// Nothing: a cumulative cell, an outstanding cell or a first age.
    Nothing,
    /// The origin's latest observed cumulative paid.
    Observed(f64),
    /// The cumulative paid of the year's cell at this index.
    Year(usize),
}

/// A cell of the year: its observation with no value yet, and for an
/// incremental paid cell what its increment is measured from.
#[derive(Debug, Clone, Copy)]
struct YearCell {
    obs: Obs,
    before: Before,
}

/// A cell's parameters `(k_er, k_p, RLR, RRF)` from one `theta` of the
/// full model, every origin and age with a level.
struct Parameters {
    b: [f64; 4],
    u: [Vec<f64>; 4],
    v: [Vec<f64>; 4],
    ms: [(f64, f64); 4],
}

impl Parameters {
    fn at(&self, origin: usize, k: usize) -> [f64; 4] {
        std::array::from_fn(|p| {
            let v = self.v[p].get(k).copied().unwrap_or(0.0);
            ((self.b[p] + self.u[p][origin] + v) * self.ms[p].1).exp() * self.ms[p].0
        })
    }
}

/// A fitted [`Compartmental`]'s coming year (decision 20).
pub(super) struct CompartmentalYear<'a> {
    compartmental: &'a Compartmental,
    data: &'a Data,
    /// The model of the observed cells with every level.
    full: Model,
    /// The paid cells, then the outstanding.
    cells: Vec<YearCell>,
    n_paid: usize,
    /// Each origin's last paid cell in the year, by index into `cells`.
    last_paid: Vec<Option<usize>>,
}

impl<'a> CompartmentalYear<'a> {
    /// The coming year of a fit of `compartmental` to `data`.
    pub(super) fn new(compartmental: &'a Compartmental, data: &'a Data) -> Result<Self> {
        let full = Model::full(
            data,
            compartmental.priors,
            compartmental.rate_order == super::RateOrder::Ordered,
            compartmental.response(),
        )?;
        let n = data.origins.len();
        let valuation = |i: usize, k: usize| {
            data.origins[i]
                .start()
                .add_months(i64::from(data.age(k)) - 1)
        };
        let at = data
            .paid
            .iter()
            .chain(&data.outstanding)
            .map(|&(i, k, _)| valuation(i, k))
            .max()
            .expect("a fit has observed cells");
        let end = at.add_months(12);
        // The ages of origin `i` in the year after its latest age `from`.
        let ages = |i: usize, latest: Option<usize>| {
            (latest.map_or(0, |k| k + 1)..data.n_ages).take_while(move |&k| valuation(i, k) <= end)
        };
        let template = |i: usize, k: usize, paid: bool| -> Obs {
            let incremental = paid && full.response.incremental && k > 0;
            Obs {
                origin: i,
                k,
                t: full.times[k],
                t_before: incremental.then(|| full.times[k - 1]),
                y: 0.0,
                ln_y: 0.0,
                w: if full.response.loss_ratio {
                    1.0
                } else {
                    data.premium[i]
                },
                paid,
            }
        };
        let mut cells = Vec::new();
        let mut last_paid = vec![None; n];
        for (i, last) in last_paid.iter_mut().enumerate() {
            let latest = data.latest_paid[i];
            for k in ages(i, latest.map(|(k, _)| k)) {
                let obs = template(i, k, true);
                let before = match (obs.t_before, *last, latest) {
                    (None, _, _) => Before::Nothing,
                    (Some(_), Some(j), _) => Before::Year(j),
                    (Some(_), None, Some((_, v))) => Before::Observed(v),
                    // An incremental cell after the first age always has
                    // the paid of the age before: the year starts right
                    // after the latest paid.
                    (Some(_), None, None) => unreachable!("the year starts at age 0"),
                };
                *last = Some(cells.len());
                cells.push(YearCell { obs, before });
            }
        }
        let n_paid = cells.len();
        for i in 0..n {
            for k in ages(i, data.latest_outstanding[i]) {
                cells.push(YearCell {
                    obs: template(i, k, false),
                    before: Before::Nothing,
                });
            }
        }
        Ok(Self {
            compartmental,
            data,
            full,
            cells,
            n_paid,
            last_paid,
        })
    }

    /// `theta` with the centred effects back on the sampling scale `z`.
    fn to_sampled(&self, theta: &[f64]) -> Vec<f64> {
        self.recentre(theta, false)
    }

    /// A `theta` of [`Model::full`] with every effect centred.
    fn to_centred(&self, x: &[f64]) -> Vec<f64> {
        self.recentre(x, true)
    }

    fn recentre(&self, theta: &[f64], centre: bool) -> Vec<f64> {
        let lay = &self.full.layout;
        let mut x = theta.to_vec();
        let (tau1, tau2) = (theta[super::LOG_TAU].exp(), theta[super::LOG_TAU + 1].exp());
        let rho = theta[super::ATANH_RHO].tanh();
        let rho_c = (1.0 - rho * rho).sqrt();
        let scale = |at: Option<usize>| at.map_or(1.0, |l| theta[l].exp());
        let (n, n_ages) = (self.full.n_levels(), self.full.n_dev_levels());
        for i in 0..n {
            let a = lay.z0 + 2 * i;
            if centre {
                x[a] = tau1 * theta[a];
                x[a + 1] = tau2 * (rho * theta[a] + rho_c * theta[a + 1]);
            } else {
                x[a] = theta[a] / tau1;
                x[a + 1] = (theta[a + 1] / tau2 - rho * x[a]) / rho_c;
            }
        }
        for i in 0..n {
            for r in [KER, KP] {
                if let Some(z) = lay.z_rate[r] {
                    let tau = scale(lay.tau_rate[r]);
                    x[z + i] = if centre {
                        tau * theta[z + i]
                    } else {
                        theta[z + i] / tau
                    };
                }
            }
        }
        for k in 0..n_ages {
            for p in 0..4 {
                if let Some(z) = lay.z_dev[p] {
                    let tau = scale(lay.tau_dev[p]);
                    x[z + k] = if centre {
                        tau * theta[z + k]
                    } else {
                        theta[z + k] / tau
                    };
                }
            }
        }
        x
    }

    /// The log Jacobian of the centred effects' map back to `z`:
    /// `-(log τ₁ + log τ₂ + log sqrt(1 - ρ²))` per origin pair and `-log τ`
    /// per other effect.
    fn log_jacobian(&self, theta: &[f64]) -> f64 {
        let lay = &self.full.layout;
        let rho = theta[super::ATANH_RHO].tanh();
        let per_origin =
            theta[super::LOG_TAU] + theta[super::LOG_TAU + 1] + 0.5 * (1.0 - rho * rho).ln();
        let mut per_rate = 0.0;
        for r in [KER, KP] {
            if let (Some(l), Some(_)) = (lay.tau_rate[r], lay.z_rate[r]) {
                per_rate += theta[l];
            }
        }
        let mut per_age = 0.0;
        for p in 0..4 {
            if let (Some(l), Some(_)) = (lay.tau_dev[p], lay.z_dev[p]) {
                per_age += theta[l];
            }
        }
        -((per_origin + per_rate) * self.full.n_levels() as f64
            + per_age * self.full.n_dev_levels() as f64)
    }

    /// The fit's draws `sampled` (the observed cells' model `observed`, on
    /// the unconstrained scale) with a standard normal `z` for every level
    /// the observed cells do not have, draw `d` from stream `d` of `seed`,
    /// then centred: the posterior of [`Model::full`], whose draws this
    /// year takes. It counts the draws as `PosteriorDraws::transform`
    /// visits them, which is in order.
    pub(super) fn opening_draws(
        &self,
        observed: &Model,
        sampled: &PosteriorDraws,
        seed: u64,
    ) -> Result<PosteriorDraws> {
        // `transform` maps the draws in order, so the counter is the draw.
        let d = Cell::new(0u64);
        Ok(sampled.transform(self.full.sampled_names(), |theta| {
            let mut rng = StreamRng::new(seed, d.get());
            d.set(d.get() + 1);
            self.to_centred(&embed(observed, &self.full, theta, &mut rng))
        })?)
    }

    /// The parameters of every cell given `theta`.
    fn parameters(&self, theta: &[f64]) -> Parameters {
        let theta = &self.to_sampled(theta);
        let (_, _, b_ker) = self.full.rates(theta);
        let (u, v) = self.full.effects(theta);
        Parameters {
            b: [
                b_ker,
                theta[super::B_KP],
                theta[super::B_RLR],
                theta[super::B_RRF],
            ],
            u,
            v,
            ms: self.full.priors.positive().map(lognormal),
        }
    }

    /// The mean amount of the paid cell of origin `i` at age `k` (its
    /// increment under incremental paid), as the fit's simulations take it.
    fn paid_mean(&self, p: &Parameters, i: usize, k: usize) -> f64 {
        let response = self.full.response;
        let obs = Obs {
            origin: i,
            k,
            t: self.full.times[k],
            t_before: (response.incremental && k > 0).then(|| self.full.times[k - 1]),
            y: 0.0,
            ln_y: 0.0,
            w: if response.loss_ratio {
                1.0
            } else {
                self.data.premium[i]
            },
            paid: true,
        };
        let mean = self.full.cell_mean(&obs, p.at(i, k));
        if response.loss_ratio {
            mean * self.data.premium[i]
        } else {
            mean
        }
    }

    /// Each origin's expected paid to the square's last age given `theta`,
    /// less its paid at the age `from` (its value `base`): under
    /// incremental paid the expected increments after `from`, under
    /// cumulative the expected paid at the last age less `base`. A
    /// log-normal response's mean is its median times `e^(σ²/2)`. 0 for an
    /// origin at the last age.
    fn reserve(&self, p: &Parameters, sigma_pd: f64, i: usize, from: Option<(usize, f64)>) -> f64 {
        let last = self.data.n_ages - 1;
        if from.is_some_and(|(k, _)| k == last) {
            return 0.0;
        }
        let factor = if self.full.response.log_normal() {
            (0.5 * sigma_pd * sigma_pd).exp()
        } else {
            1.0
        };
        if self.full.response.incremental {
            let first = from.map_or(0, |(k, _)| k + 1);
            (first..=last).map(|k| self.paid_mean(p, i, k)).sum::<f64>() * factor
        } else {
            self.paid_mean(p, i, last) * factor - from.map_or(0.0, |(_, v)| v)
        }
    }

    /// The response of year cell `j` (an amount or a loss ratio, an
    /// increment under incremental paid) from the year's amounts.
    fn response(&self, j: usize, year: &[f64]) -> f64 {
        let cell = &self.cells[j];
        let amount = year[j]
            - match cell.before {
                Before::Nothing => 0.0,
                Before::Observed(v) => v,
                Before::Year(b) => year[b],
            };
        if self.full.response.loss_ratio {
            amount / self.data.premium[cell.obs.origin]
        } else {
            amount
        }
    }

    /// The year's cells with their values, as the data's cells.
    fn extended(&self, year: &[f64]) -> Result<Model> {
        let cell = |(c, &v): (&YearCell, &f64)| (c.obs.origin, c.obs.k, v);
        let data = self.data.extended(
            self.cells[..self.n_paid]
                .iter()
                .zip(&year[..self.n_paid])
                .map(cell),
            self.cells[self.n_paid..]
                .iter()
                .zip(&year[self.n_paid..])
                .map(cell),
        );
        Model::full(
            &data,
            self.full.priors,
            self.full.ordered,
            self.full.response,
        )
    }

    /// `σ_OS` and `σ_PD` of `theta`.
    fn sigmas(&self, theta: &[f64]) -> [f64; 2] {
        let ls = self.full.log_sigma0();
        [theta[ls].exp(), theta[ls + 1].exp()]
    }
}

/// `theta` of `observed` (the observed cells' model) as a `theta` of
/// `full` (the same cells with every level): the shared entries copied,
/// a standard normal from `rng` for each `z` of a level `observed` lacks,
/// in `full`'s order.
fn embed(observed: &Model, full: &Model, theta: &[f64], rng: &mut StreamRng) -> Vec<f64> {
    let (lo, lf) = (&observed.layout, &full.layout);
    let mut normal = || norm_quantile(rng.next_open01());
    let mut x = vec![0.0; full.dim()];
    x[..lf.z0].copy_from_slice(&theta[..lo.z0]);
    for (j, &i) in full.level_origin.iter().enumerate() {
        let at = lf.z0 + 2 * j;
        match observed.level[i] {
            Some(o) => x[at..at + 2].copy_from_slice(&theta[lo.z0 + 2 * o..lo.z0 + 2 * o + 2]),
            None => {
                x[at] = normal();
                x[at + 1] = normal();
            }
        }
    }
    for r in [KER, KP] {
        if let (Some(zf), Some(zo)) = (lf.z_rate[r], lo.z_rate[r]) {
            for (j, &i) in full.level_origin.iter().enumerate() {
                x[zf + j] = observed.level[i].map_or_else(&mut normal, |o| theta[zo + o]);
            }
        }
    }
    for p in 0..4 {
        if let (Some(zf), Some(zo)) = (lf.z_dev[p], lo.z_dev[p]) {
            for (k, level) in full.dev_level.iter().enumerate() {
                let d = level.expect("the full model has every age");
                x[zf + d] = observed.dev_level[k].map_or_else(&mut normal, |o| theta[zo + o]);
            }
        }
    }
    let (ls, lsf) = (observed.log_sigma0(), full.log_sigma0());
    x[lsf..lsf + 2].copy_from_slice(&theta[ls..ls + 2]);
    x
}

impl YearModel for CompartmentalYear<'_> {
    fn n_origins(&self) -> usize {
        self.data.origins.len()
    }

    fn n_year(&self) -> usize {
        self.cells.len()
    }

    /// Each cell's response about its mean with its type's `σ`, as the
    /// fit's simulations draw it; a paid increment summed onto the paid
    /// before it.
    fn draw_year(&self, theta: &[f64], rng: &mut StreamRng, year: &mut [f64]) {
        let p = self.parameters(theta);
        let sigma = self.sigmas(theta);
        let response = self.full.response;
        for j in 0..self.cells.len() {
            let cell = self.cells[j];
            let c = &cell.obs;
            let mean = self.full.cell_mean(c, p.at(c.origin, c.k));
            let s = sigma[usize::from(c.paid)];
            let e = norm_quantile(rng.next_open01());
            let value = if response.log_normal() {
                mean * (s * e).exp()
            } else {
                mean + s * e
            };
            let amount = if response.loss_ratio {
                value * self.data.premium[c.origin]
            } else {
                value
            };
            year[j] = amount
                + match cell.before {
                    Before::Nothing => 0.0,
                    Before::Observed(v) => v,
                    Before::Year(b) => year[b],
                };
        }
    }

    /// Each cell's term of the model's log density, `-r² / (2σ²) - log σ`
    /// with `r` the residual on the likelihood's scale.
    fn year_log_likelihood(&self, theta: &[f64], year: &[f64]) -> f64 {
        let p = self.parameters(theta);
        let ls = self.full.log_sigma0();
        let log_sigma = [theta[ls], theta[ls + 1]];
        let inv = log_sigma.map(|l| (-l).exp());
        let mut ll = 0.0;
        for (j, cell) in self.cells.iter().enumerate() {
            let mut c = cell.obs;
            c.y = self.response(j, year);
            if self.full.response.log_normal() {
                c.ln_y = c.y.ln();
            }
            let mean = self.full.cell_mean(&c, p.at(c.origin, c.k));
            let t = usize::from(c.paid);
            let r = self.full.residual(&c, mean) * inv[t];
            ll -= r * r * 0.5 + log_sigma[t];
        }
        ll
    }

    fn paid(&self, year: &[f64], paid: &mut [f64]) {
        for (i, p) in paid.iter_mut().enumerate() {
            let latest = self.data.latest_paid[i].map_or(0.0, |(_, v)| v);
            *p = self.last_paid[i].map_or(0.0, |j| year[j] - latest);
        }
    }

    fn opening(&self, theta: &[f64], reserve: &mut [f64]) {
        let p = self.parameters(theta);
        let sigma_pd = self.sigmas(theta)[1];
        for (i, r) in reserve.iter_mut().enumerate() {
            *r = self.reserve(&p, sigma_pd, i, self.data.latest_paid[i]);
        }
    }

    fn closing(&self, theta: &[f64], year: &[f64], reserve: &mut [f64]) {
        let p = self.parameters(theta);
        let sigma_pd = self.sigmas(theta)[1];
        for (i, r) in reserve.iter_mut().enumerate() {
            let from = match self.last_paid[i] {
                // Under incremental paid only the position matters, and
                // the engine passes no year (`closing_uses_year`).
                Some(j) => Some((self.cells[j].obs.k, year.get(j).copied().unwrap_or(0.0))),
                None => self.data.latest_paid[i],
            };
            *r = self.reserve(&p, sigma_pd, i, from);
        }
    }

    /// Under incremental paid the reserve after the year is the expected
    /// increments after it, whatever the year's values; under cumulative
    /// paid it is measured from the year's last paid.
    fn closing_uses_year(&self) -> bool {
        !self.full.response.incremental
    }

    /// Sampled as [`Compartmental::fit`] samples, from the extended model's
    /// own start and pilot run, with the fit's sampler; the draws on the
    /// unconstrained scale of the model with every level.
    fn refit(&self, year: &[f64], seed: u64) -> Result<(PosteriorDraws, bool)> {
        let model = self.extended(year)?;
        let (_, sampled, diagnostics) = self.compartmental.sample(&model, seed)?;
        let sampled = sampled.transform(model.sampled_names(), |x| self.to_centred(x))?;
        Ok((sampled, !diagnostics.warnings.is_empty()))
    }

    /// The log density of the centred draws: that of [`Model::full`] (with
    /// the year's cells when `year` has values) at `theta` mapped back to
    /// the sampled `z` scale, plus the log Jacobian of that map
    /// (`log_jacobian`:
    /// `-(log τ₁ + log τ₂ + ½ log(1 - ρ²))` per origin pair and `-log τ`
    /// per other effect). Without it the density would be the sampled
    /// scale's, not the stored draws', which check 5 of the conformance
    /// suite catches.
    fn log_posterior(&self, theta: &[f64], year: &[f64]) -> f64 {
        let x = self.to_sampled(theta);
        let f = self.log_jacobian(theta)
            + if year.is_empty() {
                self.full.log_density::<f64>(&x)
            } else {
                match self.extended(year) {
                    Ok(model) => model.log_density::<f64>(&x),
                    Err(_) => f64::NAN,
                }
            };
        if f.is_finite() { f } else { f64::NAN }
    }
}
