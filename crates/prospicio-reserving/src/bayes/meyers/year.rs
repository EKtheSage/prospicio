//! Meyers' models' coming year, for the one-year view
//! (`docs/design/reserving-bayes.md`, decision 19): [`MeyersYear`]
//! implements [`YearModel`] on the unconstrained scale NUTS samples.
//!
//! The year's values are the cumulative values `C_wd` of the cells valued
//! in the twelve months after the valuation, origin by origin and then age
//! by age. Given a posterior draw the log of each is normal with mean
//! `mu_wd` and variance `sigma_d^2`, independently in CRC and CSR. In CAY
//! each mean also takes `rho` times the previous origin's residual (its
//! innovation, `log C - mean`) at the same age, so the cells of an age
//! form a moving average of order 1 down the origins: a year cell's
//! residual is known given the draw, the observed cells and the year's
//! earlier cells, provided its previous origin's cell at that age is
//! observed or in the year ([`MeyersYear::new`] checks it). Nothing else
//! that the year informs is drawn afresh: every parameter is in the draw,
//! and the residual of an unobserved cell the year does not reach (the
//! previous origin's at the oldest age, for an origin that stays open) is
//! integrated out of the expected ultimate, which makes its variance
//! `(1 + rho^2) sigma_m^2`.

use prospicio_bayes::nuts::PosteriorDraws;
use prospicio_core::StreamRng;
use prospicio_math::special::norm_quantile;

use super::{Cell, Meyers, Model, S_FLOOR, cell_index, logistic};
use crate::bayes::laplace::Posterior;
use crate::bayes::one_year::YearModel;
use crate::chain_ladder::ChainLadderFit;
use crate::error::{Error, Result};
use crate::one_year_bootstrap::YearCells;
use crate::triangle::{Segment, Triangle};

/// Where a cell of an age's chain of residuals comes from.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Source {
    /// An observed cell of the model, by index.
    Observed(usize),
    /// A value of the year, by index.
    Year(usize),
}

/// A fitted [`Meyers`] segment's coming year (decision 19).
pub(super) struct MeyersYear<'a> {
    meyers: &'a Meyers,
    model: &'a Model,
    /// `log P_w` per origin.
    offsets: Vec<f64>,
    /// Each origin's latest observed position.
    latest: Vec<usize>,
    /// Each origin's latest observed cumulative value.
    latest_value: Vec<f64>,
    /// The year's cells `(w, d)`, origin by origin and then age by age.
    year: Vec<(usize, usize)>,
    /// Each origin's last cell in the year, by index into `year`.
    last_cell: Vec<Option<usize>>,
    /// CAY: for each age with a cell in the year, the chain of cells whose
    /// residuals lead to them: origins `0, 1, ...` in order, observed and
    /// then in the year.
    chains: Vec<(usize, Vec<Source>)>,
    /// CAY: the same chain at the oldest age, over every origin observed
    /// there or taken there by the year.
    ultimate_chain: Vec<Source>,
    /// CAY: each year cell's previous origin's cell at the same age, by
    /// index into the extended model's cells (the observed, then the
    /// year's).
    year_prev: Vec<Option<usize>>,
}

impl<'a> MeyersYear<'a> {
    /// The coming year of `segment` of `triangle` under `model` (the
    /// segment's observed cells), with the chain ladder's latest positions
    /// and values and each origin's premium. CAY's year needs each cell's
    /// previous origin observed or in the year at the same age.
    pub(super) fn new(
        meyers: &'a Meyers,
        model: &'a Model,
        triangle: &Triangle,
        segment: &Segment,
        chain_ladder: &ChainLadderFit,
        premiums: &[f64],
    ) -> Result<Self> {
        let (n, m) = (model.n_origins, model.n_dev);
        let latest = chain_ladder.latest_position.clone();
        let mut last = latest.clone();
        // No tail: the year stops at the oldest age.
        for y in YearCells::of(
            segment,
            &latest,
            |o, d| triangle.valuation_of(o + segment.origin_offset, d + segment.dev_offset),
            false,
        ) {
            last[y.origin] = y.last;
        }
        let year: Vec<(usize, usize)> = (0..n)
            .flat_map(|w| (latest[w] + 1..=last[w]).map(move |d| (w, d)))
            .collect();
        let mut index = vec![vec![None; m]; n];
        let mut last_cell = vec![None; n];
        for (j, &(w, d)) in year.iter().enumerate() {
            index[w][d] = Some(j);
            last_cell[w] = Some(j);
        }
        let observed = cell_index(&model.cells, n, m);
        let n_obs = model.cells.len();
        let mut year_prev = vec![None; year.len()];
        let (mut chains, mut ultimate_chain) = (Vec::new(), Vec::new());
        if model.cay() {
            for (j, &(w, d)) in year.iter().enumerate() {
                if w == 0 {
                    continue;
                }
                year_prev[j] = match (observed[w - 1][d], index[w - 1][d]) {
                    (Some(i), _) => Some(i),
                    (None, Some(k)) => Some(n_obs + k),
                    (None, None) => {
                        return Err(Error::Bayes(format!(
                            "origin {} has a cell at development age {} months in the coming \
                             year but origin {} is neither observed nor in the year there; \
                             CAY's one-year view needs the previous origin's cell at each age",
                            segment.origins[w],
                            segment.ages[d],
                            segment.origins[w - 1]
                        )));
                    }
                };
            }
            let chain = |d: usize| -> Vec<Source> {
                (0..n)
                    .map_while(|w| match (observed[w][d], index[w][d]) {
                        (Some(i), _) => Some(Source::Observed(i)),
                        (None, Some(j)) => Some(Source::Year(j)),
                        (None, None) => None,
                    })
                    .collect()
            };
            let mut ages: Vec<usize> = year.iter().map(|&(_, d)| d).collect();
            ages.sort_unstable();
            ages.dedup();
            chains = ages.into_iter().map(|d| (d, chain(d))).collect();
            ultimate_chain = chain(m - 1);
        }
        Ok(Self {
            meyers,
            model,
            offsets: premiums.iter().map(|p| p.ln()).collect(),
            latest,
            latest_value: chain_ladder.latest.clone(),
            year,
            last_cell,
            chains,
            ultimate_chain,
            year_prev,
        })
    }

    /// `mu_wd` without CAY's `rho` term, from an unconstrained draw.
    fn mu(&self, theta: &[f64], w: usize, d: usize) -> f64 {
        let cell = Cell {
            w,
            d,
            y: 0.0,
            offset: self.offsets[w],
            prev: None,
        };
        self.model.mu(theta, &cell)
    }

    /// `rho` (0 outside CAY).
    fn rho(&self, theta: &[f64]) -> f64 {
        if self.model.cay() {
            theta[self.model.after_beta()].tanh()
        } else {
            0.0
        }
    }

    /// `sigma_d^2` for each age, from the draw's `u`.
    fn variances(&self, theta: &[f64]) -> Vec<f64> {
        let u = &theta[self.model.u0()..];
        let mut var = vec![0.0; self.model.n_dev];
        let mut acc = 0.0;
        for d in (0..var.len()).rev() {
            acc += S_FLOOR + (1.0 - S_FLOOR) * logistic(u[d]);
            var[d] = acc;
        }
        var
    }

    /// The log of a chain cell's value: observed, or the year's.
    fn log_value(&self, source: Source, year: &[f64]) -> f64 {
        match source {
            Source::Observed(i) => self.model.cells[i].y,
            Source::Year(j) => year[j].ln(),
        }
    }

    /// Each origin's expected ultimate less its latest value, the year's
    /// when `year` has values, the observed otherwise; 0 for an origin at
    /// the oldest age.
    fn reserves(&self, theta: &[f64], year: &[f64], reserve: &mut [f64]) {
        let model = self.model;
        let m = model.n_dev;
        let u = theta[model.u0() + m - 1];
        let var = S_FLOOR + (1.0 - S_FLOOR) * logistic(u);
        let rho = self.rho(theta);
        let with_year = !year.is_empty();
        // CAY: the residual of the last origin of the oldest age's chain,
        // and how many origins the chain holds.
        let (mut known, mut last) = (0, 0.0);
        if model.cay() {
            for (w, &source) in self.ultimate_chain.iter().enumerate() {
                if !with_year && matches!(source, Source::Year(_)) {
                    break;
                }
                last = self.log_value(source, year) - self.mu(theta, w, m - 1) - rho * last;
                known = w + 1;
            }
        }
        for (w, r) in reserve.iter_mut().enumerate() {
            let in_year = if with_year { self.last_cell[w] } else { None };
            let at = in_year.map_or(self.latest[w], |j| self.year[j].1);
            if at + 1 == m {
                *r = 0.0;
                continue;
            }
            let mut mean = self.mu(theta, w, m - 1);
            let mut v = var;
            if model.cay() && w > 0 {
                // Origins before `known` are at the oldest age.
                if w == known {
                    mean += rho * last;
                } else {
                    v *= 1.0 + rho * rho;
                }
            }
            let base = in_year.map_or(self.latest_value[w], |j| year[j]);
            *r = (mean + 0.5 * v).exp() - base;
        }
    }

    /// The model with the year's cells added as observed.
    fn extended(&self, year: &[f64]) -> Model {
        let mut model = self.model.clone();
        model
            .cells
            .extend(self.year.iter().zip(year).zip(&self.year_prev).map(
                |((&(w, d), &c), &prev)| Cell {
                    w,
                    d,
                    y: c.ln(),
                    offset: self.offsets[w],
                    prev,
                },
            ));
        model
    }
}

impl YearModel for MeyersYear<'_> {
    fn n_origins(&self) -> usize {
        self.model.n_origins
    }

    fn n_year(&self) -> usize {
        self.year.len()
    }

    fn draw_year(&self, theta: &[f64], rng: &mut StreamRng, year: &mut [f64]) {
        let var = self.variances(theta);
        if !self.model.cay() {
            for (c, &(w, d)) in year.iter_mut().zip(&self.year) {
                let z = norm_quantile(rng.next_open01());
                *c = (self.mu(theta, w, d) + var[d].sqrt() * z).exp();
            }
            return;
        }
        let rho = self.rho(theta);
        for (d, chain) in &self.chains {
            let sigma = var[*d].sqrt();
            let mut previous = 0.0;
            for (w, &source) in chain.iter().enumerate() {
                let mean = self.mu(theta, w, *d) + rho * previous;
                previous = match source {
                    Source::Observed(i) => self.model.cells[i].y - mean,
                    Source::Year(j) => {
                        let e = sigma * norm_quantile(rng.next_open01());
                        year[j] = (mean + e).exp();
                        e
                    }
                };
            }
        }
    }

    /// The normal log density of each year cell's `log C` given the draw
    /// and the cells before it, without `-log C` (constant in the draw).
    fn year_log_likelihood(&self, theta: &[f64], year: &[f64]) -> f64 {
        let var = self.variances(theta);
        let term = |e: f64, v: f64| -0.5 * (v.ln() + e * e / v);
        if !self.model.cay() {
            return self
                .year
                .iter()
                .zip(year)
                .map(|(&(w, d), c)| term(c.ln() - self.mu(theta, w, d), var[d]))
                .sum();
        }
        let rho = self.rho(theta);
        let mut ll = 0.0;
        for (d, chain) in &self.chains {
            let mut previous = 0.0;
            for (w, &source) in chain.iter().enumerate() {
                let e = self.log_value(source, year) - self.mu(theta, w, *d) - rho * previous;
                if matches!(source, Source::Year(_)) {
                    ll += term(e, var[*d]);
                }
                previous = e;
            }
        }
        ll
    }

    fn paid(&self, year: &[f64], paid: &mut [f64]) {
        for (w, p) in paid.iter_mut().enumerate() {
            *p = self.last_cell[w].map_or(0.0, |j| year[j] - self.latest_value[w]);
        }
    }

    fn opening(&self, theta: &[f64], reserve: &mut [f64]) {
        self.reserves(theta, &[], reserve);
    }

    fn closing(&self, theta: &[f64], year: &[f64], reserve: &mut [f64]) {
        self.reserves(theta, year, reserve);
    }

    /// Sampled as [`Meyers::fit`] samples, from the extended model's own
    /// least-squares start and pilot run; the draws on the unconstrained
    /// scale.
    fn refit(&self, year: &[f64], seed: u64) -> Result<(PosteriorDraws, bool)> {
        let (_, sampled, diagnostics) = self.meyers.sample(&self.extended(year), seed)?;
        Ok((sampled, !diagnostics.warnings.is_empty()))
    }

    /// The draws are stored on the unconstrained scale, an affine map of
    /// the whitened one NUTS samples, so no Jacobian enters.
    fn log_posterior(&self, theta: &[f64], year: &[f64]) -> f64 {
        let mut grad = vec![0.0; theta.len()];
        if year.is_empty() {
            return self.model.log_posterior(theta, &mut grad);
        }
        self.extended(year).log_posterior(theta, &mut grad)
    }
}
