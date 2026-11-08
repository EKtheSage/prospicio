//! The fits of a compartmental model and their summaries, segments and one-year views.

use prospicio_bayes::elpd::Loo;
use prospicio_bayes::nuts::{ParamSummary, PosteriorDraws};
use prospicio_core::{Lag, Period};
use prospicio_prob::{Distribution, KeyValue, PredictiveDistribution, Provenance};

use super::super::Diagnostics;
use super::super::one_year::{self, BayesOneYearFit, BayesOneYearSegment};
use super::data::Data;
#[cfg(doc)]
use super::fit::Compartmental;
use super::model::Model;
#[cfg(doc)]
use super::settings::{Fitted, RateOrder};
use crate::bootstrap::{component_sums, push_moments};
use crate::error::{Error, Result};
use crate::pattern_draws::DevelopmentPatterns;
use crate::segments::{FitTable, SegmentFits};

/// Paths drawn from the priors alone ([`Compartmental::prior_predictive`]):
/// the mean outstanding and cumulative paid of each origin at each age of
/// the square, dimensions `origin` and `age` (months).
#[derive(Debug, Clone)]
pub struct CompartmentalPaths {
    /// The mean outstanding.
    pub outstanding: PredictiveDistribution,
    /// The mean cumulative paid.
    pub paid: PredictiveDistribution,
}

/// A fitted [`Compartmental`].
#[derive(Debug, Clone)]
pub struct CompartmentalFit {
    /// The origins, those of the premium column.
    pub origins: Vec<Period>,
    /// Each origin's premium, the latest value of the premium column.
    pub premium: Vec<f64>,
    /// Each origin's latest observed paid (0 with none).
    pub latest_paid: Vec<f64>,
    /// The origins with no observed cell, whose effects each simulation
    /// draws from the population (decision 2).
    pub population_origins: Vec<Period>,
    /// Posterior draws, by chain: on the natural scale `k_er`, `k_p`, `RLR`
    /// and `RRF` (the population values, at a zero effect), `sigma_os`,
    /// `sigma_pd`, then `RLR[o]` and `RRF[o]` for each origin `o` with an
    /// observed cell (at a zero development effect), with origin effects on
    /// the rates `k_er[o]` and `k_p[o]` too, and with development effects
    /// `dev_ker[a]`, `dev_kp[a]`, `dev_RLR[a]` and `dev_RRF[a]`, each
    /// development age `a` (months) with an observed cell's effect on the
    /// `o` scale; and on the scale the paper prints, the intercepts `b_ker`,
    /// `b_kp`, `b_RLR`, `b_RRF`, the effect sds `tau_RLR` and `tau_RRF` (and
    /// `tau_ker`, `tau_kp`) on the `o` scale, `rho` (and `tau_dev_ker`,
    /// `tau_dev_kp`, `tau_dev_RLR`, `tau_dev_RRF`), `log_sigma_os` and
    /// `log_sigma_pd`. Each name in parentheses is there only when the
    /// priors have that effect. With one response fitted there is one
    /// `sigma` and one `log_sigma`: `sigma_pd` under [`Fitted::PaidOnly`],
    /// `sigma_os` (the incurred's) under [`Fitted::IncurredOnly`].
    pub posterior: PosteriorDraws,
    /// R-hat, ESS and divergences, with warnings for failed thresholds and
    /// (under [`RateOrder::Free`]) for draws with `k_er <= k_p`.
    pub diagnostics: Diagnostics,
    /// The number of posterior draws whose population rates have
    /// `k_er <= k_p` (0 under [`RateOrder::Ordered`]).
    pub rate_swaps: usize,
    /// With effects on the rates (each cell its own `k_er` and `k_p`), the
    /// share of posterior draws and observed cells with `k_er <= k_p`: a
    /// diagnostic, not a warning, since each cell's means are invariant
    /// under the swap on their own (decision 2). `None` when the rates are
    /// shared.
    pub cell_rate_swaps: Option<f64>,
    /// Joint posterior predictive distribution of each origin's reserve, the
    /// paid at the square's last age less the latest paid: dimension
    /// `origin`, one component per origin, with the layout of
    /// [`OdpBootstrapFit::reserves`](crate::OdpBootstrapFit::reserves).
    pub reserves: PredictiveDistribution,
    /// Posterior predictive cumulative paid of each future cell of the square
    /// (after each origin's latest paid), dimensions `origin` and `age`
    /// (months), with the same simulations as the reserves. Under
    /// [`Fitted::IncurredOnly`] no paid cell is fitted and each value is
    /// the expected paid given the draw, without observation noise.
    pub paid: PredictiveDistribution,
    /// Posterior predictive outstanding (the incurred, when it is the second
    /// response) of each future cell of the square (after each origin's
    /// latest one), the same layout; `None` under [`Fitted::PaidOnly`],
    /// which fits no second response.
    pub outstanding: Option<PredictiveDistribution>,
    /// Each origin's posterior mean ultimate loss ratio `RLR_i RRF_i`, at a
    /// zero development effect (for an origin drawn from the population, the
    /// mean over its effect too). The model's ultimate, without process
    /// error past the square.
    pub ultimate_loss_ratio: Vec<f64>,
    /// The paid development pattern of the posterior draw each simulation
    /// took (`reserving-v02.md` decision 28 item 7): at each age of the
    /// square, the expected cumulative paid of an origin with nothing
    /// observed (the population's rates, at a zero origin effect, with each
    /// observed age's development effects) over that at the square's last
    /// age, where it is exactly 1. Row `i` is simulation `i` of
    /// [`reserves`](Self::reserves), [`paid`](Self::paid) and
    /// [`outstanding`](Self::outstanding).
    pub patterns: DevelopmentPatterns,
    pub(super) model: Model,
    /// The same draws on the unconstrained scale of `theta`, for the
    /// one-year view (`docs/design/reserving-bayes.md`, decision 20).
    pub(super) sampled: PosteriorDraws,
    /// The observed cells, for the one-year view.
    pub(super) data: Data,
}

/// One segment of [`Compartmental::fit_segments`]: its [`CompartmentalFit`],
/// whose `reserves`, `paid` and `outstanding` are the segment's part of the
/// joint distributions of [`CompartmentalFits`], after any reordering for
/// the rank correlation between segments.
#[derive(Debug, Clone)]
pub struct CompartmentalSegment {
    /// The segment's fit: posterior, diagnostics, rate swaps, `summary()`,
    /// `loo()`, `to_long()` and the rest.
    pub fit: CompartmentalFit,
}

/// The fits of every segment of a triangle ([`Compartmental::fit_segments`]),
/// with joint distributions across segments.
#[derive(Debug, Clone)]
pub struct CompartmentalFits {
    /// Each segment's fit.
    pub segments: SegmentFits<CompartmentalSegment>,
    /// Joint posterior predictive distribution of the reserve by segment
    /// and origin: the key names and `origin` are its dimensions, and its
    /// components run over the origins of each segment in turn, like the
    /// rows of [`to_long`](Self::to_long).
    pub reserves: PredictiveDistribution,
    /// The future cumulative paid of every segment, dimensions the key names,
    /// `origin` and `age` (months), from the same simulations as
    /// [`reserves`](Self::reserves).
    pub paid: PredictiveDistribution,
    /// The future outstanding (or incurred) of every segment, the same
    /// layout; `None` under [`Fitted::PaidOnly`].
    pub outstanding: Option<PredictiveDistribution>,
    /// Each segment's paid development pattern in every simulation, moving
    /// with the segment's reserves and cells (`reserving-v02.md` decision
    /// 28), as in [`CompartmentalFit::patterns`].
    pub patterns: DevelopmentPatterns,
}

impl CompartmentalFits {
    /// The joint distributions of `segments`' own parts, simulation by
    /// simulation, with `provenance` (and their `output`).
    pub(super) fn assemble(
        segments: SegmentFits<CompartmentalSegment>,
        provenance: Provenance,
    ) -> Result<Self> {
        let joint = |part: fn(&CompartmentalFit) -> &PredictiveDistribution,
                     output: &str|
         -> Result<PredictiveDistribution> {
            let first = part(&segments.fits[0].fit);
            let mut dims = segments.key_names.clone();
            dims.extend(first.dims().iter().cloned());
            let mut components = Vec::new();
            for (label, s) in segments.iter() {
                for key in part(&s.fit).components() {
                    let mut k: Vec<KeyValue> =
                        label.parts().iter().map(|p| p.as_str().into()).collect();
                    k.extend(key.iter().cloned());
                    components.push(k);
                }
            }
            let n_sims = first.n_sims();
            let mut draws = Vec::with_capacity(n_sims * components.len());
            for i in 0..n_sims {
                for s in &segments.fits {
                    let p = part(&s.fit);
                    let n = p.n_components();
                    draws.extend_from_slice(&p.draw_matrix()[i * n..(i + 1) * n]);
                }
            }
            let mut p = provenance.clone();
            p.parameters.retain(|(k, _)| k != "output");
            Ok(PredictiveDistribution::from_draws(
                dims,
                components,
                draws,
                p.param("output", output),
            )?)
        };
        // Every segment has the same settings, so all or none fit a second
        // response.
        let outstanding = if segments.fits[0].fit.outstanding.is_some() {
            Some(joint(
                |f| f.outstanding.as_ref().expect("every segment fits it"),
                "outstanding",
            )?)
        } else {
            None
        };
        let patterns = DevelopmentPatterns::joined(
            &segments.labels,
            segments
                .fits
                .iter()
                .map(|s| s.fit.patterns.clone())
                .collect(),
        );
        Ok(Self {
            reserves: joint(|f| &f.reserves, "reserves")?,
            paid: joint(|f| &f.paid, "paid")?,
            outstanding,
            patterns,
            segments,
        })
    }

    /// The segments' tables, stacked in index order under the key columns.
    fn stack(&self, tables: Vec<FitTable>) -> FitTable {
        stack_segments(&self.segments, tables)
    }

    /// One row per segment × origin: the columns of
    /// [`CompartmentalFit::to_long`] (`premium`, `latest_paid`, the `mean`
    /// and `std_dev` of the reserve, `ultimate_loss_ratio`).
    pub fn to_long(&self) -> FitTable {
        self.stack(self.segments.fits.iter().map(|s| s.fit.to_long()).collect())
    }

    /// One row per segment: the columns of [`CompartmentalFit::totals`].
    pub fn totals(&self) -> FitTable {
        self.stack(self.segments.fits.iter().map(|s| s.fit.totals()).collect())
    }

    /// One row per segment × future cell of [`paid`](Self::paid), as
    /// [`CompartmentalFit::paid_table`].
    pub fn paid_table(&self) -> Result<FitTable> {
        let tables = self.segments.fits.iter().map(|s| s.fit.paid_table());
        Ok(self.stack(tables.collect::<Result<_>>()?))
    }

    /// The same for [`outstanding`](Self::outstanding); an
    /// [`Error::NotSupported`] under [`Fitted::PaidOnly`].
    pub fn outstanding_table(&self) -> Result<FitTable> {
        let tables = self.segments.fits.iter().map(|s| s.fit.outstanding_table());
        Ok(self.stack(tables.collect::<Result<_>>()?))
    }

    /// The one segment chosen as in [`SegmentFits::position`], with its part
    /// of the joint distributions (same dimensions).
    pub fn segment(&self, keys: &[(&str, &str)]) -> Result<Self> {
        let segments = self.segments.segment(keys)?;
        Self::assemble(segments, self.reserves.provenance().clone())
    }
}

/// The segments' tables, stacked in index order under the key columns.
fn stack_segments<T>(segments: &SegmentFits<T>, tables: Vec<FitTable>) -> FitTable {
    let rows: Vec<usize> = tables
        .iter()
        .map(|t| t.origin.as_ref().map_or(1, Vec::len))
        .collect();
    let mut stacked = FitTable {
        keys: segments.key_columns(|s| rows[s]),
        origin: tables[0].origin.as_ref().map(|_| Vec::new()),
        age: tables[0].age.as_ref().map(|_| Vec::new()),
        values: tables[0]
            .values
            .iter()
            .map(|(name, _)| (name.clone(), Vec::new()))
            .collect(),
    };
    for t in tables {
        if let (Some(all), Some(o)) = (&mut stacked.origin, t.origin) {
            all.extend(o);
        }
        if let (Some(all), Some(a)) = (&mut stacked.age, t.age) {
            all.extend(a);
        }
        for ((_, all), (_, v)) in stacked.values.iter_mut().zip(t.values) {
            all.extend(v);
        }
    }
    stacked
}

/// The one-year view of a compartmental model in every segment of a
/// triangle ([`Compartmental::one_year_segments`]), with the layout of
/// [`CompartmentalFits`]: segments in index order, the joint claims
/// development result across them. Each segment holds the view
/// [`Compartmental::one_year`] gives it, as a [`BayesOneYearSegment`] of its
/// [`CompartmentalSegment`].
#[derive(Debug, Clone)]
pub struct CompartmentalOneYearFits {
    /// Each segment's posterior, opening reserve and Pareto `k̂`.
    pub segments: SegmentFits<BayesOneYearSegment<CompartmentalSegment>>,
    /// Joint distribution of the claims development result by segment and
    /// origin: the key names and `origin` are its dimensions, and its
    /// components run over the origins of each segment in turn.
    pub cdr: PredictiveDistribution,
    /// Why the views are approximate, or `None`, as
    /// [`BayesOneYearFit::approximate`]: the same for every segment, as the
    /// model is the same.
    pub approximate: Option<String>,
}

impl CompartmentalOneYearFits {
    /// The components of [`cdr`](Self::cdr) that belong to segment `s`.
    pub(super) fn range(&self, s: usize) -> std::ops::Range<usize> {
        let n = |t: &BayesOneYearSegment<CompartmentalSegment>| t.fit.fit.origins.len();
        let start: usize = self.segments.fits[..s].iter().map(n).sum();
        start..start + n(&self.segments.fits[s])
    }

    /// One row per segment × origin: `premium`, `latest_paid`,
    /// `opening_ultimate`, `opening_reserve`, and the `cdr_mean` and
    /// `cdr_std_dev` of the claims development result.
    pub fn to_long(&self) -> FitTable {
        let sums = component_sums(&self.cdr);
        let tables = self
            .segments
            .fits
            .iter()
            .enumerate()
            .map(|(s, seg)| {
                let fit = &seg.fit.fit;
                let mut table = FitTable {
                    keys: Vec::new(),
                    origin: Some(fit.origins.clone()),
                    age: None,
                    values: vec![
                        ("premium".into(), fit.premium.clone()),
                        ("latest_paid".into(), fit.latest_paid.clone()),
                        ("opening_ultimate".into(), seg.opening_ultimate.clone()),
                        ("opening_reserve".into(), seg.opening_reserve.clone()),
                    ],
                };
                push_moments(&mut table, "cdr_", &sums[self.range(s)]);
                table
            })
            .collect();
        stack_segments(&self.segments, tables)
    }

    /// One row per segment: the totals of [`to_long`](Self::to_long)'s
    /// columns, the segment's `flagged_share`, and the `cdr_mean` and
    /// `cdr_std_dev` of its total claims development result.
    pub fn totals(&self) -> FitTable {
        let tables = self
            .segments
            .fits
            .iter()
            .enumerate()
            .map(|(s, seg)| {
                let fit = &seg.fit.fit;
                let sum = |v: &[f64]| vec![v.iter().sum::<f64>()];
                let total = crate::bootstrap::sums(&self.cdr, self.range(s));
                FitTable {
                    keys: Vec::new(),
                    origin: None,
                    age: None,
                    values: vec![
                        ("premium".into(), sum(&fit.premium)),
                        ("latest_paid".into(), sum(&fit.latest_paid)),
                        ("opening_ultimate".into(), sum(&seg.opening_ultimate)),
                        ("opening_reserve".into(), sum(&seg.opening_reserve)),
                        ("flagged_share".into(), vec![seg.flagged_share()]),
                        ("cdr_mean".into(), vec![total.mean()]),
                        ("cdr_std_dev".into(), vec![total.std_dev()]),
                    ],
                }
            })
            .collect();
        stack_segments(&self.segments, tables)
    }

    /// The one segment chosen as in [`SegmentFits::position`], with its
    /// part of the joint claims development result (same dimensions).
    pub fn segment(&self, keys: &[(&str, &str)]) -> Result<Self> {
        let s = self.segments.position(keys)?;
        let range = self.range(s);
        let n = self.cdr.n_components();
        let part = self
            .cdr
            .draw_matrix()
            .chunks_exact(n)
            .flat_map(|row| row[range.clone()].iter().copied())
            .collect();
        let cdr = PredictiveDistribution::from_draws(
            self.cdr.dims().to_vec(),
            self.cdr.components()[range].to_vec(),
            part,
            self.cdr.provenance().clone(),
        )?;
        Ok(Self {
            segments: SegmentFits {
                key_names: self.segments.key_names.clone(),
                labels: vec![self.segments.labels[s].clone()],
                fits: vec![self.segments.fits[s].clone()],
            },
            cdr,
            approximate: self.approximate.clone(),
        })
    }
}

/// The one-year view of a compartmental model
/// ([`Compartmental::one_year`]): its tables, by the premium column's
/// origins, as [`CompartmentalFit`]'s.
impl BayesOneYearFit<CompartmentalFit> {
    /// The share of simulations whose `k̂` is above the threshold (or
    /// NaN), whether or not they were refitted.
    pub fn flagged_share(&self) -> f64 {
        one_year::flagged_share(&self.pareto_k, self.k_threshold)
    }

    /// One row per origin: `premium`, `latest_paid`, `opening_ultimate`,
    /// `opening_reserve`, and the `cdr_mean` and `cdr_std_dev` of the
    /// claims development result.
    pub fn to_long(&self) -> FitTable {
        let mut table = FitTable {
            keys: Vec::new(),
            origin: Some(self.fit.origins.clone()),
            age: None,
            values: vec![
                ("premium".into(), self.fit.premium.clone()),
                ("latest_paid".into(), self.fit.latest_paid.clone()),
                ("opening_ultimate".into(), self.opening_ultimate.clone()),
                ("opening_reserve".into(), self.opening_reserve.clone()),
            ],
        };
        push_moments(&mut table, "cdr_", &component_sums(&self.cdr));
        table
    }

    /// One row: the totals of [`to_long`](Self::to_long)'s columns, the
    /// `flagged_share`, and the `cdr_mean` and `cdr_std_dev` of the total
    /// claims development result.
    pub fn totals(&self) -> FitTable {
        let total = self.cdr.total();
        let sum = |v: &[f64]| vec![v.iter().sum::<f64>()];
        FitTable {
            keys: Vec::new(),
            origin: None,
            age: None,
            values: vec![
                ("premium".into(), sum(&self.fit.premium)),
                ("latest_paid".into(), sum(&self.fit.latest_paid)),
                ("opening_ultimate".into(), sum(&self.opening_ultimate)),
                ("opening_reserve".into(), sum(&self.opening_reserve)),
                ("flagged_share".into(), vec![self.flagged_share()]),
                ("cdr_mean".into(), vec![total.mean()]),
                ("cdr_std_dev".into(), vec![total.std_dev()]),
            ],
        }
    }
}

/// [`CompartmentalFit::loo`]: PSIS-LOO over every observed cell, with the
/// elpd split by type.
#[derive(Debug, Clone)]
pub struct CompartmentalLoo {
    /// PSIS-LOO over the outstanding cells and then the paid cells, each by
    /// origin and age.
    pub loo: Loo,
    /// The number of outstanding cells, which come first.
    pub n_outstanding: usize,
    /// The elpd of the outstanding cells.
    pub elpd_outstanding: f64,
    /// The elpd of the paid cells.
    pub elpd_paid: f64,
}

impl CompartmentalFit {
    /// Posterior summary of each parameter: mean, sd, 5/50/95% quantiles,
    /// R-hat, bulk and tail ESS.
    pub fn summary(&self) -> Result<Vec<ParamSummary>> {
        Ok(self.posterior.summary()?)
    }

    /// PSIS-LOO of the observed cells, with each cell's relative efficiency
    /// from the chains. The pointwise term is the full log density of each
    /// cell's response under the fit's likelihood (brms' `log_lik`): normal,
    /// or log-normal including its `-ln y` term, on the fit's scale (amounts
    /// or loss ratios to premium) and paid response (cumulative or
    /// incremental). So elpd_loo compares only fits with the same scale and
    /// paid response: model 2's elpd on loss ratios does not compare with
    /// model 1's on amounts. Check each cell's Pareto k: model 2 on the
    /// papers' training triangle has a few above 0.7 (up to about 1.3), so
    /// there the estimate is rough.
    pub fn loo(&self) -> Result<CompartmentalLoo> {
        let model = &self.model;
        let n = model.cells.len();
        let post = &self.posterior;
        let s = post.n_draws();
        let mut ll = Vec::with_capacity(s * n);
        for nat in post.values().chunks_exact(post.dim()) {
            model.log_likelihoods(nat, &mut ll);
        }
        let chains = post.chains();
        let per_chain = s / chains;
        let r_eff: Vec<f64> = (0..n)
            .map(|i| {
                let by_chain: Vec<Vec<f64>> = (0..chains)
                    .map(|c| {
                        (0..per_chain)
                            .map(|k| ll[(c * per_chain + k) * n + i].exp())
                            .collect()
                    })
                    .collect();
                let refs: Vec<&[f64]> = by_chain.iter().map(Vec::as_slice).collect();
                let ess = prospicio_bayes::ess_mean(&refs).unwrap_or(s as f64);
                if ess.is_finite() && ess > 0.0 {
                    ess / s as f64
                } else {
                    1.0
                }
            })
            .collect();
        let loo = prospicio_bayes::elpd::loo(&ll, n, Some(&r_eff))?;
        let n_os = model.n_outstanding;
        let elpd_outstanding = loo.estimate.pointwise[..n_os].iter().sum();
        let elpd_paid = loo.estimate.pointwise[n_os..].iter().sum();
        Ok(CompartmentalLoo {
            loo,
            n_outstanding: n_os,
            elpd_outstanding,
            elpd_paid,
        })
    }

    /// One row per origin: `premium`, `latest_paid`, the `mean` and
    /// `std_dev` of the posterior predictive reserve, and the posterior mean
    /// `ultimate_loss_ratio`.
    pub fn to_long(&self) -> FitTable {
        let sums = component_sums(&self.reserves);
        FitTable {
            keys: Vec::new(),
            origin: Some(self.origins.clone()),
            age: None,
            values: vec![
                ("premium".into(), self.premium.clone()),
                ("latest_paid".into(), self.latest_paid.clone()),
                ("mean".into(), sums.iter().map(|d| d.mean()).collect()),
                ("std_dev".into(), sums.iter().map(|d| d.std_dev()).collect()),
                (
                    "ultimate_loss_ratio".into(),
                    self.ultimate_loss_ratio.clone(),
                ),
            ],
        }
    }

    /// One row: the total `premium` and `latest_paid`, and the `mean` and
    /// `std_dev` of the posterior predictive total reserve.
    pub fn totals(&self) -> FitTable {
        let total = self.reserves.total();
        FitTable {
            keys: Vec::new(),
            origin: None,
            age: None,
            values: vec![
                ("premium".into(), vec![self.premium.iter().sum()]),
                ("latest_paid".into(), vec![self.latest_paid.iter().sum()]),
                ("mean".into(), vec![total.mean()]),
                ("std_dev".into(), vec![total.std_dev()]),
            ],
        }
    }

    /// One row per future cell of [`paid`](Self::paid): its origin and age,
    /// and the `mean`, `std_dev` and 5%, 50% and 95% quantiles (`q05`,
    /// `q50`, `q95`) of the posterior predictive cumulative paid.
    pub fn paid_table(&self) -> Result<FitTable> {
        cell_table(&self.paid)
    }

    /// The same for [`outstanding`](Self::outstanding); an
    /// [`Error::NotSupported`] under [`Fitted::PaidOnly`], which fits no second
    /// response.
    pub fn outstanding_table(&self) -> Result<FitTable> {
        match &self.outstanding {
            Some(outstanding) => cell_table(outstanding),
            None => Err(Error::NotSupported {
                setting: "Fitted::PaidOnly",
                method: "CompartmentalFit::outstanding_table",
                reason: "a fit of paid alone has no outstanding cells",
            }),
        }
    }

    /// The reserves, the future paid cells and, when fitted, the future
    /// cells of the second response: the parts of a segment's row in
    /// [`Compartmental::fit_segments`].
    pub(super) fn parts(&self) -> Vec<&PredictiveDistribution> {
        let mut parts = vec![&self.reserves, &self.paid];
        parts.extend(&self.outstanding);
        parts
    }
}

/// A long table of a cell distribution with dimensions `origin` and `age`.
fn cell_table(cells: &PredictiveDistribution) -> Result<FitTable> {
    let sums = component_sums(cells);
    let mut origin = Vec::with_capacity(sums.len());
    let mut age = Vec::with_capacity(sums.len());
    for key in cells.components() {
        match (&key[0], &key[1]) {
            (KeyValue::Period(o), KeyValue::Int(a)) => {
                origin.push(*o);
                age.push(*a as Lag);
            }
            _ => unreachable!("cell keys are (origin, age)"),
        }
    }
    let q = |p: f64| -> Result<Vec<f64>> {
        sums.iter()
            .map(|d| Ok(d.quantile(p)?))
            .collect::<Result<Vec<f64>>>()
    };
    Ok(FitTable {
        keys: Vec::new(),
        origin: Some(origin),
        age: Some(age),
        values: vec![
            ("mean".into(), sums.iter().map(|d| d.mean()).collect()),
            ("std_dev".into(), sums.iter().map(|d| d.std_dev()).collect()),
            ("q05".into(), q(0.05)?),
            ("q50".into(), q(0.5)?),
            ("q95".into(), q(0.95)?),
        ],
    })
}
