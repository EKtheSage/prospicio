//! The hierarchical Bornhuetter-Ferguson (`HierarchicalBf`,
//! `docs/design/reserving-bayes.md`, decision 30, "Validation plan").
//!
//! Item 1 (one program and a tight hyperprior reduce to `BayesBfNormalised`)
//! and the one-year conformance suite are unit tests of the model's file and
//! `reserving_bayes_one_year_conformance.rs`. This file holds items 2, 3, 4
//! and 6, and the one-year limit of item 1 (the PR 17 check that the one-year
//! view of one program with a tight hyperprior is `BayesBfNormalised`'s):
//!
//! * Item 2, shrinkage as Buhlmann-Straub: twenty simulated programs of
//!   fully developed squares, one shared pattern (`omega_scale` 1e-6), a
//!   small within `cv` and small process error. Each program's posterior
//!   mean `log(L s_c)` against `HierarchicalCredibility` on the programs'
//!   log developed ratios, and the shrinkage `Z_c` against
//!   `credibility_premiums` at the posterior mean structure parameters.
//! * Item 3, simulation recovery of `L`, `tau` and `omega`. Each panel draws
//!   its truth from the priors the fit uses (a lognormal `L`, half-normal
//!   `tau` and `omega`), so that the rank of the truth among the posterior
//!   draws is uniform when the model is right (simulation-based calibration);
//!   a flat prior on `log L` has no draws. `cv` is the model's, the pattern
//!   `beta` a fixed one, and the cells' scale is simulated at `kappa` and
//!   plugged in from each panel's chain ladders as the model does, so the
//!   plug-in's error is part of what the study measures. Over 100 panels the
//!   90% interval of each of `L`, `tau` and `omega` must cover its truth 80
//!   to 97 times and the ranks (ten bins) must pass a chi-square test at
//!   1%. The 100-panel study runs under `RISK_RS_FULL_BAYES` (weekly and
//!   manual CI runs) and by hand as the ignored test `recovery_full`; a pull
//!   request runs five panels, which only check that the fits complete and
//!   pass the sampler's thresholds.
//! * Item 6, the thin-program property: panels of five programs of two or
//!   three origins. A program of two origins has no degree of freedom of its
//!   own (`BayesBfNormalised` refuses it), so the comparison with separate
//!   `BayesBfNormalised` fits at a vague a priori is on the programs of
//!   three origins; the hierarchical fit has a degree of freedom for each.
//!
//! * Item 4, Stan parity: three programs (EVW with its prior as premium, cut
//!   to six ages, GenIns with premium, RAA with a synthetic premium) against
//!   `reference/reserving_hierarchical_bf_stan.csv`, from
//!   `scripts/reserving_hierarchical_bf_stan.py` and
//!   `scripts/stan/hierarchical_bf.stan` (CmdStan 2.39.0 compiled and ran, so
//!   the parity is claimed).

use std::time::Instant;

use prospicio_bayes::credibility::{
    CredibilityPriors, HierarchicalCredibility, ScalePrior, credibility_premiums,
};
use prospicio_core::StreamRng;
use prospicio_math::special::{digamma, gamma_inc, trigamma};
use prospicio_prob::Distribution;
use prospicio_reserving::bayes::{BayesBfNormalised, HierarchicalBf, HierarchicalBfFit};
use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Period, Triangle};
use prospicio_validation::{check, full_bayes, reference, stan_triangle, triangle as dataset};
use rayon::prelude::*;

// ---------------------------------------------------------------------
// A seeded simulator of the model
// ---------------------------------------------------------------------

struct Rng(StreamRng);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(StreamRng::new(seed, 0))
    }

    fn uniform(&mut self) -> f64 {
        self.0.next_open01()
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }

    /// Gamma with this shape and scale (Marsaglia and Tsang).
    fn gamma(&mut self, shape: f64, scale: f64) -> f64 {
        if shape < 1.0 {
            let u = self.uniform();
            return self.gamma(shape + 1.0, scale) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.uniform();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v * scale;
            }
        }
    }
}

/// The hyperparameters of a simulated panel.
#[derive(Clone, Debug)]
struct Truth {
    level: f64,
    tau: f64,
    omega: f64,
    cv: f64,
    kappa: f64,
    /// The shared log share ratios `beta_2..beta_m`.
    beta: Vec<f64>,
}

/// The shape of a panel: origins per program, ages, whether every origin is
/// observed at every age (a fully developed square), the premium of each
/// program's first origin and its growth per origin.
struct Layout {
    origins: Vec<usize>,
    ages: usize,
    square: bool,
    base: Vec<f64>,
    growth: f64,
}

/// One simulated program.
struct Program {
    label: String,
    cumulative: Vec<Vec<f64>>,
    premium: Vec<f64>,
    /// The true expected reserve of each origin: its expected future cells.
    expected_reserve: Vec<f64>,
}

struct Panel {
    programs: Vec<Program>,
}

#[allow(clippy::needless_range_loop)]
fn simulate(rng: &mut Rng, truth: &Truth, layout: &Layout) -> Panel {
    let m = layout.ages;
    let shape = truth.cv.powi(-2);
    let mut programs = Vec::new();
    for (c, &n) in layout.origins.iter().enumerate() {
        let z = rng.normal();
        let ratio = truth.level * (truth.tau * z - 0.5 * truth.tau * truth.tau).exp();
        let mut beta = vec![0.0; m];
        for d in 1..m {
            beta[d] = truth.beta[d - 1] + truth.omega * rng.normal();
        }
        let norm: f64 = beta.iter().map(|b| b.exp()).sum();
        let share: Vec<f64> = beta.iter().map(|b| b.exp() / norm).collect();
        let premium: Vec<f64> = (0..n)
            .map(|w| layout.base[c] * (1.0 + layout.growth * w as f64))
            .collect();
        let mean_premium = premium.iter().sum::<f64>() / n as f64;
        let phi = truth.kappa * mean_premium;
        let (mut cumulative, mut expected_reserve) = (vec![], vec![]);
        for w in 0..n {
            let u = rng.gamma(shape, ratio * premium[w] / shape);
            let last = if layout.square {
                m - 1
            } else {
                (m - 1).min(n - 1 - w)
            };
            let (mut cum, mut row, mut future) = (0.0, vec![], 0.0);
            for d in 0..m {
                let mu = u * share[d];
                if d <= last {
                    cum += rng.gamma(mu / phi, phi);
                    row.push(cum);
                } else {
                    future += mu;
                }
            }
            cumulative.push(row);
            expected_reserve.push(future);
        }
        programs.push(Program {
            label: format!("p{c:02}"),
            cumulative,
            premium,
            expected_reserve,
        });
    }
    Panel { programs }
}

/// A triangle with a segment for each of `programs`.
fn triangle(programs: &[&Program]) -> Triangle {
    let (mut key, mut origin, mut age, mut paid, mut premium) =
        (vec![], vec![], vec![], vec![], vec![]);
    for p in programs {
        for (w, row) in p.cumulative.iter().enumerate() {
            for (d, &x) in row.iter().enumerate() {
                key.push(p.label.as_str());
                origin.push(Month::january(2000 + w as i32));
                age.push(12 * (d as u32 + 1));
                paid.push(x);
                premium.push(p.premium[w]);
            }
        }
    }
    Triangle::from_long(&Long {
        keys: &[("program", &key)],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("paid", &paid), ("premium", &premium)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .expect("a triangle")
}

fn whole(panel: &Panel) -> Triangle {
    triangle(&panel.programs.iter().collect::<Vec<_>>())
}

// ---------------------------------------------------------------------
// Posterior summaries
// ---------------------------------------------------------------------

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn sd(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt()
}

/// All draws of a sampled parameter, chains pooled.
fn draws(fit: &HierarchicalBfFit, name: &str) -> Vec<f64> {
    let j = fit.posterior.index(name).expect("a sampled name");
    fit.posterior.parameter(j).concat()
}

/// Draws of `log(L s_c)`.
fn log_ratio(fit: &HierarchicalBfFit, c: usize) -> Vec<f64> {
    let (level, log_tau, z) = (
        draws(fit, "log_level"),
        draws(fit, "log_tau"),
        draws(fit, &format!("z[{}]", c + 1)),
    );
    (0..level.len())
        .map(|i| {
            let tau = log_tau[i].exp();
            level[i] + tau * z[i] - 0.5 * tau * tau
        })
        .collect()
}

/// Draws of `log L - tau^2 / 2`, the mean of `log(L s_c)`.
fn log_mu(fit: &HierarchicalBfFit) -> Vec<f64> {
    let (level, log_tau) = (draws(fit, "log_level"), draws(fit, "log_tau"));
    level
        .iter()
        .zip(&log_tau)
        .map(|(l, t)| l - 0.5 * (2.0 * t).exp())
        .collect()
}

fn fit_hier(model: &HierarchicalBf, tri: &Triangle, what: &str) -> HierarchicalBfFit {
    let start = Instant::now();
    let fit = model.fit(tri, "paid", "premium").expect("fits");
    eprintln!(
        "HierarchicalBf {what}: fit in {:.2?}; {:?}",
        start.elapsed(),
        fit.diagnostics
    );
    fit
}

// ---------------------------------------------------------------------
// Item 2: shrinkage as Buhlmann-Straub
// ---------------------------------------------------------------------

/// Twenty programs of five origins, fully developed over four ages, equal
/// premiums within a program, a within `cv` of 0.1 and process error of
/// about 1% (`kappa` 1e-4), one shared pattern (`omega` 0). The truth is
/// `L` 0.7 and `tau` 0.07, so that the programs' own levels are shrunk by
/// about a third (`Z` near 0.65): at the note's `cv` of 0.05 and a `tau` of
/// 0.25 `Z` is 0.99 and the check says little.
fn buhlmann_straub_panel(seed: u64) -> Panel {
    let truth = Truth {
        level: 0.7,
        tau: 0.07,
        omega: 0.0,
        cv: 0.1,
        kappa: 1e-4,
        beta: vec![-0.15, -0.85, -1.25],
    };
    let mut rng = Rng::new(seed);
    let layout = Layout {
        origins: vec![5; 20],
        ages: 4,
        square: true,
        base: (0..20)
            .map(|_| 200.0 * (0.4 * rng.normal()).exp())
            .collect(),
        growth: 0.0,
    };
    simulate(&mut rng, &truth, &layout)
}

/// What item 2 measures on one simulated panel. A gap in "sd" is in units of
/// the program's posterior sd of `log(L s_c)`; a `Z` gap is over the programs
/// whose own level is at least one sd of the levels from the collective's
/// (elsewhere `Z` is a ratio of two small numbers).
struct Shrinkage {
    /// Against `HierarchicalCredibility`'s posterior mean premiums: the
    /// largest mean gap, the largest `Z` gap against `credibility_premiums`
    /// at its posterior mean structure parameters (the note's check), and
    /// the pooled slope's gap to that `Z`.
    hc_mean_gap: f64,
    hc_z_gap: f64,
    hc_slope_gap: f64,
    /// Against `credibility_premiums` at the sampled model's own structure
    /// (its fixed within variance and each posterior draw of `tau^2`),
    /// averaged over the draws: the same three gaps.
    mean_gap: f64,
    z_gap: f64,
    slope_gap: f64,
    divergences: usize,
}

fn buhlmann_straub(seed: u64, verbose: bool) -> Shrinkage {
    let panel = buhlmann_straub_panel(seed);
    let tri = whole(&panel);
    let model = HierarchicalBf {
        apriori: 0.7,
        level_cv: None,
        cv: 0.1,
        omega_scale: 1e-6,
        n_sims: 10,
        seed,
        ..HierarchicalBf::default()
    };
    let fit = fit_hier(&model, &tri, &format!("Buhlmann-Straub panel seed {seed}"));
    assert!(fit.diagnostics.warnings.is_empty(), "{:?}", fit.diagnostics);

    // Every origin is at ultimate (its last cumulative value). The credibility
    // model's data are the log developed ratios, moved from the mean of the
    // log of a gamma to the log of its mean (the model's `L s_c P` is the
    // mean), and its within variance the log gamma's: trigamma(1 / cv^2).
    let shape = model.cv.powi(-2);
    let shift = digamma(shape) - shape.ln();
    let within_model = trigamma(shape);
    let ratios: Vec<Vec<f64>> = panel
        .programs
        .iter()
        .map(|p| {
            p.cumulative
                .iter()
                .zip(&p.premium)
                .map(|(row, premium)| (row.last().unwrap() / premium).ln() - shift)
                .collect()
        })
        .collect();
    let weights = vec![vec![1.0; 5]; 20];
    let credibility = HierarchicalCredibility::default()
        .priors(CredibilityPriors {
            between: vec![ScalePrior::HalfNormal(model.tau_scale)],
            ..CredibilityPriors::default()
        })
        .fit(&ratios, &weights, &[])
        .expect("credibility fits");
    let structure = credibility.summary().expect("summary");
    let hc = credibility_premiums(
        &ratios,
        &weights,
        &[],
        structure[0].mean,
        &[structure[1].mean],
        None,
    )
    .expect("credibility premiums");
    // The model's structure: its fixed within variance and, for every tenth
    // posterior draw, tau^2; the premiums and credibility factors averaged.
    let tau2: Vec<f64> = draws(&fit, "log_tau")
        .iter()
        .map(|t| (2.0 * t).exp())
        .collect();
    let (mut premium, mut factor) = (vec![0.0; 20], vec![0.0; 20]);
    let used: Vec<f64> = tau2.iter().step_by(10).copied().collect();
    for &t2 in &used {
        let r = credibility_premiums(&ratios, &weights, &[], within_model, &[t2], None)
            .expect("credibility premiums");
        for c in 0..20 {
            premium[c] += r.premiums[c] / used.len() as f64;
            factor[c] += r.credibility[c] / used.len() as f64;
        }
    }

    // The shrinkage Z_c = (post_c - mu) / (own_c - mu), with one definition
    // of the group's own level (the mean of its adjusted logs) and `mu`
    // the posterior mean of `log L - tau^2 / 2`.
    let mu = mean(&log_mu(&fit));
    let own: Vec<f64> = ratios.iter().map(|g| mean(g)).collect();
    let spread = sd(&own);
    let mut out = Shrinkage {
        hc_mean_gap: 0.0,
        hc_z_gap: 0.0,
        hc_slope_gap: 0.0,
        mean_gap: 0.0,
        z_gap: 0.0,
        slope_gap: 0.0,
        divergences: fit.diagnostics.divergences,
    };
    let (mut cross, mut square) = (0.0, 0.0);
    for c in 0..20 {
        let draws_c = log_ratio(&fit, c);
        let (post, s) = (mean(&draws_c), sd(&draws_c));
        out.hc_mean_gap = out
            .hc_mean_gap
            .max((post - credibility.premiums[c]).abs() / s);
        out.mean_gap = out.mean_gap.max((post - premium[c]).abs() / s);
        cross += (post - mu) * (own[c] - mu);
        square += (own[c] - mu).powi(2);
        if (own[c] - mu).abs() >= spread {
            let z = (post - mu) / (own[c] - mu);
            out.hc_z_gap = out.hc_z_gap.max((z - hc.credibility[c]).abs());
            out.z_gap = out.z_gap.max((z - factor[c]).abs());
        }
        if verbose && c < 3 {
            eprintln!(
                "  program {c}: model {post:.4} (sd {s:.4}); credibility {:.4}, at the model's \
                 structure {:.4}; fit.shrinkage {:.3}",
                credibility.premiums[c], premium[c], fit.programs[c].shrinkage
            );
        }
    }
    let slope = cross / square;
    out.hc_slope_gap = (slope - hc.credibility[0]).abs();
    out.slope_gap = (slope - mean(&factor)).abs();
    eprintln!(
        "  credibility: within var {:.5}, between var {:.5}, Z {:.3}; model: within var \
         {within_model:.5}, tau^2 {:.5}, mean Z {:.3}; pooled slope {slope:.3}",
        structure[0].mean,
        structure[1].mean,
        hc.credibility[0],
        mean(&tau2),
        mean(&factor)
    );
    eprintln!(
        "  gaps against credibility: mean {:.3} sd, Z {:.4}, slope {:.4}; at the model's \
         structure: mean {:.3} sd, Z {:.4}, slope {:.4}",
        out.hc_mean_gap, out.hc_z_gap, out.hc_slope_gap, out.mean_gap, out.z_gap, out.slope_gap
    );
    out
}

/// Item 2's checks. Against `credibility_premiums` at the sampled model's
/// own structure (its fixed within variance, each draw of `tau^2`) the
/// shrinkage is Buhlmann-Straub's: over seeds 1 to 6 (the sampler's defaults,
/// target accept 0.99, no divergence and no warning; measured) the pooled
/// slope is within 0.0195 of the mean credibility factor, each program's
/// posterior mean within 0.156 sd and its `Z` within 0.058; the tolerances are
/// 0.03, 0.25 sd and 0.10, at least 1.5 times the largest. The note proposed
/// 0.1 sd and 0.05; 0.1 sd was exceeded on three seeds (the programs' `Z` is a
/// ratio of two small numbers) and is not claimed.
///
/// Against `HierarchicalCredibility`'s own posterior (the note's check) the
/// agreement is looser, because it estimates the within variance from the 80
/// degrees of freedom of the data (0.0076 to 0.0113 over the seeds) where this
/// model fixes it at the log gamma's 0.01005: `Z` differs by up to 0.15, the
/// means by up to 0.41 sd and the pooled slope by up to 0.13 (measured), so
/// those are only guarded (0.6 sd, 0.2).
///
/// A pull request checks seed 1; `RISK_RS_FULL_BAYES` checks all six.
#[test]
fn shrinkage_is_buhlmann_straub() {
    // Seed 1 in a pull request, all six under `RISK_RS_FULL_BAYES`.
    let seeds = if full_bayes() { 1..=6 } else { 1..=1 };
    for seed in seeds {
        let s = buhlmann_straub(seed, seed == 1);
        assert_eq!(s.divergences, 0, "seed {seed}");
        assert!(s.mean_gap < 0.25, "seed {seed}: mean gap {} sd", s.mean_gap);
        assert!(s.z_gap < 0.10, "seed {seed}: Z gap {}", s.z_gap);
        assert!(
            s.slope_gap < 0.03,
            "seed {seed}: pooled slope gap {}",
            s.slope_gap
        );
        assert!(
            s.hc_mean_gap < 0.6,
            "seed {seed}: gap to HC {} sd",
            s.hc_mean_gap
        );
        assert!(
            s.hc_slope_gap < 0.2,
            "seed {seed}: slope gap to HC {}",
            s.hc_slope_gap
        );
    }
}

/// The seed variation behind item 2's tolerances (`--ignored --nocapture`).
#[test]
#[ignore]
fn buhlmann_straub_seed_sweep() {
    for seed in 1..=6 {
        let s = buhlmann_straub(seed, false);
        eprintln!("seed {seed}: divergences {}", s.divergences);
    }
}

// ---------------------------------------------------------------------
// Item 3: simulation recovery of L, tau and omega
// ---------------------------------------------------------------------

/// The hyperprior of the fit and of the panels' truths.
const LEVEL_MEAN: f64 = 0.7;
const LEVEL_CV: f64 = 0.2;
const TAU_SCALE: f64 = 0.3;
const OMEGA_SCALE: f64 = 0.5;
/// Ages of a recovery panel and the shared log share ratios (shares 0.35,
/// 0.30, 0.15, 0.10, 0.10).
const RECOVERY_AGES: usize = 5;
const RECOVERY_BETA: [f64; 4] = [-0.1542, -0.8473, -1.2528, -1.2528];

/// What one recovery panel gave for `L`, `tau` and `omega`.
struct Recovery {
    /// The fraction of the posterior draws below the truth.
    rank: [f64; 3],
    covered: [bool; 3],
    divergences: usize,
    warnings: usize,
    max_rhat: f64,
    min_ess: f64,
    seconds: f64,
}

/// A panel of `programs` programs of three to ten origins on five ages, its
/// truth drawn from the hyperprior of the fit, and the model that fits it.
fn recovery_setup(programs: usize, index: usize) -> (Truth, Triangle, HierarchicalBf) {
    let seed = 10_000 * programs as u64 + index as u64;
    let mut rng = Rng::new(seed);
    let variance = (1.0 + LEVEL_CV * LEVEL_CV).ln();
    let truth = Truth {
        level: (LEVEL_MEAN.ln() - 0.5 * variance + variance.sqrt() * rng.normal()).exp(),
        tau: TAU_SCALE * rng.normal().abs(),
        omega: OMEGA_SCALE * rng.normal().abs(),
        cv: 0.1,
        kappa: 0.01,
        beta: RECOVERY_BETA.to_vec(),
    };
    // Three to ten origins, and a program of at least five among them so
    // that the oldest age is observed.
    let layout = Layout {
        origins: (0..programs).map(|c| 3 + (3 * c) % 8).collect(),
        ages: RECOVERY_AGES,
        square: false,
        base: (0..programs)
            .map(|_| 200.0 * (0.5 * rng.normal()).exp())
            .collect(),
        growth: 0.05,
    };
    let panel = simulate(&mut rng, &truth, &layout);
    let model = HierarchicalBf {
        apriori: LEVEL_MEAN,
        level_cv: Some(LEVEL_CV),
        cv: truth.cv,
        tau_scale: TAU_SCALE,
        omega_scale: OMEGA_SCALE,
        n_sims: 10,
        seed,
        ..HierarchicalBf::default()
    };
    (truth, whole(&panel), model)
}

/// A recovery panel ([`recovery_setup`]) and its fit.
fn recovery_panel(programs: usize, index: usize) -> Recovery {
    let (truth, tri, model) = recovery_setup(programs, index);
    let start = Instant::now();
    let fit = model
        .fit(&tri, "paid", "premium")
        .unwrap_or_else(|e| panic!("panel {programs}/{index} {truth:?}: {e}"));
    let seconds = start.elapsed().as_secs_f64();
    if !fit.diagnostics.warnings.is_empty() {
        eprintln!(
            "  panel {programs}/{index} (L {:.3}, tau {:.3}, omega {:.3}): {} divergences; {}",
            truth.level,
            truth.tau,
            truth.omega,
            fit.diagnostics.divergences,
            fit.diagnostics.warnings.join("; ")
        );
    }
    let omega = fit.posterior.index("log_omega").expect("a sampled name");
    let d = fit.posterior.dim();
    let values = fit.posterior.values();
    let n = fit.posterior.n_draws() as f64;
    let mut out = Recovery {
        rank: [0.0; 3],
        covered: [false; 3],
        divergences: fit.diagnostics.divergences,
        warnings: fit.diagnostics.warnings.len(),
        max_rhat: fit.diagnostics.max_rhat,
        min_ess: fit
            .diagnostics
            .min_ess_bulk
            .min(fit.diagnostics.min_ess_tail),
        seconds,
    };
    // The stored scale has log L, log tau and log omega.
    for (k, (truth_k, column)) in [(truth.level, 0), (truth.tau, 1), (truth.omega, omega)]
        .into_iter()
        .enumerate()
    {
        let mut sample: Vec<f64> = values.chunks_exact(d).map(|t| t[column].exp()).collect();
        sample.sort_by(f64::total_cmp);
        let below = sample.iter().filter(|&&v| v < truth_k).count() as f64;
        out.rank[k] = below / n;
        let (lo, hi) = (sample[(0.05 * n) as usize], sample[(0.95 * n) as usize]);
        out.covered[k] = lo <= truth_k && truth_k <= hi;
    }
    out
}

/// `P(chi^2_k > x)`.
fn chi_square_sf(x: f64, k: f64) -> f64 {
    gamma_inc(0.5 * k, 0.5 * x).1
}

struct Study {
    coverage: [usize; 3],
    chi_square: [f64; 3],
    p_value: [f64; 3],
    warnings: usize,
    divergences: usize,
    max_rhat: f64,
    min_ess: f64,
}

/// `n` panels of `programs` programs, fitted in parallel.
fn recovery_study(programs: usize, n: usize) -> Study {
    let start = Instant::now();
    let results: Vec<Recovery> = (0..n)
        .into_par_iter()
        .map(|i| recovery_panel(programs, i))
        .collect();
    let mut study = Study {
        coverage: [0; 3],
        chi_square: [0.0; 3],
        p_value: [0.0; 3],
        warnings: results.iter().filter(|r| r.warnings > 0).count(),
        divergences: results.iter().map(|r| r.divergences).sum(),
        max_rhat: results.iter().map(|r| r.max_rhat).fold(0.0, f64::max),
        min_ess: results
            .iter()
            .map(|r| r.min_ess)
            .fold(f64::INFINITY, f64::min),
    };
    for (k, name) in ["L", "tau", "omega"].into_iter().enumerate() {
        study.coverage[k] = results.iter().filter(|r| r.covered[k]).count();
        let mut bins = [0usize; 10];
        for r in &results {
            bins[((r.rank[k] * 10.0) as usize).min(9)] += 1;
        }
        let expected = n as f64 / 10.0;
        study.chi_square[k] = bins
            .iter()
            .map(|&b| (b as f64 - expected).powi(2) / expected)
            .sum();
        study.p_value[k] = chi_square_sf(study.chi_square[k], 9.0);
        eprintln!(
            "recovery, {programs} programs, {n} panels: {name} covered {}/{n}, rank bins \
             {bins:?}, chi-square {:.2} (p {:.3})",
            study.coverage[k], study.chi_square[k], study.p_value[k]
        );
    }
    let seconds: Vec<f64> = results.iter().map(|r| r.seconds).collect();
    eprintln!(
        "recovery, {programs} programs: {} divergent transitions, {} fits with warnings; fit \
         time {:.1}s mean, {:.1}s max; wall {:.1?}",
        results.iter().map(|r| r.divergences).sum::<usize>(),
        study.warnings,
        mean(&seconds),
        seconds.iter().copied().fold(0.0, f64::max),
        start.elapsed()
    );
    study
}

/// The 100-panel study's bounds, as the note gives them: a 90% interval
/// covers the truth 80 to 97 times in 100, and the ranks pass a chi-square
/// test (10 bins; here at 1%, since six tests are run).
fn assert_recovery(study: &Study, programs: usize) {
    for (k, name) in ["L", "tau", "omega"].into_iter().enumerate() {
        assert!(
            (80..=97).contains(&study.coverage[k]),
            "{programs} programs: the 90% interval of {name} covered {} of 100",
            study.coverage[k]
        );
        assert!(
            study.p_value[k] > 0.01,
            "{programs} programs: the ranks of {name} are not uniform: chi-square {:.2}, p {:.4}",
            study.chi_square[k],
            study.p_value[k]
        );
    }
}

/// Five panels of 5 and of 20 programs in a pull request (the fits complete
/// with R-hat below 1.05; five panels cannot test coverage), and the
/// 100-panel study of each under `RISK_RS_FULL_BAYES`.
///
/// Recorded run (measured, default seeds `10_000 * programs + index`, the
/// defaults with target accept 0.99): 5 programs, the 90% interval covered
/// `L` 85, `tau` 89 and `omega` 90 times in 100, the rank chi-squares were
/// 6.4, 12.6 and 7.4 (p 0.70, 0.18, 0.60), 1 divergent transition in 100 fits
/// and 29 fits with a sampler warning (mostly `log_tau`); 20 programs, 88, 95
/// and 88, chi-squares 10.6, 11.6 and 7.6 (p 0.30, 0.24, 0.58), no divergent
/// transition and 11 fits with a warning (`log_tau` and, in a few, `z`). The
/// five panels of the pull request's run had no divergence and one fit with a
/// warning (R-hat 1.011 on two coordinates) among the 5-program panels, none
/// among the 20-program panels.
#[test]
fn recovery_of_the_hyperparameters() {
    let n = if full_bayes() { 100 } else { 5 };
    for programs in [5, 20] {
        let study = recovery_study(programs, n);
        if n >= 100 {
            assert_recovery(&study, programs);
        } else {
            // The sampler's thresholds are strict for tau of five programs
            // (29 of 100 fits warned, all R-hat or ESS marginally past the
            // limit): the smoke run guards against a failed sampler, not a
            // marginal one.
            assert!(
                study.max_rhat < 1.05,
                "{programs} programs: R-hat {}",
                study.max_rhat
            );
            assert!(
                study.min_ess > 100.0,
                "{programs} programs: ESS {}",
                study.min_ess
            );
            assert!(
                study.divergences <= 2,
                "{programs} programs: {} divergences",
                study.divergences
            );
        }
    }
}

/// The 100-panel study by hand: `cargo test --release -p prospicio-validation
/// --features slow-bayes --test reserving_bayes_hierarchical_bf recovery_full
/// -- --ignored --nocapture`.
#[test]
#[ignore]
fn recovery_full() {
    for programs in [5, 20] {
        let study = recovery_study(programs, 100);
        assert_recovery(&study, programs);
    }
}

/// One group of `HBF_PROGRAMS` programs and `HBF_PANELS` panels, to time and
/// to explore (`--ignored --nocapture`).
#[test]
#[ignore]
fn recovery_probe() {
    let var = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    recovery_study(var("HBF_PROGRAMS", 5), var("HBF_PANELS", 10));
}

// ---------------------------------------------------------------------
// Item 6: the thin-program property
// ---------------------------------------------------------------------

/// Three ages, so that the programs of two or three origins and the
/// separate fits reserve to the same last age.
const THIN_BETA: [f64; 2] = [-0.2513, -0.8109];
/// Origins of the five programs: the first three have a degree of freedom of
/// their own (six cells, five chain-ladder parameters), the last two have
/// none.
const THIN_ORIGINS: [usize; 5] = [3, 3, 3, 2, 2];

struct Thin {
    /// Per program of three origins: the hierarchical and the separate fit's
    /// reserve sd, and their errors against the true expected reserve, all
    /// relative to the program's total premium.
    hierarchical_sd: Vec<f64>,
    separate_sd: Vec<f64>,
    /// The same sds per root scale: the reserve's sd is proportional to the
    /// root of the plugged-in scale (process variance `phi mu`, parameter
    /// variance proportional to `phi`).
    hierarchical_sd_per_scale: Vec<f64>,
    separate_sd_per_scale: Vec<f64>,
    hierarchical_error: Vec<f64>,
    separate_error: Vec<f64>,
}

fn thin_panel(index: usize, separate: &BayesBfNormalised) -> Thin {
    let seed = 50_000 + index as u64;
    let index = index % 100_000;
    let mut rng = Rng::new(seed);
    let truth = Truth {
        level: 0.7,
        tau: 0.25,
        omega: 0.3,
        cv: 0.1,
        kappa: 0.01,
        beta: THIN_BETA.to_vec(),
    };
    let layout = Layout {
        origins: THIN_ORIGINS.to_vec(),
        ages: 3,
        square: false,
        base: (0..5).map(|_| 200.0 * (0.3 * rng.normal()).exp()).collect(),
        growth: 0.05,
    };
    let panel = simulate(&mut rng, &truth, &layout);
    let hierarchical = HierarchicalBf {
        cv: truth.cv,
        n_sims: 2_000,
        seed,
        ..HierarchicalBf::default()
    }
    .fit(&whole(&panel), "paid", "premium")
    .expect("fits");
    let mut out = Thin {
        hierarchical_sd: vec![],
        separate_sd: vec![],
        hierarchical_sd_per_scale: vec![],
        separate_sd_per_scale: vec![],
        hierarchical_error: vec![],
        separate_error: vec![],
    };
    for (c, program) in panel.programs.iter().enumerate() {
        let alone = triangle(&[program]);
        let premium: f64 = program.premium.iter().sum();
        if THIN_ORIGINS[c] < 3 {
            // No degree of freedom: no scale to plug in.
            assert!(
                separate.fit(&alone, "paid", "premium").is_err(),
                "a program of two origins is fitted alone"
            );
            continue;
        }
        let alone = BayesBfNormalised { seed, ..*separate }
            .fit(&alone, "paid", "premium")
            .expect("fits");
        let truth_reserve: f64 = program.expected_reserve.iter().sum();
        let h = hierarchical.programs[c].reserves.total();
        let s = alone.reserves.total();
        eprintln!(
            "  thin {index}/{c}: scale {:.3} (alone) vs {:.3} (pooled); sd {:.4} vs {:.4}; mean              {:.4} vs {:.4}, truth {:.4}",
            alone.scale,
            hierarchical.programs[c].scale,
            s.std_dev() / premium,
            h.std_dev() / premium,
            s.mean() / premium,
            h.mean() / premium,
            truth_reserve / premium
        );
        out.hierarchical_sd_per_scale
            .push(h.std_dev() / premium / hierarchical.programs[c].scale.sqrt());
        out.separate_sd_per_scale
            .push(s.std_dev() / premium / alone.scale.sqrt());
        out.hierarchical_sd.push(h.std_dev() / premium);
        out.separate_sd.push(s.std_dev() / premium);
        out.hierarchical_error
            .push((h.mean() - truth_reserve).abs() / premium);
        out.separate_error
            .push((s.mean() - truth_reserve).abs() / premium);
    }
    out
}

/// `n` panels of five programs of two or three origins on three ages.
///
/// A program of three origins fitted alone has one degree of freedom, so its
/// plugged-in scale is a chi-square with one degree of freedom times the
/// truth: over the panels it ranged from 0.035 to 17.8 where the pooled scale
/// ranged within a factor of 20, and the separate fit's reserve sd followed
/// it. The sds are therefore compared both raw (reported: the hierarchical
/// fit's is below in about half of the programs) and per root scale, which is
/// what the data's information fixes (asserted: below in at least 80% of the
/// programs). The
/// mean is asserted to be closer to the true expected reserve on average.
///
/// Measured over four groups of 40 panels (offsets 0, 40, 80 and 120; the
/// defaults with target accept 0.99): per root scale the hierarchical sd was
/// below in 111 to 116 of 120 programs (asserted: 80%), the sd ratio averaged
/// 0.773 to 0.795 (asserted: below 0.9; largest ratio 1.08 to 1.22), the mean
/// was closer to the truth in 80 to 84 programs, and the mean absolute error
/// was 0.0229 to 0.0269 of the premium against 0.0327 to 0.0383 for the
/// separate fits (ratio 0.66 to 0.75; asserted: below 0.85).
fn thin_programs(n: usize, offset: usize) {
    let separate = BayesBfNormalised {
        apriori: 1.0,
        cv: f64::INFINITY,
        n_sims: 2_000,
        ..BayesBfNormalised::default()
    };
    let start = Instant::now();
    let panels: Vec<Thin> = (0..n)
        .into_par_iter()
        .map(|i| thin_panel(offset + i, &separate))
        .collect();
    let all = |f: fn(&Thin) -> &Vec<f64>| -> Vec<f64> {
        panels.iter().flat_map(|p| f(p).iter().copied()).collect()
    };
    let (h_sd, s_sd) = (all(|p| &p.hierarchical_sd), all(|p| &p.separate_sd));
    let (h_rs, s_rs) = (
        all(|p| &p.hierarchical_sd_per_scale),
        all(|p| &p.separate_sd_per_scale),
    );
    let (h_err, s_err) = (all(|p| &p.hierarchical_error), all(|p| &p.separate_error));
    let count = |a: &[f64], b: &[f64]| a.iter().zip(b).filter(|(x, y)| x < y).count();
    let ratios: Vec<f64> = h_rs.iter().zip(&s_rs).map(|(h, s)| h / s).collect();
    eprintln!(
        "thin programs, {n} panels ({} programs of three origins): hierarchical sd below the          separate fit's in {} (raw) and {} (per root scale; ratio {:.3} on average, largest          {:.3}); mean closer to the truth in {}; mean absolute error {:.4} against {:.4} of the          premium; wall {:.1?}",
        h_sd.len(),
        count(&h_sd, &s_sd),
        count(&h_rs, &s_rs),
        mean(&ratios),
        ratios.iter().copied().fold(0.0, f64::max),
        count(&h_err, &s_err),
        mean(&h_err),
        mean(&s_err),
        start.elapsed()
    );
    assert!(mean(&ratios) < 0.9, "sd ratios {ratios:?}");
    assert!(
        count(&h_rs, &s_rs) as f64 >= 0.8 * h_rs.len() as f64,
        "sd ratios {ratios:?}"
    );
    assert!(
        mean(&h_err) < 0.85 * mean(&s_err),
        "error {} against {}",
        mean(&h_err),
        mean(&s_err)
    );
}

#[test]
fn thin_programs_borrow_strength() {
    thin_programs(if full_bayes() { 40 } else { 10 }, 0);
}

#[test]
#[ignore]
fn thin_programs_probe() {
    let n = std::env::var("HBF_PANELS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    let offset = std::env::var("HBF_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    thin_programs(n, offset);
}

// ---------------------------------------------------------------------
// Item 4: Stan parity
// ---------------------------------------------------------------------

/// The settings of the reference case (`scripts/reserving_hierarchical_bf_stan.py`).
fn stan_model(seed: u64) -> HierarchicalBf {
    HierarchicalBf {
        apriori: 0.7,
        level_cv: Some(0.5),
        cv: 0.1,
        development_sd: 10.0,
        tau_scale: 0.5,
        omega_scale: 0.5,
        seed,
        ..HierarchicalBf::default()
    }
}

/// RAA with a synthetic premium of 40,000 in 1981 rising by 2,000 a year.
fn raa_with_premium() -> Triangle {
    let raa = dataset("raa");
    let [_, _, n_origins, n_ages] = raa.shape();
    let (mut origin, mut age, mut paid, mut premium) = (vec![], vec![], vec![], vec![]);
    for w in 0..n_origins {
        for d in 0..n_ages {
            if let Some(v) = raa.get(0, 0, w, d) {
                origin.push(Month::january(1981 + w as i32));
                age.push(raa.development()[d]);
                paid.push(v);
                premium.push(40_000.0 + 2_000.0 * w as f64);
            }
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("paid", &paid), ("premium", &premium)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .expect("a triangle")
}

/// The three programs of the reference case, in the segment order of their
/// labels: EVW's triangle with its prior ultimates as premium cut to its
/// first six ages (all ten origins), GenIns with premium, and RAA with a
/// premium of 40,000 in 1981 rising by 2,000 a year.
/// Validation item 1's fits on RAA with a synthetic premium: one program, a
/// tight hyperprior and spreads near 0 give `BayesBfNormalised`'s reserves,
/// in total and by origin, within `0.1 s` and 10% (GenIns is a unit test).
#[test]
fn a_tight_hyperprior_reduces_to_the_normalised_fit_on_raa() {
    let tri = raa_with_premium();
    let hierarchical = HierarchicalBf {
        apriori: 0.5,
        level_cv: Some(1e-6),
        cv: 0.2,
        tau_scale: 1e-6,
        omega_scale: 1e-6,
        n_sims: 8_000,
        seed: 21,
        ..HierarchicalBf::default()
    }
    .fit(&tri, "paid", "premium")
    .expect("a fit");
    let normalised = BayesBfNormalised {
        apriori: 0.5,
        cv: 0.2,
        n_sims: 8_000,
        seed: 22,
        ..BayesBfNormalised::default()
    }
    .fit(&tri, "paid", "premium")
    .expect("a fit");
    let (a, b) = (&hierarchical.reserves, &normalised.reserves);
    let total_sd = b.std_dev();
    assert!(
        (a.mean() - b.mean()).abs() < 0.1 * total_sd,
        "total means {} {}",
        a.mean(),
        b.mean()
    );
    assert!(
        (a.std_dev() / total_sd - 1.0).abs() < 0.1,
        "total sds {} {total_sd}",
        a.std_dev()
    );
    let (wa, wb) = (a.n_components(), b.n_components());
    assert_eq!(wa, wb);
    let (da, db) = (a.draw_matrix(), b.draw_matrix());
    for w in 0..wa {
        let x: Vec<f64> = da.chunks_exact(wa).map(|row| row[w]).collect();
        let y: Vec<f64> = db.chunks_exact(wb).map(|row| row[w]).collect();
        if sd(&y) > 0.0 {
            assert!(
                (mean(&x) - mean(&y)).abs() < 0.1 * sd(&y),
                "component {w}: {} {}",
                mean(&x),
                mean(&y)
            );
            assert!(
                (sd(&x) / sd(&y) - 1.0).abs() < 0.1,
                "component {w}: {} {}",
                sd(&x),
                sd(&y)
            );
        }
    }
}

/// Validation item 5's limit: the one-year view of one program with a tight
/// hyperprior and spreads near 0 is `BayesBfNormalised`'s, both reweighting
/// only so that refit noise stays out: the opening reserves and the
/// claims development result's total and origins' means within `0.1 s` and
/// its sds within 10%.
#[test]
fn a_tight_hyperprior_reduces_the_one_year_view_to_the_normalised_one() {
    use prospicio_reserving::bayes::{BayesOneYear, OneYearRefit};
    let tri = raa_with_premium();
    let settings = BayesOneYear {
        refit: Some(OneYearRefit::ReweightOnly),
        n_sims: Some(4_000),
        ..BayesOneYear::default()
    };
    let hierarchical = HierarchicalBf {
        apriori: 0.5,
        level_cv: Some(1e-6),
        cv: 0.2,
        tau_scale: 1e-6,
        omega_scale: 1e-6,
        n_sims: 8_000,
        seed: 31,
        ..HierarchicalBf::default()
    }
    .one_year(&tri, "paid", "premium", &settings)
    .expect("a view");
    let normalised = BayesBfNormalised {
        apriori: 0.5,
        cv: 0.2,
        n_sims: 8_000,
        seed: 32,
        ..BayesBfNormalised::default()
    }
    .one_year(&tri, "paid", "premium", &settings)
    .expect("a view");
    let (a, b) = (&hierarchical.cdr, &normalised.cdr);
    let total_sd = b.std_dev();
    assert!(
        (a.mean() - b.mean()).abs() < 0.1 * total_sd,
        "total CDR means {} {}",
        a.mean(),
        b.mean()
    );
    assert!(
        (a.std_dev() / total_sd - 1.0).abs() < 0.1,
        "total CDR sds {} {total_sd}",
        a.std_dev()
    );
    let total = |v: &[f64]| v.iter().sum::<f64>();
    assert!(
        (total(&hierarchical.opening_reserve) - total(&normalised.opening_reserve)).abs()
            < 0.1 * total_sd,
        "opening reserves {} {}",
        total(&hierarchical.opening_reserve),
        total(&normalised.opening_reserve)
    );
    let (wa, wb) = (a.n_components(), b.n_components());
    assert_eq!(wa, wb);
    let (da, db) = (a.draw_matrix(), b.draw_matrix());
    for w in 0..wa {
        let x: Vec<f64> = da.chunks_exact(wa).map(|row| row[w]).collect();
        let y: Vec<f64> = db.chunks_exact(wb).map(|row| row[w]).collect();
        if sd(&y) > 0.0 {
            assert!(
                (mean(&x) - mean(&y)).abs() < 0.1 * sd(&y),
                "origin {w}: CDR means {} {}",
                mean(&x),
                mean(&y)
            );
            assert!(
                (sd(&x) / sd(&y) - 1.0).abs() < 0.1,
                "origin {w}: CDR sds {} {}",
                sd(&x),
                sd(&y)
            );
        }
    }
}

fn stan_fit(seed: u64) -> HierarchicalBfFit {
    let start = Instant::now();
    let fit = stan_model(seed)
        .fit(&stan_triangle(), "paid", "premium")
        .expect("fits");
    eprintln!(
        "HierarchicalBf Stan case seed {seed}: fit in {:.2?}; {:?}",
        start.elapsed(),
        fit.diagnostics
    );
    fit
}

/// Draws of the shares `p_cd` of program `c` (from 0) at age `d` (from 0).
fn share_draws(fit: &HierarchicalBfFit, c: usize, d: usize) -> Vec<f64> {
    let m = fit.ages.len();
    let omega: Vec<f64> = draws(fit, "log_omega").iter().map(|l| l.exp()).collect();
    let beta: Vec<Vec<f64>> = (2..=m).map(|k| draws(fit, &format!("beta[{k}]"))).collect();
    let delta: Vec<Vec<f64>> = (2..=m)
        .map(|k| draws(fit, &format!("delta[{},{k}]", c + 1)))
        .collect();
    (0..omega.len())
        .map(|i| {
            let mut b = vec![0.0; m];
            for k in 1..m {
                b[k] = beta[k - 1][i] + omega[i] * delta[k - 1][i];
            }
            let norm: f64 = b.iter().map(|v| v.exp()).sum();
            b[d].exp() / norm
        })
        .collect()
}

fn moment(v: &[f64], quantity: &str) -> f64 {
    if quantity.ends_with("mean") {
        mean(v)
    } else {
        sd(v)
    }
}

/// Our value for a reference row.
fn stan_value(fit: &HierarchicalBfFit, q: &str, arg: &str) -> Option<f64> {
    let program = |label: &str| fit.programs.iter().position(|p| p.label.parts() == [label]);
    let reserve = |arg: &str| -> prospicio_prob::Sampled {
        if arg == "total" {
            return fit.reserves.total().clone();
        }
        let (label, rest) = arg.split_once(':').expect("program:origin");
        let program = &fit.programs[program(label).expect("a program")];
        if rest == "total" {
            program.reserves.total().clone()
        } else {
            let origin = Period::year(rest.parse().expect("origin year"));
            program
                .reserves
                .marginal(&vec![origin.into()])
                .expect("origin")
        }
    };
    Some(match q {
        "kappa" => fit.scale_per_premium,
        "scale" => fit.programs[program(arg)?].scale,
        "hyper_mean" | "hyper_sd" => {
            let v: Vec<f64> = match arg {
                "level" => draws(fit, "log_level").iter().map(|l| l.exp()).collect(),
                "tau" => draws(fit, "log_tau").iter().map(|l| l.exp()).collect(),
                "omega" => draws(fit, "log_omega").iter().map(|l| l.exp()).collect(),
                _ => return None,
            };
            moment(&v, q)
        }
        "ratio_mean" | "ratio_sd" => {
            let v: Vec<f64> = log_ratio(fit, program(arg)?)
                .iter()
                .map(|l| l.exp())
                .collect();
            moment(&v, q)
        }
        "share_mean" | "share_sd" => {
            let (label, d) = arg.split_once(':')?;
            moment(
                &share_draws(fit, program(label)?, d.parse::<usize>().ok()? - 1),
                q,
            )
        }
        "reserve_mean" => reserve(arg).mean(),
        "reserve_sd" => reserve(arg).std_dev(),
        "reserve_q05" => reserve(arg).quantile(0.05).ok()?,
        "reserve_q50" => reserve(arg).quantile(0.5).ok()?,
        "reserve_q95" => reserve(arg).quantile(0.95).ok()?,
        _ => return None,
    })
}

/// The same model in Stan (CmdStan 2.39.0, `reference/reserving_hierarchical_bf_stan.csv`):
/// the plugged-in scales (rel 1e-9, so the formula `phi_c = kappa Pbar_c` is
/// the runner's), the hyperparameters, each program's loss ratio and shares
/// (the shares of EVW's ages 7 to 10, which it never observes, among them)
/// and the reserves by origin and program and in total. The reference pools 8
/// Stan runs of 4 chains of 10,000 draws (seeds 20261105 to 20261112, no
/// divergence but one in 320,000 draws, R-hat at most 1.0008 in each), so
/// that its own Monte Carlo error is not spent from the tolerances. Our side
/// runs the defaults (4 chains of 1,000 draws, 10,000 simulations, seed 0).
///
/// With `s` Stan's posterior sd the tolerances are means within `0.12 s`,
/// sds within 14% (30% for EVW's loss ratio and reserves and the totals,
/// whose tail is heavy), the 5% and 50% percentiles of the reserves within
/// `0.10 s`, the 95% within `0.26 s` (`0.9 s` for EVW's reserves and the
/// totals). Over seeds 0 to 23 (`seed_sweep`) the largest errors were
/// `0.080 s`, 9% (19.6%), `0.062 s`, `0.172 s` and `0.621 s`, so each
/// tolerance is at least 1.45 times its family's worst (measured). Seed 0
/// passes every row, at most `0.048 s`, 3.1% (9.3%), `0.085 s` (`0.135 s`)
/// of them.
///
/// A row-by-row tolerance cannot see a bias of 10% in the sds (the worst
/// sd error of a single row is 9%), so the mean signed relative error of the
/// sds is asserted as well: over the sds that are not heavy-tailed (the
/// hyperparameters, shares, GenIns' and RAA's loss ratios and reserves) it
/// was within `0.009` of zero on every seed and is bounded by `0.03`, which
/// catches a 10% (and a 5%) bias; over the heavy-tailed sds (EVW's loss
/// ratio and reserves and the totals) it averaged -0.037 (standard
/// deviation 0.091 from seed to seed, from -0.160 to 0.187) and is bounded
/// by `0.28`, which catches only a bias of about 30%.
///
/// The sampler's defaults now pass the thresholds on this case: over seeds 0
/// to 23 no seed had a divergent transition, the largest R-hat was 1.0104 and
/// the smallest bulk and tail ESS were 673 and 602. One seed in 24 had a
/// warning (seed 0, the test's own: R-hat 1.0104 on `alpha[1,3]`, just above
/// the threshold of 1.01), so the test asserts no divergence, R-hat below
/// 1.02 and ESS above 400 (the warning's own limit, 100 a chain), not an
/// empty list of warnings.
#[test]
fn stan_parity() {
    let fit = stan_fit(0);
    let d = &fit.diagnostics;
    assert_eq!(d.divergences, 0, "{d:?}");
    assert!(d.max_rhat < 1.02, "{d:?}");
    assert!(d.min_ess_bulk > 400.0 && d.min_ess_tail > 400.0, "{d:?}");
    let cases = reference("reserving_hierarchical_bf_stan.csv");
    check(&cases, |case| {
        stan_value(&fit, case.get("quantity"), case.get("arg"))
    });
    let (light, heavy) = sd_bias(&fit, &cases);
    eprintln!(
        "mean signed relative sd error: {light:.4} (not heavy-tailed), {heavy:.4} (heavy-tailed)"
    );
    assert!(light.abs() < 0.03, "sd bias {light}");
    assert!(
        heavy.abs() < 0.28,
        "sd bias of the heavy-tailed sds {heavy}"
    );
}

/// The mean over the reference's sd rows of `got / expected - 1`, for the rows
/// that are not heavy-tailed and for those that are (EVW's loss ratio and
/// reserves and the totals).
fn sd_bias(fit: &HierarchicalBfFit, cases: &[prospicio_validation::Case]) -> (f64, f64) {
    let (mut light, mut heavy) = (vec![], vec![]);
    for case in cases {
        let (q, arg) = (case.get("quantity"), case.get("arg"));
        if !q.ends_with("_sd") {
            continue;
        }
        let expected = case.number("expected").expect("expected");
        if expected == 0.0 {
            continue;
        }
        let got = stan_value(fit, q, arg).expect("a row");
        let error = got / expected - 1.0;
        let is_heavy = (q == "ratio_sd" && arg == "evw")
            || (q == "reserve_sd" && (arg.starts_with("evw") || arg == "total"));
        if is_heavy { &mut heavy } else { &mut light }.push(error);
    }
    (mean(&light), mean(&heavy))
}

/// The seed variation behind the parity tolerances: every reference row for
/// seeds `0..HBF_SEEDS` (default 24), written to `target/hbf_sweep.csv`
/// (`seed,quantity,arg,got,expected,abs_tol,rel_tol`). Run in release:
/// `cargo test --release -p prospicio-validation --features slow-bayes --test
/// reserving_bayes_hierarchical_bf seed_sweep -- --ignored --nocapture`.
#[test]
#[ignore]
fn seed_sweep() {
    let seeds: u64 = std::env::var("HBF_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let cases = reference("reserving_hierarchical_bf_stan.csv");
    let mut diagnostics = String::from(
        "seed,divergences,max_rhat,min_ess_bulk,min_ess_tail,warnings
",
    );
    let mut out = String::from(
        "seed,quantity,arg,got,expected,abs_tol,rel_tol,warnings
",
    );
    for seed in 0..seeds {
        let fit = stan_fit(seed);
        let d = &fit.diagnostics;
        diagnostics.push_str(&format!(
            "{seed},{},{},{},{},{}
",
            d.divergences,
            d.max_rhat,
            d.min_ess_bulk,
            d.min_ess_tail,
            d.warnings.len()
        ));
        for case in &cases {
            let (q, arg) = (case.get("quantity"), case.get("arg"));
            let got = stan_value(&fit, q, arg).expect("a row");
            out.push_str(&format!(
                "{seed},{q},{arg},{got:e},{:e},{:e},{:e},{}
",
                case.number("expected").expect("expected"),
                case.number("abs_tol").unwrap_or(0.0),
                case.number("rel_tol").unwrap_or(0.0),
                fit.diagnostics.warnings.len()
            ));
        }
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/hbf_sweep.csv");
    std::fs::write(&path, out).expect("writes the sweep");
    std::fs::write(
        path.with_file_name("hbf_sweep_diagnostics.csv"),
        diagnostics,
    )
    .expect("writes the diagnostics");
    eprintln!("wrote {}", path.display());
}

// ---------------------------------------------------------------------
// The sampler's target acceptance (issue #762, item 4)
// ---------------------------------------------------------------------

/// A sweep of the sampler's `target_accept` (default 0.99) over panels, for
/// the user's decision on #762 (measured, not asserted). For each value of
/// `HBF_TARGET_ACCEPTS` (comma-separated, default `0.9,0.95,0.99`), each
/// panel and each seed of `0..HBF_SWEEP_SEEDS` (default 4), one fit at the
/// default sampler otherwise: its divergences, R-hat, whether it warned and
/// its wall time; on the reference panel also the gap of the three
/// programs' loss-ratio sds to the Stan reference (`ratio_sd` rows of
/// `reserving_hierarchical_bf_stan.csv`). The panels are the Stan reference
/// (EVW cut to six ages, GenIns, RAA), GenIns and RAA alone (programs on
/// the same ages), all three cut to six ages, and recovery panels of item 3
/// with `HBF_SWEEP_PROGRAMS` programs (comma-separated, default `5,10,20`;
/// diagonal scale below 8 programs, dense from 8). Prints one
/// row per value and panel (means over seeds, worst R-hat) and writes every
/// fit to `target/hbf_target_accept.csv`. Run in release:
/// `cargo test --release -p prospicio-validation --features slow-bayes --test
/// reserving_bayes_hierarchical_bf target_accept_sweep -- --ignored --nocapture`.
#[test]
#[ignore]
fn target_accept_sweep() {
    let accepts: Vec<f64> = std::env::var("HBF_TARGET_ACCEPTS")
        .unwrap_or_else(|_| "0.9,0.95,0.99".into())
        .split(',')
        .map(|s| s.trim().parse().expect("a target acceptance"))
        .collect();
    let seeds: u64 = std::env::var("HBF_SWEEP_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let cases = reference("reserving_hierarchical_bf_stan.csv");
    let ratio_sd: Vec<(String, f64)> = cases
        .iter()
        .filter(|c| c.get("quantity") == "ratio_sd")
        .map(|c| {
            (
                c.get("arg").to_string(),
                c.number("expected").expect("expected"),
            )
        })
        .collect();
    // (name, triangle, model at seed, compared with Stan)
    type Panel = (
        &'static str,
        Triangle,
        Box<dyn Fn(u64) -> HierarchicalBf>,
        bool,
    );
    let recovery = |programs: usize| -> (Triangle, HierarchicalBf) {
        let (_, tri, model) = recovery_setup(programs, 0);
        (tri, model)
    };
    let programs: Vec<usize> = std::env::var("HBF_SWEEP_PROGRAMS")
        .unwrap_or_else(|_| "5,10,20".into())
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().expect("a number of programs"))
        .collect();
    let mut panels: Vec<Panel> = vec![
        ("reference", stan_triangle(), Box::new(stan_model), true),
        (
            "genins+raa",
            prospicio_validation::stan_triangle_of(&["genins", "raa"]),
            Box::new(stan_model),
            false,
        ),
        (
            "all cut to 6",
            prospicio_validation::stan_triangle_cut(&["evw", "genins", "raa"], 6),
            Box::new(stan_model),
            false,
        ),
    ];
    for &n in &programs {
        let (tri, m) = recovery(n);
        let name: &'static str = Box::leak(format!("recovery {n}").into_boxed_str());
        panels.push((
            name,
            tri,
            Box::new(move |seed| HierarchicalBf { seed, ..m.clone() }),
            false,
        ));
    }
    let mut csv = String::from(
        "target_accept,panel,seed,divergences,max_rhat,min_ess_bulk,min_ess_tail,warned,seconds,ratio_sd_gaps\n",
    );
    println!(
        "| target_accept | panel | divergences (mean) | max R-hat (worst) | warned | fit s (mean) | loss-ratio sd / Stan - 1 (evw, genins, raa; mean) |"
    );
    println!("|---|---|---|---|---|---|---|");
    for &accept in &accepts {
        for (name, tri, model, stan) in &panels {
            let (mut div, mut rhat, mut warned, mut secs) = (0.0, 0.0_f64, 0, 0.0);
            let mut gaps = vec![0.0; ratio_sd.len()];
            for seed in 0..seeds {
                let mut m = model(seed);
                m.sampler.target_accept = accept;
                m.n_sims = 100;
                let start = Instant::now();
                let fit = m.fit(tri, "paid", "premium").expect("fits");
                let s = start.elapsed().as_secs_f64();
                let d = &fit.diagnostics;
                let row_gaps: Vec<f64> = if *stan {
                    ratio_sd
                        .iter()
                        .map(|(arg, expected)| {
                            stan_value(&fit, "ratio_sd", arg).expect("a row") / expected - 1.0
                        })
                        .collect()
                } else {
                    vec![]
                };
                for (g, r) in gaps.iter_mut().zip(&row_gaps) {
                    *g += r / seeds as f64;
                }
                div += d.divergences as f64 / seeds as f64;
                rhat = rhat.max(d.max_rhat);
                warned += usize::from(!d.warnings.is_empty());
                secs += s / seeds as f64;
                csv.push_str(&format!(
                    "{accept},{name},{seed},{},{},{},{},{},{s:.2},{}\n",
                    d.divergences,
                    d.max_rhat,
                    d.min_ess_bulk,
                    d.min_ess_tail,
                    !d.warnings.is_empty(),
                    row_gaps
                        .iter()
                        .map(|g| format!("{g:.4}"))
                        .collect::<Vec<_>>()
                        .join(";")
                ));
            }
            let gap_text = if *stan {
                gaps.iter()
                    .map(|g| format!("{g:+.3}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            } else {
                "n/a".into()
            };
            println!(
                "| {accept} | {name} | {div:.2} | {rhat:.4} | {warned} of {seeds} | {secs:.1} | {gap_text} |"
            );
        }
    }
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/hbf_target_accept.csv");
    std::fs::write(&path, csv).expect("writes the sweep");
    eprintln!("wrote {}", path.display());
}
