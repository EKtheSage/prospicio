//! The simulation study of excess-layer development
//! (`docs/design/reserving-excess.md`, PR 2, decisions 11 and 16).
//!
//! Each simulated book has ten accident years observed at ten annual ages.
//! Claims arrive by a Poisson count, are reported with a lag of up to four
//! years and develop in size claim by claim to ultimate at the oldest age,
//! so the ground-up triangle, its claim counts, the layer triangles and the
//! true layer ultimates all come from the same claims. The upper triangle
//! is fitted and the total layer ultimate is compared with the truth.
//!
//! The arms (the first four share their random numbers):
//!
//! * `scale`: a lognormal ultimate severity and a pure scale shift, every
//!   claim at age `j` the ultimate one over `c_j`: the model's own
//!   assumptions.
//! * `capped`: the same claims, the ground-up column limited at 250,000 per
//!   claim, so the shifts come from the numerical solve.
//! * `shape`: the same mean development, but the log-sd of the claims at an
//!   early age is smaller than at ultimate (shape drift), each claim
//!   developing by a quantile map.
//! * `late`: the scale shift, but the claims reported after the first age
//!   are half the size of the early ones.
//! * `pareto`: claims above 100,000 from a Pareto whose shape falls from 3.0
//!   to 2.2 (Pinto and Gogol's drift, each claim by a quantile map), fitted
//!   both by `ParetoShape` with the true shapes on the losses excess of
//!   100,000 and by the scale shift on the ground-up column.
//!
//! Each arm has a dense layer, a thin one (a few claims an accident year,
//! where the own chain ladder still fits) and a sparse one, whose early
//! cells are mostly zero. The methods are the layer's own chain ladder, the derived
//! pattern fully selected, and its credibility blend with the layer's own
//! factors (`ExcessPattern::credibility_weights` on the layer claim counts)
//! over a grid of `k`, with the cap at 10% and without it. Errors are of
//! the total layer ultimate over the ten accident years. Where the layer's
//! own chain ladder cannot be fitted (an age at which every earlier value is
//! zero) the simulation is counted and left out of the ratios, which are
//! taken on the simulations where every method has an estimate; that
//! favours the own chain ladder.
//!
//! The acceptance thresholds of decision 11 are asserted where the model's
//! assumptions hold (`scale`, `capped`, and `pareto` fitted by
//! `ParetoShape`): on the sparse layer the derived pattern's RMSE, and the
//! capped blend's at every `k` of the grid, below 0.8 times the own chain
//! ladder's (and the derived pattern's on the thin layer too); on the dense
//! layer the derived pattern's mean error within two Monte Carlo standard
//! errors of zero. Under shape drift and late claims of another size the
//! scale shift is biased by construction; those arms are reported, and the
//! direction of each bias is pinned, not tuned away. The results are in
//! `knowledge/findings/excess-simulation-study.md`.
//!
//! The full study is 500 simulations of every arm, reproducible by `SEED`
//! (each simulation its own stream, so the results do not depend on the
//! threads). It takes under a second in a debug build, so it runs in every
//! `cargo test`, with no smoke version. Print the tables with
//!
//! ```text
//! cargo test --release -p prospicio-validation --test reserving_excess_simulation -- --nocapture
//! ```

use prospicio_core::{Lag, StreamRng};
use prospicio_prob::{Lognormal, Pareto};
use prospicio_reserving::excess::{Basis, ExcessPattern, Layer, SeverityByMaturity, ShapeBase};
use prospicio_reserving::{
    ChainLadder, Development, DevelopmentColumn, Grain, Long, Month, PatternWeight, Tail, Triangle,
};
use rayon::prelude::*;

const SEED: u64 = 192;
const N_SIMS: usize = 500;
const N_ORIGINS: usize = 10;
const N_AGES: usize = 10;
/// Probability that a claim is reported at the first, second, ... age.
const REPORT: [f64; 5] = [0.55, 0.25, 0.12, 0.05, 0.03];

/// The lognormal arms: claims per accident year, the ultimate severity's
/// mean and log-sd, the severity development `c_j` to ultimate and, for
/// shape drift, the log-sd at each age as a share of the ultimate one.
const LAMBDA: f64 = 400.0;
const MEAN: f64 = 30_000.0;
const SIGMA: f64 = 1.8;
const SHIFT: [f64; N_AGES] = [2.0, 1.6, 1.35, 1.2, 1.12, 1.07, 1.04, 1.02, 1.01, 1.0];
const SD_SHARE: [f64; N_AGES] = [0.8, 0.85, 0.89, 0.93, 0.95, 0.97, 0.985, 0.995, 1.0, 1.0];
/// The ground-up cap of the `capped` arm and the size of late claims
/// relative to early ones in the `late` arm.
const CAP: f64 = 250_000.0;
const LATE_SIZE: f64 = 0.5;

/// The `pareto` arm: claims above `THRESHOLD` per accident year and the
/// Pareto shape at each age (the last is ultimate).
const LAMBDA_PARETO: f64 = 150.0;
const THRESHOLD: f64 = 100_000.0;
const ALPHA: [f64; N_AGES] = [3.0, 2.8, 2.6, 2.45, 2.35, 2.28, 2.24, 2.22, 2.2, 2.2];

/// The credibility constants and the cap on one origin's influence.
const K_GRID: [f64; 8] = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0];
const MAX_INFLUENCE: f64 = 0.1;
const NO_CAP: f64 = 1e12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    Scale,
    Capped,
    Shape,
    Late,
    Pareto,
}

impl Arm {
    const ALL: [Arm; 5] = [Arm::Scale, Arm::Capped, Arm::Shape, Arm::Late, Arm::Pareto];

    fn name(self) -> &'static str {
        match self {
            Arm::Scale => "scale",
            Arm::Capped => "capped",
            Arm::Shape => "shape",
            Arm::Late => "late",
            Arm::Pareto => "pareto",
        }
    }

    /// The dense, the thin and the sparse layer.
    fn layers(self) -> [(&'static str, Layer); 3] {
        let layer = |a: f64, l: f64| Layer::new(a, l).unwrap();
        match self {
            Arm::Pareto => [
                ("dense", layer(150_000.0, 150_000.0)),
                ("thin", layer(400_000.0, 400_000.0)),
                ("sparse", layer(1_000_000.0, 1_000_000.0)),
            ],
            _ => [
                ("dense", layer(50_000.0, 50_000.0)),
                ("thin", layer(250_000.0, 250_000.0)),
                ("sparse", layer(1_000_000.0, 1_000_000.0)),
            ],
        }
    }

    /// Whether the derived pattern's model holds, so the thresholds apply.
    fn matched(self) -> bool {
        matches!(self, Arm::Scale | Arm::Capped | Arm::Pareto)
    }
}

/// One claim: its accident year, the age position it is reported at and
/// its size at every age (meaningful from the report on).
struct Claim {
    origin: usize,
    report: usize,
    size: [f64; N_AGES],
}

fn ages() -> Vec<Lag> {
    (1..=N_AGES as Lag).map(|d| 12 * d).collect()
}

fn normal(rng: &mut StreamRng) -> f64 {
    let (u, v) = (rng.next_open01(), rng.next_open01());
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

fn poisson(mean: f64, rng: &mut StreamRng) -> usize {
    let u = rng.next_open01();
    let (mut k, mut p) = (0usize, (-mean).exp());
    let mut cdf = p;
    while u > cdf && p > 0.0 {
        k += 1;
        p *= mean / k as f64;
        cdf += p;
    }
    k
}

fn report_lag(rng: &mut StreamRng) -> usize {
    let u = rng.next_open01();
    let mut cdf = 0.0;
    for (r, p) in REPORT.iter().enumerate() {
        cdf += p;
        if u <= cdf {
            return r;
        }
    }
    REPORT.len() - 1
}

fn mu() -> f64 {
    MEAN.ln() - SIGMA * SIGMA / 2.0
}

/// The claims of simulation `sim` in `arm`. The lognormal arms draw the
/// same counts, normals and lags for the same `sim`.
fn claims(arm: Arm, sim: usize) -> Vec<Claim> {
    let pareto = arm == Arm::Pareto;
    let stream = if pareto { 1_000_000 + sim } else { sim } as u64;
    let mut rng = StreamRng::new(SEED, stream);
    let mut out = Vec::new();
    for origin in 0..N_ORIGINS {
        let n = poisson(if pareto { LAMBDA_PARETO } else { LAMBDA }, &mut rng);
        for _ in 0..n {
            let report = report_lag(&mut rng);
            let mut size = [0.0; N_AGES];
            if pareto {
                // The same quantile at every age: Pareto(k, alpha_j) at age j.
                let u = rng.next_open01();
                for (j, s) in size.iter_mut().enumerate() {
                    *s = THRESHOLD * u.powf(-1.0 / ALPHA[j]);
                }
            } else {
                let z = normal(&mut rng);
                let late = if arm == Arm::Late && report > 0 {
                    LATE_SIZE.ln()
                } else {
                    0.0
                };
                for (j, s) in size.iter_mut().enumerate() {
                    *s = if arm == Arm::Shape {
                        // log-sd SD_SHARE[j] * SIGMA, mean MEAN / SHIFT[j].
                        let sd = SD_SHARE[j] * SIGMA;
                        let m = mu() + SIGMA * SIGMA / 2.0 - SHIFT[j].ln() - sd * sd / 2.0;
                        (m + sd * z).exp()
                    } else {
                        (mu() + SIGMA * z + late).exp() / SHIFT[j]
                    };
                }
            }
            out.push(Claim {
                origin,
                report,
                size,
            });
        }
    }
    out
}

fn in_layer(x: f64, layer: Layer) -> f64 {
    (x - layer.attachment).clamp(0.0, layer.limit)
}

/// The upper triangle of the columns, each a sum over the claims reported
/// by the age of a function of the claim's size there.
fn triangle(claims: &[Claim], columns: &[(String, Box<dyn Fn(f64) -> f64>)]) -> Triangle {
    let ages = ages();
    let mut origin = Vec::new();
    let mut age = Vec::new();
    let mut values = vec![Vec::new(); columns.len()];
    for o in 0..N_ORIGINS {
        let of_origin: Vec<&Claim> = claims.iter().filter(|cl| cl.origin == o).collect();
        for d in 0..N_AGES - o {
            origin.push(Month::january(2010 + o as i32));
            age.push(ages[d]);
            let mut sums = vec![0.0; columns.len()];
            for cl in of_origin.iter().filter(|cl| cl.report <= d) {
                for (sum, (_, f)) in sums.iter_mut().zip(columns) {
                    *sum += f(cl.size[d]);
                }
            }
            for (v, sum) in values.iter_mut().zip(sums) {
                v.push(sum);
            }
        }
    }
    let named: Vec<(&str, &[f64])> = columns
        .iter()
        .zip(&values)
        .map(|((name, _), v)| (name.as_str(), v.as_slice()))
        .collect();
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &named,
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// The total ultimate of `column` by the chain ladder with `pattern`
/// selected at `weight`, or the layer's own chain ladder without one.
fn chain_ladder(
    tri: &Triangle,
    column: &str,
    pattern: Option<(&ExcessPattern, PatternWeight)>,
) -> Option<f64> {
    let cl = match pattern {
        Some((p, weight)) => ChainLadder {
            development: Development {
                pattern: Some(p.selected(weight)),
                ..Default::default()
            },
            tail: p.tail(),
        },
        None => ChainLadder::default(),
    };
    cl.fit(tri, column)
        .ok()
        .map(|fit| fit.ultimate.iter().sum())
}

/// The methods compared, in table order.
fn method_names() -> Vec<String> {
    let mut names = vec!["own".to_string(), "derived".to_string()];
    for k in K_GRID {
        names.push(format!("blend k={k}"));
    }
    for k in K_GRID {
        names.push(format!("uncapped k={k}"));
    }
    names.push("scale shift".to_string());
    names
}

/// One simulation of `arm`: per layer, the true layer ultimate and the
/// error of every method (`None` where it has no estimate; "scale shift"
/// only in the `pareto` arm).
fn simulate(arm: Arm, sim: usize) -> Vec<(f64, Vec<Option<f64>>)> {
    let claims = claims(arm, sim);
    let layers = arm.layers();
    let mut columns: Vec<(String, Box<dyn Fn(f64) -> f64>)> = vec![
        ("gu".into(), Box::new(|x| x)),
        ("count".into(), Box::new(|_| 1.0)),
        ("gu capped".into(), Box::new(|x: f64| x.min(CAP))),
        ("base".into(), Box::new(|x: f64| (x - THRESHOLD).max(0.0))),
    ];
    for (i, &(_, layer)) in layers.iter().enumerate() {
        columns.push((format!("layer {i}"), Box::new(move |x| in_layer(x, layer))));
        columns.push((
            format!("layer {i} n"),
            Box::new(move |x| f64::from(u8::from(x > layer.attachment))),
        ));
    }
    let tri = triangle(&claims, &columns);
    let lognormal = || Box::new(Lognormal::new(mu(), SIGMA).unwrap());
    let ultimate_pareto = || Box::new(Pareto::new(THRESHOLD, ALPHA[N_AGES - 1]).unwrap());

    layers
        .iter()
        .enumerate()
        .map(|(i, &(_, layer))| {
            let truth: f64 = claims
                .iter()
                .map(|c| in_layer(c.size[N_AGES - 1], layer))
                .sum();
            let (column, counts, model) = match arm {
                Arm::Capped => (
                    "gu capped",
                    Some("count"),
                    SeverityByMaturity::scale_shift_capped(lognormal(), CAP),
                ),
                Arm::Pareto => (
                    "base",
                    None,
                    SeverityByMaturity::pareto_shape(
                        THRESHOLD,
                        ages().into_iter().zip(ALPHA).collect(),
                        ALPHA[N_AGES - 1],
                        ShapeBase::Losses(Layer::unlimited(THRESHOLD).unwrap()),
                    ),
                ),
                _ => (
                    "gu",
                    Some("count"),
                    SeverityByMaturity::scale_shift(lognormal()),
                ),
            };
            let derive = |column: &str, counts: Option<&str>, model: &SeverityByMaturity| {
                ExcessPattern::from_ground_up(
                    &tri,
                    column,
                    Basis::Incurred,
                    counts,
                    &Tail::default(),
                    layer,
                    model,
                )
                .unwrap()
            };
            let p = derive(column, counts, &model);
            let name = format!("layer {i}");
            let n = format!("layer {i} n");
            let err = |est: Option<f64>| est.map(|e| e - truth);
            let blend = |k: f64, cap: f64| {
                let w = p
                    .credibility_weights(&tri, &name, Some(&n), k, cap)
                    .unwrap();
                err(chain_ladder(&tri, &name, Some((&p, w))))
            };
            let mut row = vec![
                err(chain_ladder(&tri, &name, None)),
                err(chain_ladder(
                    &tri,
                    &name,
                    Some((&p, PatternWeight::All(1.0))),
                )),
            ];
            row.extend(K_GRID.iter().map(|&k| blend(k, MAX_INFLUENCE)));
            row.extend(K_GRID.iter().map(|&k| blend(k, NO_CAP)));
            row.push(if arm == Arm::Pareto {
                let scale = SeverityByMaturity::scale_shift(ultimate_pareto());
                let q = derive("gu", Some("count"), &scale);
                err(chain_ladder(
                    &tri,
                    &name,
                    Some((&q, PatternWeight::All(1.0))),
                ))
            } else {
                None
            });
            (truth, row)
        })
        .collect()
}

/// The statistics of one method on one layer.
#[derive(Debug, Clone, Copy)]
struct Stats {
    /// Simulations with an estimate.
    n: usize,
    mean: f64,
    /// Monte Carlo standard error of the mean.
    mcse: f64,
    rmse: f64,
    /// RMSE over the own chain ladder's on the simulations where both have
    /// an estimate.
    ratio: f64,
}

struct LayerResult {
    arm: Arm,
    layer: &'static str,
    mean_truth: f64,
    own_failures: usize,
    stats: Vec<Option<Stats>>,
}

fn study(n_sims: usize) -> Vec<LayerResult> {
    let names = method_names();
    let mut out = Vec::new();
    for arm in Arm::ALL {
        let runs: Vec<Vec<(f64, Vec<Option<f64>>)>> = (0..n_sims)
            .into_par_iter()
            .map(|s| simulate(arm, s))
            .collect();
        for (l, &(layer_name, _)) in arm.layers().iter().enumerate() {
            // The mean of the simulated true ultimates, the scale of the
            // relative errors.
            let mean_truth = runs.iter().map(|r| r[l].0).sum::<f64>() / n_sims as f64;
            let errors =
                |m: usize| -> Vec<Option<f64>> { runs.iter().map(|r| r[l].1[m]).collect() };
            let own = errors(0);
            let own_failures = own.iter().filter(|e| e.is_none()).count();
            let stats = (0..names.len())
                .map(|m| {
                    let e: Vec<f64> = errors(m).into_iter().flatten().collect();
                    if e.is_empty() {
                        return None;
                    }
                    let n = e.len() as f64;
                    let mean = e.iter().sum::<f64>() / n;
                    let var = e.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
                    let rmse = (e.iter().map(|x| x * x).sum::<f64>() / n).sqrt();
                    let both: Vec<(f64, f64)> = errors(m)
                        .into_iter()
                        .zip(&own)
                        .filter_map(|(a, b)| Some((a?, (*b)?)))
                        .collect();
                    let sq =
                        |f: fn(&(f64, f64)) -> f64| both.iter().map(|p| f(p).powi(2)).sum::<f64>();
                    let ratio = (sq(|p| p.0) / sq(|p| p.1)).sqrt();
                    Some(Stats {
                        n: e.len(),
                        mean,
                        mcse: (var / n).sqrt(),
                        rmse,
                        ratio,
                    })
                })
                .collect();
            out.push(LayerResult {
                arm,
                layer: layer_name,
                mean_truth,
                own_failures,
                stats,
            });
        }
    }
    out
}

fn print(results: &[LayerResult]) {
    let names = method_names();
    for r in results {
        println!(
            "\n{} / {} layer: mean true ultimate {:.0}, own chain ladder failed in {} simulations",
            r.arm.name(),
            r.layer,
            r.mean_truth,
            r.own_failures
        );
        println!(
            "{:<16} {:>5} {:>10} {:>8} {:>10} {:>8} {:>8}",
            "method", "n", "mean err", "/truth", "mcse", "rmse", "ratio"
        );
        for (name, s) in names.iter().zip(&r.stats) {
            if let Some(s) = s {
                println!(
                    "{:<16} {:>5} {:>10.0} {:>8.4} {:>10.0} {:>8.4} {:>8.3}",
                    name,
                    s.n,
                    s.mean,
                    s.mean / r.mean_truth,
                    s.mcse,
                    s.rmse / r.mean_truth,
                    s.ratio
                );
            }
        }
    }
}

fn get<'a>(results: &'a [LayerResult], arm: Arm, layer: &str) -> &'a LayerResult {
    results
        .iter()
        .find(|r| r.arm == arm && r.layer == layer)
        .unwrap()
}

#[test]
fn simulation_study() {
    let start = std::time::Instant::now();
    let results = study(N_SIMS);
    print(&results);
    println!(
        "\n{} simulations per arm in {:.1?}",
        N_SIMS,
        start.elapsed()
    );

    let names = method_names();
    let at = |name: &str| names.iter().position(|n| n == name).unwrap();
    for arm in Arm::ALL.into_iter().filter(|a| a.matched()) {
        let sparse = get(&results, arm, "sparse");
        let derived = sparse.stats[at("derived")].unwrap();
        assert!(
            derived.ratio < 0.8,
            "{}: the derived pattern's RMSE ratio on the sparse layer is {}",
            arm.name(),
            derived.ratio
        );
        for k in K_GRID {
            let blend = sparse.stats[at(&format!("blend k={k}"))].unwrap();
            assert!(
                blend.ratio < 0.8,
                "{}: the blend's RMSE ratio at k = {k} is {}",
                arm.name(),
                blend.ratio
            );
        }
        // The cap is what keeps a small k safe.
        let capped = sparse.stats[at("blend k=1")].unwrap();
        let uncapped = sparse.stats[at("uncapped k=1")].unwrap();
        assert!(capped.rmse < 0.5 * uncapped.rmse, "{}: the cap", arm.name());
        // The thin layer: the own chain ladder fits, the derived pattern
        // still beats it.
        let thin = get(&results, arm, "thin").stats[at("derived")].unwrap();
        assert!(
            thin.ratio < 0.8,
            "{}: thin layer ratio {}",
            arm.name(),
            thin.ratio
        );
        let dense = get(&results, arm, "dense").stats[at("derived")].unwrap();
        assert!(
            dense.mean.abs() <= 2.0 * dense.mcse,
            "{}: the derived pattern's mean error on the dense layer is {} (MC SE {})",
            arm.name(),
            dense.mean,
            dense.mcse
        );
    }

    // The findings where the scale shift's assumptions fail, pinned by
    // their direction: more than two Monte Carlo standard errors away
    // from zero.
    let derived =
        |arm: Arm, layer: &str, method: &str| get(&results, arm, layer).stats[at(method)].unwrap();
    let below = |s: Stats| s.mean < -2.0 * s.mcse;
    // Shape drift: the high layers develop more than a scale shift says.
    assert!(below(derived(Arm::Shape, "sparse", "derived")));
    assert!(below(derived(Arm::Shape, "thin", "derived")));
    assert!(below(derived(Arm::Pareto, "sparse", "scale shift")));
    assert!(below(derived(Arm::Pareto, "thin", "scale shift")));
    // Smaller late claims: understated on the dense layer, where the
    // understated severity shift outweighs the overstated count factor,
    // and no bias to speak of on the sparse one, where they cancel.
    assert!(below(derived(Arm::Late, "dense", "derived")));
    let late = derived(Arm::Late, "sparse", "derived");
    assert!(late.mean.abs() <= 2.0 * late.mcse);
}
