//! The compartmental model's one-year view (`Compartmental::one_year`,
//! `docs/design/reserving-bayes.md`, decision 20) on the papers' company
//! 337 workers' compensation (`data/clrd_wkcomp_337.csv`), trained as
//! `reserving_compartmental.rs` trains it (calendar years up to 1996, and
//! 1997 with premium only), at the defaults of each model (model 1: 4
//! chains of 1,000 warm-up and 2,000 kept draws; model 2: 4 chains of
//! 2,000 and 8,000) and model 1's default one-year refit mode (moment
//! matching, then a refit for what it cannot rescue), set explicitly with
//! each view's number of simulations.
//!
//! There is no external reference for a Bayesian one-year view of these
//! models, so the checks are internal, against refits on the same years.
//! The year adds 20 cells (paid and outstanding of every origin's next
//! diagonal, 1997's first among them) to 90, so most simulations are
//! flagged (`k̂` above 0.7). For model 1:
//!
//! * `mean_z`: the default's total claims development result has mean
//!   zero within Monte Carlo error over `SIMS` simulations;
//! * `rescued`: the share of flagged simulations moment matching brings to
//!   `k̂` at most 0.7 is above a floor;
//! * `gap_matched`: on the first `REFIT_SIMS` simulations, the root mean
//!   square gap between a rescued simulation's total and its `RefitAll`
//!   value, in units of the total's sd, is below a bound and below
//!   `gap_reweighted`, the same simulations' gap without matching;
//! * `life_ratio`: the default's total sd over the lifetime reserve's is
//!   in a band around its sweep mean.
//!
//! Every bound is the sweep's mean minus (floors) or plus (bounds) at
//! least 4 seed-to-seed sds over seeds 0 to 19, and every seed passes
//! (`seed_sweep`, ignored: `SWEEP_SEEDS=20 cargo test --release -p
//! prospicio-validation --features slow-bayes --test
//! reserving_compartmental_one_year -- --ignored seed_sweep --nocapture`).
//!
//! Model 2 is measured (`model_2_comparison`, ignored): moment matching
//! rescues none of its simulations, and a refit takes about a minute, so
//! by the user's decision of 2026-10-16 its default view reweights only,
//! with a warning that it is approximate, and every model's default view
//! runs at most 2,000 simulations. `the_default_view_of_model_2_reweights`
//! checks those defaults end to end; it runs only when
//! `RISK_RS_FULL_BAYES` is set, as in CI's weekly and manual runs
//! (`prospicio_validation::full_bayes`). A pull request runs model 1 alone
//! and without its `RefitAll` comparison (the smoke run: `mean_z`,
//! `rescued` and `life_ratio` at the sizes and bounds above, about 3
//! minutes in release locally, against 6 for the full test). `runtime`
//! (ignored) measures both models' wall times.

use std::sync::OnceLock;
use std::time::Instant;

use prospicio_prob::{Distribution, PredictiveDistribution};
use prospicio_reserving::bayes::{
    BayesOneYear, BayesOneYearFit, Compartmental, CompartmentalFit, CompartmentalPriors,
    LightRefit, OneYearRefit, Outstanding,
};
use prospicio_reserving::{DevelopmentColumn, Grain, Lag, Long, Month, Triangle};
use prospicio_validation::full_bayes;

/// Simulations of model 1's default view.
const SIMS: usize = 500;
/// Of those, the simulations refitted with `RefitAll` for the gaps.
const REFIT_SIMS: usize = 40;

/// Model 1's floor on `rescued`, bound on `gap_matched` and band on
/// `life_ratio`, from `seed_sweep` over seeds 0 to 19 (mean, sd, extreme
/// seed): `rescued` 0.955 (0.017, min 0.911); `gap_matched` 0.051
/// (0.025, max 0.150 at seed 14, the next 0.072), against
/// `gap_reweighted` 0.156 (0.037), above it at every seed; `life_ratio`
/// 0.804 (0.031, from 0.756 to 0.865). The gap's tail is heavy (one seed
/// at 4 sds), so its bound is 0.20, above the mean plus 4 sds (0.150).
const RESCUED_FLOOR: f64 = 0.88;
const GAP_BOUND: f64 = 0.20;
const LIFE_BAND: (f64, f64) = (0.68, 0.93);

/// Model 2's comparison: simulations matched (no refit) and refitted.
const SIMS_2: usize = 200;
const REFIT_SIMS_2: usize = 8;

/// The rows of `data/clrd_wkcomp_337.csv`: origin, age in months, paid,
/// incurred, premium.
fn rows() -> Vec<(i32, Lag, f64, f64, f64)> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/clrd_wkcomp_337.csv");
    let text = std::fs::read_to_string(path).expect("data/clrd_wkcomp_337.csv");
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    assert_eq!(
        lines.next(),
        Some("origin,development,paid,incurred,case_incurred,premium,premium_net")
    );
    lines
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (
                f[0].parse().unwrap(),
                f[1].parse().unwrap(),
                f[2].parse().unwrap(),
                f[3].parse().unwrap(),
                f[5].parse().unwrap(),
            )
        })
        .collect()
}

/// The papers' training triangle: the cells at calendar years up to 1996,
/// columns `paid`, `incurred` and `premium` (direct), and 1997 with its
/// premium only.
fn training() -> &'static Triangle {
    static TRIANGLE: OnceLock<Triangle> = OnceLock::new();
    TRIANGLE.get_or_init(|| {
        let all = rows();
        let mut rows: Vec<_> = all
            .iter()
            .copied()
            .filter(|r| r.0 + (r.1 / 12) as i32 - 1 <= 1996)
            .collect();
        assert_eq!(rows.len(), 45);
        let premium_1997 = all.iter().find(|r| r.0 == 1997).unwrap().4;
        rows.push((1997, 12, f64::NAN, f64::NAN, premium_1997));
        let origin: Vec<Month> = rows.iter().map(|r| Month::january(r.0)).collect();
        let ages: Vec<Lag> = rows.iter().map(|r| r.1).collect();
        let column = |k: usize| -> Vec<f64> { rows.iter().map(|r| [r.2, r.3, r.4][k]).collect() };
        let (paid, incurred, premium) = (column(0), column(1), column(2));
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[
                ("paid", &paid),
                ("incurred", &incurred),
                ("premium", &premium),
            ],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    })
}

/// The bits of each value, so that a NaN Pareto `k` compares equal to itself.
fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn sd(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}

fn totals(d: &PredictiveDistribution) -> Vec<f64> {
    let c = d.n_components();
    d.draw_matrix().chunks(c).map(|r| r.iter().sum()).collect()
}

/// Model 1 or 2 at its defaults.
fn preset(two: bool) -> Compartmental {
    if two {
        Compartmental::model_2(CompartmentalPriors::gesmann_morris_2())
    } else {
        Compartmental::new(CompartmentalPriors::gesmann_morris())
    }
}

/// The one-year view of model 1 or 2 and its wall time.
fn view(
    two: bool,
    n_sims: usize,
    seed: u64,
    refit: OneYearRefit,
    max_refits: Option<usize>,
) -> (BayesOneYearFit<CompartmentalFit>, f64) {
    let t = Instant::now();
    let view = Compartmental {
        n_sims,
        seed,
        ..preset(two)
    }
    .one_year(
        training(),
        "paid",
        Outstanding::FromIncurred("incurred"),
        "premium",
        &BayesOneYear {
            refit: Some(refit),
            max_refits,
            // By default the view runs at most 2,000 simulations.
            n_sims: Some(n_sims),
            ..BayesOneYear::default()
        },
    )
    .unwrap_or_else(|e| panic!("model {}: {e}", if two { 2 } else { 1 }));
    (view, t.elapsed().as_secs_f64())
}

/// What one model measures at one seed.
#[derive(Debug, Clone, Copy)]
struct Measured {
    mean_z: f64,
    flagged_share: f64,
    rescued: f64,
    /// Simulations the default refitted (or, with a budget, left
    /// unrefitted).
    refitted: usize,
    gap_matched: f64,
    gap_reweighted: f64,
    /// The rms gap between reweighting only and the refit over every
    /// flagged simulation among the first `refit_sims`.
    gap_flagged_reweighted: f64,
    /// The rescued simulations' total sd under the default and under
    /// `ReweightOnly`, over their sd refitted.
    sd_matched: f64,
    sd_reweighted: f64,
    life_ratio: f64,
    /// Wall times: `ReweightOnly` and the default at `sims`, `RefitAll`
    /// at `refit_sims`.
    times: [f64; 3],
}

/// Model 1 (or 2) at one seed: `sims` simulations reweighted only and at
/// the default (with `max_refits`), the first `refit_sims` refitted (none
/// with 0: the gaps and sd ratios are then not measured).
fn measure(
    two: bool,
    sims: usize,
    refit_sims: usize,
    max_refits: Option<usize>,
    seed: u64,
) -> Measured {
    let name = if two { "model 2" } else { "model 1" };
    let (only, t_only) = view(two, sims, seed, OneYearRefit::ReweightOnly, None);
    let (default, t_default) = view(two, sims, seed, OneYearRefit::default(), max_refits);
    let (all, t_all) = if refit_sims > 0 {
        let (all, t) = view(two, refit_sims, seed, OneYearRefit::RefitAll, None);
        (Some(all), t)
    } else {
        (None, 0.0)
    };
    // The same years under every setting.
    assert_eq!(
        bits(&only.pareto_k),
        bits(&default.pareto_k),
        "{name}: the default's years"
    );
    if let Some(all) = &all {
        assert_eq!(
            bits(&all.pareto_k),
            bits(&only.pareto_k[..refit_sims]),
            "{name}: RefitAll's years"
        );
    }
    let (t_only_, t_def) = (totals(&only.cdr), totals(&default.cdr));
    let t_all_ = all.as_ref().map(|a| totals(&a.cdr)).unwrap_or_default();
    let s = sd(&t_def);
    let mean_z = mean(&t_def) / (s / (sims as f64).sqrt());
    let n_flagged = only
        .pareto_k
        .iter()
        .filter(|&&k| k.is_nan() || k > 0.7)
        .count();
    let rescued = if n_flagged == 0 {
        1.0
    } else {
        default.moment_matched.len() as f64 / n_flagged as f64
    };
    let matched: Vec<usize> = default
        .moment_matched
        .iter()
        .copied()
        .filter(|&i| i < refit_sims)
        .collect();
    let pick = |x: &[f64]| -> Vec<f64> { matched.iter().map(|&i| x[i]).collect() };
    let rms = |x: &[f64]| -> f64 {
        let sq: Vec<f64> = matched
            .iter()
            .map(|&i| ((x[i] - t_all_[i]) / s).powi(2))
            .collect();
        if sq.is_empty() { 0.0 } else { mean(&sq).sqrt() }
    };
    let sd_refit = sd(&pick(&t_all_));
    let flagged_refit: Vec<f64> = (0..refit_sims)
        .filter(|&i| only.pareto_k[i].is_nan() || only.pareto_k[i] > 0.7)
        .map(|i| ((t_only_[i] - t_all_[i]) / s).powi(2))
        .collect();
    let life = default.fit.reserves.total().std_dev();
    let m = Measured {
        mean_z,
        flagged_share: only.flagged_share(),
        rescued,
        refitted: default.refitted.len() + default.unrefitted.len(),
        gap_matched: rms(&t_def),
        gap_reweighted: rms(&t_only_),
        gap_flagged_reweighted: if flagged_refit.is_empty() {
            0.0
        } else {
            mean(&flagged_refit).sqrt()
        },
        sd_matched: sd(&pick(&t_def)) / sd_refit,
        sd_reweighted: sd(&pick(&t_only_)) / sd_refit,
        life_ratio: s / life,
        times: [t_only, t_default, t_all],
    };
    println!("{name} seed {seed}: {m:?}");
    m
}

/// Model 1's checks. A pull request (the smoke run, without
/// `RISK_RS_FULL_BAYES`) checks `mean_z`, `rescued` and `life_ratio` at the
/// same sizes and bounds and skips the `RefitAll` comparison (40 refits,
/// more than half the test's time), so not `gap_matched`; the weekly and
/// manual runs check everything.
#[test]
fn the_one_year_view_of_model_1_matches_its_refits() {
    let full = full_bayes();
    let m = measure(false, SIMS, if full { REFIT_SIMS } else { 0 }, None, 0);
    assert!(m.mean_z.abs() < 5.0, "mean z {}", m.mean_z);
    assert!(m.rescued >= RESCUED_FLOOR, "rescued {}", m.rescued);
    if full {
        assert!(m.gap_matched < GAP_BOUND, "gap {}", m.gap_matched);
        assert!(
            m.gap_matched < m.gap_reweighted,
            "matching {} is no closer than reweighting {}",
            m.gap_matched,
            m.gap_reweighted
        );
    }
    assert!(
        (LIFE_BAND.0..LIFE_BAND.1).contains(&m.life_ratio),
        "one-year over lifetime sd {}",
        m.life_ratio
    );
}

/// Model 2's default one-year view (the user's decision of 2026-10-16):
/// 2,000 simulations from the model's 10,000, every flagged one reweighted
/// (none moment-matched or refitted), with the warning that the view is
/// approximate (an explicit refit mode carries none: the unit test
/// `the_one_year_defaults_follow_the_model`). Full run only
/// (`RISK_RS_FULL_BAYES`): about a minute and a half in release.
#[test]
fn the_default_view_of_model_2_reweights() {
    if !full_bayes() {
        eprintln!("skipped: model 2's default view runs when RISK_RS_FULL_BAYES is set");
        return;
    }
    let model = preset(true);
    assert_eq!(model.n_sims, 10_000);
    let default = model
        .one_year(
            training(),
            "paid",
            Outstanding::FromIncurred("incurred"),
            "premium",
            &BayesOneYear::default(),
        )
        .unwrap();
    assert_eq!(default.pareto_k.len(), Compartmental::ONE_YEAR_SIMS);
    assert!(
        default.flagged_share() > 0.9,
        "flagged {}",
        default.flagged_share()
    );
    assert!(default.moment_matched.is_empty() && default.refitted.is_empty());
    assert!(default.unrefitted.is_empty());
    let warning = default.approximate.as_deref().expect("the default warns");
    assert!(warning.contains("approximate"), "{warning}");
    let parameters = &default.cdr.provenance().parameters;
    for (key, value) in [("refit", "ReweightOnly"), ("one_year_sims", "2000")] {
        assert!(
            parameters.contains(&(key.into(), value.into())),
            "{key} in {parameters:?}"
        );
    }
}

/// The papers' training triangle once under each of `names`, as the
/// segments of a `lob` key.
fn lines(names: &[&str]) -> Triangle {
    let all = rows();
    let mut rows: Vec<_> = all
        .iter()
        .copied()
        .filter(|r| r.0 + (r.1 / 12) as i32 - 1 <= 1996)
        .collect();
    let premium_1997 = all.iter().find(|r| r.0 == 1997).unwrap().4;
    rows.push((1997, 12, f64::NAN, f64::NAN, premium_1997));
    let (mut lob, mut origin, mut ages) = (vec![], vec![], vec![]);
    let (mut paid, mut incurred, mut premium) = (vec![], vec![], vec![]);
    for name in names {
        for r in &rows {
            lob.push(*name);
            origin.push(Month::january(r.0));
            ages.push(r.1);
            paid.push(r.2);
            incurred.push(r.3);
            premium.push(r.4);
        }
    }
    let key = [("lob", &lob[..])];
    Triangle::from_long(&Long {
        keys: &key,
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[
            ("paid", &paid),
            ("incurred", &incurred),
            ("premium", &premium),
        ],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// The sum of the columns `range` of each row of `d`.
fn range_totals(d: &PredictiveDistribution, range: std::ops::Range<usize>) -> Vec<f64> {
    let c = d.n_components();
    d.draw_matrix()
        .chunks(c)
        .map(|r| r[range.clone()].iter().sum())
        .collect()
}

/// The sd of the total claims development result of `one_year` on the
/// training triangle alone, to compare with a segment's part of
/// `one_year_segments`.
fn single_sd(model: &Compartmental, settings: &BayesOneYear) -> f64 {
    let view = model
        .one_year(
            training(),
            "paid",
            Outstanding::FromIncurred("incurred"),
            "premium",
            settings,
        )
        .unwrap();
    sd(&totals(&view.cdr))
}

/// Spearman's rho of two samples.
fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let ranks = |v: &[f64]| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
        let mut r = vec![0.0; v.len()];
        for (k, &i) in idx.iter().enumerate() {
            r[i] = k as f64;
        }
        r
    };
    let (rx, ry) = (ranks(x), ranks(y));
    let (mx, my) = (mean(&rx), mean(&ry));
    let sxy: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = rx.iter().map(|a| (a - mx).powi(2)).sum();
    let syy: f64 = ry.iter().map(|b| (b - my).powi(2)).sum();
    sxy / (sxx * syy).sqrt()
}

/// `Compartmental::one_year_segments` on two copies of company 337 (model 1,
/// reweighting only, the pull request's smoke run: two fits and `SIMS`
/// simulations a segment): each segment's total claims development result
/// has mean zero within Monte Carlo error and the sd of `one_year` on the
/// triangle alone within a band (other streams, so another posterior sample
/// and another set of years), the copies are rank-uncorrelated (shared
/// random numbers would make them almost perfectly correlated), and
/// `segment` returns its part of the joint result. Under
/// `RISK_RS_FULL_BAYES` the default refit mode (moment matching, then
/// refits) runs over the segments too, `max_refits` a budget of each.
#[test]
fn the_one_year_view_over_segments_matches_the_views_of_its_segments() {
    let model = Compartmental {
        n_sims: SIMS,
        seed: 0,
        ..preset(false)
    };
    let only = BayesOneYear {
        refit: Some(OneYearRefit::ReweightOnly),
        n_sims: Some(SIMS),
        ..BayesOneYear::default()
    };
    let t = Instant::now();
    let views = model
        .one_year_segments(
            &lines(&["a", "b"]),
            "paid",
            Outstanding::FromIncurred("incurred"),
            "premium",
            &only,
        )
        .unwrap();
    println!(
        "one_year_segments, 2 segments, {SIMS} simulations: {:.1} s",
        t.elapsed().as_secs_f64()
    );
    assert_eq!(views.cdr.n_components(), 20);
    assert_eq!(views.approximate, None);
    let alone = single_sd(&model, &only);
    let parts: Vec<Vec<f64>> = (0..2)
        .map(|k| range_totals(&views.cdr, 10 * k..10 * (k + 1)))
        .collect();
    for (k, part) in parts.iter().enumerate() {
        let z = mean(part) / (sd(part) / (SIMS as f64).sqrt());
        assert!(z.abs() < 5.0, "segment {k}: mean z {z}");
        let ratio = sd(part) / alone;
        println!("segment {k}: sd {} against one_year's {alone}", sd(part));
        assert!(
            (0.75..1.33).contains(&ratio),
            "segment {k}: sd ratio {ratio}"
        );
        assert_eq!(views.segments.fits[k].pareto_k.len(), SIMS);
    }
    let rho = spearman(&parts[0], &parts[1]);
    assert!(
        rho.abs() < 4.5 / (SIMS as f64).sqrt(),
        "rank correlation {rho}"
    );
    // A segment is its part of the joint result.
    let b = views.segment(&[("lob", "b")]).unwrap();
    assert_eq!(totals(&b.cdr), parts[1]);
    assert_eq!(
        bits(&b.segments.fits[0].pareto_k),
        bits(&views.segments.fits[1].pareto_k)
    );
    assert_eq!(views.totals().n_rows(), 2);

    if full_bayes() {
        let default = BayesOneYear {
            n_sims: Some(100),
            max_refits: Some(2),
            ..BayesOneYear::default()
        };
        let views = model
            .one_year_segments(
                &lines(&["a", "b"]),
                "paid",
                Outstanding::FromIncurred("incurred"),
                "premium",
                &default,
            )
            .unwrap();
        for (k, s) in views.segments.fits.iter().enumerate() {
            assert!(s.refitted.len() <= 2, "segment {k} refits");
            let dealt = s.moment_matched.len() + s.refitted.len() + s.unrefitted.len();
            let flagged = s
                .pareto_k
                .iter()
                .filter(|&&p| p.is_nan() || p > 0.7)
                .count();
            assert_eq!(dealt, flagged, "segment {k}: every flagged simulation");
        }
    }
}

/// Model 2's default view over segments: the model's defaults per segment
/// (every flagged simulation reweighted, at most 2,000 simulations) and the
/// warning that the views are approximate. Full run only
/// (`RISK_RS_FULL_BAYES`): two model-2 fits.
#[test]
fn the_default_view_of_model_2_over_segments_reweights() {
    if !full_bayes() {
        eprintln!("skipped: model 2's default view runs when RISK_RS_FULL_BAYES is set");
        return;
    }
    let views = preset(true)
        .one_year_segments(
            &lines(&["a", "b"]),
            "paid",
            Outstanding::FromIncurred("incurred"),
            "premium",
            &BayesOneYear::default(),
        )
        .unwrap();
    assert_eq!(views.cdr.n_sims(), Compartmental::ONE_YEAR_SIMS);
    let warning = views.approximate.as_deref().expect("the default warns");
    assert!(warning.contains("approximate"), "{warning}");
    for s in &views.segments.fits {
        assert!(s.flagged_share() > 0.9, "flagged {}", s.flagged_share());
        assert!(s.moment_matched.is_empty() && s.refitted.is_empty() && s.unrefitted.is_empty());
    }
    let parameters = &views.cdr.provenance().parameters;
    for (key, value) in [("refit", "ReweightOnly"), ("segments", "2")] {
        assert!(
            parameters.contains(&(key.into(), value.into())),
            "{key} in {parameters:?}"
        );
    }
}

fn summary(name: &str, row: &str, v: &[f64]) {
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = v.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "{name:8} {row:15} mean {:9.4} sd {:8.4} min {min:9.4} max {max:9.4} mean-4sd {:9.4} \
         mean+4sd {:9.4}",
        mean(v),
        sd(v),
        mean(v) - 4.0 * sd(v),
        mean(v) + 4.0 * sd(v)
    );
}

fn summarise(name: &str, ms: &[Measured]) {
    for (row, f) in [
        ("mean_z", (|m: &Measured| m.mean_z) as fn(&Measured) -> f64),
        ("flagged_share", |m| m.flagged_share),
        ("rescued", |m| m.rescued),
        ("refitted", |m| m.refitted as f64),
        ("gap_matched", |m| m.gap_matched),
        ("gap_reweighted", |m| m.gap_reweighted),
        ("gap_flag_rew", |m| m.gap_flagged_reweighted),
        ("sd_matched", |m| m.sd_matched),
        ("sd_reweighted", |m| m.sd_reweighted),
        ("life_ratio", |m| m.life_ratio),
        ("t_reweight", |m| m.times[0]),
        ("t_default", |m| m.times[1]),
        ("t_refit_all", |m| m.times[2]),
    ] {
        let v: Vec<f64> = ms.iter().map(f).collect();
        summary(name, row, &v);
    }
}

fn seeds(default: u64) -> u64 {
    std::env::var("SWEEP_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Model 1's seed-to-seed spread of the measured values (ignored; release
/// build, minutes a seed).
#[test]
#[ignore]
fn seed_sweep() {
    let ms: Vec<Measured> = (0..seeds(20))
        .map(|seed| measure(false, SIMS, REFIT_SIMS, None, seed))
        .collect();
    summarise("model 1", &ms);
}

/// Model 2 at a few seeds (ignored; release build, many minutes a seed):
/// `SIMS_2` simulations moment-matched and never refitted (`max_refits`
/// 0), the first `REFIT_SIMS_2` also refitted, for the share matching
/// rescues and the gaps.
#[test]
#[ignore]
fn model_2_comparison() {
    let ms: Vec<Measured> = (0..seeds(3))
        .map(|seed| measure(true, SIMS_2, REFIT_SIMS_2, Some(0), seed))
        .collect();
    summarise("model 2", &ms);
}

/// Wall times (ignored; release build): model 1 at 10,000 simulations
/// reweighting only and moment matching (its default mode); model 2 at
/// 10,000 reweighting only (its default mode) and at 1,000 moment matching
/// without refits (`max_refits` 0), with the number of refits matching
/// would make.
#[test]
#[ignore]
fn runtime() {
    for two in [false, true] {
        let (only, t_only) = view(two, 10_000, 0, OneYearRefit::ReweightOnly, None);
        let (sims, budget) = if two {
            (1_000, Some(0))
        } else {
            (10_000, None)
        };
        let (default, t_default) = view(two, sims, 0, OneYearRefit::default(), budget);
        println!(
            "model {}: 10,000 simulations ReweightOnly {t_only:.1} s (flagged {:.3}); {sims} \
             simulations at the default with max_refits {budget:?} {t_default:.1} s (matched \
             {}, refitted {}, left unrefitted {})",
            if two { 2 } else { 1 },
            only.flagged_share(),
            default.moment_matched.len(),
            default.refitted.len(),
            default.unrefitted.len()
        );
    }
}

/// The light refit settings to try: `LIGHT="chains,tune,draws;..."`.
fn light_settings() -> Vec<LightRefit> {
    let text = std::env::var("LIGHT").unwrap_or_else(|_| "4,150,250;4,300,500".into());
    text.split(';')
        .map(|c| {
            let v: Vec<usize> = c.split(',').map(|x| x.trim().parse().unwrap()).collect();
            LightRefit {
                chains: v[0],
                tune: v[1],
                draws: v[2],
            }
        })
        .collect()
}

/// What a light refit's view of model 2 gives against full refits.
#[derive(Debug, Clone, Copy)]
struct LightGap {
    /// The rms gap of the simulations' totals to the full refits', in
    /// units of the total's sd, and its mean.
    rms: f64,
    mean: f64,
    /// The totals' sd over the full refits' sd.
    sd_ratio: f64,
    /// Refits whose closing reserve warned.
    warned: usize,
}

/// Model 2's first `sims` years refitted in full (cached by
/// `LIGHT_CACHE`, a path prefix, when set) and with each of `lights`, all
/// `RefitAll`: the gaps of each light refit, and reweighting only's rms
/// gap, in units of the total's sd over 1,000 reweighted simulations.
fn light_gaps(sims: usize, seed: u64, lights: &[LightRefit]) -> (f64, Vec<LightGap>) {
    let (only, t_fit) = view(true, 1_000, seed, OneYearRefit::ReweightOnly, None);
    let s = sd(&totals(&only.cdr));
    println!("reweight sd of the total {s:.1}; the fit and 1,000 reweightings {t_fit:.1} s");
    let cache = std::env::var("LIGHT_CACHE")
        .ok()
        .map(|p| std::path::PathBuf::from(format!("{p}.{seed}.{sims}.txt")));
    let cached = cache
        .as_ref()
        .and_then(|c| std::fs::read_to_string(c).ok())
        .map(|text| {
            text.lines()
                .map(|l| l.parse().unwrap())
                .collect::<Vec<f64>>()
        });
    let full_totals = cached.unwrap_or_else(|| {
        let (full, t_full) = view(true, sims, seed, OneYearRefit::RefitAll, None);
        println!(
            "full refits {t_full:.1} s incl. the fit, warned {} of {sims}",
            full.refit_warnings
        );
        let t = totals(&full.cdr);
        if let Some(c) = &cache {
            let text: Vec<String> = t.iter().map(|v| format!("{v:?}")).collect();
            std::fs::write(c, text.join("\n")).unwrap();
        }
        t
    });
    let gap = |x: &[f64]| -> (f64, f64) {
        let d: Vec<f64> = x
            .iter()
            .zip(&full_totals)
            .map(|(a, b)| (a - b) / s)
            .collect();
        (
            (d.iter().map(|v| v * v).sum::<f64>() / d.len() as f64).sqrt(),
            mean(&d),
        )
    };
    let reweighted = gap(&totals(&only.cdr)[..sims]).0;
    let gaps = lights
        .iter()
        .map(|&light| {
            let t = Instant::now();
            let v = Compartmental {
                n_sims: sims,
                seed,
                ..preset(true)
            }
            .one_year(
                training(),
                "paid",
                Outstanding::FromIncurred("incurred"),
                "premium",
                &BayesOneYear {
                    refit: Some(OneYearRefit::RefitAll),
                    n_sims: Some(sims),
                    light_refit: Some(light),
                    ..BayesOneYear::default()
                },
            )
            .unwrap();
            let wall = t.elapsed().as_secs_f64();
            let (rms, mean) = gap(&totals(&v.cdr));
            let g = LightGap {
                rms,
                mean,
                sd_ratio: sd(&totals(&v.cdr)) / sd(&full_totals),
                warned: v.refit_warnings,
            };
            println!(
                "{light:?}: wall {wall:.1} s ({:.2} s a refit beyond the fit), rms gap to \
                 full {rms:.3}, mean {mean:.3}, warned {} of {sims}, sd ratio {:.3}",
                (wall - t_fit) / sims as f64,
                g.warned,
                g.sd_ratio
            );
            g
        })
        .collect();
    (reweighted, gaps)
}

/// Model 2's warm-started, shorter refits (`BayesOneYear::light_refit`,
/// decision 24) against its full refits, simulation by simulation, on the
/// first `LIGHT_SIMS` years of seed `LIGHT_SEED` (ignored; release build,
/// many minutes).
#[test]
#[ignore]
fn model_2_light_refit() {
    let env = |name: &str, default: u64| -> u64 {
        std::env::var(name)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    };
    let (reweighted, _) = light_gaps(
        env("LIGHT_SIMS", 16) as usize,
        env("LIGHT_SEED", 0),
        &light_settings(),
    );
    println!("reweight only vs full: rms gap {reweighted:.3}");
}

/// The default light refit of model 2 comes within a small gap of its full
/// refits on the first 16 years of seed 0 (measured 0.035 on those, mean
/// 0.006, sd ratio 1.015; seed 0 was not used to choose the default, which
/// was tuned on seed 1 at 0.029; the full refits' own noise is about 0.03),
/// where reweighting only is 0.52 off; its sd of the total is the full
/// refits' within 10%. Runs only with `RISK_RS_FULL_BAYES` (about 10
/// minutes in release): the full refits are what costs.
#[test]
fn the_light_refit_of_model_2_matches_its_full_refits() {
    if !full_bayes() {
        eprintln!("skipped: set RISK_RS_FULL_BAYES to run the full refits it compares with");
        return;
    }
    let (reweighted, gaps) = light_gaps(16, 0, &[LightRefit::default()]);
    let g = gaps[0];
    assert!(g.rms < 0.08, "{g:?}");
    assert!(g.mean.abs() < 0.04, "{g:?}");
    assert!((g.sd_ratio - 1.0).abs() < 0.10, "{g:?}");
    assert!(g.rms < reweighted / 2.0, "{g:?} against {reweighted}");
}

/// Model 2's default view with a light refit (ignored; release build): its
/// wall time and what it did, at 2,000 simulations (the default count) and
/// the default light sampler, against the default (reweighting only).
#[test]
#[ignore]
fn light_runtime() {
    let sims: usize = std::env::var("LIGHT_SIMS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000);
    let (_, t_reweight) = view(true, sims, 0, OneYearRefit::ReweightOnly, None);
    println!("model 2, {sims} simulations reweighted only: {t_reweight:.1} s");
    let t = Instant::now();
    let v = Compartmental {
        n_sims: sims,
        ..preset(true)
    }
    .one_year(
        training(),
        "paid",
        Outstanding::FromIncurred("incurred"),
        "premium",
        &BayesOneYear {
            n_sims: Some(sims),
            light_refit: Some(light_settings()[0]),
            ..BayesOneYear::default()
        },
    )
    .unwrap();
    let totals = totals(&v.cdr);
    println!(
        "model 2, {sims} simulations, default light refit: {:.1} s, flagged {:.3}, refitted {}, \
         warned {}, approximate {:?}, total cdr mean {:.0} sd {:.0}",
        t.elapsed().as_secs_f64(),
        v.flagged_share(),
        v.refitted.len(),
        v.refit_warnings,
        v.approximate,
        mean(&totals),
        sd(&totals)
    );
}
