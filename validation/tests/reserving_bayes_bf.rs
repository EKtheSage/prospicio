//! The Bayesian Bornhuetter–Ferguson (`BayesBf`,
//! `docs/design/reserving-bayes.md`, decisions 9 and 10) against the same
//! model in Stan, and against England, Verrall and Wüthrich (2012), whose
//! Table 1 triangle and prior ultimates are `data/evw2012.csv`.
//!
//! * Stan parity: `reference/reserving_bayes_bf_stan.csv`, from
//!   `scripts/reserving_bayes_bf_stan.py` (CmdStan 2.39.0, 4 chains of
//!   25,000 draws, dense metric), four cases: GenIns with premium, apriori
//!   0.5 and cv 0.2; EVW's triangle at their `a = 100` (cv 0.1) with
//!   `development_sd` 10 (the default) and 100, and at their `a = 0`
//!   (cv infinite) with 100. Our side runs the defaults otherwise (4
//!   chains of 1,000 draws, 10,000 simulations, seed 0). With `s` Stan's
//!   posterior sd: means within `0.1 s`, sds within 15%, reserve
//!   percentiles within `0.3 s`, each at least 4.7 times the largest
//!   seed-to-seed sd of its family over seeds 0 to 199 (`seed_sweep`: at
//!   most `0.021 s`, 2.75% and `0.063 s`). Seeds 0 to 199 pass every row.
//! * EVW's printed results (their Tables 2 and 3, Gibbs sampling with
//!   1,000,000 draws and their flat prior on `log gamma_d`, here
//!   `development_sd` 100): the total reserve's mean within 0.5% and its
//!   root mean square error of prediction within 5% at `a = 100` and
//!   `a = 0`; at `a = infinity`, which they give in closed form, the closed
//!   form itself to 2 (their `phi` is rounded to 14,714) and a fit at cv
//!   0.001 within 0.5% and 5%. Over seeds 0 to 199 our seed-to-seed sd is
//!   at most 0.10% on these means and 1.1% on these sds (largest
//!   errors 0.29% and 3.5%), so each tolerance is at least 4.5 times it.

use std::sync::OnceLock;
use std::time::Instant;

use prospicio_prob::Distribution;
use prospicio_reserving::bayes::{BayesBf, BayesBfFit};
use prospicio_reserving::{BornhuetterFerguson, ChainLadder, Period, Triangle};
use prospicio_validation::{Case, check, reference, triangle_columns};

/// The triangle, loss column and exposure column of a dataset.
fn data(name: &str) -> (Triangle, &'static str) {
    match name {
        "genins_premium" => (triangle_columns(name, &["paid", "premium"]), "premium"),
        "evw2012" => (triangle_columns(name, &["paid", "prior"]), "prior"),
        _ => panic!("no dataset {name}"),
    }
}

fn model(apriori: f64, cv: f64, development_sd: f64, seed: u64) -> BayesBf {
    BayesBf {
        apriori,
        cv,
        development_sd,
        seed,
        ..BayesBf::default()
    }
}

fn run(name: &str, apriori: f64, cv: f64, development_sd: f64, seed: u64) -> BayesBfFit {
    let (tri, exposure) = data(name);
    let start = Instant::now();
    let fit = model(apriori, cv, development_sd, seed)
        .fit(&tri, "paid", exposure)
        .expect("fits");
    eprintln!(
        "{name} cv {cv} sd {development_sd} seed {seed}: BayesBf fit in {:.2?}; {:?}",
        start.elapsed(),
        fit.diagnostics
    );
    fit
}

/// The seed-0 fit of each reference case, keyed by its `params`.
fn fit(case: &Case) -> &'static BayesBfFit {
    static FITS: OnceLock<std::sync::Mutex<Vec<(String, &'static BayesBfFit)>>> = OnceLock::new();
    let key = format!("{}/{}", case.get("dataset"), case.get("params"));
    let fits = FITS.get_or_init(Default::default);
    let mut fits = fits.lock().unwrap();
    if let Some((_, f)) = fits.iter().find(|(k, _)| *k == key) {
        return f;
    }
    let f: &'static BayesBfFit = Box::leak(Box::new(run(
        case.get("dataset"),
        case.param("params", "apriori"),
        case.param("params", "cv"),
        case.param("params", "development_sd"),
        0,
    )));
    fits.push((key, f));
    f
}

/// Our value for a reference row.
fn value(fit: &BayesBfFit, case: &Case) -> Option<f64> {
    let arg = case.get("arg");
    let reserve = || {
        if arg == "total" {
            fit.reserves.total().clone()
        } else {
            let origin = Period::year(arg.parse().expect("origin year"));
            fit.reserves.marginal(&vec![origin.into()]).expect("origin")
        }
    };
    Some(match case.get("quantity") {
        "scale" => fit.scale,
        q @ ("mean" | "sd") => {
            // The draws' moments directly: summary() also computes R-hat
            // and ESS, slow in a debug build.
            let j = fit.posterior.names().iter().position(|n| n == arg)?;
            let draws = fit.posterior.parameter(j).concat();
            let n = draws.len() as f64;
            let mean = draws.iter().sum::<f64>() / n;
            let var = draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
            if q == "mean" { mean } else { var.sqrt() }
        }
        "reserve_mean" => reserve().mean(),
        "reserve_sd" => reserve().std_dev(),
        "reserve_q05" => reserve().quantile(0.05).ok()?,
        "reserve_q50" => reserve().quantile(0.5).ok()?,
        "reserve_q95" => reserve().quantile(0.95).ok()?,
        _ => return None,
    })
}

#[test]
fn stan_parity() {
    let cases = reference("reserving_bayes_bf_stan.csv");
    check(&cases, |case| value(fit(case), case));
}

/// The defaults pass every threshold on each reference case (seeds 0 to
/// 49 gave no divergence; see `docs/design/reserving-bayes.md`).
#[test]
fn defaults_pass_the_diagnostics() {
    let cases = reference("reserving_bayes_bf_stan.csv");
    for case in cases.iter().filter(|c| c.get("quantity") == "scale") {
        let d = &fit(case).diagnostics;
        let name = format!("{} {}", case.get("dataset"), case.get("params"));
        assert!(d.warnings.is_empty(), "{name}: {d:?}");
        assert_eq!(d.divergences, 0, "{name}: {d:?}");
    }
}

/// EVW's strong-prior limit (their sec. 3.3, `a = infinity`): the total
/// reserve's mean `sum m_w S_d / M_d` over the future cells and its root
/// mean square error of prediction, process `phi` times the mean plus the
/// parameter variance `sum_d (sum of the open m_w)^2 phi S_d / M_d^2`.
fn strong_prior_limit(tri: &Triangle, fit: &BayesBfFit) -> (f64, f64) {
    let (latest, m, phi) = (
        &fit.chain_ladder.latest_position,
        &fit.prior_ultimate,
        fit.scale,
    );
    let (no, nd) = (latest.len(), latest[0] + 1);
    let (mut s, mut big_m) = (vec![0.0; nd], vec![0.0; nd]);
    for w in 0..no {
        let mut previous = 0.0;
        for d in 0..=latest[w] {
            // Segment 0, column 0 (`paid`).
            let cum = tri.get(0, 0, w, d).unwrap();
            s[d] += cum - previous;
            big_m[d] += m[w];
            previous = cum;
        }
    }
    let (mut mean, mut var) = (0.0, 0.0);
    for d in 0..nd {
        let open: f64 = (0..no).filter(|&w| d > latest[w]).map(|w| m[w]).sum();
        mean += open * s[d] / big_m[d];
        var += open * open * phi * s[d] / (big_m[d] * big_m[d]);
    }
    (mean, (var + phi * mean).sqrt())
}

#[test]
fn england_verrall_wuthrich_2012() {
    let (tri, _) = data("evw2012");
    let cl = ChainLadder::default().fit(&tri, "paid").unwrap();
    // Their Table 2: the chain ladder's reserve, 6,047,059.
    assert!((cl.total_reserve() - 6_047_059.0).abs() < 1.0);
    let fits: Vec<(f64, &BayesBfFit)> = [0.1, f64::INFINITY]
        .into_iter()
        .map(|cv| {
            let case = reference("reserving_bayes_bf_stan.csv")
                .into_iter()
                .find(|c| {
                    c.get("dataset") == "evw2012"
                        && c.param("params", "development_sd") == 100.0
                        && c.param("params", "cv") == cv
                })
                .expect("a reference case");
            (cv, fit(&case))
        })
        .collect();
    // (cv, their MMSE total reserve, their MSEP^(1/2)): Table 3 at a = 100
    // and Table 2 (= Table 3) at a = 0.
    for ((cv, f), (mean, msep)) in fits
        .iter()
        .zip([(6_145_526.0, 422_526.0), (6_049_398.0, 430_160.0)])
    {
        let total = f.reserves.total();
        eprintln!(
            "EVW cv {cv}: ours {:.0} ({:.0}), theirs {mean:.0} ({msep:.0})",
            total.mean(),
            total.std_dev()
        );
        assert!(
            (total.mean() / mean - 1.0).abs() < 0.005,
            "cv {cv}: {}",
            total.mean()
        );
        assert!(
            (total.std_dev() / msep - 1.0).abs() < 0.05,
            "cv {cv}: {}",
            total.std_dev()
        );
    }

    // a = infinity: their closed form, 6,644,047 (395,012), from phi
    // 14,714; ours from the Pearson scale 14,714.08.
    let (mean, msep) = strong_prior_limit(&tri, fits[0].1);
    assert!((mean - 6_644_047.0).abs() < 2.0, "{mean}");
    assert!((msep - 395_012.0).abs() < 2.0, "{msep}");
    let tight = run("evw2012", 1.0, 1e-3, 100.0, 0);
    let total = tight.reserves.total();
    eprintln!(
        "EVW cv 0.001: ours {:.0} ({:.0}), closed form {mean:.0} ({msep:.0})",
        total.mean(),
        total.std_dev()
    );
    assert!(
        (total.mean() / mean - 1.0).abs() < 0.005,
        "{}",
        total.mean()
    );
    assert!(
        (total.std_dev() / msep - 1.0).abs() < 0.05,
        "{}",
        total.std_dev()
    );

    // The classical Bornhuetter–Ferguson with the same prior ultimates
    // takes the chain ladder's pattern and gives 7,356,578, far from the
    // strong-prior limit: the row priors reshape the Bayesian pattern.
    let bf = BornhuetterFerguson {
        apriori: 1.0.into(),
        ..Default::default()
    }
    .fit(&tri, "paid", "prior")
    .unwrap()
    .total_reserve();
    assert!((bf - 7_356_578.47).abs() < 0.01, "{bf}");
}

/// The Monte Carlo error behind the parity tolerances: every reference
/// row for seeds `0..BF_SEEDS` (default 20), written to
/// `target/bayes_bf_sweep.csv`. Run in a release build:
/// `cargo test --release -p prospicio-validation --test reserving_bayes_bf
/// seed_sweep -- --ignored --nocapture`.
#[test]
#[ignore]
fn seed_sweep() {
    let seeds: u64 = std::env::var("BF_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let cases = reference("reserving_bayes_bf_stan.csv");
    let mut keys: Vec<(String, String)> = Vec::new();
    for c in &cases {
        let key = (c.get("dataset").to_string(), c.get("params").to_string());
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    let mut out = String::from(
        "seed,dataset,params,quantity,arg,got,expected,abs_tol,rel_tol,divergences,warnings\n",
    );
    for seed in 0..seeds {
        for (dataset, params) in &keys {
            let case = cases
                .iter()
                .find(|c| c.get("dataset") == dataset && c.get("params") == params)
                .unwrap();
            let fit = run(
                dataset,
                case.param("params", "apriori"),
                case.param("params", "cv"),
                case.param("params", "development_sd"),
                seed,
            );
            let d = &fit.diagnostics;
            if dataset == "evw2012"
                && case.param("params", "cv") == 0.1
                && case.param("params", "development_sd") == 100.0
            {
                // The strong-prior check of england_verrall_wuthrich_2012.
                let tight = run(dataset, 1.0, 1e-3, 100.0, seed);
                let (mean, msep) = strong_prior_limit(&data(dataset).0, &tight);
                let total = tight.reserves.total();
                for (q, got, want) in [
                    ("reserve_mean", total.mean(), mean),
                    ("reserve_sd", total.std_dev(), msep),
                ] {
                    out.push_str(&format!(
                        "{seed},{dataset},apriori=1;cv=0.001;development_sd=100,{q},total,{got:e},\
                         {want},0,{},{},{}\n",
                        if q == "reserve_mean" { 0.005 } else { 0.05 },
                        tight.diagnostics.divergences,
                        tight.diagnostics.warnings.len()
                    ));
                }
            }
            for case in cases
                .iter()
                .filter(|c| c.get("dataset") == dataset && c.get("params") == params)
            {
                let got = value(&fit, case).unwrap_or(f64::NAN);
                out.push_str(&format!(
                    "{seed},{dataset},{params},{},{},{got:e},{},{},{},{},{}\n",
                    case.get("quantity"),
                    case.get("arg"),
                    case.get("expected"),
                    case.get("abs_tol"),
                    case.get("rel_tol"),
                    d.divergences,
                    d.warnings.len()
                ));
            }
        }
    }
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../target/bayes_bf_sweep.csv");
    std::fs::write(path, out).unwrap();
    eprintln!("wrote {path}");
}
