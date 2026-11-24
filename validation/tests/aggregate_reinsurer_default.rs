//! Reinsurer default on a tower against closed forms (#980,
//! `docs/design/reinsurer-default.md`, Validation): `reinsurer_default.csv`,
//! written by `validation/scripts/aggregate_reinsurer_default.py` from the
//! formulas by quadrature, with no simulation.
//!
//! The test simulates lognormal annual gross losses, applies a tower of two
//! annual excess layers (`Tower::apply_annual`) and each case's `Panel`, and
//! compares the simulated means and frequencies with the reference. Monte
//! Carlo noise dominates, so a simulated value passes when it is within
//! [`MAX_Z`] of its own standard errors of the closed form; the file's
//! `abs_tol` and `rel_tol` are the reference's own accuracy (what the weekly
//! rerun of the generator is held to), far below that noise. `check` still
//! sees every row: an estimate inside the band is handed to it as the
//! expected value, one outside as itself, so it fails and prints both.

use std::collections::HashMap;

use prospicio_aggregate::{Layer, Panel, Reinsurer, Tower};
use prospicio_core::StreamRng;
use prospicio_prob::{Distribution, Empirical, KeyValue, Lognormal, PredictiveDistribution};
use prospicio_validation::{Case, check, reference};

/// Simulated years: enough that the probabilities of default (10 % to
/// 25 %, the joint default about 4.6 %) and the 1 % of years above the
/// gross loss's 99th percentile are resolved.
const N_YEARS: usize = 300_000;
/// The gross losses' seed (stream 0), apart from the panels' seed.
const GROSS_SEED: u64 = 980;
/// The default draws' seed, the same for every case.
const DEFAULT_SEED: u64 = 2026;
/// A simulated value passes within this many of its own standard errors.
const MAX_Z: f64 = 4.0;
/// The tower's layers, as the generator names them.
const LAYERS: [&str; 2] = ["A", "B"];

/// A simulated value and its standard error.
#[derive(Debug, Clone, Copy)]
struct Estimate {
    value: f64,
    se: f64,
}

/// The sample mean and its standard error.
fn mean(xs: &[f64]) -> Estimate {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
    Estimate {
        value: m,
        se: (v / n).sqrt(),
    }
}

/// The sample variance and its standard error, `sqrt((m4 - s^4) / n)`.
fn variance(xs: &[f64]) -> Estimate {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    let s2 = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
    let m4 = xs.iter().map(|x| (x - m).powi(4)).sum::<f64>() / n;
    Estimate {
        value: s2,
        se: ((m4 - s2 * s2) / n).sqrt(),
    }
}

fn column(pd: &PredictiveDistribution, kind: &str, name: &str) -> Vec<f64> {
    let key = vec![KeyValue::from(kind), KeyValue::from(name)];
    Empirical::draws(&pd.marginal(&key).unwrap()).to_vec()
}

/// The case's panel, from its `params`: reinsurers `R1`, `R2`, ... with
/// their probability, recovery, collateral, correlation and lines.
fn panel(case: &Case) -> Panel {
    let at = |name: &str| case.param("params", name);
    let names: Vec<String> = (1..=at("reinsurers") as usize)
        .map(|k| format!("R{k}"))
        .collect();
    let reinsurers = names
        .iter()
        .map(|r| {
            Reinsurer::new(r, at(&format!("p_{r}")), at(&format!("rec_{r}")))
                .unwrap()
                .collateral(at(&format!("col_{r}")))
                .unwrap()
                .correlation(at(&format!("rho_{r}")))
                .unwrap()
        })
        .collect();
    let mut panel = Panel::new(reinsurers).unwrap();
    for r in &names {
        for l in LAYERS {
            let line = at(&format!("line_{r}_{l}"));
            if line > 0.0 {
                panel = panel.line(r, l, line).unwrap();
            }
        }
    }
    panel
}

/// One case's panel applied to the years: the tower's result after
/// default, and the reinsurers' credit loss and defaults.
struct Applied {
    after: PredictiveDistribution,
    by: PredictiveDistribution,
}

#[test]
fn reinsurer_default_matches_the_closed_forms() {
    let cases = reference("reinsurer_default.csv");
    // Every case shares the gross loss and the tower.
    let first = &cases[0];
    let shared = |c: &Case, name: &str| c.param("params", name);
    let tower_params = [
        "meanlog", "sdlog", "limit_A", "attach_A", "limit_B", "attach_B",
    ];
    for c in &cases {
        for name in tower_params {
            assert_eq!(shared(c, name), shared(first, name), "{name}");
        }
    }
    let (meanlog, sdlog) = (shared(first, "meanlog"), shared(first, "sdlog"));
    let gross = Lognormal::new(meanlog, sdlog)
        .unwrap()
        .sample(&mut StreamRng::new(GROSS_SEED, 0), N_YEARS);
    let tower = Tower::new(
        LAYERS
            .iter()
            .map(|&l| {
                let (limit, attach) = (
                    shared(first, &format!("limit_{l}")),
                    shared(first, &format!("attach_{l}")),
                );
                Layer::xol(l, limit, attach).unwrap()
            })
            .collect(),
    )
    .unwrap();
    let years = tower.apply_annual(&gross).unwrap();

    // The layers cede min((G - a)^+, l) in every year, so a swapped limit
    // and attachment fails here rather than as a z-score.
    let mut ceded = HashMap::new();
    for l in LAYERS {
        let (limit, attach) = (
            shared(first, &format!("limit_{l}")),
            shared(first, &format!("attach_{l}")),
        );
        let c = column(&years, "ceded", l);
        for (g, x) in gross.iter().zip(&c) {
            assert_eq!(*x, (g - attach).max(0.0).min(limit), "{l} at {g}");
        }
        ceded.insert(l, c);
    }
    let net = column(&years, "net", "retained");
    // The years above the gross loss's 99th percentile, by rank, as the
    // panel ranks them.
    let mut order: Vec<usize> = (0..N_YEARS).collect();
    order.sort_by(|&a, &b| gross[b].total_cmp(&gross[a]));
    let top = &order[..N_YEARS / 100];

    let mut applied: HashMap<String, Applied> = HashMap::new();
    let mut estimate = |case: &Case| -> Option<Estimate> {
        let name = case.get("distribution");
        let arg = case.get("arg");
        if name == "gross" {
            return match case.get("quantity") {
                "gross_mean" => Some(mean(&gross)),
                "ceded_mean" => ceded.get(arg).map(|c| mean(c)),
                _ => None,
            };
        }
        let a = applied.entry(name.to_string()).or_insert_with(|| {
            let p = panel(case);
            Applied {
                after: p.apply(&years, DEFAULT_SEED).unwrap(),
                by: p.by_reinsurer(&years, DEFAULT_SEED).unwrap(),
            }
        });
        let default = |r: &str| column(&a.by, "default", r);
        // The year's credit loss over every layer.
        let total = || -> Vec<f64> {
            let by_layer: Vec<Vec<f64>> = LAYERS
                .iter()
                .map(|l| column(&a.after, "credit_loss", l))
                .collect();
            (0..N_YEARS)
                .map(|i| by_layer.iter().map(|c| c[i]).sum())
                .collect()
        };
        match (case.get("quantity"), arg) {
            ("credit_loss_mean", "total") => Some(mean(&total())),
            ("credit_loss_var", "total") => Some(variance(&total())),
            ("credit_loss_mean", r) => Some(mean(&column(&a.by, "credit_loss", r))),
            ("credit_loss_var", r) => Some(variance(&column(&a.by, "credit_loss", r))),
            ("credit_loss_mean_by_layer", l) => Some(mean(&column(&a.after, "credit_loss", l))),
            ("net_after_default_minus_net_mean", "") => {
                let after = column(&a.after, "net_after_default", "retained");
                let diff: Vec<f64> = after.iter().zip(&net).map(|(x, y)| x - y).collect();
                Some(mean(&diff))
            }
            ("default_frequency", r) => Some(mean(&default(r))),
            ("default_given_top_1pct", r) => {
                let d = default(r);
                Some(mean(&top.iter().map(|&i| d[i]).collect::<Vec<_>>()))
            }
            ("joint_default_frequency", "R1+R2") => {
                let (d1, d2) = (default("R1"), default("R2"));
                Some(mean(&d1.iter().zip(&d2).map(|(x, y)| x * y).collect::<Vec<_>>()))
            }
            _ => None,
        }
    };

    let mut seen: HashMap<(String, String, String), (f64, Estimate)> = HashMap::new();
    println!(
        "{:<18} {:<34} {:<6} {:>14} {:>14} {:>12} {:>7}",
        "case", "quantity", "arg", "expected", "simulated", "se", "z"
    );
    check(&cases, |case| {
        let est = estimate(case)?;
        let want = case.number("expected").unwrap();
        let z = (est.value - want) / est.se;
        println!(
            "{:<18} {:<34} {:<6} {:>14.8} {:>14.8} {:>12.3e} {:>7.2}",
            case.get("distribution"),
            case.get("quantity"),
            case.get("arg"),
            want,
            est.value,
            est.se,
            z
        );
        seen.insert(
            (
                case.get("distribution").into(),
                case.get("quantity").into(),
                case.get("arg").into(),
            ),
            (want, est),
        );
        // The Monte Carlo band: inside it the estimate counts as the
        // closed form, outside it `check` reports the estimate itself.
        Some(if z.abs() <= MAX_Z { want } else { est.value })
    });

    // The test has power: every standard error is a few percent of the
    // value at most, and the factor moves the credit loss far further than
    // the band.
    for ((name, quantity, arg), (want, est)) in &seen {
        assert!(
            est.se <= 0.05 * want.abs(),
            "{name} {quantity} {arg}: se {} against {want}",
            est.se
        );
    }
    let key = |name: &str| (name.into(), "credit_loss_mean".into(), "R1".into());
    let (factor, est) = seen[&key("d_factor")];
    let (independent, _) = seen[&key("d_independent")];
    assert!(
        (factor - independent).abs() > 25.0 * est.se,
        "{factor} {independent} {}",
        est.se
    );
}
