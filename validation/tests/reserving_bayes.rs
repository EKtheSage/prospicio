//! The Bayesian ODP chain ladder (`BayesOdp`, `docs/design/reserving-bayes.md`,
//! decision 10) against the same model in Stan, and against the chain
//! ladder and the ODP bootstrap on GenIns.
//!
//! * Stan parity: `reference/reserving_bayes_stan.csv`, from
//!   `scripts/reserving_bayes_stan.py` (CmdStan 2.39.0, 4 chains of 25,000
//!   draws, dense metric). Our side runs the defaults: 4 chains of 1,000
//!   draws and 10,000 simulations, seed 0. With `s` Stan's posterior sd,
//!   means must agree within `0.1 s` (our MCSE is about `0.02 s`, Stan's
//!   `0.003 s`), sds within 8% (our MCSE about 1.2%), and the 5%, 50% and
//!   95% reserve percentiles within `0.15 s` (about 0.02 to 0.03 `s` of
//!   MCSE from 10,000 simulations). `phi` and `c0`, the plug-ins, agree to
//!   1e-9. Different samplers never give the same draws, so parity is on
//!   summaries.
//! * Sanity, not parity: on GenIns the posterior mean reserve is within 2%
//!   of the chain ladder's and of the ODP bootstrap's mean, and its sd
//!   within 10% of the bootstrap's (England and Verrall: vague priors give
//!   back the bootstrap's answer).
//! * `loo()` on GenIns gives a finite elpd and a Pareto k for each of the
//!   55 cells; on RAA, which has a negative increment, it is an error.

use std::sync::OnceLock;
use std::time::Instant;

use prospicio_prob::Distribution;
use prospicio_reserving::bayes::{BayesOdp, BayesOdpFit};
use prospicio_reserving::{ChainLadder, OdpBootstrap, Period};
use prospicio_validation::{check, reference, triangle};

fn fit(name: &str) -> &'static BayesOdpFit {
    static GENINS: OnceLock<BayesOdpFit> = OnceLock::new();
    static RAA: OnceLock<BayesOdpFit> = OnceLock::new();
    let cell = match name {
        "genins" => &GENINS,
        "raa" => &RAA,
        _ => panic!("no dataset {name}"),
    };
    cell.get_or_init(|| {
        let start = Instant::now();
        let fit = BayesOdp::default()
            .fit(&triangle(name), "values")
            .expect("fits");
        eprintln!(
            "{name}: BayesOdp default fit in {:.2?}; diagnostics {:?}",
            start.elapsed(),
            fit.diagnostics
        );
        fit
    })
}

#[test]
fn stan_parity() {
    let cases = reference("reserving_bayes_stan.csv");
    check(&cases, |case| {
        let fit = fit(case.get("dataset"));
        let arg = case.get("arg");
        let quantity = case.get("quantity");
        let reserve = || {
            if arg == "total" {
                fit.reserves.total().clone()
            } else {
                let origin = Period::year(arg.parse().expect("origin year"));
                fit.reserves.marginal(&vec![origin.into()]).expect("origin")
            }
        };
        Some(match quantity {
            "scale" => fit.scale,
            "c0" => fit
                .reserves
                .provenance()
                .parameters
                .iter()
                .find(|(k, _)| k == "c0")?
                .1
                .parse()
                .ok()?,
            "mean" | "sd" => {
                // The draws' moments directly: summary() also computes
                // R-hat and ESS, slow in a debug build for every case.
                let j = fit.posterior.names().iter().position(|n| n == arg)?;
                let draws = fit.posterior.parameter(j).concat();
                let n = draws.len() as f64;
                let mean = draws.iter().sum::<f64>() / n;
                let var = draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
                if quantity == "mean" { mean } else { var.sqrt() }
            }
            "reserve_mean" => reserve().mean(),
            "reserve_sd" => reserve().std_dev(),
            "reserve_q05" => reserve().quantile(0.05).ok()?,
            "reserve_q50" => reserve().quantile(0.5).ok()?,
            "reserve_q95" => reserve().quantile(0.95).ok()?,
            _ => return None,
        })
    });
}

#[test]
fn defaults_pass_the_diagnostics() {
    for name in ["genins", "raa"] {
        let d = &fit(name).diagnostics;
        assert!(d.max_rhat < 1.01, "{name}: {d:?}");
        assert!(d.min_ess_bulk > 400.0, "{name}: {d:?}");
    }
    assert!(
        fit("genins").diagnostics.warnings.is_empty(),
        "{:?}",
        fit("genins").diagnostics
    );
}

#[test]
fn genins_agrees_with_the_chain_ladder_and_the_bootstrap() {
    let tri = triangle("genins");
    let bayes = fit("genins").reserves.total();
    let cl = ChainLadder::default()
        .fit(&tri, "values")
        .unwrap()
        .total_reserve();
    let boot = OdpBootstrap {
        n_sims: 20_000,
        seed: 1,
        ..Default::default()
    }
    .fit(&tri, "values")
    .unwrap();
    let boot = boot.reserves.total();
    let (mean, sd) = (bayes.mean(), bayes.std_dev());
    eprintln!(
        "GenIns total reserve: Bayes mean {mean:.0} sd {sd:.0}; chain ladder {cl:.0}; \
         bootstrap mean {:.0} sd {:.0}",
        boot.mean(),
        boot.std_dev()
    );
    assert!((mean / cl - 1.0).abs() < 0.02, "{mean} vs {cl}");
    assert!(
        (mean / boot.mean() - 1.0).abs() < 0.02,
        "{mean} vs {}",
        boot.mean()
    );
    assert!(
        (sd / boot.std_dev() - 1.0).abs() < 0.10,
        "{sd} vs {}",
        boot.std_dev()
    );
}

#[test]
fn loo_on_genins_and_not_on_raa() {
    let start = Instant::now();
    let loo = fit("genins").loo().unwrap();
    eprintln!(
        "GenIns loo in {:.2?}: elpd {:.2} (se {:.2}), p_loo {:.2}, max k {:.3}",
        start.elapsed(),
        loo.estimate.elpd,
        loo.estimate.se,
        loo.estimate.p,
        loo.pareto_k
            .iter()
            .fold(f64::NEG_INFINITY, |m, &k| m.max(k))
    );
    assert_eq!(loo.pareto_k.len(), 55);
    assert_eq!(loo.estimate.pointwise.len(), 55);
    assert!(loo.estimate.elpd.is_finite() && loo.estimate.se.is_finite());
    assert!(loo.pareto_k.iter().all(|k| k.is_finite()));
    assert!(fit("raa").loo().is_err());
}
