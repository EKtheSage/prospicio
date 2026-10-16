//! `PRO.BAYES.ODP`, the Bayesian ODP chain ladder sampled with NUTS,
//! reached through the catalogue as Excel reaches it:
//!
//! * at its defaults, against the same model in CmdStan 2.39
//!   (`validation/reference/reserving_bayes_stan.csv`, 4 chains of 25,000
//!   draws) on RAA and GenIns: every parameter's posterior mean and SD and
//!   every origin's and the total's reserve mean, SD and 5%, 50% and 95%
//!   quantiles, at the file's tolerances (four times the Monte Carlo error
//!   of these defaults);
//! * every view is the library's own fit with the same settings, exactly,
//!   since the fit is reproducible by seed;
//! * the chains converge at the defaults (R-hat under 1.01, no
//!   divergences), and shorter chains are flagged on the sheet;
//! * the cache returns the fit a refit gives, and a new seed a new fit.

mod common;

use common::{genins, n, num, raa, reference, t, text};
use prospicio_bayes::nuts::Sampler;
use prospicio_prob::Distribution;
use prospicio_reserving::bayes::BayesOdp;
use prospicio_xll::call;
use prospicio_xll::reserving::triangle;
use prospicio_xll::sheet::{Grid, Value};

/// `PRO.BAYES.ODP(RAA, output, 500, 4, 7, 4000)`: small enough for a
/// debug build.
fn small(output: &str) -> Grid {
    call(
        "pro_bayes_odp",
        vec![raa(), t(output), n(500.0), n(4.0), n(7.0), n(4000.0)],
    )
}

#[test]
fn views_are_the_library_fit() {
    let model = BayesOdp {
        sampler: Sampler {
            chains: 4,
            tune: 500,
            draws: 500,
            ..BayesOdp::default().sampler
        },
        n_sims: 4000,
        seed: 7,
        ..BayesOdp::default()
    };
    let fit = model.fit(&triangle(&raa()).unwrap(), "values").unwrap();

    let ex = small("reserves");
    assert_eq!((ex.rows, ex.columns), (12, 9), "{:?}", ex.as_scalar());
    assert_eq!(ex.get(11, 0), &Value::from("Total"));
    let total = num(ex.get(11, 2));
    assert_eq!(total, fit.reserves.mean());

    let ps = small("params");
    let summary = fit.summary().unwrap();
    assert_eq!(ps.rows, summary.len() + 1);
    assert_eq!(ps.get(1, 0), &Value::from("c"));
    for (r, p) in summary.iter().enumerate() {
        assert_eq!(ps.get(r + 1, 0), &Value::Text(p.name.clone()));
        assert_eq!(num(ps.get(r + 1, 1)), p.mean);
        assert_eq!(num(ps.get(r + 1, 6)), p.rhat);
    }

    let d = small("diagnostics");
    assert_eq!(num(d.get(0, 1)), fit.diagnostics.max_rhat);
    assert_eq!(num(d.get(3, 1)), fit.diagnostics.divergences as f64);
    assert_eq!(num(d.get(6, 1)), fit.scale);
}

#[test]
fn matches_stan_at_the_defaults() {
    let ps = Grid::from_rows(vec![vec![
        Value::Number(0.05),
        Value::Number(0.5),
        Value::Number(0.95),
    ]]);
    let mut checked = 0;
    for (dataset, tri, first) in [("raa", raa(), 1981), ("genins", genins().0, 2001)] {
        let ex = call(
            "pro_bayes_odp",
            vec![
                tri.clone(),
                Grid::missing(),
                Grid::missing(),
                Grid::missing(),
                Grid::missing(),
                Grid::missing(),
                ps.clone(),
            ],
        );
        let params = call("pro_bayes_odp", vec![tri.clone(), t("params")]);
        let diag = call("pro_bayes_odp", vec![tri, t("diagnostics")]);
        assert!(
            num(diag.get(0, 1)) < 1.01,
            "{dataset} R-hat {:?}",
            diag.get(0, 1)
        );
        assert_eq!(num(diag.get(3, 1)), 0.0, "{dataset} divergences");
        assert_eq!(diag.get(diag.rows - 1, 1), &Value::from("none"));
        for row in reference("reserving_bayes_stan.csv") {
            if row[0] != dataset || row[1] != "bayes_odp" {
                continue;
            }
            let (quantity, arg) = (row[2].as_str(), row[3].as_str());
            let got = match quantity {
                "scale" => num(diag.get(6, 1)),
                "mean" | "sd" => {
                    let r = (1..params.rows)
                        .find(|&r| params.get(r, 0) == &Value::from(arg))
                        .unwrap_or_else(|| panic!("{dataset} {arg}"));
                    num(params.get(r, if quantity == "mean" { 1 } else { 2 }))
                }
                q if q.starts_with("reserve_") => {
                    let r = if arg == "total" {
                        ex.rows - 1
                    } else {
                        arg.parse::<usize>().unwrap() - first + 1
                    };
                    let col = match q {
                        "reserve_mean" => 2,
                        "reserve_sd" => 3,
                        "reserve_q05" => 5,
                        "reserve_q50" => 6,
                        "reserve_q95" => 7,
                        other => panic!("{other}"),
                    };
                    num(ex.get(r, col))
                }
                _ => continue,
            };
            let expected: f64 = row[4].parse().unwrap();
            let (abs_tol, rel_tol): (f64, f64) = (row[5].parse().unwrap(), row[6].parse().unwrap());
            assert!(
                (got - expected).abs() <= abs_tol.max(rel_tol * expected.abs()),
                "{dataset} {quantity} {arg}: {got} vs {expected} (tol {abs_tol}, {rel_tol})"
            );
            checked += 1;
        }
    }
    assert!(checked > 150, "checked {checked}");

    // Chains of 500 draws leave one R-hat just above 1.01 on seed 7,
    // and the sheet is told.
    let short = small("diagnostics");
    assert!(num(short.get(0, 1)) > 1.01);
    assert_eq!(short.get(short.rows - 1, 0), &Value::from("Warning"));
    match short.get(short.rows - 1, 1) {
        Value::Text(w) => assert!(w.contains("R-hat"), "{w}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_cache_returns_the_refit_and_a_new_seed_a_new_fit() {
    let a = small("reserves");
    let b = small("reserves");
    assert_eq!(a, b);
    let c = call(
        "pro_bayes_odp",
        vec![raa(), Grid::missing(), n(500.0), n(4.0), n(8.0), n(4000.0)],
    );
    assert_ne!(num(a.get(11, 2)), num(c.get(11, 2)));
    // A different view of a kept fit is cheap and consistent.
    let p = small("params");
    assert_eq!(p.get(0, 6), &Value::from("R-hat"));
}

#[test]
fn errors_come_back_as_text() {
    let e = |a: Vec<Grid>| text(&call("pro_bayes_odp", a)).to_owned();
    assert!(e(vec![raa(), t("trace")]).contains("output must be"));
    assert!(e(vec![raa(), Grid::missing(), n(0.0)]).contains("draws"));
    assert!(e(vec![raa(), Grid::missing(), Grid::missing(), n(9.0)]).contains("chains"));
    let mut a = vec![raa()];
    a.extend(vec![Grid::missing(); 6]);
    a.push(n(-1.0));
    assert!(e(a).contains("prior_sd"));
}
