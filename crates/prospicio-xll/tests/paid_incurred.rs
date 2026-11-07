//! Paid and incurred developed together, reached through the catalogue
//! as Excel reaches them: the Munich chain ladder against R ChainLadder's
//! `MunichChainLadder` and chainladder-python's `MunichAdjustment` on
//! Quarg and Mack's example (`validation/reference/reserving_munich_*.csv`)
//! and R's printed totals, and Merz and Wüthrich's paid-incurred chain
//! against R ChainLadder's `PaidIncurredChain` on USAA and Dahms' data
//! (`validation/reference/reserving_paid_incurred_r.csv`) and the paper's
//! printed Hertig reserves, the same cases the crate's parity tests use.

mod common;

use std::collections::BTreeMap;

use common::{n, num, reference, t, validation};
use prospicio_xll::call as run;
use prospicio_xll::sheet::{Grid, Value};

/// The paid and incurred columns of `validation/data/<name>.csv` as two
/// sheet ranges, and the origin years as labels.
fn pair(name: &str) -> (Grid, Grid, Grid) {
    let mut rows: BTreeMap<i32, Vec<(f64, f64)>> = BTreeMap::new();
    let text = validation(&format!("data/{name}.csv"));
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let col = |n: &str| header.iter().position(|h| *h == n).unwrap();
    let (p, i) = (col("paid"), col("incurred"));
    for line in lines {
        let f: Vec<&str> = line.split(',').collect();
        rows.entry(f[0].parse().unwrap())
            .or_default()
            .push((f[p].parse().unwrap(), f[i].parse().unwrap()));
    }
    let ages = rows.values().map(Vec::len).max().unwrap();
    let grid = |pick: fn(&(f64, f64)) -> f64| {
        Grid::from_rows(
            rows.values()
                .map(|r| {
                    (0..ages)
                        .map(|c| r.get(c).map_or(Value::Empty, |x| Value::Number(pick(x))))
                        .collect()
                })
                .collect(),
        )
    };
    let labels: Vec<f64> = rows.keys().map(|&y| f64::from(y)).collect();
    (grid(|x| x.0), grid(|x| x.1), Grid::column(&labels))
}

/// A call that must succeed.
fn call(symbol: &str, args: Vec<Grid>) -> Result<Grid, String> {
    let g = run(symbol, args);
    match g.as_scalar() {
        Some(Value::Text(s)) if s.starts_with("#PROSPICIO!") => Err(s.clone()),
        _ => Ok(g),
    }
}

fn blank() -> Grid {
    Grid::scalar(Value::Empty)
}

fn close(got: f64, want: f64, abs: f64, rel: f64) -> bool {
    (got - want).abs() <= abs.max(rel * want.abs())
}

/// The settings of a Munich reference arm: the sigma, then the tails.
fn munich_args(test: &str, origins: Option<&Grid>) -> Vec<Grid> {
    let (paid, incurred, labels) = pair("mcl");
    let mut parts = test.split(";tail=");
    let sigma = t(parts.next().unwrap());
    let (tp, ti) = match parts.next() {
        None => (blank(), blank()),
        Some("log-linear") => (t("log-linear"), t("log-linear")),
        Some(constants) => {
            let (p, i) = constants.split_once('/').unwrap();
            (n(p.parse().unwrap()), n(i.parse().unwrap()))
        }
    };
    let mut args = vec![paid, incurred, sigma, tp, ti];
    if let Some(o) = origins {
        args.push(o.clone());
    } else {
        args.push(labels);
    }
    args
}

fn check_munich(file: &str) -> usize {
    let mut checked = 0;
    let mut cache: BTreeMap<String, (Grid, Grid)> = BTreeMap::new();
    for row in reference(file) {
        let (test, quantity, arg) = (&row[1], row[2].as_str(), &row[3]);
        let (summary, params) = cache.entry(test.clone()).or_insert_with(|| {
            let args = munich_args(test, None);
            (
                call("pro_res_munich", args.clone()).unwrap(),
                call("pro_res_munich_params", args[..5].to_vec()).unwrap(),
            )
        });
        let ages = params.rows - 3;
        let origin_row = || -> usize {
            let year: f64 = arg.parse().unwrap();
            (1..summary.rows)
                .find(|&r| *summary.get(r, 0) == Value::Number(year))
                .unwrap()
        };
        let age_row = || -> usize { arg.parse::<usize>().unwrap() / 12 };
        let got = match quantity {
            "lambda_paid" => num(params.get(ages + 1, 1)),
            "lambda_incurred" => num(params.get(ages + 2, 1)),
            "ultimate_paid" => num(summary.get(origin_row(), 4)),
            "ultimate_incurred" => num(summary.get(origin_row(), 5)),
            "q" => num(params.get(age_row(), 1)),
            "q_inverse" => num(params.get(age_row(), 2)),
            "rho_paid" => num(params.get(age_row(), 3)),
            "rho_incurred" => num(params.get(age_row(), 4)),
            "sigma_paid" => num(params.get(age_row(), 6)),
            "sigma_incurred" => num(params.get(age_row(), 8)),
            "tail_paid" => num(params.get(ages, 5)),
            "tail_sigma_paid" => num(params.get(ages, 6)),
            "tail_incurred" => num(params.get(ages, 7)),
            "tail_sigma_incurred" => num(params.get(ages, 8)),
            other => panic!("unknown quantity {other}"),
        };
        let (want, abs, rel) = (
            row[4].parse().unwrap(),
            row[5].parse().unwrap(),
            row[6].parse().unwrap(),
        );
        assert!(
            close(got, want, abs, rel),
            "{file} {test} {quantity} {arg}: {got} vs {want}"
        );
        checked += 1;
    }
    checked
}

#[test]
fn munich_matches_r_chainladder() {
    assert_eq!(check_munich("reserving_munich_r.csv"), 176);
}

#[test]
fn munich_matches_chainladder_python() {
    assert_eq!(check_munich("reserving_munich_python.csv"), 110);
}

#[test]
fn munich_totals_match_rs_printed_summary() {
    // R ChainLadder 0.2.21: summary(MunichChainLadder(MCLpaid,
    // MCLincurred))$Totals, which on this data falls back to Mack's sigma.
    let s = call("pro_res_munich", munich_args("mack", Some(&blank()))).unwrap();
    let total = s.rows - 1;
    assert_eq!(*s.get(total, 0), Value::from("Total"));
    assert!((num(s.get(total, 4)) - 32_121.496_994_989_8).abs() < 1e-8);
    assert!((num(s.get(total, 5)) - 32_719.512_501_042_9).abs() < 1e-8);
    assert!((num(s.get(total, 6)) - 0.981_722_970_168_516).abs() < 1e-12);
    // The youngest origin's P/I, 0.407 on the latest values, is drawn to
    // 0.980; the default origin labels are 1, 2, ...
    assert_eq!(*s.get(7, 0), Value::Number(7.0));
    assert!((num(s.get(7, 3)) - 0.407_009_159_697_332).abs() < 1e-12);
    assert!((num(s.get(7, 6)) - 0.980_301_200_225_115).abs() < 1e-12);
    // Reserve and IBNR are the ultimates less the latest values.
    for r in 1..s.rows {
        assert!((num(s.get(r, 7)) - (num(s.get(r, 4)) - num(s.get(r, 1)))).abs() < 1e-9);
        assert!((num(s.get(r, 8)) - (num(s.get(r, 5)) - num(s.get(r, 2)))).abs() < 1e-9);
    }
}

#[test]
fn munich_takes_a_sigma_for_each_column() {
    let both = munich_args("mack", None);
    let mut split = both.clone();
    split[2] = Grid::from_rows(vec![vec![Value::from("mack"), Value::from("Mack")]]);
    assert_eq!(
        call("pro_res_munich", both).unwrap(),
        call("pro_res_munich", split.clone()).unwrap()
    );
    split[2] = Grid::from_rows(vec![vec![Value::from("mack"), Value::from("log-linear")]]);
    let mixed = call("pro_res_munich", split).unwrap();
    let ll = call("pro_res_munich", munich_args("log-linear", None)).unwrap();
    assert_ne!(mixed, ll);
}

#[test]
fn munich_errors_name_the_problem() {
    let (paid, incurred, _) = pair("mcl");
    let err = |args: Vec<Grid>| call("pro_res_munich", args).unwrap_err();
    let rest = || vec![blank(), blank(), blank(), blank()];
    let mut a = vec![paid.clone(), common::raa()];
    a.extend(rest());
    assert!(err(a).contains("the same size"), "size");
    let mut shorter = incurred.clone();
    shorter.cells[shorter.columns - 1] = Value::Empty; // oldest origin's last cell
    let mut a = vec![paid.clone(), shorter];
    a.extend(rest());
    assert!(err(a).contains("the same cells"));
    let mut a = vec![paid, incurred];
    a.extend(rest());
    a[2] = t("chain");
    assert!(err(a).contains("sigma must be"));
}

/// R's arm of the PIC on `dataset`: its convention, everything else
/// estimated.
fn pic_args(dataset: &str, convention: &str) -> Vec<Grid> {
    let (paid, incurred, labels) = pair(dataset);
    let mut args = vec![paid, incurred];
    args.extend((0..3).map(|_| blank()));
    args.push(t(convention));
    args.extend((0..4).map(|_| blank()));
    args.push(labels);
    args
}

#[test]
fn pic_matches_r_chainladder() {
    let mut fits: BTreeMap<String, (Grid, Grid)> = BTreeMap::new();
    let mut checked = 0;
    for row in reference("reserving_paid_incurred_r.csv") {
        let (dataset, quantity, arg) = (row[0].as_str(), row[1].as_str(), &row[2]);
        let (summary, params) = fits.entry(dataset.to_owned()).or_insert_with(|| {
            let args = pic_args(dataset, "r");
            (
                call("pro_res_pic", args.clone()).unwrap(),
                call("pro_res_pic_params", args[..10].to_vec()).unwrap(),
            )
        });
        let periods = params.rows - 1;
        let period = || arg.parse::<usize>().unwrap() + 1;
        let origin = || {
            let year: f64 = arg.parse().unwrap();
            (1..summary.rows)
                .find(|&r| *summary.get(r, 0) == Value::Number(year))
                .unwrap()
        };
        let total = summary.rows - 1;
        let got = match quantity {
            "sigma" => num(params.get(period(), 1)),
            "tau" => num(params.get(period(), 2)),
            "phi" => num(params.get(period(), 3)),
            "phi_sd" => num(params.get(period(), 4)),
            "psi" => num(params.get(period(), 5)),
            "psi_sd" => num(params.get(period(), 6)),
            // R's beta_j belongs to the origin whose latest period is j.
            "beta" => num(summary.get(periods - arg.parse::<usize>().unwrap(), 3)),
            "ultimate" => num(summary.get(origin(), 4)),
            "reserve" => num(summary.get(origin(), 5)),
            "std_err" => num(summary.get(origin(), 6)),
            "ultimate_after_first" => num(summary.get(total, 4)) - num(summary.get(1, 4)),
            "total_reserve" => num(summary.get(total, 5)),
            "total_std_err" => num(summary.get(total, 6)),
            other => panic!("unknown quantity {other}"),
        };
        let (want, abs, rel) = (
            row[3].parse().unwrap(),
            row[4].parse().unwrap(),
            row[5].parse().unwrap(),
        );
        assert!(
            close(got, want, abs, rel),
            "{dataset} {quantity} {arg}: {got} vs {want}"
        );
        checked += 1;
    }
    assert_eq!(checked, 192);
}

#[test]
fn pic_hertig_matches_merz_and_wuthrichs_printed_reserves() {
    // Merz and Wüthrich (Variance 7(1), 2013), Table 5, "Hertig's model,
    // no tail factor", on Dahms' data: the PIC with the incurred side
    // switched off by a very large tau, Mack's rule for the last sigma.
    let mut args = pic_args("dahms", "model");
    args[3] = Grid::from_rows(vec![vec![Value::Number(1e4); 9]]);
    args[4] = t("mack");
    let s = call("pro_res_pic", args).unwrap();
    let printed = [
        0.0,
        47_060.0,
        336_189.0,
        549_682.0,
        655_906.0,
        1_190_955.0,
        1_115_656.0,
        1_611_611.0,
        2_310_950.0,
        1_954_075.0,
    ];
    for (o, want) in printed.iter().enumerate() {
        let got = num(s.get(o + 1, 5));
        assert!((got - want).abs() <= 1.0, "origin {o}: {got} vs {want}");
    }
}

#[test]
fn pic_priors_come_in_pairs_and_a_tight_prior_holds_its_mean() {
    let mut args = pic_args("usaa", "model");
    args[6] = Grid::from_rows(vec![vec![Value::Number(0.1); 10]]);
    let err = call("pro_res_pic", args.clone()).unwrap_err();
    assert!(err.contains("phi_mean and phi_sd"), "{err}");
    // A tight prior on Phi_9 pulls its posterior mean onto the prior's.
    let mut means = vec![Value::Empty; 10];
    let mut sds = vec![Value::Number(1e6); 10];
    means[9] = Value::Number(0.0005);
    sds[9] = Value::Number(1e-9);
    // Blank cells are skipped, so give every mean.
    for m in means.iter_mut().take(9) {
        *m = Value::Number(0.0);
    }
    args[6] = Grid::from_rows(vec![means]);
    args[7] = Grid::from_rows(vec![sds]);
    let p = call("pro_res_pic_params", args[..10].to_vec()).unwrap();
    assert!(
        (num(p.get(10, 3)) - 0.0005).abs() < 1e-8,
        "{:?}",
        p.get(10, 3)
    );
    // The model's convention differs from R's by about 0.5% on USAA.
    let model = call("pro_res_pic", pic_args("usaa", "model")).unwrap();
    let r = call("pro_res_pic", pic_args("usaa", "r")).unwrap();
    let (a, b) = (num(model.get(11, 5)), num(r.get(11, 5)));
    assert!(a != b && ((a - b) / b).abs() < 0.01, "{a} vs {b}");
}
