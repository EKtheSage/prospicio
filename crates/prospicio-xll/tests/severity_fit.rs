//! `PRO.SEV.FIT`, reached through the catalogue as Excel reaches it:
//!
//! * the Pareto families against the R package Pareto 2.4.5
//!   (`validation/reference/pareto_fit_r.csv`: reporting thresholds,
//!   censoring, weights and truncation), at its tolerances;
//! * the body families on their closed forms untruncated, and equal to
//!   `fit_severity` with thresholds and censoring;
//! * a fit's parameters fed to `PRO.SEV.CDF`, the goodness-of-fit table
//!   and the comparison of the body families.

use crate::common;

use common::{n, num, reference, t, text};
use prospicio_prob::severity_fit::{Family, fit_severity};
use prospicio_prob::{Distribution, LargeLosses};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

/// The losses of `validation/scripts/r_pareto_fit.R`.
const LOSSES: [f64; 14] = [
    1100.0, 1300.0, 1750.0, 2000.0, 2600.0, 3500.0, 4100.0, 5200.0, 7000.0, 9000.0, 12000.0,
    18000.0, 25000.0, 40000.0,
];

fn column(x: &[f64]) -> Grid {
    Grid::column(x)
}

fn missing() -> Grid {
    Grid::missing()
}

/// The estimate of the parameter in row `row` of a `"params"` table.
fn estimate(out: &Grid, label: &str) -> f64 {
    let r = (0..out.rows)
        .find(|&r| out.get(r, 0) == &Value::from(label))
        .unwrap_or_else(|| panic!("no row {label} in {out:?}"));
    num(out.get(r, 1))
}

#[test]
fn pareto_fits_match_r() {
    let mut checked = 0;
    for row in reference("pareto_fit_r.csv") {
        let (model, spec) = (row[0].as_str(), row[1].as_str());
        if model == "model" {
            continue;
        }
        let fields: std::collections::HashMap<&str, &str> = spec
            .split(';')
            .filter_map(|kv| kv.split_once('='))
            .collect();
        let list = |k: &str| -> Vec<f64> {
            match fields.get(k).copied().unwrap_or("") {
                "" => vec![],
                v => v.split('|').map(|x| x.parse().unwrap()).collect(),
            }
        };
        let or_missing = |v: Vec<f64>| if v.is_empty() { missing() } else { column(&v) };
        let (r, c, w, thresholds) = (list("r"), list("c"), list("w"), list("t"));
        let censored = if c.is_empty() {
            missing()
        } else {
            Grid::from_rows(c.iter().map(|&x| vec![Value::Bool(x == 1.0)]).collect())
        };
        let truncation = match (fields.get("truncation"), fields.get("type")) {
            (None, _) => missing(),
            (Some(v), None) => n(v.parse().unwrap()),
            (Some(v), Some(kind)) => Grid::from_rows(vec![vec![
                Value::Number(v.parse().unwrap()),
                Value::from(*kind),
            ]]),
        };
        let t_arg = if thresholds.len() == 1 {
            n(thresholds[0])
        } else {
            Grid::row(&thresholds)
        };
        let out = call(
            "pro_sev_fit",
            vec![
                column(&LOSSES),
                t(model),
                t("params"),
                t_arg,
                truncation,
                or_missing(r),
                censored,
                or_missing(w),
            ],
        );
        let index: usize = row[2].parse().unwrap();
        let got = match model {
            "pareto" => estimate(&out, "alpha"),
            "gen_pareto" => estimate(&out, ["alpha_ini", "alpha_tail"][index]),
            _ => estimate(&out, &format!("alpha above {}", thresholds[index])),
        };
        let (want, abs, rel): (f64, f64, f64) = (
            row[3].parse().unwrap(),
            row[4].parse().unwrap(),
            row[5].parse().unwrap(),
        );
        assert!(
            (got - want).abs() <= abs.max(rel * want.abs()),
            "{model} {spec} [{index}]: {got} against R's {want}"
        );
        checked += 1;
    }
    assert!(checked >= 30, "{checked}");
}

#[test]
fn the_lognormal_is_the_mean_and_sd_of_the_logs() {
    let out = call("pro_sev_fit", vec![column(&LOSSES), t("lognormal")]);
    let logs: Vec<f64> = LOSSES.iter().map(|x| x.ln()).collect();
    let k = logs.len() as f64;
    let mu = logs.iter().sum::<f64>() / k;
    let sd = (logs.iter().map(|l| (l - mu).powi(2)).sum::<f64>() / k).sqrt();
    assert!((estimate(&out, "meanlog") - mu).abs() < 1e-7);
    assert!((estimate(&out, "sdlog") / sd - 1.0).abs() < 1e-7);
    // The standard errors sd / sqrt(n) and sd / sqrt(2n), in the third column.
    assert!((num(out.get(1, 2)) / (sd / k.sqrt()) - 1.0).abs() < 1e-4);
    assert!((num(out.get(2, 2)) / (sd / (2.0 * k).sqrt()) - 1.0).abs() < 1e-4);
    assert_eq!(estimate(&out, "Losses"), 14.0);
    assert!((estimate(&out, "AIC") - (4.0 - 2.0 * estimate(&out, "Log-likelihood"))).abs() < 1e-9);
    assert_eq!(out.get(out.rows - 2, 1), &Value::Bool(true));
    assert_eq!(out.get(out.rows - 1, 1), &Value::from("none"));
}

#[test]
fn body_fits_with_thresholds_and_censoring_are_fit_severity() {
    let r: Vec<f64> = LOSSES
        .iter()
        .enumerate()
        .map(|(i, _)| if i % 3 == 0 { 1000.0 } else { 0.0 })
        .collect();
    let c: Vec<bool> = LOSSES.iter().map(|&x| x >= 18_000.0).collect();
    let w: Vec<f64> = (0..14).map(|i| 1.0 + (i % 2) as f64).collect();
    let data = LargeLosses::new(LOSSES.to_vec())
        .unwrap()
        .reporting_thresholds(r.clone())
        .unwrap()
        .censored(c.clone())
        .unwrap()
        .weights(w.clone())
        .unwrap();
    for family in [
        Family::Lognormal,
        Family::Gamma,
        Family::Weibull,
        Family::Burr,
    ] {
        let want = fit_severity(family, &data).unwrap();
        let out = call(
            "pro_sev_fit",
            vec![
                column(&LOSSES),
                t(family.name()),
                missing(),
                missing(),
                missing(),
                column(&r),
                Grid::from_rows(c.iter().map(|&b| vec![Value::Bool(b)]).collect()),
                column(&w),
            ],
        );
        for (i, name) in family.parameter_names().iter().enumerate() {
            assert_eq!(
                estimate(&out, name),
                want.parameters()[i],
                "{family:?} {name}"
            );
            assert_eq!(out.get(1 + i, 2), &Value::number(want.standard_errors()[i]));
        }
        assert_eq!(estimate(&out, "Log-likelihood"), want.log_likelihood());
        assert_eq!(estimate(&out, "BIC"), want.bic());
    }
}

#[test]
fn a_fit_feeds_the_severity_functions() {
    let fit = call("pro_sev_fit", vec![column(&LOSSES), t("burr")]);
    let p: Vec<Grid> = (1..4)
        .map(|r| Grid::scalar(fit.get(r, 1).clone()))
        .collect();
    let data = LargeLosses::new(LOSSES.to_vec()).unwrap();
    let want = fit_severity(Family::Burr, &data).unwrap();
    for x in [1500.0, 10_000.0] {
        let cdf = call(
            "pro_sev_cdf",
            vec![n(x), t("burr"), p[0].clone(), p[1].clone(), p[2].clone()],
        );
        assert!((num(&cdf.cells[0]) - want.dist().cdf(x)).abs() < 1e-14);
    }
    // A Pareto's t and alpha are p1 and p2.
    let fit = call(
        "pro_sev_fit",
        vec![column(&LOSSES), t("pareto"), missing(), n(1000.0)],
    );
    let cdf = call(
        "pro_sev_cdf",
        vec![
            n(5000.0),
            t("pareto"),
            Grid::scalar(fit.get(1, 1).clone()),
            Grid::scalar(fit.get(2, 1).clone()),
        ],
    );
    let alpha = estimate(&fit, "alpha");
    assert!((num(&cdf.cells[0]) - (1.0 - 0.2f64.powf(alpha))).abs() < 1e-14);
}

#[test]
fn gof_compares_with_the_empirical_survival() {
    let out = call(
        "pro_sev_fit",
        vec![column(&LOSSES), t("lognormal"), t("gof")],
    );
    assert_eq!((out.rows, out.columns), (15, 6));
    assert_eq!(out.get(0, 3), &Value::from("Empirical survival"));
    // Uncensored and untruncated: the empirical survival steps by 1/14.
    for (i, &loss) in LOSSES.iter().enumerate() {
        assert_eq!(num(out.get(i + 1, 0)), loss);
        assert!((num(out.get(i + 1, 3)) - (13 - i) as f64 / 14.0).abs() < 1e-12);
    }
    let params = call("pro_sev_fit", vec![column(&LOSSES), t("lognormal")]);
    let ks = (1..15)
        .map(|r| {
            let m = num(out.get(r, 4));
            let before = if r == 1 { 1.0 } else { num(out.get(r - 1, 3)) };
            (num(out.get(r, 3)) - m).abs().max((before - m).abs())
        })
        .fold(0.0, f64::max);
    assert!((estimate(&params, "KS distance") - ks).abs() < 1e-12);
}

#[test]
fn compare_fits_every_body_family() {
    let out = call(
        "pro_sev_fit",
        vec![column(&LOSSES), missing(), t("compare")],
    );
    assert_eq!((out.rows, out.columns), (8, 13));
    let names: Vec<&Value> = (1..8).map(|r| out.get(r, 0)).collect();
    assert_eq!(
        names,
        [
            "lognormal",
            "gamma",
            "weibull",
            "loglogistic",
            "exponential",
            "burr",
            "lomax"
        ]
        .map(Value::from)
        .iter()
        .collect::<Vec<_>>()
    );
    let lognormal = call("pro_sev_fit", vec![column(&LOSSES), t("lognormal")]);
    assert_eq!(out.get(1, 1), lognormal.get(1, 1));
    assert_eq!(out.get(1, 3), &Value::Empty);
    assert_eq!(out.get(1, 5), &Value::Number(estimate(&lognormal, "AIC")));
    // The exponential's mean is the losses' mean, untruncated.
    let mean = LOSSES.iter().sum::<f64>() / LOSSES.len() as f64;
    assert!((num(out.get(5, 1)) - mean).abs() < 1e-9 * mean);
    assert_eq!(out.get(5, 2), &Value::Empty);
    // On these Pareto-like losses the Burr runs to its Pareto limit, with
    // the highest likelihood: the note says so.
    assert!(
        matches!(out.get(6, 12), Value::Text(s) if s.contains("Pareto limit")),
        "{:?}",
        out.get(6, 12)
    );
    assert!(num(out.get(6, 4)) > num(out.get(1, 4)));
}

/// `validation/data/severity_fits.csv`'s 89 `plain` losses, all reported
/// above 150,000.
fn plain_losses() -> Vec<f64> {
    common::validation("data/severity_fits.csv")
        .lines()
        .skip(1)
        .filter(|l| l.starts_with("plain,"))
        .map(|l| l.split(',').nth(1).unwrap().parse().unwrap())
        .collect()
}

/// SciPy's log-likelihood of each family on the `plain` losses.
fn scipy_log_likelihood(family: &str) -> Option<f64> {
    reference("severity_fits_scipy.csv")
        .into_iter()
        .find(|r| r[0] == family && r[1] == "plain" && r[2] == "log_likelihood")
        .map(|r| r[3].parse().unwrap())
}

#[test]
fn compare_with_t_ranks_nine_families_by_aic() {
    let y = plain_losses();
    assert_eq!(y.len(), 89);
    let args = |family: Grid, t: Grid| {
        vec![
            column(&y),
            family,
            common::t("compare"),
            t,
            missing(),
            n(150_000.0),
        ]
    };
    let out = call("pro_sev_fit", args(missing(), n(150_000.0)));
    assert_eq!((out.rows, out.columns), (10, 13));
    let mut names: Vec<String> = (1..10).map(|r| text_cell(out.get(r, 0))).collect();
    for r in 1..10 {
        let family = text_cell(out.get(r, 0));
        let ll = num(out.get(r, 4));
        // The issue's tolerance; SciPy has no Burr here.
        if let Some(expected) = scipy_log_likelihood(&family) {
            assert!((ll - expected).abs() < 1e-3, "{family}: {ll} against {expected}");
        }
        if r > 1 {
            assert!(num(out.get(r, 5)) >= num(out.get(r - 1, 5)), "{out:?}");
        }
    }
    // The Pareto families carry t as p1, as PRO.SEV.CDF takes them.
    let row = |name: &str| (1..10).find(|&r| out.get(r, 0) == &Value::from(name)).unwrap();
    assert_eq!(out.get(row("pareto"), 1), &Value::Number(150_000.0));
    assert_eq!(out.get(row("pareto"), 3), &Value::Empty);
    assert_eq!(out.get(row("pareto"), 12), &Value::from("none"));
    assert_eq!(out.get(row("gen_pareto"), 1), &Value::Number(150_000.0));
    // Riegel's GPD from t and the Lomax conditioned on t reach the same
    // likelihood.
    assert!((num(out.get(row("gen_pareto"), 4)) - num(out.get(row("lomax"), 4))).abs() < 1e-3);
    names.sort();
    let mut all: Vec<String> = Family::ALL.iter().map(|f| f.name().to_owned()).collect();
    all.sort();
    assert_eq!(names, all);

    // "params" scores the Pareto families on the same terms.
    let params = call(
        "pro_sev_fit",
        vec![
            column(&y),
            common::t("pareto"),
            missing(),
            n(150_000.0),
            missing(),
            n(150_000.0),
        ],
    );
    assert!((estimate(&params, "Log-likelihood") - num(out.get(row("pareto"), 4))).abs() < 1e-9);
    assert!((estimate(&params, "AIC") - num(out.get(row("pareto"), 5))).abs() < 1e-9);
    assert!((estimate(&params, "BIC") - num(out.get(row("pareto"), 6))).abs() < 1e-9);
    let riegel = call(
        "pro_sev_fit",
        vec![
            column(&y),
            common::t("gen_pareto"),
            missing(),
            n(150_000.0),
            missing(),
            n(150_000.0),
        ],
    );
    let expected = scipy_log_likelihood("gen_pareto").unwrap();
    assert!((estimate(&riegel, "Log-likelihood") - expected).abs() < 1e-3);
    assert!(estimate(&riegel, "BIC") > estimate(&riegel, "AIC"));

    // A range of names restricts the rows, still ranked; blank cells skip.
    let pick = Grid::from_rows(vec![
        vec![Value::from("Lomax")],
        vec![Value::Empty],
        vec![Value::from("pareto")],
        vec![Value::from("exponential")],
    ]);
    let some = call("pro_sev_fit", args(pick, n(150_000.0)));
    assert_eq!((some.rows, some.columns), (4, 13));
    let mut picked: Vec<String> = (1..4).map(|r| text_cell(some.get(r, 0))).collect();
    let ll: Vec<f64> = (1..4).map(|r| num(some.get(r, 5))).collect();
    assert!(ll.windows(2).all(|w| w[0] <= w[1]), "{some:?}");
    for r in 1..4 {
        let full = row(&text_cell(some.get(r, 0)));
        assert_eq!(some.get(r, 4), out.get(full, 4));
    }
    picked.sort();
    assert_eq!(picked, ["exponential", "lomax", "pareto"]);

    // Without t a range of body families keeps Family::ALL's order.
    let body = call(
        "pro_sev_fit",
        args(
            Grid::from_rows(vec![vec![Value::from("lomax"), Value::from("gamma")]]),
            missing(),
        ),
    );
    assert_eq!(body.get(1, 0), &Value::from("gamma"));
    assert_eq!(body.get(2, 0), &Value::from("lomax"));
}

/// SciPy's value of `quantity` for `family`'s fit to the `plain` losses
/// (`fit_tails_scipy.csv`), with its absolute and relative tolerances.
fn scipy_tail(family: &str, quantity: &str) -> (f64, f64, f64) {
    let r = reference("fit_tails_scipy.csv")
        .into_iter()
        .find(|r| r[0] == family && r[1] == quantity)
        .unwrap_or_else(|| panic!("no {family} {quantity}"));
    (r[2].parse().unwrap(), r[3].parse().unwrap(), r[4].parse().unwrap())
}

#[test]
fn compare_and_params_give_the_ks_p_value_and_tail_moments() {
    let y = plain_losses();
    let out = call(
        "pro_sev_fit",
        vec![
            column(&y),
            missing(),
            common::t("compare"),
            n(150_000.0),
            missing(),
            n(150_000.0),
        ],
    );
    let header: Vec<String> = (0..13).map(|c| text_cell(out.get(0, c))).collect();
    assert_eq!(
        header[7..11],
        ["KS distance", "KS p", "Tail mean", "Tail CV"].map(String::from)
    );
    for r in 1..10 {
        let family = text_cell(out.get(r, 0));
        if family == "burr" {
            continue;
        }
        for (c, q) in [(7, "ks_distance"), (8, "ks_p_value"), (9, "tail_mean"), (10, "tail_cv")] {
            let (want, abs, rel) = scipy_tail(&family, q);
            match out.get(r, c) {
                Value::Text(s) if s == "inf" => assert_eq!(want, f64::INFINITY, "{family} {q}"),
                v => {
                    let got = num(v);
                    let err = (got - want).abs();
                    assert!(err <= abs || err <= rel * want.abs(), "{family} {q}: {got} against {want}");
                }
            }
        }
    }
    // A Pareto's alpha below 2: a tail mean, an infinite CV.
    let pareto = (1..10)
        .find(|&r| out.get(r, 0) == &Value::from("pareto"))
        .unwrap();
    assert_eq!(out.get(pareto, 10), &Value::from("inf"));
    // "params" has the same cells as rows.
    let params = call(
        "pro_sev_fit",
        vec![
            column(&y),
            common::t("pareto"),
            missing(),
            n(150_000.0),
            missing(),
            n(150_000.0),
        ],
    );
    let cell = |label: &str| {
        let r = (0..params.rows)
            .find(|&r| params.get(r, 0) == &Value::from(label))
            .unwrap();
        params.get(r, 1).clone()
    };
    for (label, c) in [("KS distance", 7), ("KS p", 8), ("Tail mean", 9), ("Tail CV", 10)] {
        assert_eq!(&cell(label), out.get(pareto, c), "{label}");
    }
}

#[test]
fn compare_names_what_it_cannot_rank() {
    let y = plain_losses();
    let with = |family: Grid, t: Grid, thresholds: Grid, upper: Grid| {
        let mut a = vec![
            column(&y),
            family,
            common::t("compare"),
            t,
            missing(),
            thresholds,
        ];
        a.extend([missing(), missing(), upper]);
        text(&call("pro_sev_fit", a)).to_owned()
    };
    // A threshold below t would condition the Pareto families on t and
    // the body families on less.
    let e = with(missing(), n(150_000.0), n(100_000.0), missing());
    assert!(e.contains("every reporting threshold at least t"), "{e}");
    let e = with(missing(), n(150_000.0), missing(), missing());
    assert!(e.contains("every reporting threshold at least t"), "{e}");
    let e = with(common::t("pareto"), missing(), n(150_000.0), missing());
    assert!(e.contains("only from a threshold t"), "{e}");
    let e = with(common::t("piecewise_pareto"), n(150_000.0), n(150_000.0), missing());
    assert!(e.contains("a range of the names"), "{e}");
    let e = with(missing(), n(150_000.0), n(150_000.0), n(1e9));
    assert!(e.contains("takes no truncation from above"), "{e}");
    let mut a = vec![column(&y), missing(), common::t("compare"), missing(), n(1e6)];
    a.push(n(150_000.0));
    let e = call("pro_sev_fit", a);
    assert!(text(&e).contains("takes no truncation"), "{}", text(&e));
}

/// SciPy's survival of `family` conditional on 150,000 at a row of the
/// sorted `plain` losses (`fit_curves_scipy.csv`).
fn scipy_curve(family: &str, row: usize) -> (f64, f64) {
    let r = reference("fit_curves_scipy.csv")
        .into_iter()
        .find(|r| r[0] == family && r[1] == row.to_string())
        .unwrap_or_else(|| panic!("no {family} row {row}"));
    (r[3].parse().unwrap(), r[4].parse().unwrap())
}

#[test]
fn curves_set_every_fit_beside_the_plotting_positions() {
    let y = plain_losses();
    let args = |family: Grid, points: Grid| {
        vec![
            column(&y),
            family,
            common::t("curves"),
            n(150_000.0),
            missing(),
            n(150_000.0),
            missing(),
            missing(),
            missing(),
            points,
        ]
    };
    let out = call("pro_sev_fit", args(missing(), missing()));
    // 89 losses, under 200 points: every loss, and all nine families.
    assert_eq!((out.rows, out.columns), (90, 11));
    assert_eq!(out.get(0, 0), &Value::from("Loss"));
    assert_eq!(out.get(0, 1), &Value::from("Empirical"));
    // The families in the comparison's order, by AIC.
    let compare = call(
        "pro_sev_fit",
        vec![
            column(&y),
            missing(),
            common::t("compare"),
            n(150_000.0),
            missing(),
            n(150_000.0),
        ],
    );
    for c in 2..11 {
        assert_eq!(out.get(0, c), compare.get(c - 1, 0));
    }
    for row in [1, 2, 45, 89] {
        let (e, _) = scipy_curve("empirical", row);
        assert!((num(out.get(row, 1)) - e).abs() < 1e-15);
        for c in 2..11 {
            let family = text_cell(out.get(0, c));
            if family == "burr" {
                continue;
            }
            let (s, tol) = scipy_curve(&family, row);
            let got = num(out.get(row, c));
            assert!((got - s).abs() <= tol, "{family} row {row}: {got} against {s}");
        }
    }
    // Thinned to 10 rows: numpy's rows, the plotting positions of all 89.
    let thin = call("pro_sev_fit", args(common::t("lognormal"), n(10.0)));
    assert_eq!((thin.rows, thin.columns), (11, 3));
    let kept = [1, 11, 21, 30, 40, 50, 60, 69, 79, 89];
    for (k, &row) in kept.iter().enumerate() {
        assert_eq!(thin.get(k + 1, 0), out.get(row, 0));
        assert_eq!(thin.get(k + 1, 1), out.get(row, 1));
    }
    let e = call("pro_sev_fit", args(missing(), n(1.0)));
    assert!(text(&e).contains("points must be"), "{}", text(&e));
    let e = call(
        "pro_sev_fit",
        vec![
            column(&y),
            common::t("lognormal"),
            missing(),
            missing(),
            missing(),
            missing(),
            missing(),
            missing(),
            missing(),
            n(10.0),
        ],
    );
    assert!(text(&e).contains("points is taken only by"), "{}", text(&e));
    // With t blank, the body families from the lowest reporting threshold.
    let body = call(
        "pro_sev_fit",
        vec![column(&LOSSES), missing(), common::t("curves")],
    );
    assert_eq!((body.rows, body.columns), (15, 9));
    assert_eq!(body.get(1, 0), &Value::Number(1100.0));
    assert!((num(body.get(1, 1)) - 13.5 / 14.0).abs() < 1e-15);
}

#[test]
fn the_exponential_is_a_severity_by_its_mean() {
    let cdf = call(
        "pro_sev_cdf",
        vec![n(1000.0), common::t("exponential"), n(2000.0)],
    );
    assert!((num(cdf.get(0, 0)) - (1.0 - (-0.5f64).exp())).abs() < 1e-15);
    let fit = call("pro_sev_fit", vec![column(&LOSSES), common::t("exponential")]);
    let mean = LOSSES.iter().sum::<f64>() / LOSSES.len() as f64;
    assert!((estimate(&fit, "mean") - mean).abs() < 1e-9 * mean);
    let e = call(
        "pro_sev_cdf",
        vec![n(1000.0), common::t("exponential"), n(2000.0), n(3.0)],
    );
    assert!(text(&e).contains("leave p2 blank"), "{}", text(&e));
}

fn text_cell(v: &Value) -> String {
    match v {
        Value::Text(s) => s.clone(),
        other => panic!("expected text, got {other:?}"),
    }
}

#[test]
fn blank_cells_are_skipped_and_bad_inputs_named() {
    let mut cells: Vec<f64> = LOSSES.to_vec();
    cells.push(f64::NAN);
    let with_blank = Grid::column(&cells); // NaN becomes a blank cell.
    let a = call("pro_sev_fit", vec![with_blank, t("gamma")]);
    let b = call("pro_sev_fit", vec![column(&LOSSES), t("gamma")]);
    assert_eq!(a, b);
    let e = call(
        "pro_sev_fit",
        vec![column(&LOSSES), t("gamma"), missing(), n(1000.0)],
    );
    assert!(
        text(&e).contains("t is taken only by the Pareto families"),
        "{}",
        text(&e)
    );
    let e = call(
        "pro_sev_fit",
        vec![
            column(&LOSSES),
            t("lognormal"),
            missing(),
            missing(),
            missing(),
            column(&[0.0, 0.0]),
        ],
    );
    assert!(text(&e).contains("thresholds (2 x 1)"), "{}", text(&e));
    let e = call(
        "pro_sev_fit",
        vec![column(&LOSSES), t("pareto"), missing(), n(2000.0)],
    );
    assert!(text(&e).starts_with("#PROSPICIO!"), "{}", text(&e));
    let e = call("pro_sev_fit", vec![column(&LOSSES), t("normal")]);
    assert!(text(&e).contains("family must be"), "{}", text(&e));
}

/// The 40 untruncated losses of `validation/scripts/scipy_ks_bootstrap.py`.
fn bootstrap_losses() -> Vec<f64> {
    common::validation("data/ks_bootstrap.csv")
        .lines()
        .skip(1)
        .map(|l| l.parse().unwrap())
        .collect()
}

/// A row of family names, as `family` takes them.
fn names(families: &[&str]) -> Grid {
    Grid::from_rows(vec![families.iter().map(|&f| Value::from(f)).collect()])
}

#[test]
fn ks_bootstrap_matches_scipy_and_the_core() {
    // SciPy's goodness_of_fit(statistic="ks") on the same 40 losses: the
    // bootstrap p-value within four standard errors of the two Monte Carlo
    // estimates, and the core's cells at the same seed.
    let y = bootstrap_losses();
    // The core's own test checks all five families SciPy fits; here the two
    // cheapest show the wiring, in Family::ALL's order.
    let families = ["exponential", "lognormal"];
    let out = call(
        "pro_sev_fit_ks",
        vec![
            column(&y),
            names(&families),
            missing(),
            missing(),
            missing(),
            missing(),
            missing(),
            missing(),
            n(9_999.0),
            n(7.0),
        ],
    );
    let header: Vec<String> = (0..6).map(|c| text_cell(out.get(0, c))).collect();
    assert_eq!(
        header,
        ["Family", "KS distance", "KS p", "Bootstrap p", "Replicates", "Failed"].map(String::from)
    );
    let data = LargeLosses::new(y).unwrap();
    let scipy = reference("ks_bootstrap_scipy.csv");
    let want = |family: &str, q: &str| -> (f64, f64) {
        let r = scipy.iter().find(|r| r[0] == family && r[1] == q).unwrap();
        (r[2].parse().unwrap(), r[3].parse().unwrap())
    };
    for (r, family) in ["lognormal", "exponential"]
        .into_iter()
        .enumerate()
        .map(|(i, f)| (i + 1, f))
    {
        assert_eq!(text_cell(out.get(r, 0)), family);
        let (d, d_tol) = want(family, "ks_distance");
        assert!((num(out.get(r, 1)) - d).abs() <= d_tol, "{family} distance");
        let (p, p_tol) = want(family, "ks_bootstrap_p_value");
        assert!((num(out.get(r, 3)) - p).abs() <= p_tol, "{family} p {}", num(out.get(r, 3)));
        assert_eq!(num(out.get(r, 4)) + num(out.get(r, 5)), 9_999.0);
        // The core at the same seed.
        let fit = fit_severity(Family::from_name(family).unwrap(), &data).unwrap();
        let core = prospicio_prob::severity_fit::ks_bootstrap(&fit, &data, None, 9_999, 7).unwrap();
        let classical = prospicio_prob::severity_fit::goodness_of_fit(fit.dist(), &data)
            .unwrap()
            .ks_p_value;
        assert_eq!(num(out.get(r, 1)), core.max_distance);
        assert_eq!(num(out.get(r, 2)), classical);
        assert_eq!(num(out.get(r, 3)), core.p_value);
        assert_eq!(num(out.get(r, 4)), core.distances.len() as f64);
        assert_eq!(num(out.get(r, 5)), 0.0);
    }
}

#[test]
fn ks_bootstrap_with_t_follows_compare_and_takes_limits() {
    let y = plain_losses();
    let args = |family: Grid, limits: Grid, replicates: f64| {
        vec![
            column(&y),
            family,
            n(150_000.0),
            n(150_000.0),
            missing(),
            missing(),
            missing(),
            limits,
            n(replicates),
            n(3.0),
        ]
    };
    let out = call("pro_sev_fit_ks", args(missing(), missing(), 49.0));
    let compare = call(
        "pro_sev_fit",
        vec![column(&y), missing(), t("compare"), n(150_000.0), missing(), n(150_000.0)],
    );
    // Nine families, in "compare"'s AIC order, each bootstrapped.
    assert_eq!(out.rows, 10);
    for r in 1..10 {
        assert_eq!(text_cell(out.get(r, 0)), text_cell(compare.get(r, 0)));
        assert_eq!(num(out.get(r, 1)), num(compare.get(r, 7)));
        assert_eq!(num(out.get(r, 2)), num(compare.get(r, 8)));
        let p = num(out.get(r, 3));
        assert!(p > 0.0 && p <= 1.0);
        assert_eq!(num(out.get(r, 4)) + num(out.get(r, 5)), 49.0);
    }
    // One limit for every loss: each replicate is censored there, as the
    // core does with that limit per loss.
    let data = LargeLosses::new(y.clone())
        .unwrap()
        .reporting_thresholds(vec![150_000.0; y.len()])
        .unwrap();
    let limited = call("pro_sev_fit_ks", args(t("lognormal"), n(2.0e6), 49.0));
    let fit = fit_severity(Family::Lognormal, &data).unwrap();
    let core =
        prospicio_prob::severity_fit::ks_bootstrap(&fit, &data, Some(&vec![2.0e6; y.len()]), 49, 3)
            .unwrap();
    assert_eq!(num(limited.get(1, 3)), core.p_value);
    // A limit below a loss is that family's error; a bad count is the call's.
    let low = call("pro_sev_fit_ks", args(t("lognormal"), n(200_000.0), 49.0));
    assert!(text_cell(low.get(1, 1)).contains("limit"), "{low:?}");
    for bad in [0.0, 1.5, 1.0e6] {
        let e = call("pro_sev_fit_ks", args(t("lognormal"), missing(), bad));
        assert!(text(&e).contains("replicates"), "{e:?}");
    }
}
