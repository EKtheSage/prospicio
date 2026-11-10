//! Zero-inflated and hurdle counts, and quantile and expectile regression,
//! as Excel reaches them (`docs/design/xll.md` decision 55): every number
//! `prospicio-validation` checks against R pscl 1.5.9, quantreg 5.97 and
//! expectreg 0.54 (`validation/reference/zero_counts_pscl.csv`,
//! `quantile_expectile_r.csv`), read off the sheet functions' tables, to
//! the reference's own tolerances.

use crate::common;

use common::{fails, n, ok, t, validation};
use prospicio_xll::sheet::{Grid, Value};

/// `validation/data/<name>.csv` as a sheet table: its header, then its
/// rows of numbers.
fn table(name: &str) -> Grid {
    let text = validation(&format!("data/{name}"));
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let mut rows = vec![
        lines
            .next()
            .unwrap()
            .split(',')
            .map(Value::from)
            .collect::<Vec<_>>(),
    ];
    rows.extend(lines.map(|l| {
        l.split(',')
            .map(|v| Value::Number(v.parse().unwrap()))
            .collect()
    }));
    Grid::from_rows(rows)
}

fn names(names: &[&str]) -> Grid {
    Grid::from_rows(vec![names.iter().map(|&s| Value::from(s)).collect()])
}

fn none() -> Grid {
    Grid::missing()
}

/// The reference rows: case, quantity, arg, expected, abs_tol, rel_tol.
fn cases(name: &str) -> Vec<(String, String, String, f64, f64, f64)> {
    validation(&format!("reference/{name}"))
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("case,"))
        .map(|l| {
            let f: Vec<&str> = l.splitn(7, ',').collect();
            (
                f[0].to_owned(),
                f[1].to_owned(),
                f[2].to_owned(),
                f[3].parse().unwrap(),
                f[4].parse().unwrap(),
                f[5].parse().unwrap(),
            )
        })
        .collect()
}

fn close(what: &str, got: f64, want: f64, abs: f64, rel: f64) {
    assert!(
        (got - want).abs() <= abs + rel * want.abs(),
        "{what}: {got} against {want}"
    );
}

fn number(v: &Value) -> f64 {
    match v {
        Value::Number(x) => *x,
        other => panic!("expected a number, got {other:?}"),
    }
}

/// A design column's name from a result's Term and Level cells.
fn name_of(term: &Value, level: &Value) -> String {
    match (term, level) {
        (Value::Text(t), Value::Empty) => t.clone(),
        (Value::Text(t), Value::Text(l)) => format!("{t}[{l}]"),
        other => panic!("not a term: {other:?}"),
    }
}

/// The value next to a label in column 0.
fn labelled(out: &Grid, label: &str, column: usize) -> Option<f64> {
    (0..out.rows)
        .find(|&r| *out.get(r, 0) == Value::from(label))
        .map(|r| number(out.get(r, column)))
}

/// A two-part case's arguments, `data, response, terms, model, count,
/// zero_terms, zero_link, exposure, weights`.
fn two_part_args(case: &str, bio: &Grid, book: &Grid) -> Vec<Grid> {
    let all = names(&["fem", "mar", "kid5", "phd", "ment"]);
    let (data, response, terms, exposure) = if case.starts_with("bio") {
        (bio.clone(), t("art"), all, none())
    } else {
        (book.clone(), t("y"), names(&["x", "urban"]), t("exposure"))
    };
    let model = if case.contains("_h") {
        t("hurdle")
    } else {
        none()
    };
    let count = if case.contains("nb") {
        t("negative_binomial")
    } else {
        none()
    };
    let (zero_terms, zero_link) = match case {
        "bio_zip_probit" => (t("ment"), t("probit")),
        "bio_hnb_cloglog" => (names(&["kid5", "ment"]), t("cloglog")),
        c if c.starts_with("book") => (t("x"), none()),
        _ => (none(), none()),
    };
    let weights = if case == "book_zinb_weighted" {
        t("weight")
    } else {
        none()
    };
    vec![
        data, response, terms, model, count, zero_terms, zero_link, exposure, weights,
    ]
}

#[test]
fn zero_inflated_and_hurdle_counts_match_pscl() {
    let (bio, book) = (table("biochemists.csv"), table("zero_book.csv"));
    let mut checked = 0;
    let mut last: Option<(String, Grid, Grid, Grid, Grid)> = None;
    for (case, quantity, arg, want, abs, rel) in cases("zero_counts_pscl.csv") {
        if last.as_ref().is_none_or(|l| l.0 != case) {
            let args = two_part_args(&case, &bio, &book);
            let summary = ok("pro_prc_counts", args.clone());
            let predict = |output: &str| {
                let mut a = args[..3].to_vec();
                a.push(args[0].clone());
                a.extend_from_slice(&args[3..]);
                a.push(t(output));
                a.push(n(4.0));
                ok("pro_prc_counts_predict", a)
            };
            last = Some((
                case.clone(),
                summary,
                predict("mean"),
                predict("zero"),
                predict("probabilities"),
            ));
        }
        let (_, summary, mean, zero, probs) = last.as_ref().unwrap();
        let what = format!("{case} {quantity} {arg}");
        let quantity = quantity.trim_end_matches("_pscl");
        let row = || arg.parse::<usize>().unwrap() - 1;
        let got = match quantity {
            "coef" | "se" => {
                let (part, name) = match arg.strip_prefix("count_") {
                    Some(n) => ("Count", n),
                    None => ("Zero", arg.strip_prefix("zero_").unwrap()),
                };
                (1..summary.rows)
                    .find(|&r| {
                        *summary.get(r, 0) == Value::from(part)
                            && name_of(summary.get(r, 1), summary.get(r, 2)) == name
                    })
                    .map(|r| number(summary.get(r, if quantity == "coef" { 3 } else { 4 })))
            }
            "theta" => labelled(summary, "Theta", 1),
            "se_log_theta" => labelled(summary, "SE log(theta)", 1),
            "loglik" => labelled(summary, "Log-likelihood", 1),
            "aic" => labelled(summary, "AIC", 1),
            "bic" => labelled(summary, "BIC", 1),
            "mean" => Some(number(mean.get(row(), 0))),
            "prob_zero" => Some(number(zero.get(row(), 0))),
            "prob" => {
                let (i, k) = arg.split_once(':').unwrap();
                let (i, k): (usize, usize) = (i.parse().unwrap(), k.parse().unwrap());
                assert_eq!(number(probs.get(0, k)), k as f64);
                Some(number(probs.get(i, k)))
            }
            q @ ("vuong" | "vuong_p") => {
                let label = match arg.as_str() {
                    "raw" => "Raw",
                    "aic" => "AIC-corrected",
                    _ => "BIC-corrected",
                };
                labelled(summary, label, if q == "vuong" { 1 } else { 2 })
            }
            other => panic!("unknown quantity {other}"),
        };
        let got = got.unwrap_or_else(|| panic!("{what}: not on the sheet"));
        close(&what, got, want, abs, rel);
        checked += 1;
    }
    assert!(checked > 500, "{checked} values checked");
}

#[test]
fn the_counts_summary_lays_out_both_parts_then_statistics() {
    let bio = table("biochemists.csv");
    let mut args = two_part_args("bio_zip", &bio, &bio);
    args[5] = n(1.0);
    let out = ok("pro_prc_counts", args.clone());
    assert_eq!(*out.get(0, 0), Value::from("Part"));
    // Six count coefficients and the zero part's intercept alone.
    assert_eq!(*out.get(7, 0), Value::from("Zero"));
    assert_eq!(*out.get(7, 1), Value::from("(Intercept)"));
    assert!(out.get(8, 0).is_blank());
    assert!(labelled(&out, "Theta", 1).is_none());
    assert_eq!(labelled(&out, "Parameters", 1), Some(7.0));
    // Prior weights drop Vuong's test, which ignores them.
    assert!(labelled(&out, "Raw", 1).is_some());
    let mut w = args.clone();
    w[8] = t("phd");
    assert!(labelled(&ok("pro_prc_counts", w), "Raw", 1).is_none());
    let mut a = args;
    a[1] = t("phd");
    let msg = fails("pro_prc_counts", a);
    assert!(msg.contains("whole numbers"), "{msg}");
}

#[test]
fn counts_errors() {
    let bio = table("biochemists.csv");
    let args = two_part_args("bio_zip", &bio, &bio);
    for (i, value, message) in [
        (3, "tobit", "model must be"),
        (4, "binomial", "count must be"),
        (6, "log", "zero_link must be"),
    ] {
        let mut a = args.clone();
        a[i] = t(value);
        let msg = fails("pro_prc_counts", a);
        assert!(msg.contains(message), "{msg}");
    }
}

/// A quantile case's arguments, `data, response, terms, tau, method,
/// weights, bandwidth`.
fn linear_args(data: &str, tau: f64, method: &str, bofinger: bool) -> Vec<Grid> {
    let (table, response, terms, weights) = if data == "book" {
        (
            table("quantile_book.csv"),
            t("y"),
            names(&["x", "z"]),
            t("w"),
        )
    } else {
        (table("engel.csv"), t("foodexp"), t("income"), none())
    };
    vec![
        table,
        response,
        terms,
        n(tau),
        if method == "expectile" {
            t("expectile")
        } else {
            none()
        },
        weights,
        if bofinger { t("bofinger") } else { none() },
    ]
}

#[test]
fn quantile_and_expectile_regression_match_quantreg_and_expectreg() {
    let mut checked = 0;
    for (case, quantity, arg, want, abs, rel) in cases("quantile_expectile_r.csv") {
        let (data, tau) = case.rsplit_once('_').unwrap();
        // The P-spline expectile needs expectreg's basis: not on the sheet.
        // rq's "fn" method is the same vertex to its duality gap.
        if data == "engel_pspline" || quantity == "efitted" && data != "engel" {
            continue;
        }
        let tau: f64 = tau.parse().unwrap();
        let method = if quantity.starts_with('e') {
            "expectile"
        } else {
            "quantile"
        };
        let args = linear_args(data, tau, method, quantity == "se_bofinger");
        let what = format!("{case} {quantity} {arg}");
        let got = if quantity == "efitted" {
            let mut a = args[..4].to_vec();
            a.push(args[0].clone());
            a.extend_from_slice(&args[4..]);
            a.push(none());
            let out = ok("pro_prc_quantreg_predict", a);
            number(out.get(arg.parse::<usize>().unwrap() - 1, 0))
        } else {
            let out = ok("pro_prc_quantreg", args);
            assert_eq!(*out.get(0, 2), Value::Text(format!("Estimate {tau}")));
            match quantity.as_str() {
                "objective" => labelled(&out, "Objective", 2).unwrap(),
                q => {
                    let r = (1..out.rows)
                        .find(|&r| name_of(out.get(r, 0), out.get(r, 1)) == arg)
                        .unwrap_or_else(|| panic!("{what}: no row"));
                    let se = matches!(q, "se" | "se_bofinger" | "ese");
                    number(out.get(r, if se { 3 } else { 2 }))
                }
            }
        };
        close(&what, got, want, abs, rel);
        checked += 1;
    }
    assert!(checked > 60, "{checked} values checked");
}

#[test]
fn several_taus_side_by_side_and_rearranged() {
    let mut args = linear_args("engel", 0.0, "quantile", false);
    args[3] = Grid::column(&[0.9, 0.1, 0.5]);
    let out = ok("pro_prc_quantreg", args.clone());
    assert_eq!(out.columns, 8);
    assert_eq!(*out.get(0, 6), Value::from("Estimate 0.5"));
    assert!(labelled(&out, "Bandwidth", 2).unwrap() > 0.0);
    let single = |tau: f64| {
        let mut a = args.clone();
        a[3] = n(tau);
        ok("pro_prc_quantreg", a)
    };
    assert_eq!(out.get(2, 4), single(0.1).get(2, 2));
    let mut p = args[..4].to_vec();
    p.push(args[0].clone());
    p.extend_from_slice(&args[4..]);
    p.push(Grid::scalar(Value::Bool(true)));
    let rearranged = ok("pro_prc_quantreg_predict", p.clone());
    assert_eq!(rearranged.columns, 3);
    for r in 0..rearranged.rows {
        let row: Vec<f64> = (0..3).map(|c| number(rearranged.get(r, c))).collect();
        assert!(row[0] <= row[1] && row[1] <= row[2]);
    }
    // Expectiles take no bandwidth.
    let mut e = args.clone();
    e[4] = t("expectile");
    e[6] = t("bofinger");
    assert!(fails("pro_prc_quantreg", e).contains("bandwidth"));
    let mut bad = args;
    bad[3] = n(1.5);
    assert!(fails("pro_prc_quantreg", bad).contains("tau must be"));
}
