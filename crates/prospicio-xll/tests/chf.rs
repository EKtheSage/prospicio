//! `PRO.CAP.CHF`, `PRO.CAP.STABLE` and `PRO.CAP.STABLE.PARETO` (#779,
//! #780; `docs/design/xll.md`, decision 67), reached through the catalogue
//! as Excel reaches them, against the `aggregate` 2.1.0 references the
//! Probability lane's tests check `prospicio_prob::chf` with
//! (`validation/reference/chf_*_aggregate.csv`, `chf_tvar_aggregate.csv`
//! among them, and
//! `stable_algebra_aggregate.csv`), at their tolerances.

use std::collections::BTreeMap;

use crate::common;

use common::{fails, n, ok, t};
use prospicio_prob::chf::{ChfLaw, ChfOptions, invert};
use prospicio_prob::levy::TweediePower;
use prospicio_xll::sheet::{Grid, Value};

/// One row of a reference file, by column name.
type Case = BTreeMap<String, String>;

/// A reference file's rows (no field in these files holds a comma).
fn reference(name: &str) -> Vec<Case> {
    let text = common::validation(&format!("reference/{name}"));
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .unwrap()
        .split(',')
        .map(str::to_string)
        .collect();
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| header.iter().cloned().zip(l.split(',').map(str::to_string)).collect())
        .collect()
}

/// `name=value` from a case's `params` field.
fn param(case: &Case, name: &str) -> f64 {
    case["params"]
        .split(';')
        .find_map(|kv| kv.strip_prefix(name)?.strip_prefix('='))
        .unwrap_or_else(|| panic!("{name} not in {}", case["params"]))
        .parse()
        .unwrap()
}

fn number(case: &Case, field: &str) -> f64 {
    case[field].parse().unwrap_or(0.0)
}

/// Whether `got` meets the case's `abs_tol` or `rel_tol`, as
/// `prospicio_validation::check` reads them.
fn passes(case: &Case, got: f64) -> bool {
    let want: f64 = case["expected"].parse().unwrap();
    let e = (got - want).abs();
    got == want || e <= number(case, "abs_tol") || e <= number(case, "rel_tol") * want.abs()
}

/// A sheet cell's number, NaN for a blank (a non-finite result).
fn value(v: &Value) -> f64 {
    match v {
        Value::Number(x) => *x,
        Value::Empty => f64::NAN,
        other => panic!("expected a number, got {other:?}"),
    }
}

/// A two-column summary as label to value.
fn table(g: &Grid) -> BTreeMap<String, Value> {
    assert_eq!(g.columns, 2);
    (0..g.rows)
        .map(|r| match g.get(r, 0) {
            Value::Text(s) => (s.clone(), g.get(r, 1).clone()),
            other => panic!("expected a label, got {other:?}"),
        })
        .collect()
}

fn column(v: &[f64]) -> Grid {
    Grid::from_rows(v.iter().map(|&x| vec![Value::Number(x)]).collect())
}

/// The Excel law and params of a reference case, `None` for a law with
/// no sheet form (the scaled and shifted ones).
fn sheet_law(case: &Case) -> Option<(&'static str, Vec<f64>)> {
    let p = |name: &str| param(case, name);
    Some(match case["distribution"].as_str() {
        "normal" => ("normal", vec![p("mu"), p("sigma")]),
        "gamma" => ("gamma", vec![p("shape"), p("scale")]),
        "inverse_gaussian" => ("inverse_gaussian", vec![p("mean"), p("shape")]),
        "tweedie" | "tweedie_power" => ("tweedie", vec![p("power"), p("mean"), p("dispersion")]),
        "normal_inverse_gaussian" => (
            "normal_inverse_gaussian",
            vec![p("alpha"), p("beta"), p("delta"), p("mu")],
        ),
        "variance_gamma" => (
            "variance_gamma",
            vec![p("sigma"), p("nu"), p("theta"), p("mu")],
        ),
        "tempered_stable" => (
            "tempered_stable",
            vec![p("C"), p("G"), p("M"), p("Y"), p("mu")],
        ),
        // parameterization 1 is S0 (the Probability lane's tests).
        "stable" if p("parameterization") == 1.0 => (
            "stable_s0",
            vec![p("alpha"), p("beta"), p("scale"), p("loc")],
        ),
        "stable" => ("stable", vec![p("alpha"), p("beta"), p("scale"), p("loc")]),
        _ => return None,
    })
}

/// The `log2`, `step` and `tilt` arguments of a case: `bs` 0 sizes the
/// bucket, `tilt` -1 is automatic and -2 none.
fn grid_args(case: &Case) -> [Grid; 4] {
    let bs = param(case, "bs");
    let tilt = if case["params"].contains("tilt=") {
        match param(case, "tilt") {
            -1.0 => Grid::missing(),
            -2.0 => Grid::scalar(Value::Bool(false)),
            tau => n(tau),
        }
    } else {
        Grid::missing()
    };
    [
        n(param(case, "log2")),
        if bs > 0.0 { n(bs) } else { Grid::missing() },
        Grid::missing(),
        tilt,
    ]
}

fn check_file(name: &str) {
    let cases = reference(name);
    let mut groups: BTreeMap<(String, String), Vec<&Case>> = BTreeMap::new();
    for c in &cases {
        groups
            .entry((c["distribution"].clone(), c["params"].clone()))
            .or_default()
            .push(c);
    }
    let mut checked = 0;
    let mut failures = Vec::new();
    for ((distribution, params), cases) in &groups {
        let Some((law, values)) = sheet_law(cases[0]) else {
            continue;
        };
        let mut ps: Vec<f64> = cases
            .iter()
            .filter(|c| matches!(c["quantity"].as_str(), "quantile" | "tvar"))
            .map(|c| number(c, "arg"))
            // The sheet's probabilities are in (0, 1).
            .filter(|p| *p > 0.0 && *p < 1.0)
            .collect();
        ps.sort_by(f64::total_cmp);
        ps.dedup();
        let [log2, step, x_min, tilt] = grid_args(cases[0]);
        let call = |probabilities: Grid, output: &str| {
            ok(
                "pro_cap_chf",
                vec![
                    t(law),
                    Grid::row(&values),
                    probabilities,
                    t(output),
                    log2.clone(),
                    step.clone(),
                    x_min.clone(),
                    tilt.clone(),
                ],
            )
        };
        let summary = table(&call(Grid::row(&ps), "summary"));
        let masses = call(Grid::missing(), "masses");
        for c in cases {
            let got = match c["quantity"].as_str() {
                "step" => value(&summary["Step"]),
                "x_min" => value(&summary["Left edge"]),
                "truncation_error" => value(&summary["Truncation error"]),
                "outside_below" => value(&summary["Mass below"]),
                "outside_above" => value(&summary["Mass above"]),
                "tilt_left" => value(&summary["Tilt left"]),
                "tilt_right" => value(&summary["Tilt right"]),
                "edge_error" => value(&summary["Edge error"]),
                "sum" => value(&summary["Total mass"]),
                "mean" => value(&summary["Mean"]),
                "sd" => value(&summary["SD"]),
                "quantile" => {
                    let p = number(c, "arg");
                    let k = ps.iter().position(|&q| q == p).unwrap();
                    let label = format!("VaR {}", label_percent(p));
                    assert!(summary.contains_key(&label), "{label} in {k}");
                    value(&summary[&label])
                }
                "mass" => value(masses.get(number(c, "arg") as usize + 1, 1)),
                "tvar" if !ps.contains(&number(c, "arg")) => continue,
                "tvar" => value(&summary[&format!("TVaR {}", label_percent(number(c, "arg")))]),
                other => panic!("{name}: no sheet value for {other}"),
            };
            checked += 1;
            if !passes(c, got) {
                failures.push(format!(
                    "{distribution} {params} {} {}: got {got:e}, want {}",
                    c["quantity"], c["arg"], c["expected"]
                ));
            }
        }
    }
    assert!(checked > 0, "{name}: nothing checked");
    assert!(
        failures.is_empty(),
        "{name}: {} of {checked} cases failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// `0.995` as the summary labels it, `"99.5%"`.
fn label_percent(p: f64) -> String {
    let s = format!("{:.6}", 100.0 * p);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    format!("{s}%")
}

#[test]
fn tvar_matches_aggregate() {
    check_file("chf_tvar_aggregate.csv");
}

#[test]
fn light_laws_match_aggregate() {
    check_file("chf_aggregate.csv");
}

#[test]
fn levy_laws_match_aggregate() {
    check_file("chf_levy_aggregate.csv");
}

#[test]
fn tweedie_at_any_power_matches_aggregate() {
    check_file("chf_tweedie_aggregate.csv");
}

#[test]
fn tilted_stable_laws_match_aggregate() {
    check_file("chf_tilt_aggregate.csv");
}

#[test]
fn split_stable_laws_match_aggregate() {
    check_file("chf_split_aggregate.csv");
}

/// The summary is the core's: the masses, moments and quantiles of
/// `invert` at the same options (the TVaR `ChfDistribution::tvar`), the report
/// as it is.
#[test]
fn summary_and_masses_are_the_cores() {
    let law = ChfLaw::TweediePower(TweediePower::new(1.5, 100.0, 2.0).unwrap());
    let d = invert(&law, &ChfOptions::with_log2(12)).unwrap();
    let args = |output: &str| {
        vec![
            t("tweedie"),
            column(&[1.5, 100.0, 2.0]),
            Grid::row(&[0.5, 0.99]),
            t(output),
            n(12.0),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
        ]
    };
    let s = table(&ok("pro_cap_chf", args("summary")));
    assert_eq!(value(&s["Mean"]), prospicio_prob::Distribution::mean(&d));
    assert_eq!(
        value(&s["VaR 99%"]),
        prospicio_prob::Distribution::quantile(&d, 0.99).unwrap()
    );
    assert_eq!(value(&s["TVaR 99%"]), d.tvar(0.99).unwrap());
    assert!(value(&s["TVaR 99%"]) > value(&s["VaR 99%"]));
    assert!((value(&s["Law mean"]) - 100.0).abs() < 1e-12);
    assert_eq!(value(&s["Points"]), 4096.0);
    assert_eq!(s["Step given"], Value::Bool(false));
    // This Tweedie's dropped frequencies, 1.8e-6 at 2^12 buckets (as in
    // aggregate), are above the 1e-6 flag.
    assert_eq!(value(&s["Truncation error"]), d.report().truncation_error);
    assert_eq!(s["Truncated"], Value::Bool(true));
    assert!(value(&s["Edge error"]).is_nan(), "a light law has no edge error");
    let m = ok("pro_cap_chf", args("masses"));
    assert_eq!((m.rows, m.columns), (4097, 2));
    assert_eq!(*m.get(0, 0), Value::from("x"));
    for k in [0, 1, 100, 4095] {
        assert_eq!(value(m.get(k + 1, 0)), d.x(k));
        assert_eq!(value(m.get(k + 1, 1)), d.masses()[k]);
    }
}

/// Defaults: blank trailing parameters, the default probabilities and
/// `log2`, a given step and left edge, and the tilt switched off.
#[test]
fn defaults_and_options() {
    let base = |params: Grid, rest: [Grid; 6]| {
        let mut args = vec![t("stable"), params];
        args.extend(rest);
        table(&ok("pro_cap_chf", args))
    };
    let blank = || Grid::missing();
    // alpha, beta alone: scale 1, location 0.
    let short = base(
        Grid::row(&[1.5, 1.0]),
        [blank(), blank(), n(12.0), blank(), blank(), blank()],
    );
    let full = base(
        Grid::row(&[1.5, 1.0, 1.0, 0.0]),
        [blank(), blank(), n(12.0), blank(), blank(), blank()],
    );
    assert_eq!(short, full);
    for label in ["VaR 99%", "VaR 99.5%", "VaR 99.9%", "TVaR 99.9%"] {
        assert!(short.contains_key(label), "{label}");
    }
    assert!(value(&short["Tilt right"]) > 0.0, "a right power tail is tilted");
    assert_eq!(value(&short["Tilt left"]), 0.0);
    // 2^16 points if log2 is omitted.
    let wide = base(
        Grid::row(&[1.5, 1.0]),
        [blank(), blank(), blank(), blank(), blank(), blank()],
    );
    assert_eq!(value(&wide["Points"]), 65536.0);
    // FALSE: no tilt.
    let off = base(
        Grid::row(&[1.5, 1.0]),
        [
            blank(),
            blank(),
            n(12.0),
            blank(),
            blank(),
            Grid::scalar(Value::Bool(false)),
        ],
    );
    assert_eq!(value(&off["Tilt right"]), 0.0);
    // A given grid.
    let given = table(&ok(
        "pro_cap_chf",
        vec![
            t("normal"),
            Grid::row(&[0.0, 1.0]),
            n(0.9),
            Grid::missing(),
            n(10.0),
            n(0.02),
            n(-10.0),
            Grid::missing(),
        ],
    ));
    assert_eq!(value(&given["Step"]), 0.02);
    assert_eq!(value(&given["Left edge"]), -10.0);
    assert_eq!(given["Step given"], Value::Bool(true));
    assert!((value(&given["VaR 90%"]) - 1.28).abs() < 0.02);
}

#[test]
fn bad_inputs_are_errors() {
    let call = |law: &str, params: Grid, rest: [Grid; 6]| {
        let mut args = vec![t(law), params];
        args.extend(rest);
        fails("pro_cap_chf", args)
    };
    let b = || Grid::missing();
    let none = || [b(), b(), b(), b(), b(), b()];
    let e = call("cauchy", Grid::row(&[1.0]), none());
    assert!(e.contains("law must be") && e.contains("tempered_stable"), "{e}");
    let e = call("normal", Grid::row(&[1.0]), none());
    assert!(e.contains("mu, sigma") && e.contains("sigma is missing"), "{e}");
    let e = call("gamma", Grid::row(&[1.0, 2.0, 3.0]), none());
    assert!(e.contains("takes 2 values"), "{e}");
    let e = call("normal", Grid::from_rows(vec![vec![Value::Number(0.0), Value::Number(1.0)]; 2]), none());
    assert!(e.contains("row or a column"), "{e}");
    let e = call("normal", Grid::row(&[0.0, -1.0]), none());
    assert!(e.contains("sigma"), "{e}");
    let e = call("normal", Grid::row(&[0.0, 1.0]), [b(), t("density"), b(), b(), b(), b()]);
    assert!(e.contains("output must be"), "{e}");
    let e = call("normal", Grid::row(&[0.0, 1.0]), [b(), b(), n(21.0), b(), b(), b()]);
    assert!(e.contains("log2 must be a whole number from 1 to 20"), "{e}");
    let e = call("normal", Grid::row(&[0.0, 1.0]), [b(), t("masses"), n(20.0), b(), b(), b()]);
    assert!(e.contains("from 1 to 19"), "{e}");
    let e = call("normal", Grid::row(&[0.0, 1.0]), [b(), b(), b(), b(), n(-3.0), b()]);
    assert!(e.contains("step"), "{e}");
    let e = call("normal", Grid::row(&[0.0, 1.0]), [b(), b(), b(), b(), b(), n(-1.0)]);
    assert!(e.contains("tilt must be"), "{e}");
    let e = call("normal", Grid::row(&[0.0, 1.0]), [n(1.5), b(), b(), b(), b(), b()]);
    assert!(e.contains("probabilities"), "{e}");
    let e = fails("pro_cap_stable", vec![n(2.5), n(0.0), b(), b(), b()]);
    assert!(e.contains("alpha"), "{e}");
    let e = fails("pro_cap_stable", vec![n(1.5), n(0.0), b(), b(), t("S2")]);
    assert!(e.contains("parameterization must be"), "{e}");
    let e = fails("pro_cap_stable_pareto", vec![n(10.0), n(2.5), n(1.0), n(5.0)]);
    assert!(e.contains("alpha"), "{e}");
}

/// `PRO.CAP.STABLE` and `PRO.CAP.STABLE.PARETO` against `aggregate`'s
/// stable laws, built from their parameters and as a compound Pareto's
/// limit.
#[test]
fn stable_figures_match_aggregate() {
    let labels = [
        ("alpha", "Alpha"),
        ("beta", "Beta"),
        ("scale", "Scale"),
        ("loc", "Location (S1)"),
        ("loc0", "Location (S0)"),
        ("tail_constant", "Right tail constant"),
        ("left_tail_constant", "Left tail constant"),
        ("prob_above_loc", "P(X > location)"),
    ];
    let mut checked = 0;
    let mut failures = Vec::new();
    for c in reference("stable_algebra_aggregate.csv") {
        let s = match c["arg"].as_str() {
            "identity" => table(&ok(
                "pro_cap_stable",
                vec![
                    n(param(&c, "x_alpha")),
                    n(param(&c, "x_beta")),
                    n(param(&c, "x_scale")),
                    n(param(&c, "x_loc")),
                    Grid::missing(),
                ],
            )),
            "compound_pareto" => table(&ok(
                "pro_cap_stable_pareto",
                vec![
                    n(param(&c, "frequency")),
                    n(param(&c, "alpha")),
                    n(param(&c, "theta")),
                    n(param(&c, "sev_mean")),
                ],
            )),
            _ => continue,
        };
        let label = labels
            .iter()
            .find(|(q, _)| *q == c["quantity"])
            .unwrap_or_else(|| panic!("no label for {}", c["quantity"]))
            .1;
        let got = value(&s[label]);
        let want: f64 = c["expected"].parse().unwrap();
        checked += 1;
        if !(passes(&c, got) || (want.is_nan() && got.is_nan())) {
            failures.push(format!("{} {}: got {got:e}, want {want}", c["params"], c["quantity"]));
        }
    }
    assert!(checked >= 40, "{checked} cases");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// `PRO.CAP.STABLE`'s first four values are `PRO.CAP.CHF`'s params for
/// `"stable"`, and `"S0"` reads the location as Nolan's.
#[test]
fn stable_tables_feed_the_engine() {
    let s = ok(
        "pro_cap_stable_pareto",
        vec![n(50.0), n(1.6), n(3.0), n(12.0)],
    );
    let params = Grid::from_rows((0..4).map(|r| vec![s.get(r, 1).clone()]).collect());
    let total = table(&ok(
        "pro_cap_chf",
        vec![
            t("stable"),
            params,
            n(0.99),
            Grid::missing(),
            n(14.0),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
        ],
    ));
    // The limit's mean is the compound's, 50 x 12.
    assert!((value(&total["Law mean"]) - 600.0).abs() < 1e-9);
    assert!(value(&total["VaR 99%"]) > 600.0);
    let s1 = table(&ok(
        "pro_cap_stable",
        vec![n(1.5), n(0.4), n(2.0), n(1.0), t("S1")],
    ));
    let s0 = table(&ok(
        "pro_cap_stable",
        vec![
            n(1.5),
            n(0.4),
            n(2.0),
            Grid::scalar(s1["Location (S0)"].clone()),
            t("s0"),
        ],
    ));
    assert!((value(&s0["Location (S1)"]) - 1.0).abs() < 1e-13);
}
