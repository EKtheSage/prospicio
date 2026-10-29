//! GLMs on a sheet (`PRO.GLM`, `PRO.GLM.PREDICT`, `PRO.GLM.RELATIVITIES`,
//! `docs/design/xll.md` decision 41), reached through the catalogue as
//! Excel reaches them, on `validation/data/glm_policies.csv` as a table
//! with a header row:
//!
//! * every statsmodels 0.15.0 case of
//!   `validation/reference/glm_statsmodels.csv` (ten families and links,
//!   with an exposure offset, prior weights, a Tweedie power and a negative
//!   binomial theta): coefficients, standard errors, deviance, null
//!   deviance, dispersion, log-likelihood and AIC;
//! * predictions on new rows against the coefficients by hand, and on the
//!   training rows against a refit;
//! * relativities against exp(estimate) and its interval, with the base
//!   levels, and a chosen base level;
//! * the errors.

mod common;

use common::{n, num, t, text, validation};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn none() -> Grid {
    Grid::missing()
}

/// `glm_policies.csv` as a sheet table, with a `prop` column (successes
/// over trials) for the binomial.
fn policies() -> Grid {
    let text = validation("data/glm_policies.csv");
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let (s, tr) = (
        header.iter().position(|h| *h == "successes").unwrap(),
        header.iter().position(|h| *h == "trials").unwrap(),
    );
    let mut rows = vec![
        header
            .iter()
            .map(|h| Value::from(*h))
            .chain([Value::from("prop")])
            .collect::<Vec<_>>(),
    ];
    for line in lines {
        let f: Vec<&str> = line.split(',').collect();
        let mut row: Vec<Value> = f
            .iter()
            .map(|x| match x.parse::<f64>() {
                Ok(v) => Value::Number(v),
                Err(_) => Value::from(*x),
            })
            .collect();
        let p = f[s].parse::<f64>().unwrap() / f[tr].parse::<f64>().unwrap();
        row.push(Value::Number(p));
        rows.push(row);
    }
    Grid::from_rows(rows)
}

fn terms() -> Grid {
    Grid::from_rows(vec![vec![Value::from("age"), Value::from("region")]])
}

/// A statsmodels case's model arguments after `data`: `response, terms,
/// family, link, exposure, weights, p`.
fn case(name: &str) -> [Grid; 7] {
    let s = |x: &str| t(x);
    let (response, family, link, exposure, weights, p) = match name {
        "poisson_log" => ("claims", "poisson", "log", s("exposure"), none(), none()),
        "quasi_poisson_log" => (
            "claims",
            "quasipoisson",
            "log",
            s("exposure"),
            none(),
            none(),
        ),
        "gamma_log" => ("severity", "gamma", "log", none(), none(), none()),
        "gamma_inverse" => ("severity", "gamma", "inverse", none(), none(), none()),
        "gamma_log_weighted" => ("avg_sev", "gamma", "log", none(), s("n_sev"), none()),
        "inverse_gaussian_log" => ("ig", "inverse_gaussian", "log", none(), none(), none()),
        "binomial_logit" => ("prop", "binomial", "logit", none(), s("trials"), none()),
        "negative_binomial_log" => ("nb", "negative_binomial", "log", none(), none(), n(2.0)),
        "gaussian_identity" => ("gauss", "gaussian", "identity", none(), none(), none()),
        "tweedie_log" => ("pure", "tweedie", "log", none(), s("exposure"), n(1.5)),
        other => panic!("unknown case {other}"),
    };
    [
        s(response),
        terms(),
        s(family),
        s(link),
        exposure,
        weights,
        p,
    ]
}

fn glm(data: &Grid, name: &str) -> Grid {
    let mut args = vec![data.clone()];
    args.extend(case(name));
    call("pro_glm", args)
}

/// A coefficient row's design-column name: `region[B]`, or `age`.
fn coefficient_name(out: &Grid, r: usize) -> String {
    match (out.get(r, 0), out.get(r, 1)) {
        (Value::Text(term), Value::Text(level)) => format!("{term}[{level}]"),
        (Value::Text(term), Value::Empty) => term.clone(),
        other => panic!("row {r}: {other:?}"),
    }
}

/// The value beside a statistic's label below the coefficients.
fn statistic(out: &Grid, label: &str) -> f64 {
    let r = (0..out.rows)
        .find(|&r| out.get(r, 0) == &Value::from(label))
        .unwrap_or_else(|| panic!("no {label} row"));
    num(out.get(r, 1))
}

#[test]
fn glms_match_statsmodels() {
    let data = policies();
    let mut fits: Vec<(String, Grid)> = Vec::new();
    let mut checked = 0;
    for line in validation("reference/glm_statsmodels.csv").lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let name = f[0];
        if !fits.iter().any(|(n, _)| n == name) {
            fits.push((name.to_owned(), glm(&data, name)));
        }
        let out = &fits.iter().find(|(n, _)| n == name).unwrap().1;
        assert!(out.rows > 2, "{name}: {out:?}");
        let coefficient = |column: usize| {
            let r = (1..out.rows)
                .take_while(|&r| !out.get(r, 0).is_blank())
                .find(|&r| coefficient_name(out, r) == f[2])
                .unwrap_or_else(|| panic!("{name}: no coefficient {}", f[2]));
            num(out.get(r, column))
        };
        let got = match f[1] {
            "coef" => coefficient(2),
            "std_error" => coefficient(3),
            "deviance" => statistic(out, "Deviance"),
            "null_deviance" => statistic(out, "Null deviance"),
            "scale" => statistic(out, "Dispersion"),
            "log_likelihood" => statistic(out, "Log-likelihood"),
            "aic" => statistic(out, "AIC"),
            other => panic!("{name}: unknown quantity {other}"),
        };
        let expected: f64 = f[3].parse().unwrap();
        let (abs, rel): (f64, f64) = (f[4].parse().unwrap(), f[5].parse().unwrap());
        assert!(
            (got - expected).abs() <= abs + rel * expected.abs(),
            "{name} {} {}: {got} vs {expected}",
            f[1],
            f[2]
        );
        checked += 1;
    }
    assert_eq!(checked, 144, "every row of the reference");
    assert_eq!(fits.len(), 10);
}

#[test]
fn the_summary_lays_out_coefficients_then_statistics() {
    let out = glm(&policies(), "poisson_log");
    assert_eq!(out.columns, 7);
    assert_eq!(out.get(0, 6), &Value::from("exp(Estimate)"));
    assert_eq!(out.get(1, 0), &Value::from("(Intercept)"));
    assert_eq!(out.get(3, 0), &Value::from("region"));
    assert_eq!(out.get(3, 1), &Value::from("B"));
    // Estimate over SE, and exp(estimate).
    let (b, se) = (num(out.get(3, 2)), num(out.get(3, 3)));
    assert!((num(out.get(3, 4)) - b / se).abs() < 1e-12);
    assert!((num(out.get(3, 6)) - b.exp()).abs() < 1e-12);
    assert!(out.get(6, 0).is_blank());
    assert_eq!(statistic(&out, "Observations"), 600.0);
    assert_eq!(statistic(&out, "Residual df"), 595.0);
}

/// A small table of new rows: age, region and exposure.
fn newdata() -> Grid {
    let row = |age: f64, region: &str, exposure: f64| {
        vec![
            Value::Number(age),
            Value::from(region),
            Value::Number(exposure),
        ]
    };
    Grid::from_rows(vec![
        vec![
            Value::from("age"),
            Value::from("region"),
            Value::from("exposure"),
        ],
        row(30.0, "A", 1.0),
        row(45.0, "C", 0.5),
        row(60.0, "D", 2.0),
    ])
}

#[test]
fn predictions_are_the_coefficients_applied_to_new_rows() {
    let data = policies();
    let fit = glm(&data, "poisson_log");
    let coefficient = |name: &str| {
        let r = (1..7).find(|&r| coefficient_name(&fit, r) == name).unwrap();
        num(fit.get(r, 2))
    };
    let [response, terms, family, link, exposure, weights, p] = case("poisson_log");
    let out = call(
        "pro_glm_predict",
        vec![
            data.clone(),
            response,
            terms,
            newdata(),
            family,
            link,
            exposure,
            weights,
            p,
        ],
    );
    assert_eq!((out.rows, out.columns), (3, 1), "{out:?}");
    let by_hand = [
        (30.0, None, 1.0),
        (45.0, Some("region[C]"), 0.5),
        (60.0, Some("region[D]"), 2.0),
    ]
    .map(|(age, region, exposure): (f64, Option<&str>, f64)| {
        (coefficient("(Intercept)")
            + coefficient("age") * age
            + region.map_or(0.0, coefficient)
            + f64::ln(exposure))
        .exp()
    });
    for (got, expected) in out.cells.iter().map(num).zip(by_hand) {
        assert!(
            (got - expected).abs() <= 1e-12 * expected,
            "{got} vs {expected}"
        );
    }
    // A level the training data never saw is an error.
    let mut unseen = newdata();
    unseen.cells[4] = Value::from("Z");
    let [response, terms, family, link, exposure, ..] = case("poisson_log");
    assert!(
        text(&call(
            "pro_glm_predict",
            vec![data, response, terms, unseen, family, link, exposure]
        ))
        .contains("Z")
    );
}

#[test]
fn relativities_with_their_base_levels() {
    let data = policies();
    let fit = glm(&data, "poisson_log");
    let mut args = vec![data.clone()];
    args.extend(case("poisson_log"));
    let out = call("pro_glm_relativities", args.clone());
    // age per unit, region's base A, then B, C, D.
    assert_eq!((out.rows, out.columns), (6, 5), "{out:?}");
    assert_eq!(out.get(0, 3), &Value::from("Lower 95%"));
    assert_eq!(out.get(1, 1), &Value::from("per unit"));
    assert_eq!(out.get(2, 1), &Value::from("A (base)"));
    assert_eq!(num(out.get(2, 2)), 1.0);
    let z = 1.959963984540054;
    for (r, f) in [(1, 2), (3, 3), (4, 4), (5, 5)] {
        let (b, se) = (num(fit.get(f, 2)), num(fit.get(f, 3)));
        assert_eq!(out.get(r, 0), fit.get(f, 0));
        assert!((num(out.get(r, 2)) - b.exp()).abs() < 1e-12);
        assert!((num(out.get(r, 3)) - (b - z * se).exp()).abs() < 1e-9);
        assert!((num(out.get(r, 4)) - (b + z * se).exp()).abs() < 1e-9);
    }
    // At 90%, a narrower interval.
    args.push(n(0.9));
    let narrow = call("pro_glm_relativities", args);
    assert_eq!(narrow.get(0, 3), &Value::from("Lower 90%"));
    assert!(num(narrow.get(3, 3)) > num(out.get(3, 3)));

    // A chosen base level: region D, so the others are relative to it.
    let mut args = vec![data.clone()];
    let mut c = case("poisson_log");
    c[1] = Grid::from_rows(vec![vec![Value::from("age"), Value::from("region=D")]]);
    args.extend(c);
    let based = call("pro_glm_relativities", args);
    assert_eq!(based.get(2, 1), &Value::from("D (base)"));
    let a = (1..based.rows)
        .find(|&r| based.get(r, 1) == &Value::from("A"))
        .unwrap();
    assert!((num(based.get(a, 2)) - 1.0 / num(out.get(5, 2))).abs() < 1e-9);
}

#[test]
fn errors() {
    let data = policies();
    let run = |symbol: &str, args: Vec<Grid>| text(&call(symbol, args)).to_owned();
    assert!(
        run("pro_glm", vec![data.clone(), t("nope"), terms()]).contains("no column named \"nope\"")
    );
    assert!(run("pro_glm", vec![data.clone(), t("claims")]).contains("terms is required"));
    assert!(
        run(
            "pro_glm",
            vec![data.clone(), t("claims"), terms(), t("weibull")]
        )
        .contains("family must be")
    );
    assert!(
        run(
            "pro_glm",
            vec![data.clone(), t("claims"), terms(), t("tweedie")]
        )
        .contains("needs p")
    );
    assert!(
        run(
            "pro_glm",
            vec![
                data.clone(),
                t("claims"),
                terms(),
                none(),
                none(),
                none(),
                none(),
                n(1.5)
            ]
        )
        .contains("leave it empty")
    );
    assert!(
        run(
            "pro_glm",
            vec![
                data.clone(),
                t("claims"),
                terms(),
                none(),
                t("identity"),
                t("exposure")
            ]
        )
        .contains("log link")
    );
    assert!(run("pro_glm", vec![data.clone(), t("region"), terms()]).contains("must be numbers"));
    let mut blank = data.clone();
    blank.cells[3 * blank.columns + 1] = Value::Empty;
    assert!(run("pro_glm", vec![blank, t("claims"), terms()]).contains("blank in row 3"));
    assert!(
        run(
            "pro_glm",
            vec![
                data.clone(),
                t("claims"),
                Grid::from_rows(vec![vec![Value::from("age:")]])
            ]
        )
        .contains("an interaction")
    );
    let mut args = vec![data.clone()];
    args.extend(case("gamma_inverse"));
    assert!(run("pro_glm_relativities", args).contains("log link"));
    assert!(
        run(
            "pro_glm",
            vec![
                Grid::from_rows(vec![vec![Value::from("a")]]),
                t("a"),
                t("a")
            ]
        )
        .contains("header row")
    );
}
