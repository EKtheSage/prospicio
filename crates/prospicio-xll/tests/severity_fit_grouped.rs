//! The newer severity fits, reached through the catalogue as Excel
//! reaches them: the Lomax in `PRO.SEV.FIT` and the severity functions,
//! truncation from above, and the fits to grouped losses
//! (`PRO.SEV.FIT.GROUPED`) and by the minimum limited-mean distance
//! (`PRO.SEV.FIT.LEV`) against Bahnemann's *Distributions for Actuaries*
//! (CAS Monograph 2), Ex 2.8, 2.10 and P2.16, as `prospicio_prob`'s own
//! tests reproduce them.

mod common;

use common::{n, num, t, text};
use prospicio_prob::severity_fit::{Family, fit_severity};
use prospicio_prob::{Distribution, LargeLosses};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn estimate(out: &Grid, label: &str) -> f64 {
    let r = (0..out.rows)
        .find(|&r| out.get(r, 0) == &Value::from(label))
        .unwrap_or_else(|| panic!("no row {label} in {out:?}"));
    num(out.get(r, 1))
}

const LOSSES: [f64; 10] = [
    120.0, 300.0, 450.0, 800.0, 1500.0, 2600.0, 2900.0, 650.0, 90.0, 2100.0,
];

/// `PRO.SEV.FIT` with its trailing arguments: `[thresholds, censored,
/// weights, upper]` after losses, family and output.
fn fit(family: &str, tail: [Grid; 4]) -> Grid {
    let [r, c, w, u] = tail;
    call(
        "pro_sev_fit",
        vec![
            Grid::column(&LOSSES),
            t(family),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            r,
            c,
            w,
            u,
        ],
    )
}

fn none() -> [Grid; 4] {
    [
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
    ]
}

#[test]
fn the_lomax_is_fitted_and_priced() {
    // Heavy-tailed losses (`validation/scripts/r_pareto_fit.R`'s): on
    // light-tailed ones the Lomax runs to its exponential limit.
    let heavy = [
        1100.0, 1300.0, 1750.0, 2000.0, 2600.0, 3500.0, 4100.0, 5200.0, 7000.0, 9000.0, 12000.0,
        18000.0, 25000.0, 40000.0,
    ];
    let out = call("pro_sev_fit", vec![Grid::column(&heavy), t("lomax")]);
    let want = fit_severity(Family::Lomax, &LargeLosses::new(heavy.to_vec()).unwrap()).unwrap();
    let (a, s) = (estimate(&out, "alpha"), estimate(&out, "scale"));
    assert_eq!([a, s], [want.parameters()[0], want.parameters()[1]]);
    assert_eq!(estimate(&out, "AIC"), want.aic());
    // The parameters feed the severity functions: S(x) = (1 + x/θ)^(-α).
    let cdf = call("pro_sev_cdf", vec![n(1000.0), t("lomax"), n(a), n(s)]);
    let want = 1.0 - (1.0 + 1000.0 / s).powf(-a);
    assert!(
        (num(&cdf.cells[0]) - want).abs() < 1e-12,
        "{a} {s}: {:?} vs {want}",
        cdf.cells[0]
    );
    let e = call(
        "pro_sev_cdf",
        vec![n(1.0), t("lomax"), n(2.0), n(3.0), n(4.0)],
    );
    assert!(text(&e).contains("p3 is taken only"), "{}", text(&e));
}

/// Losses kept only up to 3,000: each conditioned on `X ≤ 3000`, the
/// likelihood in closed form for the lognormal.
#[test]
fn truncation_from_above_conditions_each_loss() {
    let [r, c, w, _] = none();
    let out = fit("lognormal", [r, c, w, n(3000.0)]);
    let (m, s) = (estimate(&out, "meanlog"), estimate(&out, "sdlog"));
    let ll = |m: f64, s: f64| {
        let phi = 0.5 * (1.0 + erf((3000f64.ln() - m) / (s * std::f64::consts::SQRT_2)));
        LOSSES
            .iter()
            .map(|&v| {
                -(v * s * (2.0 * std::f64::consts::PI).sqrt()).ln()
                    - 0.5 * ((v.ln() - m) / s).powi(2)
            })
            .sum::<f64>()
            - LOSSES.len() as f64 * phi.ln()
    };
    let got = estimate(&out, "Log-likelihood");
    assert!(
        (got - ll(m, s)).abs() < 1e-8 * got.abs(),
        "{got} vs {}",
        ll(m, s)
    );
    for (dm, ds) in [(1e-3, 0.0), (-1e-3, 0.0), (0.0, 1e-3), (0.0, -1e-3)] {
        assert!(ll(m + dm, s + ds) < got);
    }
    let plain = fit("lognormal", none());
    assert!(s > estimate(&plain, "sdlog"));
    // One bound per loss gives the same fit; the gof table stays below it.
    let per_loss = {
        let [r, c, w, _] = none();
        fit("lognormal", [r, c, w, Grid::column(&[3000.0; 10])])
    };
    assert_eq!(per_loss, out);
    let gof = call(
        "pro_sev_fit",
        vec![
            Grid::column(&LOSSES),
            t("lognormal"),
            t("gof"),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            n(3000.0),
        ],
    );
    assert!((1..gof.rows).all(|r| num(gof.get(r, 5)) <= 3000.0));

    // A bound below a loss, and a bound on a Pareto fit, are refused.
    let [r, c, w, _] = none();
    let e = fit("lognormal", [r, c, w, n(100.0)]);
    assert!(text(&e).contains("truncated_above"), "{}", text(&e));
    let e = call(
        "pro_sev_fit",
        vec![
            Grid::column(&LOSSES),
            t("pareto"),
            Grid::missing(),
            n(80.0),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            n(3000.0),
        ],
    );
    assert!(text(&e).contains("upper is taken only"), "{}", text(&e));
}

/// The error function through the standard normal cdf (std has no erf).
fn erf(x: f64) -> f64 {
    let z = prospicio_prob::Normal::new(0.0, 1.0).unwrap();
    2.0 * z.cdf(x * std::f64::consts::SQRT_2) - 1.0
}

/// Bahnemann's six bands of 200 losses (Ex 2.6 to 2.8, P2.16).
fn bands(open: Value) -> (Grid, Grid) {
    let mut bounds: Vec<Value> = [0.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0]
        .map(Value::Number)
        .to_vec();
    bounds.push(open);
    (
        Grid::from_rows(vec![bounds]),
        Grid::row(&[42.0, 61.0, 47.0, 26.0, 14.0, 10.0]),
    )
}

#[test]
fn grouped_fits_reproduce_bahnemann() {
    let (b, c) = bands(Value::Empty);
    // Ex 2.8: the minimum chi-square lognormal (the book's printed one is
    // not the minimum; `knowledge/findings/grouped-severity-fits.md`).
    let out = call(
        "pro_sev_fit_grouped",
        vec![b.clone(), c.clone(), t("lognormal"), t("chi_square")],
    );
    assert!((estimate(&out, "meanlog") - 7.2857).abs() < 5e-5);
    assert!((estimate(&out, "sdlog") - 0.4570).abs() < 5e-5);
    assert!((estimate(&out, "Chi-square") - 0.614).abs() < 5e-4);
    assert_eq!(estimate(&out, "Degrees of freedom"), 3.0);
    let p = estimate(&out, "p-value");
    assert!(0.89 < p && p < 0.90, "{p}");
    assert_eq!(estimate(&out, "Losses"), 200.0);
    // "inf" opens the top band as a blank does.
    let (b_inf, _) = bands(Value::from("inf"));
    assert_eq!(
        call(
            "pro_sev_fit_grouped",
            vec![b_inf, c.clone(), t("lognormal"), t("chi_square")]
        ),
        out
    );
    // Maximum likelihood (the default) has standard errors and does no
    // better on chi-square.
    let ml = call(
        "pro_sev_fit_grouped",
        vec![b.clone(), c.clone(), t("lognormal")],
    );
    assert!(estimate(&ml, "Chi-square") >= estimate(&out, "Chi-square"));
    assert!(num(ml.get(1, 2)) > 0.0 && num(ml.get(2, 2)) > 0.0);
    // P2.16: a gamma by minimum chi-square.
    let g = call(
        "pro_sev_fit_grouped",
        vec![b.clone(), c.clone(), t("gamma"), t("chi_square")],
    );
    assert!((num(g.get(1, 1)) - 4.7432).abs() < 5e-5);
    assert!((num(g.get(2, 1)) - 337.31).abs() < 5e-3);
    // The table: the expected counts add up to the 200 observed, the open
    // band's upper bound blank.
    let table = call(
        "pro_sev_fit_grouped",
        vec![b, c, t("lognormal"), Grid::missing(), t("table")],
    );
    assert_eq!((table.rows, table.columns), (7, 4));
    let expected: f64 = (1..7).map(|r| num(table.get(r, 3))).sum();
    assert!((expected - 200.0).abs() < 1e-9);
    assert_eq!(table.get(6, 1), &Value::Empty);
    assert_eq!(num(table.get(6, 0)), 3000.0);
}

#[test]
fn lev_distance_reproduces_bahnemann_ex_2_10() {
    // 1,500 claims under a 300,000 limit: the empirical limited means
    // from the counts below and totals at each size.
    let x = [
        1000.0, 5000.0, 10000.0, 25000.0, 50000.0, 100000.0, 200000.0, 300000.0,
    ];
    let below = [
        729.0, 1096.0, 1208.0, 1326.0, 1391.0, 1440.0, 1468.0, 1500.0,
    ];
    let total = [
        225138.0, 1102272.0, 1918947.0, 3752091.0, 6007543.0, 9234739.0, 13100561.0, 22343455.0,
    ];
    let e: Vec<f64> = (0..8)
        .map(|j| (total[j] + x[j] * (1500.0 - below[j])) / 1500.0)
        .collect();
    let out = call(
        "pro_sev_fit_lev",
        vec![Grid::column(&x), Grid::column(&e), t("lognormal")],
    );
    assert!((estimate(&out, "meanlog") - 6.9852).abs() < 5e-5);
    assert!((estimate(&out, "sdlog") - 2.5850).abs() < 5e-5);
    assert_eq!(out.get(4, 1), &Value::Bool(true));
    let table = call(
        "pro_sev_fit_lev",
        vec![
            Grid::column(&x),
            Grid::column(&e),
            t("lognormal"),
            Grid::missing(),
            t("table"),
        ],
    );
    let fitted: Vec<i64> = (1..9)
        .map(|r| num(table.get(r, 2)).round() as i64)
        .collect();
    assert_eq!(fitted, [648, 2091, 3239, 5410, 7580, 10168, 13069, 14850]);
    // Equal weights change nothing.
    let weighted = call(
        "pro_sev_fit_lev",
        vec![
            Grid::column(&x),
            Grid::column(&e),
            t("lognormal"),
            Grid::column(&[1.0; 8]),
        ],
    );
    assert_eq!(weighted, out);
}

#[test]
fn bad_inputs_are_named() {
    let (b, c) = bands(Value::Empty);
    let e = call(
        "pro_sev_fit_grouped",
        vec![b.clone(), c.clone(), t("pareto")],
    );
    assert!(text(&e).contains("family must be"), "{}", text(&e));
    let e = call(
        "pro_sev_fit_grouped",
        vec![b.clone(), c.clone(), t("gamma"), t("mle")],
    );
    assert!(text(&e).contains("method must be"), "{}", text(&e));
    let e = call(
        "pro_sev_fit_grouped",
        vec![b, c.clone(), t("gamma"), Grid::missing(), t("chart")],
    );
    assert!(text(&e).contains("output must be"), "{}", text(&e));
    let short = Grid::row(&[0.0, 1000.0]);
    let e = call("pro_sev_fit_grouped", vec![short, c, t("gamma")]);
    assert!(text(&e).contains("bounds"), "{}", text(&e));
    let e = call(
        "pro_sev_fit_lev",
        vec![
            Grid::column(&[1.0, 2.0, 3.0]),
            Grid::column(&[1.0, 1.5, 1.8]),
            t("gamma"),
            Grid::column(&[1.0, 1.0]),
        ],
    );
    assert!(text(&e).contains("weights must have 3"), "{}", text(&e));
}
