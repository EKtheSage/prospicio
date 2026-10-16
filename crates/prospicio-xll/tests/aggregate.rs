//! Aggregate distributions (`PRO.AGG.STATS`, `PRO.AGG.LAYERS`), reached
//! through the catalogue as Excel reaches them:
//!
//! * per-occurrence layer statistics against `aggregate` 1.0.1's
//!   `reins_stats_df` (`validation/reference/reinsurance_aggregate.csv`,
//!   books C, S and T, on `aggregate`'s grid: step 1 to the cap);
//! * the ground-up moments against `aggregate`'s on mpmath
//!   (`moments_aggregate.csv`), to the grid's discretization;
//! * VaR and TVaR against the Panjer recursion on the same grid (a second
//!   algorithm), and a layer's annual mean and SD against the collective
//!   model's closed forms.

mod common;

use std::collections::HashMap;

use common::{n, num, reference, t, text};
use prospicio_aggregate::{CollectiveModel, panjer};
use prospicio_prob::{Distribution, Gamma, Grid as PGrid, PanjerClass, Poisson};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn fields(params: &str) -> HashMap<&str, &str> {
    params
        .split(';')
        .map(|kv| kv.split_once('=').unwrap())
        .collect()
}

/// count_mean, dispersion, family, p1, p2, p3 for a reference book.
fn book(f: &HashMap<&str, &str>) -> Vec<Grid> {
    let v = |k: &str| f[k].parse::<f64>().unwrap();
    let lambda = v("lambda");
    let dispersion = f
        .get("mixing_cv")
        .map_or(1.0, |c| 1.0 + c.parse::<f64>().unwrap().powi(2) * lambda);
    let (family, p1, p2) = match f["severity"] {
        "gamma" => ("gamma_cv", v("mean"), v("cv")),
        "lognormal" => ("lognormal_cv", v("mean"), v("cv")),
        "weibull" => ("weibull", v("shape"), v("scale")),
        other => panic!("{other}"),
    };
    vec![
        n(lambda),
        n(dispersion),
        t(family),
        n(p1),
        n(p2),
        Grid::missing(),
    ]
}

fn row_of(g: &Grid, label: &str) -> usize {
    (0..g.rows)
        .find(|&r| g.get(r, 0) == &Value::from(label))
        .unwrap_or_else(|| panic!("no row {label}"))
}

#[test]
fn layer_statistics_match_aggregate() {
    let mut checked = 0;
    let mut fits: HashMap<String, Grid> = HashMap::new();
    for row in reference("reinsurance_aggregate.csv").into_iter().skip(1) {
        let (name, params, quantity) = (&row[0], &row[1], &row[2]);
        let Some((col, stat)) = quantity.split_once(':') else {
            continue;
        };
        let Some(i) = col.strip_prefix("layer.") else {
            continue;
        };
        if !["C", "S", "T"].contains(&name.as_str()) {
            continue;
        }
        let f = fields(params);
        let ex = fits.entry(params.clone()).or_insert_with(|| {
            let occ: Vec<Vec<f64>> = f["occ"]
                .split('+')
                .map(|l| l.split(':').map(|x| x.parse().unwrap()).collect())
                .collect();
            let r =
                |k: usize| Grid::from_rows(vec![occ.iter().map(|l| Value::Number(l[k])).collect()]);
            let cap: f64 = f["cap"].parse().unwrap();
            let mut args = book(&f);
            args.extend([r(1), r(2), r(0), n(1.0), n(cap + 1.0)]);
            call("pro_agg_layers", args)
        });
        assert_eq!(ex.rows, 15, "{name}: {:?}", ex.as_scalar());
        let label = match stat {
            "pr_attach" => "Pr attach",
            "pr_detach" => "Pr detach",
            "pr_loss" => "Pr loss",
            "lol" => "Loss on line",
            "freq_mean" => "Count mean",
            "freq_cv" => "Count CV",
            "freq_skew" => "Count skewness",
            "sev_mean" => "Severity mean",
            "sev_cv" => "Severity CV",
            "sev_skew" => "Severity skewness",
            "agg_mean" => "Annual mean",
            "agg_cv" => "Annual CV",
            "agg_skew" => "Annual skewness",
            other => panic!("{other}"),
        };
        let got = num(ex.get(row_of(ex, label), i.parse().unwrap()));
        let expected: f64 = row[4].parse().unwrap();
        let (abs_tol, rel_tol): (f64, f64) = (row[5].parse().unwrap(), row[6].parse().unwrap());
        assert!(
            (got - expected).abs() <= abs_tol.max(rel_tol * expected.abs()),
            "{name} {quantity}: {got} vs {expected}"
        );
        checked += 1;
    }
    assert_eq!(checked, 60);
}

#[test]
fn ground_up_moments_match_aggregate_to_the_grid() {
    let mut checked = 0;
    let mut books: HashMap<String, Grid> = HashMap::new();
    for row in reference("moments_aggregate.csv").into_iter().skip(1) {
        let q = row[2].as_str();
        if !["mean", "cv", "skew"].contains(&q) {
            continue;
        }
        let s = books
            .entry(row[1].clone())
            .or_insert_with(|| call("pro_agg_stats", book(&fields(&row[1]))));
        assert_eq!(s.columns, 2, "{}: {:?}", row[0], s.as_scalar());
        let label = match q {
            "mean" => "Mean",
            "cv" => "CV",
            _ => "Skewness",
        };
        let got = num(s.get(row_of(s, label), 1));
        let expected: f64 = row[4].parse().unwrap();
        // Exact ground up, at the reference's tolerance or 1e-11: the
        // severity's third moment comes through SeverityDist, a few ulps
        // from the family's own closed form.
        let (abs_tol, rel_tol): (f64, f64) = (row[5].parse().unwrap(), row[6].parse().unwrap());
        let rel_tol = rel_tol.max(1e-11);
        assert!(
            (got - expected).abs() <= abs_tol.max(rel_tol * expected.abs()),
            "{} {q}: {got} vs {expected}",
            row[0]
        );
        checked += 1;
    }
    assert_eq!(checked, 15);
}

#[test]
fn var_and_tvar_match_panjer_and_layers_their_closed_forms() {
    // Poisson 3 of a gamma with mean 1,000 and CV 1.5, on 4,096 buckets of
    // 10: the FFT's VaR and TVaR against Panjer's on the same severity.
    let args = vec![
        n(3.0),
        Grid::missing(),
        t("gamma_cv"),
        n(1000.0),
        n(1.5),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        Grid::from_rows(vec![vec![Value::Number(0.9), Value::Number(0.99)]]),
        n(10.0),
        n(4096.0),
    ];
    let s = call("pro_agg_stats", args);
    let sev = Gamma::from_mean_cv(1000.0, 1.5).unwrap();
    let (grid, _) = PGrid::rounding(&sev, 10.0, 4096).unwrap();
    let (agg, _) = panjer(&Poisson::new(3.0).unwrap(), &grid, 4096).unwrap();
    for (p, label) in [(0.9, "90%"), (0.99, "99%")] {
        let var = num(s.get(row_of(&s, &format!("VaR {label}")), 1));
        let tvar = num(s.get(row_of(&s, &format!("TVaR {label}")), 1));
        assert!((var - agg.quantile(p).unwrap()).abs() < 1e-9, "{p}");
        assert!((tvar - agg.tvar(p).unwrap()).abs() < 1e-6 * tvar, "{p}");
        assert!(tvar > var);
    }
    assert_eq!(num(s.get(row_of(&s, "Points"), 1)), 4096.0);
    assert!((num(s.get(row_of(&s, "Grid mean"), 1)) - agg.mean()).abs() < 1e-6);

    // A per-occurrence layer on the automatic grid: 2,000 xs 1,000 under
    // a negative binomial with dispersion 2.
    let s = call(
        "pro_agg_stats",
        vec![
            n(3.0),
            n(2.0),
            t("gamma_cv"),
            n(1000.0),
            n(1.5),
            Grid::missing(),
            n(2000.0),
            n(1000.0),
        ],
    );
    let model = CollectiveModel::new(PanjerClass::from_mean_dispersion(3.0, 2.0).unwrap(), sev);
    let mean = model.layer_mean(2000.0, 1000.0);
    let sd = model.layer_std_dev(2000.0, 1000.0);
    assert!(
        (num(s.get(0, 1)) / mean - 1.0).abs() < 1e-4,
        "{} vs {mean}",
        num(s.get(0, 1))
    );
    assert!((num(s.get(1, 1)) / sd - 1.0).abs() < 1e-4);
    assert!(num(s.get(row_of(&s, "Aliasing error"), 1)) < 1e-10);
    let var = num(s.get(row_of(&s, "VaR 99.9%"), 1));
    assert!(var > 2000.0 && var < num(s.get(row_of(&s, "TVaR 99.9%"), 1)));
}

#[test]
fn errors_come_back_as_text() {
    let base = || {
        vec![
            n(3.0),
            Grid::missing(),
            t("gamma_cv"),
            n(1000.0),
            n(1.5),
            Grid::missing(),
        ]
    };
    let e = |a: Vec<Grid>, f: &str| text(&call(f, a)).to_owned();
    let mut a = base();
    a.extend([Grid::missing(), Grid::missing(), Grid::missing(), n(10.0)]);
    assert!(e(a, "pro_agg_stats").contains("both step and points"));
    let mut a = base();
    a.extend([Grid::missing(), Grid::missing(), n(1.5)]);
    assert!(e(a, "pro_agg_stats").contains("(0, 1)"));
    let mut a = base();
    a[1] = n(-1.0);
    assert!(e(a, "pro_agg_stats").contains("dispersion"));
    let mut a = base();
    a.extend([
        Grid::from_rows(vec![vec![Value::Number(1.0), Value::Number(2.0)]]),
        Grid::from_rows(vec![vec![Value::Number(0.0); 3]]),
    ]);
    assert!(e(a, "pro_agg_layers").contains("one per limit"));
}
