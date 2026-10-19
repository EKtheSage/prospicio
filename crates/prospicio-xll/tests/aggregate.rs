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
//!   model's closed forms;
//! * the occurrence PML (`PRO.AGG.PML`) against `aggregate` 1.0.1's `oep`
//!   (`oep_aggregate.csv`), and its aggregate PML against Panjer's VaR;
//! * aggregate covers (`PRO.AGG.COVER`) on the total net of occurrence
//!   layers against `reins_stats_df`'s `agg` block (books R and T), a stop
//!   loss against Panjer's on the same grid, and a cover on the
//!   occurrence layer's ceded total against `PRO.AGG.LAYERS`;
//! * a tower's views priced by a distortion (`PRO.TOWER.PRICE`) against
//!   `reins_price_df` (books C, R, S and T, PH 0.7), and annual terms
//!   against `Tower::on_grid` and the closed-form CCoC price.

mod common;

use std::collections::HashMap;

use common::{n, num, reference, t, text};
use prospicio_aggregate::pml::occurrence_exceedance;
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
fn occurrence_pml_matches_aggregate() {
    let mut checked = 0;
    let mut books: HashMap<String, Grid> = HashMap::new();
    for row in reference("oep_aggregate.csv").into_iter().skip(1) {
        let p: f64 = row[3].parse().unwrap();
        let s = books.entry(row[1].clone()).or_insert_with(|| {
            let ps: Vec<f64> = reference("oep_aggregate.csv")
                .into_iter()
                .filter(|r| r[1] == row[1] && r[2] == "loss")
                .map(|r| 1.0 / r[3].parse::<f64>().unwrap())
                .collect();
            let mut args = book(&fields(&row[1]));
            args.push(Grid::column(&ps));
            call("pro_agg_pml", args)
        });
        assert_eq!(s.columns, 6, "{}: {:?}", row[0], s.as_scalar());
        let r = (1..s.rows)
            .find(|&r| (num(s.get(r, 1)) / p - 1.0).abs() < 1e-12)
            .unwrap_or_else(|| panic!("{} no row for {p}", row[0]));
        let col = match row[2].as_str() {
            "loss" => 2,
            "severity_survival" => 4,
            "occurrence_return_period" => 5,
            other => panic!("{other}"),
        };
        let got = num(s.get(r, col));
        let expected: f64 = row[4].parse().unwrap();
        let (abs_tol, rel_tol): (f64, f64) = (row[5].parse().unwrap(), row[6].parse().unwrap());
        assert!(
            (got - expected).abs() <= abs_tol.max(rel_tol * expected.abs()),
            "{} {} at {p}: {got} vs {expected}",
            row[0],
            row[2]
        );
        assert_eq!(num(s.get(r, 0)), 1.0 / p);
        checked += 1;
    }
    assert_eq!(checked, 72);
}

#[test]
fn aggregate_pml_is_panjers_var_and_clustering_is_exact() {
    // A negative binomial with dispersion 2 on 4,096 buckets of 10: the
    // aggregate PML is Panjer's VaR on the same grid, the occurrence PML
    // inverts 1 − G_N(F(x)), and the total's PML is at least the largest
    // occurrence's.
    let nb = PanjerClass::from_mean_dispersion(3.0, 2.0).unwrap();
    let PanjerClass::NegativeBinomial(count) = nb else {
        panic!("{nb:?}")
    };
    let s = call(
        "pro_agg_pml",
        vec![
            n(3.0),
            n(2.0),
            t("gamma_cv"),
            n(1000.0),
            n(1.5),
            Grid::missing(),
            Grid::row(&[10.0, 100.0, 0.002]),
            n(10.0),
            n(4096.0),
        ],
    );
    assert_eq!((s.rows, s.columns), (4, 6), "{:?}", s.as_scalar());
    assert_eq!(s.get(0, 2), &Value::from("Occurrence PML"));
    let sev = Gamma::from_mean_cv(1000.0, 1.5).unwrap();
    let (grid, _) = PGrid::rounding(&sev, 10.0, 4096).unwrap();
    let (agg, _) = panjer(&count, &grid, 4096).unwrap();
    for (r, p) in [(1, 0.1), (2, 0.01), (3, 0.002)] {
        assert!((num(s.get(r, 1)) - p).abs() < 1e-15);
        let oep = num(s.get(r, 2));
        assert!((occurrence_exceedance(&count, &sev, oep) / p - 1.0).abs() < 1e-9);
        let aep = num(s.get(r, 3));
        assert!((aep - agg.quantile(1.0 - p).unwrap()).abs() < 1e-9, "{p}");
        assert!(aep >= oep - 10.0, "{p}: {aep} < {oep}");
    }
    // Clustering lowers the OEP at a given mean: the Poisson's PML is
    // higher.
    let poisson = call(
        "pro_agg_pml",
        vec![n(3.0), Grid::missing(), t("gamma_cv"), n(1000.0), n(1.5)],
    );
    assert_eq!(poisson.rows, 10);
    let at_100 = |g: &Grid| {
        let r = (1..g.rows).find(|&r| num(g.get(r, 0)) == 100.0).unwrap();
        num(g.get(r, 2))
    };
    assert!(at_100(&poisson) > at_100(&s));
}

#[test]
fn aggregate_covers_match_aggregate() {
    let mut checked = 0;
    let mut fits: HashMap<String, Grid> = HashMap::new();
    for row in reference("reinsurance_aggregate.csv").into_iter().skip(1) {
        let Some(stat) = row[2].strip_prefix("agglayer:") else {
            continue;
        };
        let f = fields(&row[1]);
        let ex = fits.entry(row[1].clone()).or_insert_with(|| {
            // The book's policy limit (`1000 xs 0`, `cap`) as one more
            // occurrence layer, unlimited and ceded away: the net is then
            // the capped severity's, as `aggregate` rounds it.
            let cap = Value::Number(f["cap"].parse().unwrap());
            let occ: Vec<Vec<Value>> = f["occ"]
                .split('+')
                .map(|l| {
                    l.split(':')
                        .map(|x| Value::Number(x.parse().unwrap()))
                        .collect()
                })
                .chain([vec![Value::Number(1.0), Value::from("unlimited"), cap]])
                .collect();
            let r = |k: usize| Grid::from_rows(vec![occ.iter().map(|l| l[k].clone()).collect()]);
            let (limit, retention) = f["agg"].split_once(':').unwrap();
            let mut args = book(&f);
            args.extend([
                n(limit.parse().unwrap()),
                n(retention.parse().unwrap()),
                Grid::missing(),
                r(1),
                r(2),
                r(0),
                Grid::missing(),
                n(1.0),
                n(f64::from(1 << 14)),
            ]);
            call("pro_agg_cover", args)
        });
        assert_eq!(
            (ex.rows, ex.columns),
            (9, 2),
            "{}: {:?}",
            row[0],
            ex.as_scalar()
        );
        let label = match stat {
            "pr_attach" => "Pr attach",
            "pr_detach" => "Pr detach",
            "lol" => "Loss on line",
            "agg_mean" => "Annual mean",
            "agg_cv" => "Annual CV",
            "agg_skew" => "Annual skewness",
            other => panic!("{other}"),
        };
        let got = num(ex.get(row_of(ex, label), 1));
        let expected: f64 = row[4].parse().unwrap();
        let (abs_tol, rel_tol): (f64, f64) = (row[5].parse().unwrap(), row[6].parse().unwrap());
        assert!(
            (got - expected).abs() <= abs_tol.max(rel_tol * expected.abs()),
            "{} {stat}: {got} vs {expected}",
            row[0]
        );
        checked += 1;
    }
    assert_eq!(checked, 12);
}

#[test]
fn a_stop_loss_is_panjers_and_a_cover_on_the_ceded_total_its_layer() {
    // Poisson 3 of a gamma with mean 1,000 and CV 1.5 on 4,096 buckets of
    // 10: 2,000 xs 5,000 of the year and an unlimited cover above 4,000.
    let mut args = vec![
        n(3.0),
        Grid::missing(),
        t("gamma_cv"),
        n(1000.0),
        n(1.5),
        Grid::missing(),
        Grid::from_rows(vec![vec![Value::Number(2000.0), Value::from("unlimited")]]),
        Grid::row(&[5000.0, 4000.0]),
        n(0.5),
    ];
    args.extend([
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
    ]);
    args.extend([n(10.0), n(4096.0)]);
    let s = call("pro_agg_cover", args);
    assert_eq!((s.rows, s.columns), (9, 3), "{:?}", s.as_scalar());
    assert_eq!(s.get(0, 1), &Value::from("50% of 2000 xs 5000"));
    assert_eq!(s.get(0, 2), &Value::from("50% of unlimited xs 4000"));
    let sev = Gamma::from_mean_cv(1000.0, 1.5).unwrap();
    let (grid, _) = PGrid::rounding(&sev, 10.0, 4096).unwrap();
    let (agg, _) = panjer(&Poisson::new(3.0).unwrap(), &grid, 4096).unwrap();
    let excess = |r: f64, l: f64| -> f64 {
        agg.probs()
            .iter()
            .enumerate()
            .map(|(j, p)| p * (agg.x(j) - r).clamp(0.0, l))
            .sum()
    };
    assert!((num(s.get(row_of(&s, "Subject mean"), 1)) - agg.mean()).abs() < 1e-8);
    let mean = |c: usize| num(s.get(row_of(&s, "Annual mean"), c));
    assert!((mean(1) - 0.5 * excess(5000.0, 2000.0)).abs() < 1e-8);
    assert!((mean(2) - 0.5 * excess(4000.0, f64::INFINITY)).abs() < 1e-8);
    let lol = num(s.get(row_of(&s, "Loss on line"), 1));
    assert!((lol - mean(1) / 1000.0).abs() < 1e-15);
    assert_eq!(num(s.get(row_of(&s, "Loss on line"), 2)), 0.0);
    assert_eq!(num(s.get(row_of(&s, "Pr detach"), 2)), 0.0);
    let p_attach = 1.0 - agg.cdf(4000.0);
    assert!((num(s.get(row_of(&s, "Pr attach"), 2)) - p_attach).abs() < 1e-12);

    // Unlimited xs 0 on the ceded total of 2,000 xs 1,000 per claim is
    // that layer's annual loss, as PRO.AGG.LAYERS has it.
    let base = || {
        vec![
            n(3.0),
            n(2.0),
            t("gamma_cv"),
            n(1000.0),
            n(1.5),
            Grid::missing(),
        ]
    };
    let mut a = base();
    a.extend([
        t("unlimited"),
        n(0.0),
        Grid::missing(),
        n(2000.0),
        n(1000.0),
    ]);
    a.extend([Grid::missing(), t("ceded"), n(10.0), n(4096.0)]);
    let c = call("pro_agg_cover", a);
    assert_eq!(c.columns, 2, "{:?}", c.as_scalar());
    let mut a = base();
    a.extend([n(2000.0), n(1000.0), Grid::missing(), n(10.0), n(4096.0)]);
    let l = call("pro_agg_layers", a);
    for label in ["Annual mean", "Annual SD", "Annual skewness"] {
        let (x, y) = (
            num(c.get(row_of(&c, label), 1)),
            num(l.get(row_of(&l, label), 1)),
        );
        assert!((x / y - 1.0).abs() < 1e-9, "{label}: {x} vs {y}");
    }
    assert_eq!(
        num(c.get(row_of(&c, "Subject mean"), 1)),
        num(c.get(row_of(&c, "Annual mean"), 1))
    );
}

#[test]
fn tower_prices_match_aggregate() {
    let mut checked = 0;
    let mut fits: HashMap<String, Grid> = HashMap::new();
    for row in reference("reinsurance_aggregate.csv").into_iter().skip(1) {
        let Some(rest) = row[2].strip_prefix("price:") else {
            continue;
        };
        let (view, stat) = rest.split_once(':').unwrap();
        let f = fields(&row[1]);
        let occ: Vec<Vec<f64>> = f["occ"]
            .split('+')
            .map(|l| l.split(':').map(|x| x.parse().unwrap()).collect())
            .collect();
        let ex = fits.entry(row[1].clone()).or_insert_with(|| {
            let r = |k: usize| Grid::row(&occ.iter().map(|l| l[k]).collect::<Vec<_>>());
            let (cover_limit, cover_retention) = match f.get("agg") {
                Some(a) => {
                    let (l, r) = a.split_once(':').unwrap();
                    (n(l.parse().unwrap()), n(r.parse().unwrap()))
                }
                None => (Grid::missing(), Grid::missing()),
            };
            let mut args = book(&f);
            args.extend([
                n(f["cap"].parse().unwrap()),
                Grid::missing(),
                r(1),
                r(2),
                r(0),
            ]);
            args.extend([
                Grid::missing(),
                Grid::missing(),
                cover_limit,
                cover_retention,
            ]);
            args.extend([t("ph"), n(0.7), n(1.0), n(f64::from(1 << 14))]);
            call("pro_tower_price", args)
        });
        assert_eq!(ex.columns, 8, "{}: {:?}", row[0], ex.as_scalar());
        let label = match view {
            "gross" => "Gross".to_owned(),
            "net" => "Net".to_owned(),
            "ceded_occ" if occ.len() == 1 => format!("{} xs {}", occ[0][1], occ[0][2]),
            "ceded_occ" | "stage0" => "Occurrence total".to_owned(),
            "ceded" => {
                let (l, r) = f["agg"].split_once(':').unwrap();
                format!("Aggregate cover {l} xs {r}")
            }
            other => panic!("{other}"),
        };
        let col = if stat == "L" { 1 } else { 4 };
        let got = num(ex.get(row_of(ex, &label), col));
        let expected: f64 = row[4].parse().unwrap();
        let (abs_tol, rel_tol): (f64, f64) = (row[5].parse().unwrap(), row[6].parse().unwrap());
        assert!(
            (got - expected).abs() <= abs_tol.max(rel_tol * expected.abs()),
            "{} {rest}: {got} vs {expected}",
            row[0]
        );
        checked += 1;
    }
    assert_eq!(checked, 28);
}

#[test]
fn a_tower_with_annual_terms_is_the_librarys_and_ccoc_is_closed_form() {
    use prospicio_aggregate::{Layer, Tower};
    // Poisson 3 of a gamma with mean 1,000 and CV 1.5 on 4,096 buckets of
    // 10: 2,000 xs 1,000 with an AAD of 500 and an AAL of 4,000 at 50%,
    // and 5,000 xs 3,000; CCoC at 10%.
    let mut args = vec![
        n(3.0),
        Grid::missing(),
        t("gamma_cv"),
        n(1000.0),
        n(1.5),
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        Grid::row(&[2000.0, 5000.0]),
        Grid::row(&[1000.0, 3000.0]),
        Grid::row(&[0.5, 1.0]),
        Grid::row(&[500.0, 0.0]),
        Grid::from_rows(vec![vec![Value::Number(4000.0), Value::from("unlimited")]]),
        Grid::missing(),
        Grid::missing(),
        t("ccoc"),
        n(0.1),
        n(10.0),
        n(4096.0),
    ];
    let s = call("pro_tower_price", args.clone());
    assert_eq!((s.rows, s.columns), (6, 8), "{:?}", s.as_scalar());
    let sev = Gamma::from_mean_cv(1000.0, 1.5).unwrap();
    let (grid, _) = PGrid::rounding(&sev, 10.0, 4096).unwrap();
    let tower = Tower::new(vec![
        Layer::xol("a", 2000.0, 1000.0)
            .unwrap()
            .share(0.5)
            .unwrap()
            .aggregate_deductible(500.0)
            .unwrap()
            .aggregate_limit(4000.0)
            .unwrap(),
        Layer::xol("b", 5000.0, 3000.0).unwrap(),
    ])
    .unwrap();
    let grids = tower
        .on_grid(&Poisson::new(3.0).unwrap(), &grid, 4096)
        .unwrap();
    let r = row_of(&s, "50% of 2000 xs 1000");
    let ceded = &grids.ceded[0];
    assert!((num(s.get(r, 1)) - ceded.mean()).abs() < 1e-9);
    assert!((num(s.get(r, 2)) - ceded.variance().sqrt()).abs() < 1e-9);
    // CCoC: P = (E X + r max X) / (1 + r), max X the share of the AAL.
    let p = (ceded.mean() + 0.1 * 2000.0) / 1.1;
    assert!(
        (num(s.get(r, 4)) - p).abs() < 1e-6,
        "{} vs {p}",
        num(s.get(r, 4))
    );
    assert!((num(s.get(r, 7)) - p / 1000.0).abs() < 1e-9);
    assert!((num(s.get(r, 5)) - (p - ceded.mean())).abs() < 1e-6);
    // A layer with annual terms: the total and the net are not one
    // compound, and say so.
    let note = |label: &str| match s.get(row_of(&s, label), 1) {
        Value::Text(t) => t.clone(),
        other => panic!("{label}: {other:?}"),
    };
    assert!(note("Occurrence total").contains("not one compound"));
    assert!(note("Net").contains("not one compound"));
    assert_eq!(s.get(row_of(&s, "Gross"), 7), &Value::Empty);

    // Without annual terms the net and the views add up: gross loss is
    // the layers' and the net's, up to what the gross total's tail wraps
    // around 4,096 points (3e-7 of its mean; the ceded and net are
    // lighter).
    args[11] = Grid::missing();
    args[12] = Grid::missing();
    let s = call("pro_tower_price", args);
    let loss = |label: &str| num(s.get(row_of(&s, label), 1));
    assert!(
        (loss("Gross") / (loss("Occurrence total") + loss("Net")) - 1.0).abs() < 1e-6,
        "{s:?}"
    );
    assert!(
        (loss("Occurrence total") - loss("50% of 2000 xs 1000") - loss("5000 xs 3000")).abs()
            < 1e-8
    );
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
    let mut a = base();
    a.push(n(1.0));
    assert!(e(a, "pro_agg_pml").contains("above 1"));
    let mut a = base();
    a[0] = n(0.01);
    a.push(n(10.0));
    assert!(e(a, "pro_agg_pml").contains("any occurrence"));
    let mut a = base();
    a.extend([n(100.0), Grid::missing()]);
    assert!(e(a, "pro_agg_cover").contains("retentions is required"));
    let mut a = base();
    a.extend([n(100.0), n(0.0), Grid::missing()]);
    a.extend([
        Grid::missing(),
        Grid::missing(),
        Grid::missing(),
        t("ceded"),
    ]);
    assert!(e(a, "pro_agg_cover").contains("needs occurrence layers"));
    let mut a = base();
    a.extend([n(100.0), n(0.0), Grid::missing(), n(50.0), n(0.0)]);
    a.extend([Grid::missing(), t("gross")]);
    assert!(e(a, "pro_agg_cover").contains("\"net\" (default)"));
    let tower = |distortion: &str, cover: Option<f64>| {
        let mut a = base();
        a.extend([Grid::missing(), Grid::missing(), n(500.0), n(100.0)]);
        a.extend([Grid::missing(), Grid::missing(), Grid::missing()]);
        a.extend([cover.map_or(Grid::missing(), n), Grid::missing()]);
        a.extend([t(distortion), n(0.7)]);
        e(a, "pro_tower_price")
    };
    assert!(tower("gamma", None).contains("\"ccoc\", \"ph\""));
    assert!(tower("ph", Some(1000.0)).contains("retention"));
}
