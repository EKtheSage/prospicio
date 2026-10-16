//! Clark's growth curves (`PRO.CLARK`, `PRO.CLARK.PARAMS`), reached through
//! the catalogue as Excel reaches them, against R ChainLadder 0.2.21's
//! `ClarkLDF` on RAA and `ClarkCapeCod` on GenIns with its premium, run to
//! convergence (`optim=tight` in
//! `validation/reference/reserving_clark_r.csv`, rel_tol 1e-5): every
//! parameter, every origin's and the total's ultimate, reserve and
//! standard errors, for both curves and both `max_age` settings.

mod common;

use std::collections::BTreeMap;

use common::{genins, n, num, raa, reference, t, text};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn spec(method: &str) -> (String, BTreeMap<String, String>) {
    let mut parts = method.split(';');
    let name = parts.next().unwrap().to_owned();
    let kv = parts
        .map(|p| p.split_once('=').unwrap())
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
    (name, kv)
}

#[test]
fn matches_r_chainladder() {
    let mut fits: BTreeMap<(String, String), (Grid, Grid)> = BTreeMap::new();
    let mut checked = 0;
    for row in reference("reserving_clark_r.csv") {
        let (dataset, method, quantity, arg) = (&row[0], &row[1], row[2].as_str(), &row[3]);
        let (name, s) = spec(method);
        if s["optim"] != "tight" {
            continue;
        }
        let (exhibit, params) = fits
            .entry((dataset.clone(), method.clone()))
            .or_insert_with(|| {
                let (tri, exposure) = match name.as_str() {
                    "clark_ldf" => (raa_or_genins(dataset), Grid::missing()),
                    _ => genins(),
                };
                let curve = t(&s["curve"]);
                let max_age = if s["max_age"] == "inf" {
                    t("inf")
                } else {
                    n(s["max_age"].parse().unwrap())
                };
                let exhibit = call(
                    "pro_clark",
                    vec![
                        tri.clone(),
                        exposure.clone(),
                        curve.clone(),
                        max_age.clone(),
                    ],
                );
                let params = call("pro_clark_params", vec![tri, exposure, curve, max_age]);
                (exhibit, params)
            });
        assert_eq!(
            exhibit.rows,
            12,
            "{dataset} {method}: {:?}",
            exhibit.as_scalar()
        );
        let param = |label: &str| -> f64 {
            (0..params.rows)
                .find(|&r| params.get(r, 0) == &Value::from(label))
                .map(|r| num(params.get(r, 1)))
                .unwrap()
        };
        // The row of an origin year: RAA runs 1981..1990, GenIns 2001..2010.
        let first = if dataset == "raa" { 1980 } else { 2000 };
        let origin_row = || arg.parse::<usize>().ok().map(|y| y - first);
        let got = match quantity {
            "omega" => param("Omega"),
            "theta" => param("Theta"),
            "sigma2" => param("Scale"),
            "elr" => param("ELR"),
            "total_ultimate" => num(exhibit.get(11, 2)),
            "total_reserve" => num(exhibit.get(11, 3)),
            "total_process_se" => num(exhibit.get(11, 4)),
            "total_parameter_se" => num(exhibit.get(11, 5)),
            "total_standard_error" => num(exhibit.get(11, 6)),
            "ultimate" | "reserve" | "process_se" | "parameter_se" | "standard_error" => {
                let r = origin_row().unwrap();
                let col = [
                    "ultimate",
                    "reserve",
                    "process_se",
                    "parameter_se",
                    "standard_error",
                ]
                .iter()
                .position(|q| *q == quantity)
                .unwrap()
                    + 2;
                num(exhibit.get(r, col))
            }
            _ => continue,
        };
        let expected: f64 = row[4].parse().unwrap();
        let (abs_tol, rel_tol): (f64, f64) = (row[5].parse().unwrap(), row[6].parse().unwrap());
        assert!(
            (got - expected).abs() <= abs_tol.max(rel_tol * expected.abs()),
            "{dataset} {method} {quantity} {arg}: {got} vs {expected}"
        );
        checked += 1;
    }
    assert!(checked > 300, "checked {checked}");
}

fn raa_or_genins(dataset: &str) -> Grid {
    match dataset {
        "raa" => raa(),
        _ => genins().0,
    }
}

#[test]
fn ldf_method_has_no_elr_and_errors_come_back_as_text() {
    let params = call("pro_clark_params", vec![raa()]);
    assert_eq!(params.get(0, 1), &Value::from("loglogistic"));
    assert!((0..params.rows).all(|r| params.get(r, 0) != &Value::from("ELR")));
    assert!(
        text(&call(
            "pro_clark",
            vec![raa(), Grid::missing(), t("gompertz")]
        ))
        .contains("curve must be")
    );
    assert!(
        text(&call(
            "pro_clark",
            vec![raa(), Grid::missing(), Grid::missing(), n(-1.0)]
        ))
        .contains("max_age")
    );
}
