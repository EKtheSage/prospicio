//! Classical credibility (`PRO.CRED.STANDARD`, `PRO.CRED.Z`,
//! `PRO.BUHLMANN`, `PRO.BUHLMANN.PARAMS`), reached through the catalogue
//! as Excel reaches it, against R's `actuar` 3.3.7
//! (`validation/reference/credibility_actuar.csv`): the Bühlmann–Straub
//! model on Hachemeister's data with the unbiased and the iterative
//! estimator, Bühlmann's model (no weights), a fit with blank cells, and
//! the limited fluctuation standards, every value at the file's
//! tolerances.

mod common;

use common::{n, num, reference, t, text};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

/// Hachemeister's ratios and weights as 5 x 12 ranges.
fn hachemeister() -> (Vec<Vec<Value>>, Vec<Vec<Value>>) {
    let mut ratios = vec![vec![Value::Empty; 12]; 5];
    let mut weights = ratios.clone();
    for row in reference("hachemeister.csv").into_iter().skip(1) {
        let i: usize = row[0].parse().unwrap();
        let j: usize = row[1].parse().unwrap();
        ratios[i - 1][j - 1] = Value::Number(row[2].parse().unwrap());
        weights[i - 1][j - 1] = Value::Number(row[3].parse().unwrap());
    }
    (ratios, weights)
}

#[test]
fn matches_actuar() {
    let (ratios, weights) = hachemeister();
    let mut checked = 0;
    for row in reference("credibility_actuar.csv").into_iter().skip(1) {
        let (kind, params, quantity) = (row[0].as_str(), row[1].as_str(), row[2].as_str());
        let field = |k: &str| {
            params
                .split(';')
                .find_map(|kv| kv.strip_prefix(&format!("{k}=")))
                .map(str::to_owned)
        };
        let got = if kind == "limited_fluctuation" {
            let p: f64 = field("p").unwrap().parse().unwrap();
            let k: f64 = field("k").unwrap().parse().unwrap();
            num(call("pro_cred_standard", vec![n(p), n(k)])
                .as_scalar()
                .unwrap())
        } else {
            let (mut r, mut w) = (ratios.clone(), weights.clone());
            if let Some(cells) = field("missing") {
                for cell in cells.split('|') {
                    let (i, j) = cell.split_once('.').unwrap();
                    let (i, j): (usize, usize) = (i.parse().unwrap(), j.parse().unwrap());
                    r[i - 1][j - 1] = Value::Empty;
                    w[i - 1][j - 1] = Value::Empty;
                }
            }
            let weights = if field("weights").as_deref() == Some("yes") {
                Grid::from_rows(w)
            } else {
                Grid::missing()
            };
            let args = vec![Grid::from_rows(r), weights, t(&field("estimator").unwrap())];
            let group = || row[3].parse::<usize>().unwrap();
            match quantity {
                "credibility" | "premium" => {
                    let ex = call("pro_buhlmann", args);
                    assert_eq!((ex.rows, ex.columns), (6, 5), "{:?}", ex.as_scalar());
                    num(ex.get(group(), if quantity == "credibility" { 3 } else { 4 }))
                }
                _ => {
                    let ps = call("pro_buhlmann_params", args);
                    let r = match quantity {
                        "collective_mean" => 0,
                        "between_variance" => 1,
                        "within_variance" => 2,
                        other => panic!("{other}"),
                    };
                    num(ps.get(r, 1))
                }
            }
        };
        let expected: f64 = row[4].parse().unwrap();
        let (abs_tol, rel_tol): (f64, f64) = (row[5].parse().unwrap(), row[6].parse().unwrap());
        assert!(
            (got - expected).abs() <= abs_tol.max(rel_tol * expected.abs()),
            "{kind} {params} {quantity} {}: {got} vs {expected}",
            row[3]
        );
        checked += 1;
    }
    assert!(checked >= 57, "checked {checked}");
}

#[test]
fn standards_table_square_root_rule_and_errors() {
    // A column of p against a row of k: the textbook table, 1,082 at 90%/5%.
    let ps = Grid::from_rows(vec![vec![Value::Number(0.9)], vec![Value::Number(0.95)]]);
    let ks = Grid::from_rows(vec![vec![Value::Number(0.05), Value::Number(0.1)]]);
    let table = call("pro_cred_standard", vec![ps, ks]);
    assert_eq!((table.rows, table.columns), (2, 2));
    assert!((num(table.get(0, 0)) - 1082.21).abs() < 0.01);
    // Halving k quadruples the standard.
    assert!((num(table.get(0, 0)) / num(table.get(0, 1)) - 4.0).abs() < 1e-12);
    // Severity and pure premium standards scale by cv^2 and 1 + cv^2.
    let f = num(table.get(0, 0));
    let sev = call(
        "pro_cred_standard",
        vec![n(0.9), n(0.05), t("severity"), n(2.0)],
    );
    let pp = call(
        "pro_cred_standard",
        vec![n(0.9), n(0.05), t("pure_premium"), n(2.0)],
    );
    assert!((num(sev.as_scalar().unwrap()) - 4.0 * f).abs() < 1e-9);
    assert!((num(pp.as_scalar().unwrap()) - 5.0 * f).abs() < 1e-9);

    let zs = Grid::from_rows(vec![vec![
        Value::Number(-5.0),
        Value::Number(270.55),
        Value::Number(5000.0),
    ]]);
    let z = call("pro_cred_z", vec![zs, n(1082.2)]);
    assert_eq!(num(z.get(0, 0)), 0.0);
    assert!((num(z.get(0, 1)) - 0.5).abs() < 1e-4);
    assert_eq!(num(z.get(0, 2)), 1.0);

    let e = |f: &str, a: Vec<Grid>| text(&call(f, a)).to_owned();
    assert!(e("pro_cred_standard", vec![n(0.9), n(0.05), t("severity")]).contains("cv"));
    assert!(e("pro_cred_standard", vec![n(0.9), n(0.05), t("loss")]).contains("kind must be"));
    assert!(e("pro_cred_z", vec![n(1.0), n(0.0)]).contains("positive"));
    let one = Grid::from_rows(vec![vec![Value::Number(1.0), Value::Number(2.0)]]);
    assert!(e("pro_buhlmann", vec![one.clone()]).contains("two groups"));
    let (r, _) = hachemeister();
    assert!(e("pro_buhlmann", vec![Grid::from_rows(r.clone()), one]).contains("shape of ratios"));
    assert!(
        e(
            "pro_buhlmann",
            vec![Grid::from_rows(r), Grid::missing(), t("bayes")]
        )
        .contains("estimator must be")
    );
}

#[test]
fn group_labels() {
    let (r, w) = hachemeister();
    let labels = Grid::from_rows(vec![
        ["CA", "NY", "TX", "FL", "IL"].map(Value::from).to_vec(),
    ]);
    let ex = call(
        "pro_buhlmann",
        vec![
            Grid::from_rows(r),
            Grid::from_rows(w),
            Grid::missing(),
            labels,
        ],
    );
    assert_eq!(ex.get(0, 0), &Value::from("Group"));
    assert_eq!(ex.get(4, 0), &Value::from("FL"));
}
