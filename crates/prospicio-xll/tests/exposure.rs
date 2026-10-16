//! Exposure curves (`PRO.EXPOSURE.*`, `PRO.SWISSRE.C*`), reached through
//! the catalogue as Excel reaches them:
//!
//! * `G(x)` and the mean destruction rate of the Swiss Re curves
//!   `c = 1.5 ... 5` and three MBBEFD curves against mpmath's quadrature of
//!   the survival (`validation/reference/mbbefd_mpmath.csv`), at its
//!   tolerances;
//! * layer shares that add up, the Swiss Re calibrations recovering `c`,
//!   the quantile inverse to the MBBEFD's cdf, and a tabulated curve's
//!   closed forms.

mod common;

use common::{n, num, reference, t, text};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

/// `curve, p1, p2, p3` for a reference row's `curve` column.
fn curve(spec: &str) -> Vec<Grid> {
    let kv: Vec<(&str, f64)> = spec
        .split(';')
        .map(|p| p.split_once('=').unwrap())
        .map(|(k, v)| (k, v.parse().unwrap()))
        .collect();
    match kv.as_slice() {
        [("c", c)] => vec![t("swiss_re"), n(*c), Grid::missing(), Grid::missing()],
        [("b", b), ("g", g)] => vec![t("mbbefd"), n(*b), n(*g), Grid::missing()],
        _ => panic!("{spec}"),
    }
}

#[test]
fn curves_and_mean_rates_match_mpmath() {
    let mut checked = 0;
    for row in reference("mbbefd_mpmath.csv").into_iter().skip(1) {
        let c = curve(&row[0]);
        let got = match row[1].as_str() {
            "G" => {
                let mut a = vec![n(row[2].parse().unwrap())];
                a.extend(c);
                num(call("pro_exposure_curve", a).as_scalar().unwrap())
            }
            "mean" => {
                let stats = call("pro_exposure_stats", c);
                assert_eq!(stats.get(3, 0), &Value::from("Mean rate"));
                num(stats.get(3, 1))
            }
            other => panic!("{other}"),
        };
        let expected: f64 = row[3].parse().unwrap();
        let (abs_tol, rel_tol): (f64, f64) = (row[4].parse().unwrap(), row[5].parse().unwrap());
        assert!(
            (got - expected).abs() <= abs_tol.max(rel_tol * expected.abs()),
            "{} {} {}: {got} vs {expected}",
            row[0],
            row[1],
            row[2]
        );
        checked += 1;
    }
    assert_eq!(checked, 56);
}

fn with(first: Vec<Grid>, c: &[Grid]) -> Vec<Grid> {
    let mut a = first;
    a.extend(c.iter().cloned());
    a
}

#[test]
fn shares_add_up_and_swiss_re_c_is_recovered() {
    let c3 = [t("swiss_re"), n(3.0)];
    // A column of attachments against a row of MPLs, layers of 2.5m: the
    // four layers of a 10m risk add up to 1.
    let atts = Grid::from_rows(
        (0..4)
            .map(|k| vec![Value::Number(2.5e6 * k as f64)])
            .collect(),
    );
    let mpls = Grid::from_rows(vec![vec![Value::Number(10e6), Value::Number(20e6)]]);
    let shares = call("pro_exposure_share", with(vec![n(2.5e6), atts, mpls], &c3));
    assert_eq!((shares.rows, shares.columns), (4, 2));
    let total: f64 = (0..4).map(|r| num(shares.get(r, 0))).sum();
    assert!((total - 1.0).abs() < 1e-12);
    // Concave: each layer takes less than the one below.
    assert!((1..4).all(|r| num(shares.get(r, 0)) < num(shares.get(r - 1, 0))));
    // An unlimited layer takes everything above its attachment.
    let top = call(
        "pro_exposure_share",
        with(vec![t("unlimited"), n(5e6), n(10e6)], &c3),
    );
    let g = call("pro_exposure_curve", with(vec![n(0.5)], &c3));
    assert!((num(top.as_scalar().unwrap()) - (1.0 - num(g.as_scalar().unwrap()))).abs() < 1e-12);

    // c from a mean rate and from a layer's share.
    let mean = num(call("pro_exposure_stats", c3.to_vec()).get(3, 1));
    let c = call("pro_swissre_c", vec![n(mean)]);
    assert!((num(c.as_scalar().unwrap()) - 3.0).abs() < 1e-9);
    let c = call(
        "pro_swissre_c_layer",
        vec![n(2.5e6), n(2.5e6), n(10e6), shares.get(1, 0).clone().into()],
    );
    assert!((num(c.as_scalar().unwrap()) - 3.0).abs() < 1e-8);
}

#[test]
fn quantile_inverts_the_mbbefd_and_stats() {
    // b = 0.5, g = 4: F(x) = 1 - (1 - b) / ((g - 1) b^(1 - x) + 1 - g b) below 1,
    // a total loss with probability 1/4.
    let m = [t("mbbefd"), n(0.5), n(4.0)];
    let f = |x: f64| 1.0 - 0.5 / (3.0 * 0.5f64.powf(1.0 - x) + 1.0 - 2.0);
    for x in [0.1, 0.4, 0.7] {
        let q = call("pro_exposure_quantile", with(vec![n(f(x))], &m));
        assert!((num(q.as_scalar().unwrap()) - x).abs() < 1e-9, "{x}");
    }
    let q = call("pro_exposure_quantile", with(vec![n(0.9)], &m));
    assert_eq!(num(q.as_scalar().unwrap()), 1.0);
    let stats = call("pro_exposure_stats", m.to_vec());
    assert_eq!(stats.get(0, 1), &Value::from("mbbefd"));
    assert_eq!(num(stats.get(4, 1)), 0.25);
}

#[test]
fn tabulated_curve() {
    let row = |v: &[f64]| Grid::from_rows(vec![v.iter().map(|&x| Value::Number(x)).collect()]);
    let tab = [
        t("tabulated"),
        row(&[0.0, 0.1, 0.5, 1.0]),
        row(&[0.0, 0.4, 0.8, 1.0]),
    ];
    let g = call("pro_exposure_curve", with(vec![n(0.3)], &tab));
    assert!((num(g.as_scalar().unwrap()) - 0.6).abs() < 1e-15);
    // Slopes 4, 1, 0.4: mean rate 1/4, a total loss with probability 0.1.
    let stats = call("pro_exposure_stats", tab.to_vec());
    assert_eq!(stats.get(0, 1), &Value::from("tabulated"));
    assert!((num(stats.get(3, 1)) - 0.25).abs() < 1e-15);
    assert_eq!(stats.get(1, 1), &Value::Empty);
    let q = call("pro_exposure_quantile", with(vec![n(0.95)], &tab));
    assert_eq!(num(q.as_scalar().unwrap()), 1.0);
    // Log-log runs above the chord between points past the first.
    let mut ll = tab.to_vec();
    ll.push(t("log-log"));
    let lin = call("pro_exposure_curve", with(vec![n(0.3)], &tab));
    let log = call("pro_exposure_curve", with(vec![n(0.3)], &ll));
    assert!(num(log.as_scalar().unwrap()) > num(lin.as_scalar().unwrap()));
}

#[test]
fn errors_come_back_as_text() {
    let e = |f: &str, a: Vec<Grid>| text(&call(f, a)).to_owned();
    assert!(
        e("pro_exposure_curve", vec![n(0.5), t("salzmann"), n(1.0)])
            .contains("curve must be one of")
    );
    assert!(e("pro_exposure_curve", vec![n(1.5), t("swiss_re"), n(3.0)]).contains("[0, 1]"));
    assert!(
        e(
            "pro_exposure_curve",
            vec![n(0.5), t("swiss_re"), n(3.0), n(2.0)]
        )
        .contains("p2 is not taken")
    );
    assert!(
        e(
            "pro_exposure_share",
            vec![n(1.0), n(0.0), n(-1.0), t("swiss_re"), n(3.0)]
        )
        .contains("mpl")
    );
    assert!(e("pro_exposure_quantile", vec![n(1.0), t("swiss_re"), n(3.0)]).contains("(0, 1)"));
    assert!(e("pro_swissre_c", vec![n(1.5)]).starts_with("#PROSPICIO!"));
    let col = Grid::from_rows(vec![vec![Value::Number(1.0)], vec![Value::Number(2.0)]]);
    let three = Grid::from_rows(vec![vec![Value::Number(1.0)]; 3]);
    assert!(
        e(
            "pro_exposure_share",
            vec![col, three, n(10.0), t("swiss_re"), n(3.0)]
        )
        .contains("do not combine")
    );
}
