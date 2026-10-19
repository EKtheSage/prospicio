//! The copula functions, reached through the catalogue as Excel reaches
//! them: the Gaussian and Archimedean distribution functions against
//! mpmath at 30 digits (`validation/reference/copula_mpmath.csv`, written
//! by `validation/scripts/mpmath_copula.py`) at its tolerances, simulated
//! uniforms against their copulas' rank correlations and the crate's own
//! draws, and Iman–Conover against `prospicio_prob`'s.

mod common;

use common::{n, num, reference, t, text};
use prospicio_core::StreamRng;
use prospicio_prob::copula::{Archimedean, ArchimedeanCopula, Copula, iman_conover};
use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn param(params: &str, k: &str) -> f64 {
    params
        .split(';')
        .find_map(|kv| kv.strip_prefix(&format!("{k}=")))
        .unwrap()
        .parse()
        .unwrap()
}

#[test]
fn cdfs_match_mpmath() {
    let mut checked = 0;
    for row in reference("copula_mpmath.csv") {
        let (family, params) = (row[0].as_str(), row[1].as_str());
        let (Ok(u), Ok(v)) = (row[3].parse::<f64>(), row[4].parse::<f64>()) else {
            continue;
        };
        let (parameter, mut point) = match family {
            "gaussian" => (param(params, "rho"), vec![u, v]),
            "clayton" | "gumbel" | "frank" | "joe" => (param(params, "theta"), vec![u, v]),
            _ => continue,
        };
        if params.contains("w=") {
            point.push(param(params, "w"));
        }
        let got = num(&call(
            "pro_copula_cdf",
            vec![Grid::row(&point), t(family), n(parameter)],
        )
        .cells[0]);
        let (want, abs, rel): (f64, f64, f64) = (
            row[5].parse().unwrap(),
            row[6].parse().unwrap(),
            row[7].parse().unwrap(),
        );
        assert!(
            (got - want).abs() <= abs.max(rel * want.abs()),
            "{family} {params} at {point:?}: {got} against {want}"
        );
        checked += 1;
    }
    assert_eq!(checked, 39);
}

/// Several points in one call, one a row; a matrix gives the Gaussian its
/// correlation as a single number does.
#[test]
fn points_are_rows() {
    let points = Grid::from_rows(vec![
        vec![Value::Number(0.5), Value::Number(0.5)],
        vec![Value::Number(0.3), Value::Number(1.0)],
        vec![Value::Number(0.0), Value::Number(0.7)],
    ]);
    let r = Grid::from_rows(vec![
        vec![Value::Number(1.0), Value::Number(0.5)],
        vec![Value::Number(0.5), Value::Number(1.0)],
    ]);
    let a = call(
        "pro_copula_cdf",
        vec![points.clone(), t("gaussian"), n(0.5)],
    );
    let b = call("pro_copula_cdf", vec![points, t("gaussian"), r]);
    assert_eq!(a, b);
    assert_eq!((a.rows, a.columns), (3, 1));
    assert!((num(a.get(0, 0)) - 1.0 / 3.0).abs() < 1e-15);
    assert_eq!(num(a.get(1, 0)), 0.3);
    assert_eq!(num(a.get(2, 0)), 0.0);
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let ranks = |v: &[f64]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].total_cmp(&v[j]));
        let mut r = vec![0.0; v.len()];
        for (k, &i) in idx.iter().enumerate() {
            r[i] = k as f64;
        }
        r
    };
    let (rx, ry) = (ranks(x), ranks(y));
    let m = (x.len() as f64 - 1.0) / 2.0;
    let cov: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - m) * (b - m)).sum();
    let var: f64 = rx.iter().map(|a| (a - m) * (a - m)).sum();
    cov / var
}

fn column(g: &Grid, j: usize) -> Vec<f64> {
    (0..g.rows).map(|i| num(g.get(i, j))).collect()
}

/// Row `i` is stream `i` of the seed, as the crate's `simulate` draws it;
/// the Gaussian's rank correlation is `(6 / π) asin(ρ / 2)`.
#[test]
fn simulated_uniforms_follow_their_copula() {
    let s = call(
        "pro_copula_sim",
        vec![t("clayton"), n(2.0), n(2000.0), n(3.0), n(7.0)],
    );
    assert_eq!((s.rows, s.columns), (2000, 3));
    let c = ArchimedeanCopula::new(Archimedean::Clayton, 2.0, 3).unwrap();
    let mut u = [0.0; 3];
    for i in [0, 1, 1999] {
        c.sample(&mut StreamRng::new(7, i as u64), &mut u);
        assert_eq!(column(&s, 0)[i], u[0]);
        assert_eq!(column(&s, 2)[i], u[2]);
    }
    assert!(
        s.cells
            .iter()
            .all(|v| matches!(v, Value::Number(x) if *x > 0.0 && *x < 1.0))
    );
    assert_eq!(
        s,
        call(
            "pro_copula_sim",
            vec![t("clayton"), n(2.0), n(2000.0), n(3.0), n(7.0)]
        )
    );

    let g = call("pro_copula_sim", vec![t("gaussian"), n(0.5), n(20000.0)]);
    assert_eq!((g.rows, g.columns), (20000, 2));
    let want = 6.0 / std::f64::consts::PI * (0.25f64).asin();
    let got = spearman(&column(&g, 0), &column(&g, 1));
    assert!((got - want).abs() < 0.02, "{got} vs {want}");

    let tc = call(
        "pro_copula_sim",
        vec![t("t"), n(0.5), n(100.0), n(4.0), Grid::missing(), n(3.0)],
    );
    assert_eq!((tc.rows, tc.columns), (100, 4));
}

/// Each column keeps its values; their rank correlation comes near the
/// target; the sheet's result is the crate's.
#[test]
fn iman_conover_keeps_the_marginals() {
    let n_sims = 2000;
    let x: Vec<f64> = (0..n_sims).map(|i| (i as f64).powf(1.3)).collect();
    let y: Vec<f64> = (0..n_sims).map(|i| ((i * 7919) % n_sims) as f64).collect();
    let draws = Grid::from_rows(
        x.iter()
            .zip(&y)
            .map(|(a, b)| vec![Value::Number(*a), Value::Number(*b)])
            .collect(),
    );
    let out = call("pro_iman_conover", vec![draws.clone(), n(0.7), n(3.0)]);
    assert_eq!((out.rows, out.columns), (n_sims, 2));
    for (j, original) in [&x, &y].into_iter().enumerate() {
        let mut a = column(&out, j);
        let mut b = original.clone();
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        assert_eq!(a, b);
    }
    let rho = spearman(&column(&out, 0), &column(&out, 1));
    assert!((rho - 0.7).abs() < 0.03, "{rho}");

    let flat: Vec<f64> = x.iter().zip(&y).flat_map(|(a, b)| [*a, *b]).collect();
    let pd = PredictiveDistribution::from_draws(
        vec!["column".into()],
        vec![vec![KeyValue::Int(0)], vec![KeyValue::Int(1)]],
        flat,
        Provenance::new("test"),
    )
    .unwrap();
    let joined = iman_conover(&pd, &[1.0, 0.7, 0.7, 1.0], 3).unwrap();
    for i in [0, 1, n_sims - 1] {
        assert_eq!(
            joined.row(i).unwrap(),
            &[num(out.get(i, 0)), num(out.get(i, 1))]
        );
    }

    // t scores and shuffled rows keep the marginals too.
    let s = call(
        "pro_iman_conover",
        vec![
            draws,
            n(0.7),
            n(3.0),
            n(4.0),
            Grid::scalar(Value::Bool(true)),
        ],
    );
    let first = column(&s, 0);
    assert!(first.windows(2).any(|w| w[1] < w[0]));
    let mut a = first;
    a.sort_by(f64::total_cmp);
    assert_eq!(a, x);
}

#[test]
fn bad_inputs_are_named() {
    let e = call(
        "pro_copula_cdf",
        vec![Grid::row(&[0.5, 0.5]), t("t"), n(0.5), n(3.0)],
    );
    assert!(text(&e).contains("no closed-form"), "{}", text(&e));
    let e = call(
        "pro_copula_cdf",
        vec![Grid::row(&[0.5, 0.5, 0.5]), t("gaussian"), n(0.5)],
    );
    assert!(text(&e).contains("two dimensions"), "{}", text(&e));
    let e = call(
        "pro_copula_cdf",
        vec![Grid::row(&[0.5, 0.5]), t("clayton"), n(2.0), n(3.0)],
    );
    assert!(
        text(&e).contains("nu is taken only by the t"),
        "{}",
        text(&e)
    );
    let e = call(
        "pro_copula_cdf",
        vec![Grid::row(&[0.5, 0.5]), t("plackett"), n(2.0)],
    );
    assert!(text(&e).contains("family must be"), "{}", text(&e));
    let e = call("pro_copula_sim", vec![t("t"), n(0.5), n(10.0)]);
    assert!(text(&e).contains("nu"), "{}", text(&e));
    let r = Grid::from_rows(vec![
        vec![Value::Number(1.0), Value::Number(0.5)],
        vec![Value::Number(0.5), Value::Number(1.0)],
    ]);
    let e = call("pro_copula_sim", vec![t("gaussian"), r, n(10.0), n(3.0)]);
    assert!(text(&e).contains("dim is 3"), "{}", text(&e));
    let e = call(
        "pro_iman_conover",
        vec![Grid::column(&[1.0, 2.0, 3.0]), n(0.5)],
    );
    assert!(text(&e).contains("at least two columns"), "{}", text(&e));
}
