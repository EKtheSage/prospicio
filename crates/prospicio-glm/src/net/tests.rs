use super::*;

fn design() -> (Design, Vec<f64>) {
    let n = 40u32;
    let x1: Vec<f64> = (0..n).map(|i| f64::from(i) / 4.0).collect();
    let x2: Vec<f64> = (0..n).map(|i| (f64::from(i) * 7.3).sin()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let k = i as usize;
            2.0 + 0.8 * x1[k] + 0.3 * x2[k] + (f64::from(i) * 3.1).cos()
        })
        .collect();
    let d = Design::new(
        vec!["(Intercept)".into(), "x1".into(), "x2".into()],
        vec![vec![1.0; n as usize], x1, x2],
    )
    .unwrap();
    (d, y)
}

/// At `λ = 0` the elastic net is the unpenalized GLM.
#[test]
fn lambda_zero_is_the_glm() {
    let (d, y) = design();
    let net = ElasticNet::new(Family::Gaussian, Link::Identity, 0.5, 0.0)
        .fit(&d, &y)
        .unwrap();
    let glm = Glm::new(Family::Gaussian, Link::Identity)
        .fit(&d, &y)
        .unwrap();
    for (a, b) in net.coefficients().iter().zip(glm.coefficients()) {
        assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()), "{a} vs {b}");
    }
}

/// The KKT conditions hold for the lasso: zero coefficients have
/// gradients within `λ`, non-zero ones exactly `-λ sign(b)`.
#[test]
fn lasso_kkt() {
    let (d, y) = design();
    let spec = ElasticNet::new(Family::Gaussian, Link::Identity, 1.0, 0.0);
    let lmax = spec.lambda_max(&d, &y).unwrap();
    for ratio in [0.9, 0.3, 0.05] {
        let lambda = lmax * ratio;
        let fit = spec.with_lambda(lambda).fit(&d, &y).unwrap();
        let prep = spec.prepare(&d, &y).unwrap();
        let eta = d.linear_predictor(fit.coefficients());
        let grad = prep.gradient(&spec, &d, &y, &eta);
        for (j, (&b, g)) in fit.coefficients().iter().zip(&grad).enumerate().skip(1) {
            if b == 0.0 {
                assert!(g.abs() <= lambda * (1.0 + 1e-9));
            } else {
                assert!((g + lambda * b.signum()).abs() < 1e-8 * lambda, "{j}");
            }
        }
    }
}

/// A Poisson lasso on 15 hinges `max(0, age - k)` at neighbouring
/// knots, non-negative: the collinear basis on which plain coordinate
/// descent creeps. Each fit along the path meets the optimality
/// conditions with its bounds: a zero coefficient's gradient points
/// out of the feasible side by at most `λ`, a free one balances it.
#[test]
fn hinges_under_bounds_meet_the_optimality_conditions() {
    let ages: Vec<f64> = (0..300u32).map(|i| 18.0 + f64::from(i % 60)).collect();
    let y: Vec<f64> = (0..300u32)
        .map(|i| {
            let a = ages[i as usize];
            let rate = 1.0 + 2.0 * (-(a - 18.0) / 10.0).exp() + 0.01 * (a - 50.0).max(0.0);
            (rate * (1.0 + 0.3 * (f64::from(i) * 2.7).sin())).round()
        })
        .collect();
    let mut names = vec!["(Intercept)".to_string()];
    let mut cols = vec![vec![1.0; ages.len()]];
    for j in 0..15u32 {
        let knot = 20.0 + 3.0 * f64::from(j);
        names.push(format!("h{j}"));
        cols.push(ages.iter().map(|a| (a - knot).max(0.0)).collect());
    }
    let d = Design::new(names, cols).unwrap();
    let mut lower = vec![0.0; 16];
    lower[1] = f64::NEG_INFINITY;
    let spec = ElasticNet::new(Family::Poisson, Link::Log, 1.0, 0.0).lower_limits(lower);
    let lambdas = spec.lambda_path(&d, &y, 20, 1e-4).unwrap();
    let prep = spec.prepare(&d, &y).unwrap();
    let mut binding = false;
    for fit in spec.path(&d, &y, &lambdas).unwrap() {
        let lambda = fit.lambda();
        let eta = d.linear_predictor(fit.coefficients());
        let grad = prep.gradient(&spec, &d, &y, &eta);
        for (j, (&b, &g)) in fit.coefficients().iter().zip(&grad).enumerate().skip(1) {
            // On the standardized scale, where the penalty is λ pfⱼ.
            let pen = lambda * prep.pf[j];
            let tol = 1e-6 * lambda;
            if b == 0.0 && j > 1 {
                // At the bound 0: only an increase is allowed.
                assert!(g >= -pen - tol, "h{} at {lambda}: {g}", j - 1);
                binding |= g > pen + tol;
            } else if b == 0.0 {
                assert!(g.abs() <= pen + tol);
            } else {
                assert!((g + pen * b.signum()).abs() < tol, "h{} at {lambda}", j - 1);
            }
            assert!(j == 1 || b >= 0.0);
        }
    }
    // Some bound holds a coefficient the lasso alone would make negative.
    assert!(binding);
}

/// Warm-started paths give the cold-start fits.
#[test]
fn path_matches_single_fits() {
    let (d, y) = design();
    let spec = ElasticNet::new(Family::Gaussian, Link::Identity, 0.5, 0.0);
    let lambdas = spec.lambda_path(&d, &y, 8, 1e-3).unwrap();
    let path = spec.path(&d, &y, &lambdas).unwrap();
    for fit in &path {
        let cold = spec.with_lambda(fit.lambda()).fit(&d, &y).unwrap();
        for (a, b) in fit.coefficients().iter().zip(cold.coefficients()) {
            assert!((a - b).abs() < 1e-8 * (1.0 + b.abs()));
        }
    }
    // At λ_max itself the penalized coefficients are zero to rounding.
    assert!(path[0].coefficients()[1..].iter().all(|&b| b.abs() < 1e-10));
}

#[test]
fn rejects_bad_parameters() {
    let (d, y) = design();
    assert!(
        ElasticNet::new(Family::Gaussian, Link::Identity, 1.5, 1.0)
            .fit(&d, &y)
            .is_err()
    );
    assert!(
        ElasticNet::new(Family::Gaussian, Link::Identity, 1.0, -1.0)
            .fit(&d, &y)
            .is_err()
    );
    let bad = ElasticNet::new(Family::Gaussian, Link::Identity, 1.0, 1.0).penalty_factor(vec![1.0]);
    assert!(bad.fit(&d, &y).is_err());
}
