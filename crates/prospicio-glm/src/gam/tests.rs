use super::*;
use prospicio_models::{Family, Link};

#[test]
fn constrained_basis_sums_to_zero_over_the_data() {
    let x: Vec<f64> = (0..50).map(|i| f64::from(i).sqrt()).collect();
    let b = Basis::fit(&PSpline::new("x"), &x).unwrap();
    for col in b.columns(&x) {
        assert!(col.iter().sum::<f64>().abs() < 1e-10);
    }
}

#[test]
fn huge_smoothing_gives_a_straight_line() {
    // λ → ∞ leaves only the unpenalized linear part: the GLM on x.
    let x: Vec<f64> = (0..60).map(|i| f64::from(i) / 10.0).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|v| 1.0 + 0.5 * v + (3.0 * v).sin() * 0.3)
        .collect();
    let d = Design::new(
        vec!["(Intercept)".into(), "x".into()],
        vec![vec![1.0; 60], x.clone()],
    )
    .unwrap();
    let glm = Glm::new(Family::Gaussian, Link::Identity);
    let gam = Gam::new(glm, vec![PSpline::new("x")]).smoothing(Smoothing::Fixed(vec![1e8]));
    let fit = gam.fit(&d, &y).unwrap();
    let line = glm.fit(&d, &y).unwrap();
    for (a, b) in fit.fitted().iter().zip(line.fitted()) {
        assert!((a - b).abs() < 1e-6, "{a} {b}");
    }
    assert!((fit.edf() - 2.0).abs() < 1e-6);
    // At λ → ∞ the log-likelihood is the line's, and the AIC counts the
    // edf, here the line's two coefficients.
    assert!((fit.log_likelihood() - line.log_likelihood()).abs() < 1e-5);
    assert!((fit.aic() - line.aic()).abs() < 1e-5);
}

#[test]
fn the_log_likelihood_is_the_familys_at_the_fit_and_round_trips() {
    let x: Vec<f64> = (0..80).map(|i| f64::from(i) / 10.0).collect();
    let y: Vec<f64> = x.iter().map(|v| ((2.0 * v).sin() + 1.5).round()).collect();
    let d = Design::new(
        vec!["(Intercept)".into(), "x".into()],
        vec![vec![1.0; 80], x.clone()],
    )
    .unwrap();
    let glm = Glm::new(Family::Poisson, Link::Log);
    let fit = Gam::new(glm, vec![PSpline::new("x")]).fit(&d, &y).unwrap();
    let want: f64 = y
        .iter()
        .zip(fit.fitted())
        .map(|(&k, &m)| k * m.ln() - m - ln_gamma(k + 1.0))
        .sum();
    assert!((fit.log_likelihood() - want).abs() < 1e-9);
    assert!((fit.aic() - (-2.0 * want + 2.0 * fit.edf())).abs() < 1e-9);
    let back = GamFit::from_json(&fit.to_json()).unwrap();
    assert_eq!(back.log_likelihood(), fit.log_likelihood());
}
