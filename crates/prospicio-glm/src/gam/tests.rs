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

#[test]
fn canonical_links_have_the_expected_information_as_observed() {
    // REML reuses IRLS's expected information for these pairs (#969):
    // check it equals the observed information, with weights and an offset.
    let n = 200;
    let x: Vec<f64> = (0..n).map(|i| f64::from(i) / 50.0).collect();
    let w: Vec<f64> = (0..n).map(|i| 0.5 + f64::from(i % 5) / 4.0).collect();
    let offset: Vec<f64> = (0..n).map(|i| 0.1 * f64::from(i % 3)).collect();
    let u = |i: i32| (f64::from(i) * 0.618_034).fract();
    let cases: [(Family, Link, Vec<f64>); 5] = [
        (
            Family::Gaussian,
            Link::Identity,
            (0..n).map(|i| 1.0 + u(i)).collect(),
        ),
        (
            Family::Poisson,
            Link::Log,
            (0..n).map(|i| (2.0 * u(i)).floor()).collect(),
        ),
        (
            Family::Binomial,
            Link::Logit,
            (0..n).map(|i| f64::from(u(i) > 0.4)).collect(),
        ),
        (
            Family::Gamma,
            Link::Inverse,
            (0..n).map(|i| 2.0 + u(i)).collect(),
        ),
        (
            Family::InverseGaussian,
            Link::InverseSquared,
            (0..n).map(|i| 2.0 + u(i)).collect(),
        ),
    ];
    for (family, link, y) in cases {
        assert!(canonical(family, link));
        let glm = Glm::new(family, link);
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; 200], x.clone()],
        )
        .unwrap()
        .with_weights(w.clone())
        .unwrap()
        .with_offset(if link == Link::Log {
            offset.clone()
        } else {
            vec![0.0; 200]
        })
        .unwrap();
        let fit = irls(&glm, &d, &y, |i| family.initial_mu(y[i], w[i], 2.0), None).unwrap();
        let observed = observed_information(&glm, &d, &y, &fit.mu);
        let scale = observed.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for (a, b) in observed.iter().zip(&fit.information) {
            assert!(
                (a - b).abs() <= 1e-9 * scale,
                "{family:?} {link:?}: {a} vs {b}"
            );
        }
    }
    assert!(!canonical(Family::Gamma, Link::Log));
    assert!(!canonical(
        Family::NegativeBinomial { theta: 1.0 },
        Link::Log
    ));
}

#[test]
fn saturated_gamma_by_weight_groups_matches_the_row_sum() {
    let y: Vec<f64> = (0..300).map(|i| 0.5 + f64::from(i % 17) / 3.0).collect();
    let w: Vec<f64> = (0..300).map(|i| [1.0, 2.5, 0.0, 1.0][i % 4]).collect();
    for family in [Family::Gamma, Family::Gaussian, Family::Poisson] {
        let sat = Saturated::new(family, &y, &w).unwrap();
        for phi in [0.01, 0.3, 1.0, 7.0] {
            let want = saturated_log_likelihood(family, &y, &w, phi).unwrap();
            let got = sat.at(&y, &w, phi);
            assert!(
                (got - want).abs() <= 1e-11 * want.abs(),
                "{family:?} {phi}: {got} vs {want}"
            );
        }
    }
    assert!(Saturated::new(Family::Tweedie { power: 1.5 }, &y, &w).is_err());
}
