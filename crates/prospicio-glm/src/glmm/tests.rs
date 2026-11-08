use super::*;

#[test]
fn gauss_hermite_integrates_polynomials() {
    let (x, lw) = gauss_hermite(7);
    let w: Vec<f64> = x
        .iter()
        .zip(&lw)
        .map(|(xi, l)| (l - xi * xi).exp())
        .collect();
    let pi = std::f64::consts::PI;
    let m = |k: i32| {
        x.iter()
            .zip(&w)
            .map(|(xi, wi)| wi * xi.powi(k))
            .sum::<f64>()
    };
    assert!((m(0) - pi.sqrt()).abs() < 1e-13);
    assert!((m(2) - pi.sqrt() / 2.0).abs() < 1e-13);
    assert!((m(4) - 3.0 * pi.sqrt() / 4.0).abs() < 1e-12);
    assert!((m(12) - 10395.0 * pi.sqrt() / 64.0).abs() < 1e-9);
}

/// Forty rows in eight groups with a group effect and one covariate.
fn book() -> (Design, Vec<String>, Vec<f64>) {
    let n = 40;
    let x: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64 / 10.0 - 0.5).collect();
    let labels: Vec<String> = (0..n).map(|i| format!("g{}", i % 8)).collect();
    let effect = [0.3, -0.2, 0.5, -0.4, 0.1, 0.0, -0.1, 0.2];
    let y: Vec<f64> = (0..n)
        .map(|i| {
            (2.0 * (1.0 + 0.4 * x[i] + effect[i % 8]).exp()
                * (1.0 + 0.3 * ((i * 13) % 5) as f64 - 0.6))
                .round()
                .max(0.0)
        })
        .collect();
    let design = Design::new(
        vec!["(Intercept)".into(), "x".into()],
        vec![vec![1.0; n], x],
    )
    .unwrap();
    (design, labels, y)
}

#[test]
fn one_node_is_the_laplace_fit() {
    let (design, labels, y) = book();
    let g = [Grouping::new("g", labels)];
    let laplace = Glmm::new(Family::Poisson).fit(&design, &g, &y).unwrap();
    let one = Glmm::new(Family::Poisson)
        .quadrature(1)
        .fit(&design, &g, &y)
        .unwrap();
    assert!(laplace.converged() && one.converged());
    assert!((laplace.log_likelihood() - one.log_likelihood()).abs() < 1e-9);
    for (a, b) in laplace.coefficients().iter().zip(one.coefficients()) {
        assert!((a - b).abs() < 1e-7, "{a} {b}");
    }
    // More nodes move the fit only a little.
    let nine = Glmm::new(Family::Poisson)
        .quadrature(9)
        .fit(&design, &g, &y)
        .unwrap();
    assert!((nine.log_likelihood() - laplace.log_likelihood()).abs() < 0.1);
    assert!((nine.sd()[0] - laplace.sd()[0]).abs() < 0.05 * laplace.sd()[0]);
}

#[test]
fn the_fit_is_a_maximum_and_predicts_by_label() {
    let (design, labels, y) = book();
    let g = [Grouping::new("g", labels.clone())];
    let fit = Glmm::new(Family::Poisson).fit(&design, &g, &y).unwrap();
    assert_eq!(fit.random_effects()[0].levels.len(), 8);
    // The training rows' means are the predictions at their labels.
    let mu = fit.predict(&design, &g).unwrap();
    for (a, b) in mu.iter().zip(fit.fitted()) {
        assert!((a - b).abs() < 1e-9 * b);
    }
    // Without the factor, or at an unseen level, the population mean.
    let population = fit.predict(&design, &[]).unwrap();
    let unseen = fit
        .predict(&design, &[Grouping::new("g", vec!["new".into(); 40])])
        .unwrap();
    assert_eq!(population, unseen);
    let re = &fit.random_effects()[0];
    for j in 0..8 {
        assert!(re.conditional_variance[0][j] > 0.0);
        assert!(re.prediction_variance[0][j] > re.conditional_variance[0][j]);
    }
    assert!(fit.standard_errors().iter().all(|s| *s > 0.0));
    assert!(fit.aic() > -2.0 * fit.log_likelihood());
}

#[test]
fn groups_without_an_effect_run_to_a_zero_sd() {
    // Every group has the same responses, so the maximum is at σ = 0.
    let n = 40;
    let y: Vec<f64> = (0..n).map(|i| [1.0, 3.0, 2.0, 5.0, 0.0][i / 8]).collect();
    let labels: Vec<String> = (0..n).map(|i| format!("g{}", i % 8)).collect();
    let design = Design::new(vec!["(Intercept)".into()], vec![vec![1.0; n]]).unwrap();
    let fit = Glmm::new(Family::Poisson)
        .fit(&design, &[Grouping::new("g", labels)], &y)
        .unwrap();
    // The search stops at the boundary instead of at the iteration
    // limit; β and its SE are the GLM's: log 2.2 and 1 / √(40 · 2.2).
    assert!(fit.converged() && fit.iterations() < 100);
    assert!(fit.sd()[0] < 1e-4);
    assert!(fit.sd_standard_errors()[0].is_nan());
    assert!((fit.coefficients()[0] - 2.2f64.ln()).abs() < 1e-6);
    assert!((fit.standard_errors()[0] - 1.0 / 88f64.sqrt()).abs() < 1e-5);
    let effects = &fit.random_effects()[0];
    assert!(effects.values[0].iter().all(|v| v.abs() < 1e-4));
    assert!(effects.prediction_variance[0].iter().all(|v| v.is_finite()));
}

#[test]
fn slopes_predict_by_label_and_covariate() {
    let (design, labels, y) = book();
    let x = design.column(1).to_vec();
    let g = [Grouping::new("g", labels.clone()).slope("x", x.clone())];
    let fit = Glmm::new(Family::Poisson).fit(&design, &g, &y).unwrap();
    assert!(fit.converged());
    assert_eq!(fit.group_names(), ["g", "g"]);
    assert_eq!(fit.term_names(), [INTERCEPT, "x"]);
    assert_eq!(fit.parameters().len(), 2 + 3);
    let re = &fit.random_effects()[0];
    assert_eq!(re.terms, [INTERCEPT, "x"]);
    assert_eq!((re.values.len(), re.values[1].len()), (2, 8));
    let r = &fit.correlations()[0];
    assert!(r[0] == 1.0 && r[3] == 1.0 && (r[1] - r[2]).abs() < 1e-15 && r[1].abs() < 1.0);
    // The training rows' means are the predictions at their labels and
    // covariates.
    let mu = fit.predict(&design, &g).unwrap();
    for (a, b) in mu.iter().zip(fit.fitted()) {
        assert!((a - b).abs() < 1e-9 * b);
    }
    // A factor given without the slope's values is an error.
    let bare = [Grouping::new("g", labels.clone())];
    assert!(
        fit.predict(&design, &bare)
            .unwrap_err()
            .to_string()
            .contains("slope x")
    );
    // The slope adds a parameter for its sd and one for the
    // correlation, and cannot lower the likelihood.
    let intercepts = Glmm::new(Family::Poisson)
        .fit(&design, &[Grouping::new("g", labels.clone())], &y)
        .unwrap();
    assert!(fit.log_likelihood() >= intercepts.log_likelihood() - 1e-6);
    // Independent terms: a factor for each, on the same labels.
    let split = [
        Grouping::new("g", labels.clone()),
        Grouping::new("g", labels).without_intercept().slope("x", x),
    ];
    let diag = Glmm::new(Family::Poisson).fit(&design, &split, &y).unwrap();
    assert_eq!(diag.parameters().len(), 2 + 2);
    assert!(diag.log_likelihood() <= fit.log_likelihood() + 1e-6);
    let mu = diag.predict(&design, &split).unwrap();
    for (a, b) in mu.iter().zip(diag.fitted()) {
        assert!((a - b).abs() < 1e-9 * b);
    }
}

#[test]
fn nested_labels_join_the_levels() {
    let a = Grouping::new("a", ["x", "x", "y"].map(String::from).to_vec());
    let b = Grouping::new("b", ["1", "2", "1"].map(String::from).to_vec());
    let ab = Grouping::nested(&a, &b).unwrap();
    assert_eq!(ab.name, "a:b");
    assert_eq!(ab.labels, ["x:1", "x:2", "y:1"]);
    let short = Grouping::new("b", vec!["1".into()]);
    assert!(Grouping::nested(&a, &short).is_err());
}

#[test]
fn bad_inputs_are_errors() {
    let (design, labels, y) = book();
    let g = [Grouping::new("g", labels.clone())];
    let msg = |r: Result<GlmmFit>| r.unwrap_err().to_string();
    assert!(
        msg(Glmm::new(Family::Gaussian).fit(&design, &g, &y)).contains("Poisson, gamma or Tweedie")
    );
    assert!(
        Glmm::new(Family::Tweedie { power: 2.5 })
            .fit(&design, &g, &y)
            .is_err()
    );
    assert!(
        msg(Glmm::new(Family::Poisson).fit(&design, &[], &y)).contains("at least one grouping")
    );
    let two = [g[0].clone(), Grouping::new("h", labels.clone())];
    assert!(
        msg(Glmm::new(Family::Poisson)
            .quadrature(5)
            .fit(&design, &two, &y))
        .contains("single grouping factor with a single term")
    );
    assert!(
        Glmm::new(Family::Poisson)
            .quadrature(0)
            .fit(&design, &g, &y)
            .is_err()
    );
    let slope = [g[0].clone().slope("x", vec![0.5; 40])];
    assert!(
        msg(Glmm::new(Family::Poisson)
            .quadrature(5)
            .fit(&design, &slope, &y))
        .contains("2 terms")
    );
    let none = [g[0].clone().without_intercept()];
    assert!(
        msg(Glmm::new(Family::Poisson).fit(&design, &none, &y))
            .contains("neither an intercept nor a slope")
    );
    let ragged = [g[0].clone().slope("x", vec![0.5; 3])];
    assert!(
        msg(Glmm::new(Family::Poisson).fit(&design, &ragged, &y)).contains("3 values for 40 rows")
    );
    let short = [Grouping::new("g", vec!["a".into()])];
    assert!(
        msg(Glmm::new(Family::Poisson).fit(&design, &short, &y)).contains("1 labels for 40 rows")
    );
    // The gamma needs every response positive.
    let mut zero = y.clone();
    zero[0] = 0.0;
    assert!(msg(Glmm::new(Family::Gamma).fit(&design, &g, &zero)).contains("outside the gamma"));
}
