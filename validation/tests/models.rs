//! Model parity: GLMs against statsmodels on a synthetic portfolio
//! (`validation/scripts/statsmodels_glm.py`), with sandwich standard errors
//! (`statsmodels_glm_robust.py`).

use prospicio_glm::{Dispersion, Glm, GlmFit, Robust};
use prospicio_models::{Column, Design, Family, Frame, Link, Model, Terms};
use prospicio_validation::{check, reference};

/// `validation/data/glm_policies.csv` as named columns.
fn policies() -> Vec<(String, Vec<String>)> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/glm_policies.csv");
    let text = std::fs::read_to_string(path).expect("glm_policies.csv");
    let mut lines = text.lines();
    let header: Vec<String> = lines.next().unwrap().split(',').map(String::from).collect();
    let mut cols: Vec<Vec<String>> = vec![Vec::new(); header.len()];
    for line in lines {
        for (c, v) in cols.iter_mut().zip(line.split(',')) {
            c.push(v.to_string());
        }
    }
    header.into_iter().zip(cols).collect()
}

fn numeric(data: &[(String, Vec<String>)], name: &str) -> Vec<f64> {
    data.iter()
        .find(|(n, _)| n == name)
        .unwrap()
        .1
        .iter()
        .map(|v| v.parse().unwrap())
        .collect()
}

/// Intercept, age and region in treatment coding, reference "A".
fn design(data: &[(String, Vec<String>)]) -> Design {
    let region = data.iter().find(|(n, _)| n == "region").unwrap().1.clone();
    let frame = Frame::new(vec![
        ("age".into(), Column::Numeric(numeric(data, "age"))),
        ("region".into(), Column::Categorical(region)),
    ])
    .unwrap();
    Terms::new()
        .intercept()
        .numeric("age")
        .factor("region")
        .fit(&frame)
        .unwrap()
        .design(&frame)
        .unwrap()
}

fn fit_case(case: &str, data: &[(String, Vec<String>)]) -> GlmFit {
    let (glm, y, d) = case_data(case, data);
    glm.fit(&d, &y).unwrap_or_else(|e| panic!("{case}: {e}"))
}

/// The spec, response and design of a statsmodels case.
fn case_data(case: &str, data: &[(String, Vec<String>)]) -> (Glm, Vec<f64>, Design) {
    let d = design(data);
    let log_exposure: Vec<f64> = numeric(data, "exposure").iter().map(|e| e.ln()).collect();
    let (glm, y, d) = match case {
        "poisson_log" => (
            Glm::new(Family::Poisson, Link::Log),
            numeric(data, "claims"),
            d.with_offset(log_exposure).unwrap(),
        ),
        "quasi_poisson_log" => (
            Glm::over_dispersed_poisson(),
            numeric(data, "claims"),
            d.with_offset(log_exposure).unwrap(),
        ),
        "gamma_log" => (
            Glm::new(Family::Gamma, Link::Log),
            numeric(data, "severity"),
            d,
        ),
        "gamma_inverse" => (
            Glm::new(Family::Gamma, Link::Inverse),
            numeric(data, "severity"),
            d,
        ),
        "gamma_log_weighted" => (
            Glm::new(Family::Gamma, Link::Log),
            numeric(data, "avg_sev"),
            d.with_weights(numeric(data, "n_sev")).unwrap(),
        ),
        "inverse_gaussian_log" => (
            Glm::new(Family::InverseGaussian, Link::Log),
            numeric(data, "ig"),
            d,
        ),
        "binomial_logit" => {
            let trials = numeric(data, "trials");
            let y = numeric(data, "successes")
                .iter()
                .zip(&trials)
                .map(|(s, n)| s / n)
                .collect();
            (
                Glm::new(Family::Binomial, Link::Logit),
                y,
                d.with_weights(trials).unwrap(),
            )
        }
        "negative_binomial_log" => (
            Glm::new(Family::NegativeBinomial { theta: 2.0 }, Link::Log),
            numeric(data, "nb"),
            d,
        ),
        "gaussian_identity" => (
            Glm::new(Family::Gaussian, Link::Identity),
            numeric(data, "gauss"),
            d,
        ),
        "tweedie_log" => (
            Glm::new(Family::Tweedie { power: 1.5 }, Link::Log).dispersion(Dispersion::Pearson),
            numeric(data, "pure"),
            d.with_weights(numeric(data, "exposure")).unwrap(),
        ),
        other => panic!("unknown case {other}"),
    };
    (glm, y, d)
}

#[test]
fn glms_match_statsmodels() {
    let data = policies();
    let cases = reference("glm_statsmodels.csv");
    let mut fits: Vec<(String, GlmFit)> = Vec::new();
    check(&cases, |c| {
        let name = c.get("case");
        if !fits.iter().any(|(n, _)| n == name) {
            fits.push((name.to_string(), fit_case(name, &data)));
        }
        let fit = &fits.iter().find(|(n, _)| n == name).unwrap().1;
        let term = c.get("term");
        let j = fit.names().iter().position(|n| n == term);
        match c.get("quantity") {
            "coef" => Some(fit.coefficients()[j?]),
            "std_error" => Some(fit.std_errors()[j?]),
            "deviance" => Some(fit.deviance()),
            "null_deviance" => Some(fit.null_deviance()),
            "scale" => Some(fit.dispersion()),
            "log_likelihood" => Some(fit.log_likelihood()),
            "aic" => Some(fit.aic()),
            _ => None,
        }
    });
}

#[test]
fn robust_std_errors_match_statsmodels() {
    let data = policies();
    let groups: Vec<usize> = numeric(&data, "age")
        .iter()
        .map(|a| (a / 5.0).floor() as usize)
        .collect();
    let cases = reference("glm_robust_statsmodels.csv");
    let mut fits: Vec<(String, GlmFit, Design, Vec<f64>)> = Vec::new();
    check(&cases, |c| {
        let name = c.get("case");
        if !fits.iter().any(|f| f.0 == name) {
            let (glm, y, d) = case_data(name, &data);
            let fit = glm.fit(&d, &y).unwrap();
            fits.push((name.to_string(), fit, d, y));
        }
        let (_, fit, d, y) = fits.iter().find(|f| f.0 == name).unwrap();
        let kind = match c.get("kind") {
            "hc0" => Robust::Hc0,
            "cluster" => Robust::Cluster(&groups),
            _ => return None,
        };
        let j = fit.names().iter().position(|n| n == c.get("term"))?;
        Some(fit.robust_std_errors(d, y, kind).unwrap()[j])
    });
}

#[test]
fn gams_match_mgcv() {
    use prospicio_glm::gam::{Gam, GamFit, PSpline, Smoothing};
    use prospicio_models::Fitted;
    let data = policies();
    let cases = reference("gam_mgcv.csv");
    let fit_case = |case: &str| -> GamFit {
        let reml = case.starts_with("reml_");
        let case = match case.strip_prefix("reml_").unwrap_or(case) {
            "gaussian" => "gaussian_identity",
            "poisson" => "poisson_log",
            "gamma" => "gamma_log",
            other => other,
        };
        let d = design(&data);
        let log_exposure: Vec<f64> = numeric(&data, "exposure").iter().map(|e| e.ln()).collect();
        if case == "two_smooths" {
            // wavy ~ region + s(age, k = 12) + s(exposure, k = 8), by REML.
            let mut names = d.names().to_vec();
            let mut cols: Vec<Vec<f64>> = (0..d.n_cols()).map(|j| d.column(j).to_vec()).collect();
            names.push("exposure".into());
            cols.push(numeric(&data, "exposure"));
            let d = prospicio_models::Design::new(names, cols).unwrap();
            return Gam::new(
                Glm::new(Family::Gaussian, Link::Identity),
                vec![
                    PSpline::new("age").n_basis(12),
                    PSpline::new("exposure").n_basis(8),
                ],
            )
            .smoothing(Smoothing::Reml)
            .fit(&d, &numeric(&data, "wavy"))
            .unwrap_or_else(|e| panic!("{case}: {e}"));
        }
        let (glm, y, d, k) = match case {
            "gaussian_identity" => (
                Glm::new(Family::Gaussian, Link::Identity),
                numeric(&data, "gauss"),
                d,
                10,
            ),
            "poisson_log" => (
                Glm::new(Family::Poisson, Link::Log),
                numeric(&data, "claims"),
                d.with_offset(log_exposure).unwrap(),
                10,
            ),
            "gamma_log" => (
                Glm::new(Family::Gamma, Link::Log),
                numeric(&data, "severity"),
                d,
                10,
            ),
            "wavy_gaussian" => (
                Glm::new(Family::Gaussian, Link::Identity),
                numeric(&data, "wavy"),
                d,
                12,
            ),
            "wavy_poisson" => (
                Glm::new(Family::Poisson, Link::Log),
                numeric(&data, "wavy_claims"),
                d.with_offset(log_exposure).unwrap(),
                12,
            ),
            other => panic!("unknown case {other}"),
        };
        let gam = Gam::new(glm, vec![PSpline::new("age").n_basis(k)]);
        let gam = if reml {
            gam.smoothing(Smoothing::Reml)
        } else {
            gam
        };
        gam.fit(&d, &y).unwrap_or_else(|e| panic!("{case}: {e}"))
    };
    let mut fits: Vec<(String, GamFit)> = Vec::new();
    check(&cases, |c| {
        let name = c.get("case");
        if !fits.iter().any(|(n, _)| n == name) {
            fits.push((name.to_string(), fit_case(name)));
        }
        let fit = &fits.iter().find(|(n, _)| n == name).unwrap().1;
        match c.get("quantity") {
            "deviance" => Some(fit.deviance()),
            "edf" => Some(fit.edf()),
            "scale" => Some(fit.dispersion()),
            "score" => {
                // mgcv stops short of the REML minimum where it is flat:
                // never above its score.
                if name.starts_with("reml_") {
                    let want = c.number("expected")?;
                    assert!(
                        fit.score() <= want + 1e-9 * want.abs(),
                        "{name}: REML score above mgcv's"
                    );
                }
                Some(fit.score())
            }
            "reml_scale" => fit.reml_scale(),
            "coef" => {
                let j = fit.names().iter().position(|n| n == c.get("arg"))?;
                Some(fit.coefficients()[j])
            }
            "fitted" => Some(fit.fitted()[c.number("arg")? as usize]),
            _ => None,
        }
    });
    // Predicting on the training design reproduces the fitted values.
    let (_, fit) = &fits[0];
    let pred = fit.predict(&design(&data)).unwrap();
    assert!((pred[5] - fit.fitted()[5]).abs() < 1e-9);
}

#[test]
fn cyclic_gams_match_mgcv() {
    use prospicio_glm::gam::{Gam, GamFit, PSpline, Smoothing};
    use prospicio_math::spline::{cyclic_bspline_basis, cyclic_difference_penalty};
    use prospicio_models::Fitted;
    // The data in validation/scripts/r_gam_cyclic.R.
    let n = 300;
    let (mut t, mut gauss, mut count) = (Vec::new(), Vec::new(), Vec::new());
    for i in 1..=n {
        let ti = ((37 * i) % 300) as f64 / 25.0 + 0.5 / 25.0;
        let e = ((7919 * i) % 13) as f64 / 6.0 - 1.0;
        let a = std::f64::consts::TAU * ti / 12.0;
        t.push(ti);
        gauss.push(2.0 + a.sin() + 0.5 * (2.0 * a).cos() + 0.4 * e);
        count.push(
            ((0.5 + 0.6 * a.sin()).exp() + 1.2 * e + 1.5)
                .round()
                .max(0.0),
        );
    }
    let names = vec!["(Intercept)".to_string(), "t".to_string()];
    let design = Design::new(names.clone(), vec![vec![1.0; n], t]).unwrap();
    let ends = Design::new(names, vec![vec![1.0; 2], vec![0.0, 12.0]]).unwrap();
    let fit_case = |case: &str| -> GamFit {
        let spline = PSpline::cyclic("t");
        let spline = if case == "cyclic_range_gaussian" {
            spline
        } else {
            spline.period(0.0, 12.0)
        };
        let (glm, y) = if case.ends_with("poisson") {
            (Glm::new(Family::Poisson, Link::Log), &count)
        } else {
            (Glm::new(Family::Gaussian, Link::Identity), &gauss)
        };
        let gam = Gam::new(glm, vec![spline]);
        let gam = if case.starts_with("reml_") {
            gam.smoothing(Smoothing::Reml)
        } else {
            gam
        };
        gam.fit(&design, y)
            .unwrap_or_else(|e| panic!("{case}: {e}"))
    };
    let penalty = cyclic_difference_penalty(10, 2);
    let mut fits: Vec<(String, GamFit)> = Vec::new();
    check(&reference("gam_cyclic_mgcv.csv"), |c| {
        let name = c.get("case");
        if name == "basis" {
            let arg = c.get("arg");
            let (a, b) = arg.split_once(';')?;
            return match c.get("quantity") {
                "raw" => {
                    let x: f64 = a.parse().ok()?;
                    Some(cyclic_bspline_basis(x, 0.0, 12.0, 10, 3)[b.parse::<usize>().ok()?])
                }
                "penalty" => {
                    Some(penalty[a.parse::<usize>().ok()? * 10 + b.parse::<usize>().ok()?])
                }
                _ => None,
            };
        }
        if !fits.iter().any(|(n, _)| n == name) {
            fits.push((name.to_string(), fit_case(name)));
        }
        let fit = &fits.iter().find(|(n, _)| n == name).unwrap().1;
        match c.get("quantity") {
            "deviance" => Some(fit.deviance()),
            "edf" => Some(fit.edf()),
            "scale" => Some(fit.dispersion()),
            "score" => Some(fit.score()),
            "reml_scale" => fit.reml_scale(),
            "fitted" => Some(fit.fitted()[c.number("arg")? as usize]),
            "predict" => {
                let p = fit.predict(&ends).ok()?;
                Some(if c.get("arg") == "0" { p[0] } else { p[1] })
            }
            _ => None,
        }
    });
}

#[test]
fn monotone_gams_match_scam() {
    use prospicio_glm::gam::{Gam, GamFit, PSpline, Smoothing};
    use prospicio_models::Fitted;
    // The data in validation/scripts/r_gam_monotone.R.
    let n = 300;
    let logistic = |v: f64| 1.0 / (1.0 + (-v).exp());
    let (mut x, mut rise, mut fall) = (Vec::new(), Vec::new(), Vec::new());
    for i in 1..=n {
        let xi = (((37 * i) % 300) as f64 + 0.5) / 300.0;
        let e = ((7919 * i) % 13) as f64 / 6.0 - 1.0;
        x.push(xi);
        rise.push(1.0 + 2.0 * logistic(10.0 * (xi - 0.5)) + 0.3 * e);
        fall.push(
            ((2.5 - 2.0 * logistic(12.0 * (xi - 0.4))).exp() + 1.5 * e)
                .round()
                .max(0.0),
        );
    }
    let design = Design::new(
        vec!["(Intercept)".to_string(), "x".to_string()],
        vec![vec![1.0; n], x],
    )
    .unwrap();
    let cases = reference("gam_monotone_scam.csv");
    let lambda_of = |case: &str| -> f64 {
        cases
            .iter()
            .find(|c| c.get("case") == case && c.get("quantity") == "lambda")
            .and_then(|c| c.number("expected"))
            .unwrap()
    };
    let fit_case = |case: &str| -> GamFit {
        // An exact_ case is the selected fit, against the refined optimum.
        let case = case.strip_prefix("exact_").unwrap_or(case);
        let (glm, y, spline) = if case.starts_with("mpi") {
            (
                Glm::new(Family::Gaussian, Link::Identity),
                &rise,
                PSpline::increasing("x"),
            )
        } else {
            (
                Glm::new(Family::Poisson, Link::Log),
                &fall,
                PSpline::decreasing("x"),
            )
        };
        let gam = Gam::new(glm, vec![spline]);
        let gam = if case.ends_with("fixed") {
            gam.smoothing(Smoothing::Fixed(vec![lambda_of(case)]))
        } else {
            gam
        };
        gam.fit(&design, y)
            .unwrap_or_else(|e| panic!("{case}: {e}"))
    };
    let mut fits: Vec<(String, GamFit)> = Vec::new();
    check(&cases, |c| {
        let name = c.get("case");
        if !fits.iter().any(|(n, _)| n == name) {
            fits.push((name.to_string(), fit_case(name)));
        }
        let fit = &fits.iter().find(|(n, _)| n == name).unwrap().1;
        match c.get("quantity") {
            "lambda" => Some(fit.lambdas()[0]),
            "deviance" => Some(fit.deviance()),
            "edf" => Some(fit.edf()),
            "score" => Some(fit.score()),
            "scale" => Some(fit.dispersion()),
            "fitted" => Some(fit.fitted()[c.number("arg")? as usize]),
            "coef" => Some(fit.coefficients()[c.number("arg")? as usize]),
            _ => None,
        }
    });
    // Monotone in x, inside the data's range and past it (where the
    // smooth holds its end value).
    for (name, fit) in &fits {
        let grid: Vec<f64> = (0..=200).map(|i| f64::from(i) / 100.0 - 0.5).collect();
        let g = Design::new(
            vec!["(Intercept)".to_string(), "x".to_string()],
            vec![vec![1.0; 201], grid],
        )
        .unwrap();
        let p = fit.predict(&g).unwrap();
        let up = name.trim_start_matches("exact_").starts_with("mpi");
        assert!(
            p.windows(2).all(|w| if up {
                w[1] >= w[0] - 1e-12
            } else {
                w[1] <= w[0] + 1e-12
            }),
            "{name} is not monotone"
        );
    }
}

#[test]
fn mcmc_diagnostics_match_posterior() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mcmc_chains.csv");
    let text = std::fs::read_to_string(path).expect("mcmc_chains.csv");
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let rows: Vec<Vec<f64>> = lines
        .map(|l| l.split(',').map(|v| v.parse().unwrap()).collect())
        .collect();
    let chains_of = |variable: &str| -> Vec<Vec<f64>> {
        let col = header.iter().position(|h| *h == variable).unwrap();
        let mut chains = vec![Vec::new(); 4];
        for r in &rows {
            chains[r[0] as usize].push(r[col]);
        }
        chains
    };
    let cases = reference("mcmc_posterior.csv");
    check(&cases, |c| {
        let chains = chains_of(c.get("variable"));
        let refs: Vec<&[f64]> = chains.iter().map(Vec::as_slice).collect();
        match c.get("quantity") {
            "rhat" => prospicio_bayes::rhat(&refs).ok(),
            "ess_bulk" => prospicio_bayes::ess_bulk(&refs).ok(),
            "ess_tail" => prospicio_bayes::ess_tail(&refs).ok(),
            "ess_mean" => prospicio_bayes::ess_mean(&refs).ok(),
            "mcse_mean" => prospicio_bayes::mcse_mean(&refs).ok(),
            _ => None,
        }
    });
}

/// Intercept, age, age²/100 and region dummies, as `r_glmnet.R` builds them.
fn net_design(data: &[(String, Vec<String>)]) -> Design {
    let age = numeric(data, "age");
    let region = &data.iter().find(|(n, _)| n == "region").unwrap().1;
    let dummy = |level: &str| -> Vec<f64> {
        region
            .iter()
            .map(|r| f64::from(u8::from(r == level)))
            .collect()
    };
    Design::new(
        vec![
            "(Intercept)".into(),
            "age".into(),
            "age2".into(),
            "region[B]".into(),
            "region[C]".into(),
            "region[D]".into(),
        ],
        vec![
            vec![1.0; age.len()],
            age.clone(),
            age.iter().map(|a| a * a / 100.0).collect(),
            dummy("B"),
            dummy("C"),
            dummy("D"),
        ],
    )
    .unwrap()
}

fn net_case(
    case: &str,
    data: &[(String, Vec<String>)],
) -> (prospicio_glm::net::ElasticNet, Design, Vec<f64>) {
    use prospicio_glm::net::ElasticNet;
    let d = net_design(data);
    let log_exposure: Vec<f64> = numeric(data, "exposure").iter().map(|e| e.ln()).collect();
    let net = |family, link, alpha| ElasticNet::new(family, link, alpha, 0.0);
    match case {
        "gaussian_lasso" => (
            net(Family::Gaussian, Link::Identity, 1.0),
            d,
            numeric(data, "gauss"),
        ),
        "gaussian_enet_raw" => (
            net(Family::Gaussian, Link::Identity, 0.5).standardize(false),
            d,
            numeric(data, "gauss"),
        ),
        "gaussian_weighted" => (
            net(Family::Gaussian, Link::Identity, 0.7),
            d.with_weights(numeric(data, "exposure")).unwrap(),
            numeric(data, "gauss"),
        ),
        "poisson_lasso" => (
            net(Family::Poisson, Link::Log, 1.0),
            d.with_offset(log_exposure).unwrap(),
            numeric(data, "claims"),
        ),
        "poisson_ridge" => (
            net(Family::Poisson, Link::Log, 0.0),
            d.with_offset(log_exposure).unwrap(),
            numeric(data, "claims"),
        ),
        "binomial_enet" => {
            let trials = numeric(data, "trials");
            let y = numeric(data, "successes")
                .iter()
                .zip(&trials)
                .map(|(s, n)| s / n)
                .collect();
            (
                net(Family::Binomial, Link::Logit, 0.5),
                d.with_weights(trials).unwrap(),
                y,
            )
        }
        "gamma_log_enet" => (
            net(Family::Gamma, Link::Log, 0.5),
            d,
            numeric(data, "severity"),
        ),
        // r_glmnet_relax.R: bounds and relaxed fits, all lasso.
        "poisson_relaxed" | "poisson_nonneg" | "poisson_nonneg_relaxed" => {
            let mut spec = net(Family::Poisson, Link::Log, 1.0);
            if case.contains("nonneg") {
                spec = spec.lower_limits(vec![0.0; 6]);
            }
            (
                spec,
                d.with_offset(log_exposure).unwrap(),
                numeric(data, "claims"),
            )
        }
        "gaussian_relaxed" => (
            net(Family::Gaussian, Link::Identity, 1.0),
            d,
            numeric(data, "gauss"),
        ),
        "gaussian_bounded" => {
            let (inf, ninf) = (f64::INFINITY, f64::NEG_INFINITY);
            (
                net(Family::Gaussian, Link::Identity, 1.0)
                    .lower_limits(vec![ninf, ninf, ninf, ninf, -2.0, ninf])
                    .upper_limits(vec![inf, inf, 0.1, inf, inf, 5.0]),
                d,
                numeric(data, "gauss"),
            )
        }
        other => panic!("unknown case {other}"),
    }
}

/// glmnet's `(λ, α)` as ours. For the Gaussian, glmnet scales `y` to unit
/// (weighted, population) standard deviation `s_y` before fitting and
/// reports `λ` back on `y`'s scale, which leaves the lasso part of the
/// penalty as stated but divides the ridge part by `s_y`. The same
/// solution in our parameterization has `λ α` unchanged and
/// `λ (1 - α)` divided by `s_y`.
fn from_glmnet(
    spec: &prospicio_glm::net::ElasticNet,
    d: &Design,
    y: &[f64],
    lambda: f64,
) -> prospicio_glm::net::ElasticNet {
    if spec.family != Family::Gaussian || spec.alpha == 1.0 {
        return spec.with_lambda(lambda);
    }
    let w = d.weights();
    let total: f64 = w.iter().sum();
    let mean = y.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / total;
    let sd = (y
        .iter()
        .zip(w)
        .map(|(a, b)| b * (a - mean).powi(2))
        .sum::<f64>()
        / total)
        .sqrt();
    let l1 = lambda * spec.alpha;
    let l2 = lambda * (1.0 - spec.alpha) / sd;
    let mut out = spec.with_lambda(l1 + l2);
    out.alpha = l1 / (l1 + l2);
    out
}

#[test]
fn elastic_nets_match_glmnet() {
    let data = policies();
    let cases = reference("elastic_net_glmnet.csv");
    check(&cases, |c| {
        let (spec, d, y) = net_case(c.get("case"), &data);
        match c.get("quantity") {
            "lambda_max" => spec.lambda_max(&d, &y).ok(),
            "coef" | "deviance" => {
                let fit = from_glmnet(&spec, &d, &y, c.number("arg")?)
                    .fit(&d, &y)
                    .ok()?;
                if c.get("quantity") == "deviance" {
                    return Some(fit.deviance());
                }
                let j = fit.names().iter().position(|n| n == c.get("term"))?;
                Some(fit.coefficients()[j])
            }
            _ => None,
        }
    });
}

#[test]
fn elastic_net_cross_validation_matches_cv_glmnet() {
    use prospicio_models::resample::Split;
    let data = policies();
    let cases = reference("elastic_net_cv_glmnet.csv");
    let n = numeric(&data, "age").len();
    // Row i (from 0) in fold i mod 5, as the script's foldid.
    let splits: Vec<Split> = (0..5)
        .map(|f| Split {
            train: (0..n).filter(|i| i % 5 != f).collect(),
            test: (0..n).filter(|i| i % 5 == f).collect(),
        })
        .collect();
    let mut runs: Vec<(String, prospicio_glm::net::CvPath)> = Vec::new();
    check(&cases, |c| {
        let name = c.get("case");
        if !runs.iter().any(|(k, _)| k == name) {
            let lambdas: Vec<f64> = cases
                .iter()
                .filter(|r| r.get("case") == name && r.get("quantity") == "mean")
                .map(|r| r.number("arg").unwrap())
                .collect();
            let (spec, d, y) = net_case(name, &data);
            let cv = spec.cross_validate(&d, &y, &lambdas, &splits).ok()?;
            runs.push((name.to_string(), cv));
        }
        let cv = &runs.iter().find(|(k, _)| k == name)?.1;
        let at = |l: f64| cv.lambdas.iter().position(|&x| x == l);
        match c.get("quantity") {
            "mean" => Some(cv.mean[at(c.number("arg")?)?]),
            "se" => Some(cv.se[at(c.number("arg")?)?]),
            "lambda_min" => Some(cv.lambda_min()),
            "lambda_1se" => Some(cv.lambda_1se()),
            _ => None,
        }
    });
}

#[test]
fn bounded_and_relaxed_elastic_nets_match_glmnet() {
    let data = policies();
    let cases = reference("elastic_net_relax_glmnet.csv");
    check(&cases, |c| {
        let (spec, d, y) = net_case(c.get("case"), &data);
        match c.get("quantity") {
            "lambda_max" => spec.lambda_max(&d, &y).ok(),
            "coef" => {
                let mut fit = spec.with_lambda(c.number("arg")?).fit(&d, &y).ok()?;
                if !c.get("gamma").is_empty() {
                    fit = fit.relax(&d, &y, c.number("gamma")?).ok()?;
                }
                let j = fit.names().iter().position(|n| n == c.get("term"))?;
                Some(fit.coefficients()[j])
            }
            _ => None,
        }
    });
}

#[test]
fn relaxed_cross_validation_matches_cv_glmnet() {
    use prospicio_models::resample::Split;
    let data = policies();
    let cases = reference("elastic_net_relax_cv_glmnet.csv");
    let n = numeric(&data, "age").len();
    let splits: Vec<Split> = (0..5)
        .map(|f| Split {
            train: (0..n).filter(|i| i % 5 != f).collect(),
            test: (0..n).filter(|i| i % 5 == f).collect(),
        })
        .collect();
    // As the script records them: glmnet's γ = 0 is 1e-5.
    let gammas = [1e-5, 0.25, 0.5, 0.75, 1.0];
    let mut runs: Vec<(String, prospicio_glm::net::RelaxedCv)> = Vec::new();
    check(&cases, |c| {
        let name = c.get("case");
        if !runs.iter().any(|(k, _)| k == name) {
            let lambdas: Vec<f64> = cases
                .iter()
                .filter(|r| r.get("case") == name && r.get("quantity") == "mean")
                .filter(|r| r.get("gamma") == "1")
                .map(|r| r.number("arg").unwrap())
                .collect();
            let (spec, d, y) = net_case(name, &data);
            let cv = spec
                .cross_validate_relaxed(&d, &y, &lambdas, &gammas, &splits)
                .ok()?;
            runs.push((name.to_string(), cv));
        }
        let cv = &runs.iter().find(|(k, _)| k == name)?.1;
        let at = || -> Option<&prospicio_glm::net::CvPath> {
            let gamma = c.number("gamma")?;
            let g = cv.gammas.iter().position(|&x| x == gamma)?;
            Some(&cv.paths[g])
        };
        let k = |path: &prospicio_glm::net::CvPath| {
            path.lambdas
                .iter()
                .position(|&x| Some(x) == c.number("arg"))
        };
        match c.get("quantity") {
            "mean" => at().and_then(|p| Some(p.mean[k(p)?])),
            "se" => at().and_then(|p| Some(p.se[k(p)?])),
            "lambda_min" => Some(cv.lambda_min()),
            "gamma_min" => Some(cv.gamma_min()),
            "lambda_1se" => Some(cv.lambda_1se()),
            "gamma_1se" => Some(cv.gamma_1se()),
            _ => None,
        }
    });
}

#[test]
fn tweedie_power_estimate_matches_statsmodels() {
    use prospicio_glm::tweedie::{TweedieGlm, tweedie_profile};
    let data = policies();
    let d = net_design(&data);
    // Intercept, age and region: drop the age² column `net_design` adds.
    let keep: Vec<usize> = (0..d.n_cols())
        .filter(|&j| d.names()[j] != "age2")
        .collect();
    let d = Design::new(
        keep.iter().map(|&j| d.names()[j].clone()).collect(),
        keep.iter().map(|&j| d.column(j).to_vec()).collect(),
    )
    .unwrap()
    .with_weights(numeric(&data, "exposure"))
    .unwrap();
    let y = numeric(&data, "pure");
    let powers = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8];
    let prof = tweedie_profile(Link::Log, &d, &y, &powers).unwrap();
    let fit = TweedieGlm::new(Link::Log).fit(&d, &y).unwrap();
    // The grid's refinement and Brent's method agree.
    assert!((prof.power - fit.power()).abs() < 1e-3);
    let cases = reference("tweedie_profile_statsmodels.csv");
    check(&cases, |c| {
        let at = |p: f64| powers.iter().position(|&q| q == p);
        match c.get("quantity") {
            "log_likelihood" => Some(prof.log_likelihood[at(c.number("arg")?)?]),
            "dispersion" => Some(prof.dispersion[at(c.number("arg")?)?]),
            "power" => Some(fit.power()),
            "max_log_likelihood" => Some(fit.log_likelihood()),
            "dispersion_at_power" => Some(fit.dispersion()),
            "interval_lower" => Some(fit.interval().0),
            "interval_upper" => Some(fit.interval().1),
            _ => None,
        }
    });
}

#[test]
fn family_scores_match_scipy() {
    let cases = reference("family_scores_scipy.csv");
    check(&cases, |c| {
        let family = match c.get("family") {
            "gaussian" => Family::Gaussian,
            "poisson" => Family::Poisson,
            "binomial" => Family::Binomial,
            "negative_binomial" => Family::NegativeBinomial {
                theta: c.param("params", "theta"),
            },
            "gamma" => Family::Gamma,
            "inverse_gaussian" => Family::InverseGaussian,
            _ => return None,
        };
        let (mu, phi, w) = (
            c.param("params", "mu"),
            c.param("params", "phi"),
            c.param("params", "w"),
        );
        let y = c.number("y")?;
        match c.get("quantity") {
            "log_density" => family.log_density(y, mu, phi, w).ok(),
            "cdf_lower" => family.cdf_bounds(y, mu, phi, w).ok().map(|b| b.0),
            "cdf_upper" => family.cdf_bounds(y, mu, phi, w).ok().map(|b| b.1),
            _ => None,
        }
    });
}

#[test]
fn elpd_matches_loo() {
    use prospicio_bayes::elpd::{loo, lppd, waic};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/loo_loglik.csv");
    let text = std::fs::read_to_string(path).expect("loo_loglik.csv");
    let mut lines = text.lines();
    let n = lines.next().unwrap().split(',').count();
    // Draws by observations, row-major.
    let ll: Vec<f64> = lines
        .flat_map(|l| {
            l.split(',')
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect();
    let l = loo(&ll, n, None).unwrap();
    let w = waic(&ll, n).unwrap();
    let cases = reference("elpd_loo.csv");
    check(&cases, |c| {
        let at = || c.number("index").map(|i| i as usize);
        match c.get("quantity") {
            "elpd_loo" => Some(l.estimate.elpd),
            "p_loo" => Some(l.estimate.p),
            "looic" => Some(l.estimate.ic),
            "se_elpd_loo" => Some(l.estimate.se),
            "pointwise_elpd_loo" => Some(l.estimate.pointwise[at()?]),
            "pareto_k" => Some(l.pareto_k[at()?]),
            "elpd_waic" => Some(w.elpd),
            "p_waic" => Some(w.p),
            "waic" => Some(w.ic),
            "se_elpd_waic" => Some(w.se),
            "lppd" => lppd(&ll, n).ok(),
            _ => None,
        }
    });
}

#[test]
fn stacking_and_pseudo_bma_weights_match_the_reference() {
    // validation/scripts/stacking_weights.py: held-out log densities of four
    // claim-count models, one column per model.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/stacking_lpd.csv");
    let text = std::fs::read_to_string(path).expect("stacking_lpd.csv");
    let mut lines = text.lines();
    let models: Vec<String> = lines.next().unwrap().split(',').map(String::from).collect();
    let mut lpd = vec![Vec::new(); models.len()];
    for line in lines {
        for (col, v) in lpd.iter_mut().zip(line.split(',')) {
            col.push(v.parse::<f64>().unwrap());
        }
    }
    let stacking = prospicio_models::stack::stacking_weights(&lpd).unwrap();
    let pseudo = prospicio_models::stack::pseudo_bma_weights(&lpd, None).unwrap();
    check(&reference("stacking_weights.csv"), |c| {
        let k = models.iter().position(|m| m == c.get("model"))?;
        match c.get("method") {
            "stacking" => Some(stacking[k]),
            "pseudo_bma" => Some(pseudo[k]),
            _ => None,
        }
    });
}

#[test]
fn bayes_glm_posteriors_match_grid_integration() {
    // validation/scripts/bayes_glm_grid.py: exact posterior moments by grid
    // integration; the first 200 policies, x = (age - 50) / 10.
    use prospicio_bayes::glm::{BayesGlm, Sampler};
    let data = policies();
    let n = 200;
    let x: Vec<f64> = numeric(&data, "age")[..n]
        .iter()
        .map(|a| (a - 50.0) / 10.0)
        .collect();
    let design = |offset: Option<Vec<f64>>| {
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; n], x.clone()],
        )
        .unwrap();
        match offset {
            Some(o) => d.with_offset(o).unwrap(),
            None => d,
        }
    };
    let sampler = Sampler {
        chains: 4,
        tune: 1000,
        draws: 1000,
        seed: 20261005,
        ..Sampler::default()
    };
    let log_e: Vec<f64> = numeric(&data, "exposure")[..n]
        .iter()
        .map(|e| e.ln())
        .collect();
    let poisson = BayesGlm::new(Family::Poisson, Link::Log)
        .sampler(sampler)
        .fit(&design(Some(log_e)), &numeric(&data, "claims")[..n])
        .unwrap();
    let gauss: Vec<f64> = numeric(&data, "gauss")[..n]
        .iter()
        .map(|g| (g - 200.0) / 10.0)
        .collect();
    let gaussian = BayesGlm::new(Family::Gaussian, Link::Identity)
        .sampler(sampler)
        .fit(&design(None), &gauss)
        .unwrap();
    let summaries = [
        ("poisson", poisson.summary().unwrap()),
        ("gaussian", gaussian.summary().unwrap()),
    ];
    for (_, s) in &summaries {
        assert!(
            s.iter().all(|p| p.rhat < 1.01 && p.ess_bulk > 400.0),
            "{s:?}"
        );
    }
    assert_eq!(poisson.divergences() + gaussian.divergences(), 0);
    check(&reference("bayes_glm_grid.csv"), |c| {
        let s = &summaries.iter().find(|(n, _)| *n == c.get("case"))?.1;
        let k = match c.get("param") {
            "b0" => 0,
            "b1" => 1,
            "dispersion" => 2,
            _ => return None,
        };
        match c.get("quantity") {
            "mean" => Some(s[k].mean),
            "sd" => Some(s[k].sd),
            _ => None,
        }
    });
}

/// Columns of a CSV in `validation/data/` with a header row.
fn data_columns(name: &str) -> Vec<(String, Vec<f64>)> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join(name);
    let text = std::fs::read_to_string(path).expect(name);
    let mut lines = text.lines();
    let header: Vec<String> = lines.next().unwrap().split(',').map(String::from).collect();
    let mut cols = vec![Vec::new(); header.len()];
    for line in lines {
        for (c, v) in cols.iter_mut().zip(line.split(',')) {
            c.push(v.parse::<f64>().unwrap());
        }
    }
    header.into_iter().zip(cols).collect()
}

#[test]
fn bayesian_and_hierarchical_stacking_match_grid_integration() {
    // validation/scripts/stacking_grid.py: exact posterior moments by grid
    // integration.
    use prospicio_bayes::glm::Sampler;
    use prospicio_bayes::stacking::{BayesStacking, HierarchicalStacking};
    let col =
        |cols: &[(String, Vec<f64>)], n: &str| cols.iter().find(|c| c.0 == n).unwrap().1.clone();
    let lpd = data_columns("stacking_lpd.csv");
    let pair = vec![
        col(&lpd, "poisson_intercept"),
        col(&lpd, "poisson_age_region"),
    ];
    let age = numeric(&policies(), "age");
    let n = age.len() as f64;
    let mean = age.iter().sum::<f64>() / n;
    let sd = (age.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / n).sqrt();
    let x: Vec<f64> = age.iter().map(|a| (a - mean) / (2.0 * sd)).collect();
    let syn = data_columns("stacking_synthetic.csv");
    let sampler = Sampler {
        chains: 4,
        tune: 1000,
        draws: 1000,
        seed: 20261005,
        ..Sampler::default()
    };
    let bayes = BayesStacking {
        sampler,
        ..BayesStacking::default()
    }
    .fit(&pair)
    .unwrap();
    let spec = HierarchicalStacking {
        sampler,
        ..HierarchicalStacking::default()
    };
    let hier = spec.fit(&pair, std::slice::from_ref(&x)).unwrap();
    let synth = spec
        .fit(&[col(&syn, "lpd_a"), col(&syn, "lpd_b")], &[col(&syn, "x")])
        .unwrap();
    for f in [&bayes, &hier, &synth] {
        assert_eq!(f.divergences(), 0);
        assert!(
            f.rhat_ess()
                .unwrap()
                .iter()
                .all(|(r, e)| *r < 1.01 && *e > 400.0)
        );
    }
    let moments = |v: &[f64]| {
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n;
        (
            m,
            (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt(),
        )
    };
    // The first model's weight in each Bayesian-stacking draw.
    let w_bayes: Vec<f64> = bayes
        .alpha_draws()
        .iter()
        .map(|z| 1.0 / (1.0 + (-z).exp()))
        .collect();
    check(&reference("stacking_grid.csv"), |c| {
        let fit = match c.get("case") {
            "bayes" => {
                let (m, s) = moments(&w_bayes);
                return match c.get("quantity") {
                    "mean" => Some(m),
                    "sd" => Some(s),
                    _ => None,
                };
            }
            "hierarchical" => &hier,
            "synthetic" => &synth,
            _ => return None,
        };
        let param = c.get("param");
        if let Some(at) = param.strip_prefix("weight_at_") {
            let xv: f64 = at.parse().ok()?;
            return Some(fit.weights(&[vec![xv]]).unwrap()[0]);
        }
        let draws = match param {
            "alpha" => fit.alpha_draws(),
            "beta" => fit.beta_draws(),
            _ => return None,
        };
        let (m, s) = moments(&draws);
        match c.get("quantity") {
            "mean" => Some(m),
            "sd" => Some(s),
            _ => None,
        }
    });
}

#[test]
fn nuts_posterior_predictive_matches_the_conjugate_negative_binomial() {
    use prospicio_bayes::nuts::{LogDensity, Sampler, sample};
    use prospicio_prob::{Counting, Distribution, KeyValue, NegativeBinomial, Provenance};

    // Poisson counts with a Gamma(2, rate 0.5) prior on the rate λ. The
    // posterior is Gamma(2 + Σy, 0.5 + n), and next year's count is
    // negative binomial with shape 2 + Σy and scale 1 / (0.5 + n).
    let y = [3.0, 5.0, 2.0, 4.0, 6.0, 1.0, 3.0, 4.0];
    let (a, b) = (2.0 + y.iter().sum::<f64>(), 0.5 + y.len() as f64);

    /// log λ, with the Jacobian: (a) θ - b e^θ.
    struct LogRate {
        a: f64,
        b: f64,
    }
    impl LogDensity for LogRate {
        fn dim(&self) -> usize {
            1
        }
        fn log_density(&self, x: &[f64], g: &mut [f64]) -> Option<f64> {
            let lambda = x[0].exp();
            g[0] = self.a - self.b * lambda;
            Some(self.a * x[0] - self.b * lambda)
        }
    }

    let s = Sampler {
        chains: 4,
        tune: 500,
        draws: 2000,
        seed: 17,
        ..Sampler::default()
    };
    let post = sample(&LogRate { a, b }, &[(a / b).ln()], s).unwrap();
    let rate = post
        .transform(vec!["lambda".into()], |x| vec![x[0].exp()])
        .unwrap();
    let summary = &rate.summary().unwrap()[0];
    let (mean, sd) = (a / b, a.sqrt() / b);
    assert!(summary.rhat < 1.01, "{summary:?}");
    assert!(
        (summary.mean - mean).abs() < 4.0 * sd / summary.ess_bulk.sqrt(),
        "{summary:?}"
    );
    assert!((summary.sd / sd - 1.0).abs() < 0.05, "{summary:?}");

    // Posterior predictive count: draw λ, then a Poisson count.
    let n = 40_000;
    let pd = rate
        .predictive(
            vec!["year".into()],
            vec![vec![KeyValue::from(2026)]],
            n,
            23,
            Provenance::new("gamma_poisson"),
            |theta, rng, row| {
                // Inverse transform of Poisson(λ).
                let (lambda, u) = (theta[0], rng.next_open01());
                let (mut k, mut p) = (0.0, (-lambda).exp());
                let mut cdf = p;
                while u > cdf {
                    k += 1.0;
                    p *= lambda / k;
                    cdf += p;
                }
                row[0] = k;
            },
        )
        .unwrap();
    let exact = NegativeBinomial::new(a, 1.0 / b).unwrap();
    let draws = pd.total();
    for k in 0..12u64 {
        let want = exact.cdf(k);
        let got = draws.cdf(k as f64);
        let se = (want * (1.0 - want) / n as f64).sqrt();
        // Draws within a simulation are independent, but the posterior
        // draws repeat across simulations; allow for that.
        assert!(
            (got - want).abs() < 6.0 * se + 2e-3,
            "k = {k}: {got} vs {want}"
        );
    }
    assert!((draws.mean() - Counting::mean(&exact)).abs() < 0.05);
}

/// RAA as an over-dispersed Poisson GLM on incremental cells (intercept,
/// origin and development dummies, first level dropped): the observed
/// cells' design and responses, the future cells' design, and the Chain
/// Ladder reserve.
fn raa_odp() -> (
    prospicio_models::Design,
    Vec<f64>,
    prospicio_models::Design,
    f64,
) {
    use prospicio_models::Design;

    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/raa.csv"),
    )
    .unwrap();
    let n = 10;
    let mut cum = vec![vec![f64::NAN; n]; n];
    for l in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("origin"))
    {
        let v: Vec<f64> = l.split(',').map(|x| x.parse().unwrap()).collect();
        cum[(v[0] - 1981.0) as usize][(v[1] / 12.0) as usize - 1] = v[2];
    }
    let dummies = |i: usize, j: usize| -> Vec<f64> {
        let mut r = vec![1.0];
        r.extend((1..n).map(|k| f64::from(u8::from(i == k))));
        r.extend((1..n).map(|k| f64::from(u8::from(j == k))));
        r
    };
    let design = |cells: &[(usize, usize)]| {
        let mut cols = vec![Vec::new(); 1 + 2 * (n - 1)];
        for &(i, j) in cells {
            for (c, v) in cols.iter_mut().zip(dummies(i, j)) {
                c.push(v);
            }
        }
        let names = (0..cols.len()).map(|k| format!("x{k}")).collect();
        Design::new(names, cols).unwrap()
    };
    let observed: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (0..n - i).map(move |j| (i, j)))
        .collect();
    let future: Vec<(usize, usize)> = (1..n)
        .flat_map(|i| (n - i..n).map(move |j| (i, j)))
        .collect();
    let y = observed
        .iter()
        .map(|&(i, j)| {
            if j == 0 {
                cum[i][0]
            } else {
                cum[i][j] - cum[i][j - 1]
            }
        })
        .collect();

    // Chain Ladder: volume-weighted development factors.
    let mut ultimate: Vec<f64> = (0..n).map(|i| cum[i][n - 1 - i]).collect();
    for j in 0..n - 1 {
        let (num, den) =
            (0..n - 1 - j).fold((0.0, 0.0), |(a, b), i| (a + cum[i][j + 1], b + cum[i][j]));
        for (i, u) in ultimate.iter_mut().enumerate() {
            if j >= n - 1 - i {
                *u *= num / den;
            }
        }
    }
    let latest: f64 = (0..n).map(|i| cum[i][n - 1 - i]).sum();
    (
        design(&observed),
        y,
        design(&future),
        ultimate.iter().sum::<f64>() - latest,
    )
}

#[test]
fn over_dispersed_poisson_fits_raa_with_its_negative_increment() {
    // RAA's 1982 origin loses 103 between 72 and 84 months. The
    // over-dispersed Poisson GLM with origin and development factors
    // still reproduces the Chain Ladder reserve.
    use prospicio_glm::Glm;
    use prospicio_models::{Fitted, Model};

    let (design, y, future, cl_reserve) = raa_odp();
    assert!(y.iter().any(|v| *v < 0.0));
    let fit = Glm::over_dispersed_poisson().fit(&design, &y).unwrap();
    let glm_reserve: f64 = fit.predict(&future).unwrap().iter().sum();
    assert!(
        (glm_reserve - cl_reserve).abs() < 1e-6 * cl_reserve,
        "{glm_reserve} vs {cl_reserve}"
    );
    assert!(fit.dispersion() > 0.0 && fit.log_likelihood().is_nan());
    // A Poisson with the dispersion fixed at 1 still refuses it.
    assert!(
        Glm::new(
            prospicio_models::Family::Poisson,
            prospicio_models::Link::Log
        )
        .fit(&design, &y)
        .is_err()
    );
}

#[test]
fn mean_preserving_draws_centre_the_odp_reserve_on_the_chain_ladder() {
    use prospicio_glm::{Glm, ParameterDraws};
    use prospicio_models::Model;
    use prospicio_prob::Distribution;

    let (design, y, future, cl_reserve) = raa_odp();
    let fit = Glm::over_dispersed_poisson().fit(&design, &y).unwrap();
    let n = 20_000;
    let reserve = |parameters| {
        let total = fit
            .predict_distribution_with(&future, n, 5, parameters)
            .unwrap()
            .total()
            .clone();
        (total.mean(), total.std_dev() / (n as f64).sqrt())
    };
    // Normal draws of β overstate the mean through exp(); shifting each
    // cell by -x'Σx/2 removes that, keeping parameter and process error.
    let (normal, se) = reserve(ParameterDraws::Normal);
    let (centred, se_c) = reserve(ParameterDraws::MeanPreserving);
    let (fixed, se_f) = reserve(ParameterDraws::Fixed);
    assert!(
        (centred - cl_reserve).abs() < 4.0 * se_c,
        "{centred} vs {cl_reserve}"
    );
    assert!(
        (fixed - cl_reserve).abs() < 4.0 * se_f,
        "{fixed} vs {cl_reserve}"
    );
    assert!(normal - cl_reserve > 4.0 * se, "{normal} vs {cl_reserve}");
    // Parameter error widens the distribution.
    assert!(se_c > se_f);
}

/// Tensor product smooths against mgcv's `te(x, z, bs = "ps", np = FALSE)`
/// (`validation/scripts/r_gam_tensor.R`): a surface by REML and GCV,
/// Poisson counts by REML, and a P-spline beside the tensor.
#[test]
fn tensor_gams_match_mgcv() {
    use prospicio_glm::gam::{Gam, GamFit, PSpline, Smoothing, TensorSpline};
    use prospicio_models::{Design, Fitted};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/gam_surface.csv");
    let text = std::fs::read_to_string(path).expect("gam_surface.csv");
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let rows: Vec<Vec<f64>> = lines
        .map(|l| l.split(',').map(|v| v.parse().unwrap()).collect())
        .collect();
    let col = |name: &str| -> Vec<f64> {
        let j = header.iter().position(|h| *h == name).unwrap();
        rows.iter().map(|r| r[j]).collect()
    };
    let n = rows.len();
    // `w` only where a smooth replaces it; otherwise it would be linear.
    let with_w = Design::new(
        vec!["(Intercept)".into(), "x".into(), "z".into(), "w".into()],
        vec![vec![1.0; n], col("x"), col("z"), col("w")],
    )
    .unwrap();
    let design = Design::new(
        vec!["(Intercept)".into(), "x".into(), "z".into()],
        vec![vec![1.0; n], col("x"), col("z")],
    )
    .unwrap();
    let te = || TensorSpline::new(&["x", "z"]);
    let fit_case = |case: &str| -> GamFit {
        let (family, link, y, smooths, smoothing) = match case {
            "te_gaussian_reml" => (
                Family::Gaussian,
                Link::Identity,
                "y",
                vec![],
                Smoothing::Reml,
            ),
            "te_gaussian_gcv" => (
                Family::Gaussian,
                Link::Identity,
                "y",
                vec![],
                Smoothing::Auto,
            ),
            "te_poisson_reml" => (
                Family::Poisson,
                Link::Log,
                "counts",
                vec![],
                Smoothing::Reml,
            ),
            "s_te_reml" => (
                Family::Gaussian,
                Link::Identity,
                "y2",
                vec![PSpline::new("w").n_basis(8)],
                Smoothing::Reml,
            ),
            other => panic!("unknown case {other}"),
        };
        let d = if smooths.is_empty() { &design } else { &with_w };
        Gam::new(Glm::new(family, link), smooths)
            .tensor(te())
            .smoothing(smoothing)
            .fit(d, &col(y))
            .unwrap_or_else(|e| panic!("{case}: {e}"))
    };
    let cases = reference("gam_tensor_mgcv.csv");
    let mut fits: Vec<(String, GamFit)> = Vec::new();
    check(&cases, |c| {
        let name = c.get("case");
        if !fits.iter().any(|(n, _)| n == name) {
            fits.push((name.to_string(), fit_case(name)));
        }
        let fit = &fits.iter().find(|(n, _)| n == name).unwrap().1;
        match c.get("quantity") {
            "deviance" => Some(fit.deviance()),
            "edf" => Some(fit.edf()),
            "scale" => Some(fit.dispersion()),
            "score" => {
                // Never above mgcv's: it stops where the score is flat.
                let want = c.number("expected")?;
                assert!(
                    fit.score() <= want + 1e-9 * want.abs(),
                    "{name}: score above mgcv's"
                );
                Some(fit.score())
            }
            "reml_scale" => fit.reml_scale(),
            "fitted" => Some(fit.fitted()[c.number("arg")? as usize]),
            _ => None,
        }
    });
    // Two smoothing parameters for the tensor, and the saved fit predicts
    // the same.
    let (_, fit) = &fits[0];
    assert_eq!(fit.lambdas().len(), 2);
    let back = GamFit::from_json(&fit.to_json()).unwrap();
    let (a, b) = (
        fit.predict(&design).unwrap(),
        back.predict(&design).unwrap(),
    );
    assert!((a[7] - b[7]).abs() < 1e-12 && (a[7] - fit.fitted()[7]).abs() < 1e-9);
}

/// Credibility premiums given the structure parameters, against
/// `actuar::cm` on Hachemeister's data: Bühlmann–Straub with one level, and
/// Bühlmann–Gisler's hierarchy (two cohorts of states) with each of
/// actuar's estimators. The structure parameters are read from the
/// reference and plugged in; the premiums, factors and collective mean are
/// checked (`validation/scripts/r_credibility_hierarchical.R`).
#[test]
fn credibility_premiums_match_actuar() {
    use prospicio_bayes::credibility::credibility_premiums;

    let data = reference("hachemeister.csv");
    let mut ratios = vec![vec![0.0; 12]; 5];
    let mut weights = vec![vec![0.0; 12]; 5];
    for row in &data {
        let i = row.number("state").unwrap() as usize - 1;
        let j = row.number("quarter").unwrap() as usize - 1;
        ratios[i][j] = row.number("ratio").unwrap();
        weights[i][j] = row.number("weight").unwrap();
    }
    let cases = reference("credibility_hierarchical_actuar.csv");
    let value = |params: &str, quantity: &str, arg: &str| -> f64 {
        cases
            .iter()
            .find(|c| {
                c.get("params") == params && c.get("quantity") == quantity && c.get("arg") == arg
            })
            .and_then(|c| c.number("expected"))
            .unwrap_or_else(|| panic!("{params} {quantity} {arg}"))
    };
    check(&cases, |c| {
        let params = c.get("params");
        let hierarchy = params.contains("levels=cohort");
        let levels: Vec<Vec<usize>> = if hierarchy {
            vec![vec![0, 1, 0, 1, 1]]
        } else {
            Vec::new()
        };
        let s2 = value(params, "within_variance", "");
        let between: Vec<f64> = if hierarchy {
            vec![
                value(params, "between_variance", "0"),
                value(params, "between_variance", "1"),
            ]
        } else {
            vec![value(params, "between_variance", "0")]
        };
        let p = credibility_premiums(&ratios, &weights, &levels, s2, &between, None).ok()?;
        let index = || Some(c.number("arg")? as usize - 1);
        match c.get("quantity") {
            "within_variance" => Some(s2),
            "between_variance" => Some(between[c.number("arg")? as usize]),
            "collective_mean" => Some(p.collective_mean),
            "credibility" => Some(p.credibility[index()?]),
            "premium" => Some(p.premiums[index()?]),
            "cohort_credibility" => Some(p.level_credibility[0][index()?]),
            "cohort_premium" => Some(p.level_premiums[0][index()?]),
            _ => None,
        }
    });
}

/// The Bayesian Bühlmann–Straub model sampled with NUTS against its exact
/// posterior moments by grid integration
/// (`validation/scripts/credibility_bayes_grid.py`), on Hachemeister's
/// data with half-normal priors of 60,000 on `σ` and 1,500 on `τ`.
#[test]
fn bayesian_credibility_matches_grid_integration() {
    use prospicio_bayes::credibility::{
        CredibilityPriors, HierarchicalCredibility, Sampler, ScalePrior,
    };

    let data = reference("hachemeister.csv");
    let mut ratios = vec![vec![0.0; 12]; 5];
    let mut weights = vec![vec![0.0; 12]; 5];
    for row in &data {
        let i = row.number("state").unwrap() as usize - 1;
        let j = row.number("quarter").unwrap() as usize - 1;
        ratios[i][j] = row.number("ratio").unwrap();
        weights[i][j] = row.number("weight").unwrap();
    }
    let fit = HierarchicalCredibility::default()
        .priors(CredibilityPriors {
            collective_mean: None,
            within: Some(ScalePrior::HalfNormal(60000.0)),
            between: vec![ScalePrior::HalfNormal(1500.0)],
        })
        .sampler(Sampler {
            chains: 4,
            tune: 1000,
            draws: 2500,
            seed: 20261016,
            ..Sampler::default()
        })
        .fit(&ratios, &weights, &[])
        .unwrap();
    let summary = fit.summary().unwrap();
    assert!(summary.iter().all(|s| s.rhat < 1.01), "{summary:?}");
    check(&reference("credibility_bayes_grid.csv"), |c| {
        let param = c.get("param");
        let index = |name: &str| -> Option<usize> {
            param
                .strip_prefix(name)?
                .strip_prefix('[')?
                .strip_suffix(']')?
                .parse()
                .ok()
        };
        match (param, c.get("quantity")) {
            ("within_variance", q) => Some(if q == "mean" {
                summary[0].mean
            } else {
                summary[0].sd
            }),
            ("between_variance[0]", q) => Some(if q == "mean" {
                summary[1].mean
            } else {
                summary[1].sd
            }),
            ("collective_mean", _) => Some(fit.collective_mean),
            _ => {
                if let Some(i) = index("premium") {
                    Some(fit.premiums[i])
                } else {
                    index("credibility").map(|i| fit.credibility[i])
                }
            }
        }
    });
}

/// The leave-one-out PIT against R `loo`'s `E_loo` of `y_rep <= y` under
/// the PSIS weights (`validation/scripts/r_loo_pit.R`), on the normal model
/// of `elpd_matches_loo` with one replicated dataset per draw.
#[test]
fn loo_pit_matches_loo() {
    use prospicio_bayes::ppc::loo_pit;
    let read = |name: &str| -> (usize, Vec<f64>) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(name);
        let text = std::fs::read_to_string(path).expect(name);
        let mut lines = text.lines();
        let n = lines.next().unwrap().split(',').count();
        let values = lines
            .flat_map(|l| {
                l.split(',')
                    .map(|v| v.parse::<f64>().unwrap())
                    .collect::<Vec<_>>()
            })
            .collect();
        (n, values)
    };
    let (n, ll) = read("data/loo_loglik.csv");
    let (_, y_rep) = read("data/loo_pit_yrep.csv");
    let (_, y) = read("data/loo_pit_y.csv");
    assert_eq!(y.len(), n);
    let p = loo_pit(&y, &y_rep, &ll, None, 0).unwrap();
    check(&reference("loo_pit.csv"), |c| match c.get("quantity") {
        "loo_pit" => Some(p.pit[c.number("index")? as usize]),
        _ => None,
    });
}

/// The negative binomial GLM with `θ` estimated, against R's
/// `MASS::glm.nb(Days ~ Eth + Sex + Age + Lrn, data = quine)` (MASS
/// 7.3-60; `validation/scripts/r_glm_nb.R` writes the data and prints
/// these values).
#[test]
fn negative_binomial_theta_matches_mass_glm_nb() {
    use prospicio_glm::negbin::NegativeBinomialGlm;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/quine.csv");
    let text = std::fs::read_to_string(path).expect("quine.csv");
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .unwrap()
        .split(',')
        .map(|h| h.trim_matches('"').to_string())
        .collect();
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); header.len()];
    for line in lines {
        for (c, v) in cols.iter_mut().zip(line.split(',')) {
            c.push(v.parse().unwrap());
        }
    }
    let n = cols[0].len();
    let mut names = vec!["(Intercept)".to_string()];
    names.extend(header[1..].iter().cloned());
    let mut columns = vec![vec![1.0; n]];
    columns.extend(cols[1..].iter().cloned());
    let d = Design::new(names, columns).unwrap();
    let fit = NegativeBinomialGlm::new(Link::Log)
        .fit(&d, &cols[0])
        .unwrap();
    let close = |a: f64, b: f64, tol: f64| (a - b).abs() <= tol * b.abs().max(1.0);
    assert!(
        close(fit.theta(), 1.27489264581101, 1e-7),
        "{}",
        fit.theta()
    );
    assert!(
        close(fit.se_theta(), 0.161035178973597, 1e-6),
        "{}",
        fit.se_theta()
    );
    assert!(
        close(fit.log_likelihood(), -546.575509144992, 1e-9),
        "{}",
        fit.log_likelihood()
    );
    let want = [
        2.89458001692052,
        -0.569371703399029,
        0.0823202641037287,
        -0.448428148590178,
        0.0880801397149139,
        0.356900947762783,
        0.292109142827854,
    ];
    for (got, want) in fit.glm().coefficients().iter().zip(want) {
        assert!(close(*got, want, 1e-7), "{got} vs {want}");
    }
}

/// R's `anova(small, big, test = "F")` and `test = "Chisq"`, `BIC` and
/// variance inflation factors (`1 / (1 - R^2)` by `lm`) on the quine
/// design; `validation/scripts/r_glm_nested.R` prints these values.
#[test]
fn nested_glm_tests_match_r_anova() {
    use prospicio_glm::nested::{f_test, likelihood_ratio_test};
    use prospicio_models::metrics::vif;
    let (header, cols) = quine();
    let n = cols[0].len();
    let design = |k: usize| {
        let mut names = vec!["(Intercept)".to_string()];
        names.extend(header[1..=k].iter().cloned());
        let mut columns = vec![vec![1.0; n]];
        columns.extend(cols[1..=k].iter().cloned());
        Design::new(names, columns).unwrap()
    };
    let (small, big) = (design(2), design(6));
    let close = |a: f64, b: f64, tol: f64| (a - b).abs() <= tol * b.abs().max(1e-300);
    let log_days: Vec<f64> = cols[0].iter().map(|d| (d + 1.0).ln()).collect();
    let gaussian = Glm::new(Family::Gaussian, Link::Identity);
    let (s, b) = (
        gaussian.fit(&small, &log_days).unwrap(),
        gaussian.fit(&big, &log_days).unwrap(),
    );
    let f = f_test(&s, &b).unwrap();
    assert!(close(f.statistic, 1.2950554093611, 1e-10), "{f:?}");
    assert!(close(f.p_value, 0.274969956093846, 1e-9), "{f:?}");
    assert_eq!((f.df, f.df_resid), (4.0, 139.0));
    assert!(close(f.deviance_drop, 5.12796060003762, 1e-10));
    let lr = likelihood_ratio_test(&s, &b).unwrap();
    assert!(close(lr.p_value, 0.26930041235005, 1e-9), "{lr:?}");
    let poisson = Glm::new(Family::Poisson, Link::Log);
    let (sp, bp) = (
        poisson.fit(&small, &cols[0]).unwrap(),
        poisson.fit(&big, &cols[0]).unwrap(),
    );
    let lr = likelihood_ratio_test(&sp, &bp).unwrap();
    assert!(close(lr.statistic, 178.384532973985, 1e-9), "{lr:?}");
    assert!(close(lr.p_value, 1.65752518873386e-37, 1e-6), "{lr:?}");
    // R's BIC counts no dispersion for the Poisson.
    assert!(close(bp.bic(), 2320.06887663732, 1e-11), "{}", bp.bic());
    let v = vif(&big).unwrap();
    let want = [
        1.00326550828182,
        1.08256046985237,
        2.09706084577139,
        1.91041499741150,
        1.85737573451634,
        1.44140349290722,
    ];
    assert!(v[0].is_nan());
    for (got, want) in v[1..].iter().zip(want) {
        assert!(close(*got, want, 1e-11), "{v:?}");
    }
}

/// The quine data of `validation/data/quine.csv`: its header and columns,
/// `Days` first.
fn quine() -> (Vec<String>, Vec<Vec<f64>>) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/quine.csv");
    let text = std::fs::read_to_string(path).expect("quine.csv");
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .unwrap()
        .split(',')
        .map(|h| h.trim_matches('"').to_string())
        .collect();
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); header.len()];
    for line in lines {
        for (c, v) in cols.iter_mut().zip(line.split(',')) {
            c.push(v.parse().unwrap());
        }
    }
    (header, cols)
}

/// R's `residuals.glm` (response, Pearson, deviance, working), working
/// weights, `hatvalues`, `rstandard`, `cooks.distance` and
/// `predict(type = "response", se.fit = TRUE)` on the quine design, rows 1,
/// 50 and 146; `validation/scripts/r_glm_diagnostics.R` prints these.
#[test]
fn glm_diagnostics_match_r() {
    let (header, cols) = quine();
    let n = cols[0].len();
    let mut names = vec!["(Intercept)".to_string()];
    names.extend(header[1..].iter().cloned());
    let mut columns = vec![vec![1.0; n]];
    columns.extend(cols[1..].iter().cloned());
    let design = Design::new(names, columns).unwrap();
    let rows = [0, 49, 145];
    let check_rows = |label: &str, got: &[f64], want: [f64; 3], tol: f64| {
        for (k, &i) in rows.iter().enumerate() {
            let (g, w) = (got[i], want[k]);
            assert!(
                (g - w).abs() <= tol * w.abs(),
                "{label} row {i}: {g} against {w}"
            );
        }
    };
    let check_all = |fit: &GlmFit, y: &[f64], tol: f64, want: [[f64; 3]; 10]| {
        let d = fit.diagnostics(&design, y).unwrap();
        let se = fit.prediction_se(&design).unwrap();
        let labels = [
            "response",
            "pearson",
            "deviance",
            "working",
            "weights",
            "hat",
            "rstandard",
            "rstandard_pearson",
            "cooks",
            "se",
        ];
        let got: [&[f64]; 10] = [
            &d.response,
            &d.pearson,
            &d.deviance,
            &d.working,
            &d.working_weights,
            &d.leverage,
            &d.standardized_deviance,
            &d.standardized_pearson,
            &d.cooks_distance,
            &se,
        ];
        for ((label, got), want) in labels.iter().zip(got).zip(want) {
            check_rows(label, got, want, tol);
        }
        let sum_sq: f64 = d.deviance.iter().map(|r| r * r).sum();
        assert!((sum_sq - fit.deviance()).abs() < 1e-9 * fit.deviance());
    };
    let poisson = Glm::new(Family::Poisson, Link::Log)
        .fit(&design, &cols[0])
        .unwrap();
    check_all(
        &poisson,
        &cols[0],
        1e-8,
        [
            [-23.17672035966231, 6.17910643217398, 23.40815851088593],
            [-4.61904722731264, 1.87842609619715, 6.349332916043],
            [-6.01850067312018, 1.7322701201506, 5.22406442044183],
            [-0.920561535758869, 0.571034766532261, 1.722221269990082],
            [25.1767203602903, 10.8208935677958, 13.5918414895791],
            [0.1043236171522393, 0.0450395922904775, 0.0355182959781139],
            [-6.35935035206074, 1.77264973950398, 5.31938625785304],
            [-4.88064074535815, 1.92221264534186, 6.46518716882067],
            [0.396357569174138, 0.0248951016628828, 0.2198985132111394],
            [1.620656205333161, 0.698117923072381, 0.694808641916209],
        ],
    );
    let days1: Vec<f64> = cols[0].iter().map(|d| d + 1.0).collect();
    let gamma = Glm::new(Family::Gamma, Link::Log)
        .fit(&design, &days1)
        .unwrap();
    check_all(
        &gamma,
        &days1,
        // R's Gamma fit moves by about 1e-8 in its coefficients when
        // restarted from its own estimates, even at `epsilon = 1e-14`.
        1e-6,
        [
            [-23.97912834927767, 5.32274173824275, 22.28535875532715],
            [-0.888802930874514, 0.419865370598274, 1.418127108875726],
            [-1.61718788235856, 0.372299108023747, 1.03453737515037],
            [-0.888802930874514, 0.419865370598274, 1.418127108875726],
            [1.0, 1.0, 1.0],
            [0.0715119069064243, 0.053063888006668, 0.0423797414730783],
            [-1.953301460484155, 0.445275317405806, 1.230400470039053],
            [-1.07353021989492, 0.502165280903462, 1.686613072903431],
            [
                0.01268038760198572,
                0.00201870897737342,
                0.01798447905736641,
            ],
            [6.19897847056201, 2.50915740297474, 2.7796278647662],
        ],
    );
}

/// R's `poly(x, 3)` columns and their `predict` on new values, and the
/// fitted values and predictions of `lm(y ~ ns(x, knots = c(10, 25)))`,
/// which a natural spline in another basis must reproduce;
/// `validation/scripts/r_terms_spline.R` prints these.
#[test]
// R's printed digits, kept as printed.
#[allow(clippy::excessive_precision)]
fn orthogonal_polys_and_natural_splines_match_r() {
    use prospicio_models::Fitted;
    let x = vec![1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 37.0, 46.0];
    let y: Vec<f64> = x
        .iter()
        .map(|v: &f64| (v / 7.0).sin() * 10.0 + v / 3.0)
        .collect();
    let train = Frame::new(vec![("x".into(), Column::Numeric(x.clone()))]).unwrap();
    let new = Frame::new(vec![("x".into(), Column::Numeric(vec![0.0, 20.0, 50.0]))]).unwrap();
    let close = |a: f64, b: f64, tol: f64| (a - b).abs() <= tol * b.abs().max(1.0);
    let poly = Terms::new().orthogonal_poly("x", 3).fit(&train).unwrap();
    assert_eq!(poly.names(), ["poly(x)1", "poly(x)2", "poly(x)3"]);
    let want_train = [
        [
            -0.352221734672833409,
            -0.330874962874479872,
            -0.288181419277772799,
            -0.224141103882712189,
            -0.138754016689298015,
            -0.032020157697530312,
            0.096060473092590914,
            0.245487875681065726,
            0.416262050067894018,
            0.608382996253075903,
        ],
        [
            0.35122943174541682,
            0.280273990988766653,
            0.148710777919145137,
            -0.022764870576091854,
            -0.203109949165910131,
            -0.350933784075597166,
            -0.414498033086762585,
            -0.331716685537337852,
            -0.030156062321576135,
            0.572965184109947123,
        ],
        [
            -0.3465733261516025232,
            -0.1995422180872851448,
            0.0393833325172275941,
            0.2732597379272244109,
            0.3899959748244423707,
            0.3023428142476387026,
            0.0012117948466838812,
            -0.3776760605498226675,
            -0.4738117850534137032,
            0.391409735478907117,
        ],
    ];
    let want_new = [
        [
            -0.373568506471186945,
            0.053366929495883855,
            0.693770083446490049,
        ],
        [
            0.42563409531662572,
            -0.40710684134127828,
            0.93069886459138995,
        ],
        [
            -0.51316818879627024,
            0.11467618565785501,
            1.2685068455087305,
        ],
    ];
    let (dt, dn) = (poly.design(&train).unwrap(), poly.design(&new).unwrap());
    for j in 0..3 {
        for (g, w) in dt.column(j).iter().zip(want_train[j]) {
            assert!(close(*g, w, 1e-12), "poly {j}: {g} against {w}");
        }
        for (g, w) in dn.column(j).iter().zip(want_new[j]) {
            assert!(close(*g, w, 1e-12), "poly new {j}: {g} against {w}");
        }
    }
    let spline = Terms::new()
        .intercept()
        .natural_spline("x", &[10.0, 25.0])
        .fit(&train)
        .unwrap();
    assert_eq!(spline.names().len(), 4);
    let fit = Glm::new(Family::Gaussian, Link::Identity)
        .fit(&spline.design(&train).unwrap(), &y)
        .unwrap();
    let fitted = [
        1.241601572941901,
        3.1981637211770977,
        6.9609391541493313,
        11.6654219067441307,
        14.617484841300687,
        12.1661124528715039,
        5.4444506082478128,
        1.5608468083302753,
        6.0187503707099408,
        17.083713098715517,
    ];
    for (g, w) in fit.fitted().iter().zip(fitted) {
        assert!(close(*g, w, 1e-9), "ns fitted {g} against {w}");
    }
    let predicted = fit.predict(&spline.design(&new).unwrap()).unwrap();
    for (g, w) in predicted.iter().zip([
        -0.72122511127239264,
        7.71926159866044337,
        22.49421993091769423,
    ]) {
        assert!(close(*g, w, 1e-9), "ns predicted {g} against {w}");
    }
}

/// `validation/data/<name>` as named string columns, quotes stripped.
fn glmm_data(name: &str) -> Vec<(String, Vec<String>)> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join(name);
    let text = std::fs::read_to_string(path).expect(name);
    let mut lines = text.lines();
    let unquote = |v: &str| v.trim_matches('"').to_string();
    let header: Vec<String> = lines.next().unwrap().split(',').map(unquote).collect();
    let mut cols: Vec<Vec<String>> = vec![Vec::new(); header.len()];
    for line in lines {
        for (c, v) in cols.iter_mut().zip(line.split(',')) {
            c.push(unquote(v));
        }
    }
    header.into_iter().zip(cols).collect()
}

fn glmm_column<'a>(data: &'a [(String, Vec<String>)], name: &str) -> &'a [String] {
    &data.iter().find(|(n, _)| n == name).unwrap().1
}

/// A GLMM case of `validation/reference/glmm_r.csv`: the design (factors
/// in R's treatment coding, R's reference level), grouping factors,
/// response and family.
fn glmm_case(
    dataset: &str,
    model: &str,
) -> (
    prospicio_glm::glmm::Glmm,
    Design,
    Vec<prospicio_glm::glmm::Grouping>,
    Vec<f64>,
) {
    use prospicio_glm::glmm::{Glmm, Grouping};
    let num = |data: &[(String, Vec<String>)], name: &str| -> Vec<f64> {
        glmm_column(data, name)
            .iter()
            .map(|v| v.parse().unwrap())
            .collect()
    };
    let cat = |data: &[(String, Vec<String>)], name: &str| {
        Column::Categorical(glmm_column(data, name).to_vec())
    };
    let group = |data: &[(String, Vec<String>)], name: &str| {
        Grouping::new(name, glmm_column(data, name).to_vec())
    };
    let (terms, frame, groups, y, offset) = match dataset {
        "salamanders" => {
            let data = glmm_data("glmm_salamanders.csv");
            let frame = Frame::new(vec![
                ("spp".into(), cat(&data, "spp")),
                ("mined".into(), cat(&data, "mined")),
            ])
            .unwrap();
            let terms = Terms::new()
                .intercept()
                .factor_with_reference("spp", "GP")
                .factor_with_reference("mined", "yes");
            let groups = vec![group(&data, "site")];
            let y = num(&data, "count");
            (terms, frame, groups, y, None)
        }
        "grouseticks" => {
            let data = glmm_data("glmm_grouseticks.csv");
            let frame = Frame::new(vec![
                ("year".into(), cat(&data, "year")),
                ("height".into(), Column::Numeric(num(&data, "height"))),
            ])
            .unwrap();
            let terms = Terms::new()
                .intercept()
                .factor_with_reference("year", "95")
                .numeric("height");
            let location = group(&data, "location");
            let brood = Grouping::nested(&location, &group(&data, "brood")).unwrap();
            let y = num(&data, "ticks");
            (terms, frame, vec![location, brood], y, None)
        }
        _ => {
            let data = glmm_data("glmm_simulated.csv");
            let frame = Frame::new(vec![
                ("x".into(), Column::Numeric(num(&data, "x"))),
                ("urban".into(), cat(&data, "urban")),
            ])
            .unwrap();
            let terms = Terms::new()
                .intercept()
                .numeric("x")
                .factor_with_reference("urban", "no");
            let mut groups = vec![group(&data, "territory")];
            if model != "tweedie_one" {
                groups.push(group(&data, "class"));
            }
            let (y, offset) = match model {
                "poisson" => (
                    num(&data, "count"),
                    Some(num(&data, "exposure").iter().map(|e| e.ln()).collect()),
                ),
                "gamma" => (num(&data, "severity"), None),
                _ => (num(&data, "pure"), None),
            };
            (terms, frame, groups, y, offset)
        }
    };
    let mut design = terms.fit(&frame).unwrap().design(&frame).unwrap();
    if let Some(o) = offset {
        design = design.with_offset(o).unwrap();
    }
    let family = match model {
        m if m.starts_with("poisson") => Family::Poisson,
        "gamma" => Family::Gamma,
        _ => Family::Tweedie { power: 1.5 },
    };
    let mut spec = Glmm::new(family);
    if model == "poisson_agq9" {
        spec = spec.quadrature(9);
    }
    (spec, design, groups, y)
}

/// Random-intercept GLMMs against glmmTMB (Laplace; Poisson, gamma and
/// Tweedie with power 1.5; one, nested and crossed factors) and lme4's
/// `glmer` (Laplace, and quadrature with 9 nodes): coefficients and their
/// standard errors, each factor's standard deviation and its log's
/// standard error, the dispersion, the log-likelihood, and every level's
/// predicted intercept with its conditional (lme4) or prediction
/// (glmmTMB) variance. `validation/scripts/r_glmm.R` writes the data and
/// the reference.
#[test]
fn glmm_matches_glmmtmb_and_glmer() {
    use std::collections::BTreeMap;
    let cases = reference("glmm_r.csv");
    let mut fits = BTreeMap::new();
    // R's coefficient names: `spp[PR]` is `sppPR`.
    let r_name = |name: &str| name.replace(['[', ']'], "");
    check(&cases, |case| {
        // glmer's Laplace fit is the same model as glmmTMB's Poisson.
        let model = match case.get("model") {
            "poisson_glmer" => "poisson",
            m => m,
        };
        let key = (case.get("dataset").to_string(), model.to_string());
        let fit = fits.entry(key.clone()).or_insert_with(|| {
            let (spec, design, groups, y) = glmm_case(&key.0, &key.1);
            let fit = spec
                .fit(&design, &groups, &y)
                .unwrap_or_else(|e| panic!("{key:?}: {e}"));
            assert!(fit.converged(), "{key:?} did not converge");
            fit
        });
        let arg = case.get("arg");
        let coef = || fit.names().iter().position(|n| r_name(n) == arg);
        let factor = |name: &str| fit.group_names().iter().position(|n| n == name);
        let level = |kind: &str| -> Option<f64> {
            let (g, l) = arg.split_once('|')?;
            let re = fit.random_effects().iter().find(|r| r.name == g)?;
            let j = re.levels.iter().position(|v| v == l)?;
            Some(match kind {
                "ranef" => re.values[j],
                "conditional_var" => re.conditional_variance[j],
                _ => re.prediction_variance[j],
            })
        };
        match case.get("quantity") {
            "coef" => coef().map(|j| fit.coefficients()[j]),
            "se" => coef().map(|j| fit.standard_errors()[j]),
            "sd" => factor(arg).map(|k| fit.sd()[k]),
            "se_log_sd" => factor(arg).map(|k| fit.sd_standard_errors()[k] / fit.sd()[k]),
            "dispersion" => Some(fit.dispersion()),
            "se_log_dispersion" => Some(fit.dispersion_standard_error() / fit.dispersion()),
            "loglik" => Some(fit.log_likelihood()),
            q => level(q),
        }
    });
}

/// Double GLMs against `dglm` 1.8.6 corrected (its mean step's weights and
/// offset, `knowledge/findings/dglm-mean-step-weights.md`), by ML and REML:
/// the mean's and dispersion's coefficients with their standard errors,
/// `m2loglik` and the REML criterion, with prior weights and with an
/// offset on the log dispersion too. On the Gaussian and Poisson without
/// offsets the corrected loop is `dglm`; `validation/scripts/r_dglm.R`
/// checks that and writes the reference.
#[test]
fn double_glms_match_dglm() {
    use prospicio_glm::dglm::DoubleGlm;
    use std::collections::BTreeMap;
    let data = glmm_data("glmm_simulated.csv");
    let num = |name: &str| -> Vec<f64> {
        glmm_column(&data, name)
            .iter()
            .map(|v| v.parse().unwrap())
            .collect()
    };
    let frame = Frame::new(vec![
        ("x".into(), Column::Numeric(num("x"))),
        (
            "urban".into(),
            Column::Categorical(glmm_column(&data, "urban").to_vec()),
        ),
    ])
    .unwrap();
    let design = |terms: Terms| terms.fit(&frame).unwrap().design(&frame).unwrap();
    let both = || {
        Terms::new()
            .intercept()
            .numeric("x")
            .factor_with_reference("urban", "no")
    };
    let urban = || {
        Terms::new()
            .intercept()
            .factor_with_reference("urban", "no")
    };
    let mean = design(both());
    let cases = reference("dglm_r.csv");
    let mut fits = BTreeMap::new();
    let r_name = |name: &str| name.replace(['[', ']'], "");
    check(&cases, |case| {
        let key = (case.get("case").to_string(), case.get("method").to_string());
        let fit = fits.entry(key.clone()).or_insert_with(|| {
            let (family, link, y, z, x) = match key.0.as_str() {
                "gaussian" => (
                    Family::Gaussian,
                    Link::Identity,
                    num("severity"),
                    design(both()),
                    mean.clone(),
                ),
                "poisson" => (
                    Family::Poisson,
                    Link::Log,
                    num("count"),
                    design(urban()),
                    mean.clone(),
                ),
                "poisson_offset" => (
                    Family::Poisson,
                    Link::Log,
                    num("count"),
                    design(urban()),
                    mean.clone()
                        .with_offset(num("exposure").iter().map(|e| e.ln()).collect())
                        .unwrap(),
                ),
                "gamma" => (
                    Family::Gamma,
                    Link::Log,
                    num("severity"),
                    design(both()),
                    mean.clone(),
                ),
                "gaussian_weighted" => (
                    Family::Gaussian,
                    Link::Identity,
                    num("severity"),
                    design(both()),
                    mean.clone().with_weights(num("exposure")).unwrap(),
                ),
                "gamma_doffset" => (
                    Family::Gamma,
                    Link::Log,
                    num("severity"),
                    design(urban())
                        .with_offset(num("exposure").iter().map(|e| e.ln()).collect())
                        .unwrap(),
                    mean.clone(),
                ),
                _ => (
                    Family::Tweedie { power: 1.5 },
                    Link::Log,
                    num("pure"),
                    design(urban()),
                    mean.clone(),
                ),
            };
            let mut spec = DoubleGlm::new(family, link);
            if key.1 == "reml" {
                spec = spec.reml();
            }
            spec.fit(&x, &z, &y)
                .unwrap_or_else(|e| panic!("{key:?}: {e}"))
        });
        let arg = case.get("arg");
        let at = |names: &[String]| names.iter().position(|n| r_name(n) == arg);
        match case.get("quantity") {
            "coef" => at(fit.names()).map(|j| fit.coefficients()[j]),
            "se" => at(fit.names()).map(|j| fit.standard_errors()[j]),
            "dcoef" => at(fit.dispersion_names()).map(|j| fit.dispersion_coefficients()[j]),
            "dse" => at(fit.dispersion_names()).map(|j| fit.dispersion_standard_errors()[j]),
            "m2loglik" => Some(fit.minus_two_log_likelihood()),
            "reml_criterion" => Some(fit.reml_criterion()),
            _ => None,
        }
    });
}
