//! Model parity: GLMs against statsmodels on a synthetic portfolio
//! (`validation/scripts/statsmodels_glm.py`), with sandwich standard errors
//! (`statsmodels_glm_robust.py`).

use act_glm::{Dispersion, Glm, GlmFit, Robust};
use act_models::{Column, Design, Family, Frame, Link, Model, Terms};
use act_validation::{check, reference};

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
    use act_glm::gam::{Gam, GamFit, PSpline};
    use act_models::Fitted;
    let data = policies();
    let cases = reference("gam_mgcv.csv");
    let fit_case = |case: &str| -> GamFit {
        let d = design(&data);
        let log_exposure: Vec<f64> = numeric(&data, "exposure").iter().map(|e| e.ln()).collect();
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
        Gam::new(glm, vec![PSpline::new("age").n_basis(k)])
            .fit(&d, &y)
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
            "deviance" => Some(fit.deviance()),
            "edf" => Some(fit.edf()),
            "scale" => Some(fit.dispersion()),
            "score" => Some(fit.score()),
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
            "rhat" => act_bayes::rhat(&refs).ok(),
            "ess_bulk" => act_bayes::ess_bulk(&refs).ok(),
            "ess_tail" => act_bayes::ess_tail(&refs).ok(),
            "ess_mean" => act_bayes::ess_mean(&refs).ok(),
            "mcse_mean" => act_bayes::mcse_mean(&refs).ok(),
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
) -> (act_glm::net::ElasticNet, Design, Vec<f64>) {
    use act_glm::net::ElasticNet;
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
    spec: &act_glm::net::ElasticNet,
    d: &Design,
    y: &[f64],
    lambda: f64,
) -> act_glm::net::ElasticNet {
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
    use act_models::resample::Split;
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
    let mut runs: Vec<(String, act_glm::net::CvPath)> = Vec::new();
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
fn tweedie_power_estimate_matches_statsmodels() {
    use act_glm::tweedie::{TweedieGlm, tweedie_profile};
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
    use act_bayes::elpd::{loo, lppd, waic};
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
    let stacking = act_models::stack::stacking_weights(&lpd).unwrap();
    let pseudo = act_models::stack::pseudo_bma_weights(&lpd, None).unwrap();
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
    use act_bayes::glm::{BayesGlm, Sampler};
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
