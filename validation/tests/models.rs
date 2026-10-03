//! Model parity: GLMs against statsmodels on a synthetic portfolio
//! (`validation/scripts/statsmodels_glm.py`).

use act_glm::{Dispersion, Glm, GlmFit};
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
    glm.fit(&d, &y).unwrap_or_else(|e| panic!("{case}: {e}"))
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
