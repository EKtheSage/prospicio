//! GLM parity at scale against statsmodels
//! (`validation/scripts/statsmodels_fremtpl2.py`): the claim frequency GLM
//! of Noll, Salzmann and Wüthrich (2018) on all 678,013 policies of
//! freMTPL2freq, a logistic regression of whether a policy has a claim on
//! the same design, and a gamma severity GLM on freMTPL2sev joined to the
//! portfolio.
//!
//! The data is downloaded, not committed: `python
//! validation/scripts/fetch_fremtpl2.py` writes
//! `validation/data/external/freMTPL2freq_glm.csv` and
//! `freMTPL2sev_glm.csv`. Run it with
//! `cargo test --release -p prospicio-validation --test fremtpl2`. Without
//! the files, or in an unoptimized build, the tests say so and pass, unless
//! `RISK_RS_REQUIRE_FREMTPL2` is set, as CI's `fremtpl2` job sets it after
//! fetching (and caching) the files.

use prospicio_glm::{Glm, GlmFit, Robust};
use prospicio_models::{Column, Design, Frame, Model, Terms};
use prospicio_validation::{check, reference};

const DATA: &str = "data/external/freMTPL2freq_glm.csv";
const SEV_DATA: &str = "data/external/freMTPL2sev_glm.csv";

/// A prepared file (`DATA` or `SEV_DATA`) as named columns, or `None` if it
/// has not been fetched.
fn portfolio(file: &str) -> Option<Vec<(String, Vec<String>)>> {
    let required = std::env::var_os("RISK_RS_REQUIRE_FREMTPL2").is_some();
    if cfg!(debug_assertions) && !required {
        eprintln!("skipping freMTPL2 parity in an unoptimized build: run it with --release");
        return None;
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(file);
    let Ok(text) = std::fs::read_to_string(&path) else {
        assert!(
            !required,
            "{} is missing: run python validation/scripts/fetch_fremtpl2.py",
            path.display()
        );
        eprintln!(
            "skipping freMTPL2 parity: {} is missing (python validation/scripts/fetch_fremtpl2.py)",
            path.display()
        );
        return None;
    };
    let mut lines = text.lines();
    let header: Vec<String> = lines.next()?.split(',').map(String::from).collect();
    let mut cols: Vec<Vec<String>> = vec![Vec::new(); header.len()];
    for line in lines {
        for (c, v) in cols.iter_mut().zip(line.split(',')) {
            c.push(v.to_string());
        }
    }
    Some(header.into_iter().zip(cols).collect())
}

fn column<'a>(data: &'a [(String, Vec<String>)], name: &str) -> &'a [String] {
    &data.iter().find(|(n, _)| n == name).unwrap().1
}

fn numeric(data: &[(String, Vec<String>)], name: &str) -> Vec<f64> {
    column(data, name)
        .iter()
        .map(|v| v.parse().unwrap())
        .collect()
}

/// The GLM design of Noll, Salzmann and Wüthrich, without an offset.
fn design(data: &[(String, Vec<String>)]) -> Design {
    let numeric_terms = ["Area", "BonusMalus", "Density"];
    let factor_terms = [
        "VehPower", "VehAge", "DrivAge", "VehBrand", "VehGas", "Region",
    ];
    let mut columns = Vec::new();
    for name in numeric_terms {
        columns.push((name.to_string(), Column::Numeric(numeric(data, name))));
    }
    for name in factor_terms {
        columns.push((
            name.to_string(),
            Column::Categorical(column(data, name).to_vec()),
        ));
    }
    let frame = Frame::new(columns).unwrap();
    Terms::new()
        .intercept()
        .numeric("Area")
        .factor("VehPower")
        .factor_with_reference("VehAge", "6-12")
        .factor_with_reference("DrivAge", "41-50")
        .numeric("BonusMalus")
        .factor("VehBrand")
        .factor("VehGas")
        .numeric("Density")
        .factor_with_reference("Region", "R24")
        .fit(&frame)
        .unwrap()
        .design(&frame)
        .unwrap()
}

/// Checks a fit (and its HC0 standard errors, as case `<case>_hc0`)
/// against the reference rows of `case`.
fn check_fit(case: &str, d: &Design, y: &[f64], fit: &GlmFit) {
    let hc0 = fit.robust_std_errors(d, y, Robust::Hc0).unwrap();
    let hc0_case = format!("{case}_hc0");
    let cases: Vec<_> = reference("fremtpl2_statsmodels.csv")
        .into_iter()
        .filter(|c| c.get("case") == case || c.get("case") == hc0_case)
        .collect();
    assert!(!cases.is_empty(), "no reference rows for {case}");
    check(&cases, |c| {
        let j = fit.names().iter().position(|n| n == c.get("term"));
        match (c.get("case") == hc0_case, c.get("quantity")) {
            (true, "std_error") => Some(hc0[j?]),
            (false, "coef") => Some(fit.coefficients()[j?]),
            (false, "std_error") => Some(fit.std_errors()[j?]),
            (false, "deviance") => Some(fit.deviance()),
            (false, "null_deviance") => Some(fit.null_deviance()),
            (false, "log_likelihood") => Some(fit.log_likelihood()),
            (false, "aic") => Some(fit.aic()),
            (false, "scale") => Some(fit.dispersion()),
            _ => None,
        }
    });
}

#[test]
fn fremtpl2_frequency_glm_matches_statsmodels() {
    let Some(data) = portfolio(DATA) else {
        return;
    };
    let log_exposure = numeric(&data, "Exposure").iter().map(|e| e.ln()).collect();
    let d = design(&data).with_offset(log_exposure).unwrap();
    let y = numeric(&data, "ClaimNb");
    assert_eq!(y.len(), 678_013);
    let poisson: GlmFit = Glm::new(
        prospicio_models::Family::Poisson,
        prospicio_models::Link::Log,
    )
    .fit(&d, &y)
    .unwrap();
    let quasi = Glm::over_dispersed_poisson().fit(&d, &y).unwrap();
    let hc0 = poisson.robust_std_errors(&d, &y, Robust::Hc0).unwrap();
    let cases: Vec<_> = reference("fremtpl2_statsmodels.csv")
        .into_iter()
        .filter(|c| c.get("case").starts_with("poisson") || c.get("case").starts_with("quasi"))
        .collect();
    check(&cases, |c| {
        let fit = match c.get("case") {
            "quasi_poisson_log" => &quasi,
            _ => &poisson,
        };
        let j = fit.names().iter().position(|n| n == c.get("term"));
        match (c.get("case"), c.get("quantity")) {
            ("poisson_log_hc0", "std_error") => Some(hc0[j?]),
            (_, "coef") => Some(fit.coefficients()[j?]),
            (_, "std_error") => Some(fit.std_errors()[j?]),
            (_, "deviance") => Some(fit.deviance()),
            (_, "null_deviance") => Some(fit.null_deviance()),
            (_, "log_likelihood") => Some(fit.log_likelihood()),
            (_, "aic") => Some(fit.aic()),
            (_, "scale") => Some(fit.dispersion()),
            _ => None,
        }
    });
}

#[test]
fn fremtpl2_has_claim_logit_matches_statsmodels() {
    let Some(data) = portfolio(DATA) else {
        return;
    };
    let d = design(&data);
    let y: Vec<f64> = numeric(&data, "ClaimNb")
        .iter()
        .map(|&n| if n > 0.0 { 1.0 } else { 0.0 })
        .collect();
    let fit = Glm::new(
        prospicio_models::Family::Binomial,
        prospicio_models::Link::Logit,
    )
    .fit(&d, &y)
    .unwrap();
    check_fit("binomial_logit", &d, &y, &fit);
}

#[test]
fn fremtpl2_gamma_severity_matches_statsmodels() {
    let Some(sev) = portfolio(SEV_DATA) else {
        return;
    };
    let d = design(&sev).with_weights(numeric(&sev, "NSev")).unwrap();
    let y = numeric(&sev, "AvgClaim");
    assert_eq!(y.len(), 24_944);
    let fit = Glm::new(prospicio_models::Family::Gamma, prospicio_models::Link::Log)
        .fit(&d, &y)
        .unwrap();
    check_fit("gamma_log_weighted", &d, &y, &fit);
}
