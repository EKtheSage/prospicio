//! GLM parity at scale: the claim frequency GLM of Noll, Salzmann and
//! Wüthrich (2018) on all 678,013 policies of freMTPL2freq against
//! statsmodels (`validation/scripts/statsmodels_fremtpl2.py`).
//!
//! The data is downloaded, not committed: `python
//! validation/scripts/fetch_fremtpl2.py` writes
//! `validation/data/external/freMTPL2freq_glm.csv`. Run it with
//! `cargo test --release -p act-validation --test fremtpl2` (about 30 s;
//! some six minutes unoptimized). Without the file, or in an unoptimized
//! build, the test says so and passes, unless `RISK_RS_REQUIRE_FREMTPL2`
//! is set, as CI's `fremtpl2` job sets it after fetching (and caching)
//! the file.

use act_glm::{Glm, GlmFit, Robust};
use act_models::{Column, Design, Frame, Model, Terms};
use act_validation::{check, reference};

const DATA: &str = "data/external/freMTPL2freq_glm.csv";

/// The prepared portfolio as named columns, or `None` if it has not been
/// fetched.
fn portfolio() -> Option<Vec<(String, Vec<String>)>> {
    let required = std::env::var_os("RISK_RS_REQUIRE_FREMTPL2").is_some();
    if cfg!(debug_assertions) && !required {
        eprintln!("skipping freMTPL2 parity in an unoptimized build: run it with --release");
        return None;
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(DATA);
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

/// The GLM design of Noll, Salzmann and Wüthrich, with the offset
/// log(Exposure).
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
    let log_exposure = numeric(data, "Exposure").iter().map(|e| e.ln()).collect();
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
        .with_offset(log_exposure)
        .unwrap()
}

#[test]
fn fremtpl2_frequency_glm_matches_statsmodels() {
    let Some(data) = portfolio() else {
        return;
    };
    let d = design(&data);
    let y = numeric(&data, "ClaimNb");
    assert_eq!(y.len(), 678_013);
    let poisson: GlmFit = Glm::new(act_models::Family::Poisson, act_models::Link::Log)
        .fit(&d, &y)
        .unwrap();
    let quasi = Glm::over_dispersed_poisson().fit(&d, &y).unwrap();
    let hc0 = poisson.robust_std_errors(&d, &y, Robust::Hc0).unwrap();
    let cases = reference("fremtpl2_statsmodels.csv");
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
