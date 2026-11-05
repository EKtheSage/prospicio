//! GLM parity at scale against statsmodels
//! (`validation/scripts/statsmodels_fremtpl2.py`): the claim frequency GLM
//! of Noll, Salzmann and Wüthrich (2018) on all 678,013 policies of
//! freMTPL2freq, a logistic regression of whether a policy has a claim on
//! the same design, and a gamma severity GLM on freMTPL2sev joined to the
//! portfolio; and the tabular networks' embeddings of VehBrand and Region
//! against a CANN on their indicator columns (`prospicio-nn`, #577).
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

/// The network's features as Schelldorfer and Wüthrich (2019) feed them:
/// VehPower a number, not the GLM's factor, beside the other columns, and
/// the GLM's classes of VehAge and DrivAge.
fn network_design(data: &[(String, Vec<String>)]) -> Design {
    let mut columns = Vec::new();
    for name in ["Area", "VehPower", "BonusMalus", "Density"] {
        columns.push((name.to_string(), Column::Numeric(numeric(data, name))));
    }
    for name in ["VehAge", "DrivAge", "VehBrand", "VehGas", "Region"] {
        columns.push((
            name.to_string(),
            Column::Categorical(column(data, name).to_vec()),
        ));
    }
    let frame = Frame::new(columns).unwrap();
    let mut terms = Terms::new();
    for name in ["Area", "VehPower", "BonusMalus", "Density"] {
        terms = terms.numeric(name);
    }
    for name in ["VehAge", "DrivAge", "VehBrand", "VehGas", "Region"] {
        terms = terms.factor(name);
    }
    terms.fit(&frame).unwrap().design(&frame).unwrap()
}

/// Schelldorfer and Wüthrich's comparison (2019, section 4) of a CANN on
/// indicator columns with one that enters VehBrand and Region as
/// two-dimensional embeddings, on a learning and a test sample of the
/// portfolio (every fifth policy held out). Both start at the GLM fitted on
/// the learning sample, so the comparison is of what each network adds.
#[test]
fn fremtpl2_embeddings_beat_the_one_hot_cann() {
    use prospicio_models::metrics::mean_deviance;
    use prospicio_models::{Family, Fitted, Link};
    use prospicio_nn::{Head, TabularNet, Target};

    let Some(data) = portfolio(DATA) else {
        return;
    };
    let log_exposure: Vec<f64> = numeric(&data, "Exposure").iter().map(|e| e.ln()).collect();
    let glm_design = design(&data).with_offset(log_exposure).unwrap();
    let features = network_design(&data);
    let y = numeric(&data, "ClaimNb");
    let (learn, test): (Vec<usize>, Vec<usize>) = (0..y.len()).partition(|i| i % 5 != 0);
    let pick = |rows: &[usize]| rows.iter().map(|&i| y[i]).collect::<Vec<f64>>();
    let (yl, yt) = (pick(&learn), pick(&test));
    let (gl, gt) = (glm_design.select(&learn), glm_design.select(&test));
    let glm = Glm::new(Family::Poisson, Link::Log).fit(&gl, &yl).unwrap();
    let glm_test = mean_deviance(Family::Poisson, &yt, &glm.predict(&gt).unwrap(), None).unwrap();
    let eta = |d: &Design| d.linear_predictor(glm.coefficients());
    let dl = features.select(&learn).with_offset(eta(&gl)).unwrap();
    let dt = features.select(&test).with_offset(eta(&gt)).unwrap();
    let net = |spec: TabularNet| {
        let mut spec = spec
            .hidden(&[20, 15, 10])
            .head(Head::new("freq", Family::Poisson));
        spec.epochs = 15;
        spec.batch_size = 10_000;
        spec.learning_rate = 1e-2;
        spec.seed = 1;
        let fit = spec.fit_heads(&dl, &[Target::new(&yl)]).unwrap();
        mean_deviance(Family::Poisson, &yt, &fit.predict(&dt).unwrap(), None).unwrap()
    };
    let one_hot = net(TabularNet::new());
    let embedded = net(TabularNet::new()
        .embedding("VehBrand", 2)
        .embedding("Region", 2));
    eprintln!(
        "test mean deviance: GLM {glm_test:.6}, one-hot CANN {one_hot:.6}, embeddings {embedded:.6}"
    );
    assert!(one_hot < glm_test, "{one_hot} vs the GLM's {glm_test}");
    assert!(
        embedded < one_hot,
        "{embedded} vs the one-hot CANN's {one_hot}"
    );
}
