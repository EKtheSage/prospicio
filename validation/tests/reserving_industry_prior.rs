//! The industry prior (`IndustryPrior`, `reserving-bayes.md` decision 29,
//! issue #604) recovers the parameters of simulated companies.
//!
//! No reference implementation builds this prior, so the check is
//! recovery: companies drawn from the model the prior feeds, with known
//! between-company and within-company cvs, mean loss ratio and pattern
//! spread, and the estimates compared with the truth within a few Monte
//! Carlo standard errors of the estimators (each derived in its test). The
//! companies are drawn as the normalised Bayesian BF with a common
//! multiplier describes one program:
//!
//! ```text
//! s_c ~ Gamma(shape 1 / cb^2, mean 1)
//! U_cw | s_c ~ Gamma(shape 1 / cw^2, mean s_c apriori premium)
//! beta_cd = mu_d + sd_d z_cd,   p_cd = exp(beta_cd) / sum_k exp(beta_ck)
//! increment_cwd ~ Gamma(mean U_cw p_cd, variance phi U_cw p_cd)  (or exact)
//! ```
//!
//! Each company is a 10 x 10 upper triangle. Runs in the plain `cargo
//! test` (no NUTS), in about a second in release.

use prospicio_core::StreamRng;
use prospicio_prob::{Distribution, Gamma};
use prospicio_reserving::{
    DevelopmentColumn, Grain, IndustryPrior, IndustryPriorSettings, Lag, Long, Month, Triangle,
};

const N_ORIGINS: usize = 10;
const APRIORI: f64 = 0.7;
const CB: f64 = 0.15;
const CW: f64 = 0.1;
const PREMIUM: f64 = 1000.0;
/// The true pattern's mean log share ratios: shares falling from 0.3.
const MU: [f64; 9] = [-0.2, -0.5, -0.8, -1.1, -1.4, -1.7, -2.0, -2.3, -2.6];
const SD: [f64; 9] = [0.1, 0.1, 0.12, 0.12, 0.15, 0.15, 0.2, 0.2, 0.25];

fn normal(rng: &mut StreamRng) -> f64 {
    let (u, v) = (rng.next_open01(), rng.next_open01());
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

fn gamma(rng: &mut StreamRng, mean: f64, cv: f64) -> f64 {
    let shape = cv.powi(-2);
    Gamma::new(shape, mean / shape).unwrap().sample(rng, 1)[0]
}

/// Company `c`'s triangle, with process scale `phi` (`None`: the cells are
/// exactly `U p`).
fn company(c: u64, phi: Option<f64>) -> Triangle {
    let mut rng = StreamRng::new(20261102, c);
    let s = gamma(&mut rng, 1.0, CB);
    let beta: Vec<f64> = std::iter::once(0.0)
        .chain(MU.iter().zip(SD).map(|(m, sd)| m + sd * normal(&mut rng)))
        .collect();
    let total: f64 = beta.iter().map(|b| b.exp()).sum();
    let shares: Vec<f64> = beta.iter().map(|b| b.exp() / total).collect();
    let (mut origin, mut ages, mut values) = (vec![], vec![], vec![]);
    for w in 0..N_ORIGINS {
        let u = gamma(&mut rng, s * APRIORI * PREMIUM, CW);
        let mut cum = 0.0;
        for (d, p) in shares.iter().enumerate().take(N_ORIGINS - w) {
            let mean = u * p;
            cum += match phi {
                None => mean,
                Some(phi) => gamma(&mut rng, mean, (phi / mean).sqrt()),
            };
            origin.push(Month::january(2000 + w as i32));
            ages.push(12 * (d as Lag + 1));
            values.push(cum);
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("paid", &values)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

fn panel(k: u64, phi: Option<f64>) -> (Vec<Triangle>, Vec<Vec<f64>>) {
    let triangles: Vec<Triangle> = (0..k).map(|c| company(c, phi)).collect();
    (triangles, vec![vec![PREMIUM; N_ORIGINS]; k as usize])
}

fn prior(k: u64, phi: Option<f64>) -> IndustryPrior {
    let (triangles, premiums) = panel(k, phi);
    IndustryPrior::from_triangles(&triangles, "paid", &premiums).unwrap()
}

/// The marginal loss-ratio moments of one origin: the mean of a company's
/// `n` origins has variance `a^2 ((1 + cb^2)(1 + cw^2 / n) - 1)`.
fn company_mean_sd(n: f64) -> f64 {
    APRIORI * ((1.0 + CB * CB) * (1.0 + CW * CW / n) - 1.0).sqrt()
}

/// Exact cells: each company's chain ladder returns its pattern and
/// expected ultimates, so the estimators see the model's parameters with
/// sampling error only. 2,000 companies.
#[test]
fn exact_companies_recover_the_parameters() {
    let k = 2000;
    let prior = prior(k, None);
    let report = &prior.report;
    assert_eq!(report.n_companies_used, k as usize);
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    // The mean pattern has 46% developed at 24 months and 62% at 36, so
    // the 8 oldest origins count (min_developed 0.5): n = 8 a company.
    let n = report.n_origins_used as f64 / k as f64;
    assert_eq!(n, 8.0);
    let kf = k as f64;

    // apriori: the mean of k company means.
    let se = company_mean_sd(n) / kf.sqrt();
    println!("apriori {} se {se}", prior.apriori);
    assert!((prior.apriori - APRIORI).abs() < 4.0 * se);
    assert!((prior.apriori_se / se - 1.0).abs() < 0.1);

    // within: a pooled variance on k (n - 1) degrees of freedom; a gamma
    // with cv 0.1 has excess kurtosis 6 cv^2, so the relative se of the
    // variance is about sqrt((2 + 0.06) / (k (n - 1))), halved for the cv.
    let rel = ((2.06) / (kf * (n - 1.0))).sqrt() / 2.0;
    println!("within {} rel se {rel}", prior.within_cv);
    assert!((prior.within_cv / CW - 1.0).abs() < 4.0 * rel);
    assert!(prior.process_cv.abs() < 1e-6, "{}", prior.process_cv);
    assert!((prior.cv - prior.within_cv).abs() < 1e-12);

    // between: a variance on k - 1 degrees of freedom less the within
    // part; the gamma's excess kurtosis 6 cb^2 adds to the 2.
    let var_means = (company_mean_sd(n) / APRIORI).powi(2);
    let rel = var_means * ((2.0 + 6.0 * CB * CB) / kf).sqrt() / (2.0 * CB * CB);
    println!("between {} rel se {rel}", prior.between_cv);
    assert!((prior.between_cv / CB - 1.0).abs() < 4.0 * rel);
    assert_eq!(prior.common_cv(), Some(prior.between_cv));

    // The pattern: mean and sd of k exact log share ratios per age.
    for d in 0..9 {
        let (m, s) = (prior.pattern.mean[d], prior.pattern.sd[d]);
        assert!(
            (m - MU[d]).abs() < 4.0 * SD[d] / kf.sqrt(),
            "age {d}: {m} {}",
            MU[d]
        );
        assert!(
            (s / SD[d] - 1.0).abs() < 4.0 / (2.0 * kf).sqrt(),
            "age {d}: {s} {}",
            SD[d]
        );
        assert_eq!(report.companies_by_age[d], k as usize);
    }
    // The truncated prior: apriori to 36 months, the tail factor its
    // inverse share.
    let cut = prior.at_age(36).unwrap();
    let developed = prior.percent_developed[2];
    assert!((cut.apriori - prior.apriori * developed).abs() < 1e-12);
    assert!((cut.tail_factor * developed - 1.0).abs() < 1e-12);
    assert_eq!(cut.pattern.ages, vec![12, 24, 36]);
}

/// ODP process error in the cells (an ultimate's process cv about
/// `sqrt(phi / U)`, 0.05 here): the realised within cv includes it, which
/// is the prior's `cv` by default; with `subtract_process` the `cv` takes
/// it out with each company's Pearson scale; the between cv and the mean
/// are still recovered. The companies' own chain
/// ladders now carry estimation error into the immature origins and the
/// pattern, which the tolerances allow for (10% of the cvs).
#[test]
fn process_error_is_taken_out_of_the_within_cv() {
    let phi = 0.0025 * APRIORI * PREMIUM;
    let k = 1500;
    let (triangles, premiums) = panel(k, Some(phi));
    let default = IndustryPrior::from_triangles(&triangles, "paid", &premiums).unwrap();
    let prior = IndustryPriorSettings {
        subtract_process: true,
        ..IndustryPriorSettings::default()
    }
    .from_triangles(&triangles, "paid", &premiums)
    .unwrap();
    assert_eq!(default.cv, default.within_cv);
    assert_eq!(default.within_cv, prior.within_cv);
    assert_eq!(default.between_cv, prior.between_cv);
    println!(
        "apriori {} between {} within {} process {} cv {}",
        prior.apriori, prior.between_cv, prior.within_cv, prior.process_cv, prior.cv
    );
    let n = prior.report.n_origins_used as f64 / k as f64;
    let se = company_mean_sd(n) / (k as f64).sqrt();
    assert!((prior.apriori - APRIORI).abs() < 4.0 * se + 0.002);
    let process = (phi / (APRIORI * PREMIUM)).sqrt();
    assert!(prior.within_cv > CW, "{}", prior.within_cv);
    assert!((prior.process_cv / process - 1.0).abs() < 0.25, "{process}");
    assert!((prior.cv / CW - 1.0).abs() < 0.1);
    assert!((prior.between_cv / CB - 1.0).abs() < 0.1);
    // The pattern's spread now includes each company's estimation error,
    // so it is at least the true spread.
    for d in 0..9 {
        assert!((prior.pattern.mean[d] - MU[d]).abs() < 0.1, "age {d}");
        assert!(prior.pattern.sd[d] > 0.9 * SD[d], "age {d}");
    }
}

/// A small panel warns, and the settings move what counts.
#[test]
fn a_small_panel_warns() {
    let prior = prior(6, None);
    assert!(
        prior
            .report
            .warnings
            .iter()
            .any(|w| w.contains("only 6 companies")),
        "{:?}",
        prior.report.warnings
    );
    let triangles: Vec<Triangle> = (0..6).map(|c| company(c, None)).collect();
    let premiums = vec![vec![PREMIUM; N_ORIGINS]; 6];
    let mature = IndustryPriorSettings {
        min_developed: 0.95,
        min_companies: 3,
        ..IndustryPriorSettings::default()
    }
    .from_triangles(&triangles, "paid", &premiums)
    .unwrap();
    assert!(mature.report.n_origins_used < prior.report.n_origins_used);
    assert!(
        mature.report.warnings.is_empty(),
        "{:?}",
        mature.report.warnings
    );
}
