//! The posterior predictive loss ratio of a new underwriting year
//! (`docs/design/reserving-bayes.md`, decision 28, issue #598):
//! `BayesBfNormalisedFit::new_year` and `MeyersFit::new_year`.
//!
//! There is no external reference implementation of a new year from these
//! models, so the checks are exact distributions the model implies:
//!
//! * **Conjugate case** (the normalised BF on a square triangle, every
//!   origin observed at every age): each origin's shares sum to 1 whatever
//!   the pattern draw, so the likelihood of `U_w` is `U_w^(S_w / phi)
//!   exp(-U_w / phi)` apart from the pattern, the posterior of the common
//!   multiplier `s` is a one-dimensional density known in closed form up to
//!   its constant (each `U_w` integrated out of its gamma prior), and the
//!   new year's loss ratio is `Gamma(1 / cv^2, mean apriori s)` mixed over
//!   it. Its cdf, computed by quadrature, is compared with the simulated
//!   quantiles; at `cv -> 0` the posterior of `s` is the gamma `Gamma(k +
//!   T / phi, rate k + M / phi)`, whose mean and variance the new year
//!   matches at `cv` 0.01.
//! * **Tight prior**: without the common multiplier, or with a vanishing
//!   one, the new year reproduces the prior (`apriori`, `cv`).
//! * **Pooling**: with a vague common multiplier and origins that differ
//!   only by noise, the new year's mean is the origins' posterior mean loss
//!   ratio and its cv at least the origin-to-origin `cv`.
//! * **Meyers** (group 353 Commercial Auto): given the posterior draws, the
//!   new year's log loss ratio has a known mean and variance under each
//!   level rule (a Student t level plus `sigma_m` noise, and in CAY `rho`
//!   times the newest origin's residual); and the 1997 origin held out of
//!   a fit on 1988 to 1996 lands inside its new-year interval.

use prospicio_math::special::gamma_inc;
use prospicio_prob::Distribution;
use prospicio_reserving::bayes::{
    BayesBfNormalised, Meyers, MeyersFit, MeyersModel, NewOriginLevel,
};
use prospicio_reserving::{
    DevelopmentColumn, Grain, Lag, Long, Month, ProcessDistribution, Triangle,
};
use prospicio_validation::triangle_columns;

/// A cumulative annual triangle from incremental `rows` (row `w` the
/// increments observed for origin `w`) with a `premium` column.
fn from_increments(rows: &[Vec<f64>], premium: &[f64]) -> Triangle {
    let (mut origin, mut ages, mut values, mut prem) = (vec![], vec![], vec![], vec![]);
    for (w, row) in rows.iter().enumerate() {
        let mut cum = 0.0;
        for (d, &x) in row.iter().enumerate() {
            cum += x;
            origin.push(Month::january(2001 + w as i32));
            ages.push(12 * (d as Lag + 1));
            values.push(cum);
            prem.push(premium[w]);
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("paid", &values), ("premium", &prem)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// Increments of `n` origins with premium 1,000 and loss ratios
/// `ratio(w)`, a fixed pattern, and a deterministic +-10% wobble so the
/// Pearson scale is not 0; the first `observed(w)` ages of origin `w`.
fn book(n: usize, ratio: impl Fn(usize) -> f64, observed: impl Fn(usize) -> usize) -> Triangle {
    let pattern = [0.30, 0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02];
    assert_eq!(pattern.len(), n);
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|w| {
            (0..observed(w))
                .map(|d| {
                    let wobble = 1.0 + 0.1 * ((3 * w + 5 * d + 1) as f64).sin();
                    1000.0 * ratio(w) * pattern[d] * wobble
                })
                .collect()
        })
        .collect();
    from_increments(&rows, &vec![1000.0; n])
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    sorted[((p * sorted.len() as f64) as usize).min(sorted.len() - 1)]
}

fn sorted(draws: &[f64]) -> Vec<f64> {
    let mut v = draws.to_vec();
    v.sort_by(f64::total_cmp);
    v
}

/// The conjugate case: on a square triangle the new year's loss ratio is
/// the gamma mixture over the exact posterior of `s`, at a moderate prior
/// (`cv` 0.3, `common_cv` 0.5) and in the limit `cv -> 0`, where the
/// posterior of `s` is a gamma.
#[test]
fn on_a_square_the_new_year_is_the_exact_gamma_mixture() {
    let tri = book(8, |w| 0.6 * (1.0 + 0.15 * (2.0 * w as f64).cos()), |_| 8);
    for (cv, common) in [(0.3, 0.5), (0.01, 0.5)] {
        let fit = BayesBfNormalised {
            apriori: 0.5,
            cv,
            common_cv: Some(common),
            process: ProcessDistribution::None,
            seed: 11,
            n_sims: 100,
            ..BayesBfNormalised::default()
        }
        .fit(&tri, "paid", "premium")
        .unwrap();
        assert!(
            fit.diagnostics.warnings.is_empty(),
            "{:?}",
            fit.diagnostics.warnings
        );
        let phi = fit.scale;
        let (a, k) = (cv.powi(-2), common.powi(-2));
        // Each origin's total and prior mean ultimate.
        let totals: Vec<f64> = fit.chain_ladder.latest.clone();
        let m: Vec<f64> = fit.prior_ultimate.clone();
        // log p(s), s on a log grid, each U_w integrated out of its gamma:
        // s^(k-1) e^(-k s) prod_w (a / (m_w s))^a (a / (m_w s) + 1 / phi)^-(a + S_w / phi).
        let grid: Vec<f64> = (0..20_000)
            .map(|i| (-3.0 + 4.0 * i as f64 / 19_999.0).exp())
            .collect();
        let log_post: Vec<f64> = grid
            .iter()
            .map(|&s| {
                let mut l = (k - 1.0) * s.ln() - k * s + s.ln(); // + ln s: the log grid's Jacobian
                for (t, mw) in totals.iter().zip(&m) {
                    let rate = a / (mw * s);
                    l += a * rate.ln() - (a + t / phi) * (rate + 1.0 / phi).ln();
                }
                l
            })
            .collect();
        let top = log_post.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weight: Vec<f64> = log_post.iter().map(|l| (l - top).exp()).collect();
        let total: f64 = weight.iter().sum();
        let s_mean: f64 = grid.iter().zip(&weight).map(|(s, w)| s * w).sum::<f64>() / total;
        // The new year's loss ratio cdf: Gamma(a, mean 0.5 s) mixed over s.
        let cdf = |x: f64| -> f64 {
            grid.iter()
                .zip(&weight)
                .map(|(&s, w)| w * gamma_inc(a, x * a / (0.5 * s)).0)
                .sum::<f64>()
                / total
        };
        // The exact sd: E[(0.5 s)^2] (1 + cv^2) - (0.5 E[s])^2.
        let s_second: f64 = grid
            .iter()
            .zip(&weight)
            .map(|(s, w)| s * s * w)
            .sum::<f64>()
            / total;
        let exact_sd = (0.25 * s_second * (1.0 + cv * cv) - (0.5 * s_mean).powi(2)).sqrt();
        let year = fit.new_year(2000.0, 100_000, 3).unwrap();
        let simulated_sd = year.loss_ratio.std_dev();
        println!("cv {cv}: sd {simulated_sd:.6} against exact {exact_sd:.6}");
        assert!(
            (simulated_sd / exact_sd - 1.0).abs() < 0.015,
            "{simulated_sd} {exact_sd}"
        );
        let draws = sorted(year.loss_ratio.draw_matrix());
        let mean = year.loss_ratio.mean();
        println!(
            "cv {cv}: new-year mean {mean:.5} against 0.5 E[s] {:.5}, phi {phi:.3}",
            0.5 * s_mean
        );
        assert!(
            (mean / (0.5 * s_mean) - 1.0).abs() < 0.005,
            "{mean} {s_mean}"
        );
        for p in [0.05, 0.25, 0.5, 0.75, 0.95] {
            let q = quantile(&draws, p);
            let f = cdf(q);
            println!("  p {p}: simulated quantile {q:.5}, exact cdf there {f:.4}");
            assert!((f - p).abs() < 0.01, "p {p}: cdf {f}");
        }
        if cv < 0.05 {
            // The gamma limit: s ~ Gamma(k + T / phi, rate k + M / phi).
            let t: f64 = totals.iter().sum();
            let mm: f64 = m.iter().sum();
            let (shape, rate) = (k + t / phi, k + mm / phi);
            let exact = 0.5 * shape / rate;
            println!("  gamma limit: 0.5 E[s] = {exact:.5}");
            assert!((mean / exact - 1.0).abs() < 0.005, "{mean} {exact}");
            // Its variance, with the new origin's own gamma (cv 1%) on top:
            // E[(0.5 s)^2] (1 + cv^2) - (0.5 E[s])^2.
            let second = 0.25 * shape * (shape + 1.0) / (rate * rate);
            let sd = (second * (1.0 + cv * cv) - exact * exact).sqrt();
            let simulated = year.loss_ratio.std_dev();
            println!("  gamma limit: sd {sd:.6}, simulated {simulated:.6}");
            // At cv 0.01 each origin's ultimate still moves by about 1%,
            // which widens the posterior of s by about 4% over the limit.
            assert!((simulated / sd - 1.0).abs() < 0.06, "{simulated} {sd}");
        }
    }
}

/// The tight-prior limit reproduces the prior: without the common
/// multiplier exactly (any data), and with a vanishing one (`common_cv`
/// 0.001) within Monte Carlo error; the cv is `cv` in both.
#[test]
fn a_tight_prior_reproduces_apriori_and_cv() {
    let tri = triangle_columns("genins_premium", &["paid", "premium"]);
    for common in [None, Some(0.001)] {
        let fit = BayesBfNormalised {
            apriori: 0.7,
            cv: 0.25,
            common_cv: common,
            process: ProcessDistribution::None,
            seed: 2,
            n_sims: 100,
            ..BayesBfNormalised::default()
        }
        .fit(&tri, "paid", "premium")
        .unwrap();
        let year = fit.new_year(1.2e7, 100_000, 9).unwrap();
        let (mean, cv) = (
            year.loss_ratio.mean(),
            year.loss_ratio.std_dev() / year.loss_ratio.mean(),
        );
        println!("common_cv {common:?}: mean {mean:.5}, cv {cv:.5}");
        assert!((mean / 0.7 - 1.0).abs() < 0.004, "{mean}");
        assert!((cv / 0.25 - 1.0).abs() < 0.015, "{cv}");
    }
}

/// Pooling: ten origins whose loss ratios differ only by the wobble (0.6
/// each), a wrong prior (`apriori` 0.9) held loosely in level (`common_cv`
/// 10) and at `cv` 0.2 across origins. The new year's mean is the origins'
/// posterior mean loss ratio, near the data's 0.6 rather than the prior's,
/// and its cv at least 0.2.
#[test]
fn identical_origins_pool_into_the_new_year() {
    let tri = book(8, |_| 0.6, |w| 8 - w);
    let fit = BayesBfNormalised {
        apriori: 0.9,
        cv: 0.2,
        common_cv: Some(10.0),
        process: ProcessDistribution::None,
        seed: 4,
        n_sims: 100,
        ..BayesBfNormalised::default()
    }
    .fit(&tri, "paid", "premium")
    .unwrap();
    assert!(
        fit.diagnostics.warnings.is_empty(),
        "{:?}",
        fit.diagnostics.warnings
    );
    let origins: Vec<f64> = fit
        .expected_ultimate()
        .iter()
        .zip(&fit.exposure)
        .map(|(u, e)| u / e)
        .collect();
    let pooled = origins.iter().sum::<f64>() / origins.len() as f64;
    let year = fit.new_year(1000.0, 100_000, 1).unwrap();
    let (mean, cv) = (
        year.loss_ratio.mean(),
        year.loss_ratio.std_dev() / year.loss_ratio.mean(),
    );
    println!("origins' posterior loss ratios {origins:.4?}, mean {pooled:.4}");
    println!("new year: mean {mean:.4}, cv {cv:.4}");
    assert!((mean / pooled - 1.0).abs() < 0.03, "{mean} {pooled}");
    assert!((mean - 0.6).abs() < 0.05, "{mean}");
    assert!(cv >= 0.2, "{cv}");
}

/// The rows of `data/clrd_comauto_353.csv` with origin, age in months,
/// paid, incurred and premium, at calendar years up to `last`.
fn comauto(last: i32) -> Triangle {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/clrd_comauto_353.csv");
    let text = std::fs::read_to_string(path).unwrap();
    let rows: Vec<(i32, Lag, f64, f64, f64)> = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (
                f[0].parse().unwrap(),
                f[1].parse().unwrap(),
                f[2].parse().unwrap(),
                f[3].parse().unwrap(),
                f[4].parse().unwrap(),
            )
        })
        .filter(|r: &(i32, Lag, f64, f64, f64)| r.0 <= last && r.0 + (r.1 / 12) as i32 - 1 <= last)
        .collect();
    let origin: Vec<Month> = rows.iter().map(|r| Month::january(r.0)).collect();
    let ages: Vec<Lag> = rows.iter().map(|r| r.1).collect();
    let col = |k: usize| {
        rows.iter()
            .map(|r| [r.2, r.3, r.4][k])
            .collect::<Vec<f64>>()
    };
    let (paid, incurred, premium) = (col(0), col(1), col(2));
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[
            ("paid", &paid),
            ("incurred", &incurred),
            ("premium", &premium),
        ],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// The mean and variance of the new year's log loss ratio that `fit`'s
/// posterior implies under `level`, averaged over the draws.
fn implied_log_moments(fit: &MeyersFit, level: NewOriginLevel, origin_sd: f64) -> (f64, f64) {
    let post = &fit.posterior;
    let names = post.names();
    let at = |name: &str| names.iter().position(|n| n == name);
    let n = fit.premium.len();
    let m = fit.chain_ladder.development.development.len();
    let sigma_m = at(&format!("sigma[{m}]")).unwrap();
    let rho = at("rho");
    let (mut first, mut second, mut within) = (0.0, 0.0, 0.0);
    let draws = post.n_draws();
    for d in 0..draws {
        let theta = post.draw(d);
        let logelr = theta[0];
        let levels: Vec<f64> = (1..=n)
            .map(|w| {
                logelr
                    + if w == 1 {
                        0.0
                    } else {
                        theta[at(&format!("alpha[{w}]")).unwrap()]
                    }
            })
            .collect();
        let (centre, spread) = match level {
            NewOriginLevel::Prior => (logelr, origin_sd * origin_sd),
            NewOriginLevel::Exchangeable => {
                let nf = n as f64;
                let mean = levels.iter().sum::<f64>() / nf;
                let var = levels.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / (nf - 1.0);
                // A Student t with n - 1 degrees of freedom has variance
                // (n - 1) / (n - 3) times its scale squared.
                (mean, var * (1.0 + 1.0 / nf) * (nf - 1.0) / (nf - 3.0))
            }
        };
        let s2 = theta[sigma_m].powi(2);
        // CAY: rho times the newest origin's residual, N(0, sigma_m^2) when
        // it is open.
        let r2 = rho.map_or(0.0, |i| theta[i].powi(2));
        first += centre;
        second += centre * centre;
        within += spread + s2 * (1.0 + r2);
    }
    let draws = draws as f64;
    let mean = first / draws;
    (mean, second / draws - mean * mean + within / draws)
}

/// Meyers' models on group 353: the new year's log loss ratio has the mean
/// and variance the posterior implies under each level rule, and Meyers'
/// own `origin_sd` (3.162) makes the prior rule's spread useless while the
/// exchangeable one follows the triangle's.
#[test]
fn meyers_new_year_has_the_implied_moments() {
    let tri = comauto(1997);
    for (kind, column) in [
        (MeyersModel::Crc, "paid"),
        (MeyersModel::Csr, "paid"),
        (MeyersModel::Cay, "incurred"),
    ] {
        let fit = Meyers {
            model: kind,
            n_sims: 100,
            seed: 1,
            ..Meyers::default()
        }
        .fit(&tri, column, "premium")
        .unwrap();
        for level in [NewOriginLevel::Exchangeable, NewOriginLevel::Prior] {
            let year = fit.new_year_with_level(5000.0, level, 200_000, 7).unwrap();
            let logs: Vec<f64> = year
                .loss_ratio
                .draw_matrix()
                .iter()
                .map(|x| x.ln())
                .collect();
            let n = logs.len() as f64;
            let mean = logs.iter().sum::<f64>() / n;
            let var = logs.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / n;
            let (m, v) = implied_log_moments(&fit, level, 3.162);
            println!(
                "{} {} {:?}: log LR mean {mean:.4} (implied {m:.4}), sd {:.4} (implied {:.4}), \
                 median LR {:.4}",
                kind.name(),
                column,
                level,
                var.sqrt(),
                v.sqrt(),
                quantile(&sorted(year.loss_ratio.draw_matrix()), 0.5)
            );
            assert!((mean - m).abs() < 0.01 + 0.01 * v.sqrt(), "{mean} {m}");
            assert!((var / v - 1.0).abs() < 0.03, "{var} {v}");
            match level {
                NewOriginLevel::Prior => assert!(var.sqrt() > 3.0),
                NewOriginLevel::Exchangeable => assert!(var.sqrt() < 0.5),
            }
        }
    }
}

/// A held-out year: CRC fitted to paid at calendar years up to 1996 (nine
/// origins), the 1997 origin's paid at 120 months (4,139) and its premium
/// (4,962) give a loss ratio that falls inside the new year's 1% to 99%
/// interval. One year is one draw from the predictive, so this guards
/// against gross errors only; the percentile is printed.
#[test]
fn a_held_out_year_falls_inside_its_new_year_interval() {
    let tri = comauto(1996);
    let fit = Meyers {
        n_sims: 100,
        seed: 3,
        ..Meyers::default()
    }
    .fit(&tri, "paid", "premium")
    .unwrap();
    let year = fit.new_year(4962.0, 100_000, 2).unwrap();
    assert_eq!(year.origin.to_string(), "1997");
    let draws = year.loss_ratio.draw_matrix();
    let actual = 4139.0 / 4962.0;
    let below = draws.iter().filter(|&&x| x <= actual).count() as f64 / draws.len() as f64;
    let s = sorted(draws);
    println!(
        "1997 paid loss ratio {actual:.4} at percentile {below:.3} of the new year \
         (median {:.4}, 5% {:.4}, 95% {:.4})",
        quantile(&s, 0.5),
        quantile(&s, 0.05),
        quantile(&s, 0.95)
    );
    assert!((0.01..=0.99).contains(&below), "{below}");
}
