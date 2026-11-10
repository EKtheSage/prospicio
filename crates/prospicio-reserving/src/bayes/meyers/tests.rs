use super::model::softplus;
use super::*;
use crate::bayes::laplace::tests::assert_gradient;
use crate::triangle::Long;
use crate::{DevelopmentColumn, Grain, Month};
use prospicio_bayes::nuts::LogDensity;
use prospicio_core::Period;
use prospicio_prob::Distribution;

/// Meyers' illustrative insurer, group 353 Commercial Auto paid
/// (Meyers 2019, Table 2.2; validation/data/clrd_comauto_353.csv), the
/// cells at calendar years up to 1997.
const PAID: [[f64; 10]; 10] = [
    [
        952., 1529., 2813., 3647., 3724., 3832., 3899., 3907., 3911., 3912.,
    ],
    [
        849., 1564., 2202., 2432., 2468., 2487., 2513., 2526., 2531., 2527.,
    ],
    [
        983., 2211., 2830., 3832., 4039., 4065., 4102., 4155., 4268., 4274.,
    ],
    [
        1657., 2685., 3169., 3600., 3900., 4320., 4332., 4338., 4341., 4341.,
    ],
    [
        932., 1940., 2626., 3332., 3368., 3491., 3531., 3540., 3540., 3583.,
    ],
    [
        1162., 2402., 2799., 2996., 3034., 3042., 3230., 3238., 3241., 3268.,
    ],
    [
        1478., 2980., 3945., 4714., 5462., 5680., 5682., 5683., 5684., 5684.,
    ],
    [
        1240., 2080., 2607., 3080., 3678., 4116., 4117., 4125., 4128., 4128.,
    ],
    [
        1326., 2412., 3367., 3843., 3965., 4127., 4133., 4141., 4142., 4144.,
    ],
    [
        1413., 2683., 3173., 3674., 3805., 4005., 4020., 4095., 4132., 4139.,
    ],
];
const PREMIUM: [f64; 10] = [
    5812., 4908., 5454., 5165., 5214., 5230., 4992., 5466., 5226., 4962.,
];

/// The training triangle of `rows` (the first `10 - w` cells of row
/// `w`), columns `paid` and `premium`, origins from 1988.
fn triangle(rows: &[[f64; 10]], premium: &[f64]) -> Triangle {
    let n = rows.len();
    let (mut origin, mut ages, mut paid, mut prem) = (vec![], vec![], vec![], vec![]);
    for (w, row) in rows.iter().enumerate() {
        for (d, &v) in row.iter().enumerate().take(n - w) {
            origin.push(Month::january(1988 + w as i32));
            ages.push(12 * (d as crate::Lag + 1));
            paid.push(v);
            prem.push(premium[w]);
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("paid", &paid), ("premium", &prem)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

fn comauto() -> Triangle {
    triangle(&PAID, &PREMIUM)
}

fn quick(model: MeyersModel, seed: u64) -> Meyers {
    Meyers {
        model,
        sampler: Sampler {
            tune: 300,
            draws: 300,
            ..Meyers::default().sampler
        },
        n_sims: 2_000,
        seed,
        ..Meyers::default()
    }
}

fn prepared(model: MeyersModel) -> Model {
    let tri = comauto();
    let segment = tri.segment("paid").unwrap();
    let cl = ChainLadder::default()
        .fit_segment(&segment, &segment.ages)
        .unwrap();
    let premium = origin_exposure(&segment, &tri.segment("premium").unwrap(), "premium").unwrap();
    Meyers {
        model,
        ..Meyers::default()
    }
    .model_of(&segment, &cl, &premium)
    .unwrap()
}

/// A point off the mode with `gamma` or `rho` away from 0.
fn off_mode(model: &Model) -> Vec<f64> {
    let mut x = model.start();
    for (j, v) in x.iter_mut().enumerate() {
        *v += 0.01 * (j as f64 % 7.0 - 3.0);
    }
    match model.kind {
        MeyersModel::Crc => {}
        MeyersModel::Csr => x[model.after_beta()] = 0.04,
        MeyersModel::Cay => x[model.after_beta()] = 0.6,
    }
    x
}

#[test]
fn the_gradient_matches_a_finite_difference() {
    for kind in [MeyersModel::Crc, MeyersModel::Csr, MeyersModel::Cay] {
        let model = prepared(kind);
        assert_eq!(model.dim(), if kind == MeyersModel::Crc { 29 } else { 30 });
        assert_eq!(model.cells.len(), 55);
        let x = off_mode(&model);
        assert_gradient(|t, g| model.log_posterior(t, g), &x);
        let whitened = Whitened::regularised(&model, &model.start()).unwrap();
        let z: Vec<f64> = (0..model.dim())
            .map(|j| 0.3 * ((j % 3) as f64 - 1.0))
            .collect();
        assert_gradient(|z, g| whitened.log_density(z, g).unwrap(), &z);
    }
}

#[test]
fn the_generic_density_matches_the_hand_written_one() {
    use crate::bayes::autodiff::{assert_oracle, posterior_points};
    for kind in [MeyersModel::Crc, MeyersModel::Csr, MeyersModel::Cay] {
        let model = prepared(kind);
        // The pilot run's scale at the start: the posterior's bulk,
        // not the joint mode deep in the funnel.
        let whitened = Whitened::regularised(&model, &model.start()).unwrap();
        let points = posterior_points(&whitened, model.dim(), 20);
        assert_oracle(&format!("Meyers {}", kind.name()), &model, &points);
    }
}

/// The benchmark's row for Meyers' CRC on group 353's paid column
/// (`validation/data/clrd_comauto_353.csv`, the training cells in
/// [`PAID`]): 29 parameters, 55 cells, at the start point.
pub(crate) fn bench_crc_353() {
    let model = prepared(MeyersModel::Crc);
    let theta = model.start();
    crate::bayes::autodiff::bench_row::<_, 29>(
        "Meyers CRC, group 353 paid",
        &model,
        &theta,
        Some(&|t: &[f64], g: &mut [f64]| model.log_posterior(t, g)),
    );
}

#[test]
fn csr_at_gamma_zero_is_crc_plus_the_gamma_prior() {
    let (crc, csr) = (prepared(MeyersModel::Crc), prepared(MeyersModel::Csr));
    let theta = off_mode(&crc);
    // The same point with gamma = 0 inserted before u.
    let mut with_gamma = theta[..crc.after_beta()].to_vec();
    with_gamma.push(0.0);
    with_gamma.extend(&theta[crc.after_beta()..]);
    let (mut g1, mut g2) = (vec![0.0; crc.dim()], vec![0.0; csr.dim()]);
    let a = crc.log_posterior(&theta, &mut g1);
    let b = csr.log_posterior(&with_gamma, &mut g2);
    // The gamma prior at 0 is 0 (up to its constant, left out).
    assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    let mut g2_without = g2.clone();
    g2_without.remove(csr.after_beta());
    for (x, y) in g1.iter().zip(&g2_without) {
        assert!((x - y).abs() < 1e-12);
    }
}

#[test]
fn cay_at_rho_zero_is_crc_plus_the_rho_prior() {
    let (crc, cay) = (prepared(MeyersModel::Crc), prepared(MeyersModel::Cay));
    let theta = off_mode(&crc);
    let k = crc.after_beta();
    let mut with_v = theta[..k].to_vec();
    with_v.push(0.0);
    with_v.extend(&theta[k..]);
    let (mut g1, mut g2) = (vec![0.0; crc.dim()], vec![0.0; cay.dim()]);
    let a = crc.log_posterior(&theta, &mut g1);
    let b = cay.log_posterior(&with_v, &mut g2);
    // The Beta(2, 2) prior with its Jacobian at v = 0: 2 log(1/4).
    assert!((b - a - 2.0 * 0.25_f64.ln()).abs() < 1e-10, "{a} vs {b}");
    let mut g2_without = g2.clone();
    g2_without.remove(k);
    for (x, y) in g1.iter().zip(&g2_without) {
        assert!((x - y).abs() < 1e-10, "{x} vs {y}");
    }
    // At rho = 0 the rho gradient is the sum over cells after the first
    // origin of (r_i / v_i) r_(w-1)d: not zero on these data.
    assert!(g2[k].abs() > 1e-3, "{}", g2[k]);
}

#[test]
fn cay_follows_meyers_recursion() {
    // Meyers' CAY.R: cells sorted by (age, origin), and
    // mu[i] = base[i] + rho (y[i - 1] - mu[i - 1]) for origins after
    // the first; here recomputed that way from a natural-scale point.
    let model = prepared(MeyersModel::Cay);
    let nat = model.natural(&off_mode(&model));
    let rho = nat[model.after_beta()];
    assert!((rho - 0.6_f64.tanh()).abs() < 1e-15);
    let mut order: Vec<usize> = (0..model.cells.len()).collect();
    order.sort_by_key(|&i| (model.cells[i].d, model.cells[i].w));
    let mut expected = vec![0.0; model.cells.len()];
    let (mut last_y, mut last_mu) = (0.0, 0.0);
    for &i in &order {
        let c = &model.cells[i];
        let mut mu = model.mu(&nat, c);
        if c.w > 0 {
            mu += rho * (last_y - last_mu);
        }
        let sigma = nat[model.u0() + c.d];
        expected[i] =
            -0.5 * ((c.y - mu) / sigma).powi(2) - sigma.ln() - 0.5 * std::f64::consts::TAU.ln();
        (last_y, last_mu) = (c.y, mu);
    }
    let mut got = Vec::new();
    model.log_densities(&nat, &mut Vec::new(), &mut got);
    for (g, e) in got.iter().zip(&expected) {
        assert!((g - e).abs() < 1e-12, "{g} vs {e}");
    }
    // The log posterior without priors is their sum less the
    // constant, plus the log Jacobian of each u_i.
    let unpriored = Model {
        priors: false,
        ..model.clone()
    };
    let theta = off_mode(&model);
    let mut grad = vec![0.0; model.dim()];
    let f = unpriored.log_posterior(&theta, &mut grad);
    let jacobian: f64 = theta[model.u0()..]
        .iter()
        .map(|&u| -(softplus(-u) + softplus(u)))
        .sum();
    let n = model.cells.len() as f64;
    let sum: f64 = expected.iter().sum();
    assert!(
        (f - (sum + 0.5 * n * std::f64::consts::TAU.ln() + jacobian)).abs() < 1e-9,
        "{f} vs {sum}"
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn cay_draws_the_oldest_age_as_a_moving_average() {
    // Given one natural-scale draw with a large sigma_m and rho, the
    // log ultimates down age m follow Meyers' recursion: origin 2's
    // mean takes rho times origin 1's observed residual, and the open
    // origins after it are a moving average of order 1, with lag-one
    // correlation rho / (1 + rho^2) and none further (Meyers 2019,
    // eq. 8.3). On group 353 sigma_m is near 0.013, so this barely
    // moves the predictive there; this test is what checks it.
    let model = prepared(MeyersModel::Cay);
    let (n, k) = (model.n_origins, model.after_beta());
    let (rho, sigma) = (0.8, 0.5);
    let mut nat = vec![0.0; model.dim()];
    nat[0] = -0.4;
    for (w, alpha) in nat.iter_mut().enumerate().take(n).skip(1) {
        *alpha = 0.05 * w as f64;
    }
    nat[k] = rho;
    nat[model.dim() - 1] = sigma;
    let premiums = PREMIUM;
    let mut closed = vec![false; n];
    closed[0] = true;
    let mut latest = vec![0.0_f64; n];
    latest[0] = 3912.0;
    let base = |w: usize| premiums[w].ln() + nat[0] + if w > 0 { nat[w] } else { 0.0 };
    let r1 = latest[0].ln() - base(0);
    let sims = 200_000;
    let mut rng = StreamRng::new(7, 0);
    let mut row = vec![0.0; n];
    let mut logs = vec![Vec::with_capacity(sims); n];
    for _ in 0..sims {
        model.draw_ultimates(&nat, &premiums, &closed, &latest, &mut rng, &mut row);
        assert_eq!(row[0], latest[0]);
        for (w, u) in row.iter().enumerate() {
            logs[w].push(u.ln());
        }
    }
    let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
    let cov = |x: &[f64], y: &[f64]| {
        let (mx, my) = (mean(x), mean(y));
        x.iter()
            .zip(y)
            .map(|(a, b)| (a - mx) * (b - my))
            .sum::<f64>()
            / (x.len() as f64 - 1.0)
    };
    let se = sigma / (sims as f64).sqrt();
    assert!((mean(&logs[1]) - (base(1) + rho * r1)).abs() < 5.0 * se);
    assert!((cov(&logs[1], &logs[1]).sqrt() / sigma - 1.0).abs() < 0.01);
    for (w, log) in logs.iter().enumerate().skip(2) {
        assert!((mean(log) - base(w)).abs() < 5.0 * se * 1.3);
        let sd = cov(log, log).sqrt();
        assert!((sd / (sigma * (1.0 + rho * rho).sqrt()) - 1.0).abs() < 0.01);
    }
    let corr = |a: usize, b: usize| {
        cov(&logs[a], &logs[b]) / (cov(&logs[a], &logs[a]) * cov(&logs[b], &logs[b])).sqrt()
    };
    let ma1 = rho / (1.0 + rho * rho);
    for w in 2..n - 1 {
        assert!((corr(w, w + 1) - ma1).abs() < 0.01, "{}", corr(w, w + 1));
    }
    assert!(corr(3, 5).abs() < 0.01);
    // Origin 2 has no earlier random term: its correlation with origin
    // 3 is rho / sqrt(1 + rho^2).
    assert!((corr(1, 2) - rho / (1.0 + rho * rho).sqrt()).abs() < 0.01);
    // CRC draws the same origins independently.
    let crc = prepared(MeyersModel::Crc);
    let mut crc_nat = nat.clone();
    crc_nat.remove(k);
    let mut a = Vec::with_capacity(sims);
    let mut b = Vec::with_capacity(sims);
    for _ in 0..sims {
        crc.draw_ultimates(&crc_nat, &premiums, &closed, &latest, &mut rng, &mut row);
        a.push(row[3].ln());
        b.push(row[4].ln());
    }
    assert!((cov(&a, &b) / (cov(&a, &a) * cov(&b, &b)).sqrt()).abs() < 0.01);
}

#[test]
fn the_rho_prior_is_meyers_beta() {
    // With no cells and only the rho prior, the density of v is
    // Beta(a, a) of r = (1 - tanh v) / 2 times |dr/dv| = 2 r (1 - r),
    // up to its constant, so differences from v = 0 match exactly.
    for a in [2.0, 0.7, 5.0] {
        let model = Model {
            cells: Vec::new(),
            rho_shape: a,
            ..prepared(MeyersModel::Cay)
        };
        let k = model.after_beta();
        let mut theta = vec![0.0; model.dim()];
        let mut grad = vec![0.0; model.dim()];
        let base = model.log_posterior(&theta, &mut grad);
        for v in [-3.0, -0.4, 0.9, 2.5_f64] {
            theta[k] = v;
            let f = model.log_posterior(&theta, &mut grad) - base;
            let r = (1.0 - v.tanh()) / 2.0;
            let density = |r: f64| (a - 1.0) * (r * (1.0 - r)).ln() + (2.0 * r * (1.0 - r)).ln();
            let expected = density(r) - density(0.5);
            assert!(
                (f - expected).abs() < 1e-10,
                "a {a} v {v}: {f} vs {expected}"
            );
        }
    }
}

#[test]
fn the_logit_gives_meyers_uniform_prior_on_s() {
    // Meyers' floor: a_ig < 1e5, s = 1 - exp(-1 / a_ig).
    assert!((S_FLOOR + (-1e-5_f64).exp_m1()).abs() < 1e-20);
    // With no cells and no priors, the density in u is
    // log l + log(1 - l) for l = logistic(u): the U(S_FLOOR, 1) density
    // of s = S_FLOOR + (1 - S_FLOOR) l times |ds/du|.
    let model = Model {
        cells: Vec::new(),
        priors: false,
        ..prepared(MeyersModel::Crc)
    };
    let mut theta = vec![0.0; model.dim()];
    let mut grad = vec![0.0; model.dim()];
    for u in [-6.0, -1.0, 0.0, 0.5, 3.0] {
        theta[model.u0()..].fill(u);
        let f = model.log_posterior(&theta, &mut grad);
        let s = logistic(u);
        let expected = model.n_dev as f64 * (s * (1.0 - s)).ln();
        assert!((f - expected).abs() < 1e-12, "u {u}: {f} vs {expected}");
    }
    // Far down, every step is the floor.
    theta[model.u0()..].fill(-60.0);
    let (var, _, _) = model.variances(&theta);
    assert!((var[model.n_dev - 1] / S_FLOOR - 1.0).abs() < 1e-12);
    assert!((var[0] / (10.0 * S_FLOOR) - 1.0).abs() < 1e-12);
    // Its integral over u is 1 (trapezoid on a wide grid).
    let h = 1e-3;
    let total: f64 = (-40_000..=40_000)
        .map(|k| {
            let s = logistic(k as f64 * h);
            s * (1.0 - s) * h
        })
        .sum();
    assert!((total - 1.0).abs() < 1e-9, "{total}");
}

#[test]
fn the_start_is_least_squares_with_falling_sigmas() {
    for kind in [MeyersModel::Crc, MeyersModel::Csr, MeyersModel::Cay] {
        let model = prepared(kind);
        let start = model.start();
        // The location part is least squares: its gradient without
        // priors, at equal sigmas, is zero.
        let ols = Model {
            priors: false,
            ..model.clone()
        };
        // All sigma_d^2 equal (to 0.5, up to 1e-22): only s_m is not
        // negligible.
        let mut theta = start.clone();
        theta[model.u0()..].fill(-50.0);
        theta[model.dim() - 1] = 0.0;
        let mut grad = vec![0.0; model.dim()];
        ols.log_posterior(&theta, &mut grad);
        let worst = grad[..model.after_beta()]
            .iter()
            .fold(0.0_f64, |m, g| m.max(g.abs()));
        // Up to the floor's 1e-4 relative difference between the sigmas.
        assert!(worst < 1e-3, "{kind:?}: gradient {worst}");
        if kind != MeyersModel::Crc {
            assert_eq!(start[model.after_beta()], 0.0);
        }
        let sigma = &model.natural(&start)[model.u0()..];
        assert!(sigma.windows(2).all(|p| p[0] > p[1]), "{sigma:?}");
        assert!(sigma.iter().all(|&s| s > 0.0 && s < 1.0), "{sigma:?}");
    }
}

#[test]
fn the_joint_mode_is_deep_in_the_funnel() {
    // Why Meyers samples on a pilot scale and not at the mode as
    // BayesOdp does: the joint mode puts sigma[10] far below its
    // posterior mean (0.020 in Stan's reference), where a Laplace scale
    // says little about the bulk.
    let model = prepared(MeyersModel::Crc);
    let theta = match crate::bayes::laplace::mode(&model, model.start()) {
        Ok(theta) => theta,
        // Newton creeps down the funnel and may not converge.
        Err(Error::Bayes(_)) => return,
        Err(e) => panic!("{e}"),
    };
    let sigma = model.natural(&theta)[model.dim() - 1];
    assert!(sigma < 0.005, "sigma[10] at the mode {sigma}");
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn a_seed_replays_whatever_the_threads() {
    let tri = comauto();
    let a = quick(MeyersModel::Csr, 3)
        .fit(&tri, "paid", "premium")
        .unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let b = pool.install(|| {
        quick(MeyersModel::Csr, 3)
            .fit(&tri, "paid", "premium")
            .unwrap()
    });
    assert_eq!(a.posterior, b.posterior);
    assert_eq!(a.reserves.draw_matrix(), b.reserves.draw_matrix());
    let c = quick(MeyersModel::Csr, 4)
        .fit(&tri, "paid", "premium")
        .unwrap();
    assert_ne!(a.posterior.values(), c.posterior.values());
    assert_ne!(a.reserves.draw_matrix(), c.reserves.draw_matrix());
    // The sampler's own seed is not used.
    let mut d = quick(MeyersModel::Csr, 3);
    d.sampler.seed = 99;
    assert_eq!(
        d.fit(&tri, "paid", "premium").unwrap().posterior,
        a.posterior
    );
    // CAY's sequential prediction replays too.
    let e = quick(MeyersModel::Cay, 3)
        .fit(&tri, "paid", "premium")
        .unwrap();
    let f = pool.install(|| {
        quick(MeyersModel::Cay, 3)
            .fit(&tri, "paid", "premium")
            .unwrap()
    });
    assert_eq!(e.posterior, f.posterior);
    assert_eq!(e.reserves.draw_matrix(), f.reserves.draw_matrix());
}

#[test]
fn a_short_run_returns_the_fit_with_warnings() {
    let model = Meyers {
        sampler: Sampler {
            tune: 50,
            draws: 20,
            ..Meyers::default().sampler
        },
        n_sims: 100,
        ..Meyers::default()
    };
    let fit = model.fit(&comauto(), "paid", "premium").unwrap();
    assert!(fit.diagnostics.min_ess_bulk < 400.0);
    assert!(
        fit.diagnostics
            .warnings
            .iter()
            .any(|w| w.contains("bulk ESS")),
        "{:?}",
        fit.diagnostics.warnings
    );
}

#[test]
fn the_pattern_is_the_next_origins_development_effect() {
    for kind in [MeyersModel::Crc, MeyersModel::Csr, MeyersModel::Cay] {
        let model = prepared(kind);
        let nat = model.natural(&off_mode(&model));
        let m = model.n_dev;
        let mut shares = vec![0.0; m];
        model.pattern(&nat, &mut shares);
        assert_eq!(shares[m - 1], 1.0, "{kind:?}");
        // CSR scales each beta by the speed-up of the origin after the
        // newest, (1 - gamma)^10 here; CRC and CAY (whose rho is off 0)
        // take beta alone.
        let g = match kind {
            MeyersModel::Csr => {
                let gamma = nat[model.after_beta()];
                assert!(gamma.abs() > 1e-3);
                (1.0 - gamma).powi(10)
            }
            _ => 1.0,
        };
        for (d, share) in shares[..m - 1].iter().enumerate() {
            let beta = nat[model.beta0() + d];
            assert!((share - (beta * g).exp()).abs() < 1e-15, "{kind:?} {d}");
        }
        // The share times the median value at the oldest age is the
        // median value at age d of that origin, whatever its premium.
        let premiums = [PREMIUM.as_slice(), &[6000.0]].concat();
        let oldest = model.mu_at(&nat, &premiums, 10, m - 1).exp();
        for (d, share) in shares.iter().enumerate() {
            let at_d = model.mu_at(&nat, &premiums, 10, d).exp();
            assert!((share * oldest / at_d - 1.0).abs() < 1e-12, "{kind:?} {d}");
        }
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn each_simulations_pattern_is_its_posterior_draws() {
    let tri = comauto();
    let segment = tri.segment("paid").unwrap();
    // CAY's pattern is CRC's function of the draw (the cheap test above).
    for kind in [MeyersModel::Crc, MeyersModel::Csr] {
        let meyers = quick(kind, 21);
        let fit = meyers.fit(&tri, "paid", "premium").unwrap();
        let patterns = fit.patterns.segment(&[]).unwrap();
        assert_eq!(fit.patterns.n_sims(), 2_000);
        assert_eq!(patterns.ages(), segment.ages);
        // The ultimates and patterns of one predictive run, the pattern
        // computed from the draw each simulation took: the fit's
        // pattern row i is simulation i's.
        let latest = &fit.chain_ladder.latest;
        let closed: Vec<bool> = fit
            .chain_ladder
            .latest_position
            .iter()
            .map(|&d| d + 1 == 10)
            .collect();
        let replay = fit
            .posterior
            .predictive(
                vec!["column".into()],
                (0..20).map(|j| vec![KeyValue::Int(j)]).collect(),
                meyers.n_sims,
                derive_seed(meyers.seed, Stream::Predictive),
                Provenance::new("replay"),
                |nat, rng: &mut StreamRng, row| {
                    let (ultimates, shares) = row.split_at_mut(10);
                    fit.model
                        .draw_ultimates(nat, &fit.premium, &closed, latest, rng, ultimates);
                    fit.model.pattern(nat, shares);
                },
            )
            .unwrap();
        for (i, row) in replay.draw_matrix().as_chunks::<20>().0.iter().enumerate() {
            let ultimates = &fit.ultimates.draw_matrix()[i * 10..(i + 1) * 10];
            assert_eq!(&row[..10], ultimates, "{kind:?} simulation {i}");
            assert_eq!(&row[10..], patterns.draw(i), "{kind:?} simulation {i}");
        }
        // The same predictive run without the pattern columns, as the
        // fit drew it before patterns: the fit's ultimates are bit for
        // bit these, so writing the patterns moved no random number.
        let without = fit
            .posterior
            .predictive(
                vec!["origin".into()],
                (0..10).map(|j| vec![KeyValue::Int(j)]).collect(),
                meyers.n_sims,
                derive_seed(meyers.seed, Stream::Predictive),
                Provenance::new("replay without patterns"),
                |nat, rng: &mut StreamRng, row| {
                    fit.model
                        .draw_ultimates(nat, &fit.premium, &closed, latest, rng, row);
                },
            )
            .unwrap();
        assert_eq!(
            without.draw_matrix(),
            fit.ultimates.draw_matrix(),
            "{kind:?}: the ultimates moved when the patterns were added"
        );
        // Every share is positive and the pattern ends at exactly 1, but
        // nothing in the model keeps beta_d rising or below 0: a share
        // falls in nearly every simulation, mostly from 60 months on.
        let mean = patterns.mean();
        assert_eq!(mean[9], 1.0);
        let falling = (0..2_000)
            .filter(|&i| patterns.draw(i).windows(2).any(|w| w[1] < w[0]))
            .count();
        // The number of simulations whose share falls into each age.
        let mut by_age = [0usize; 10];
        for i in 0..2_000 {
            for (k, w) in patterns.draw(i).windows(2).enumerate() {
                by_age[k + 1] += usize::from(w[1] < w[0]);
            }
        }
        eprintln!("{kind:?}: mean {mean:?}, falling in {falling} of 2000, by age {by_age:?}");
        for i in 0..2_000 {
            assert!(patterns.draw(i).iter().all(|&x| x > 0.0 && x.is_finite()));
        }
        // So the cash-flow engine's monthly patterns refuse them by
        // default, and hold each share at the largest before it on
        // request.
        assert!(falling > 0);
        let basis = crate::MonthlyBasis::default();
        let refused = fit
            .patterns
            .monthly(&[], &basis, 120, crate::Monotone::Refuse)
            .unwrap_err();
        assert!(refused.to_string().contains("falls"), "{refused}");
        let held = fit
            .patterns
            .monthly(&[], &basis, 120, crate::Monotone::RunningMax)
            .unwrap();
        assert_eq!(held.len(), 2_000);
        for row in &held {
            assert!(row.windows(2).all(|w| w[0] <= w[1]));
            assert_eq!(row[119], 1.0);
        }
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn ultimates_and_reserves_by_origin() {
    let fit = quick(MeyersModel::Crc, 5)
        .fit(&comauto(), "paid", "premium")
        .unwrap();
    for pd in [&fit.ultimates, &fit.reserves] {
        assert_eq!(pd.dims(), ["origin"]);
        assert_eq!(pd.n_components(), 10);
        assert_eq!(pd.n_sims(), 2_000);
    }
    let oldest = vec![Period::year(1988).into()];
    let reserve = fit.reserves.marginal(&oldest).unwrap();
    assert_eq!((reserve.mean(), reserve.std_dev()), (0.0, 0.0));
    assert_eq!(fit.ultimates.marginal(&oldest).unwrap().mean(), 3912.0);
    // Reserves are the ultimates less the latest values, draw by draw.
    for (row_u, row_r) in fit
        .ultimates
        .draw_matrix()
        .as_chunks::<10>()
        .0
        .iter()
        .zip(fit.reserves.draw_matrix().as_chunks::<10>().0)
    {
        for ((u, r), l) in row_u.iter().zip(row_r).zip(&fit.chain_ladder.latest) {
            assert_eq!(*r, u - l);
        }
    }
    let names = fit.posterior.names();
    assert_eq!(names.len(), 29);
    assert_eq!(names[0], "logelr");
    assert_eq!(names[9], "alpha[10]");
    assert_eq!(names[10], "beta[1]");
    assert_eq!(names[19], "sigma[1]");
    assert_eq!(names[28], "sigma[10]");
    assert_eq!(fit.premium, PREMIUM);
    let long = fit.to_long();
    assert_eq!(long.n_rows(), 10);
    let ulr: Vec<f64> = (long.column("ultimate").unwrap().iter())
        .zip(&PREMIUM)
        .map(|(u, p)| u / p)
        .collect();
    assert_eq!(long.column("ulr").unwrap(), ulr);
    for column in [
        "premium",
        "mean",
        "std_dev",
        "ultimate_mean",
        "ultimate_std_dev",
    ] {
        assert!(long.column(column).is_some(), "{column}");
    }
    // The posterior mean ultimate over the premium, per origin and in
    // total.
    let ulr_mean: Vec<f64> = (long.column("ultimate_mean").unwrap().iter())
        .zip(&PREMIUM)
        .map(|(u, p)| u / p)
        .collect();
    assert_eq!(long.column("ulr_mean").unwrap(), ulr_mean);
    assert_eq!(long.values.last().unwrap().0, "ulr_mean");
    let totals = fit.totals();
    assert_eq!(
        totals.column("ulr_mean").unwrap(),
        [totals.column("ultimate_mean").unwrap()[0] / PREMIUM.iter().sum::<f64>()]
    );
    assert_eq!(
        totals.column("premium").unwrap(),
        [PREMIUM.iter().sum::<f64>()]
    );
    assert!((totals.column("mean").unwrap()[0] - fit.reserves.total().mean()).abs() < 1e-6);
    assert!(
        (totals.column("ultimate_mean").unwrap()[0] - fit.ultimates.total().mean()).abs() < 1e-6
    );
    assert_eq!(fit.development_table().n_rows(), 10);
    assert_eq!(fit.summary().unwrap().len(), 29);
    let loo = fit.loo().unwrap();
    assert_eq!(loo.pareto_k.len(), 55);
    assert!(loo.estimate.elpd.is_finite());
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn provenance_records_the_settings() {
    let fit = quick(MeyersModel::Csr, 5)
        .fit(&comauto(), "paid", "premium")
        .unwrap();
    assert_eq!(fit.posterior.names()[19], "gamma");
    assert_eq!(fit.model(), MeyersModel::Csr);
    for (pd, output) in [(&fit.ultimates, "ultimates"), (&fit.reserves, "reserves")] {
        let provenance = pd.provenance();
        assert_eq!(provenance.model, "meyers_csr");
        let keys: Vec<&str> = provenance
            .parameters
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        for key in [
            "logelr_mean",
            "logelr_sd",
            "origin_sd",
            "development_sd",
            "gamma_sd",
            "chains",
            "tune",
            "draws",
            "target_accept",
            "max_depth",
            "n_sims",
            "seed",
            "column",
            "premium",
            "output",
            "posterior_draws",
        ] {
            assert!(keys.contains(&key), "{key} missing from {keys:?}");
        }
        assert!(
            provenance
                .parameters
                .iter()
                .any(|(k, v)| k == "output" && v == output)
        );
    }
    let crc = quick(MeyersModel::Crc, 5)
        .fit(&comauto(), "paid", "premium")
        .unwrap();
    assert_eq!(crc.reserves.provenance().model, "meyers_crc");
    let cay = quick(MeyersModel::Cay, 5)
        .fit(&comauto(), "paid", "premium")
        .unwrap();
    assert_eq!(cay.model(), MeyersModel::Cay);
    assert_eq!(cay.posterior.names()[19], "rho");
    let rho = cay.posterior.parameter(19).concat();
    assert!(rho.iter().all(|r| r.abs() < 1.0));
    let provenance = cay.reserves.provenance();
    assert_eq!(provenance.model, "meyers_cay");
    assert!(
        provenance
            .parameters
            .iter()
            .any(|(k, v)| k == "rho_shape" && v == "2")
    );
    assert!(
        !provenance.parameters.iter().any(|(k, _)| k == "gamma_sd"),
        "{:?}",
        provenance.parameters
    );
    // The oldest origin keeps its value; the CAY chain starts from its
    // observed residual.
    let oldest = vec![Period::year(1988).into()];
    assert_eq!(cay.ultimates.marginal(&oldest).unwrap().mean(), 3912.0);
    assert_eq!(cay.loo().unwrap().pareto_k.len(), 55);
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn errors() {
    let tri = comauto();
    for (name, model) in [
        (
            "logelr_sd",
            Meyers {
                logelr_sd: 0.0,
                ..Meyers::default()
            },
        ),
        (
            "origin_sd",
            Meyers {
                origin_sd: f64::NAN,
                ..Meyers::default()
            },
        ),
        (
            "development_sd",
            Meyers {
                development_sd: -1.0,
                ..Meyers::default()
            },
        ),
        (
            "gamma_sd",
            Meyers {
                gamma_sd: f64::INFINITY,
                ..Meyers::default()
            },
        ),
        (
            "logelr_mean",
            Meyers {
                logelr_mean: f64::NAN,
                ..Meyers::default()
            },
        ),
        (
            "n_sims",
            Meyers {
                n_sims: 0,
                ..Meyers::default()
            },
        ),
        (
            "rho_shape",
            Meyers {
                model: MeyersModel::Cay,
                rho_shape: 0.0,
                ..Meyers::default()
            },
        ),
    ] {
        let e = model.fit(&tri, "paid", "premium").unwrap_err();
        assert!(
            matches!(e, Error::InvalidSetting { name: n, .. } if n == name),
            "{name}: {e}"
        );
    }
    assert!(matches!(
        Meyers::default().fit(&tri, "incurred", "premium"),
        Err(Error::UnknownColumn(_))
    ));

    // A non-positive cell: 1990 at 24 months.
    let mut rows = PAID;
    rows[2][1] = 0.0;
    let e = quick(MeyersModel::Crc, 0)
        .fit(&triangle(&rows, &PREMIUM), "paid", "premium")
        .unwrap_err();
    assert!(
        matches!(&e, Error::Bayes(m) if m.starts_with("origin 1990 at development age 24 months is 0")),
        "{e}"
    );
    // A missing or non-positive premium.
    let mut premium = PREMIUM;
    premium[4] = 0.0;
    let e = quick(MeyersModel::Crc, 0)
        .fit(&triangle(&PAID, &premium), "paid", "premium")
        .unwrap_err();
    assert!(
        matches!(&e, Error::InvalidExposure { origin, .. } if origin == "1992"),
        "{e}"
    );
    // Too few cells: 3 origins, 6 cells, 3 + 2 + 3 = 8 parameters.
    let small = triangle(&PAID[..3], &PREMIUM[..3]);
    let e = quick(MeyersModel::Crc, 0)
        .fit(&small, "paid", "premium")
        .unwrap_err();
    assert!(
        matches!(&e, Error::Bayes(m) if m.starts_with("6 observed cells for 8 parameters")),
        "{e}"
    );
    // CAY without the previous origin's cell: 1992 stops at 36 months,
    // 1993 runs to 60.
    let (mut origin, mut ages, mut paid, mut prem) = (vec![], vec![], vec![], vec![]);
    for (w, row) in PAID.iter().enumerate() {
        let cells = if w == 4 { 3 } else { 10 - w };
        for (d, &v) in row.iter().enumerate().take(cells) {
            origin.push(Month::january(1988 + w as i32));
            ages.push(12 * (d as crate::Lag + 1));
            paid.push(v);
            prem.push(PREMIUM[w]);
        }
    }
    let ragged = Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("paid", &paid), ("premium", &prem)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    let e = quick(MeyersModel::Cay, 0)
        .fit(&ragged, "paid", "premium")
        .unwrap_err();
    assert!(
        matches!(&e, Error::Bayes(m) if m.starts_with(
            "origin 1993 is observed at development age 48 months but origin 1992 is not"
        )),
        "{e}"
    );
    // CRC takes the same triangle.
    quick(MeyersModel::Crc, 0)
        .fit(&ragged, "paid", "premium")
        .unwrap();

    let origin = [2020, 2020, 2021].map(Month::january);
    let two = Triangle::from_long(&Long {
        keys: &[("lob", &["a", "a", "a", "b", "b", "b"])],
        origin: &[origin, origin].concat(),
        development: DevelopmentColumn::Age(&[12, 24, 12, 12, 24, 12]),
        values: &[
            ("paid", &[1.0, 2.0, 1.0, 1.0, 2.0, 1.0]),
            ("premium", &[3.0; 6]),
        ],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    assert!(matches!(
        quick(MeyersModel::Crc, 0).fit(&two, "paid", "premium"),
        Err(Error::MultipleSegments(2))
    ));
}

/// Group 353's paid triangle twice, as lines `a` and `b` of a `lob`
/// key, with its premium.
fn twins() -> Triangle {
    let (mut keys, mut origin, mut ages, mut paid, mut prem) =
        (vec![], vec![], vec![], vec![], vec![]);
    for line in ["a", "b"] {
        for (w, row) in PAID.iter().enumerate() {
            for (d, &v) in row.iter().enumerate().take(10 - w) {
                keys.push(line);
                origin.push(Month::january(1988 + w as i32));
                ages.push(12 * (d as crate::Lag + 1));
                paid.push(v);
                prem.push(PREMIUM[w]);
            }
        }
    }
    Triangle::from_long(&Long {
        keys: &[("lob", &keys)],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("paid", &paid), ("premium", &prem)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// Each line's total of `pd`, simulation by simulation.
fn line_totals(pd: &PredictiveDistribution) -> Vec<Vec<f64>> {
    let by = pd.aggregate(&["lob"]).unwrap();
    let n = by.n_components();
    (0..n)
        .map(|j| by.draw_matrix().chunks(n).map(|r| r[j]).collect())
        .collect()
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let ranks = |v: &[f64]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
        let mut r = vec![0.0; v.len()];
        for (k, &i) in idx.iter().enumerate() {
            r[i] = k as f64;
        }
        r
    };
    let (rx, ry) = (ranks(x), ranks(y));
    let m = (x.len() as f64 - 1.0) / 2.0;
    let sxy: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - m) * (b - m)).sum();
    let sxx: f64 = rx.iter().map(|a| (a - m).powi(2)).sum();
    sxy / sxx
}

/// Every row's ultimates less its reserves are the latest values.
fn assert_ultimates_follow_reserves(fits: &MeyersFits) {
    let latest: Vec<f64> = fits
        .segments
        .fits
        .iter()
        .flat_map(|s| s.chain_ladder.latest.clone())
        .collect();
    let n = latest.len();
    for (u, r) in fits
        .ultimates
        .draw_matrix()
        .chunks_exact(n)
        .zip(fits.reserves.draw_matrix().chunks_exact(n))
    {
        for ((u, r), l) in u.iter().zip(r).zip(&latest) {
            assert_eq!(*r, u - l);
        }
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn each_segment_has_its_own_seeds() {
    let tri = twins();
    let fits = quick(MeyersModel::Crc, 9)
        .fit_segments(&tri, "paid", "premium")
        .unwrap();
    for pd in [&fits.reserves, &fits.ultimates] {
        assert_eq!(pd.dims(), ["lob", "origin"]);
        assert_eq!(pd.n_components(), 20);
        assert_eq!(pd.n_sims(), 2_000);
    }
    assert_eq!(fits.segments.len(), 2);
    let (a, b) = (&fits.segments.fits[0], &fits.segments.fits[1]);
    assert_eq!(a.premium, PREMIUM);
    assert_eq!(a.model(), MeyersModel::Crc);
    assert_ne!(a.posterior.values(), b.posterior.values());
    // Not the single-segment fit's seeds either.
    let alone = quick(MeyersModel::Crc, 9)
        .fit(&comauto(), "paid", "premium")
        .unwrap();
    assert_ne!(a.posterior.values(), alone.posterior.values());
    assert_ultimates_follow_reserves(&fits);
    // Shared streams would make the twins' totals almost perfectly
    // rank correlated; their own make them independent.
    let totals = line_totals(&fits.reserves);
    let r = spearman(&totals[0], &totals[1]);
    assert!(r.abs() < 4.5 / 2_000f64.sqrt(), "rank correlation {r}");

    let long = fits.to_long();
    assert_eq!(long.n_rows(), 20);
    for column in ["premium", "mean", "ultimate_mean", "ultimate_std_dev"] {
        assert!(long.column(column).is_some(), "{column}");
    }
    let ulr_mean: Vec<f64> = (long.column("ultimate_mean").unwrap().iter())
        .zip(PREMIUM.iter().chain(&PREMIUM))
        .map(|(u, p)| u / p)
        .collect();
    assert_eq!(long.column("ulr_mean").unwrap(), ulr_mean);
    let t = fits.totals();
    assert_eq!(t.n_rows(), 2);
    let premium: f64 = PREMIUM.iter().sum();
    assert_eq!(t.column("premium").unwrap(), [premium, premium]);
    let ultimate_mean = t.column("ultimate_mean").unwrap();
    assert_eq!(
        t.column("ulr_mean").unwrap(),
        [ultimate_mean[0] / premium, ultimate_mean[1] / premium]
    );
    let mean_b = totals[1].iter().sum::<f64>() / 2e3;
    assert!((t.column("mean").unwrap()[1] / mean_b - 1.0).abs() < 1e-9);
    assert_eq!(fits.development_table().n_rows(), 20);
    let one = fits.segment(&[("lob", "b")]).unwrap();
    assert_eq!(one.reserves.n_components(), 10);
    assert_eq!(one.ultimates.n_components(), 10);
    assert_eq!(one.segments.fits[0].posterior, b.posterior);
    assert_eq!(b.summary().unwrap().len(), 29);
    assert_eq!(b.loo().unwrap().pareto_k.len(), 55);

    for (pd, output) in [(&fits.reserves, "reserves"), (&fits.ultimates, "ultimates")] {
        let provenance = pd.provenance();
        assert_eq!(provenance.model, "meyers_crc");
        for (key, value) in [
            ("segments", "2"),
            ("output", output),
            ("dependence", "Independent"),
        ] {
            assert!(
                provenance.parameters.contains(&(key.into(), value.into())),
                "{key} {value}: {:?}",
                provenance.parameters
            );
        }
    }

    let again = quick(MeyersModel::Crc, 9)
        .fit_segments(&tri, "paid", "premium")
        .unwrap();
    assert_eq!(again.reserves.draw_matrix(), fits.reserves.draw_matrix());
    assert_eq!(again.ultimates.draw_matrix(), fits.ultimates.draw_matrix());
}

#[test]
fn synchronized_segments_are_an_error() {
    let model = Meyers {
        dependence: SegmentDependence::Synchronized,
        ..quick(MeyersModel::Csr, 0)
    };
    let e = model.fit_segments(&twins(), "paid", "premium").unwrap_err();
    assert!(
        matches!(
            &e,
            Error::NotSupported {
                setting: "Synchronized dependence",
                ..
            }
        ),
        "{e}"
    );
}

#[test]
fn a_bad_rank_correlation_fails_before_sampling() {
    // With a sampler this long, any sampling before the check would
    // not finish.
    let slow = Meyers {
        sampler: Sampler {
            tune: 1_000_000,
            draws: 1_000_000,
            ..Sampler::default()
        },
        ..Meyers::default()
    };
    for spearman in [
        vec![1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0],
        vec![0.9, 0.5, 0.5, 1.0],
        vec![1.0, 0.5, 0.4, 1.0],
        vec![1.0, 1.5, 1.5, 1.0],
    ] {
        let model = Meyers {
            dependence: SegmentDependence::RankCorrelation { spearman },
            ..slow.clone()
        };
        let e = model.fit_segments(&twins(), "paid", "premium").unwrap_err();
        assert!(
            matches!(
                &e,
                Error::InvalidSetting {
                    name: "spearman",
                    ..
                }
            ),
            "{e}"
        );
    }
    // A bad setting fails first too.
    let e = Meyers {
        n_sims: 0,
        ..slow.clone()
    }
    .fit_segments(&twins(), "paid", "premium")
    .unwrap_err();
    assert!(
        matches!(e, Error::InvalidSetting { name: "n_sims", .. }),
        "{e}"
    );
    assert!(matches!(
        slow.fit_segments(&twins(), "paid", "exposure"),
        Err(Error::UnknownColumn(_))
    ));
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn rank_correlation_moves_reserves_and_ultimates_together() {
    let tri = twins();
    let independent = quick(MeyersModel::Cay, 10)
        .fit_segments(&tri, "paid", "premium")
        .unwrap();
    let ranked = Meyers {
        dependence: SegmentDependence::RankCorrelation {
            spearman: vec![1.0, 0.8, 0.8, 1.0],
        },
        ..quick(MeyersModel::Cay, 10)
    }
    .fit_segments(&tri, "paid", "premium")
    .unwrap();
    // Each line keeps its own simulations, as whole rows.
    let rows = |pd: &PredictiveDistribution, s: usize| {
        let mut rows: Vec<Vec<f64>> = pd
            .draw_matrix()
            .chunks(20)
            .map(|r| r[s * 10..(s + 1) * 10].to_vec())
            .collect();
        rows.sort_by(|x, y| {
            x.iter()
                .zip(y)
                .map(|(a, b)| a.total_cmp(b))
                .find(|o| o.is_ne())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rows
    };
    for s in 0..2 {
        assert_eq!(rows(&independent.reserves, s), rows(&ranked.reserves, s));
        assert_eq!(rows(&independent.ultimates, s), rows(&ranked.ultimates, s));
    }
    assert_ultimates_follow_reserves(&ranked);
    let totals = line_totals(&ranked.reserves);
    let r = spearman(&totals[0], &totals[1]);
    assert!((r - 0.8).abs() < 0.05, "rank correlation {r}");
    assert!(
        ranked
            .reserves
            .provenance()
            .parameters
            .iter()
            .any(|(k, _)| k == "rank_correlation")
    );
}

/// Group 353's paid cells as a frame, with its premium as exposure.
fn frame() -> TriangleFrame {
    TriangleFrame::new(&comauto(), "paid", Some(&PREMIUM)).unwrap()
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn a_backtest_reports_diagnostics_and_still_scores_a_short_run() {
    // One chain of 50 kept draws cannot reach 100 effective draws per
    // chain, so every split warns.
    let short = Meyers {
        sampler: Sampler {
            chains: 1,
            tune: 50,
            draws: 50,
            ..Meyers::default().sampler
        },
        ..Meyers::default()
    };
    let bt = crate::diagonal_backtest(&frame(), &[&short], 2, 200, 5, 0.9).unwrap();
    assert_eq!(bt.models(), ["meyers_crc"]);
    assert_eq!(bt.diagnostics(0).len(), 2);
    for d in bt.diagnostics(0) {
        let d = d.as_ref().expect("a Bayesian model reports diagnostics");
        assert!(!d.warnings.is_empty(), "{d:?}");
    }
    let warnings = bt.warnings();
    assert!(warnings.iter().any(|(m, s, _)| (*m, *s) == (0, 0)));
    assert!(warnings.iter().any(|(m, s, _)| (*m, *s) == (0, 1)));
    // Scored all the same.
    assert!(bt.split_scores(0, 0).iter().all(|v| v.is_finite()));
    assert!(bt.split_scores(0, 3).iter().all(|v| v.is_finite()));
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_backtest_model_is_the_fit_on_the_train_rows() {
    // The latest diagonal held out: the train rows are the triangle of
    // the first nine origins without it (row 10 w + d is origin w at
    // age d).
    let cells = frame();
    let held: Vec<usize> = (1..9).map(|w| w * 10 + 9 - w).collect();
    let train: Vec<usize> = cells
        .observed()
        .iter()
        .copied()
        .filter(|&r| r / 10 + r % 10 < 9)
        .collect();
    let short = triangle(&PAID[..9], &PREMIUM[..9]);
    for kind in [MeyersModel::Crc, MeyersModel::Csr] {
        let model = quick(kind, 12);
        assert_eq!(
            TriangleModel::name(&model),
            format!("meyers_{}", kind.name())
        );
        let f = model.forecast(&cells, &train, &held, 2_000, 12).unwrap();
        let fit = model.fit(&short, "paid", "premium").unwrap();
        // The same posterior: each test row's mean and log density from
        // the fit's draws, written out here.
        let p = fit.posterior.dim();
        let lpd = f.log_density.clone().unwrap();
        for (j, w) in (1..9).enumerate() {
            let d = 9 - w;
            let (before, after) = (PAID[w][d - 1], PAID[w][d]);
            let (mut mean, mut dens) = (0.0, 0.0);
            let n = fit.posterior.n_draws() as f64;
            for nat in fit.posterior.values().chunks_exact(p) {
                let beta = if d < 8 { nat[9 + d] } else { 0.0 };
                let g = if kind == MeyersModel::Csr {
                    (1.0 - nat[17]).powi(w as i32)
                } else {
                    1.0
                };
                let mu = PREMIUM[w].ln() + nat[0] + nat[w] + beta * g;
                let sigma = nat[p - 9 + d];
                mean += ((mu + 0.5 * sigma * sigma).exp() - before) / n;
                let z = (after.ln() - mu) / sigma;
                dens += (-0.5 * z * z).exp() / (after * sigma * std::f64::consts::TAU.sqrt()) / n;
            }
            assert!(
                (f.mean[j] / mean - 1.0).abs() < 1e-9,
                "{kind:?} {w}: {} vs {mean}",
                f.mean[j]
            );
            assert!(
                (lpd[j] - dens.ln()).abs() < 1e-9,
                "{kind:?} {w}: {} vs {}",
                lpd[j],
                dens.ln()
            );
            // The draws' mean is the mean, up to Monte Carlo error.
            let x: Vec<f64> = f
                .distribution
                .draw_matrix()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|r| r[j])
                .collect();
            let m = x.iter().sum::<f64>() / x.len() as f64;
            let sd = (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
            assert!(
                (m - mean).abs() < 5.0 * sd / (x.len() as f64).sqrt(),
                "{m} vs {mean}"
            );
        }
        assert_eq!(f.distribution.n_components(), 8);
        let d = f
            .diagnostics
            .as_ref()
            .expect("a Bayesian model reports diagnostics");
        assert!(d.max_rhat.is_finite() && d.max_rhat < 1.1, "{d:?}");
        assert_eq!(f.distribution.n_sims(), 2_000);
        let again = model.forecast(&cells, &train, &held, 2_000, 12).unwrap();
        assert_eq!(again.diagnostics.as_ref(), Some(d));
        assert_eq!(
            again.distribution.draw_matrix(),
            f.distribution.draw_matrix()
        );
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_next_cell_simulates_as_the_fit() {
    // On the whole triangle, origin 1989's cell at 120 months is its
    // ultimate: the backtest's draws of its increment are the fit's
    // reserve draws, simulation by simulation (integer cells, so the
    // rebuilt cumulative values are exact).
    let cells = frame();
    let train = cells.observed().to_vec();
    for kind in [MeyersModel::Crc, MeyersModel::Cay] {
        let model = quick(kind, 13);
        let f = model.forecast(&cells, &train, &[19], 1_000, 13).unwrap();
        let fit = Meyers {
            n_sims: 1_000,
            ..model.clone()
        }
        .fit(&comauto(), "paid", "premium")
        .unwrap();
        let reserve: Vec<f64> = fit
            .reserves
            .draw_matrix()
            .as_chunks::<10>()
            .0
            .iter()
            .map(|r| r[1])
            .collect();
        for (a, b) in f.distribution.draw_matrix().iter().zip(&reserve) {
            assert!(
                (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                "{kind:?}: {a} vs {b}"
            );
        }
        // A future cell has no response, so no density there.
        assert_eq!(f.log_density.unwrap(), [f64::NEG_INFINITY]);
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn an_origin_path_telescopes() {
    // Origin 1997's nine future cells, one path per simulation: the
    // increments sum to its ultimate less its latest value, with the
    // fit's reserve's mean and spread (independent cells would add up
    // nine variances).
    let cells = frame();
    let train = cells.observed().to_vec();
    let test: Vec<usize> = (91..100).collect();
    let model = quick(MeyersModel::Crc, 14);
    let f = model.forecast(&cells, &train, &test, 4_000, 14).unwrap();
    assert!(f.log_density.is_none());
    let sums: Vec<f64> = f
        .distribution
        .draw_matrix()
        .as_chunks::<9>()
        .0
        .iter()
        .map(|r| r.iter().sum())
        .collect();
    let fit = Meyers {
        n_sims: 4_000,
        ..model.clone()
    }
    .fit(&comauto(), "paid", "premium")
    .unwrap();
    let reserve = fit
        .reserves
        .marginal(&vec![Period::year(1997).into()])
        .unwrap();
    let m = sums.iter().sum::<f64>() / sums.len() as f64;
    let sd = (sums.iter().map(|v| (v - m).powi(2)).sum::<f64>() / sums.len() as f64).sqrt();
    let se = reserve.std_dev() / (sums.len() as f64).sqrt();
    assert!(
        (m - reserve.mean()).abs() < 6.0 * se,
        "{m} vs {}",
        reserve.mean()
    );
    assert!(
        (sd / reserve.std_dev() - 1.0).abs() < 0.1,
        "{sd} vs {}",
        reserve.std_dev()
    );
    // The analytic means telescope too.
    let total: f64 = f.mean.iter().sum();
    assert!((total / m - 1.0).abs() < 0.05, "{total} vs {m}");
}

#[test]
fn cay_paths_take_the_previous_origins_residual() {
    // A fixed draw with a large sigma and rho; origins 1996 and 1997
    // both on paths at 36 months, so 1997's mean takes 1996's drawn
    // residual, while 1996's takes 1995's observed one. 1997's path
    // runs from 24 months.
    let cells = frame();
    let train = cells.observed().to_vec();
    let test = [82, 91, 92];
    let rebuilt = Rebuilt::new(&cells, &train, &test).unwrap();
    let model = prepared(MeyersModel::Cay);
    let (k, rho, sigma) = (model.after_beta(), 0.7, 0.4);
    let mut nat = model.natural(&model.start());
    nat[k] = rho;
    for s in &mut nat[model.u0()..] {
        *s = sigma;
    }
    let at = cell_index(&model.cells, 10, 10);
    let mut residuals = Vec::new();
    model.cay_residuals(&nat, &mut residuals);
    let (m8, v8) = model.log_moments(&nat, &residuals, &at, &PREMIUM, 8, 2);
    let (m9, v9) = model.log_moments(&nat, &residuals, &at, &PREMIUM, 9, 2);
    assert!((v8 - sigma * sigma).abs() < 1e-15);
    assert!((v9 - (1.0 + rho * rho) * sigma * sigma).abs() < 1e-15);
    let ages = rebuilt.path_ages();
    assert_eq!(ages, [1, 2]);
    let mut rng = StreamRng::new(3, 0);
    let mut row = [0.0; 3];
    let n = 100_000;
    let (mut logs8, mut logs9) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        model.draw_paths(&nat, &rebuilt, &at, &ages, &mut rng, &mut row);
        logs8.push((row[0] + rebuilt.last_value[8]).ln());
        logs9.push((row[1] + row[2] + rebuilt.last_value[9]).ln());
    }
    let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
    let var = |x: &[f64]| {
        let m = mean(x);
        x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64
    };
    for (logs, m, v) in [(&logs8, m8, v8), (&logs9, m9, v9)] {
        assert!((mean(logs) - m).abs() < 5.0 * (v / n as f64).sqrt());
        assert!((var(logs) / v - 1.0).abs() < 0.02, "{} vs {v}", var(logs));
    }
    // 1997's log value moves with 1996's residual: covariance
    // rho sigma^2.
    let (a, b) = (mean(&logs8), mean(&logs9));
    let cov = logs8
        .iter()
        .zip(&logs9)
        .map(|(x, y)| (x - a) * (y - b))
        .sum::<f64>()
        / (n - 1) as f64;
    assert!((cov / (rho * sigma * sigma) - 1.0).abs() < 0.03, "{cov}");
}

#[test]
fn backtest_errors() {
    let cells = frame();
    let train = cells.observed().to_vec();
    let model = quick(MeyersModel::Crc, 0);
    // A test row at or before its origin's latest train row.
    let e = model.forecast(&cells, &train, &[5], 100, 0).unwrap_err();
    assert!(
        matches!(&e, Error::Bayes(m) if m.contains("not after")),
        "{e}"
    );
    // Origin 1988 without its 12-month row.
    let gap: Vec<usize> = train.iter().copied().filter(|&r| r != 0).collect();
    let e = model.forecast(&cells, &gap, &[19], 100, 0).unwrap_err();
    assert!(
        matches!(&e, Error::Bayes(m)
            if m.starts_with("origin 1988 has no train row at development age 12")),
        "{e}"
    );
    // Origin 1990 without train rows, between 1989 and 1991.
    let gap: Vec<usize> = train.iter().copied().filter(|&r| r / 10 != 2).collect();
    let e = model.forecast(&cells, &gap, &[19], 100, 0).unwrap_err();
    assert!(
        matches!(&e, Error::Bayes(m) if m.starts_with("origin 1990 has no train row")),
        "{e}"
    );
    // The newest origin without its only cell.
    let without: Vec<usize> = train.iter().copied().filter(|&r| r != 90).collect();
    let e = model.forecast(&cells, &without, &[91], 100, 0).unwrap_err();
    assert!(
        matches!(&e, Error::Bayes(m) if m.contains("origin without a train row")),
        "{e}"
    );
    // An age past the oldest train age.
    let young: Vec<usize> = train.iter().copied().filter(|&r| r % 10 < 8).collect();
    let e = model.forecast(&cells, &young, &[18], 100, 0).unwrap_err();
    assert!(
        matches!(&e, Error::Bayes(m) if m.contains("an age without a train row")),
        "{e}"
    );
    // Settings are checked first.
    let e = Meyers {
        gamma_sd: 0.0,
        ..model
    }
    .forecast(&cells, &train, &[19], 100, 0)
    .unwrap_err();
    assert!(
        matches!(
            e,
            Error::InvalidSetting {
                name: "gamma_sd",
                ..
            }
        ),
        "{e}"
    );
}

// The one-year view (decision 19).

/// A fully developed 4 x 4 square of paid with premium.
fn square() -> Triangle {
    let rows: [[f64; 4]; 4] = [
        [357.0, 1124.0, 1735.0, 2218.0],
        [352.0, 1236.0, 2170.0, 3353.0],
        [290.0, 1292.0, 2219.0, 3235.0],
        [310.0, 1418.0, 2195.0, 3757.0],
    ];
    let (mut origin, mut ages, mut paid, mut prem) = (vec![], vec![], vec![], vec![]);
    for (w, row) in rows.iter().enumerate() {
        for (d, &v) in row.iter().enumerate() {
            origin.push(Month::january(2001 + w as i32));
            ages.push(12 * (d as crate::Lag + 1));
            paid.push(v);
            prem.push(5000.0 + 500.0 * w as f64);
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("paid", &paid), ("premium", &prem)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// The pieces a [`year::MeyersYear`] borrows: the segment, its chain
/// ladder and premiums, and the model.
fn year_parts(meyers: &Meyers, tri: &Triangle) -> (Segment, ChainLadderFit, Vec<f64>, Model) {
    let segment = tri.segment("paid").unwrap();
    let cl = ChainLadder::default()
        .fit_segment(&segment, &segment.ages)
        .unwrap();
    let premium = origin_exposure(&segment, &tri.segment("premium").unwrap(), "premium").unwrap();
    let model = meyers.model_of(&segment, &cl, &premium).unwrap();
    (segment, cl, premium, model)
}

/// The year's pieces without sampling: the importance ratio is the
/// change in the log posterior, the expected ultimate of CRC and CSR
/// does not depend on the year, and the opening and closing reserves
/// are the means of the lifetime simulation's reserve given a draw
/// (with CAY's integrated residual), the closing one on the triangle
/// extended by the year.
#[test]
fn the_year_follows_the_model() {
    use crate::bayes::one_year::YearModel;
    let tri = comauto();
    for kind in [MeyersModel::Crc, MeyersModel::Csr, MeyersModel::Cay] {
        let meyers = Meyers {
            model: kind,
            ..Meyers::default()
        };
        let (segment, cl, premium, model) = year_parts(&meyers, &tri);
        let year = year::MeyersYear::new(&meyers, &model, &tri, &segment, &cl, &premium).unwrap();
        assert_eq!(year.n_year(), 9, "{kind:?}");
        assert_eq!(year.n_origins(), 10);
        let theta = off_mode(&model);
        let mut other = theta.clone();
        for (j, v) in other.iter_mut().enumerate() {
            *v -= 0.02 * (j as f64 % 5.0 - 2.0);
        }
        let mut rng = StreamRng::new(3, 0);
        let mut values = vec![0.0; 9];
        year.draw_year(&theta, &mut rng, &mut values);
        assert!(values.iter().all(|v| v.is_finite() && *v > 0.0));
        // 1. The ratio is the change in the log posterior.
        let change = |t: &[f64]| {
            year.log_posterior(t, &values)
                - year.log_posterior(t, &[])
                - year.year_log_likelihood(t, &values)
        };
        let (a, b) = (change(&theta), change(&other));
        assert!((a - b).abs() < 1e-9 * (1.0 + a.abs()), "{kind:?}: {a} {b}");
        // 2. The year's payment and closing reserve.
        let mut opening = vec![0.0; 10];
        let mut paid = vec![0.0; 10];
        let mut closing = vec![0.0; 10];
        year.opening(&theta, &mut opening);
        year.paid(&values, &mut paid);
        year.closing(&theta, &values, &mut closing);
        assert_eq!((opening[0], paid[0], closing[0]), (0.0, 0.0, 0.0));
        // 1989 reaches the oldest age in the year.
        assert_eq!(closing[1], 0.0);
        assert_eq!(paid[1], values[0] - cl.latest[1]);
        if kind != MeyersModel::Cay {
            for w in 2..10 {
                let moved = closing[w] + paid[w];
                assert!(
                    (moved - opening[w]).abs() < 1e-9 * opening[w].abs().max(1.0),
                    "{kind:?} origin {w}: {moved} {}",
                    opening[w]
                );
            }
        }
        // 3. The opening reserve is the lifetime reserve's mean given
        // the draw.
        let nat = model.natural(&theta);
        let closed: Vec<bool> = cl.latest_position.iter().map(|&d| d == 9).collect();
        let n = 50_000;
        let mut sum = [0.0; 10];
        let mut sq = [0.0; 10];
        let mut row = vec![0.0; 10];
        let mut rng = StreamRng::new(5, 0);
        for _ in 0..n {
            model.draw_ultimates(&nat, &premium, &closed, &cl.latest, &mut rng, &mut row);
            for w in 0..10 {
                let r = row[w] - cl.latest[w];
                sum[w] += r;
                sq[w] += r * r;
            }
        }
        for w in 1..10 {
            let mean = sum[w] / n as f64;
            let se = ((sq[w] / n as f64 - mean * mean) / n as f64).sqrt();
            assert!(
                (mean - opening[w]).abs() < 5.0 * se,
                "{kind:?} origin {w}: simulated {mean}, opening {} (se {se})",
                opening[w]
            );
        }
        // 4. The closing reserve is the lifetime reserve's mean given
        // the draw on the triangle extended by the year: each origin's
        // latest value is its year value, and 1989, which the year
        // takes to the oldest age, is closed. In CAY this checks the
        // conditioning on the year independently of `closing`: `rho`
        // times 1989's residual at the oldest age for 1990, and the
        // integrated residual of the previous origin for the others.
        // At a draw with `rho` 0.6 (CAY) and `sigma_10^2` about 0.5,
        // where both terms move the closing reserve by many standard
        // errors (near the mode they are within the noise).
        let mut wide = theta.clone();
        if kind == MeyersModel::Cay {
            wide[model.after_beta()] = 0.6f64.atanh();
        }
        wide[model.u0() + 9] = 0.0;
        year.closing(&wide, &values, &mut closing);
        let nat = model.natural(&wide);
        let mut latest = cl.latest.clone();
        for (j, &v) in values.iter().enumerate() {
            latest[j + 1] = v;
        }
        let closed: Vec<bool> = cl.latest_position.iter().map(|&d| d + 1 >= 9).collect();
        let (mut sum, mut sq) = ([0.0; 10], [0.0; 10]);
        let mut rng = StreamRng::new(7, 0);
        for _ in 0..n {
            model.draw_ultimates(&nat, &premium, &closed, &latest, &mut rng, &mut row);
            for w in 0..10 {
                let r = row[w] - latest[w];
                sum[w] += r;
                sq[w] += r * r;
            }
        }
        for w in 2..10 {
            let mean = sum[w] / n as f64;
            let se = ((sq[w] / n as f64 - mean * mean) / n as f64).sqrt();
            assert!(
                (mean - closing[w]).abs() < 5.0 * se,
                "{kind:?} origin {w}: simulated {mean}, closing {} (se {se})",
                closing[w]
            );
        }
    }
    // A square has no year.
    let closed = square();
    let meyers = Meyers::default();
    let (segment, cl, premium, model) = year_parts(&meyers, &closed);
    let year = year::MeyersYear::new(&meyers, &model, &closed, &segment, &cl, &premium).unwrap();
    assert_eq!(year.n_year(), 0);
}

/// The one-year conformance suite (decision 17) on CRC, CSR and CAY,
/// with a fully developed square as the triangle without a year.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_one_year_view_conforms() {
    for kind in [MeyersModel::Crc, MeyersModel::Csr, MeyersModel::Cay] {
        let meyers = quick(kind, 4);
        let fitted: Vec<(Triangle, Segment, MeyersSegment)> = [comauto(), square()]
            .into_iter()
            .map(|tri| {
                let segment = tri.segment("paid").unwrap();
                let premium = tri.segment("premium").unwrap();
                let (fit, _, _, _) = meyers
                    .fit_segment(
                        &segment,
                        &premium,
                        "paid",
                        "premium",
                        Stream::Chains,
                        Stream::Predictive,
                    )
                    .unwrap();
                (tri, segment, fit)
            })
            .collect();
        let years: Vec<year::MeyersYear> = fitted
            .iter()
            .map(|(tri, segment, fit)| {
                year::MeyersYear::new(
                    &meyers,
                    &fit.model,
                    tri,
                    segment,
                    &fit.chain_ladder,
                    &fit.premium,
                )
                .unwrap()
            })
            .collect();
        crate::bayes::one_year::conformance::check(
            &format!("Meyers {kind:?}"),
            &years[0],
            &fitted[0].2.sampled,
            &years[1],
            &fitted[1].2.sampled,
        );
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_one_year_view_reweights_the_fits_posterior() {
    let model = quick(MeyersModel::Csr, 6);
    let only = BayesOneYear {
        refit: Some(crate::bayes::OneYearRefit::ReweightOnly),
        ..BayesOneYear::default()
    };
    let view = model
        .one_year(&comauto(), "paid", "premium", &only)
        .unwrap();
    // The posterior and lifetime predictive are fit()'s, draw for draw.
    let fit = model.fit(&comauto(), "paid", "premium").unwrap();
    assert_eq!(view.fit.posterior, fit.posterior);
    assert_eq!(view.fit.reserves.draw_matrix(), fit.reserves.draw_matrix());
    assert_eq!(
        view.fit.ultimates.draw_matrix(),
        fit.ultimates.draw_matrix()
    );
    assert_eq!(view.cdr.dims(), ["origin"]);
    assert_eq!(view.cdr.n_components(), 10);
    assert_eq!(view.pareto_k.len(), 2_000);
    assert!(view.refitted.is_empty() && view.moment_matched.is_empty());
    // The settings' n_sims overrides the model's for the year only, and
    // gives the first simulations of the larger count.
    let fewer = model
        .one_year(
            &comauto(),
            "paid",
            "premium",
            &BayesOneYear {
                n_sims: Some(500),
                ..only
            },
        )
        .unwrap();
    assert_eq!(fewer.cdr.n_sims(), 500);
    assert_eq!(fewer.fit.reserves.n_sims(), 2_000);
    assert_eq!(fewer.pareto_k, view.pareto_k[..500]);
    assert_eq!(fewer.cdr.draw_matrix(), &view.cdr.draw_matrix()[..500 * 10]);
    // 1988 is fully developed.
    assert!(view.cdr.draw_matrix().chunks(10).all(|r| r[0] == 0.0));
    // The opening reserve is the lifetime reserve's posterior mean.
    let lifetime = fit.reserves.total();
    let opening: f64 = view.opening_reserve.iter().sum();
    let se = lifetime.std_dev() / 2_000f64.sqrt();
    assert!(
        (opening - lifetime.mean()).abs() < 5.0 * se,
        "opening {opening}, lifetime {} (se {se})",
        lifetime.mean()
    );
    assert!(view.cdr.std_dev() < fit.reserves.std_dev());
    assert_eq!(view.cdr.provenance().model, "meyers_csr_one_year");
    let long = view.to_long();
    assert_eq!(long.n_rows(), 10);
    assert!(long.column("premium").is_some());
    assert!(view.totals().column("flagged_share").is_some());
    // The default moment-matches the flagged simulations, refitting
    // only those matching cannot rescue.
    let matched = Meyers {
        n_sims: 300,
        ..model.clone()
    }
    .one_year(&comauto(), "paid", "premium", &BayesOneYear::default())
    .unwrap();
    let flagged: Vec<usize> = (0..300).filter(|&i| view.pareto_k[i] > 0.7).collect();
    assert_eq!(matched.pareto_k, view.pareto_k[..300]);
    let mut either = matched.moment_matched.clone();
    either.extend(&matched.refitted);
    either.sort_unstable();
    assert_eq!(either, flagged);
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "samples with NUTS; needs --features slow-tests"
)]
fn the_one_year_view_over_segments() {
    let only = BayesOneYear {
        refit: Some(crate::bayes::OneYearRefit::ReweightOnly),
        ..BayesOneYear::default()
    };
    let tri = twins();
    let model = quick(MeyersModel::Crc, 9);
    let views = model
        .one_year_segments(&tri, "paid", "premium", &only)
        .unwrap();
    assert_eq!(views.cdr.dims(), ["lob", "origin"]);
    assert_eq!(views.cdr.n_components(), 20);
    let lifetime = model.fit_segments(&tri, "paid", "premium").unwrap();
    for (a, b) in views.segments.fits.iter().zip(&lifetime.segments.fits) {
        assert_eq!(a.fit.posterior, b.posterior);
        assert_eq!(a.pareto_k.len(), 2_000);
    }
    let totals = line_totals(&views.cdr);
    let r = spearman(&totals[0], &totals[1]);
    assert!(r.abs() < 4.5 / 2_000f64.sqrt(), "rank correlation {r}");
    assert_eq!(views.to_long().n_rows(), 20);
    assert_eq!(views.totals().n_rows(), 2);
    assert_eq!(
        views.segment(&[("lob", "b")]).unwrap().cdr.n_components(),
        10
    );
    let correlated = Meyers {
        dependence: SegmentDependence::RankCorrelation {
            spearman: vec![1.0, 0.7, 0.7, 1.0],
        },
        ..model
    };
    let views = correlated
        .one_year_segments(&tri, "paid", "premium", &only)
        .unwrap();
    let totals = line_totals(&views.cdr);
    let r = spearman(&totals[0], &totals[1]);
    assert!((r - 0.7).abs() < 0.05, "rank correlation {r}");
}

#[test]
fn one_year_errors() {
    let e = Meyers::default()
        .one_year(
            &comauto(),
            "paid",
            "premium",
            &BayesOneYear {
                k_threshold: f64::NAN,
                ..BayesOneYear::default()
            },
        )
        .unwrap_err();
    assert!(
        matches!(
            e,
            Error::InvalidSetting {
                name: "k_threshold",
                ..
            }
        ),
        "{e}"
    );
    let e = Meyers {
        dependence: SegmentDependence::Synchronized,
        ..Meyers::default()
    }
    .one_year_segments(&twins(), "paid", "premium", &BayesOneYear::default())
    .unwrap_err();
    assert!(
        matches!(
            &e,
            Error::NotSupported {
                setting: "Synchronized dependence",
                ..
            }
        ),
        "{e}"
    );
    assert!(matches!(
        Meyers::default().one_year(&twins(), "paid", "premium", &BayesOneYear::default()),
        Err(Error::MultipleSegments(2))
    ));
    let e = Meyers::default()
        .one_year(
            &comauto(),
            "paid",
            "premium",
            &BayesOneYear {
                n_sims: Some(0),
                ..BayesOneYear::default()
            },
        )
        .unwrap_err();
    assert!(
        matches!(e, Error::InvalidSetting { name: "n_sims", .. }),
        "{e}"
    );
}

#[test]
fn the_one_year_view_runs_at_most_2000_simulations_by_default() {
    let s = BayesOneYear::default();
    assert_eq!(Meyers::default().n_sims, 10_000);
    assert_eq!(Meyers::default().one_year_sims(&s), 2_000);
    let small = Meyers {
        n_sims: 500,
        ..Meyers::default()
    };
    assert_eq!(small.one_year_sims(&s), 500);
    let more = BayesOneYear {
        n_sims: Some(10_000),
        ..s
    };
    assert_eq!(Meyers::default().one_year_sims(&more), 10_000);
    assert_eq!(small.one_year_sims(&more), 10_000);
}
