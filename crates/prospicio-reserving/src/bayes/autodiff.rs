//! Test-only support for the generic densities
//! (`docs/design/reserving-compartmental.md`, decision 4): the
//! [`GenericDensity`] trait the oracle densities beside `BayesOdp`,
//! `BayesBf` and `Meyers` implement, the oracle check against their
//! hand-written gradients, the one-compartment closed form with its
//! central-difference test, and the benchmark that decision 15 of
//! `reserving-bayes.md` owed (`#[ignore]`d; run it in release with
//! `cargo test --release -p prospicio-reserving --features bayes --lib
//! autodiff_benchmark -- --ignored --nocapture`).

use std::hint::black_box;
use std::time::{Duration, Instant};

use prospicio_core::StreamRng;

use super::dual::{CHUNK, gradient};
use super::laplace::{Posterior, Whitened};
use super::scalar::{Scalar, exprel_neg};

/// A log density written once over [`Scalar`]: its value with `f64`, its
/// gradient with [`Dual`] through [`gradient`].
pub(crate) trait GenericDensity {
    fn n_params(&self) -> usize;
    fn log_density<S: Scalar>(&self, theta: &[S]) -> S;
}

/// `n` standard normal draws from stream `stream` of `seed`, by
/// Box-Muller.
fn normals(seed: u64, stream: u64, n: usize) -> Vec<f64> {
    let mut rng = StreamRng::new(seed, stream);
    (0..n)
        .map(|_| {
            let (u, v) = (rng.next_open01(), rng.next_open01());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        })
        .collect()
}

/// `n` points on the posterior's scale: `centre + L z` for standard
/// normal `z`, with `whitened`'s `L` (a Laplace scale at the mode, or the
/// regularised one at a start point for `Meyers`).
pub(crate) fn posterior_points<P: Posterior + ?Sized>(
    whitened: &Whitened<'_, P>,
    p: usize,
    n: usize,
) -> Vec<Vec<f64>> {
    (0..n as u64)
        .map(|k| whitened.theta(&normals(20_261_013, k, p)))
        .collect()
}

/// The largest relative errors of a model's generic density against its
/// hand-written one over some points.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OracleErrors {
    /// `|generic - hand| / |hand|` of the value.
    pub(crate) value: f64,
    /// `max_j |dual_j - hand_j| / max_j |hand_j|`: each gradient's error
    /// relative to its largest component.
    pub(crate) gradient: f64,
    /// `max_j |dual_j - hand_j| / |hand_j|`, componentwise over the
    /// nonzero components.
    pub(crate) componentwise: f64,
}

/// Checks `model`'s generic value and chunked dual gradient against its
/// hand-written `log_posterior` at `points`, to 1e-12 relative (the
/// gradient both against its largest component and componentwise); returns
/// and prints the largest errors.
pub(crate) fn assert_oracle<M: Posterior + GenericDensity>(
    label: &str,
    model: &M,
    points: &[Vec<f64>],
) -> OracleErrors {
    let p = model.dim();
    assert_eq!(model.n_params(), p, "{label}: parameter counts");
    let mut errors = OracleErrors {
        value: 0.0,
        gradient: 0.0,
        componentwise: 0.0,
    };
    let (mut hand, mut dual, mut chunked) = (vec![0.0; p], vec![0.0; p], vec![0.0; p]);
    for theta in points {
        let f = model.log_posterior(theta, &mut hand);
        assert!(f.is_finite(), "{label}: the point is outside the support");
        let generic = model.log_density(theta);
        // The crate's chunk (one pass for these models) and chunks of 8,
        // several passes, which must agree exactly: each tangent is
        // computed alone.
        let fd = gradient::<CHUNK>(theta, &mut dual, |x| model.log_density(x));
        assert_eq!(fd, generic, "{label}: the dual value is the f64 value");
        gradient::<8>(theta, &mut chunked, |x| model.log_density(x));
        assert_eq!(chunked, dual, "{label}: chunks of 8");
        errors.value = errors.value.max((generic - f).abs() / f.abs());
        let scale = hand.iter().fold(0.0f64, |a, g| a.max(g.abs()));
        for (a, b) in dual.iter().zip(&hand) {
            errors.gradient = errors.gradient.max((a - b).abs() / scale);
            if *b != 0.0 {
                errors.componentwise = errors.componentwise.max((a - b).abs() / b.abs());
            }
        }
    }
    println!(
        "oracle {label}: value {:.1e}, gradient {:.1e} (componentwise {:.1e}) at {} points",
        errors.value,
        errors.gradient,
        errors.componentwise,
        points.len()
    );
    assert!(errors.value <= 1e-12, "{label}: value {:e}", errors.value);
    assert!(
        errors.gradient <= 1e-12,
        "{label}: gradient {:e}",
        errors.gradient
    );
    assert!(
        errors.componentwise <= 1e-12,
        "{label}: componentwise gradient {:e}",
        errors.componentwise
    );
    errors
}

/// The one-compartment closed form (the compartmental note, decision 2) per
/// unit of premium: `OS(t) / (Π RLR) = k_er g(t)` and
/// `PD(t) / (Π RLR RRF) = (1 - e^(-k_p t)) - k_p g(t)`, with
/// `g(t) = t e^(-k_p t) (1 - e^(-Δ t)) / (Δ t)` and `Δ = k_er - k_p`,
/// finite and smooth through `k_er = k_p` by [`exprel_neg`].
pub(crate) fn one_os<S: Scalar>(k_er: S, k_p: S, t: f64) -> (S, S) {
    let decay = k_p * (-t);
    let g = decay.exp() * t * exprel_neg((k_er - k_p) * t);
    (k_er * g, -decay.exp_m1() - k_p * g)
}

/// Cells of a triangle: `(origin, age, and two observed values)`.
type Cells = Vec<(usize, f64, f64, f64)>;

/// A hand-written log posterior with its gradient.
type HandGradient<'a> = &'a dyn Fn(&[f64], &mut [f64]) -> f64;

/// A synthetic training triangle shaped like the papers' company 337
/// (accident years 1988 to 1996 with 9 down to 1 ages): `(origin, age,
/// outstanding, cumulative paid)` and the premiums, generated from the
/// one-OS model near the paper's model 1 posterior (`k_er` 1.75, `k_p`
/// 0.43) with a deterministic 3% wobble. It sizes the benchmark; it is not
/// company 337.
fn synthetic_337() -> (Cells, Vec<f64>) {
    let premium: Vec<f64> = (0..9).map(|i| 100_000.0 + 3_000.0 * i as f64).collect();
    let mut cells = Vec::new();
    for (i, &prem) in premium.iter().enumerate() {
        let rlr = 0.7 * (0.2 * (1.44 + 0.1 * (i as f64 - 4.0))).exp();
        let rrf = 0.8 * (-0.136f64).exp();
        for t in 1..=(9 - i) {
            let (os, pd) = one_os(1.75, 0.43, t as f64);
            let wobble = |k: f64| 1.0 + 0.03 * (7.0 * i as f64 + k * t as f64).sin();
            cells.push((
                i,
                t as f64,
                prem * rlr * os * wobble(3.0),
                prem * rlr * rrf * pd * wobble(5.0),
            ));
        }
    }
    (cells, premium)
}

/// `log` of a half Student-t(`nu`, 0, `s`) density at `tau = e^l`, less its
/// constant, with the Jacobian `l` of the log map.
fn half_t_on_log<S: Scalar>(l: S, nu: f64, s: f64) -> S {
    let x = l.exp() / s;
    l - (x * x / nu).ln_1p() * ((nu + 1.0) / 2.0)
}

/// Gesmann and Morris' model 1 with one OS compartment (the compartmental
/// note, decision 2) on a synthetic triangle: 27 parameters,
/// `(b_ker, b_kp, b_RLR, b_RRF, log tau_1, log tau_2, atanh rho, z_11,
/// z_12, .., z_91, z_92, log sigma_OS, log sigma_PD)`.
pub(crate) struct OneOsModel1 {
    cells: Cells,
    premium: Vec<f64>,
}

impl OneOsModel1 {
    pub(crate) fn synthetic() -> Self {
        let (cells, premium) = synthetic_337();
        Self { cells, premium }
    }

    /// A point near the posterior.
    pub(crate) fn point(&self) -> Vec<f64> {
        let mut x = vec![-5.4, -8.49, 1.44, -1.36, -0.45, -0.3, 0.2];
        x.extend((0..18).map(|k| 0.3 * ((k % 5) as f64 - 2.0)));
        x.extend([8.25, 6.66]);
        x
    }
}

impl GenericDensity for OneOsModel1 {
    fn n_params(&self) -> usize {
        27
    }

    fn log_density<S: Scalar>(&self, x: &[S]) -> S {
        let n = self.premium.len();
        let mut f = S::from_f64(0.0);
        // b ~ N(0, 1) and z ~ N(0, 1).
        for &v in x[..4].iter().chain(&x[7..7 + 2 * n]) {
            f -= v * v * 0.5;
        }
        f += half_t_on_log(x[4], 10.0, 0.2) + half_t_on_log(x[5], 10.0, 0.1);
        // LKJ(1) is uniform on rho; the Jacobian of rho = tanh(v).
        let rho = x[6].tanh();
        let rho_c = (-(rho * rho) + 1.0).sqrt();
        f += (-(rho * rho) + 1.0).ln();
        let (ls_os, ls_pd) = (x[7 + 2 * n], x[8 + 2 * n]);
        // Cauchy(0, 1000) on log sigma.
        for l in [ls_os, ls_pd] {
            f -= ((l / 1000.0) * (l / 1000.0)).ln_1p();
        }
        let (inv_os, inv_pd) = ((-ls_os).exp(), (-ls_pd).exp());
        let k_er = (x[0] * 0.1).exp() * 3.0;
        let k_p = (x[1] * 0.1).exp();
        let (tau1, tau2) = (x[4].exp(), x[5].exp());
        // The rates are shared, so the closed form is one per age.
        let curves: Vec<(S, S)> = (1..=n).map(|t| one_os(k_er, k_p, t as f64)).collect();
        let mut origin = Vec::with_capacity(n);
        for i in 0..n {
            let (z1, z2) = (x[7 + 2 * i], x[8 + 2 * i]);
            let u1 = tau1 * z1;
            let u2 = tau2 * (rho * z1 + rho_c * z2);
            let rlr = ((x[2] + u1) * 0.2).exp() * (0.7 * self.premium[i]);
            let rrf = ((x[3] + u2) * 0.1).exp() * 0.8;
            origin.push((rlr, rlr * rrf));
        }
        for &(i, t, os, pd) in &self.cells {
            let (cos, cpd) = curves[t as usize - 1];
            let (a, b) = origin[i];
            let r_os = (-(a * cos) + os) * inv_os;
            let r_pd = (-(b * cpd) + pd) * inv_pd;
            f -= ls_os + ls_pd + (r_os * r_os + r_pd * r_pd) * 0.5;
        }
        f
    }
}

/// Gesmann and Morris' model 2 (the compartmental note, decision 2: 87
/// parameters) on the same synthetic triangle, with three standard normal
/// parameters appended so that `p` = 90, the size decision 4 benchmarks:
/// `(b[4], log tau_AY[4], log tau_DY[4], atanh rho, z_AY[9][4],
/// z_DY[9][4], log sigma_OS, log sigma_PD, pad[3])`, each in the order
/// `k_er`, `k_p`, `RLR`, `RRF`. Each cell has its own four parameters,
/// and the incremental paid evaluates the closed form at `t` and `t - 1`
/// with them.
pub(crate) struct OneOsModel2 {
    /// `(origin, age, log OS loss ratio, log incremental paid loss ratio)`.
    cells: Cells,
}

impl OneOsModel2 {
    pub(crate) fn synthetic() -> Self {
        let (cells, premium) = synthetic_337();
        let cells = cells
            .iter()
            .enumerate()
            .map(|(k, &(i, t, os, pd))| {
                let before = if t > 1.0 { cells[k - 1].3 } else { 0.0 };
                (
                    i,
                    t,
                    (os / premium[i]).ln(),
                    ((pd - before) / premium[i]).ln(),
                )
            })
            .collect();
        Self { cells }
    }

    pub(crate) fn point(&self) -> Vec<f64> {
        let mut x = vec![-5.4, -8.49, 1.44, -1.36];
        x.extend([-1.5; 8]);
        x.push(0.2);
        x.extend((0..72).map(|k| 0.3 * ((k % 5) as f64 - 2.0)));
        x.extend([-1.6, -1.6, 0.1, -0.2, 0.3]);
        x
    }
}

impl GenericDensity for OneOsModel2 {
    fn n_params(&self) -> usize {
        90
    }

    fn log_density<S: Scalar>(&self, x: &[S]) -> S {
        const MEDIAN: [f64; 4] = [3.0, 1.0, 0.7, 0.8];
        const SCALE: [f64; 4] = [0.1, 0.1, 0.2, 0.1];
        const SD_PRIOR: [f64; 4] = [0.3, 0.3, 0.7, 0.5];
        let (z_ay, z_dy, ls) = (13, 13 + 36, 13 + 72);
        let mut f = S::from_f64(0.0);
        for &v in x[..4].iter().chain(&x[z_ay..ls]).chain(&x[ls + 2..]) {
            f -= v * v * 0.5;
        }
        for k in 0..4 {
            f += half_t_on_log(x[4 + k], 10.0, SD_PRIOR[k]);
            f += half_t_on_log(x[8 + k], 10.0, SD_PRIOR[k]);
        }
        let rho = x[12].tanh();
        let rho_c = (-(rho * rho) + 1.0).sqrt();
        f += (-(rho * rho) + 1.0).ln();
        let (ls_os, ls_pd) = (x[ls], x[ls + 1]);
        // log sigma ~ N(log 0.2, 0.2).
        for l in [ls_os, ls_pd] {
            let r = (l - 0.2f64.ln()) / 0.2;
            f -= r * r * 0.5;
        }
        let (inv_os, inv_pd) = ((-ls_os).exp(), (-ls_pd).exp());
        let tau: Vec<S> = x[4..12].iter().map(|l| l.exp()).collect();
        // Accident-year effects (RLR and RRF correlated) and
        // development-year effects (uncorrelated) on each `o`.
        let ay = |i: usize, k: usize| {
            let z = |k: usize| x[z_ay + 4 * i + k];
            match k {
                3 => tau[3] * (rho * z(2) + rho_c * z(3)),
                _ => tau[k] * z(k),
            }
        };
        let dy = |t: usize, k: usize| tau[4 + k] * x[z_dy + 4 * t + k];
        for &(i, t, y_os, y_pd) in &self.cells {
            let d = t as usize - 1;
            let o: [S; 4] = std::array::from_fn(|k| x[k] + ay(i, k) + dy(d, k));
            let par: [S; 4] = std::array::from_fn(|k| (o[k] * SCALE[k]).exp() * MEDIAN[k]);
            let (cos, cpd) = one_os(par[0], par[1], t);
            let inc = if d == 0 {
                cpd
            } else {
                cpd - one_os(par[0], par[1], t - 1.0).1
            };
            // log loss ratios: OS / premium and the paid increment / premium.
            let mu_os = (par[2] * cos).ln();
            let mu_pd = (par[2] * par[3] * inc).ln();
            let r_os = (-mu_os + y_os) * inv_os;
            let r_pd = (-mu_pd + y_pd) * inv_pd;
            f -= ls_os + ls_pd + (r_os * r_os + r_pd * r_pd) * 0.5;
        }
        f
    }
}

#[test]
fn the_closed_form_gradient_matches_central_differences() {
    // Through k_er = k_p exactly, within the series branch
    // (|Δ t| < EXPREL_SERIES = 1e-2), just inside and just outside its edge
    // (at t = 1), and away from it; the last row is the 99.9th prior
    // percentiles.
    let points = [
        (1.75, 0.43),
        (3.0, 1.0),
        (1.0, 1.0),
        (1.0, 1.0 + 3e-6),
        (0.7, 0.7 - 2e-7),
        (1.0, 1.0099),
        (1.0, 1.0101),
        (0.5, 4.0),
        (4.09, 1.36),
    ];
    let mut worst: f64 = 0.0;
    for (k_er, k_p) in points {
        for t in [0.5, 1.0, 3.0, 10.0] {
            for which in 0..2 {
                // (k_er, k_p, RLR, RRF).
                let theta = [k_er, k_p, 0.85, 0.94];
                let f = |x: &[f64]| {
                    let (os, pd) = one_os(x[0], x[1], t);
                    [x[2] * os, x[2] * x[3] * pd][which]
                };
                let mut grad = [0.0; 4];
                gradient::<4>(&theta, &mut grad, |x| {
                    let (os, pd) = one_os(x[0], x[1], t);
                    [x[2] * os, x[2] * x[3] * pd][which]
                });
                for j in 0..4 {
                    let h = 1e-5 * theta[j];
                    let (mut up, mut down) = (theta, theta);
                    up[j] += h;
                    down[j] -= h;
                    let fd = (f(&up) - f(&down)) / (2.0 * h);
                    let err = (fd - grad[j]).abs() / grad[j].abs().max(1e-3);
                    worst = worst.max(err);
                    assert!(
                        err <= 1e-7,
                        "({k_er}, {k_p}) t {t} output {which} parameter {j}: \
                         central {fd}, dual {}",
                        grad[j]
                    );
                }
            }
        }
    }
    println!("closed form: dual vs central differences, worst relative {worst:.1e}");
}

#[test]
fn the_closed_form_is_the_ode_solution() {
    // OS(t) = k_er (e^-kp t - e^-ker t) / (k_er - k_p) and its paid, away
    // from the coincidence; at it, OS = k t e^-k t.
    let (k_er, k_p, t) = (1.7f64, 0.5f64, 2.0f64);
    let (os, pd) = one_os(k_er, k_p, t);
    let g = ((-k_p * t).exp() - (-k_er * t).exp()) / (k_er - k_p);
    assert!((os - k_er * g).abs() <= 1e-15);
    let paid = (k_er * (1.0 - (-k_p * t).exp()) - k_p * (1.0 - (-k_er * t).exp())) / (k_er - k_p);
    assert!((pd - paid).abs() <= 1e-15, "{pd} vs {paid}");
    let (os, _) = one_os(0.8, 0.8, t);
    assert!((os - 0.8 * t * (-0.8 * t).exp()).abs() <= 1e-16);
}

#[test]
fn the_compartmental_generic_densities_are_finite_with_finite_gradients() {
    let m1 = OneOsModel1::synthetic();
    let m2 = OneOsModel2::synthetic();
    assert_eq!(m1.cells.len(), 45);
    for (n, x, value) in [
        (m1.n_params(), m1.point(), m1.log_density(&m1.point())),
        (m2.n_params(), m2.point(), m2.log_density(&m2.point())),
    ] {
        assert_eq!(x.len(), n);
        assert!(value.is_finite());
    }
    let mut g1 = vec![0.0; 27];
    gradient::<CHUNK>(&m1.point(), &mut g1, |x| m1.log_density(x));
    let mut g2 = vec![0.0; 90];
    gradient::<CHUNK>(&m2.point(), &mut g2, |x| m2.log_density(x));
    assert!(g1.iter().chain(&g2).all(|g| g.is_finite()));
}

/// The median time of one call of `f` in nanoseconds: 31 samples, each
/// repeating `f` for at least 2 ms.
fn median_ns(mut f: impl FnMut()) -> f64 {
    let mut reps = 1u32;
    loop {
        let start = Instant::now();
        for _ in 0..reps {
            f();
        }
        if start.elapsed() >= Duration::from_millis(2) {
            break;
        }
        reps *= 2;
    }
    let mut samples: Vec<f64> = (0..31)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..reps {
                f();
            }
            start.elapsed().as_secs_f64() * 1e9 / f64::from(reps)
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[15]
}

/// The benchmark's times for one density, in nanoseconds per value and
/// gradient: the f64 value alone, the hand-written gradient if there is
/// one, the chunked dual at `N` = 4, 8, 16 and 32, and the fixed-size dual
/// (`N = P`, one pass).
pub(crate) fn bench_row<D: GenericDensity, const P: usize>(
    label: &str,
    model: &D,
    theta: &[f64],
    hand: Option<HandGradient<'_>>,
) {
    assert_eq!(theta.len(), P);
    let mut g = vec![0.0; P];
    let value = median_ns(|| {
        black_box(model.log_density(black_box(theta)));
    });
    let hand_ns = hand.map(|h| {
        median_ns(|| {
            black_box(h(black_box(theta), &mut g));
        })
    });
    let mut chunked = |n: usize| -> f64 {
        macro_rules! run {
            ($n:literal) => {
                median_ns(|| {
                    black_box(gradient::<$n>(black_box(theta), &mut g, |x| {
                        model.log_density(x)
                    }));
                })
            };
        }
        match n {
            4 => run!(4),
            8 => run!(8),
            16 => run!(16),
            32 => run!(32),
            _ => unreachable!(),
        }
    };
    let times: Vec<f64> = [4, 8, 16, 32].into_iter().map(&mut chunked).collect();
    let fixed = median_ns(|| {
        black_box(gradient::<P>(black_box(theta), &mut g, |x| {
            model.log_density(x)
        }));
    });
    let base = hand_ns.unwrap_or(value);
    let base_name = if hand_ns.is_some() { "hand" } else { "value" };
    let us = |ns: f64| ns / 1e3;
    println!(
        "| {label} | {P} | {:.2} | {} | {} | {:.2} ({:.1}x {base_name}) |",
        us(value),
        hand_ns.map_or("n/a".into(), |h| format!("{:.2}", us(h))),
        times
            .iter()
            .zip([4, 8, 16, 32])
            .map(|(t, n)| format!(
                "N={n}: {:.2} ({:.1}x, {} passes)",
                us(*t),
                t / base,
                P.div_ceil(n)
            ))
            .collect::<Vec<_>>()
            .join(" | "),
        us(fixed),
        fixed / base,
    );
}

/// The measurement decision 15 of `reserving-bayes.md` owed: release
/// build, median time per value and gradient (µs).
#[test]
#[ignore = "a release benchmark; run with --release --ignored --nocapture"]
fn autodiff_benchmark() {
    if cfg!(debug_assertions) {
        println!("warning: a debug build; the times mean nothing");
    }
    println!(
        "| density | p | f64 value | hand-written | chunked N=4 | N=8 | N=16 | N=32 | fixed size |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    let m1 = OneOsModel1::synthetic();
    bench_row::<_, 27>("one-OS model 1, synthetic 337", &m1, &m1.point(), None);
    super::meyers::tests::bench_crc_353();
    let m2 = OneOsModel2::synthetic();
    bench_row::<_, 90>(
        "one-OS model 2 + 3 pad, synthetic 337",
        &m2,
        &m2.point(),
        None,
    );
}
