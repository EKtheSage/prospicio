//! Benchmarks of the GLM hot paths named in #969: IRLS on a
//! freMTPL2-sized portfolio and on a triangle-sized one, a GAM with REML
//! smoothing and the Tweedie power profile.
//!
//! `cargo bench -p prospicio-glm` prints the median wall time of each case.
//! Set `BENCH_QUICK=1` for one timed run per case.

use prospicio_glm::Glm;
use prospicio_glm::gam::{Gam, PSpline, Smoothing};
use prospicio_glm::tweedie::tweedie_profile;
use prospicio_models::{Design, Family, Link, Model};
use std::hint::black_box;

mod synthetic;
use synthetic::{Rng, portfolio, time_ms};

fn runs(full: usize) -> usize {
    if std::env::var_os("BENCH_QUICK").is_some() {
        1
    } else {
        full
    }
}

fn design(p: &synthetic::Portfolio) -> Design {
    Design::new(p.names.clone(), p.columns.clone())
        .unwrap()
        .with_offset(p.offset.clone())
        .unwrap()
}

fn report(name: &str, ms: f64) {
    println!("{name:<40} {ms:>12.3} ms");
}

fn main() {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("prospicio-glm benchmarks (#969), {threads} hardware threads");

    // freMTPL2 size: 678,013 rows, 60 columns.
    let big = portfolio(678_013, 60, 4, 969);
    let big_design = design(&big);
    let poisson = Glm::new(Family::Poisson, Link::Log);
    report(
        "glm_fit_poisson_n678013_p60",
        time_ms(runs(3), || {
            black_box(poisson.fit(&big_design, &big.y).unwrap());
        }),
    );

    // Triangle size, as the reserving GLMs refit it: 100 rows, 20 columns,
    // timed over 1000 fits.
    let small = portfolio(100, 20, 2, 1);
    let small_design = design(&small);
    report(
        "glm_fit_poisson_n100_p20_x1000",
        time_ms(runs(3), || {
            for _ in 0..1000 {
                black_box(poisson.fit(&small_design, &small.y).ok());
            }
        }),
    );

    // GAM with REML smoothing: 50,000 rows, 10 linear columns and a
    // P-spline in one of them.
    let gam_data = portfolio(50_000, 10, 2, 7);
    let gam_design = design(&gam_data);
    let gam = Gam::new(poisson, vec![PSpline::new("x1")]).smoothing(Smoothing::Reml);
    report(
        "gam_reml_poisson_n50000_p10_spline",
        time_ms(runs(3), || {
            black_box(gam.fit(&gam_design, &gam_data.y).unwrap());
        }),
    );

    // The same GAM for gamma severities, whose REML score searches the
    // dispersion at every trial.
    let mut rng = Rng::new(8);
    let sev: Vec<f64> = gam_data
        .y
        .iter()
        .map(|&k| 500.0 * (1.0 + k) * (0.2 - rng.uniform().ln()))
        .collect();
    let gamma = Gam::new(Glm::new(Family::Gamma, Link::Log), vec![PSpline::new("x1")])
        .smoothing(Smoothing::Reml);
    report(
        "gam_reml_gamma_n50000_p10_spline",
        time_ms(runs(3), || {
            black_box(gamma.fit(&gam_design, &sev).unwrap());
        }),
    );

    // Tweedie profile over a grid of five powers: 100,000 rows, 20 columns,
    // compound Poisson-gamma responses.
    let tw = portfolio(100_000, 20, 2, 11);
    let mut rng = Rng::new(12);
    let y: Vec<f64> =
        tw.y.iter()
            .map(|&k| (0..k as usize).map(|_| -rng.uniform().ln()).sum::<f64>() * 1000.0)
            .collect();
    let tw_design = design(&tw);
    report(
        "tweedie_profile_n100000_p20_5powers",
        time_ms(runs(3), || {
            black_box(
                tweedie_profile(Link::Log, &tw_design, &y, &[1.1, 1.3, 1.5, 1.7, 1.9]).unwrap(),
            );
        }),
    );
}
