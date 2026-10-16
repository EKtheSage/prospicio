//! `PRO.MACK.BOOT` and `PRO.MACK.BOOT.SIMS` on RAA, reached through the
//! catalogue as Excel reaches them: the same draws as
//! `prospicio_reserving::MackBootstrap`, and the reconciliation
//! `validation/tests/reserving_mack_bootstrap.rs` makes: with centred
//! residuals the mean reserve is the chain ladder's and the total's
//! standard deviation is Mack's, `sqrt(process^2 + v parameter^2)` from R
//! ChainLadder's risks with `v` = 0.981 the resampled pool's variance on
//! RAA, each within five Monte Carlo standard errors.

mod common;

use common::{n, num, raa, t, text};
use prospicio_reserving::MackBootstrap;
use prospicio_xll::call;
use prospicio_xll::reserving::triangle;
use prospicio_xll::sheet::Grid;

fn boot(sims: f64, seed: f64, process: Grid) -> Grid {
    call(
        "pro_mack_boot",
        vec![
            raa(),
            n(sims),
            n(seed),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            process,
            Grid::missing(),
        ],
    )
}

// R ChainLadder 0.2.21, MackChainLadder(RAA, est.sigma = "log-linear"):
// Total.ProcessRisk and Total.ParameterRisk
// (validation/reference/reserving_chainladder_r.csv).
const PROCESS: f64 = 24_917.265_746_524_2;
const PARAMETER: f64 = 10_084.843_499_040_4;
const RESERVE: f64 = 52_135.228_261_210_1;
/// The resampled residual pool's variance on RAA, `1 - m^2`.
const POOL_VARIANCE: f64 = 0.981;

#[test]
fn total_reconciles_with_mack() {
    let sims = 20_000.0;
    let out = boot(sims, 1.0, Grid::missing());
    assert_eq!((out.rows, out.columns), (12, 9));
    let (mean, sd) = (num(out.get(11, 2)), num(out.get(11, 3)));
    let expected_sd = (PROCESS.powi(2) + POOL_VARIANCE * PARAMETER.powi(2)).sqrt();
    // Five standard errors: of the mean, sd / sqrt(n); of the sd, with a
    // kurtosis below 6, sd * sqrt(5 / (4 n)).
    assert!(
        (mean - RESERVE).abs() < 5.0 * sd / sims.sqrt(),
        "mean {mean}"
    );
    assert!(
        (sd - expected_sd).abs() < 5.0 * sd * (5.0 / (4.0 * sims)).sqrt(),
        "sd {sd} vs {expected_sd}"
    );
    assert!((num(out.get(11, 1)) - RESERVE).abs() < 1e-6, "CL reserve");
}

#[test]
fn parameter_error_alone_is_mack_parameter_risk() {
    let sims = 20_000.0;
    let out = boot(sims, 2.0, t("none"));
    let sd = num(out.get(11, 3));
    let expected = POOL_VARIANCE.sqrt() * PARAMETER;
    assert!(
        (sd - expected).abs() < 5.0 * sd * (5.0 / (4.0 * sims)).sqrt(),
        "{sd} vs {expected}"
    );
}

#[test]
fn sims_are_the_crates_draws_and_the_exhibits() {
    let fit = MackBootstrap {
        n_sims: 1_000,
        seed: 5,
        ..MackBootstrap::default()
    }
    .fit(&triangle(&raa()).unwrap(), "values")
    .unwrap();
    let draws = call(
        "pro_mack_boot_sims",
        vec![
            raa(),
            n(1_000.0),
            n(5.0),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
        ],
    );
    assert_eq!((draws.rows, draws.columns), (1_001, 11));
    let k = fit.reserves.n_components();
    for (i, row) in fit.reserves.draw_matrix().chunks_exact(k).enumerate() {
        for (j, &x) in row.iter().enumerate() {
            assert_eq!(num(draws.get(i + 1, j)), x);
        }
    }
    let exhibit = boot(1_000.0, 5.0, Grid::missing());
    let total: f64 = (1..=1_000).map(|r| num(draws.get(r, 10))).sum::<f64>() / 1_000.0;
    let mean = num(exhibit.get(11, 2));
    assert!((total - mean).abs() <= 1e-9 * mean);
}

#[test]
fn options_and_errors() {
    for process in ["lognormal", "normal", "residuals"] {
        let out = boot(500.0, 0.0, t(process));
        assert_eq!(out.rows, 12, "{process}");
    }
    let tail = call(
        "pro_mack_boot",
        vec![
            raa(),
            n(500.0),
            n(0.0),
            Grid::missing(),
            t("simple"),
            n(1.05),
            Grid::missing(),
            Grid::missing(),
        ],
    );
    assert!(num(tail.get(11, 1)) > RESERVE, "a tail adds reserve");
    assert!(text(&boot(500.0, 0.0, t("poisson"))).contains("process must be"));
    assert!(text(&boot(0.0, 0.0, Grid::missing())).contains("sims"));
}
