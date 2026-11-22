//! A stable law with two power tails inverted on a given step (#937). The
//! split moves a normal from its wide half to its narrow one, which the
//! wide half can give up only while the whole law's truncation
//! `exp(-(pi gamma / h)^alpha)` is at most `2^-104`. Above that step the
//! wide half's transform grew with the frequency and the masses were
//! garbage (a sum of 21.2, or of -2.4e-5), returned as `Ok`.

use prospicio_prob::chf::{ChfLaw, ChfOptions, invert};
use prospicio_prob::{Stable, StableParameterization};

fn inverted(
    alpha: f64,
    beta: f64,
    step: f64,
    log2: u32,
) -> prospicio_core::Result<(f64, f64, f64)> {
    let st = Stable::new(alpha, beta, 1.0, 0.0, StableParameterization::S1).unwrap();
    let options = ChfOptions {
        log2,
        step: Some(step),
        ..ChfOptions::default()
    };
    let d = invert(&ChfLaw::Stable(st), &options)?;
    let sum: f64 = d.masses().iter().sum();
    let least = d.masses().iter().copied().fold(f64::INFINITY, f64::min);
    // The mass below the window is in the first bucket, the mass above is
    // left out and reported.
    Ok((sum + d.report().outside_above, least, d.report().edge_error))
}

/// The largest step the split can take: `pi gamma / (104 ln 2)^(1/alpha)`.
fn largest(alpha: f64) -> f64 {
    std::f64::consts::PI / (104.0 * 2f64.ln()).powf(1.0 / alpha)
}

#[test]
fn a_step_above_the_splits_limit_is_refused() {
    // The issue's two rows.
    for (alpha, beta, step) in [(1.5, 0.5, 0.5), (1.8, 0.0, 0.5)] {
        let err = inverted(alpha, beta, step, 10).unwrap_err().to_string();
        let limit = format!("{:.6}", largest(alpha));
        assert!(
            err.contains("step") && err.contains(&limit[..6]),
            "{alpha} {beta}: {err}"
        );
    }
    // Just above the limit, for several laws.
    for (alpha, beta) in [(0.7, 0.3), (1.0, -0.4), (1.5, 0.5), (1.8, 0.0), (1.95, 0.9)] {
        assert!(inverted(alpha, beta, largest(alpha) * 1.001, 10).is_err(), "{alpha} {beta}");
    }
}

#[test]
fn a_step_up_to_the_limit_gives_a_distribution() {
    // The issue's first row, below the automatic cap, and steps between
    // the automatic cap (a rung of the ladder under gamma / 10) and the
    // limit, which stay valid.
    // 2^10 buckets as in the issue, then 2^16, where the window reaches far
    // enough into both tails for their series to be exact.
    let mut cases = vec![(1.5, 0.5, 0.05, 10)];
    for (alpha, beta) in [(0.7, 0.3), (1.0, -0.4), (1.5, 0.5), (1.8, 0.0), (1.95, 0.9)] {
        cases.push((alpha, beta, largest(alpha) * 0.999, 16));
        cases.push((alpha, beta, largest(alpha) * 0.5, 16));
    }
    for (alpha, beta, step, log2) in cases {
        let (total, least, edge) = inverted(alpha, beta, step, log2).unwrap();
        // The outside masses come from the tails' series, so the total is 1
        // to a small part of the modelled edge error.
        assert!((total - 1.0).abs() < 1e-6 + 1e-3 * edge, "{alpha} {beta} {step}: total {total}");
        assert!(least > -1e-12, "{alpha} {beta} {step}: least {least}");
    }
}
