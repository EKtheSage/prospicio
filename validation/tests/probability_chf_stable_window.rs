//! A stable law through `chf::invert` with grid options given (#945): a
//! tilt asked for on a fixed grid, the window of a tilted one-sided law on
//! a given step, and `Stable::tail_mass_above` outside its range.

use prospicio_prob::chf::{ChfLaw, ChfOptions, Tilt, invert};
use prospicio_prob::{Stable, StableParameterization};

fn stable(alpha: f64, beta: f64) -> Stable {
    Stable::new(alpha, beta, 1.0, 0.0, StableParameterization::S1).unwrap()
}

#[test]
fn an_explicit_tilt_on_a_fixed_grid_is_refused() {
    let options = ChfOptions {
        log2: 12,
        step: Some(0.05),
        x_min: Some(-20.0),
        tilt: Tilt::Fixed(6.0),
        ..ChfOptions::default()
    };
    let err = invert(&ChfLaw::Stable(stable(1.5, 1.0)), &options)
        .expect_err("a tilt is not applied on a grid whose left edge is given")
        .to_string();
    assert!(err.contains("x_min"), "{err}");
}

#[test]
fn the_automatic_tilt_on_a_fixed_grid_falls_back_to_none() {
    let options = ChfOptions {
        log2: 12,
        step: Some(0.05),
        x_min: Some(-20.0),
        ..ChfOptions::default()
    };
    let d = invert(&ChfLaw::Stable(stable(1.5, 1.0)), &options).unwrap();
    assert_eq!(d.report().tilt, (0.0, 0.0));
}

#[test]
fn a_tilted_window_on_a_given_step_holds_the_body() {
    for (alpha, beta) in [(1.5, 1.0), (1.5, -1.0), (1.2, 1.0)] {
        let st = stable(alpha, beta);
        let options = ChfOptions {
            log2: 12,
            step: Some(0.001),
            ..ChfOptions::default()
        };
        let d = invert(&ChfLaw::Stable(st), &options).unwrap();
        let top = d.x_min() + 4096.0 * d.step();
        let mean = st.mean();
        assert!(
            d.x_min() <= mean && mean <= top,
            "{alpha} {beta}: window [{}, {top}] misses the mean {mean}",
            d.x_min()
        );
        // and its masses are a probability: the tilt must not magnify the
        // light side's mass below the window (#1010)
        let r = d.report();
        assert!(d.masses().iter().all(|&m| m >= -1e-9), "{alpha} {beta}");
        assert!(r.total_mass <= 1.0 + 1e-6, "{alpha} {beta}: {r:?}");
        assert!(r.total_mass + r.outside_above >= 1.0 - 1e-6, "{alpha} {beta}: {r:?}");
    }
}

#[test]
fn the_tail_mass_is_a_probability_or_an_error() {
    let st = stable(1.5, 1.0);
    // The reference: the complement of the cdf of a wide, tilted inversion.
    let wide = invert(&ChfLaw::Stable(st), &ChfOptions::with_log2(18)).unwrap();
    let (h, x0) = (wide.step(), wide.x_min());
    let mut cum = Vec::with_capacity(wide.masses().len());
    let mut acc = 0.0;
    for &m in wide.masses() {
        acc += m;
        cum.push(acc);
    }
    // Bucket k holds the mass on [x0 + (k - 1/2) h, x0 + (k + 1/2) h): take
    // the part of the bucket holding x that lies above it.
    let reference = |x: f64| {
        let k = ((x - x0) / h).round() as usize;
        let above_in_bucket = (x0 + (k as f64 + 0.5) * h - x) / h;
        1.0 - cum[k] + above_in_bucket * wide.masses()[k]
    };
    let mut answered = 0;
    for u in [-1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0, 1000.0] {
        if let Ok(v) = st.tail_mass_above(u) {
            answered += 1;
            assert!((0.0..=1.0).contains(&v), "u = {u}: {v}");
            let want = reference(u);
            assert!(
                (v - want).abs() <= 1e-3 * want + 1e-9,
                "u = {u}: {v} against {want}"
            );
        }
    }
    assert!(answered >= 3, "the far tail must still be answered");
    assert!(st.tail_mass_above(0.0).is_err());
    assert!(st.tail_mass_above(0.5).is_err());
    // alpha < 1 and small u: a partial sum that has not converged.
    let low = stable(0.7, 1.0);
    for u in [0.01, 0.1, 0.5] {
        if let Ok(v) = low.tail_mass_above(u) {
            assert!((0.0..=1.0).contains(&v), "alpha 0.7 u = {u}: {v}");
        }
    }
    // alpha = 2 has an exact tail at every point.
    let normal = stable(2.0, 0.0);
    assert!((normal.tail_mass_above(0.0).unwrap() - 0.5).abs() < 1e-12);
}
