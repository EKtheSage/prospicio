//! The economic scenario generator (`docs/design/esg-dfa.md`, validation 1
//! to 3): Vasicek's and CIR's simulated discount factors against their
//! closed-form bond prices and their exact transitions' moments; Hull-White
//! against the initial curve it reproduces; the Svensson fit against the
//! curve that generated its yields.

use prospicio_esg::{Cir, Esg, HullWhite, InitialCurve, Scenarios, Step, Svensson, Vasicek};
use prospicio_reserving::DiscountCurve;

const SIMS: usize = 20_000;

/// The mean of a column over paths and its standard error.
fn column(sims: usize, f: impl Fn(usize) -> f64) -> (f64, f64) {
    let draws: Vec<f64> = (0..sims).map(f).collect();
    let n = sims as f64;
    let mean = draws.iter().sum::<f64>() / n;
    let var = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

/// The deflators' mean at step `k` against `price`, within four standard
/// errors and a bias allowance.
fn deflator_prices(paths: &Scenarios, k: usize, price: f64, bias: f64) {
    let (mean, se) = column(paths.n_sims(), |s| paths.deflator(s, k).unwrap());
    assert!(
        (mean - price).abs() < 4.0 * se + bias,
        "step {k}: {mean} ± {se} against {price}"
    );
}

#[test]
fn vasicek_deflators_price_its_bonds_and_its_transition_has_its_moments() {
    let model = Vasicek::new(0.25, 0.045, 0.012, 0.02).unwrap();
    let paths = Esg::new(model).simulate(10, SIMS, 2026).unwrap();
    for years in [1, 5, 10] {
        deflator_prices(&paths, 12 * years, model.bond_price(0.02, years as f64), 0.0);
    }
    for years in [1usize, 10] {
        let k = 12 * years;
        let (mean, se) = column(SIMS, |s| paths.short_rate(s, k).unwrap());
        let t = years as f64;
        assert!((mean - model.mean(t)).abs() < 4.0 * se, "{mean} vs {}", model.mean(t));
        let var = column(SIMS, |s| (paths.short_rate(s, k).unwrap() - model.mean(t)).powi(2)).0;
        assert!((var / model.variance(t) - 1.0).abs() < 0.05, "{var}");
    }
}

#[test]
fn cir_deflators_price_its_bonds_and_its_transition_has_its_moments() {
    // Feller holds for the first; the second reaches zero (dof 0.8 < 1, the
    // Poisson mixture of the transition).
    for model in [
        Cir::new(0.3, 0.04, 0.06, 0.03).unwrap(),
        Cir::new(0.2, 0.02, 0.1, 0.01).unwrap(),
    ] {
        let paths = Esg::new(model).simulate(10, SIMS, 7).unwrap();
        // The trapezoid integral's bias in a monthly step is far below a
        // basis point of price.
        for years in [1, 5, 10] {
            deflator_prices(&paths, 12 * years, model.bond_price(model.r0, years as f64), 1e-4);
        }
        let annual = Esg::new(model).with_step(Step::Annual).simulate(3, SIMS, 9).unwrap();
        for k in [1, 3] {
            let t = k as f64;
            let (mean, se) = column(SIMS, |s| annual.short_rate(s, k).unwrap());
            assert!((mean - model.mean(t)).abs() < 4.0 * se, "{mean}");
            let var = column(SIMS, |s| (annual.short_rate(s, k).unwrap() - model.mean(t)).powi(2)).0;
            assert!((var / model.variance(t) - 1.0).abs() < 0.05, "{var}");
            assert!((0..SIMS).all(|s| annual.short_rate(s, k).unwrap() >= 0.0));
        }
    }
}

#[test]
fn hull_white_reproduces_the_initial_curve() {
    let initial = InitialCurve::spot(
        vec![0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0],
        vec![0.042, 0.040, 0.037, 0.036, 0.037, 0.039, 0.041, 0.045, 0.044],
    )
    .unwrap();
    let curve = DiscountCurve::spot(
        &[0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0],
        &[0.042, 0.040, 0.037, 0.036, 0.037, 0.039, 0.041, 0.045, 0.044],
    )
    .unwrap();
    let model = HullWhite::new(0.08, 0.011, initial.clone()).unwrap();
    let maturities = vec![0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0];
    let paths = Esg::new(model.clone())
        .with_maturities(maturities.clone())
        .simulate(10, SIMS, 3)
        .unwrap();
    // The model's bond prices today are the curve's, to 1e-10, and the
    // same as prospicio_reserving's DiscountCurve on the same nodes.
    for (j, &m) in maturities.iter().enumerate() {
        let p = paths.bond_price(0, 0, j).unwrap();
        assert!((p - initial.discount(m)).abs() < 1e-10);
        assert!((p - curve.factor(m)).abs() < 1e-10);
    }
    // The deflators reproduce it within simulation error, and the
    // deflated bonds are martingales: E[D(t) P(t, t + m)] = P(0, t + m).
    for years in [1usize, 5, 10] {
        let k = 12 * years;
        let t = years as f64;
        deflator_prices(&paths, k, initial.discount(t), 0.0);
        for (j, &m) in maturities.iter().enumerate().step_by(3) {
            let (mean, se) = column(SIMS, |s| {
                paths.deflator(s, k).unwrap() * paths.bond_price(s, k, j).unwrap()
            });
            let target = initial.discount(t + m);
            assert!((mean - target).abs() < 4.0 * se, "{t} + {m}: {mean} vs {target}");
        }
    }
    // A Svensson curve serves as well.
    let svensson = Svensson::new([0.045, -0.02, -0.01, 0.015], [1.5, 9.0]).unwrap();
    let model = HullWhite::new(0.08, 0.011, InitialCurve::Svensson(svensson)).unwrap();
    let paths = Esg::new(model).with_step(Step::Quarterly).simulate(5, SIMS, 4).unwrap();
    deflator_prices(&paths, 20, svensson.discount(5.0), 0.0);
    // A simulated curve converts to a DiscountCurve with the same prices.
    let c = paths.curve(17, 8).unwrap();
    let dc = DiscountCurve::spot(&c.maturities, &c.rates).unwrap();
    for (j, &m) in c.maturities.iter().enumerate() {
        assert!((dc.factor(m) - paths.bond_price(17, 8, j).unwrap()).abs() < 1e-12);
    }
}

#[test]
fn a_svensson_fit_recovers_its_curve_to_a_basis_point() {
    // A Treasury-like shape: an inverted short end and a hump.
    let truth = Svensson::new([0.048, -0.008, -0.025, 0.02], [1.2, 12.0]).unwrap();
    let maturities: Vec<f64> = (1..=30).map(f64::from).collect();
    let yields: Vec<f64> = maturities.iter().map(|&m| truth.spot(m)).collect();
    let fit = Svensson::fit(&maturities, &yields).unwrap();
    for m in [0.25, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0] {
        assert!((fit.spot(m) - truth.spot(m)).abs() < 1e-4, "{m}: {} vs {}", fit.spot(m), truth.spot(m));
    }
    // With noise of a basis point the fit stays within two.
    let noisy: Vec<f64> = yields
        .iter()
        .enumerate()
        .map(|(k, y)| y + 1e-4 * if k % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let fit = Svensson::fit(&maturities, &noisy).unwrap();
    for &m in &maturities {
        assert!((fit.spot(m) - truth.spot(m)).abs() < 2e-4, "{m}");
    }
}
