//! The insurance charge and savings of `prospicio_pricing::retro` (#981):
//! a compound Poisson-gamma aggregate on a grid by FFT against the same
//! law's closed series (a Tweedie), and the exponential's closed form.

use prospicio_aggregate::fft;
use prospicio_pricing::retro::InsuranceCharge;
use prospicio_prob::{Gamma, Grid, Poisson, Tweedie};

#[test]
fn the_charge_of_an_fft_aggregate_matches_the_tweedie_series() {
    // Three claims expected, gamma severity of shape 2 and scale 10:
    // expected aggregate 60.
    let law = Tweedie::from_poisson_gamma(3.0, 2.0, 10.0).unwrap();
    let (severity, _) = Grid::local_moment(&law.severity(), 0.01, 30_000).unwrap();
    let (aggregate, report) = fft(&Poisson::new(3.0).unwrap(), &severity, 100_000).unwrap();
    assert!(report.aliasing_error < 1e-12 && report.tail_mass < 1e-12);

    let on_grid = InsuranceCharge::from_severity(aggregate).unwrap();
    let exact = InsuranceCharge::from_severity(law).unwrap();
    assert!((on_grid.expected() / 60.0 - 1.0).abs() < 1e-9);
    assert!((exact.expected() / 60.0 - 1.0).abs() < 1e-12);
    for r in [0.0, 0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0] {
        // Local moment matching keeps the grid's limited means, so the
        // charge agrees far inside the step's 1e-2.
        assert!(
            (on_grid.charge(r) - exact.charge(r)).abs() < 1e-7,
            "charge at {r}: {} against {}",
            on_grid.charge(r),
            exact.charge(r)
        );
        assert!((on_grid.savings(r) - exact.savings(r)).abs() < 1e-7, "savings at {r}");
        assert!((on_grid.charge(r) - on_grid.savings(r) - (1.0 - r)).abs() < 1e-12);
    }
    // No claim in a year has probability exp(-3): the savings at a small
    // entry ratio are that share of it.
    assert!((exact.savings(1e-6) / 1e-6 - (-3.0f64).exp()).abs() < 1e-4);
}

#[test]
fn an_exponential_aggregate_has_the_charge_exp_of_minus_r() {
    let table = InsuranceCharge::from_severity(Gamma::new(1.0, 4_000.0).unwrap()).unwrap();
    for row in table.table(&[0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0]) {
        let r = row.entry_ratio;
        assert!((row.charge - (-r).exp()).abs() < 1e-14, "r = {r}");
        assert!((row.savings - ((-r).exp() + r - 1.0)).abs() < 1e-14, "r = {r}");
    }
}
