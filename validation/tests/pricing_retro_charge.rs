//! The insurance charge and savings of `prospicio_pricing::retro` (#981):
//! a compound Poisson-gamma aggregate on a grid by FFT against the same
//! law's closed series (a Tweedie), the exponential's closed form, and a
//! Table L from an FFT aggregate with a per-occurrence limit against its
//! definition.

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

#[test]
fn table_l_of_an_fft_aggregate_with_a_per_occurrence_limit() {
    use prospicio_prob::{Distribution, Severity};

    // Four claims expected, exponential severity of mean 20 on a step of
    // 1, each claim limited to 50.
    let claim = Gamma::new(1.0, 20.0).unwrap();
    let (severity, _) = Grid::local_moment(&claim, 1.0, 400).unwrap();
    let (limited_severity, exact) = severity.map(|x| x.min(50.0)).unwrap();
    assert!(exact);
    let count = Poisson::new(4.0).unwrap();
    let (unlimited, _) = fft(&count, &severity, 4_096).unwrap();
    let (limited, report) = fft(&count, &limited_severity, 4_096).unwrap();
    assert!(report.aliasing_error < 1e-12 && report.tail_mass < 1e-12);

    let e = unlimited.mean();
    let table_m = InsuranceCharge::from_severity(unlimited).unwrap();
    let limited_table_m = InsuranceCharge::from_severity(limited.clone()).unwrap();
    let table_l = limited_table_m.clone().table_l(e).unwrap();

    // k is the severity's share above the limit: a compound's mean is the
    // count's times the severity's.
    let k = 1.0 - limited_severity.mean() / severity.mean();
    assert!((table_l.loss_elimination() - k).abs() < 1e-12);
    // And close to the exponential's exp(-50 / 20), within the grid's step.
    assert!((k - (-2.5f64).exp()).abs() < 1e-3);

    for r in [0.0, 0.2, 0.5, 0.9, 1.0, 1.3, 2.0, 3.5, 8.0] {
        // The definition, summed over the limited aggregate's grid.
        let above: f64 = limited
            .probs()
            .iter()
            .enumerate()
            .map(|(j, p)| p * (limited.x(j) - r * e).max(0.0))
            .sum();
        assert!((table_l.charge(r) - (above / e + k)).abs() < 1e-12, "r = {r}");
        assert!((table_l.charge(r) - table_l.savings(r) - (1.0 - r)).abs() < 1e-12);
        // A limit never lowers the charge at an entry ratio.
        assert!(table_l.charge(r) >= table_m.charge(r) - 1e-12, "r = {r}");
        // The limited Table M is the same curve read at r E / E*, less k.
        let r_limited = r * e / limited_table_m.expected();
        let scaled = limited_table_m.charge(r_limited) * (1.0 - k) + k;
        assert!((table_l.charge(r) - scaled).abs() < 1e-12, "r = {r}");
    }
    assert!((table_l.charge(50.0) - k).abs() < 1e-12);
    assert_eq!(limited_table_m.loss_elimination(), 0.0);
    assert!(limited.stop_loss(0.0) > 0.0);
}
