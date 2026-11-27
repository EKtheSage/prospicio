//! The general liability experience ratio plan (`prospicio_pricing::emod`,
//! #981). ISO's plan and its tables are licensed and no public text works
//! an exhibit through, so this checks the plan's arithmetic and the
//! properties every reading of it shares
//! (`knowledge/references/iso-gl-experience-rating.md`).

use prospicio_pricing::emod::{ExperienceRatioPlan, LiabilityYear};

fn years() -> Vec<LiabilityYear> {
    vec![
        LiabilityYear::new(150_000.0, 0.15)
            .unwrap()
            .with_detrend_factor(0.93)
            .unwrap()
            .with_occurrence(60_000.0, 12_000.0)
            .unwrap()
            .with_occurrence(400_000.0, 50_000.0)
            .unwrap(),
        LiabilityYear::new(160_000.0, 0.35)
            .unwrap()
            .with_detrend_factor(0.97)
            .unwrap()
            .with_occurrence(25_000.0, 2_500.0)
            .unwrap(),
        LiabilityYear::new(170_000.0, 0.65).unwrap(),
    ]
}

#[test]
fn the_worksheet_sums() {
    let x = ExperienceRatioPlan::new(250_000.0, 0.70, 0.35)
        .unwrap()
        .modification(&years())
        .unwrap();
    // 72,000 and 450,000 held to the MSL of 250,000; then 27,500.
    assert_eq!(x.years[0].reported, 322_000.0);
    assert_eq!(x.years[1].reported, 27_500.0);
    assert_eq!(x.reported, 349_500.0);
    let unreported = 150_000.0 * 0.93 * 0.70 * 0.15
        + 160_000.0 * 0.97 * 0.70 * 0.35
        + 170_000.0 * 0.70 * 0.65;
    assert!((x.expected_unreported - unreported).abs() < 1e-9);
    assert_eq!(x.subject_loss_cost, 480_000.0);
    let aer = (349_500.0 + unreported) / 480_000.0;
    assert!((x.actual_ratio - aer).abs() < 1e-14);
    assert!((x.modification - (1.0 + 0.35 * (aer - 0.70) / 0.70)).abs() < 1e-14);
    assert!((x.credit_debit - (x.modification - 1.0)).abs() < 1e-15);
}

#[test]
fn the_modification_is_linear_in_the_credibility() {
    let at = |z: f64| {
        ExperienceRatioPlan::new(250_000.0, 0.70, z)
            .unwrap()
            .modification(&years())
            .unwrap()
            .modification
    };
    assert_eq!(at(0.0), 1.0);
    let slope = at(1.0) - 1.0;
    for z in [0.1, 0.25, 0.5, 0.9] {
        assert!((at(z) - (1.0 + z * slope)).abs() < 1e-14);
    }
}

#[test]
fn a_higher_msl_never_lowers_the_modification() {
    let mut last = f64::NEG_INFINITY;
    for msl in [10_000.0, 50_000.0, 100_000.0, 250_000.0, 1e6, f64::INFINITY] {
        let m = ExperienceRatioPlan::new(msl, 0.70, 0.35)
            .unwrap()
            .modification(&years())
            .unwrap()
            .modification;
        assert!(m >= last);
        last = m;
    }
}
