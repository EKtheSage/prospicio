//! The workers compensation split plan (`prospicio_pricing::emod`, #981)
//! against NCCI's "ABCs of Experience Rating", whose worksheet (Exhibits G
//! and H) and examples (Exhibits A and B) are transcribed here.
//!
//! The worksheet prints whole dollars, so its amounts are matched within
//! half a dollar. Its four class rows are one policy of the experience
//! period and do not add up to the summary's expected losses, and its claim
//! detail differs from the summary by 775, so the modification is checked
//! from the summary's totals (`knowledge/references/ncci-abcs-experience-rating.md`).

use prospicio_pricing::emod::{Claim, ClassExposure, SplitPlan};

fn near(got: f64, printed: f64) {
    assert!((got - printed).abs() <= 0.5, "{got} is not {printed} to the dollar");
}

fn plan() -> SplitPlan {
    // The split point "in most states", and the example's per-claim
    // accident limitation.
    SplitPlan::new(18_000.0)
        .unwrap()
        .with_per_claim_limit(250_000.0)
        .unwrap()
}

#[test]
fn exhibit_g_class_rows() {
    // Class, ELR, D-ratio, payroll; expected and expected primary losses.
    let rows = [
        ("8288", 3.50, 0.27, 250_000.0, 8_750.0, 2_363.0),
        ("8380", 0.96, 0.33, 3_025_350.0, 29_043.0, 9_584.0),
        ("8748", 0.31, 0.33, 1_645_650.0, 5_102.0, 1_684.0),
        ("8810", 0.11, 0.35, 3_000_000.0, 3_300.0, 1_155.0),
    ];
    let exposures: Vec<_> = rows
        .iter()
        .map(|&(class, elr, d, payroll, ..)| ClassExposure::new(class, payroll, elr, d).unwrap())
        .collect();
    let x = plan().modification(&exposures, &[], 0.14, 47_400.0).unwrap();
    for (row, &(_, _, d, _, expected, primary)) in x.exposures.iter().zip(&rows) {
        near(row.expected, expected);
        assert!((row.expected_primary - row.expected * d).abs() < 1e-9);
        // The worksheet rounds a row's expected losses to the dollar before
        // it applies the D-ratio: 5,102 x 0.33 = 1,683.66 prints 1,684,
        // where the unrounded 5,101.515 x 0.33 is 1,683.4995.
        near(row.expected.round() * d, primary);
    }
}

#[test]
fn exhibit_h_modification_from_the_summary_totals() {
    // E = 176,190 and E_p = 56,172 as one row; A_p = 68,584 and
    // A_e = 31,985 from claim 1700001 (49,985: 18,000 primary) and claims
    // under the split point; W = 0.14, B = 47,400.
    let exposures = [ClassExposure::new("all", 100.0, 176_190.0, 56_172.0 / 176_190.0).unwrap()];
    let claims = [49_985.0, 17_359.0, 17_000.0, 16_225.0].map(|a| Claim::new(a).unwrap());
    let x = plan().modification(&exposures, &claims, 0.14, 47_400.0).unwrap();
    near(x.expected_excess, 120_018.0);
    near(x.actual_primary, 68_584.0);
    near(x.actual_excess, 31_985.0);
    near(x.stabilizing, 150_615.0);
    near(x.weight * x.actual_excess, 4_478.0);
    near(x.weight * x.expected_excess, 16_803.0);
    near(x.adjusted_actual, 223_677.0);
    near(x.adjusted_expected, 223_590.0);
    assert_eq!((x.modification * 100.0).round() / 100.0, 1.00);
    assert!((x.modification - 223_677.38 / 223_590.0).abs() < 1e-12);
}

#[test]
fn exhibits_a_and_b_split_and_medical_only() {
    let claims = [
        Claim::new(500_000.0).unwrap(),
        Claim::new(100_000.0).unwrap(),
        Claim::new(5_000.0).unwrap(),
        Claim::new(20_000.0).unwrap().medical_only(),
    ];
    let exposures = [ClassExposure::new("8810", 1_000_000.0, 1.0, 0.3).unwrap()];
    let x = plan().modification(&exposures, &claims, 0.2, 10_000.0).unwrap();
    let got: Vec<_> = x.claims.iter().map(|r| (r.primary, r.excess)).collect();
    let want = [(18_000.0, 232_000.0), (18_000.0, 82_000.0), (5_000.0, 0.0), (5_400.0, 600.0)];
    for (g, w) in got.iter().zip(want) {
        assert!((g.0 - w.0).abs() < 1e-9 && (g.1 - w.1).abs() < 1e-9, "{g:?} is not {w:?}");
    }
}
