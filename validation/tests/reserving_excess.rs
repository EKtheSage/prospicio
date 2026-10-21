//! Excess-layer development from a ground-up triangle
//! (`docs/design/reserving-excess.md`, PR 1).
//!
//! * Bloom and Patrik, "A Methodology for Avoiding the Pitfalls of Excess
//!   Loss Development", CAS E-Forum Fall 2013, pp. 4-8: four claims with
//!   ground-up factors 3.60 and 1.25, a deductible of 350 and 5% inflation.
//!   `ExcessPattern::from_ground_up` reproduces the excess factors of Years
//!   One, Two and Three (printed to two decimals, so compared to the
//!   rounding; the exact values are computed here from the claims), the
//!   layer values 175, 3,010 and 4,112.5, and the limited factors.
//! * Their excess CDF from limited and unlimited factors and an ELF
//!   (section 4, 17.54), and Pinto and Gogol's layer identity (PCAS 74
//!   (1987), section 5, p. 248), as identities of the printed inputs.
//! * Sahasrabuddhe, "Claims Development by Layer", E-Forum Fall 2010,
//!   Example 1 (p. 18): the rebasing of a limited factor by the ratio of
//!   limited means at two exponential scales, 14.352. This checks the
//!   rebasing identity through `layer_at_shift`, not the solve for the
//!   shifts, which no source prints for a capped triangle.
//! * The capped solve by round trip: a triangle built from a known scale
//!   path gives back the scales.

use prospicio_prob::{Distribution, Gamma, Grid, Lognormal, Severity};
use prospicio_reserving::excess::{
    Basis, ExcessPattern, Layer, SeverityByMaturity, layer_at_shift,
};
use prospicio_reserving::{
    ChainLadder, DevelopmentColumn, Grain, Long, Month, PatternWeight, Tail, Triangle,
};

/// The four claims of Bloom and Patrik's example at 12 months.
const CLAIMS: [f64; 4] = [100.0, 375.0, 250.0, 500.0];

/// The ultimate claims 450, 1,687.5, 1,125 and 2,250 times `scale`, each
/// with probability 1/4, as a grid of step `112.5 scale` (every claim is a
/// point of the grid, so `lev` is exact).
fn ultimate(scale: f64) -> Grid {
    let mut probs = vec![0.0; 21];
    for j in [4, 15, 10, 20] {
        probs[j] = 0.25;
    }
    Grid::new(112.5 * scale, probs).unwrap()
}

/// One origin's ground-up triangle at 12, 24 and 36 months: the claims
/// times `scale`, developing by 3.60 and 1.25.
fn ground_up(scale: f64) -> Triangle {
    let total: f64 = CLAIMS.iter().sum::<f64>() * scale;
    Triangle::from_long(&Long {
        keys: &[],
        origin: &[Month::january(2020); 3],
        development: DevelopmentColumn::Age(&[12, 24, 36]),
        values: &[("incurred", &[total, total * 3.6, total * 4.5])],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

fn pattern(scale: f64, layer: Layer) -> ExcessPattern {
    ExcessPattern::from_ground_up(
        &ground_up(scale),
        "incurred",
        Basis::Incurred,
        None,
        &Tail::default(),
        layer,
        &SeverityByMaturity::scale_shift(Box::new(ultimate(scale))),
    )
    .unwrap()
}

/// The excess of `retention` of the four claims, each grown by `growth`
/// from 12 months, summed: the paper's excess triangle by arithmetic.
fn excess_sum(scale: f64, growth: f64, retention: f64) -> f64 {
    CLAIMS
        .iter()
        .map(|c| (c * scale * growth - retention).max(0.0))
        .sum()
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() <= tol, "{what}: got {got}, want {want}");
}

#[test]
fn bloom_patrik_the_excess_triangle_is_reproduced_by_arithmetic() {
    // The paper's excess of 350 triangle, as printed.
    for (scale, printed) in [
        (1.0, [175.0, 3010.0, 4113.0]),
        (1.05, [219.0, 3231.0, 4388.0]),
        (1.1025, [265.0, 3462.0, 4678.0]),
    ] {
        for (age, growth) in [1.0, 3.6, 4.5].into_iter().enumerate() {
            close(
                excess_sum(scale, growth, 350.0),
                printed[age],
                0.6,
                "triangle",
            );
        }
    }
}

#[test]
fn bloom_patrik_excess_factors_from_the_ground_up_triangle() {
    let layer = Layer::unlimited(350.0).unwrap();
    // (scale, printed 12:24, 24:36, 12:ult), as exact values then printed.
    let years = [
        (1.0, 17.20, 1.37, 23.50),
        (1.05, 14.77, 1.36, 20.06),
        (1.1025, 13.08, 1.35, 17.67),
    ];
    for (scale, p1224, p2436, p12ult) in years {
        let p = pattern(scale, layer);
        // The exact values from the claims, to rounding error.
        let cell = |growth| excess_sum(scale, growth, 350.0);
        close(p.layer_ldf[0], cell(3.6) / cell(1.0), 1e-11, "12:24 exact");
        close(p.layer_ldf[1], cell(4.5) / cell(3.6), 1e-11, "24:36 exact");
        close(p.layer_cdf[0], cell(4.5) / cell(1.0), 1e-11, "12:ult exact");
        // And the paper's two decimals.
        close(p.layer_ldf[0], p1224, 0.005, "12:24 printed");
        close(p.layer_ldf[1], p2436, 0.005, "24:36 printed");
        close(p.layer_cdf[0], p12ult, 0.005, "12:ult printed");
    }
    // The achieved values: Year One 17.2, 4112.5 / 3010 and 23.5, Year Two
    // 14.768, 1.3583, 20.06, Year Three 13.0797, 1.3511, 17.6719.
    let one = pattern(1.0, layer);
    close(one.layer_ldf[0], 17.2, 1e-11, "Year One 12:24");
    close(one.layer_ldf[1], 4112.5 / 3010.0, 1e-11, "Year One 24:36");
    close(one.layer_cdf[0], 23.5, 1e-11, "Year One 12:ult");
    let two = pattern(1.05, layer);
    close(two.layer_ldf[0], 14.768, 5e-4, "Year Two 12:24");
    close(two.layer_ldf[1], 1.3583, 5e-5, "Year Two 24:36");
    close(two.layer_cdf[0], 20.06, 5e-3, "Year Two 12:ult");
    let three = pattern(1.1025, layer);
    close(three.layer_ldf[0], 13.0797, 5e-5, "Year Three 12:24");
    close(three.layer_ldf[1], 1.3511, 5e-5, "Year Three 24:36");
    close(three.layer_cdf[0], 17.6719, 5e-5, "Year Three 12:ult");
    // The shifts are the ground-up factors to ultimate, and the pattern
    // carries no tail.
    for (got, want) in one.shifts.iter().zip([4.5, 1.25, 1.0]) {
        close(*got, want, 1e-12, "shift");
    }
    assert_eq!(one.layer_tail, 1.0);
}

#[test]
fn bloom_patrik_layer_values_and_limited_factors() {
    let layer = Layer::unlimited(350.0).unwrap();
    let sev = ultimate(1.0);
    // 4 claims at the maturity where the ultimate claims are divided by c:
    // 175, 3,010 and 4,112.5.
    for (c, want) in [(4.5, 175.0), (1.25, 3010.0), (1.0, 4112.5)] {
        close(
            4.0 * layer_at_shift(&sev, layer, c),
            want,
            1e-9,
            "layer value",
        );
    }
    // Limited to 350: 1,050 at 12 months and 1,400 at 24 and ultimate.
    let limited = Layer::new(0.0, 350.0).unwrap();
    let p = pattern(1.0, limited);
    close(p.layer_ldf[0], 1400.0 / 1050.0, 1e-12, "limited 12:24");
    close(p.layer_ldf[0], 1.3333, 5e-5, "limited 12:24 printed");
    close(p.layer_ldf[1], 1.0, 1e-12, "limited 24:36");
    close(
        4.0 * layer_at_shift(&sev, limited, 1.0),
        1400.0,
        1e-9,
        "limited ultimate",
    );
    // Year One's pattern on Year Three overstates its ultimate, which is
    // why the severity is trended and not the pattern (decision 5): the
    // 12-month excess of Year Three times Year One's 23.5.
    let year_three_12 = excess_sum(1.1025, 1.0, 350.0);
    let overstated = year_three_12 * pattern(1.0, layer).layer_cdf[0];
    let truth = excess_sum(1.1025, 4.5, 350.0);
    assert!(overstated / truth > 1.3);
}

#[test]
fn bloom_patrik_excess_cdf_from_limited_unlimited_and_the_elf() {
    // Section 4: the excess CDF of Year Three from the weighted limited
    // factor, the unlimited CDF 4.5 and the ELF, printed 17.54.
    let limited_at = |scale: f64, growth: f64| -> f64 {
        CLAIMS.iter().map(|c| (c * scale * growth).min(350.0)).sum()
    };
    let (y1, y2, y3) = (1.0, 1.05, 1.1025);
    let weighted_12_24 =
        (limited_at(y1, 3.6) + limited_at(y2, 3.6)) / (limited_at(y1, 1.0) + limited_at(y2, 1.0));
    let limited_cdf = weighted_12_24; // 24:36 is 1.00
    let limited_ult = limited_at(y3, 1.0) * limited_cdf;
    let unlimited_ult: f64 = CLAIMS.iter().sum::<f64>() * y3 * 4.5;
    let elf = 1.0 - limited_ult / unlimited_ult;
    let excess_cdf = elf / (1.0 / 4.5 - (1.0 - elf) / limited_cdf);
    close(excess_cdf, 17.54, 0.005, "17.54");
    // The same ELF through the printed, rounded inputs gives 17.59: the
    // paper's 17.54 needs the unrounded ones.
    let printed = 0.764 / (1.0 / 4.5 - (1.0 - 0.764) / 1.32);
    close(printed, 17.59, 0.005, "from rounded inputs");
}

#[test]
fn pinto_gogol_layer_identity() {
    // LDF(c-d) = (f(c) - f(d)) / (f(c) / CDF_c - f(d) / CDF_d), with f the
    // ultimate excess over the ultimate ground-up loss; the inputs are
    // printed to three decimals, so the printed outputs hold to 1e-3.
    let f = [0.484, 0.372, 0.240, 0.144, 0.076];
    let cdf = [3.150, 3.668, 4.485, 5.223, 6.081];
    let ldf = |c: usize, d: usize| (f[c] - f[d]) / (f[c] / cdf[c] - f[d] / cdf[d]);
    for (d, printed) in [(4, 2.891), (3, 2.697), (2, 2.437), (1, 2.144)] {
        close(ldf(0, d), printed, 1e-3, "Pinto and Gogol");
    }
    // The identity is the derived layer factor of the scale shift: a
    // layer between two retentions is the difference of two excess layers.
    let sev = Lognormal::new(9.0, 1.2).unwrap();
    let c = 3.0; // ground-up factor to ultimate
    let stop = |a: f64| layer_at_shift(&sev, Layer::unlimited(a).unwrap(), c);
    let ult = |a: f64| sev.stop_loss(a);
    let (a, b) = (50_000.0, 250_000.0);
    // Excess CDFs of each retention, f(a) = excess ultimate / ground-up.
    let cdf_a = ult(a) / stop(a);
    let cdf_b = ult(b) / stop(b);
    let f_a = ult(a) / sev.mean();
    let f_b = ult(b) / sev.mean();
    let identity = (f_a - f_b) / (f_a / cdf_a - f_b / cdf_b);
    let direct = (ult(a) - ult(b)) / (stop(a) - stop(b));
    close(identity, direct, 1e-12, "identity on a lognormal");
    let from_pattern = ExcessPattern::from_ground_up(
        &Triangle::from_long(&Long {
            keys: &[],
            origin: &[Month::january(2020); 2],
            development: DevelopmentColumn::Age(&[12, 24]),
            values: &[("incurred", &[100.0, 100.0 * c])],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap(),
        "incurred",
        Basis::Incurred,
        None,
        &Tail::default(),
        Layer::new(a, b - a).unwrap(),
        &SeverityByMaturity::scale_shift(Box::new(sev)),
    )
    .unwrap();
    close(
        from_pattern.layer_cdf[0],
        direct,
        1e-9,
        "the layer 200k xs 50k",
    );
}

#[test]
fn sahasrabuddhe_example_1_rebasing() {
    // E4 at row 10, j = 1: 12.291 [LEV_280000(1m) / LEV_280000(500k)] /
    // [LEV_28138(1m) / LEV_28138(500k)], exponential severities whose
    // scale at age 1 is 28,138 and at ultimate 280,000: a shift of
    // 280000 / 28138. Printed 14.352.
    let sev = Gamma::new(1.0, 280_000.0).unwrap();
    let c = 280_000.0 / 28_138.0;
    let limited = |u: f64, shift: f64| layer_at_shift(&sev, Layer::new(0.0, u).unwrap(), shift);
    let e4 = 12.291 * (limited(1e6, 1.0) / limited(5e5, 1.0)) / (limited(1e6, c) / limited(5e5, c));
    close(e4, 14.352, 5e-4, "Sahasrabuddhe E4");
    // The same factors by the exponential's limited mean.
    let lev = |theta: f64, x: f64| theta * (1.0 - (-x / theta).exp());
    let direct = 12.291 * (lev(280_000.0, 1e6) / lev(280_000.0, 5e5))
        / (lev(28_138.0, 1e6) / lev(28_138.0, 5e5));
    close(e4, direct, 1e-9, "against the closed form");
}

#[test]
fn the_capped_solve_gives_back_the_scales() {
    // A ground-up triangle capped at 200k per claim, built from a known
    // scale path: the solve recovers it. No source prints such an example.
    let sev = Lognormal::new(9.0, 1.4).unwrap();
    let cap = 200_000.0;
    let truth = [3.0, 1.5, 1.1, 1.0];
    let capped: Vec<f64> = truth.iter().map(|&c| sev.lev(c * cap) / c).collect();
    let tri = Triangle::from_long(&Long {
        keys: &[],
        origin: &[Month::january(2020); 4],
        development: DevelopmentColumn::Age(&[12, 24, 36, 48]),
        values: &[("incurred", &capped)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    let layer = Layer::new(100_000.0, 400_000.0).unwrap();
    let p = ExcessPattern::from_ground_up(
        &tri,
        "incurred",
        Basis::Incurred,
        None,
        &Tail::default(),
        layer,
        &SeverityByMaturity::scale_shift_capped(Box::new(sev), cap),
    )
    .unwrap();
    for (got, want) in p.shifts.iter().zip(truth) {
        close(got / want, 1.0, 1e-9, "shift");
    }
    // An uncapped column would read the same capped data as less
    // development: the cap matters.
    let uncapped = ExcessPattern::from_ground_up(
        &tri,
        "incurred",
        Basis::Incurred,
        None,
        &Tail::default(),
        layer,
        &SeverityByMaturity::scale_shift(Box::new(sev)),
    )
    .unwrap();
    assert!(uncapped.shifts[0] < 0.95 * truth[0]);
}

#[test]
fn a_sparse_layer_triangle_runs_on_the_derived_pattern() {
    // The use the pattern is for: a layer triangle whose first age is
    // empty, fitted with the derived factors fully selected.
    let layer = Layer::unlimited(350.0).unwrap();
    let p = pattern(1.0, layer);
    let tri = Triangle::from_long(&Long {
        keys: &[],
        origin: &[
            Month::january(2020),
            Month::january(2020),
            Month::january(2020),
            Month::january(2021),
            Month::january(2021),
            Month::january(2022),
        ],
        development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
        values: &[("layer", &[0.0, 3010.0, 4112.5, 0.0, 3231.0, 0.0])],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    // The triangle's own 12:24 factor does not exist (every origin is zero
    // at 12 months): without the pattern it is an error.
    assert!(ChainLadder::default().fit(&tri, "layer").is_err());
    let development = prospicio_reserving::Development {
        pattern: Some(p.selected(PatternWeight::All(1.0))),
        ..Default::default()
    };
    let cl = ChainLadder {
        development,
        ..Default::default()
    }
    .fit(&tri, "layer")
    .unwrap();
    close(cl.ldf()[0], 17.2, 1e-11, "selected 12:24");
    close(cl.ldf()[1], 4112.5 / 3010.0, 1e-11, "selected 24:36");
    // Origin 2020 is at 36 months, 2021 at 24 and 2022 at 12 (zero, so its
    // ultimate is zero: a BF prior is the way to fill it).
    close(
        cl.ultimate[1],
        3231.0 * 4112.5 / 3010.0,
        1e-9,
        "2021 ultimate",
    );
}
