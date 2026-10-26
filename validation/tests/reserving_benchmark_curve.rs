//! Benchmark-curve parity: `BenchmarkCurve` against
//! `scipy.optimize.least_squares` (`reserving-v02.md`, decision 20).
//!
//! The reference values come from `scripts/reserving_benchmark_curve_scipy.py`
//! (scipy 1.18.1, numpy 2.5.4; MINPACK's Levenberg-Marquardt, `method="lm"`,
//! on `(ln omega, ln theta)` with `ftol = xtol = gtol = 1e-15` and residuals
//! `sqrt(w) * (p - G(c(a)) / G(c(m)))`), printed and pasted here. Both
//! searches find the same least-squares minimum, so parameters agree to
//! about 1e-7 relative (our Nelder-Mead runs to 1e-12 on the objective and
//! the log parameters) and the minimized sum of squares to about 1e-9.

use prospicio_reserving::{BenchmarkCurve, GrowthCurve, PatternKind};

const AGES: [f64; 10] = [12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0, 120.0];
const PERCENT: [f64; 10] = [
    0.18, 0.47, 0.66, 0.78, 0.855, 0.90, 0.93, 0.951, 0.965, 0.975,
];
const WEIGHTS: [f64; 10] = [1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5];

struct Case {
    name: &'static str,
    settings: BenchmarkCurve,
    weighted: bool,
    omega: f64,
    theta: f64,
    sse: f64,
}

#[test]
fn benchmark_curve_matches_scipy_least_squares() {
    let cases = [
        Case {
            name: "loglogistic",
            settings: BenchmarkCurve::default(),
            weighted: false,
            omega: 2.09195678319105,
            theta: 25.5547370205669,
            sse: 0.000775555676218485,
        },
        Case {
            name: "weibull_shift_max",
            settings: BenchmarkCurve {
                curve: GrowthCurve::Weibull,
                max_age: Some(240.0),
                origin_width: Some(12.0),
            },
            weighted: false,
            omega: 1.00079725545721,
            theta: 28.350802255576,
            sse: 0.000447194375150825,
        },
        Case {
            name: "loglogistic_weighted_shift",
            settings: BenchmarkCurve {
                origin_width: Some(12.0),
                ..BenchmarkCurve::default()
            },
            weighted: true,
            omega: 1.6187953142168,
            theta: 18.9910413059938,
            sse: 0.00480838215217264,
        },
    ];
    let values: Vec<(f64, f64)> = AGES.iter().copied().zip(PERCENT).collect();
    for c in cases {
        let weights = c.weighted.then_some(&WEIGHTS[..]);
        let fit = c
            .settings
            .fit(PatternKind::PercentDeveloped, &values, weights)
            .unwrap();
        let rel = |a: f64, b: f64| (a / b - 1.0).abs();
        assert!(
            rel(fit.omega, c.omega) < 1e-7,
            "{}: omega {}",
            c.name,
            fit.omega
        );
        assert!(
            rel(fit.theta, c.theta) < 1e-7,
            "{}: theta {}",
            c.name,
            fit.theta
        );
        assert!(
            rel(fit.sum_of_squares, c.sse) < 1e-9,
            "{}: sse {}",
            c.name,
            fit.sum_of_squares
        );
    }
}
