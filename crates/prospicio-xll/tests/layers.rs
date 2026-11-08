//! Excess-layer development and the benchmark curve as Excel reaches them
//! (`docs/design/xll.md` decision 58): `PRO.RES.EXCESS.PATTERN` equal to
//! `ExcessPattern::from_ground_up` under the scale shift,
//! `PRO.RES.EXCESS.SHAPE` at Pinto and Gogol's printed 5.0 and 10^0.3, and
//! `PRO.RES.BENCHMARK.CURVE` at scipy's least-squares fit
//! (`validation/tests/reserving_benchmark_curve.rs`).

mod common;

use common::{fails, n, ok, raa, row, t};
use prospicio_prob::Lognormal;
use prospicio_reserving::Tail;
use prospicio_reserving::excess::{Basis, ExcessPattern, Layer, SeverityByMaturity};
use prospicio_xll::reserving::{COLUMN, tail, triangle};
use prospicio_xll::sheet::{Grid, Value};

fn none() -> Grid {
    Grid::missing()
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol * want.abs().max(1.0),
        "{what}: got {got}, want {want}"
    );
}

fn numbers(g: &Grid) -> Vec<f64> {
    g.cells
        .iter()
        .map(|v| match v {
            Value::Number(x) => *x,
            other => panic!("expected a number, got {other:?}"),
        })
        .collect()
}

fn excess_pattern(args: &[Grid]) -> Grid {
    ok("pro_res_excess_pattern", args.to_vec())
}

#[test]
fn the_scale_shift_equals_the_library() {
    let lib = ExcessPattern::from_ground_up(
        &triangle(&raa()).unwrap(),
        COLUMN,
        Basis::Incurred,
        None,
        &Tail::default(),
        Layer::new(25_000.0, 50_000.0).unwrap(),
        &SeverityByMaturity::scale_shift(Box::new(Lognormal::new(9.0, 1.5).unwrap())),
    )
    .unwrap();
    let args = [
        raa(),
        n(25_000.0),
        n(50_000.0),
        t("lognormal"),
        n(9.0),
        n(1.5),
        none(),
        none(),
        none(),
        none(),
        t("ldf"),
    ];
    let ldf = numbers(&excess_pattern(&args));
    assert_eq!(ldf.len(), lib.layer_ldf.len());
    for (a, b) in ldf.iter().zip(&lib.layer_ldf) {
        close(*a, *b, 1e-12, "layer ldf");
    }
    let mut cdf_args = args.clone();
    cdf_args[10] = t("cdf");
    let cdf = numbers(&excess_pattern(&cdf_args));
    for (a, b) in cdf.iter().zip(&lib.layer_cdf) {
        close(*a, *b, 1e-12, "layer cdf");
    }
    // The summary: a header, one row per age and the tail row.
    let mut summary_args = args.clone();
    summary_args[10] = none();
    let s = excess_pattern(&summary_args);
    assert_eq!(s.rows, lib.ages.len() + 2);
    assert_eq!(*s.get(0, 2), Value::from("Layer LDF"));
    close(
        common::num(s.get(1, 1)),
        lib.ground_up_ldf[0],
        1e-12,
        "ground-up ldf",
    );
    close(common::num(s.get(1, 4)), lib.shifts[0], 1e-12, "shift");
    assert_eq!(*s.get(s.rows - 1, 0), Value::from("Tail"));
    // The layer develops faster than the ground-up losses.
    assert!(ldf[0] > common::num(s.get(1, 1)));
}

#[test]
fn a_tail_and_a_cap_reach_the_library() {
    let lib = ExcessPattern::from_ground_up(
        &triangle(&raa()).unwrap(),
        COLUMN,
        Basis::Incurred,
        None,
        &tail(&n(1.05)).unwrap(),
        Layer::unlimited(50_000.0).unwrap(),
        &SeverityByMaturity::scale_shift_capped(
            Box::new(Lognormal::new(9.0, 1.5).unwrap()),
            250_000.0,
        ),
    )
    .unwrap();
    let ldf = numbers(&excess_pattern(&[
        raa(),
        n(50_000.0),
        none(),
        t("lognormal"),
        n(9.0),
        n(1.5),
        none(),
        none(),
        n(1.05),
        n(250_000.0),
        t("ldf"),
    ]));
    assert!(lib.layer_tail > 1.0);
    let mut want = lib.layer_ldf.clone();
    want.push(lib.layer_tail);
    assert_eq!(ldf.len(), want.len());
    for (a, b) in ldf.iter().zip(&want) {
        close(*a, *b, 1e-12, "capped ldf with tail");
    }
}

/// One origin at two reports: the base develops by `d`.
fn two_reports(d: f64) -> Grid {
    row(&[1.0, d])
}

#[test]
fn shape_drift_matches_pinto_and_gogol() {
    // p. 247: 2.5 (4)^0.5 = 5.0 from the losses excess of 25,000 ...
    let ldf = |tri: Grid, base: Grid| {
        numbers(&ok(
            "pro_res_excess_shape",
            vec![
                tri,
                n(100_000.0),
                none(),
                n(25_000.0),
                row(&[1.75, 1.25]),
                n(1.25),
                base,
                none(),
                none(),
                t("ldf"),
            ],
        ))
    };
    close(ldf(two_reports(2.5), n(25_000.0))[0], 5.0, 1e-12, "losses");
    // ... and from the counts above it.
    close(
        ldf(two_reports(2.5 * 0.25 / 0.75), none())[0],
        5.0,
        1e-12,
        "counts",
    );
    close(
        ldf(two_reports(2.5 * 0.25 / 0.75), t("counts"))[0],
        5.0,
        1e-12,
        "counts by name",
    );
    // Appendix B: the losses excess of 100,000 develop 10^0.3 times those
    // excess of 10,000.
    let a = 3.0;
    let got = numbers(&ok(
        "pro_res_excess_shape",
        vec![
            two_reports(a),
            n(100_000.0),
            none(),
            n(10_000.0),
            row(&[1.6, 1.3]),
            n(1.3),
            n(10_000.0),
            none(),
            none(),
            t("ldf"),
        ],
    ))[0];
    close(got / a, 10f64.powf(0.3), 1e-12, "Appendix B");
}

const AGES: [f64; 10] = [12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0, 120.0];
const PERCENT: [f64; 10] = [
    0.18, 0.47, 0.66, 0.78, 0.855, 0.90, 0.93, 0.951, 0.965, 0.975,
];
const WEIGHTS: [f64; 10] = [1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5];

fn labelled(out: &Grid, label: &str) -> f64 {
    let r = (0..out.rows)
        .find(|&r| *out.get(r, 0) == Value::from(label))
        .unwrap_or_else(|| panic!("no row {label}"));
    common::num(out.get(r, 1))
}

#[test]
fn the_benchmark_curve_matches_scipy() {
    // (curve, weighted, max_age, origin_width, omega, theta, sse) from
    // scipy.optimize.least_squares.
    let cases = [
        (
            "loglogistic",
            false,
            None,
            None,
            2.09195678319105,
            25.5547370205669,
            0.000775555676218485,
        ),
        (
            "weibull",
            false,
            Some(240.0),
            Some(12.0),
            1.00079725545721,
            28.350802255576,
            0.000447194375150825,
        ),
        (
            "loglogistic",
            true,
            None,
            Some(12.0),
            1.6187953142168,
            18.9910413059938,
            0.00480838215217264,
        ),
    ];
    for (curve, weighted, max_age, width, omega, theta, sse) in cases {
        let opt = |x: Option<f64>| x.map_or_else(none, n);
        let fit = ok(
            "pro_res_benchmark_curve",
            vec![
                row(&AGES),
                row(&PERCENT),
                none(),
                t(curve),
                if weighted { row(&WEIGHTS) } else { none() },
                opt(max_age),
                opt(width),
                none(),
                none(),
            ],
        );
        close(labelled(&fit, "Omega"), omega, 1e-7, "omega");
        close(labelled(&fit, "Theta"), theta, 1e-7, "theta");
        close(labelled(&fit, "Sum of squares"), sse, 1e-8, "sse");
        assert_eq!(*fit.get(0, 2), Value::from("Fitted"));
    }
    // The pattern and its factors to ultimate at other ages.
    let at = |output: &str| {
        numbers(&ok(
            "pro_res_benchmark_curve",
            vec![
                row(&AGES),
                row(&PERCENT),
                t("percent"),
                none(),
                none(),
                none(),
                none(),
                row(&[18.0, 30.0, 240.0]),
                t(output),
            ],
        ))
    };
    let p = at("pattern");
    let c = at("cdf");
    assert_eq!(p.len(), 3);
    for (p, c) in p.iter().zip(&c) {
        close(p * c, 1.0, 1e-12, "cdf is 1 / percent");
    }
    assert!(p[0] < p[1] && p[1] < p[2] && p[2] < 1.0);
}

#[test]
fn bad_inputs_are_errors() {
    let e = fails(
        "pro_res_benchmark_curve",
        vec![
            row(&AGES),
            row(&PERCENT[..9]),
            none(),
            none(),
            none(),
            none(),
            none(),
            none(),
            none(),
        ],
    );
    assert!(e.contains("as many cells"), "{e}");
    let e = fails(
        "pro_res_benchmark_curve",
        vec![
            row(&AGES),
            row(&PERCENT),
            none(),
            t("gompertz"),
            none(),
            none(),
            none(),
            none(),
            none(),
        ],
    );
    assert!(e.contains("curve"), "{e}");
    let e = fails(
        "pro_res_excess_shape",
        vec![
            two_reports(2.0),
            n(100_000.0),
            none(),
            n(25_000.0),
            row(&[1.5]),
            n(1.2),
            none(),
            none(),
            none(),
            none(),
        ],
    );
    assert!(e.contains("one per triangle column"), "{e}");
    let e = fails(
        "pro_res_excess_pattern",
        vec![
            raa(),
            n(25_000.0),
            none(),
            t("lognormal"),
            n(9.0),
            n(1.5),
            none(),
            none(),
            none(),
            none(),
            t("table"),
        ],
    );
    assert!(e.contains("output"), "{e}");
}

#[test]
fn counts_reach_the_library() {
    // RAA's own cells as a stand-in count triangle (any developing counts).
    let counts = raa();
    let tri =
        prospicio_xll::counts::measures(&[(COLUMN, &raa()), ("counts", &counts)], &[]).unwrap();
    let lib = ExcessPattern::from_ground_up(
        &tri,
        COLUMN,
        Basis::Incurred,
        Some("counts"),
        &Tail::default(),
        Layer::unlimited(50_000.0).unwrap(),
        &SeverityByMaturity::scale_shift(Box::new(Lognormal::new(9.0, 1.5).unwrap())),
    )
    .unwrap();
    let ldf = numbers(&excess_pattern(&[
        raa(),
        n(50_000.0),
        none(),
        t("lognormal"),
        n(9.0),
        n(1.5),
        none(),
        counts,
        none(),
        none(),
        t("ldf"),
    ]));
    for (a, b) in ldf.iter().zip(&lib.layer_ldf) {
        close(*a, *b, 1e-12, "ldf with counts");
    }
}
