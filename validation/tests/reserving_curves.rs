//! Development curves over every age (`DevelopmentCurve`,
//! `docs/design/reserving-v02.md` decision 31, issue #556).
//!
//! * Every row of `reference/reserving_curves_r.csv`: the exponential,
//!   inverse power and Hoerl curves by R's `lm` on `ln(f - 1)`, Sherman's
//!   and the growth curves by R's `nls` (the port algorithm), on the
//!   volume-weighted factors of RAA and GenIns; produced by
//!   `Rscript validation/scripts/reserving_curves_r.R` (R 4.5.3, ChainLadder
//!   0.2.21).
//! * chainladder-python 0.10.1's `TailCurve(attachment_age=60)`, the
//!   exponential attached early, from `reference/reserving_tails_python.csv`
//!   (`scripts/reserving_tails_python.py`): the curve kept from 60 months on,
//!   given to the chain ladder as a selected pattern with the curve's own
//!   tail, gives the same factors, tail and ultimates.

use std::collections::HashMap;

use prospicio_reserving::{
    ChainLadder, ChainLadderFit, CurveShape, Development, DevelopmentCurve, DevelopmentCurveFit,
    GrowthCurve, Period, Tail,
};
use prospicio_validation::{Case, check, reference, triangle};

fn shape(method: &str) -> CurveShape {
    match method {
        "exponential" => CurveShape::Exponential,
        "inverse_power" => CurveShape::InversePower,
        "hoerl" => CurveShape::Hoerl,
        "sherman" => CurveShape::Sherman,
        "loglogistic" => CurveShape::Growth(GrowthCurve::LogLogistic),
        "weibull" => CurveShape::Growth(GrowthCurve::Weibull),
        other => panic!("unknown method {other}"),
    }
}

fn param_index(quantity: &str) -> usize {
    match quantity {
        "param_a" | "param_omega" => 0,
        "param_b" | "param_theta" => 1,
        "param_c" => 2,
        other => panic!("unknown parameter {other}"),
    }
}

#[test]
fn matches_r_lm_and_nls() {
    let mut fits: HashMap<(String, String), DevelopmentCurveFit> = HashMap::new();
    check(&reference("reserving_curves_r.csv"), |c| {
        let (dataset, method, quantity) = (c.get("dataset"), c.get("method"), c.get("quantity"));
        let fit = fits
            .entry((dataset.into(), method.into()))
            .or_insert_with(|| {
                let dev = Development::default()
                    .fit(&triangle(dataset), "values")
                    .unwrap();
                DevelopmentCurve {
                    shape: shape(method),
                    ..Default::default()
                }
                .fit(&dev)
                .unwrap_or_else(|e| panic!("{dataset} {method}: {e}"))
            });
        match quantity {
            q if q.starts_with("param_") => fit.parameters.get(param_index(q)).copied(),
            "fitted" => fit.fitted.get(c.number("arg")? as usize).copied(),
            "sum_of_squares" => Some(fit.sum_of_squares),
            "r_squared" => Some(fit.r_squared),
            "tail" => Some(fit.tail(100)),
            _ => None,
        }
    });
}

/// The chain ladder with the exponential curve from 60 months on as its
/// selected pattern and the curve's tail.
fn curve_chain_ladder(dataset: &str) -> ChainLadderFit {
    let tri = triangle(dataset);
    let dev = Development::default().fit(&tri, "values").unwrap();
    let curve = DevelopmentCurve {
        keep_observed_until: Some(60),
        ..Default::default()
    }
    .fit(&dev)
    .unwrap();
    ChainLadder {
        development: Development {
            pattern: Some(curve.pattern()),
            ..Default::default()
        },
        tail: Tail::from(curve.tail(100)),
    }
    .fit(&tri, "values")
    .unwrap()
}

#[test]
fn matches_chainladder_python_tail_curve_attached_early() {
    let cases: Vec<Case> = reference("reserving_tails_python.csv")
        .into_iter()
        .filter(|c| c.get("method") == "tail_curve_attach")
        .filter(|c| {
            // Mack's risks differ by design: a selected factor carries no
            // parameter error. Factors past the oldest age are the tail's
            // spread; only their product, the tail factor, is the curve's.
            matches!(
                c.get("quantity"),
                "ldf" | "tail_factor" | "ultimate" | "reserve" | "total_ultimate" | "total_reserve"
            )
        })
        .collect();
    let mut fits: HashMap<String, ChainLadderFit> = HashMap::new();
    let mut checked = Vec::new();
    for c in cases {
        let dataset = c.get("dataset").to_string();
        let fit = fits
            .entry(dataset.clone())
            .or_insert_with(|| curve_chain_ladder(&dataset));
        if c.get("quantity") == "ldf" && c.number("arg").unwrap() as usize >= fit.ldf().len() {
            continue;
        }
        checked.push(c);
    }
    check(&checked, |c| {
        let fit = &fits[c.get("dataset")];
        let origin = || {
            let year = c.number("arg")? as i32;
            fit.origins.iter().position(|&p| p == Period::year(year))
        };
        match c.get("quantity") {
            "ldf" => fit.ldf().get(c.number("arg")? as usize).copied(),
            "tail_factor" => Some(fit.tail.factor),
            "ultimate" => Some(fit.ultimate[origin()?]),
            "reserve" => Some(fit.reserves()[origin()?]),
            "total_ultimate" => Some(fit.total_ultimate()),
            "total_reserve" => Some(fit.total_reserve()),
            _ => None,
        }
    });
}

/// `compare` fits every shape to the same points: on RAA and GenIns the
/// two- and three-parameter shapes order as R's sums of squares do.
#[test]
fn compare_gives_one_row_per_shape() {
    for dataset in ["raa", "genins"] {
        let dev = Development::default()
            .fit(&triangle(dataset), "values")
            .unwrap();
        let shapes = [
            CurveShape::Exponential,
            CurveShape::InversePower,
            CurveShape::Hoerl,
            CurveShape::Sherman,
            CurveShape::Growth(GrowthCurve::LogLogistic),
            CurveShape::Growth(GrowthCurve::Weibull),
        ];
        let table = DevelopmentCurve::default().compare(&dev, &shapes).unwrap();
        assert_eq!(table.len(), shapes.len());
        for (row, shape) in table.iter().zip(shapes) {
            assert_eq!(row.curve.shape, shape);
            assert_eq!(row.n_points, dev.ldf.len());
        }
        // Hoerl nests the exponential and the inverse power on the log
        // scale, and fits the factors better than either on both books.
        assert!(table[2].sum_of_squares < table[0].sum_of_squares);
        assert!(table[2].sum_of_squares < table[1].sum_of_squares);
    }
}
