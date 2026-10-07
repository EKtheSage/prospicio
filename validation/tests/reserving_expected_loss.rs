//! Expected-loss parity: expected loss, Bornhuetter–Ferguson, Benktander and
//! Cape Cod against chainladder-python on the CAS workers' compensation
//! triangle and GenIns with premium.
//!
//! Every row of `reference/reserving_expected_loss_python.csv` is evaluated;
//! see `scripts/reserving_expected_loss_python.py`. A reference `method` is
//! the method name followed by its settings, `name=value` separated by `;`.

use std::collections::HashMap;

use prospicio_reserving::{
    Average, Benktander, BornhuetterFerguson, CapeCod, ChainLadder, Development, ExpectedLoss,
    ExpectedLossFit, Period,
};
use prospicio_validation::{Case, check, reference, triangle_columns};

/// A fitted reference case: the expected-loss fit and, for Cape Cod, the
/// trended apriori.
struct Fitted {
    fit: ExpectedLossFit,
    trended_apriori: Option<Vec<f64>>,
}

fn fit(dataset: &str, case: &Case) -> Fitted {
    let method = case.get("method");
    let average = match method.split(';').find_map(|p| p.strip_prefix("average=")) {
        Some("volume") => Average::Volume,
        Some("simple") => Average::Simple,
        other => panic!("{method}: unknown average {other:?}"),
    };
    let chain_ladder = ChainLadder {
        development: Development {
            average,
            ..Default::default()
        },
        ..Default::default()
    };
    let tri = triangle_columns(dataset, &["paid", "premium"]);
    let param = |name| case.param("method", name);
    let fail = |e: prospicio_reserving::Error| -> ! { panic!("{dataset} {method}: {e}") };
    let (fit, trended_apriori) = match method.split(';').next().unwrap() {
        "expected_loss" => {
            let m = ExpectedLoss {
                apriori: param("apriori"),
                chain_ladder,
            };
            (
                m.fit(&tri, "paid", "premium").unwrap_or_else(|e| fail(e)),
                None,
            )
        }
        "bornhuetter_ferguson" => {
            let m = BornhuetterFerguson {
                apriori: param("apriori"),
                chain_ladder,
            };
            (
                m.fit(&tri, "paid", "premium").unwrap_or_else(|e| fail(e)),
                None,
            )
        }
        "benktander" => {
            let m = Benktander {
                apriori: param("apriori"),
                n_iters: param("n_iters") as usize,
                chain_ladder,
            };
            (
                m.fit(&tri, "paid", "premium").unwrap_or_else(|e| fail(e)),
                None,
            )
        }
        "cape_cod" => {
            let m = CapeCod {
                trend: param("trend"),
                decay: param("decay"),
                chain_ladder,
            };
            let cc = m.fit(&tri, "paid", "premium").unwrap_or_else(|e| fail(e));
            (cc.expected_loss, Some(cc.trended_apriori))
        }
        other => panic!("unknown method {other}"),
    };
    Fitted {
        fit,
        trended_apriori,
    }
}

#[test]
fn matches_chainladder_python() {
    let mut fits: HashMap<(String, String), Fitted> = HashMap::new();
    check(&reference("reserving_expected_loss_python.csv"), |case| {
        let dataset = case.get("dataset");
        let key = (dataset.to_string(), case.get("method").to_string());
        let f = fits.entry(key).or_insert_with(|| fit(dataset, case));
        let origin = || {
            let year = case.number("arg")? as i32;
            f.fit
                .chain_ladder
                .origins
                .iter()
                .position(|&p| p == Period::year(year))
        };
        match case.get("quantity") {
            "ultimate" => Some(f.fit.ultimate[origin()?]),
            "reserve" => Some(f.fit.reserves()[origin()?]),
            "total_ultimate" => Some(f.fit.total_ultimate()),
            "total_reserve" => Some(f.fit.total_reserve()),
            "apriori" => Some(f.trended_apriori.as_ref()?[origin()?]),
            "detrended_apriori" => Some(f.fit.apriori[origin()?]),
            _ => None,
        }
    });
}

#[test]
fn dataset_shapes() {
    // Latest paid and premium totals of the two exposure datasets.
    for (name, first, latest_paid, latest_premium) in [
        ("clrd_wkcomp", 1988, 11_029_320.0, 21_946_490.0),
        ("genins_premium", 2001, 34_358_090.0, 118_000_000.0),
    ] {
        let tri = triangle_columns(name, &["paid", "premium"]);
        assert_eq!(tri.shape(), [1, 2, 10, 10], "{name}");
        assert_eq!(tri.origins()[0], Period::year(first), "{name}");
        let diag = tri.latest_diagonal();
        assert_eq!(diag.values(0, 0).iter().sum::<f64>(), latest_paid, "{name}");
        assert_eq!(
            diag.values(0, 1).iter().sum::<f64>(),
            latest_premium,
            "{name}"
        );
    }
}
