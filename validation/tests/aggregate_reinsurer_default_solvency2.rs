//! The Solvency II counterparty default module for type 1 exposures
//! (`prospicio_aggregate::solvency2`, #980, `docs/design/reinsurer-default.md`
//! decision (k)) against `reinsurer_default_solvency2.csv`, which
//! `validation/scripts/aggregate_reinsurer_default.py` computes from the text
//! of Delegated Regulation (EU) 2015/35, Arts 192, 199, 200 and 201, not
//! from the Rust. Plain arithmetic on the inputs, held to 1e-12 relative.

use prospicio_aggregate::solvency2;
use prospicio_validation::{Case, check, reference};

/// A type 1 portfolio from its `params`: `n` exposures `(pd_i, lgd_i)`.
fn exposures(case: &Case) -> Vec<(f64, f64)> {
    let at = |name: &str| case.param("params", name);
    (1..=at("n") as usize)
        .map(|i| (at(&format!("pd_{i}")), at(&format!("lgd_{i}"))))
        .collect()
}

#[test]
fn solvency2_matches_the_articles() {
    let cases = reference("reinsurer_default_solvency2.csv");
    assert_eq!(cases.len(), 78);
    check(&cases, |case| {
        let arg = case.get("arg");
        let at = |name: &str| case.param("params", name);
        let value = match (case.get("distribution"), case.get("quantity")) {
            ("probability_by_step", "probability") => {
                solvency2::probability_by_step(arg.parse().unwrap()).unwrap()
            }
            ("probability_by_ratio", "probability") => {
                solvency2::probability_by_ratio(arg.parse().unwrap()).unwrap()
            }
            ("reinsurance_lgd", "lgd") => solvency2::reinsurance_lgd(
                at("recoverables"),
                at("risk_mitigation"),
                at("collateral"),
                at("lgd_share"),
                at("factor"),
            )
            .unwrap(),
            ("type1", quantity) => {
                let t = solvency2::type1(&exposures(case)).unwrap();
                match quantity {
                    "v_inter" => t.v_inter,
                    "v_intra" => t.v_intra,
                    "sigma" => t.sigma,
                    "total_lgd" => t.total_lgd,
                    "scr" => t.scr,
                    _ => return None,
                }
            }
            _ => return None,
        };
        Some(value)
    });
}
