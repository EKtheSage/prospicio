//! One-year view parity: Merz and Wüthrich's claims development result
//! against R ChainLadder's `CDR(MackChainLadder(tri), dev = "all")` on RAA,
//! GenIns, ABC and the two Merz–Wüthrich example triangles.
//!
//! Every row of `reference/reserving_cdr_r.csv` is evaluated; see
//! `scripts/reserving_cdr_r.R`. R's `Mack.S.E.` column is checked twice:
//! against Mack's standard error, and against the full run-off of the
//! claims development result, which adds up to it.

use std::collections::HashMap;

use prospicio_reserving::{
    ClaimsDevelopmentResult, Development, Mack, MackFit, Period, SigmaInterpolation,
};
use prospicio_validation::{Case, check, reference, triangle};

/// Development estimator for a reference `method`: volume-weighted factors,
/// with R's default or Mack's rule for the last sigma.
fn development(method: &str) -> Development {
    let sigma_interpolation = match method {
        "cdr" => SigmaInterpolation::LogLinear,
        "cdr_sigma_mack" => SigmaInterpolation::Mack,
        other => panic!("unknown method {other}"),
    };
    Development {
        sigma_interpolation,
        ..Default::default()
    }
}

/// Fits each (dataset, method) once and answers reference cases from it.
#[derive(Default)]
struct Fits {
    fits: HashMap<(String, String), (MackFit, ClaimsDevelopmentResult)>,
}

impl Fits {
    fn get(&mut self, dataset: &str, method: &str) -> &(MackFit, ClaimsDevelopmentResult) {
        self.fits
            .entry((dataset.to_string(), method.to_string()))
            .or_insert_with(|| {
                let mack = Mack {
                    development: development(method),
                    ..Default::default()
                }
                .fit(&triangle(dataset), "values")
                .unwrap_or_else(|e| panic!("{dataset} {method}: {e}"));
                let cdr = mack
                    .claims_development_result()
                    .unwrap_or_else(|e| panic!("{dataset} {method}: {e}"));
                (mack, cdr)
            })
    }

    /// The value of a case; `run_off` answers `Mack.S.E.` rows from the
    /// claims development result instead of from Mack.
    fn eval(&mut self, case: &Case, run_off: bool) -> Option<f64> {
        let (mack, cdr) = self.get(case.get("dataset"), case.get("method"));
        let origin = |year: &str| {
            let year: i32 = year.parse().ok()?;
            cdr.origins.iter().position(|&p| p == Period::year(year))
        };
        let arg = case.get("arg");
        // R reports one year per age; years past the run-off are zero.
        let year = |k: &str| -> Option<Option<usize>> {
            let k: usize = k.parse().ok()?;
            (k >= 1).then(|| (k <= cdr.by_calendar_year.len()).then(|| k - 1))
        };
        match case.get("quantity") {
            "reserve" => Some(mack.chain_ladder.reserves()[origin(arg)?]),
            "total_reserve" => Some(mack.chain_ladder.total_reserve()),
            "one_year_se" => Some(cdr.one_year_standard_error[origin(arg)?]),
            "total_one_year_se" => Some(cdr.total_one_year_standard_error),
            "cdr_se" => {
                let (o, k) = arg.split_once(':')?;
                let o = origin(o)?;
                Some(year(k)?.map_or(0.0, |t| cdr.by_calendar_year[t][o]))
            }
            "total_cdr_se" => Some(year(arg)?.map_or(0.0, |t| cdr.total_by_calendar_year[t])),
            "mack_se" if run_off => Some(cdr.run_off_standard_error()[origin(arg)?]),
            "mack_se" => Some(mack.standard_error[origin(arg)?]),
            "total_mack_se" if run_off => Some(cdr.total_run_off_standard_error()),
            "total_mack_se" => Some(mack.total_standard_error),
            _ => None,
        }
    }
}

#[test]
fn matches_r_cdr() {
    let mut fits = Fits::default();
    check(&reference("reserving_cdr_r.csv"), |c| fits.eval(c, false));
}

#[test]
fn run_off_adds_up_to_r_mack_standard_error() {
    let mut fits = Fits::default();
    let cases: Vec<Case> = reference("reserving_cdr_r.csv")
        .into_iter()
        .filter(|c| matches!(c.get("quantity"), "mack_se" | "total_mack_se"))
        .collect();
    check(&cases, |c| fits.eval(c, true));
}

#[test]
fn mw_dataset_shapes() {
    // R ChainLadder: sum(getLatestCumulative(MW2008)), likewise MW2014.
    for (name, n, latest_total) in [("mw2008", 9, 30_986_807.0), ("mw2014", 17, 429_117.0)] {
        let tri = triangle(name);
        assert_eq!(tri.shape(), [1, 1, n, n], "{name}");
        assert_eq!(tri.origins()[0], Period::year(2001), "{name}");
        let latest: f64 = tri.latest_diagonal().values(0, 0).iter().sum();
        assert_eq!(latest, latest_total, "{name}");
    }
}
