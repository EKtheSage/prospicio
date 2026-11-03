//! Merz and Wüthrich's (2010) paid-incurred chain against R ChainLadder's
//! `PaidIncurredChain` on its `USAApaid` and `USAAincurred` example and on
//! Dahms' (2008) triangles, the data of Merz and Wüthrich's example
//! (`scripts/reserving_paid_incurred_r.R`).
//!
//! R returns only the totals and the origins after the first; the script
//! also writes its intermediates (the variances, the incurred weights, the
//! posterior means and standard deviations) and each origin's term of the
//! msep sum, so that every stage is compared. R's `Ult.Loss` leaves out the
//! oldest origin, which prospicio includes at its last paid value.
//!
//! R leaves out the coupling of `Psi_0` with `Phi_1, ..., Phi_J` in the
//! posterior precision, so the parity runs under
//! `PicConvention::RChainLadder`; the default, the model's posterior, is
//! checked against direct conditioning in `prospicio-reserving`'s unit
//! tests, and its distance from R's is recorded below.

use prospicio_reserving::{PaidIncurredChain, PicConvention, PicFit};
use prospicio_validation::{Case, check, reference, triangle_columns};

fn fit(dataset: &str, convention: PicConvention) -> PicFit {
    let tri = triangle_columns(dataset, &["paid", "incurred"]);
    PaidIncurredChain {
        convention,
        ..Default::default()
    }
    .fit(&tri, "paid", "incurred")
    .unwrap_or_else(|e| panic!("{dataset}: {e}"))
}

fn eval(fits: &[(&str, PicFit)], case: &Case) -> Option<f64> {
    let (_, fit) = fits.iter().find(|(n, _)| *n == case.get("dataset"))?;
    let arg = case.number("arg");
    let period = || arg.map(|a| a as usize);
    let origin = || {
        let year = arg? as i32;
        fit.origins
            .iter()
            .position(|o| o.to_string() == year.to_string())
    };
    let n = fit.ages.len();
    let sd = fit.posterior_sd();
    match case.get("quantity") {
        "sigma" => fit.sigma.get(period()?).copied(),
        "tau" => fit.tau.get(period()?).copied(),
        // R's beta_j belongs to the origin whose latest period is j.
        "beta" => fit.incurred_weight.get(n - 1 - period()?).copied(),
        "phi" => fit.phi().get(period()?).copied(),
        "psi" => fit.psi().get(period()?).copied(),
        "phi_sd" => sd.get(period()?).copied(),
        "psi_sd" => sd.get(n + period()?).copied(),
        "ultimate" => Some(fit.ultimate[origin()?]),
        "reserve" => Some(fit.reserve[origin()?]),
        "std_err" => Some(fit.std_err[origin()?]),
        "ultimate_after_first" => Some(fit.total_ultimate() - fit.ultimate[0]),
        "total_reserve" => Some(fit.total_reserve()),
        "total_std_err" => Some(fit.total_std_err),
        _ => None,
    }
}

#[test]
fn matches_r_chainladder() {
    let r = PicConvention::RChainLadder;
    let fits = [("usaa", fit("usaa", r)), ("dahms", fit("dahms", r))];
    check(&reference("reserving_paid_incurred_r.csv"), |case| {
        eval(&fits, case)
    });
}

#[test]
fn on_usaa_the_ultimates_lie_between_the_paid_and_incurred_chain_ladders() {
    // A property the paper discusses rather than states: on USAA the PIC
    // ultimate of every origin after the first lies between the volume
    // weighted paid and incurred chain ladders'. Recorded here so that a
    // change in it is noticed; it is not a property of the model.
    use prospicio_reserving::ChainLadder;
    let tri = triangle_columns("usaa", &["paid", "incurred"]);
    let pic = fit("usaa", PicConvention::Paper);
    let paid = ChainLadder::default().fit(&tri, "paid").unwrap().ultimate;
    let incurred = ChainLadder::default()
        .fit(&tri, "incurred")
        .unwrap()
        .ultimate;
    for o in 1..pic.ultimate.len() {
        let (lo, hi) = (paid[o].min(incurred[o]), paid[o].max(incurred[o]));
        assert!(
            pic.ultimate[o] >= lo && pic.ultimate[o] <= hi,
            "origin {o}: {} outside [{lo}, {hi}]",
            pic.ultimate[o]
        );
    }
}

#[test]
fn the_models_posterior_is_close_to_rs() {
    // The distance between the two conventions (reserving-v02.md decision
    // 29): the model's posterior moves the total reserve and its standard
    // error by about 0.5% on USAA (1,604,986 and 111,519 against R's
    // 1,596,954 and 110,977) and by about 1.1% and 1.7% on Dahms' data.
    for (dataset, reserve, se) in [("usaa", 0.0050, 0.0049), ("dahms", 0.0114, 0.0168)] {
        let paper = fit(dataset, PicConvention::Paper);
        let r = fit(dataset, PicConvention::RChainLadder);
        let rel = |a: f64, b: f64| a / b - 1.0;
        let dr = rel(paper.total_reserve(), r.total_reserve());
        let ds = rel(paper.total_std_err, r.total_std_err);
        assert!((dr - reserve).abs() < 1e-4, "{dataset}: reserve {dr}");
        assert!((ds - se).abs() < 1e-4, "{dataset}: standard error {ds}");
    }
}
