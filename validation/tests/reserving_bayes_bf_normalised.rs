//! The Bayesian Bornhuetter–Ferguson with a normalised pattern
//! (`BayesBfNormalised`, `docs/design/reserving-bayes.md`, decision 22).
//!
//! There is no Stan reference file for this model (no CmdStan in the
//! environments that built it), so its checks are the model's own limits and
//! the contrast with `BayesBf` that motivated it (issue #410,
//! `knowledge/findings/bayes-bf-apriori-scale.md`):
//!
//! * the flat prior (`cv` infinite) reproduces the Bayesian chain ladder
//!   (`BayesOdp`) on GenIns with premium and on RAA with a synthetic
//!   premium: each origin's and the total's mean within `0.1 s` and sd
//!   within 10% (15% on RAA, whose oldest reserves are heavy-tailed), `s`
//!   the sd of `BayesOdp`'s reserve;
//! * a tight prior at the chain ladder's own ultimates gives the classical
//!   `BornhuetterFerguson` reserve (the fit's mean within 0.5%); at other
//!   a-priori loss ratios the gap to the classical reserve, which grows with
//!   the distance of the loss ratio from the chain ladder's, is reported
//!   (the closed form of the shares is the unit tests');
//! * the a-priori loss ratio is a real prior on the ultimate: on RAA with a
//!   premium, the total reserve of `BayesBfNormalised` rises with
//!   `apriori` at every `cv`, and between 0.3 and 1.2 by at least a factor
//!   2 at `cv` 0.05, where `BayesBf`'s moves by under 2% (the finding's
//!   table);
//! * the one-year view agrees with `BayesOdp`'s under a flat prior and
//!   narrows as the prior tightens;
//! * the defaults pass every MCMC threshold.

use std::sync::OnceLock;
use std::time::Instant;

use prospicio_prob::{Distribution, PredictiveDistribution};
use prospicio_reserving::bayes::{
    BayesBf, BayesBfNormalised, BayesBfNormalisedFit, BayesOneYear, OneYearRefit,
};
use prospicio_reserving::{
    BayesOdp, BornhuetterFerguson, ChainLadder, DevelopmentColumn, Grain, Long, Month, Period,
    ProcessDistribution, Triangle,
};
use prospicio_validation::{triangle, triangle_columns};

/// GenIns with R ChainLadder's `ClarkCapeCod` premium.
fn genins() -> Triangle {
    triangle_columns("genins_premium", &["paid", "premium"])
}

/// `tri`'s first column as `paid`, with a `premium` column that is
/// `premium(origin)` at every cell of the origin.
fn with_premium(tri: &Triangle, first_year: i32, premium: impl Fn(usize) -> f64) -> Triangle {
    let [_, _, n_origins, n_dev] = tri.shape();
    let (mut origin, mut age, mut value, mut prem) = (vec![], vec![], vec![], vec![]);
    for w in 0..n_origins {
        for d in 0..n_dev {
            if let Some(v) = tri.get(0, 0, w, d) {
                origin.push(Month::january(first_year + w as i32));
                age.push(tri.development()[d]);
                value.push(v);
                prem.push(premium(w));
            }
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("paid", &value), ("premium", &prem)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .expect("a triangle with premium")
}

/// RAA with a premium of 40,000 in 1981 rising by 2,000 a year, as in the
/// finding's table.
fn raa_premium() -> Triangle {
    with_premium(&triangle("raa"), 1981, |w| 40_000.0 + 2_000.0 * w as f64)
}

fn normalised(apriori: f64, cv: f64, seed: u64) -> BayesBfNormalised {
    BayesBfNormalised {
        apriori,
        cv,
        seed,
        ..BayesBfNormalised::default()
    }
}

fn fit(tri: &Triangle, model: &BayesBfNormalised) -> BayesBfNormalisedFit {
    let start = Instant::now();
    let fit = model.fit(tri, "paid", "premium").expect("fits");
    eprintln!(
        "BayesBfNormalised apriori {} cv {} seed {}: fit in {:.2?}; {:?}",
        model.apriori,
        model.cv,
        model.seed,
        start.elapsed(),
        fit.diagnostics
    );
    fit
}

/// The reserve of origin `year` or, for `None`, the total.
fn pick(r: &PredictiveDistribution, key: Option<i32>) -> impl Distribution {
    match key {
        Some(y) => r.marginal(&vec![Period::year(y).into()]).expect("origin"),
        None => r.total().clone(),
    }
}

/// The flat prior against `BayesOdp` by origin and in total.
fn flat_against_bayes_odp(tri: &Triangle, first: i32, last: i32, sd_tol: f64) {
    let bf = fit(tri, &normalised(0.5, f64::INFINITY, 0));
    let odp = BayesOdp {
        seed: 1000,
        ..BayesOdp::default()
    }
    .fit(tri, "paid")
    .expect("fits");
    let mut keys: Vec<Option<i32>> = (first + 1..=last).map(Some).collect();
    keys.push(None);
    for key in keys {
        let (a, b) = (pick(&bf.reserves, key), pick(&odp.reserves, key));
        let s = b.std_dev();
        assert!(
            (a.mean() - b.mean()).abs() < 0.1 * s,
            "{key:?}: mean {} vs {} (sd {s})",
            a.mean(),
            b.mean()
        );
        assert!(
            (a.std_dev() / s - 1.0).abs() < sd_tol,
            "{key:?}: sd {} vs {s}",
            a.std_dev()
        );
    }
}

#[test]
fn a_flat_prior_is_the_bayesian_chain_ladder_on_genins() {
    flat_against_bayes_odp(&genins(), 2001, 2010, 0.10);
}

#[test]
fn a_flat_prior_is_the_bayesian_chain_ladder_on_raa() {
    // RAA's oldest reserves rest on one or two cells and have heavy-tailed
    // draws: over seeds 0 to 3 the 1982 reserve's sd (one future cell) came
    // out 5% to 11% above `BayesOdp`'s, `BayesBf`'s own 7% below to 5% above,
    // so the sd tolerance is 15%.
    flat_against_bayes_odp(&raa_premium(), 1981, 1990, 0.15);
}

/// A tight prior at the chain ladder's ultimates gives the classical
/// reserve; at other loss ratios the closed form, and the gap to the
/// classical reserve is printed.
#[test]
fn a_tight_prior_is_the_bornhuetter_ferguson_limit() {
    let tri = genins();
    let cl = ChainLadder::default().fit(&tri, "paid").unwrap();
    // The chain ladder's ultimates as the prior ultimates (exposure = the
    // ultimates, apriori 1).
    let at_cl = with_premium(&tri, 2001, |w| cl.ultimate[w]);
    let classical = BornhuetterFerguson {
        apriori: 1.0,
        ..Default::default()
    }
    .fit(&at_cl, "paid", "premium")
    .unwrap()
    .total_reserve();
    assert!((classical / cl.total_reserve() - 1.0).abs() < 1e-9);
    let tight = BayesBfNormalised {
        apriori: 1.0,
        cv: 1e-3,
        development_sd: 100.0,
        process: ProcessDistribution::None,
        ..BayesBfNormalised::default()
    };
    let f = fit(&at_cl, &tight);
    let got = f.reserves.total().mean();
    eprintln!("at the chain ladder's ultimates: {got:.0} vs classical {classical:.0}");
    assert!(
        (got / classical - 1.0).abs() < 0.005,
        "{got} vs {classical}"
    );

    // Other loss ratios: the gap to the classical reserve.
    for apriori in [0.35, 0.45, 0.55, 0.65] {
        let model = BayesBfNormalised {
            apriori,
            ..tight.clone()
        };
        let f = fit(&tri, &model);
        let got = f.reserves.total().mean();
        let classical = BornhuetterFerguson {
            apriori,
            ..Default::default()
        }
        .fit(&tri, "paid", "premium")
        .unwrap()
        .total_reserve();
        eprintln!(
            "apriori {apriori}: {got:.0} vs classical {classical:.0} ({:+.1}%)",
            100.0 * (got / classical - 1.0)
        );
        assert!(got > 0.0);
    }
}

/// The finding's table, for the new model: the reserve follows `apriori`.
#[test]
fn the_apriori_is_a_real_prior() {
    let tri = raa_premium();
    let total = |apriori: f64, cv: f64| {
        let new = fit(&tri, &normalised(apriori, cv, 1))
            .reserves
            .total()
            .mean();
        let old = BayesBf {
            apriori,
            cv,
            seed: 1,
            ..BayesBf::default()
        }
        .fit(&tri, "paid", "premium")
        .unwrap()
        .reserves
        .total()
        .mean();
        (new, old)
    };
    eprintln!("RAA with premium, total mean reserve (BayesBfNormalised, BayesBf):");
    for cv in [0.5, 0.2, 0.05] {
        let row: Vec<(f64, f64)> = [0.3, 0.6, 1.2].iter().map(|&a| total(a, cv)).collect();
        eprintln!("cv {cv}: {row:.0?}");
        // The new model rises with the prior's level.
        assert!(
            row[0].0 < row[1].0 && row[1].0 < row[2].0,
            "cv {cv}: {row:?}"
        );
        if cv == 0.05 {
            assert!(row[2].0 > 2.0 * row[0].0, "cv {cv}: {row:?}");
            // BayesBf's level is nearly inert.
            let spread = row.iter().map(|r| r.1).fold(0.0, f64::max)
                / row.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
            assert!(spread < 1.02, "cv {cv}: {row:?}");
        }
    }
}

/// The defaults pass every MCMC threshold on the reference triangles, over
/// a flat, a moderate and a tight prior.
#[test]
fn the_defaults_pass_the_diagnostics() {
    static TRIANGLES: OnceLock<[(&str, Triangle); 2]> = OnceLock::new();
    let triangles = TRIANGLES.get_or_init(|| [("genins", genins()), ("raa", raa_premium())]);
    for (name, tri) in triangles {
        for cv in [f64::INFINITY, 0.2, 0.01] {
            let f = fit(tri, &normalised(0.5, cv, 0));
            let d = &f.diagnostics;
            assert!(d.warnings.is_empty(), "{name} cv {cv}: {d:?}");
            assert_eq!(d.divergences, 0, "{name} cv {cv}: {d:?}");
        }
    }
}

/// The one-year view (decision 17) under a flat prior is the Bayesian chain
/// ladder's, and narrows as the prior tightens: the total claims
/// development result's sd against `BayesOdp::one_year`'s, on GenIns with
/// premium.
#[test]
fn the_one_year_view_narrows_with_the_prior() {
    let tri = genins();
    let settings = BayesOneYear {
        refit: Some(OneYearRefit::ReweightOnly),
        ..BayesOneYear::default()
    };
    let odp = BayesOdp::default()
        .one_year(&tri, "paid", &settings)
        .expect("fits")
        .cdr
        .total()
        .std_dev();
    let ratio = |cv: f64| {
        let view = normalised(0.5, cv, 0)
            .one_year(&tri, "paid", "premium", &settings)
            .expect("fits");
        view.cdr.total().std_dev() / odp
    };
    let (flat, mid, tight) = (ratio(f64::INFINITY), ratio(0.2), ratio(0.05));
    eprintln!(
        "one-year total sd over BayesOdp's: flat {flat:.3}, cv 0.2 {mid:.3}, cv 0.05 {tight:.3}"
    );
    assert!((flat - 1.0).abs() < 0.15, "flat {flat}");
    assert!(mid < flat && tight < mid, "{flat} {mid} {tight}");
}
