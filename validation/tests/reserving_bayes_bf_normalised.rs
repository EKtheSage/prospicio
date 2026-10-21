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
//!   a-priori loss ratios the fit's mean is within 1.5% of the closed-form
//!   strong-prior limit (the shares `S_d / (M_d + lambda)` at the prior
//!   ultimates) and its gap to the classical reserve, which grows with the
//!   distance of the loss ratio from the chain ladder's, within 2 points of
//!   decision 22's figures;
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

    // Other loss ratios: the closed-form limit, and the gap to the
    // classical reserve.
    for (apriori, expected) in [(0.35, -0.086), (0.45, -0.011), (0.55, 0.102), (0.65, 0.284)] {
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
        let limit = strong_prior_reserve(&tri, apriori);
        eprintln!(
            "apriori {apriori}: {got:.0} vs classical {classical:.0} ({:+.1}%), \
             closed-form limit {limit:.0} ({:+.2}%)",
            100.0 * (got / classical - 1.0),
            100.0 * (got / limit - 1.0)
        );
        // The fit reaches the closed form of the strong-prior limit (the
        // shares S_d / (M_d + lambda) at the prior ultimates), not the
        // classical reserve: within 1.5% (seed 0: +0.01%, -0.22%, -0.90%
        // and -0.67%, the posterior mean of the shares below their mode,
        // as the 0.25% at the chain ladder's ultimates above).
        assert!(
            (got / limit - 1.0).abs() < 0.015,
            "apriori {apriori}: {got} vs the closed-form limit {limit}"
        );
        // The gap to the classical reserve has the sign and rough size
        // reported in decision 22 (-8.6%, -1.1%, +10.2%, +28.4%).
        let gap = got / classical - 1.0;
        assert!(
            (gap - expected).abs() < 0.02,
            "apriori {apriori}: gap {gap} vs about {expected}"
        );
    }
}

/// The strong-prior limit's total reserve: the ultimates at their prior
/// `m_w = apriori x premium_w` and the shares `p_d = S_d / (M_d + lambda)`
/// that maximise the likelihood given them, `S_d` the observed increments
/// of column `d`, `M_d` the prior ultimates of the origins observed there,
/// `lambda` the root of `sum p_d = 1` (by bisection). The reserve is
/// `sum_w m_w` times the shares of `w`'s unobserved ages.
fn strong_prior_reserve(tri: &Triangle, apriori: f64) -> f64 {
    let [_, _, n_origins, n_dev] = tri.shape();
    let paid = tri.column_position("paid").unwrap();
    let premium = tri.column_position("premium").unwrap();
    let prior: Vec<f64> = (0..n_origins)
        .map(|w| apriori * tri.get(0, premium, w, 0).unwrap())
        .collect();
    let (mut s, mut m) = (vec![0.0; n_dev], vec![0.0; n_dev]);
    for (w, &m_w) in prior.iter().enumerate() {
        for d in 0..n_dev {
            if let Some(v) = tri.get(0, paid, w, d) {
                let before = if d == 0 {
                    0.0
                } else {
                    tri.get(0, paid, w, d - 1).unwrap()
                };
                s[d] += v - before;
                m[d] += m_w;
            }
        }
    }
    let excess = |lambda: f64| s.iter().zip(&m).map(|(s, m)| s / (m + lambda)).sum::<f64>() - 1.0;
    let floor = m.iter().cloned().fold(f64::INFINITY, f64::min);
    let (mut lo, mut hi) = (-floor * (1.0 - 1e-12), floor);
    while excess(hi) > 0.0 {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if excess(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let lambda = 0.5 * (lo + hi);
    let share: Vec<f64> = s.iter().zip(&m).map(|(s, m)| s / (m + lambda)).collect();
    (0..n_origins)
        .map(|w| {
            let future: f64 = (0..n_dev)
                .filter(|&d| tri.get(0, paid, w, d).is_none())
                .map(|d| share[d])
                .sum();
            prior[w] * future
        })
        .sum()
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
