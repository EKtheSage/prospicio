//! A Bayesian Bornhuetter–Ferguson prior for an excess layer from the layer
//! share and its uncertainty (`LayerSharePrior`,
//! `docs/design/reserving-excess.md`, PR 4).
//!
//! The layer's triangle is GenIns with R ChainLadder's `ClarkCapeCod`
//! premium and a layer column that is a quarter of the paid values, the
//! layer share `s = 0.25` of the ground-up premium's loss ratio. The checks:
//!
//! * the prior's moments are exact: the fit's prior ultimate is the
//!   ground-up loss ratio times the share times the premium, and the model's
//!   `cv` is that of a product of independent variables;
//! * the posterior under a tight prior tends to decision 23's closed-form
//!   limit, applied to the layer values (the shares `S_d / (M_d + lambda)`
//!   at the layer's prior ultimates), within 1.5% in the mean;
//! * the share's uncertainty widens the predictive interval, as the prior's
//!   cv rises: the total reserve's sd rises with the share's cv, and the
//!   mean moves from the prior's implied level towards the data's;
//! * a flat share (`share_cv` infinite) gives the flat normalised BF;
//! * the share error is common to all origins by default (open decision
//!   23(a), `reserving-bayes.md` decision 25): the total's prior variance
//!   is the closed form with the covariance `cv_share^2 m_i m_j`, a vanishing
//!   share cv is the plain model, the total's interval is wider than under
//!   independent per-origin priors at share cvs 0.3, 0.6 and 1.2, and on a
//!   layer triangle generated with a wrong share the posterior of the common
//!   multiplier moves towards the truth.

use prospicio_prob::Distribution;
use prospicio_reserving::bayes::{BayesBfNormalised, LayerSharePrior, ShareDependence};
use prospicio_reserving::excess::{Layer, layer_share};
use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, ProcessDistribution, Triangle};
use prospicio_validation::triangle_columns;

const SHARE: f64 = 0.25;

/// GenIns with a `premium` column and a `layer` column, `SHARE` times paid.
fn layer_triangle() -> Triangle {
    let tri = triangle_columns("genins_premium", &["paid", "premium"]);
    let [_, _, n_origins, n_dev] = tri.shape();
    let (mut origin, mut age, mut layer, mut prem) = (vec![], vec![], vec![], vec![]);
    for w in 0..n_origins {
        for d in 0..n_dev {
            if let Some(v) = tri.get(0, 0, w, d) {
                origin.push(Month::january(2001 + w as i32));
                age.push(tri.development()[d]);
                layer.push(SHARE * v);
                prem.push(tri.get(0, 1, w, 0).unwrap());
            }
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("layer", &layer), ("premium", &prem)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .expect("a layer triangle")
}

fn base(seed: u64) -> BayesBfNormalised {
    BayesBfNormalised {
        seed,
        ..BayesBfNormalised::default()
    }
}

#[test]
fn the_prior_moments_are_exact() {
    let tri = layer_triangle();
    let prior = LayerSharePrior::new(0.5, 0.1, &[SHARE], 0.3).unwrap();
    // The cv of a product of independent variables, from raw moments.
    let raw = |m: f64, c: f64| m * m * (1.0 + c * c);
    let var = raw(0.5, 0.1) * raw(SHARE, 0.3) - (0.5 * SHARE).powi(2);
    assert!((prior.cv() - var.sqrt() / (0.5 * SHARE)).abs() < 1e-14);
    // The default is a share error common to all origins: the model's cv is
    // the loss ratio's and the share's is the multiplier's; per origin, the
    // one gamma carries the product's.
    let model = prior
        .model(&BayesBfNormalised {
            n_sims: 200,
            ..base(0)
        })
        .unwrap();
    assert_eq!(model.apriori, 0.5);
    assert_eq!((model.cv, model.common_cv), (0.1, Some(0.3)));
    let per = prior.clone().with_dependence(ShareDependence::PerOrigin);
    let per_model = per.model(&base(0)).unwrap();
    assert_eq!((per_model.cv, per_model.common_cv), (prior.cv(), None));
    let fit = prior.fit(&model, &tri, "layer", "premium").unwrap();
    for (w, &u) in fit.prior_ultimate.iter().enumerate() {
        let premium = tri.get(0, 1, w, 0).unwrap();
        assert!((u - 0.5 * SHARE * premium).abs() < 1e-9 * premium, "{w}");
    }
    // The total's prior variance, from first principles: with m_i the
    // origins' prior ultimates, the covariance between two origins is
    // cv_share^2 m_i m_j and each variance m_i^2 ((1 + c_e^2)(1 + c_s^2) - 1).
    let m = &fit.prior_ultimate;
    let (e2, s2) = (0.1_f64.powi(2), 0.3_f64.powi(2));
    let mut var = 0.0;
    for (i, mi) in m.iter().enumerate() {
        for (j, mj) in m.iter().enumerate() {
            var += if i == j {
                mi * mi * ((1.0 + e2) * (1.0 + s2) - 1.0)
            } else {
                s2 * mi * mj
            };
        }
    }
    let premium: Vec<f64> = (0..10).map(|w| tri.get(0, 1, w, 0).unwrap()).collect();
    let (mean, sd) = prior.prior_total(&premium).unwrap();
    assert!((mean - m.iter().sum::<f64>()).abs() < 1e-6 * mean);
    assert!(
        (sd - var.sqrt()).abs() < 1e-9 * sd,
        "{sd} vs {}",
        var.sqrt()
    );
    let (_, per_sd) = per.prior_total(&premium).unwrap();
    assert!(sd > per_sd);
    // A Pareto severity's share is the closed form, and feeds the prior.
    let sev = prospicio_prob::Pareto::new(1.0, 2.0).unwrap();
    let layer = Layer::unlimited(4.0).unwrap();
    assert!((layer_share(&sev, layer).unwrap() - 0.125).abs() < 1e-12);
}

/// The strong-prior limit's total reserve: ultimates at `m_w = apriori x
/// premium_w` and shares `p_d = S_d / (M_d + lambda)` with `lambda` such that
/// they sum to 1 (decision 23), on loss column `column`.
fn strong_prior_reserve(tri: &Triangle, column: &str, apriori: f64) -> f64 {
    let [_, _, n_origins, n_dev] = tri.shape();
    let paid = tri.column_position(column).unwrap();
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

#[test]
fn a_tight_prior_reaches_the_closed_form_limit_on_the_layer() {
    let tri = layer_triangle();
    for elr in [0.45, 0.55] {
        // Both factors held firmly: a combined cv of about 0.0007.
        let prior = LayerSharePrior::new(elr, 5e-4, &[SHARE], 5e-4).unwrap();
        let model = prior
            .model(&BayesBfNormalised {
                development_sd: 100.0,
                process: ProcessDistribution::None,
                ..base(0)
            })
            .unwrap();
        let fit = prior.fit(&model, &tri, "layer", "premium").unwrap();
        let got = fit.reserves.total().mean();
        let layer_tri = prior.triangle(&tri, "premium").unwrap();
        let limit = strong_prior_reserve(&layer_tri, "layer", elr * SHARE);
        eprintln!(
            "elr {elr}: {got:.0} vs closed-form limit {limit:.0} ({:+.2}%)",
            100.0 * (got / limit - 1.0)
        );
        assert!(
            (got / limit - 1.0).abs() < 0.015,
            "elr {elr}: {got} vs the closed-form limit {limit}"
        );
    }
}

#[test]
fn the_share_uncertainty_widens_the_interval() {
    let tri = layer_triangle();
    for dependence in [ShareDependence::Common, ShareDependence::PerOrigin] {
        let run = |share_cv: f64| {
            let prior = LayerSharePrior::new(0.35, 0.1, &[SHARE], share_cv)
                .unwrap()
                .with_dependence(dependence);
            let fit = prior.fit(&base(3), &tri, "layer", "premium").unwrap();
            let t = fit.reserves.total();
            eprintln!(
                "{dependence:?}, share cv {share_cv}: total cv {:.3}, reserve mean {:.0}, sd {:.0}, \
                 95% [{:.0}, {:.0}]",
                prior.cv(),
                t.mean(),
                t.std_dev(),
                t.quantile(0.025).unwrap(),
                t.quantile(0.975).unwrap()
            );
            (
                t.mean(),
                t.std_dev(),
                t.quantile(0.975).unwrap() - t.quantile(0.025).unwrap(),
            )
        };
        let rows: Vec<_> = [1e-3, 0.3, 0.6, 1.2].iter().map(|&c| run(c)).collect();
        // The prior level (0.35 x 0.25 of premium) is below the data's: a
        // wider prior hands the ultimates to the data and the mean rises.
        // The share error widens the interval a good deal over a known
        // share under either dependence; independent priors keep widening
        // it with the share's cv (PR 4), while a common share is learned
        // from all the origins and stops mattering past a cv of about 0.3.
        assert!(rows[1].1 > 1.4 * rows[0].1, "{dependence:?} sd: {rows:?}");
        assert!(
            rows[1].2 > 1.4 * rows[0].2,
            "{dependence:?} width: {rows:?}"
        );
        assert!(rows[1].0 > rows[0].0, "{dependence:?} mean: {rows:?}");
        if dependence == ShareDependence::PerOrigin {
            for w in rows.windows(2) {
                assert!(w[1].1 > w[0].1, "sd: {rows:?}");
                assert!(w[1].2 > w[0].2, "interval: {rows:?}");
            }
            assert!(rows[3].1 > 1.15 * rows[0].1, "sd widens by 15%: {rows:?}");
        }
    }
}

#[test]
fn a_flat_share_is_the_flat_normalised_bf() {
    let tri = layer_triangle();
    let prior = LayerSharePrior::new(0.35, 0.1, &[SHARE], f64::INFINITY).unwrap();
    assert!(prior.cv().is_infinite());
    let a = prior.fit(&base(5), &tri, "layer", "premium").unwrap();
    let b = BayesBfNormalised {
        apriori: 0.9,
        cv: f64::INFINITY,
        ..base(5)
    }
    .fit(&tri, "layer", "premium")
    .unwrap();
    // The flat prior has no centre: the apriori is ignored, and the same
    // seed gives the same posterior up to the exposure scale on alpha.
    let (ma, mb) = (a.reserves.total().mean(), b.reserves.total().mean());
    assert!((ma / mb - 1.0).abs() < 0.02, "{ma} vs {mb}");
}

/// The total reserve's mean, sd and 95% interval width with the share's cv
/// `share_cv`, common to all origins or per origin, at an ELR of 0.35.
fn total_row(share_cv: f64, dependence: ShareDependence, seed: u64) -> (f64, f64, f64) {
    let prior = LayerSharePrior::new(0.35, 0.1, &[SHARE], share_cv)
        .unwrap()
        .with_dependence(dependence);
    let fit = prior
        .fit(&base(seed), &layer_triangle(), "layer", "premium")
        .unwrap();
    let t = fit.reserves.total();
    (
        t.mean(),
        t.std_dev(),
        t.quantile(0.975).unwrap() - t.quantile(0.025).unwrap(),
    )
}

/// A common share error against independent per-origin priors with the same
/// marginal cv at each origin, GenIns's layer at an ELR of 0.35. The prior
/// total's interval is wider under the common error (the covariance
/// `cv_share^2 m_i m_j`, exact), which is what independent priors left out.
/// The posterior total's is narrower, not wider: the common multiplier is
/// learned from all ten origins, so each origin's ultimate is held to
/// `m_i s` with the loss ratio's own cv of 0.1, where the independent prior
/// gives each origin the whole marginal cv (0.32 to 1.21) and no way to learn
/// from the others.
#[test]
fn the_common_share_widens_the_prior_total_and_pools_the_posterior() {
    let tri = layer_triangle();
    let premium: Vec<f64> = (0..10).map(|w| tri.get(0, 1, w, 0).unwrap()).collect();
    for share_cv in [0.3, 0.6, 1.2] {
        let common = LayerSharePrior::new(0.35, 0.1, &[SHARE], share_cv).unwrap();
        let per = common.clone().with_dependence(ShareDependence::PerOrigin);
        let (_, prior_common) = common.prior_total(&premium).unwrap();
        let (_, prior_per) = per.prior_total(&premium).unwrap();
        let (cm, csd, ci) = total_row(share_cv, ShareDependence::Common, 3);
        let (im, isd, ii) = total_row(share_cv, ShareDependence::PerOrigin, 3);
        eprintln!(
            "share cv {share_cv}: prior total sd common {prior_common:.0}, independent \
             {prior_per:.0} (ratio {:.3}); posterior total sd common {csd:.0} (mean {cm:.0}, \
             95% width {ci:.0}), independent {isd:.0} (mean {im:.0}, width {ii:.0}), ratio {:.3}",
            prior_common / prior_per,
            csd / isd
        );
        assert!(prior_common > prior_per, "cv {share_cv}");
        // The marginal variances agree; the difference is all covariance.
        assert!(prior_common > 1.05 * prior_per, "cv {share_cv}");
        // The posterior total is narrower under the common share error at
        // cvs 0.6 and 1.2, and no wider (within Monte Carlo error) at 0.3.
        let allowed = if share_cv < 0.5 { 1.05 } else { 1.0 };
        assert!(csd < allowed * isd, "cv {share_cv}: {csd} vs {isd}");
    }
}

#[test]
fn a_vanishing_common_share_is_the_plain_model() {
    let tri = layer_triangle();
    // A known share (cv 0) has nothing to share: the per-origin model with
    // the loss ratio's cv alone, which is the plain model draw for draw.
    let known = LayerSharePrior::new(0.35, 0.1, &[SHARE], 0.0).unwrap();
    assert!(!known.has_common_factor());
    let a = known.fit(&base(4), &tri, "layer", "premium").unwrap();
    let plain = BayesBfNormalised {
        apriori: 0.35,
        cv: 0.1,
        ..base(4)
    }
    .fit(&prior_triangle(&tri), "layer", "layer_exposure")
    .unwrap();
    let (pa, pp) = (a.reserves.total().mean(), plain.reserves.total().mean());
    assert!((pa / pp - 1.0).abs() < 1e-6, "{pa} vs {pp}");
    // A tiny common cv gives it up to Monte Carlo error.
    let tiny = LayerSharePrior::new(0.35, 0.1, &[SHARE], 1e-4).unwrap();
    assert!(tiny.has_common_factor());
    let b = tiny.fit(&base(4), &tri, "layer", "premium").unwrap();
    let (ta, tb) = (a.reserves.total(), b.reserves.total());
    eprintln!(
        "plain sd {:.0} mean {:.0}; tiny common sd {:.0} mean {:.0}",
        ta.std_dev(),
        ta.mean(),
        tb.std_dev(),
        tb.mean()
    );
    assert!((ta.mean() / tb.mean() - 1.0).abs() < 0.01);
    assert!((ta.std_dev() / tb.std_dev() - 1.0).abs() < 0.06);
}

/// The layer triangle with the layer exposure column, as the adapter builds it.
fn prior_triangle(tri: &Triangle) -> Triangle {
    LayerSharePrior::new(0.35, 0.1, &[SHARE], 0.0)
        .unwrap()
        .triangle(tri, "premium")
        .unwrap()
}

/// Index of the common multiplier's draws in a fit's posterior.
fn log_share_draws(fit: &prospicio_reserving::bayes::BayesBfNormalisedFit) -> Vec<f64> {
    let j = fit
        .posterior
        .names()
        .iter()
        .position(|n| n == "log_share")
        .expect("a common multiplier");
    (0..fit.posterior.n_draws())
        .map(|d| fit.posterior.draw(d)[j])
        .collect()
}

/// A layer triangle generated with a wrong share: the data are `TRUE_SHARE`
/// of the paid values (so the layer is 1.4 times what the assumed share of
/// 0.25 says), the prior holds the share at 0.25 with cv 0.5 and the loss
/// ratio that the 0.25 triangle implies. The posterior of the multiplier
/// moves from its prior mean 1 towards 1.4; an independent per-origin prior
/// has no such parameter and its ultimates stay nearer the prior.
#[test]
fn the_posterior_of_the_common_share_moves_towards_the_truth() {
    const TRUE_SHARE: f64 = 0.35;
    let truth = TRUE_SHARE / SHARE;
    let assumed = layer_triangle();
    // The loss ratio the assumed layer implies: its flat-prior ultimates
    // over its layer exposure.
    let flat = LayerSharePrior::new(0.5, f64::INFINITY, &[SHARE], 0.0)
        .unwrap()
        .fit(&base(1), &assumed, "layer", "premium")
        .unwrap();
    let exposure: f64 = (0..10)
        .map(|w| SHARE * assumed.get(0, 1, w, 0).unwrap())
        .sum();
    let elr = flat.expected_ultimate().iter().sum::<f64>() / exposure;
    // The data: the assumed layer scaled to the true share.
    let [_, _, n_origins, n_dev] = assumed.shape();
    let (mut origin, mut age, mut layer, mut prem) = (vec![], vec![], vec![], vec![]);
    for w in 0..n_origins {
        for d in 0..n_dev {
            if let Some(v) = assumed.get(0, 0, w, d) {
                origin.push(Month::january(2001 + w as i32));
                age.push(assumed.development()[d]);
                layer.push(truth * v);
                prem.push(assumed.get(0, 1, w, 0).unwrap());
            }
        }
    }
    let data = Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("layer", &layer), ("premium", &prem)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    let prior = LayerSharePrior::new(elr, 0.1, &[SHARE], 0.5).unwrap();
    let fit = prior.fit(&base(2), &data, "layer", "premium").unwrap();
    let s: Vec<f64> = log_share_draws(&fit).iter().map(|t| t.exp()).collect();
    let n = s.len() as f64;
    let mean = s.iter().sum::<f64>() / n;
    let sd = (s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
    let per = prior
        .clone()
        .with_dependence(ShareDependence::PerOrigin)
        .fit(&base(2), &data, "layer", "premium")
        .unwrap();
    // The ultimates, against the truth `truth x` (the prior's).
    let prior_total: f64 = fit.prior_ultimate.iter().sum();
    let ratio = |f: &prospicio_reserving::bayes::BayesBfNormalisedFit| {
        f.expected_ultimate().iter().sum::<f64>() / prior_total
    };
    eprintln!(
        "elr {elr:.4}; s posterior mean {mean:.3} sd {sd:.3} (truth {truth}); total ultimate over \
         prior: common {:.3}, per origin {:.3}, truth {:.3}",
        ratio(&fit),
        ratio(&per),
        truth
    );
    assert!(mean > 1.2 && mean < 1.6, "s mean {mean}");
    assert!((mean - truth).abs() < (1.0 - truth).abs() / 2.0, "{mean}");
    assert!(sd > 0.0 && sd < 0.5, "{sd}");
    // The common multiplier carries the data's message to every origin at
    // once: nearer the truth than the independent priors.
    assert!((ratio(&fit) - truth).abs() < (ratio(&per) - truth).abs());
}
