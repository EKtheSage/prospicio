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
//! * a flat share (`share_cv` infinite) gives the flat normalised BF.

use prospicio_prob::Distribution;
use prospicio_reserving::bayes::{BayesBfNormalised, LayerSharePrior};
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
    let model = prior.model(&BayesBfNormalised {
        n_sims: 200,
        ..base(0)
    });
    assert_eq!(model.apriori, 0.5);
    assert_eq!(model.cv, prior.cv());
    let fit = prior.fit(&model, &tri, "layer", "premium").unwrap();
    for (w, &u) in fit.prior_ultimate.iter().enumerate() {
        let premium = tri.get(0, 1, w, 0).unwrap();
        assert!((u - 0.5 * SHARE * premium).abs() < 1e-9 * premium, "{w}");
    }
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
        let model = prior.model(&BayesBfNormalised {
            development_sd: 100.0,
            process: ProcessDistribution::None,
            ..base(0)
        });
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
    let run = |share_cv: f64| {
        let prior = LayerSharePrior::new(0.35, 0.1, &[SHARE], share_cv).unwrap();
        let fit = prior.fit(&base(3), &tri, "layer", "premium").unwrap();
        let t = fit.reserves.total();
        eprintln!(
            "share cv {share_cv}: total cv {:.3}, reserve mean {:.0}, sd {:.0}, 95% [{:.0}, {:.0}]",
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
    // wider prior hands the ultimates to the data and the mean rises, and
    // the predictive sd and interval widen with the share's cv.
    for w in rows.windows(2) {
        assert!(w[1].1 > w[0].1, "sd: {rows:?}");
        assert!(w[1].2 > w[0].2, "interval: {rows:?}");
    }
    assert!(rows[3].1 > 1.15 * rows[0].1, "sd widens by 15%: {rows:?}");
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
