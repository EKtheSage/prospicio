//! A development pattern per simulation from the Bayesian fits
//! (`reserving-v02.md` decision 28, issue #600): each simulation's pattern is
//! its posterior draw's, so it ends at 1, follows its reserves through a
//! rank correlation of segments, and averages to the fit's own pattern.

use prospicio_bayes::nuts::Sampler;
use prospicio_prob::PredictiveDistribution;
use prospicio_reserving::bayes::{BayesBf, BayesBfNormalised};
use prospicio_reserving::{
    BayesOdp, ChainLadder, DevelopmentColumn, DevelopmentPatterns, Grain, Long, SegmentDependence,
    Triangle,
};
use prospicio_validation::{triangle, triangle_columns};

fn sampler() -> Sampler {
    Sampler {
        chains: 2,
        tune: 400,
        draws: 400,
        ..Sampler::default()
    }
}

/// `tri`'s values twice, under the `lob` values `A` and `B`, with every
/// value of `B` doubled.
fn twins(tri: &Triangle) -> Triangle {
    let long = tri.to_long();
    let values = &long.values.iter().find(|(c, _)| c == "values").unwrap().1;
    let n = long.origin.len();
    let keys: Vec<&str> = (0..2 * n).map(|i| if i < n { "A" } else { "B" }).collect();
    let doubled: Vec<f64> = values
        .iter()
        .copied()
        .chain(values.iter().map(|v| 2.0 * v))
        .collect();
    Triangle::from_long(&Long {
        keys: &[("lob", &keys)],
        origin: &[long.origin.clone(), long.origin.clone()].concat(),
        development: DevelopmentColumn::Age(&[long.development.clone(), long.development].concat()),
        values: &[("values", &doubled)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// Every simulation's pattern ends at exactly 1 and none exceeds it.
fn check_ends_at_one(patterns: &DevelopmentPatterns, n_ages: usize) {
    for segment in patterns.segments() {
        assert_eq!(segment.ages().len(), n_ages);
        for i in 0..patterns.n_sims() {
            let row = segment.draw(i);
            assert_eq!(row[n_ages - 1], 1.0);
            assert!(row.iter().all(|&x| x > 0.0 && x <= 1.0), "{row:?}");
        }
    }
}

/// Every row of a rank-correlated fit's segment is a row of the same fit
/// without the correlation (its reserves matched exactly), and its pattern
/// is that row's pattern.
fn check_follows(
    independent: (&PredictiveDistribution, &DevelopmentPatterns),
    ranked: (&PredictiveDistribution, &DevelopmentPatterns),
) {
    let n = independent.0.n_components();
    let half = n / 2;
    let rows = |pd: &PredictiveDistribution, s: usize| -> Vec<Vec<u64>> {
        pd.draw_matrix()
            .chunks_exact(n)
            .map(|r| {
                r[s * half..(s + 1) * half]
                    .iter()
                    .map(|x| x.to_bits())
                    .collect()
            })
            .collect()
    };
    let mut moved = 0;
    for (s, label) in ["A", "B"].iter().enumerate() {
        let from = rows(independent.0, s);
        let to = rows(ranked.0, s);
        let p_from = independent.1.segment(&[label]).unwrap();
        let p_to = ranked.1.segment(&[label]).unwrap();
        for (i, row) in to.iter().enumerate() {
            let j = from.iter().position(|r| r == row).expect("a source row");
            assert_eq!(p_to.draw(i), p_from.draw(j), "segment {label}, row {i}");
            moved += usize::from(i != j);
        }
    }
    assert!(moved > 100, "the reorder moved only {moved} rows");
}

#[test]
fn bayes_odp_patterns_end_at_one_and_average_to_the_chain_ladders() {
    let raa = triangle("raa");
    let fit = BayesOdp {
        sampler: sampler(),
        n_sims: 2_000,
        seed: 610,
        ..BayesOdp::default()
    }
    .fit(&raa, "values")
    .unwrap();
    check_ends_at_one(&fit.patterns, 10);
    // Under the default flat priors the posterior mean of the pattern is
    // close to the chain ladder's; the youngest ages are the least
    // informed.
    let cl = ChainLadder::default().fit(&raa, "values").unwrap();
    let ldf = cl.ldf();
    let mean = fit.patterns.segment(&[]).unwrap().mean();
    for (k, m) in mean.iter().enumerate() {
        let share = 1.0 / ldf[k..].iter().product::<f64>();
        assert!(
            (m / share - 1.0).abs() < 0.1,
            "age {k}: {m} against {share}"
        );
    }
}

#[test]
fn bayes_odp_patterns_follow_the_reserves_through_a_rank_correlation() {
    let tri = twins(&triangle("raa"));
    let model = BayesOdp {
        sampler: sampler(),
        n_sims: 1_000,
        seed: 611,
        ..BayesOdp::default()
    };
    let a = model.fit_segments(&tri, "values").unwrap();
    let b = BayesOdp {
        dependence: SegmentDependence::RankCorrelation {
            spearman: vec![1.0, 0.7, 0.7, 1.0],
        },
        ..model
    }
    .fit_segments(&tri, "values")
    .unwrap();
    check_ends_at_one(&b.patterns, 10);
    check_follows((&a.reserves, &a.patterns), (&b.reserves, &b.patterns));
}

#[test]
fn bayes_bf_patterns_end_at_one() {
    let genins = triangle_columns("genins_premium", &["paid", "premium"]);
    let fit = BayesBf {
        sampler: sampler(),
        n_sims: 1_000,
        seed: 612,
        ..BayesBf::default()
    }
    .fit(&genins, "paid", "premium")
    .unwrap();
    check_ends_at_one(&fit.patterns, 10);
}

#[test]
fn bayes_bf_normalised_patterns_average_to_its_pattern() {
    let genins = triangle_columns("genins_premium", &["paid", "premium"]);
    let fit = BayesBfNormalised {
        apriori: 0.5,
        cv: 0.2,
        sampler: sampler(),
        n_sims: 4_000,
        seed: 613,
        ..BayesBfNormalised::default()
    }
    .fit(&genins, "paid", "premium")
    .unwrap();
    check_ends_at_one(&fit.patterns, 10);
    // Each simulation's pattern is the cumulative sum of its posterior
    // draw's shares, so their mean is the cumulative sum of the posterior
    // mean shares, up to which draws the simulations picked.
    let mean = fit.patterns.segment(&[]).unwrap().mean();
    let mut cumulative = 0.0;
    for (k, share) in fit.pattern().iter().enumerate() {
        cumulative += share;
        assert!(
            (mean[k] / cumulative - 1.0).abs() < 0.01,
            "age {k}: {} against {cumulative}",
            mean[k]
        );
    }
}
