//! A development pattern per simulation from the bootstraps
//! (`reserving-v02.md` decision 28, issue #600): each pattern lines up with
//! its simulation's reserves, follows them through a rank correlation,
//! averages to the chain ladder's pattern, carries the tail, and runs
//! through the cash-flow engine as its paid pattern.

use prospicio_pricing::cashflow::{
    Base, CashflowTerms, MonthlyPatterns, PatternDraws, Requirement, Term,
};
use prospicio_prob::{Empirical, KeyValue};
use prospicio_reserving::{
    DevelopmentColumn, DevelopmentPatterns, Grain, Long, MackBootstrap, Monotone, MonthlyBasis,
    OdpBootstrap, ProcessDistribution, SegmentDependence, Tail, TailConstant, Triangle,
    UltimateDraws, UnderwritingBasis,
};
use prospicio_validation::triangle;

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

fn constant_tail(factor: f64) -> Tail {
    Tail::Constant(TailConstant {
        factor,
        ..Default::default()
    })
}

fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx).powi(2);
        syy += (b - my).powi(2);
    }
    sxy / (sxx * syy).sqrt()
}

/// Every row of a rank-correlated fit's segment is a row of the same fit
/// without the correlation (the reserves of that segment, matched exactly),
/// and its pattern is that row's pattern.
fn check_follows(
    independent: (
        &prospicio_prob::PredictiveDistribution,
        &DevelopmentPatterns,
    ),
    ranked: (
        &prospicio_prob::PredictiveDistribution,
        &DevelopmentPatterns,
    ),
) {
    let n = independent.0.n_components();
    let half = n / 2;
    let rows = |pd: &prospicio_prob::PredictiveDistribution, s: usize| -> Vec<Vec<u64>> {
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
fn patterns_follow_the_reserves_through_a_rank_correlation() {
    let tri = twins(&triangle("raa"));
    let rank = SegmentDependence::RankCorrelation {
        spearman: vec![1.0, 0.7, 0.7, 1.0],
    };
    let odp = OdpBootstrap {
        n_sims: 1_000,
        seed: 600,
        ..Default::default()
    };
    let a = odp.fit_segments(&tri, "values").unwrap();
    let b = OdpBootstrap {
        dependence: rank.clone(),
        ..odp
    }
    .fit_segments(&tri, "values")
    .unwrap();
    check_follows((&a.reserves, &a.patterns), (&b.reserves, &b.patterns));

    let mack = MackBootstrap {
        n_sims: 1_000,
        seed: 601,
        tail: constant_tail(1.05),
        ..Default::default()
    };
    let a = mack.fit_segments(&tri, "values").unwrap();
    let b = MackBootstrap {
        dependence: rank,
        ..mack
    }
    .fit_segments(&tri, "values")
    .unwrap();
    check_follows((&a.reserves, &a.patterns), (&b.reserves, &b.patterns));
    // A segment picked alone keeps its patterns.
    let alone = b.segment(&[("lob", "B")]).unwrap();
    assert_eq!(alone.patterns.segments().len(), 1);
    assert_eq!(
        alone.patterns.pattern_draws(&["B"]).unwrap(),
        b.patterns.pattern_draws(&["B"]).unwrap()
    );
}

#[test]
fn a_simulations_pattern_moves_with_its_reserve() {
    // Without process error the youngest origin's reserve is its pseudo
    // value times the simulation's factor to ultimate less 1, so it moves
    // with 1 over the share at the first age (the pseudo value, one
    // resampled cell, blurs it); a pattern from another simulation does
    // not.
    let raa = triangle("raa");
    let fit = OdpBootstrap {
        n_sims: 2_000,
        seed: 602,
        process: ProcessDistribution::None,
        ..Default::default()
    }
    .fit(&raa, "values")
    .unwrap();
    let youngest: Vec<KeyValue> = vec![1990.into()];
    let reserve = Empirical::draws(&fit.reserves.marginal(&youngest).unwrap()).to_vec();
    let segment = fit.patterns.segment(&[]).unwrap();
    let cdf: Vec<f64> = (0..fit.patterns.n_sims())
        .map(|i| 1.0 / segment.draw(i)[0])
        .collect();
    let own = correlation(&reserve, &cdf);
    let mut shifted = cdf.clone();
    shifted.rotate_left(1);
    let other = correlation(&reserve, &shifted);
    // With 2,000 simulations an unrelated pair's correlation has a
    // standard deviation of about 0.022.
    assert!(own > 0.3, "{own}");
    assert!(other.abs() < 0.1, "{other}");

    // The same in Mack's bootstrap, whose process error is drawn cell by
    // cell from the observed latest value.
    let fit = MackBootstrap {
        n_sims: 2_000,
        seed: 603,
        ..Default::default()
    }
    .fit(&raa, "values")
    .unwrap();
    let ultimate = fit.ultimate_draws(&youngest).unwrap();
    let segment = fit.patterns.segment(&[]).unwrap();
    let cdf: Vec<f64> = (0..fit.patterns.n_sims())
        .map(|i| 1.0 / segment.draw(i)[0])
        .collect();
    let own = correlation(&ultimate, &cdf);
    let mut shifted = cdf.clone();
    shifted.rotate_left(1);
    let other = correlation(&ultimate, &shifted);
    assert!(own > 0.2, "{own}");
    assert!(other.abs() < 0.1, "{other}");
}

#[test]
fn the_mean_pattern_is_the_chain_ladders() {
    let raa = triangle("raa");
    let fit = OdpBootstrap {
        n_sims: 5_000,
        seed: 604,
        ..Default::default()
    }
    .fit(&raa, "values")
    .unwrap();
    let segment = fit.patterns.segment(&[]).unwrap();
    assert_eq!(segment.ages(), [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]);
    let ldf = fit.chain_ladder.ldf();
    let mean = segment.mean();
    for (k, m) in mean.iter().enumerate() {
        let share = 1.0 / ldf[k..].iter().product::<f64>();
        assert!(
            (m / share - 1.0).abs() < 0.03,
            "age {k}: {m} against {share}"
        );
    }
    for i in 0..fit.patterns.n_sims() {
        assert_eq!(segment.draw(i)[9], 1.0);
    }
}

#[test]
fn a_tail_reaches_one_a_period_after_the_oldest_age() {
    let raa = triangle("raa");
    let odp = OdpBootstrap {
        n_sims: 500,
        seed: 605,
        tail: constant_tail(1.05),
        ..Default::default()
    }
    .fit(&raa, "values")
    .unwrap();
    let mack = MackBootstrap {
        n_sims: 500,
        seed: 606,
        tail: constant_tail(1.05),
        ..Default::default()
    }
    .fit(&raa, "values")
    .unwrap();
    // The ODP bootstrap keeps a constant tail fixed, so the share at the
    // oldest age is 1 over it in every simulation.
    let at_oldest = odp.chain_ladder.latest[0] / odp.chain_ladder.ultimate[0];
    assert!((at_oldest - 1.0 / 1.05).abs() < 1e-12);
    let segment = odp.patterns.segment(&[]).unwrap();
    assert_eq!(segment.ages().len(), 11);
    assert_eq!(segment.ages()[10], 132);
    for i in 0..odp.patterns.n_sims() {
        let row = segment.draw(i);
        assert!((row[9] - at_oldest).abs() < 1e-12, "{}", row[9]);
        assert_eq!(row[10], 1.0);
    }
    // Mack's bootstrap draws the tail factor with the tail's sigma, so the
    // share at the oldest age varies about 1 over the constant.
    let segment = mack.patterns.segment(&[]).unwrap();
    assert_eq!(segment.ages()[10], 132);
    for i in 0..mack.patterns.n_sims() {
        assert_eq!(segment.draw(i)[10], 1.0);
    }
    let mean = segment.mean()[9];
    assert!((mean * 1.05 - 1.0).abs() < 0.01, "{mean}");
}

#[test]
fn monthly_patterns_run_through_the_cash_flow_engine() {
    let raa = triangle("raa");
    let fit = MackBootstrap {
        n_sims: 400,
        seed: 607,
        ..Default::default()
    }
    .fit(&raa, "values")
    .unwrap();
    let months = 144;
    let paid = fit
        .patterns
        .monthly(&[], &MonthlyBasis::default(), months, Monotone::RunningMax)
        .unwrap();
    assert_eq!(paid.len(), 400);
    for row in &paid {
        assert_eq!(row.len(), months);
        assert_eq!(row[months - 1], 1.0);
        assert!(row.windows(2).all(|w| w[1] >= w[0]));
    }
    // An underwriting year written over 12 months is less developed at 12
    // months than the accident year, and fully developed 12 months later.
    let uy = fit
        .patterns
        .monthly(
            &[],
            &MonthlyBasis::Underwriting(UnderwritingBasis::default()),
            months,
            Monotone::RunningMax,
        )
        .unwrap();
    assert!(uy[0][11] < paid[0][11]);
    assert!(
        fit.patterns
            .monthly(
                &[],
                &MonthlyBasis::Underwriting(UnderwritingBasis::default()),
                120,
                Monotone::RunningMax
            )
            .is_err()
    );

    // The terms pay by simulation 0's pattern; each simulation's run with
    // its own pattern is the terms' run with that pattern alone.
    let ramp = |m: f64| (0..months).map(|t| ((t + 1) as f64 / m).min(1.0)).collect();
    let terms = CashflowTerms::new(
        100.0,
        24,
        MonthlyPatterns::new(ramp(12.0), ramp(24.0), paid[0].clone()).unwrap(),
        Requirement::new(vec![Term::new(Base::Reserve, 1.0)]),
    )
    .unwrap();
    let premium = 60_000.0;
    let u = fit.total_loss_ratio_draws(premium).unwrap();
    let commission = vec![0.0; u.len()];
    let draws = PatternDraws::new(paid.clone()).unwrap();
    let run = terms.run_with_patterns(&u, &commission, &draws).unwrap();
    let n = run.months();
    for j in [0, 7, 399] {
        let mut alone = terms.clone();
        alone.patterns.paid = paid[j].clone();
        let one = alone.run(&u[j..=j], &commission[j..=j]).unwrap();
        assert_eq!(&run.flows()[j * n..(j + 1) * n], one.flows());
        assert_eq!(run.funded()[j], one.funded()[0]);
    }
}

#[test]
fn a_falling_pattern_is_refused_unless_held() {
    // Pseudo factors below 1 at the old ages of a nearly closed triangle.
    let raa = triangle("raa");
    let fit = OdpBootstrap {
        n_sims: 2_000,
        seed: 608,
        ..Default::default()
    }
    .fit(&raa, "values")
    .unwrap();
    let falls = (0..fit.patterns.n_sims()).any(|i| {
        let row = fit.patterns.segment(&[]).unwrap().draw(i);
        row.windows(2).any(|w| w[1] < w[0])
    });
    let refused = fit
        .patterns
        .monthly(&[], &MonthlyBasis::default(), 120, Monotone::Refuse);
    assert_eq!(refused.is_err(), falls);
    let held = fit
        .patterns
        .monthly(&[], &MonthlyBasis::default(), 120, Monotone::RunningMax)
        .unwrap();
    assert!(
        held.iter()
            .all(|row| row.windows(2).all(|w| w[1] >= w[0]) && row[119] == 1.0)
    );
}
