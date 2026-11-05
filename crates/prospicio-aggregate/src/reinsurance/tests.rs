use super::*;
use crate::simulate_events;
use prospicio_prob::provenance::SIM_INDEX_SCHEME;
use prospicio_prob::{Distribution, Lognormal, Poisson, Severity};

#[test]
fn a_negative_loss_is_refused_by_the_checked_ceded() {
    let layer = Layer::xol("10x5", 10.0, 5.0).unwrap();
    // Unchecked, a sign error cedes nothing.
    assert_eq!(layer.ceded(&[-12.0]), 0.0);
    assert_eq!(layer.try_ceded(&[12.0]), Ok(7.0));
    let err = layer.try_ceded(&[12.0, -3.0]).unwrap_err();
    assert!(err.to_string().contains("loss 2"), "{err}");
    assert!(layer.try_ceded(&[f64::INFINITY]).is_err());
    assert_eq!(
        layer.try_ceded_by_event(&[12.0, 20.0]),
        Ok(layer.ceded_by_event(&[12.0, 20.0]))
    );
    assert!(layer.try_ceded_by_event(&[-1.0]).is_err());
    let tower = Tower::new(vec![layer]).unwrap();
    assert_eq!(tower.try_ceded(&[12.0]), Ok(vec![7.0]));
    assert!(tower.try_ceded(&[f64::NAN]).is_err());
}

/// #483's three years: attritional 10, 40, 30 and large losses [],
/// [50], [5, 60], the attritional given by `part`.
fn attritional_years(part: PredictiveDistribution) -> EventSet {
    EventSet::from_years(vec![vec![], vec![50.0], vec![5.0, 60.0]], 0)
        .unwrap()
        .with_aggregate_part(&part)
        .unwrap()
}

fn unlabelled(values: Vec<f64>) -> PredictiveDistribution {
    PredictiveDistribution::from_draws(vec![], vec![vec![]], values, Provenance::new("part"))
        .unwrap()
}

fn column(result: &PredictiveDistribution, key: &[&str]) -> Vec<f64> {
    let key: Vec<KeyValue> = key.iter().map(|&k| k.into()).collect();
    prospicio_prob::Empirical::draws(&result.marginal(&key).unwrap()).to_vec()
}

fn close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-12, "{a:?} vs {b:?}");
    }
}

#[test]
fn occurrence_terms_skip_the_aggregate_part() {
    let events = attritional_years(unlabelled(vec![10.0, 40.0, 30.0]));
    let tower = Tower::new(vec![
        Layer::xol("30x25", 30.0, 25.0).unwrap(),
        Layer::quota_share("QS", 0.5).unwrap(),
        Layer::stop_loss("SL", 50.0, 40.0).unwrap(),
    ])
    .unwrap();
    let r = tower.apply(&events).unwrap();
    // The issue's table: 0, 25, 30, where one event for the year
    // would cede 0, 40, 35.
    close(&column(&r, &["ceded", "30x25"]), &[0.0, 25.0, 30.0]);
    close(&column(&r, &["ceded", "QS"]), &[5.0, 45.0, 47.5]);
    close(&column(&r, &["ceded", "SL"]), &[0.0, 50.0, 50.0]);
    let gross = column(&r, &["gross", "ground_up"]);
    close(&gross, &[10.0, 90.0, 95.0]);
    let net = column(&r, &["net", "retained"]);
    for i in 0..3 {
        let ceded: f64 = ["30x25", "QS", "SL"]
            .iter()
            .map(|l| column(&r, &["ceded", l])[i])
            .sum();
        assert!((net[i] - (gross[i] - ceded)).abs() < 1e-12);
    }
    close(
        prospicio_prob::Empirical::draws(events.totals().unwrap().total()),
        &gross,
    );
}

#[test]
fn inuring_stages_see_the_aggregate_part_net_of_earlier_ones() {
    let events = attritional_years(unlabelled(vec![10.0, 40.0, 30.0]));
    // A per risk inuring to a stop loss: the stop loss sees the
    // attritional whole and the large losses net, 10, 65, 65.
    let tower = Tower::inuring(vec![
        vec![Layer::xol("30x25", 30.0, 25.0).unwrap()],
        vec![Layer::stop_loss("SL", 50.0, 40.0).unwrap()],
    ])
    .unwrap();
    let r = tower.apply(&events).unwrap();
    close(&column(&r, &["ceded", "30x25"]), &[0.0, 25.0, 30.0]);
    close(&column(&r, &["ceded", "SL"]), &[0.0, 25.0, 25.0]);
    // A quota share inuring to a per risk: the per risk sees the large
    // losses at half, 25 and 2.5, 30.
    let tower = Tower::inuring(vec![
        vec![Layer::quota_share("QS", 0.5).unwrap()],
        vec![Layer::xol("30x25", 30.0, 25.0).unwrap()],
    ])
    .unwrap();
    let r = tower.apply(&events).unwrap();
    close(&column(&r, &["ceded", "QS"]), &[5.0, 45.0, 47.5]);
    close(&column(&r, &["ceded", "30x25"]), &[0.0, 0.0, 5.0]);
}

#[test]
fn the_aggregate_part_comes_first_in_the_year() {
    let part = PredictiveDistribution::from_draws(
        vec!["class".into()],
        vec![vec!["attritional".into()]],
        vec![10.0, 40.0, 30.0],
        Provenance::new("part"),
    )
    .unwrap();
    let events = EventSet::from_years(vec![vec![], vec![50.0], vec![5.0, 60.0]], 0)
        .unwrap()
        .with_labels(vec![("class".into(), vec!["large".into(); 3])])
        .unwrap()
        .with_aggregate_part(&part)
        .unwrap();
    assert_eq!(events.label_combinations(), [["attritional"], ["large"]]);
    // The deductible of 40 is used up on the attritional first.
    let sl = Layer::stop_loss("SL", 50.0, 40.0).unwrap();
    let r = Tower::new(vec![sl.clone()])
        .unwrap()
        .apply(&events)
        .unwrap();
    close(
        &column(&r, &["ceded", "SL", "attritional"]),
        &[0.0, 0.0, 0.0],
    );
    close(&column(&r, &["ceded", "SL", "large"]), &[0.0, 50.0, 50.0]);
    close(
        &column(&r, &["gross", "ground_up", "attritional"]),
        &[10.0, 40.0, 30.0],
    );
    close(
        &column(&r, &["net", "retained", "attritional"]),
        &[10.0, 40.0, 30.0],
    );
    // Pro rata, the same annual total shared by amount.
    let pro_rata = sl.allocation(LimitAllocation::ProRata).unwrap();
    let r = Tower::new(vec![pro_rata]).unwrap().apply(&events).unwrap();
    close(
        &column(&r, &["ceded", "SL", "attritional"]),
        &[0.0, 40.0 * 50.0 / 90.0, 30.0 * 50.0 / 95.0],
    );
    close(
        &column(&r, &["ceded", "SL", "large"]),
        &[0.0, 50.0 * 50.0 / 90.0, 65.0 * 50.0 / 95.0],
    );
}

#[test]
fn the_aggregate_part_by_peril() {
    // Attritional wind and fire beside wind and flood events: fire has
    // no events, and a per risk cedes nothing of either attritional.
    let part = PredictiveDistribution::from_draws(
        vec!["peril".into()],
        vec![vec!["wind".into()], vec!["fire".into()]],
        vec![1.0, 9.0, 2.0, 38.0, 3.0, 27.0],
        Provenance::new("part"),
    )
    .unwrap();
    let events = EventSet::from_years(vec![vec![], vec![50.0], vec![5.0, 60.0]], 0)
        .unwrap()
        .with_labels(vec![(
            "peril".into(),
            vec!["wind".into(), "flood".into(), "wind".into()],
        )])
        .unwrap()
        .with_aggregate_part(&part)
        .unwrap();
    assert_eq!(events.label_combinations(), [["fire"], ["flood"], ["wind"]]);
    let labels: Vec<Vec<String>> = events
        .aggregate_part_labels()
        .unwrap()
        .into_iter()
        .map(<[String]>::to_vec)
        .collect();
    assert_eq!(labels, [["wind"], ["fire"]]);
    assert_eq!(events.aggregate_part(1), Some(&[2.0, 38.0][..]));
    let tower = Tower::new(vec![
        Layer::xol("30x25", 30.0, 25.0).unwrap(),
        Layer::quota_share("QS", 0.5).unwrap(),
    ])
    .unwrap();
    let r = tower.apply(&events).unwrap();
    close(
        &column(&r, &["gross", "ground_up", "fire"]),
        &[9.0, 38.0, 27.0],
    );
    close(
        &column(&r, &["gross", "ground_up", "flood"]),
        &[0.0, 0.0, 5.0],
    );
    close(
        &column(&r, &["gross", "ground_up", "wind"]),
        &[1.0, 52.0, 63.0],
    );
    close(&column(&r, &["ceded", "30x25", "wind"]), &[0.0, 25.0, 30.0]);
    close(&column(&r, &["ceded", "30x25", "fire"]), &[0.0, 0.0, 0.0]);
    close(&column(&r, &["ceded", "QS", "fire"]), &[4.5, 19.0, 13.5]);
    // Summed over labels, the unlabelled result.
    let flat = attritional_years(unlabelled(vec![10.0, 40.0, 30.0]));
    let a = r.aggregate(&["kind", "layer"]).unwrap();
    let b = tower.apply(&flat).unwrap();
    close(a.draw_matrix(), b.draw_matrix());
}

#[test]
fn a_term_sees_each_years_aggregate_part() {
    let events = attritional_years(unlabelled(vec![10.0, 40.0, 30.0]));
    let years = events.term_years(3).unwrap();
    let tower = Tower::new(vec![Layer::quota_share("QS", 0.5).unwrap()]).unwrap();
    let r = tower.apply_term(&years).unwrap();
    let ceded: Vec<f64> = (1..=3)
        .map(|t| {
            let key = vec![
                KeyValue::from("ceded"),
                KeyValue::from("QS"),
                KeyValue::Int(t),
            ];
            prospicio_prob::Empirical::draws(&r.marginal(&key).unwrap())[0]
        })
        .collect();
    close(&ceded, &[5.0, 45.0, 47.5]);
}

#[test]
fn the_seven_million_example() {
    // Retain 5m; 5m xs 5m; 15m xs 10m. A 7m loss: 5m kept, 2m to layer 2.
    let l2 = Layer::xol("5x5", 5e6, 5e6).unwrap();
    let l3 = Layer::xol("15x10", 15e6, 10e6).unwrap();
    assert_eq!(l2.ceded(&[7e6]), 2e6);
    assert_eq!(l3.ceded(&[7e6]), 0.0);
    // A 30m loss exhausts both layers and leaves 5m + 5m retained.
    assert_eq!(l2.ceded(&[30e6]), 5e6);
    assert_eq!(l3.ceded(&[30e6]), 15e6);
}

#[test]
fn annual_terms() {
    let base = || Layer::xol("L", 10.0, 5.0).unwrap();
    let losses = [8.0, 20.0, 12.0]; // recoveries 3, 10, 7 = 20
    assert_eq!(base().ceded(&losses), 20.0);
    assert_eq!(
        base().aggregate_deductible(4.0).unwrap().ceded(&losses),
        16.0
    );
    assert_eq!(base().aggregate_limit(15.0).unwrap().ceded(&losses), 15.0);
    assert_eq!(base().reinstatements(0).unwrap().ceded(&losses), 10.0);
    assert_eq!(base().share(0.25).unwrap().ceded(&losses), 5.0);
    let all = base()
        .aggregate_deductible(4.0)
        .unwrap()
        .aggregate_limit(15.0)
        .unwrap()
        .share(0.5)
        .unwrap();
    assert_eq!(all.ceded(&losses), 7.5);
    assert_eq!(base().ceded(&[]), 0.0);
}

#[test]
fn loss_corridors() {
    let base = || Layer::xol("L", 10.0, 5.0).unwrap();
    let losses = [8.0, 20.0, 12.0]; // recoveries 3, 10, 7 = 20
    // After the deductible of 4, 16; the corridor keeps half of 5..9,
    // so 14; the annual limit of 12 then caps it.
    let layer = base()
        .aggregate_deductible(4.0)
        .unwrap()
        .loss_corridor(5.0, 9.0, 0.5)
        .unwrap()
        .aggregate_limit(12.0)
        .unwrap();
    assert_eq!(layer.ceded(&losses), 12.0);
    assert_eq!(layer.clone().share(0.5).unwrap().ceded(&losses), 6.0);
    // The reinsurer still pays up to the full annual limit.
    let wide = base()
        .aggregate_deductible(4.0)
        .unwrap()
        .loss_corridor(5.0, 9.0, 0.5)
        .unwrap()
        .aggregate_limit(15.0)
        .unwrap();
    assert_eq!(wide.ceded(&losses), 14.0);
    // Used up in event order: after the deductible 0, 9, 16, after the
    // corridor 0, 7, 14, after the limit 0, 7, 12.
    assert_eq!(layer.ceded_by_event(&losses), [0.0, 7.0, 5.0]);
    assert!(layer.has_annual_terms());
    assert!(
        base()
            .loss_corridor(0.0, 1.0, 1.0)
            .unwrap()
            .has_annual_terms()
    );
    assert!(!base().has_annual_terms());

    // Reinstatement premiums follow the loss after the corridor:
    // recovery 2, of which the corridor keeps 1.
    let paid = || {
        Layer::xol("10x10", 10.0, 10.0)
            .unwrap()
            .paid_reinstatements(2.0, vec![1.0])
            .unwrap()
    };
    assert!((paid().reinstatement_premium(&[12.0]) - 0.4).abs() < 1e-12);
    let kept = paid().loss_corridor(0.0, 1.0, 1.0).unwrap();
    assert!((kept.reinstatement_premium(&[12.0]) - 0.2).abs() < 1e-12);

    assert!(base().loss_corridor(-1.0, 1.0, 1.0).is_err());
    assert!(base().loss_corridor(2.0, 2.0, 1.0).is_err());
    assert!(base().loss_corridor(1.0, f64::INFINITY, 1.0).is_err());
    assert!(base().loss_corridor(1.0, 2.0, 0.0).is_err());
    assert!(base().loss_corridor(1.0, 2.0, 1.5).is_err());
}

#[test]
fn corridor_keeps_gross_equal_to_ceded_plus_net() {
    let tower = Tower::inuring(vec![
        vec![
            Layer::quota_share("QS", 0.4)
                .unwrap()
                .loss_corridor(2e6, 6e6, 0.5)
                .unwrap(),
        ],
        vec![
            Layer::xol("5x5", 5e6, 5e6)
                .unwrap()
                .loss_corridor(2e6, 8e6, 1.0)
                .unwrap()
                .reinstatements(2)
                .unwrap(),
        ],
    ])
    .unwrap();
    let result = tower.apply(&events()).unwrap();
    let mut kept = 0;
    for sim in 0..result.n_sims() {
        let row = result.row(sim).unwrap();
        assert!((row[0] - row[1] - row[2] - row[3]).abs() <= 1e-6 * row[0].max(1.0));
        // The QS cedes 40% of the year's loss less half its corridor.
        let gross = row[0];
        let want = 0.4 * (gross - 0.5 * (gross - 2e6).clamp(0.0, 4e6));
        assert!((row[1] - want).abs() <= 1e-6 * gross.max(1.0));
        kept += usize::from(gross > 2e6);
    }
    assert!(kept > 1_000);
}

#[test]
fn rejects_bad_terms() {
    assert!(Layer::xol("L", 0.0, 1.0).is_err());
    assert!(Layer::xol("L", 1.0, -1.0).is_err());
    assert!(Layer::xol("L", 1.0, 0.0).unwrap().share(1.5).is_err());
    assert!(
        Layer::xol("L", 1.0, 0.0)
            .unwrap()
            .aggregate_limit(0.0)
            .is_err()
    );
    assert!(Tower::new(vec![]).is_err());
    let l = Layer::xol("L", 1.0, 0.0).unwrap();
    assert!(Tower::new(vec![l.clone(), l]).is_err());
}

#[test]
fn premiums_from_quotes() {
    let xl = || Layer::xol("10x10", 10.0, 10.0).unwrap().share(0.4).unwrap();
    assert!((xl().rate_on_line(0.2).unwrap().premium - 0.8).abs() < 1e-12);
    assert!((xl().deposit_premium(3.0).unwrap().premium - 1.2).abs() < 1e-12);
    assert!((xl().premium_rate(0.05, 50.0).unwrap().premium - 1.0).abs() < 1e-12);
    assert!(
        Layer::quota_share("QS", 0.3)
            .unwrap()
            .rate_on_line(0.1)
            .is_err()
    );
    assert!(xl().rate_on_line(-0.1).is_err());
    assert!(xl().deposit_premium(f64::NAN).is_err());
    assert!(xl().premium_rate(0.1, f64::INFINITY).is_err());
}

#[test]
fn a_no_claims_bonus_refunds_a_share_in_a_clean_year() {
    let layer = || {
        Layer::xol("cat", 20.0, 10.0)
            .unwrap()
            .premium_rate(0.1, 100.0)
            .unwrap()
    };
    assert!(layer().no_claims_bonus(0.0, 1, 0.0).is_err());
    assert!(layer().no_claims_bonus(1.5, 1, 0.0).is_err());
    assert!(layer().no_claims_bonus(0.1, 0, 0.0).is_err());
    assert!(layer().no_claims_bonus(0.1, 1, -1.0).is_err());
    assert!(
        Layer::xol("free", 20.0, 10.0)
            .unwrap()
            .no_claims_bonus(0.1, 1, 0.0)
            .is_err()
    );
    // The guide's sliding-scale approximation, 10% at a loss ratio of 0
    // and 0 just above, on simulated years.
    let ncb = layer().no_claims_bonus(0.1, 1, 0.0).unwrap();
    let scale = layer()
        .sliding_scale(vec![(0.10, 0.0), (0.0, 1e-9)])
        .unwrap();
    let events = crate::simulate_events(
        &Poisson::new(0.5).unwrap(),
        &Lognormal::from_mean_cv(8.0, 1.0).unwrap(),
        5_000,
        3,
    )
    .unwrap();
    let a = Tower::new(vec![ncb.clone()])
        .unwrap()
        .apply(&events)
        .unwrap();
    let b = Tower::new(vec![scale]).unwrap().apply(&events).unwrap();
    let col = |pd: &PredictiveDistribution, kind: &str| {
        prospicio_prob::Empirical::draws(
            &pd.marginal(&vec![KeyValue::from(kind), KeyValue::from("cat")])
                .unwrap(),
        )
        .to_vec()
    };
    let (bonus, comm, ceded) = (
        col(&a, "no_claims_bonus"),
        col(&b, "ceding_commission"),
        col(&a, "ceded"),
    );
    let mut clean = 0;
    for j in 0..bonus.len() {
        assert!((bonus[j] - comm[j]).abs() < 1e-12, "year {j}");
        assert_eq!(bonus[j], if ceded[j] == 0.0 { 1.0 } else { 0.0 });
        clean += usize::from(ceded[j] == 0.0);
    }
    assert!(clean > 1_000 && clean < 5_000, "{clean}");
    // On the grid, the expected bonus is the simulated mean's limit.
    let grid = Tower::new(vec![ncb])
        .unwrap()
        .on_annual_grid(
            &prospicio_prob::Grid::new(
                1.0,
                vec![
                    0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25,
                ],
            )
            .unwrap(),
        )
        .unwrap();
    // An annual 0 cedes nothing, 11 and 12 cede 1 and 2: clean half the
    // time, 10% of a premium of 10 then.
    assert!((grid.expected_no_claims_bonus[0] - 0.5).abs() < 1e-12);
}

#[test]
fn commissions_on_hand_worked_years() {
    // A 50% quota share of a subject premium of 200: premium 100.
    let qs = || {
        Layer::quota_share("QS", 0.5)
            .unwrap()
            .premium_rate(1.0, 200.0)
            .unwrap()
    };
    let flat = qs().ceding_commission(0.3).unwrap();
    assert!((flat.ceding_commission_for(0.0) - 30.0).abs() < 1e-12);
    assert!((flat.ceding_commission_for(500.0) - 30.0).abs() < 1e-12);
    assert_eq!(flat.premium_for(500.0), 100.0);
    assert_eq!(flat.profit_commission_for(10.0), 0.0);

    // aggregate's worked example: 45% at 60%, 25% at 70%, 19% at 80%,
    // given out of order.
    let slide = qs()
        .sliding_scale(vec![(0.25, 0.70), (0.45, 0.60), (0.19, 0.80)])
        .unwrap();
    for (lr, c) in [
        (0.55, 0.45),
        (0.60, 0.45),
        (0.65, 0.35),
        (0.70, 0.25),
        (0.75, 0.22),
        (0.80, 0.19),
        (0.90, 0.19),
    ] {
        let got = slide.ceding_commission_for(100.0 * lr);
        assert!((got - 100.0 * c).abs() < 1e-10, "{lr}: {got}");
    }
    // A one-anchor scale is flat.
    let one = qs().sliding_scale(vec![(0.3, 0.6)]).unwrap();
    assert!((one.ceding_commission_for(90.0) - 30.0).abs() < 1e-12);
    // A sliding scale replaces a flat commission.
    let replaced = qs()
        .ceding_commission(0.1)
        .unwrap()
        .sliding_scale(vec![(0.3, 0.6)])
        .unwrap();
    assert!((replaced.ceding_commission_for(10.0) - 30.0).abs() < 1e-12);

    // aggregate's profit commission example: 25% after 10%.
    let pc = qs().profit_commission(0.25, 0.10).unwrap();
    for (lr, want) in [(0.50, 0.10), (0.60, 0.075), (0.90, 0.0), (0.95, 0.0)] {
        assert!((pc.profit_commission_for(100.0 * lr) - 100.0 * want).abs() < 1e-10);
    }
    // With a flat commission too: each is figured on its own.
    let both = qs()
        .ceding_commission(0.3)
        .unwrap()
        .profit_commission(0.25, 0.4)
        .unwrap();
    assert!((both.ceding_commission_for(40.0) - 30.0).abs() < 1e-12);
    assert!((both.profit_commission_for(40.0) - 5.0).abs() < 1e-12);
}

#[test]
fn swing_and_retro_premiums() {
    // aggregate's swing example: basic 0, lcm 1, between 100 and 300.
    let terms = LossSensitivePremium::new(0.0, 1.0, Some(100.0), Some(300.0)).unwrap();
    assert_eq!(terms.premium(50.0), 100.0);
    assert_eq!(terms.premium(200.0), 200.0);
    assert_eq!(terms.premium(400.0), 300.0);
    // The minimum defaults to the basic premium, the maximum to no cap.
    let retro = LossSensitivePremium::new(1000.0, 1.1, None, None).unwrap();
    assert_eq!(retro.minimum, 1000.0);
    assert!((retro.premium(1e6) - 1_101_000.0).abs() < 1e-6);
    // At a placed share of 40%, the amounts scale and the factor does
    // not: clip(0.4 × 20 + 1.25 × ceded, 0.4 × 50, 0.4 × 300).
    let swing = LossSensitivePremium::new(20.0, 1.25, Some(50.0), Some(300.0)).unwrap();
    let layer = Layer::xol("L", 1000.0, 0.0)
        .unwrap()
        .share(0.4)
        .unwrap()
        .swing_rated(swing)
        .unwrap();
    assert!((layer.premium_for(0.0) - 20.0).abs() < 1e-12);
    assert!((layer.premium_for(40.0) - 58.0).abs() < 1e-12);
    assert!((layer.premium_for(1000.0) - 120.0).abs() < 1e-12);
    // A flat commission applies to the swing premium.
    let comm = layer.clone().ceding_commission(0.1).unwrap();
    assert!((comm.ceding_commission_for(40.0) - 5.8).abs() < 1e-12);

    assert!(LossSensitivePremium::new(-1.0, 1.0, None, None).is_err());
    assert!(LossSensitivePremium::new(1.0, f64::NAN, None, None).is_err());
    assert!(LossSensitivePremium::new(1.0, 1.0, Some(5.0), Some(4.0)).is_err());
    assert!(LossSensitivePremium::new(1.0, 1.0, Some(f64::INFINITY), None).is_err());
}

#[test]
fn refuses_terms_that_do_not_combine() {
    let qs = || Layer::quota_share("QS", 0.5).unwrap();
    let priced = || qs().premium_rate(1.0, 200.0).unwrap();
    let swing = LossSensitivePremium::new(0.0, 1.0, None, None).unwrap();
    // A loss ratio needs a premium.
    assert!(qs().sliding_scale(vec![(0.3, 0.6)]).is_err());
    assert!(qs().profit_commission(0.2, 0.1).is_err());
    // Swing rating against loss-ratio terms and paid reinstatements, in
    // either order.
    assert!(
        priced()
            .swing_rated(swing)
            .unwrap()
            .sliding_scale(vec![(0.3, 0.6)])
            .is_err()
    );
    assert!(
        priced()
            .sliding_scale(vec![(0.3, 0.6)])
            .unwrap()
            .swing_rated(swing)
            .is_err()
    );
    assert!(
        priced()
            .profit_commission(0.2, 0.1)
            .unwrap()
            .swing_rated(swing)
            .is_err()
    );
    let xl = || Layer::xol("10x10", 10.0, 10.0).unwrap();
    assert!(
        xl().paid_reinstatements(1.0, vec![1.0])
            .unwrap()
            .swing_rated(swing)
            .is_err()
    );
    assert!(
        xl().swing_rated(swing)
            .unwrap()
            .paid_reinstatements(1.0, vec![1.0])
            .is_err()
    );
    assert!(
        xl().paid_reinstatements(1.0, vec![1.0])
            .unwrap()
            .profit_commission(0.2, 0.1)
            .is_err()
    );
    assert!(
        xl().rate_on_line(0.1)
            .unwrap()
            .sliding_scale(vec![(0.3, 0.6)])
            .unwrap()
            .paid_reinstatements(1.0, vec![1.0])
            .is_err()
    );
    // A flat commission combines with paid reinstatements and swing.
    assert!(
        xl().paid_reinstatements(1.0, vec![1.0])
            .unwrap()
            .ceding_commission(0.1)
            .is_ok()
    );
    assert!(
        xl().swing_rated(swing)
            .unwrap()
            .ceding_commission(0.1)
            .is_ok()
    );
    // Bad values.
    assert!(priced().ceding_commission(1.5).is_err());
    assert!(priced().sliding_scale(vec![]).is_err());
    assert!(
        priced()
            .sliding_scale(vec![(0.2, 0.6), (0.3, 0.7)])
            .is_err()
    );
    assert!(
        priced()
            .sliding_scale(vec![(0.3, 0.6), (0.2, 0.6)])
            .is_err()
    );
    assert!(priced().sliding_scale(vec![(1.2, 0.6)]).is_err());
    assert!(priced().profit_commission(1.5, 0.1).is_err());
    assert!(priced().profit_commission(0.2, -0.1).is_err());
}

#[test]
fn tower_reports_contract_terms() {
    let swing = LossSensitivePremium::new(2e5, 1.25, Some(4e5), Some(3e6)).unwrap();
    let tower = Tower::inuring(vec![
        vec![
            Layer::quota_share("QS", 0.3)
                .unwrap()
                .premium_rate(1.0, 20e6)
                .unwrap()
                .sliding_scale(vec![(0.35, 0.5), (0.25, 0.7)])
                .unwrap()
                .profit_commission(0.2, 0.3)
                .unwrap(),
        ],
        vec![
            Layer::xol("5x5", 5e6, 5e6)
                .unwrap()
                .swing_rated(swing)
                .unwrap(),
        ],
    ])
    .unwrap();
    let result = tower.apply(&events()).unwrap();
    let kinds: Vec<String> = result
        .components()
        .iter()
        .map(|k| format!("{}/{}", k[0], k[1]))
        .collect();
    assert_eq!(
        kinds,
        [
            "gross/ground_up",
            "ceded/QS",
            "ceded/5x5",
            "net/retained",
            "swing_premium/5x5",
            "ceding_commission/QS",
            "profit_commission/QS"
        ]
    );
    let (qs, xl) = (&tower.layers[0], &tower.layers[1]);
    let mut slid = 0;
    for sim in 0..result.n_sims() {
        let row = result.row(sim).unwrap();
        assert!((row[0] - row[1] - row[2] - row[3]).abs() <= 1e-6 * row[0].max(1.0));
        assert_eq!(row[4], xl.premium_for(row[2]));
        assert_eq!(row[5], qs.ceding_commission_for(row[1]));
        assert_eq!(row[6], qs.profit_commission_for(row[1]));
        let lr = row[1] / 6e6;
        slid += usize::from(lr > 0.5 && lr < 0.7);
    }
    // The scale slides in some years, not only at its ends.
    assert!(slid > 1_000, "{slid}");
    assert!(
        result
            .provenance()
            .parameters
            .iter()
            .any(|(k, v)| k == "layer:QS" && v.contains("sliding scale"))
    );
}

fn events() -> EventSet {
    simulate_events(
        &Poisson::new(2.0).unwrap(),
        &Lognormal::from_mean_cv(3e6, 1.5).unwrap(),
        100_000,
        11,
    )
    .unwrap()
}

#[test]
fn gross_equals_ceded_plus_net_every_year() {
    let tower = Tower::new(vec![
        Layer::xol("5x5", 5e6, 5e6)
            .unwrap()
            .reinstatements(1)
            .unwrap(),
        Layer::xol("15x10", 15e6, 10e6).unwrap().share(0.6).unwrap(),
    ])
    .unwrap();
    let result = tower.apply(&events()).unwrap();
    assert_eq!(result.n_components(), 4);
    for sim in 0..result.n_sims() {
        let row = result.row(sim).unwrap();
        assert!((row[0] - row[1] - row[2] - row[3]).abs() <= 1e-6 * row[0].max(1.0));
    }
    let by_kind = result.aggregate(&["kind"]).unwrap();
    let kinds: Vec<_> = by_kind
        .components()
        .iter()
        .map(|k| k[0].to_string())
        .collect();
    assert_eq!(kinds, ["gross", "ceded", "net"]);
    assert_eq!(result.provenance().seed, Some(11));
}

#[test]
fn mean_ceded_matches_the_exact_layer_value() {
    // With no annual terms, E[ceded] = E[N] × E[min(max(X - a, 0), l)].
    let sev = Lognormal::from_mean_cv(3e6, 1.5).unwrap();
    let layer = Layer::xol("5x5", 5e6, 5e6).unwrap();
    let exact = 2.0 * sev.layer(5e6, 5e6);
    let result = Tower::new(vec![layer]).unwrap().apply(&events()).unwrap();
    let ceded = result
        .marginal(&vec![KeyValue::from("ceded"), KeyValue::from("5x5")])
        .unwrap();
    // Four standard errors.
    let se = ceded.std_dev() / (ceded.len() as f64).sqrt();
    assert!(
        (ceded.mean() - exact).abs() < 4.0 * se,
        "{} vs {exact}",
        ceded.mean()
    );
}

#[test]
fn quota_share_and_stop_loss() {
    let qs = Layer::quota_share("QS", 0.25).unwrap();
    assert_eq!(qs.ceded(&[8.0, 4.0]), 3.0);
    assert!(Layer::quota_share("QS", 0.0).is_err());
    let sl = Layer::stop_loss("SL", 50.0, 100.0).unwrap();
    assert_eq!(sl.ceded(&[60.0]), 0.0);
    assert_eq!(sl.ceded(&[60.0, 70.0]), 30.0);
    assert_eq!(sl.ceded(&[200.0]), 50.0);
    assert!(Layer::stop_loss("SL", 50.0, f64::INFINITY).is_err());
}

#[test]
fn pro_rata_scales_every_event_by_the_same_factor() {
    // #476's rule: u = 3, 10, 7 after the occurrence terms, U = 20; the
    // deductible 4 and annual limit 15 leave 15, so f = 0.75.
    let layer = Layer::xol("L", 10.0, 5.0)
        .unwrap()
        .aggregate_deductible(4.0)
        .unwrap()
        .aggregate_limit(15.0)
        .unwrap()
        .share(0.5)
        .unwrap();
    let losses = [8.0, 20.0, 12.0];
    let pro_rata = layer.clone().allocation(LimitAllocation::ProRata).unwrap();
    let split = pro_rata.ceded_by_event(&losses);
    for (got, want) in split.iter().zip([1.125, 3.75, 2.625]) {
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
    // The annual total is the same under both rules.
    let total: f64 = split.iter().sum();
    assert!((total - layer.ceded(&losses)).abs() < 1e-12);
    assert_eq!(pro_rata.ceded(&losses), layer.ceded(&losses));
    // When the annual terms do not bite, the rules agree.
    let free = Layer::xol("F", 10.0, 5.0).unwrap();
    let free_pr = free.clone().allocation(LimitAllocation::ProRata).unwrap();
    assert_eq!(
        free.ceded_by_event(&losses),
        free_pr.ceded_by_event(&losses)
    );
    // No recovery, nothing ceded.
    assert_eq!(pro_rata.ceded_by_event(&[1.0, 2.0]), [0.0, 0.0]);
    assert!(pro_rata.ceded_by_event(&[]).is_empty());
    // Not with reinstatements pro rata as to time, either way round.
    let timed = Layer::xol("T", 10.0, 5.0)
        .unwrap()
        .paid_reinstatements(1.0, vec![1.0])
        .unwrap();
    assert!(
        timed
            .clone()
            .pro_rata_as_to_time()
            .unwrap()
            .allocation(LimitAllocation::ProRata)
            .is_err()
    );
    assert!(
        timed
            .allocation(LimitAllocation::ProRata)
            .unwrap()
            .pro_rata_as_to_time()
            .is_err()
    );
}

#[test]
fn pro_rata_changes_what_a_later_stage_sees() {
    // Stage 1: 10 xs 5 with one reinstatement (20 a year) on 15, 20, 10.
    // In event order it cedes 10, 10, 0 and leaves 5, 10, 10; pro rata
    // 8, 8, 4 and leaves 7, 12, 6. Stage 2, 5 xs 5, recovers 0 + 5 + 5
    // against 2 + 5 + 1.
    let first = Layer::xol("10x5", 10.0, 5.0)
        .unwrap()
        .reinstatements(1)
        .unwrap();
    let second = Layer::xol("5x5", 5.0, 5.0).unwrap();
    let losses = [15.0, 20.0, 10.0];
    let order = Tower::inuring(vec![vec![first.clone()], vec![second.clone()]]).unwrap();
    let pro_rata = Tower::inuring(vec![
        vec![first.allocation(LimitAllocation::ProRata).unwrap()],
        vec![second],
    ])
    .unwrap();
    assert_eq!(order.ceded(&losses), [20.0, 10.0]);
    assert_eq!(pro_rata.ceded(&losses), [20.0, 8.0]);
}

#[test]
fn labels_split_gross_ceded_and_net() {
    // Two years: wind 15 and quake 20, then quake 10 and wind 4.
    let events = EventSet::from_years(vec![vec![15.0, 20.0], vec![10.0, 4.0]], 3)
        .unwrap()
        .with_labels(vec![(
            "peril".into(),
            vec!["wind".into(), "quake".into(), "quake".into(), "wind".into()],
        )])
        .unwrap();
    let layer = Layer::xol("10x5", 10.0, 5.0)
        .unwrap()
        .aggregate_limit(15.0)
        .unwrap()
        .ceding_commission(0.1)
        .unwrap();
    let tower = Tower::new(vec![layer]).unwrap();
    let labeled = tower.apply(&events).unwrap();
    assert_eq!(labeled.dims(), ["kind", "layer", "peril"]);
    let key = |k: &str, l: &str, p: &str| -> Vec<KeyValue> { vec![k.into(), l.into(), p.into()] };
    let draws = |k: &str, l: &str, p: &str| {
        prospicio_prob::Empirical::draws(&labeled.marginal(&key(k, l, p)).unwrap()).to_vec()
    };
    // Year 1 recovers 10 (wind) then 5 of quake's 10 under the limit;
    // year 2 recovers 5 (quake) and nothing on 4.
    assert_eq!(draws("gross", "ground_up", "wind"), [15.0, 4.0]);
    assert_eq!(draws("ceded", "10x5", "wind"), [10.0, 0.0]);
    assert_eq!(draws("ceded", "10x5", "quake"), [5.0, 5.0]);
    assert_eq!(draws("net", "retained", "quake"), [15.0, 5.0]);
    assert_eq!(draws("ceding_commission", "10x5", ""), [0.0, 0.0]);
    // Summed over labels it is the unlabelled result.
    let plain = tower
        .apply(&EventSet::from_years(vec![vec![15.0, 20.0], vec![10.0, 4.0]], 3).unwrap())
        .unwrap();
    let back = labeled.aggregate(&["kind", "layer"]).unwrap();
    for k in plain.components() {
        assert_eq!(
            prospicio_prob::Empirical::draws(&back.marginal(k).unwrap()),
            prospicio_prob::Empirical::draws(&plain.marginal(k).unwrap()),
        );
    }
    // Pro rata, quake's share of year 1 is 15 × 10 / 20.
    let pro = Tower::new(vec![
        Layer::xol("10x5", 10.0, 5.0)
            .unwrap()
            .aggregate_limit(15.0)
            .unwrap()
            .allocation(LimitAllocation::ProRata)
            .unwrap(),
    ])
    .unwrap()
    .apply(&events)
    .unwrap();
    let quake = pro.marginal(&key("ceded", "10x5", "quake")).unwrap();
    assert_eq!(prospicio_prob::Empirical::draws(&quake), [7.5, 5.0]);
}

#[test]
fn a_pro_rata_stop_loss_splits_by_each_label_s_share() {
    // A stop loss of 20 xs 30 on the annual total, pro rata across the
    // year's events: each label gets its share of the gross loss.
    let events = EventSet::from_years(vec![vec![30.0, 10.0, 20.0], vec![5.0, 15.0]], 0)
        .unwrap()
        .with_labels(vec![(
            "peril".into(),
            vec![
                "wind".into(),
                "quake".into(),
                "wind".into(),
                "quake".into(),
                "wind".into(),
            ],
        )])
        .unwrap();
    let cover = Layer::stop_loss("sl", 20.0, 30.0)
        .unwrap()
        .allocation(LimitAllocation::ProRata)
        .unwrap();
    let result = Tower::new(vec![cover]).unwrap().apply(&events).unwrap();
    let at = |k: &str, l: &str, p: &str| {
        let key: Vec<KeyValue> = vec![k.into(), l.into(), p.into()];
        prospicio_prob::Empirical::draws(&result.marginal(&key).unwrap()).to_vec()
    };
    // Year 1: gross 60 (wind 50, quake 10), ceded 20, a third of each.
    let (wind, quake) = (at("ceded", "sl", "wind"), at("ceded", "sl", "quake"));
    assert!((wind[0] - 50.0 / 3.0).abs() < 1e-12 && (quake[0] - 10.0 / 3.0).abs() < 1e-12);
    // Year 2: gross 20 is below the retention.
    assert_eq!((wind[1], quake[1]), (0.0, 0.0));
    let gross = at("gross", "ground_up", "wind");
    assert!((wind[0] / gross[0] - 1.0 / 3.0).abs() < 1e-12);
}

#[test]
fn labels_split_what_an_inuring_stage_sees() {
    let events = EventSet::from_years(vec![vec![15.0, 20.0, 10.0]], 0)
        .unwrap()
        .with_labels(vec![(
            "class".into(),
            vec!["a".into(), "b".into(), "a".into()],
        )])
        .unwrap();
    let first = Layer::xol("10x5", 10.0, 5.0)
        .unwrap()
        .reinstatements(1)
        .unwrap();
    let second = Layer::xol("5x5", 5.0, 5.0).unwrap();
    let tower = Tower::inuring(vec![vec![first], vec![second]]).unwrap();
    let result = tower.apply(&events).unwrap();
    let at = |l: &str, c: &str| {
        let key: Vec<KeyValue> = vec!["ceded".into(), l.into(), c.into()];
        prospicio_prob::Empirical::draws(&result.marginal(&key).unwrap())[0]
    };
    // Nets behind the first stage: 5, 10, 10, so 0 + 5 + 5 for 5x5.
    assert_eq!((at("10x5", "a"), at("10x5", "b")), (10.0, 10.0));
    assert_eq!((at("5x5", "a"), at("5x5", "b")), (5.0, 5.0));
}

fn perils() -> EventSet {
    // Two years: quake 20 then wind 15, and wind 8, flood 30, wind 12.
    EventSet::from_years(vec![vec![20.0, 15.0], vec![8.0, 30.0, 12.0]], 4)
        .unwrap()
        .with_times(vec![0.1, 0.4, 0.2, 0.5, 0.9])
        .unwrap()
        .with_labels(vec![(
            "peril".into(),
            vec![
                "quake".into(),
                "wind".into(),
                "wind".into(),
                "flood".into(),
                "wind".into(),
            ],
        )])
        .unwrap()
}

fn draws_of(pd: &PredictiveDistribution, key: &[&str]) -> Vec<f64> {
    let key: Vec<KeyValue> = key.iter().map(|&k| k.into()).collect();
    prospicio_prob::Empirical::draws(&pd.marginal(&key).unwrap()).to_vec()
}

#[test]
fn a_scope_over_every_value_is_no_scope() {
    let layer = Layer::xol("10x5", 10.0, 5.0)
        .unwrap()
        .reinstatements(1)
        .unwrap();
    let all = layer
        .clone()
        .scope("peril", &["quake", "wind", "flood"])
        .unwrap();
    let plain = Tower::new(vec![layer]).unwrap().apply(&perils()).unwrap();
    let scoped = Tower::new(vec![all]).unwrap().apply(&perils()).unwrap();
    assert_eq!(plain.components(), scoped.components());
    for k in plain.components() {
        assert_eq!(
            prospicio_prob::Empirical::draws(&plain.marginal(k).unwrap()),
            prospicio_prob::Empirical::draws(&scoped.marginal(k).unwrap()),
        );
    }
}

#[test]
fn a_scoped_layer_cedes_what_it_cedes_on_the_filtered_set() {
    let layer = Layer::xol("10x5", 10.0, 5.0)
        .unwrap()
        .aggregate_limit(15.0)
        .unwrap()
        .paid_reinstatements(2.0, vec![1.0])
        .unwrap()
        .ceding_commission(0.1)
        .unwrap();
    let scoped = Tower::new(vec![layer.clone().scope("peril", &["wind"]).unwrap()])
        .unwrap()
        .apply(&perils())
        .unwrap()
        .aggregate(&["kind", "layer"])
        .unwrap();
    let wind = perils().filter("peril", &["wind"]).unwrap();
    assert_eq!(wind.events(0), &[15.0]);
    assert_eq!(wind.events(1), &[8.0, 12.0]);
    assert_eq!(wind.times(1), Some(&[0.2, 0.9][..]));
    let alone = Tower::new(vec![layer])
        .unwrap()
        .apply(&wind)
        .unwrap()
        .aggregate(&["kind", "layer"])
        .unwrap();
    for k in alone
        .components()
        .iter()
        .filter(|k| k[0].to_string() != "gross" && k[0].to_string() != "net")
    {
        assert_eq!(
            prospicio_prob::Empirical::draws(&scoped.marginal(k).unwrap()),
            prospicio_prob::Empirical::draws(&alone.marginal(k).unwrap()),
            "{k:?}"
        );
    }
    // Wind recovers 10, then 3 and 7 within the limit of 15.
    assert_eq!(draws_of(&scoped, &["ceded", "10x5"]), [10.0, 10.0]);
    // The net keeps every peril's loss.
    assert_eq!(draws_of(&scoped, &["net", "retained"]), [25.0, 40.0]);
}

#[test]
fn a_later_stage_sees_the_out_of_scope_losses_whole() {
    let wind_xl = Layer::xol("10x5", 10.0, 5.0)
        .unwrap()
        .scope("peril", &["wind"])
        .unwrap();
    let stop_loss = Layer::stop_loss("sl", 100.0, 20.0).unwrap();
    let tower = Tower::inuring(vec![vec![wind_xl], vec![stop_loss]]).unwrap();
    let result = tower.apply(&perils()).unwrap();
    // Year 1: wind 15 cedes 10, so the stop loss sees 20 + 5 = 25.
    // Year 2: wind 8 and 12 cede 3 and 7, so it sees 5 + 30 + 5 = 40.
    assert_eq!(draws_of(&result, &["ceded", "10x5", "wind"]), [10.0, 10.0]);
    assert_eq!(draws_of(&result, &["ceded", "10x5", "quake"]), [0.0, 0.0]);
    let sl: Vec<f64> = ["quake", "wind", "flood"]
        .iter()
        .map(|p| draws_of(&result, &["ceded", "sl", p]))
        .fold(vec![0.0; 2], |a, d| {
            a.iter().zip(d).map(|(x, y)| x + y).collect()
        });
    assert_eq!(sl, [5.0, 20.0]);
}

#[test]
fn a_scope_masks_the_aggregate_part_too() {
    // An attritional part by peril: wind 10 and 40, quake 30 and 5.
    let part = PredictiveDistribution::from_draws(
        vec!["peril".into()],
        vec![vec!["quake".into()], vec!["wind".into()]],
        vec![30.0, 10.0, 5.0, 40.0],
        Provenance::new("part"),
    )
    .unwrap();
    let events = perils().with_aggregate_part(&part).unwrap();
    let sl = Layer::stop_loss("SL", 30.0, 20.0).unwrap();
    let scoped = Tower::new(vec![sl.clone().scope("peril", &["wind"]).unwrap()])
        .unwrap()
        .apply(&events)
        .unwrap()
        .aggregate(&["kind", "layer"])
        .unwrap();
    let wind = events.filter("peril", &["wind"]).unwrap();
    assert_eq!(wind.aggregate_part(1), Some(&[40.0][..]));
    let alone = Tower::new(vec![sl])
        .unwrap()
        .apply(&wind)
        .unwrap()
        .aggregate(&["kind", "layer"])
        .unwrap();
    // Wind's years: 10 + 15 = 25 and 40 + 20 = 60, so 5 and 30.
    assert_eq!(draws_of(&scoped, &["ceded", "SL"]), [5.0, 30.0]);
    assert_eq!(draws_of(&alone, &["ceded", "SL"]), [5.0, 30.0]);
}

#[test]
fn ceded_events_add_to_the_layers_ceded_totals() {
    let events = events();
    let tower = Tower::inuring(vec![
        vec![Layer::quota_share("QS", 0.3).unwrap()],
        vec![
            Layer::xol("5x5", 5e6, 5e6)
                .unwrap()
                .reinstatements(1)
                .unwrap(),
            Layer::xol("10x10", 10e6, 10e6)
                .unwrap()
                .aggregate_limit(12e6)
                .unwrap()
                .allocation(LimitAllocation::ProRata)
                .unwrap(),
        ],
    ])
    .unwrap();
    let result = tower.apply(&events).unwrap();
    for names in [&["QS"][..], &["5x5"], &["10x10"], &["5x5", "10x10"]] {
        let ceded = tower.ceded_events(&events, names).unwrap();
        assert_eq!(ceded.n_sims(), events.n_sims());
        let totals: Vec<Vec<f64>> = names
            .iter()
            .map(|n| {
                let key: Vec<KeyValue> = vec!["ceded".into(), (*n).into()];
                prospicio_prob::Empirical::draws(&result.marginal(&key).unwrap()).to_vec()
            })
            .collect();
        for i in 0..events.n_sims() {
            let want: f64 = totals.iter().map(|t| t[i]).sum();
            let got: f64 = ceded.events(i).iter().sum();
            assert!(
                (got - want).abs() <= 1e-6 * want.max(1.0),
                "{names:?} year {i}"
            );
            assert!(ceded.events(i).iter().all(|&x| x > 0.0));
        }
    }
}

#[test]
fn an_occurrence_retro_on_two_deals_ceded_events() {
    // Deal A: wind 30 at 0.2 under 20 xs 10, ceding 20. Deal B: the
    // same wind event, 25 at 0.2, and quake 12 at 0.6 under 10 xs 10,
    // ceding 10 and 2.
    let a = EventSet::from_years(vec![vec![30.0]], 0)
        .unwrap()
        .with_times(vec![0.2])
        .unwrap()
        .with_labels(vec![("peril".into(), vec!["wind".into()])])
        .unwrap();
    let b = EventSet::from_years(vec![vec![25.0, 12.0]], 1)
        .unwrap()
        .with_times(vec![0.2, 0.6])
        .unwrap()
        .with_labels(vec![("peril".into(), vec!["wind".into(), "quake".into()])])
        .unwrap();
    let ca = Tower::new(vec![Layer::xol("A", 20.0, 10.0).unwrap()])
        .unwrap()
        .ceded_events(&a, &["A"])
        .unwrap();
    let cb = Tower::new(vec![Layer::xol("B", 10.0, 10.0).unwrap()])
        .unwrap()
        .ceded_events(&b, &["B"])
        .unwrap();
    let book = ca.merge(&cb).unwrap();
    assert_eq!(book.events(0), &[20.0, 10.0, 2.0]);
    assert_eq!(book.times(0), Some(&[0.2, 0.2, 0.6][..]));
    assert_eq!(
        book.label_combinations(),
        [vec!["quake".to_string()], vec!["wind".to_string()]]
    );
    // A 15 xs 10 retro: each deal's cession its own occurrence (10), or
    // the wind event's cessions grouped into one occurrence of 30 (15).
    let retro = Tower::new(vec![Layer::xol("retro", 15.0, 10.0).unwrap()]).unwrap();
    assert_eq!(retro.ceded(book.events(0)), [10.0]);
    let one = book.grouped(72.0).unwrap();
    assert_eq!(one.events(0), &[30.0, 2.0]);
    assert_eq!(retro.ceded(one.events(0)), [15.0]);
}

#[test]
fn ceded_events_follow_scopes_and_the_aggregate_part() {
    let part = PredictiveDistribution::from_draws(
        vec!["peril".into()],
        vec![vec!["wind".into()]],
        vec![8.0, 0.0],
        Provenance::new("part"),
    )
    .unwrap();
    let events = perils().with_aggregate_part(&part).unwrap();
    // A wind-scoped stop loss of 100 xs 20 on wind: 8 + 15 = 23 and
    // 0 + 8 + 12 = 20 cede 3 and 0, split pro rata to each entry.
    let sl = Layer::stop_loss("SL", 100.0, 20.0)
        .unwrap()
        .allocation(LimitAllocation::ProRata)
        .unwrap()
        .scope("peril", &["wind"])
        .unwrap();
    let ceded = Tower::new(vec![sl])
        .unwrap()
        .ceded_events(&events, &["SL"])
        .unwrap();
    let part0 = ceded.aggregate_part(0).unwrap()[0];
    assert!((part0 - 3.0 * 8.0 / 23.0).abs() < 1e-12);
    assert_eq!(ceded.events(0).len(), 1);
    assert!((ceded.events(0)[0] - 3.0 * 15.0 / 23.0).abs() < 1e-12);
    assert!(ceded.events(1).is_empty());
    assert!(ceded.sums_insured(0).is_none());
    // Errors: no layer, an unknown or repeated name.
    let tower = Tower::new(vec![Layer::xol("L", 10.0, 5.0).unwrap()]).unwrap();
    assert!(tower.ceded_events(&events, &[]).is_err());
    assert!(tower.ceded_events(&events, &["M"]).is_err());
    assert!(tower.ceded_events(&events, &["L", "L"]).is_err());
}

#[test]
fn a_scope_is_not_a_label_split() {
    // Quake 20 comes first: unscoped, it uses 10 of the limit of 15 and
    // wind's share is 5; scoped to wind, wind has the whole limit.
    let layer = Layer::xol("10x5", 10.0, 5.0)
        .unwrap()
        .aggregate_limit(15.0)
        .unwrap();
    let split = Tower::new(vec![layer.clone()])
        .unwrap()
        .apply(&perils())
        .unwrap();
    let scoped = Tower::new(vec![layer.scope("peril", &["wind"]).unwrap()])
        .unwrap()
        .apply(&perils())
        .unwrap();
    assert_eq!(draws_of(&split, &["ceded", "10x5", "wind"])[0], 5.0);
    assert_eq!(draws_of(&scoped, &["ceded", "10x5", "wind"])[0], 10.0);
}

#[test]
fn a_scope_needs_its_label() {
    let scoped = Tower::new(vec![
        Layer::xol("L", 10.0, 5.0)
            .unwrap()
            .scope("peril", &["wind"])
            .unwrap(),
    ])
    .unwrap();
    let plain = EventSet::from_years(vec![vec![15.0]], 0).unwrap();
    assert!(scoped.apply(&plain).is_err());
    let classes = plain
        .clone()
        .with_labels(vec![("class".into(), vec!["a".into()])])
        .unwrap();
    assert!(scoped.apply(&classes).is_err());
    let other = plain
        .with_labels(vec![("peril".into(), vec!["hail".into()])])
        .unwrap();
    assert!(scoped.apply(&other).is_err());
    assert!(scoped.apply_annual(&[15.0]).is_err());
    assert!(
        Layer::xol("L", 1.0, 1.0)
            .unwrap()
            .scope("peril", &[])
            .is_err()
    );
    assert!(perils().filter("class", &["a"]).is_err());
    assert!(
        EventSet::from_years(vec![vec![1.0]], 0)
            .unwrap()
            .filter("peril", &["a"])
            .is_err()
    );
}

#[test]
fn a_franchise_pays_from_the_ground_up() {
    let f = Layer::xol("F", 20.0, 5.0).unwrap().franchise().unwrap();
    assert_eq!(f.recovery(4.999), 0.0);
    assert_eq!(f.recovery(5.0), 5.0);
    assert_eq!(f.recovery(30.0), 20.0);
    // Above the largest loss it pays nothing.
    let sev = Lognormal::from_mean_cv(10.0, 1.0).unwrap();
    let ev = simulate_events(&Poisson::new(3.0).unwrap(), &sev, 500, 2).unwrap();
    let largest = (0..500)
        .flat_map(|i| ev.events(i).to_vec())
        .fold(0.0, f64::max);
    let high = Layer::xol("H", 1e9, largest * 1.01)
        .unwrap()
        .franchise()
        .unwrap();
    let r = Tower::new(vec![high]).unwrap().apply(&ev).unwrap();
    assert!(column(&r, &["ceded", "H"]).iter().all(|&c| c == 0.0));
    // On the grid, the franchise recovery of each point.
    let sev = prospicio_prob::Grid::new(1.0, vec![0.0, 0.2, 0.3, 0.3, 0.2]).unwrap();
    let tower = Tower::new(vec![
        Layer::xol("F", 10.0, 2.0).unwrap().franchise().unwrap(),
    ])
    .unwrap();
    let g = tower
        .on_grid(&Poisson::new(1.0).unwrap(), &sev, 64)
        .unwrap();
    // E[N] E[recovery] = 1 × (0.3 × 2 + 0.3 × 3 + 0.2 × 4).
    assert!((g.ceded[0].mean() - 2.3).abs() < 1e-9);
    assert!(Layer::surplus("S", 1.0, 2.0).unwrap().franchise().is_err());
}

#[test]
fn an_nth_event_cover_skips_the_first_qualifying_events() {
    let l = Layer::xol("2nd", 10.0, 5.0)
        .unwrap()
        .nth_event(2, 5.0)
        .unwrap();
    assert_eq!(
        l.ceded_by_event(&[12.0, 3.0, 9.0, 20.0]),
        [0.0, 0.0, 4.0, 10.0]
    );
    // Third and later events above 8.
    let third = Layer::xol("3rd", 10.0, 5.0)
        .unwrap()
        .nth_event(3, 8.0)
        .unwrap();
    assert_eq!(
        third.ceded_by_event(&[9.0, 6.0, 12.0, 7.0, 15.0]),
        [0.0, 0.0, 0.0, 0.0, 10.0]
    );
    // nth_event(1, attachment) is the plain layer on simulated years.
    let sev = Lognormal::from_mean_cv(10.0, 1.5).unwrap();
    let ev = simulate_events(&Poisson::new(4.0).unwrap(), &sev, 2_000, 5).unwrap();
    let plain = Layer::xol("L", 20.0, 8.0)
        .unwrap()
        .aggregate_limit(40.0)
        .unwrap();
    let first = plain.clone().nth_event(1, 8.0).unwrap();
    let a = Tower::new(vec![plain]).unwrap().apply(&ev).unwrap();
    let b = Tower::new(vec![first.clone()]).unwrap().apply(&ev).unwrap();
    assert_eq!(a.draw_matrix(), b.draw_matrix());
    // Paths without events in order refuse it.
    let t = Tower::new(vec![first]).unwrap();
    assert!(t.apply_annual(&[30.0]).is_err());
    let sev = prospicio_prob::Grid::new(1.0, vec![0.0, 0.5, 0.5]).unwrap();
    assert!(t.on_grid(&Poisson::new(1.0).unwrap(), &sev, 16).is_err());
    assert!(
        Layer::xol("L", 1.0, 1.0)
            .unwrap()
            .nth_event(0, 1.0)
            .is_err()
    );
}

#[test]
fn a_top_and_drop_cover_drops_once_the_underlying_layer_is_used_up() {
    let lower = Layer::xol("lower", 10.0, 10.0)
        .unwrap()
        .aggregate_limit(20.0)
        .unwrap();
    let top = Layer::xol("top", 10.0, 30.0)
        .unwrap()
        .top_and_drop("lower", 10.0)
        .unwrap();
    let tower = Tower::new(vec![lower.clone(), top.clone()]).unwrap();
    let ev = EventSet::from_years(vec![vec![25.0, 25.0, 25.0, 40.0]], 0).unwrap();
    let ce = tower.ceded_events(&ev, &["top"]).unwrap();
    // The lower layer is used up by the second event; the top then sits
    // at 10 and pays 10 on each later event.
    assert_eq!(ce.events(0), [10.0, 10.0]);
    let r = tower.apply(&ev).unwrap();
    assert_eq!(column(&r, &["ceded", "lower"]), [20.0]);
    assert_eq!(column(&r, &["ceded", "top"]), [20.0]);
    // An underlying layer that never exhausts leaves the plain layer.
    let sev = Lognormal::from_mean_cv(5.0, 1.0).unwrap();
    let years = simulate_events(&Poisson::new(2.0).unwrap(), &sev, 2_000, 8).unwrap();
    let roomy = lower.clone().aggregate_limit(1e12).unwrap();
    let drop = Tower::new(vec![roomy.clone(), top.clone()])
        .unwrap()
        .apply(&years)
        .unwrap();
    let plain_top = Layer::xol("top", 10.0, 30.0).unwrap();
    let plain = Tower::new(vec![roomy, plain_top])
        .unwrap()
        .apply(&years)
        .unwrap();
    assert_eq!(drop.draw_matrix(), plain.draw_matrix());
    // The tower checks the underlying layer.
    assert!(Tower::new(vec![top.clone()]).is_err());
    assert!(Tower::new(vec![top.clone(), lower.clone()]).is_err());
    let open = Layer::xol("lower", 10.0, 10.0).unwrap();
    assert!(Tower::new(vec![open, top.clone()]).is_err());
    assert!(Tower::inuring(vec![vec![lower.clone()], vec![top.clone()]]).is_err());
    assert!(
        Layer::xol("t", 10.0, 30.0)
            .unwrap()
            .top_and_drop("lower", 30.0)
            .is_err()
    );
    assert!(
        Layer::xol("t", 10.0, 30.0)
            .unwrap()
            .franchise()
            .unwrap()
            .top_and_drop("lower", 10.0)
            .is_err()
    );
    assert!(tower.apply_annual(&[50.0]).is_err());
}

#[test]
fn ceded_by_event_uses_annual_terms_in_order() {
    let layer = Layer::xol("L", 10.0, 5.0)
        .unwrap()
        .aggregate_deductible(4.0)
        .unwrap()
        .aggregate_limit(15.0)
        .unwrap()
        .share(0.5)
        .unwrap();
    // Recoveries 3, 10, 7: the deductible takes 3 then 1, the limit
    // stops the last 5.
    assert_eq!(layer.ceded_by_event(&[8.0, 20.0, 12.0]), [0.0, 4.5, 3.0]);
    // The same losses in another order use the terms up differently
    // but cede the same annual total.
    assert_eq!(layer.ceded_by_event(&[20.0, 12.0, 8.0]), [3.0, 3.5, 1.0]);
    assert!(layer.ceded_by_event(&[]).is_empty());
    let big = Layer::xol("5x5", 5e6, 5e6)
        .unwrap()
        .aggregate_deductible(2e6)
        .unwrap()
        .reinstatements(1)
        .unwrap();
    let events = events();
    for sim in 0..events.n_sims() {
        let losses = events.events(sim);
        let split: f64 = big.ceded_by_event(losses).iter().sum();
        assert!((split - big.ceded(losses)).abs() <= 1e-6);
    }
}

#[test]
fn later_stages_see_losses_net_of_earlier_ones() {
    // Stage 1: 10 xs 5 with a 15 annual limit; on [20, 20] it pays 10
    // then 5, leaving 10 and 15. Stage 2: a stop-loss of 20 xs 20 on
    // that net total of 25 pays 5. A layer in stage 1 alongside it
    // still sees gross.
    let tower = Tower::inuring(vec![
        vec![
            Layer::xol("A", 10.0, 5.0)
                .unwrap()
                .aggregate_limit(15.0)
                .unwrap(),
            Layer::xol("B", 100.0, 18.0).unwrap(),
        ],
        vec![Layer::stop_loss("SL", 20.0, 20.0).unwrap()],
    ])
    .unwrap();
    assert_eq!(tower.stages, [0, 0, 1]);
    // B takes 2 + 2 of the gross, so stage 2 sees 8 + 13 = 21.
    assert_eq!(tower.ceded(&[20.0, 20.0]), [15.0, 4.0, 1.0]);
    assert!(Tower::inuring(vec![vec![], vec![Layer::xol("A", 1.0, 0.0).unwrap()]]).is_err());
    assert!(
        Tower::inuring(vec![
            vec![Layer::xol("A", 1.0, 0.0).unwrap()],
            vec![Layer::xol("A", 2.0, 0.0).unwrap()],
        ])
        .is_err()
    );
}

#[test]
fn quota_share_inuring_to_a_layer_scales_it() {
    // A cession c inuring to l xs a is (1 - c) × (l / (1 - c)) xs
    // (a / (1 - c)) on gross, in every simulated year.
    let c = 0.4;
    let inuring = Tower::inuring(vec![
        vec![Layer::quota_share("QS", c).unwrap()],
        vec![
            Layer::xol("XL", 5e6, 5e6)
                .unwrap()
                .reinstatements(1)
                .unwrap(),
        ],
    ])
    .unwrap();
    let scaled = Layer::xol("XL", 5e6 / (1.0 - c), 5e6 / (1.0 - c))
        .unwrap()
        .reinstatements(1)
        .unwrap()
        .share(1.0 - c)
        .unwrap();
    let events = events();
    for sim in 0..events.n_sims() {
        let losses = events.events(sim);
        let ceded = inuring.ceded(losses);
        assert!((ceded[0] - c * losses.iter().sum::<f64>()).abs() <= 1e-6);
        assert!(
            (ceded[1] - scaled.ceded(losses)).abs() <= 1e-6,
            "year {sim}"
        );
    }
    let result = inuring.apply(&events).unwrap();
    for sim in 0..result.n_sims() {
        let row = result.row(sim).unwrap();
        assert!((row[0] - row[1] - row[2] - row[3]).abs() <= 1e-6 * row[0].max(1.0));
    }
}

#[test]
fn reinstatement_premiums_pro_rata_as_to_time() {
    let amount = Layer::xol("10x10", 10.0, 10.0)
        .unwrap()
        .paid_reinstatements(2.0, vec![1.0, 0.5])
        .unwrap();
    let timed = amount.clone().pro_rata_as_to_time().unwrap();
    assert!(timed.needs_times() && !amount.needs_times());
    // Without times: NaN; times are ignored when only pro rata as to
    // amount.
    assert!(timed.reinstatement_premium(&[22.0]).is_nan());
    assert_eq!(
        amount.reinstatement_premium_dated(&[22.0, 12.0], &[0.25, 0.5]),
        amount.reinstatement_premium(&[22.0, 12.0])
    );
    // Recoveries 5 at t = 0.4, then 10 at t = 0.7, which uses the rest
    // of the first limit and 5 of the second.
    let rp = timed.reinstatement_premium_dated(&[15.0, 20.0], &[0.4, 0.7]);
    let want = 2.0 * (1.0 * 5.0 * 0.6 + (1.0 * 5.0 + 0.5 * 5.0) * 0.3) / 10.0;
    assert!((rp - want).abs() < 1e-12, "{rp} vs {want}");
    // At t = 0 it is the amount-only premium.
    assert_eq!(
        timed.reinstatement_premium_dated(&[15.0, 25.0], &[0.0, 0.0]),
        amount.reinstatement_premium(&[15.0, 25.0])
    );
    // The annual deductible absorbs the first event, which uses no limit.
    let aad = timed.clone().aggregate_deductible(5.0).unwrap();
    let rp = aad.reinstatement_premium_dated(&[15.0, 25.0], &[0.1, 0.5]);
    assert!((rp - 2.0 * 0.5 * 1.0).abs() < 1e-12);
    assert!(
        Layer::xol("free", 10.0, 10.0)
            .unwrap()
            .reinstatements(1)
            .unwrap()
            .pro_rata_as_to_time()
            .is_err()
    );

    // One loss a year exhausting the first limit, at a uniform time: the
    // premium averages 2 × E[1 − t] = 1.
    let n = 20_000;
    let events = EventSet::from_years(vec![vec![20.0]; n], 9).unwrap();
    let one = Layer::xol("10x10", 10.0, 10.0)
        .unwrap()
        .paid_reinstatements(2.0, vec![1.0])
        .unwrap()
        .pro_rata_as_to_time()
        .unwrap();
    let tower = Tower::inuring(vec![
        vec![Layer::quota_share("QS", 0.0001).unwrap()],
        vec![one],
    ])
    .unwrap();
    assert!(tower.apply(&events).is_err());
    assert!(tower.apply_aggregate(&events.totals().unwrap()).is_err());
    let pd = tower.apply(&events.clone().with_uniform_times()).unwrap();
    let rp = pd
        .marginal(&vec![
            KeyValue::from("reinstatement_premium"),
            KeyValue::from("10x10"),
        ])
        .unwrap();
    let se = (rp.variance() / n as f64).sqrt();
    assert!((rp.mean() - 1.0).abs() < 4.0 * se, "{} ± {se}", rp.mean());
    assert!((rp.variance() / (4.0 / 12.0) - 1.0).abs() < 0.05);

    // The same losses in the second half of the year only: t is uniform
    // on [0.5, 1], so the premium averages 2 × 0.25 = 0.5 with variance
    // 4 × 0.5² / 12.
    let pd = tower
        .apply(&events.with_seasonal_times(&[0.0, 1.0]).unwrap())
        .unwrap();
    let rp = pd
        .marginal(&vec![
            KeyValue::from("reinstatement_premium"),
            KeyValue::from("10x10"),
        ])
        .unwrap();
    let se = (rp.variance() / n as f64).sqrt();
    assert!((rp.mean() - 0.5).abs() < 4.0 * se, "{} ± {se}", rp.mean());
    assert!((rp.variance() / (4.0 * 0.25 / 12.0) - 1.0).abs() < 0.05);
}

#[test]
fn reinstatement_premiums() {
    let layer = Layer::xol("10x10", 10.0, 10.0)
        .unwrap()
        .paid_reinstatements(2.0, vec![1.0, 0.5])
        .unwrap();
    // Recoveries 5 + 10 = 15: the first limit and 5 of the second,
    // so 2 × (1.0 × 10 + 0.5 × 5) / 10 = 2.5.
    assert_eq!(layer.reinstatement_premium(&[15.0, 25.0]), 2.5);
    // Exhausted (30 = three limits): both reinstatements in full.
    assert_eq!(layer.reinstatement_premium(&[40.0, 40.0, 40.0]), 3.0);
    assert_eq!(layer.reinstatement_premium(&[5.0]), 0.0);
    // The share scales the loss, not the premium, which is already
    // for the placed share.
    let half = layer.clone().share(0.5).unwrap();
    assert_eq!(half.reinstatement_premium(&[15.0, 25.0]), 2.5);
    // The annual deductible comes off first.
    let aad = layer.clone().aggregate_deductible(5.0).unwrap();
    assert_eq!(aad.reinstatement_premium(&[15.0, 25.0]), 2.0);
    assert_eq!(
        Layer::xol("F", 10.0, 0.0)
            .unwrap()
            .reinstatements(2)
            .unwrap()
            .reinstatement_premium(&[30.0]),
        0.0
    );
    assert!(
        Layer::quota_share("QS", 0.5)
            .unwrap()
            .paid_reinstatements(1.0, vec![1.0])
            .is_err()
    );
    assert!(
        Layer::xol("L", 1.0, 0.0)
            .unwrap()
            .paid_reinstatements(-1.0, vec![])
            .is_err()
    );
    assert!(
        Layer::xol("L", 1.0, 0.0)
            .unwrap()
            .paid_reinstatements(1.0, vec![f64::NAN])
            .is_err()
    );
}

#[test]
fn tower_reports_reinstatement_premiums() {
    let tower = Tower::new(vec![
        Layer::xol("5x5", 5e6, 5e6)
            .unwrap()
            .paid_reinstatements(1e6, vec![1.0])
            .unwrap(),
        Layer::xol("15x10", 15e6, 10e6).unwrap(),
    ])
    .unwrap();
    let events = events();
    let result = tower.apply(&events).unwrap();
    let kinds: Vec<String> = result
        .components()
        .iter()
        .map(|k| k[0].to_string())
        .collect();
    assert_eq!(
        kinds,
        ["gross", "ceded", "ceded", "net", "reinstatement_premium"]
    );
    for sim in 0..result.n_sims() {
        let row = result.row(sim).unwrap();
        // One reinstatement at 100%: premium × min(ceded, limit) / limit.
        let expected = 1e6 * row[1].min(5e6) / 5e6;
        assert!((row[4] - expected).abs() <= 1e-9 * expected.max(1.0));
        assert!((row[0] - row[1] - row[2] - row[3]).abs() <= 1e-6 * row[0].max(1.0));
    }
}

#[test]
fn surplus_cedes_by_sum_insured_and_inures_to_a_per_risk_xl() {
    // Retention 1m, 4 lines: capacity to 5m of sum insured.
    let surplus = Layer::surplus("surplus", 1e6, 4.0).unwrap();
    let si = [0.5e6, 2e6, 5e6, 10e6];
    let losses = [0.5e6, 2e6, 1e6, 10e6];
    // Cessions 0, 1/2, 4/5, 2/5.
    let by_risk: Vec<f64> = (0..4)
        .map(|i| surplus.ceded_with_sums_insured(&losses[i..=i], &si[i..=i]))
        .collect();
    for (got, want) in by_risk.iter().zip([0.0, 1e6, 0.8e6, 4e6]) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
    // A 1m xs 1m per-risk XL sees each loss net of the surplus.
    let tower = Tower::inuring(vec![
        vec![surplus.clone()],
        vec![Layer::xol("1x1", 1e6, 1e6).unwrap()],
    ])
    .unwrap();
    let ceded = tower.ceded_with_sums_insured(&losses, &si);
    // Nets 0.5m, 1m, 0.2m, 6m: only the last reaches the XL, for 1m.
    assert!((ceded[0] - 5.8e6).abs() < 1e-6);
    assert!((ceded[1] - 1e6).abs() < 1e-6);
    assert!(tower.needs_sums_insured());
    assert!(tower.ceded(&losses)[0].is_nan());
}

#[test]
fn surplus_needs_sums_insured_on_events() {
    let tower = Tower::new(vec![Layer::surplus("s", 1e6, 4.0).unwrap()]).unwrap();
    let events = EventSet::from_years(vec![vec![2e6], vec![]], 0).unwrap();
    assert!(tower.apply(&events).is_err());
    let events = events.with_sums_insured(vec![2e6]).unwrap();
    let result = tower.apply(&events).unwrap();
    let ceded = result
        .marginal(&vec![KeyValue::from("ceded"), KeyValue::from("s")])
        .unwrap();
    assert_eq!(prospicio_prob::Empirical::draws(&ceded), [1e6, 0.0]);
    let pd = PredictiveDistribution::from_draws(
        vec![],
        vec![vec![]],
        vec![1.0, 2.0],
        Provenance::new("test"),
    )
    .unwrap();
    assert!(tower.apply_aggregate(&pd).is_err());
    assert!(Layer::surplus("s", 0.0, 4.0).is_err());
    assert!(Layer::surplus("s", 1.0, f64::INFINITY).is_err());
    let e = EventSet::from_years(vec![vec![5.0]], 0).unwrap();
    assert!(e.clone().with_sums_insured(vec![4.0]).is_err());
    assert!(e.with_sums_insured(vec![5.0, 6.0]).is_err());
}

#[test]
fn apply_annual_matches_apply_aggregate_and_checks_its_input() {
    let cover = Layer::stop_loss("cover", 120.0, 0.0)
        .unwrap()
        .premium_rate(1.0, 100.0)
        .unwrap()
        .sliding_scale(vec![(0.3, 0.5), (0.1, 0.9)])
        .unwrap()
        .profit_commission(0.2, 0.1)
        .unwrap();
    let tower = Tower::new(vec![cover]).unwrap();
    let losses = [0.0, 40.0, 70.0, 150.0, 95.5];
    let annual = tower.apply_annual(&losses).unwrap();
    let pd = PredictiveDistribution::from_draws(
        vec!["line".into()],
        vec![vec!["a".into()]],
        losses.to_vec(),
        Provenance::new("test"),
    )
    .unwrap();
    let aggregate = tower.apply_aggregate(&pd).unwrap();
    assert_eq!(annual.components(), aggregate.components());
    assert_eq!(annual.draw_matrix(), aggregate.draw_matrix());
    assert!(tower.apply_annual(&[]).is_err());
    assert!(tower.apply_annual(&[1.0, -1.0]).is_err());
    assert!(tower.apply_annual(&[f64::NAN]).is_err());
}

#[test]
fn apply_aggregate_keeps_the_input_streams_and_samplers() {
    let tower = Tower::new(vec![Layer::stop_loss("adc", 50.0, 100.0).unwrap()]).unwrap();
    let mut source = Provenance::new("odp_bootstrap").seed(4, SIM_INDEX_SCHEME);
    source.samplers = None; // read from a file saved before samplers were recorded
    let pd =
        PredictiveDistribution::from_draws(vec![], vec![vec![]], vec![90.0, 120.0], source.clone())
            .unwrap();
    let p = tower.apply_aggregate(&pd).unwrap().provenance().clone();
    assert_eq!(p.seed, Some(4));
    assert_eq!(p.stream_scheme.as_deref(), Some(SIM_INDEX_SCHEME));
    assert_eq!(p.samplers, None);
    assert!(p.shares_streams(&source));
}

#[test]
fn apply_records_the_samplers_only_of_events_drawn_here() {
    let tower = Tower::new(vec![Layer::xol("5x5", 5.0, 5.0).unwrap()]).unwrap();
    let external = EventSet::from_years(vec![vec![7.0], vec![12.0]], 0).unwrap();
    let p = tower.apply(&external).unwrap().provenance().clone();
    assert_eq!(p.seed, Some(0));
    assert_eq!(p.stream_scheme.as_deref(), Some(SIM_INDEX_SCHEME));
    assert_eq!(p.samplers, None);
    let other = EventSet::from_years(vec![vec![1.0], vec![]], 0).unwrap();
    let q = tower.apply(&other).unwrap().provenance().clone();
    assert!(!p.replays_same_draws(&q));

    let drawn = simulate_events(
        &Poisson::new(1.0).unwrap(),
        &Lognormal::from_mean_cv(6.0, 1.0).unwrap(),
        20,
        0,
    )
    .unwrap();
    let d = tower.apply(&drawn).unwrap().provenance().clone();
    assert_eq!(
        d.samplers,
        Some(prospicio_prob::provenance::current_samplers())
    );
    assert!(d.shares_streams(&p) && !d.replays_same_draws(&p));
}
