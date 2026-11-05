use super::*;
use prospicio_prob::{Empirical, Grid, Provenance};

fn draws(pd: &PredictiveDistribution, key: &[&str]) -> Vec<f64> {
    let key: ComponentKey = key.iter().map(|&k| KeyValue::from(k)).collect();
    Empirical::draws(&pd.marginal(&key).unwrap()).to_vec()
}

fn sd(x: &[f64]) -> f64 {
    let m = x.iter().sum::<f64>() / x.len() as f64;
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}

/// The deal of #485: a sliding commission, flat brokerage and four
/// legs on the premium, a manager's fee outside NMBF.
fn deal(extra: Option<Deduction>) -> Tower {
    let mut layer = Layer::stop_loss("deal", 120.0, 0.0)
        .unwrap()
        .premium_rate(1.0, 100.0)
        .unwrap()
        .sliding_scale(vec![(0.35, 0.5), (0.15, 0.9)])
        .unwrap()
        .brokerage(0.025)
        .unwrap();
    for (label, rate) in [
        ("tpa", 0.01),
        ("tax", 0.03),
        ("ri_cost", 0.02),
        ("fronting", 0.005),
    ] {
        layer = layer
            .deduction(Deduction::premium(label, rate).unwrap())
            .unwrap();
    }
    layer = layer
        .deduction(Deduction::premium("ms_fee", 0.05).unwrap().outside_nmbf())
        .unwrap();
    if let Some(d) = extra {
        layer = layer.deduction(d).unwrap();
    }
    Tower::new(vec![layer]).unwrap()
}

const LOSSES: [f64; 6] = [20.0, 45.0, 60.0, 75.0, 95.0, 150.0];

#[test]
fn premium_legs_shift_the_total_without_spreading_it() {
    let tower = deal(None);
    let result = tower.apply_annual(&LOSSES).unwrap();
    let r = tower.deal_ratios(&result, "deal").unwrap();
    let [lr, cc, bk, td, cr, nmbf, margin] = [
        "loss_ratio",
        "ceding_commission",
        "brokerage",
        "total_deductions",
        "combined_ratio",
        "nmbf",
        "margin",
    ]
    .map(|k| draws(&r, &[k]));
    for i in 0..LOSSES.len() {
        // TD − CC − BK is the sum of the flat legs, fee included.
        assert!((td[i] - cc[i] - bk[i] - 0.115).abs() < 1e-12);
        assert!((cr[i] - lr[i] - td[i]).abs() < 1e-12);
        assert!((margin[i] - (1.0 - cr[i])).abs() < 1e-12);
        // Only the manager's fee sits outside NMBF.
        assert!((nmbf[i] - margin[i] - 0.05).abs() < 1e-12);
        assert!((bk[i] - 0.025).abs() < 1e-12);
        assert!((lr[i] - LOSSES[i].min(120.0) / 100.0).abs() < 1e-12);
    }
    assert!((sd(&td) - sd(&cc)).abs() < 1e-12);
    assert_eq!(draws(&r, &["tax"]), vec![0.03; LOSSES.len()]);
    // The amounts in the tower's result.
    assert_eq!(draws(&result, &["ms_fee", "deal"]), vec![5.0; LOSSES.len()]);
}

#[test]
fn a_loss_leg_spreads_the_total() {
    let tower = deal(Some(Deduction::loss("claims_fee", 0.04).unwrap()));
    let result = tower.apply_annual(&LOSSES).unwrap();
    let r = tower.deal_ratios(&result, "deal").unwrap();
    let (td, cc, lr) = (
        draws(&r, &["total_deductions"]),
        draws(&r, &["ceding_commission"]),
        draws(&r, &["loss_ratio"]),
    );
    let fee = draws(&r, &["claims_fee"]);
    for i in 0..LOSSES.len() {
        assert!((fee[i] - 0.04 * lr[i]).abs() < 1e-12);
    }
    assert!((sd(&td) - sd(&cc)).abs() > 1e-3);
    let fixed = deal(Some(Deduction::fixed("audit", 2.0).unwrap()));
    let r = fixed
        .deal_ratios(&fixed.apply_annual(&LOSSES).unwrap(), "deal")
        .unwrap();
    assert_eq!(draws(&r, &["audit"]), vec![0.02; LOSSES.len()]);
}

/// A 120% loss-ratio cap as an annual limit on a whole-account quota
/// share, and both scales.
fn capped(before: bool) -> Tower {
    let mut layer = Layer::quota_share("deal", 1.0)
        .unwrap()
        .aggregate_limit(120.0)
        .unwrap()
        .premium_rate(1.0, 100.0)
        .unwrap()
        .sliding_scale(vec![(0.3, 0.5), (0.1, 1.5)])
        .unwrap()
        .brokerage_scale(vec![(0.01, 0.5), (0.03, 1.5)])
        .unwrap();
    if before {
        layer = layer.scales_before_annual_terms();
    }
    Tower::new(vec![layer]).unwrap()
}

#[test]
fn the_scales_read_the_loss_ratio_before_the_cap_when_asked() {
    let losses = [40.0, 140.0];
    let after = capped(false);
    let before = capped(true);
    let a = after.apply_annual(&losses).unwrap();
    let b = before.apply_annual(&losses).unwrap();
    assert_eq!(draws(&a, &["ceded", "deal"]), [40.0, 120.0]);
    assert_eq!(draws(&b, &["ceded", "deal"]), [40.0, 120.0]);
    assert_eq!(draws(&b, &["loss_before_terms", "deal"]), [40.0, 140.0]);
    assert!(
        a.component_index(&vec!["loss_before_terms".into(), "deal".into()])
            .is_none()
    );
    // Below the cap the two agree; above it the scale reads 140%.
    let rate = |lr: f64| 0.3 - 0.2 * (lr - 0.5);
    let (cca, ccb) = (
        draws(&a, &["ceding_commission", "deal"]),
        draws(&b, &["ceding_commission", "deal"]),
    );
    assert!((cca[0] - 30.0).abs() < 1e-12 && (ccb[0] - 30.0).abs() < 1e-12);
    assert!((cca[1] - 100.0 * rate(1.2)).abs() < 1e-12);
    assert!((ccb[1] - 100.0 * rate(1.4)).abs() < 1e-12);
    let bk = draws(&b, &["brokerage", "deal"]);
    assert!((bk[1] - 100.0 * (0.01 + 0.02 * 0.9)).abs() < 1e-12);
    let r = before.deal_ratios(&b, "deal").unwrap();
    assert_eq!(draws(&r, &["loss_ratio"]), [0.4, 1.2]);
    assert_eq!(draws(&r, &["loss_ratio_before_terms"]), [0.4, 1.4]);
    let ra = after.deal_ratios(&a, "deal").unwrap();
    assert_eq!(ra.components().len() + 1, r.components().len());
}

#[test]
fn the_brokerage_key_is_held_in_the_commission_range_only_when_asked() {
    let tower = |held: bool| {
        let mut l = Layer::quota_share("deal", 1.0)
            .unwrap()
            .premium_rate(1.0, 100.0)
            .unwrap()
            .sliding_scale(vec![(0.3, 0.5), (0.1, 0.9)])
            .unwrap()
            .brokerage_scale(vec![(0.0, 0.3), (0.04, 1.3)])
            .unwrap();
        if held {
            l = l.brokerage_within_commission_range();
        }
        Tower::new(vec![l]).unwrap()
    };
    let losses = [10.0, 70.0, 130.0];
    let free = draws(
        &tower(false).apply_annual(&losses).unwrap(),
        &["brokerage", "deal"],
    );
    let held = draws(
        &tower(true).apply_annual(&losses).unwrap(),
        &["brokerage", "deal"],
    );
    let bk = |lr: f64| 100.0 * 0.04 * (lr - 0.3);
    for (got, want) in free.iter().zip([0.0, bk(0.7), bk(1.3)]) {
        assert!((got - want).abs() < 1e-12);
    }
    for (got, want) in held.iter().zip([bk(0.5), bk(0.7), bk(0.9)]) {
        assert!((got - want).abs() < 1e-12);
    }
}

#[test]
fn a_term_gives_ratios_by_year_with_the_carry_on_the_key() {
    let layer = Layer::quota_share("deal", 1.0)
        .unwrap()
        .aggregate_limit(120.0)
        .unwrap()
        .premium_rate(1.0, 100.0)
        .unwrap()
        .sliding_scale(vec![(0.3, 0.5), (0.1, 1.5)])
        .unwrap()
        .sliding_scale_carry_forward(true, false, None)
        .unwrap()
        .brokerage(0.02)
        .unwrap()
        .scales_before_annual_terms();
    let tower = Tower::new(vec![layer]).unwrap();
    let pd = |v: Vec<f64>| {
        PredictiveDistribution::from_draws(vec![], vec![vec![]], v, Provenance::new("test"))
            .unwrap()
    };
    let result = tower
        .apply_aggregate_term(&[pd(vec![180.0]), pd(vec![60.0])])
        .unwrap();
    let r = tower.deal_ratios(&result, "deal").unwrap();
    assert_eq!(r.dims(), ["ratio", "year"]);
    let at = |k: &str, y: i64| {
        let key: ComponentKey = vec![KeyValue::from(k), KeyValue::Int(y)];
        Empirical::draws(&r.marginal(&key).unwrap())[0]
    };
    assert_eq!(at("loss_ratio", 1), 1.2);
    assert_eq!(at("loss_ratio_before_terms", 1), 1.8);
    // Year 1 is beyond the scale's 150%: 30% of deficit carried, read
    // on the uncapped loss; year 2 reads 60% + 30%.
    assert!((at("ceding_commission", 1) - 0.1).abs() < 1e-12);
    assert!((at("ceding_commission", 2) - (0.3 - 0.2 * 0.4)).abs() < 1e-12);
    assert!((at("brokerage", 2) - 0.02).abs() < 1e-12);
}

#[test]
fn the_grids_refuse_a_scale_read_before_annual_terms() {
    let annual = Grid::new(10.0, vec![0.25; 4]).unwrap();
    let err = capped(true).on_annual_grid(&annual).unwrap_err();
    assert!(err.to_string().contains("before its annual terms"));
    assert!(capped(false).on_annual_grid(&annual).is_ok());
    // Without annual terms the key is the ceded loss.
    let open = Layer::quota_share("deal", 1.0)
        .unwrap()
        .premium_rate(1.0, 100.0)
        .unwrap()
        .sliding_scale(vec![(0.3, 0.5), (0.1, 1.5)])
        .unwrap()
        .scales_before_annual_terms();
    assert!(
        Tower::new(vec![open])
            .unwrap()
            .on_annual_grid(&annual)
            .is_ok()
    );
}

#[test]
fn refuses_bad_deductions_and_results() {
    assert!(Deduction::premium("", 0.1).is_err());
    assert!(Deduction::premium("brokerage", 0.1).is_err());
    assert!(Deduction::premium("nmbf", 0.1).is_err());
    assert!(Deduction::premium("tax", -0.1).is_err());
    assert!(Deduction::fixed("tax", f64::INFINITY).is_err());
    let l = Layer::quota_share("QS", 0.5)
        .unwrap()
        .deduction(Deduction::premium("tax", 0.03).unwrap())
        .unwrap();
    assert!(
        l.clone()
            .deduction(Deduction::loss("tax", 0.01).unwrap())
            .is_err()
    );
    assert!(l.clone().brokerage(1.5).is_err());
    // A brokerage scale needs a premium and refuses swing rating.
    assert!(l.clone().brokerage_scale(vec![(0.02, 0.5)]).is_err());
    let priced = l.premium_rate(1.0, 100.0).unwrap();
    assert!(priced.clone().brokerage_scale(vec![(1.2, 0.5)]).is_err());
    let scaled = priced
        .brokerage_scale(vec![(0.01, 0.5), (0.03, 0.9)])
        .unwrap();
    let swing = crate::LossSensitivePremium::new(10.0, 1.0, None, None).unwrap();
    assert!(scaled.clone().swing_rated(swing).is_err());
    let tower = Tower::new(vec![scaled]).unwrap();
    let result = tower.apply_annual(&[20.0, 40.0]).unwrap();
    assert!(tower.deal_ratios(&result, "nope").is_err());
    let other = deal(None);
    assert!(
        other
            .deal_ratios(&other.apply_annual(&[20.0]).unwrap(), "deal")
            .is_ok()
    );
    assert!(other.deal_ratios(&result, "deal").is_err());
}

/// Two sections of premium 100 each, A capped at 50 and B above 50,
/// each with its own scale, and a deal scale on both together.
fn two_sections(deal: DealScale) -> Tower {
    let scale = vec![(0.35, 0.5), (0.15, 0.9)];
    let a = Layer::stop_loss("A", 50.0, 0.0)
        .unwrap()
        .premium_rate(1.0, 100.0)
        .unwrap()
        .sliding_scale(scale.clone())
        .unwrap()
        .deduction(Deduction::premium("tax", 0.03).unwrap())
        .unwrap();
    let b = Layer::stop_loss("B", f64::INFINITY, 50.0)
        .unwrap()
        .premium_rate(1.0, 100.0)
        .unwrap()
        .sliding_scale(scale)
        .unwrap()
        .brokerage(0.02)
        .unwrap()
        .deduction(Deduction::premium("tax", 0.03).unwrap())
        .unwrap();
    Tower::new(vec![a, b]).unwrap().deal_scale(deal).unwrap()
}

fn deal_scale() -> DealScale {
    DealScale::new("deal", ["A", "B"])
        .unwrap()
        .sliding_scale(vec![(0.2, 0.5), (0.1, 0.9)])
        .unwrap()
}

#[test]
fn a_deal_scale_reads_its_sections_together() {
    let tower = two_sections(
        deal_scale()
            .brokerage_scale(vec![(0.01, 0.5), (0.03, 0.9)])
            .unwrap(),
    );
    // A loss of 160: A at 50%, B at 110%, the deal at 80%.
    let result = tower.apply_annual(&[160.0, 40.0]).unwrap();
    let deal_cc = draws(&result, &["ceding_commission", "deal"]);
    // 20% − 0.1 × 0.75 = 12.5% of 200 at 80%; 20% of 200 at 20%.
    assert!((deal_cc[0] - 25.0).abs() < 1e-12);
    assert!((deal_cc[1] - 40.0).abs() < 1e-12);
    // The sections' own scales on each loss ratio alone: not the
    // deal's 80%.
    let own =
        [&["ceding_commission", "A"], &["ceding_commission", "B"]].map(|k| draws(&result, k)[0]);
    assert!((own[0] - 35.0).abs() < 1e-12 && (own[1] - 15.0).abs() < 1e-12);
    let at_80 = 0.35 - 0.2 * 0.75;
    assert!((own[0] + own[1] - 2.0 * 100.0 * at_80).abs() > 1.0);
    // The deal's brokerage at 80%: 1% + 0.02 × 0.75 = 2.5%.
    assert!((draws(&result, &["brokerage", "deal"])[0] - 5.0).abs() < 1e-12);
    // The deal's ratios add the sections and the deal over 200.
    let r = tower.deal_ratios(&result, "deal").unwrap();
    let ratio = |k: &str| draws(&r, &[k])[0];
    assert!((ratio("loss_ratio") - 0.8).abs() < 1e-12);
    assert!((ratio("ceding_commission") - (35.0 + 15.0 + 25.0) / 200.0).abs() < 1e-12);
    assert!((ratio("brokerage") - (2.0 + 5.0) / 200.0).abs() < 1e-12);
    assert!((ratio("tax") - 0.03).abs() < 1e-12);
    // The commission's total is above the deal scale's 20% maximum,
    // as in the reference model's workbook.
    assert!(ratio("ceding_commission") > 0.2);
    let total = ratio("ceding_commission") + ratio("brokerage") + ratio("tax");
    assert!((ratio("total_deductions") - total).abs() < 1e-12);
    assert!((ratio("margin") - (1.0 - 0.8 - total)).abs() < 1e-12);
    // A section's own ratios leave the deal's scales out.
    let ra = tower.deal_ratios(&result, "A").unwrap();
    assert!((draws(&ra, &["ceding_commission"])[0] - 0.35).abs() < 1e-12);
    assert!(
        result
            .provenance()
            .parameters
            .iter()
            .any(|(k, _)| k == "deal:deal")
    );
}

#[test]
fn a_deal_carries_its_own_deficit_over_a_term() {
    let tower = two_sections(deal_scale().carry_forward(true, false, None).unwrap());
    let pd = |v: Vec<f64>| {
        PredictiveDistribution::from_draws(vec![], vec![vec![]], v, Provenance::new("test"))
            .unwrap()
    };
    // Year 1 at 100%: 10% of 200 beyond the deal's 90%, carried. Year 2
    // at 50% reads 60%: 20% − 0.1 × 0.25.
    let result = tower
        .apply_aggregate_term(&[pd(vec![200.0]), pd(vec![100.0])])
        .unwrap();
    let at = |k: &str, l: &str, y: i64| {
        let key: ComponentKey = vec![KeyValue::from(k), KeyValue::from(l), KeyValue::Int(y)];
        Empirical::draws(&result.marginal(&key).unwrap())[0]
    };
    assert!((at("ceding_commission", "deal", 1) - 20.0).abs() < 1e-12);
    assert!((at("ceding_commission", "deal", 2) - 0.175 * 200.0).abs() < 1e-12);
    // The sections carry nothing: A at 50% in year 2 reads 35%.
    assert!((at("ceding_commission", "A", 2) - 35.0).abs() < 1e-12);
    let r = tower.deal_ratios(&result, "deal").unwrap();
    assert_eq!(r.dims(), ["ratio", "year"]);
}

#[test]
fn a_deal_reads_the_sections_keys_before_annual_terms() {
    let section = |name: &str, before: bool| {
        let l = Layer::quota_share(name, 1.0)
            .unwrap()
            .aggregate_limit(60.0)
            .unwrap()
            .premium_rate(1.0, 100.0)
            .unwrap();
        if before {
            l.scales_before_annual_terms()
        } else {
            l
        }
    };
    let deal = DealScale::new("deal", ["A", "B"])
        .unwrap()
        .sliding_scale(vec![(0.3, 0.5), (0.1, 1.0)])
        .unwrap()
        .brokerage_scale(vec![(0.01, 0.5), (0.04, 2.0)])
        .unwrap()
        .brokerage_within_commission_range();
    let tower = Tower::new(vec![section("A", true), section("B", false)])
        .unwrap()
        .deal_scale(deal)
        .unwrap();
    // A loss of 150: each cedes 60; A's key is 150, B's 60.
    let result = tower.apply_annual(&[150.0]).unwrap();
    assert_eq!(draws(&result, &["loss_before_terms", "deal"]), [210.0]);
    // The deal at 105%, beyond the scale: 10% of 200.
    assert!((draws(&result, &["ceding_commission", "deal"])[0] - 20.0).abs() < 1e-12);
    // Its brokerage held at the commission's 100%: 1% + 0.03 / 3 = 2%,
    // not the 2.1% at 105%.
    assert!((draws(&result, &["brokerage", "deal"])[0] - 4.0).abs() < 1e-12);
    let r = tower.deal_ratios(&result, "deal").unwrap();
    assert!((draws(&r, &["loss_ratio"])[0] - 0.6).abs() < 1e-12);
    assert!((draws(&r, &["loss_ratio_before_terms"])[0] - 1.05).abs() < 1e-12);
    // The annual grid reads the same keys at each annual total: here 0
    // and 150, equally likely.
    let annual = Grid::new(50.0, vec![0.5, 0.0, 0.0, 0.5]).unwrap();
    let g = tower.on_annual_grid(&annual).unwrap();
    let sims = tower.apply_annual(&[0.0, 150.0]).unwrap();
    let mean = |kind: &str| draws(&sims, &[kind, "deal"]).iter().sum::<f64>() / 2.0;
    assert!((g.expected_deal_ceding_commission[0] - mean("ceding_commission")).abs() < 1e-12);
    assert!((g.expected_deal_brokerage[0] - mean("brokerage")).abs() < 1e-12);
    // A compound grid cannot: a section has annual terms. The error
    // names the deal and the section.
    let sev = Grid::new(10.0, vec![0.0, 0.5, 0.5]).unwrap();
    let poisson = prospicio_prob::Poisson::new(2.0).unwrap();
    let err = tower.on_grid(&poisson, &sev, 64).unwrap_err().to_string();
    assert!(
        err.contains("deal \"deal\"") && err.contains("\"A\""),
        "{err}"
    );
}

#[test]
fn refuses_bad_deals() {
    assert!(DealScale::new("", ["A", "B"]).is_err());
    assert!(DealScale::new("d", ["A"]).is_err());
    assert!(DealScale::new("d", ["A", "B", "A"]).is_err());
    let bare = DealScale::new("d", ["A", "B"]).unwrap();
    assert!(bare.clone().carry_forward(true, false, None).is_err());
    assert!(
        bare.clone()
            .sliding_scale(vec![(0.1, 0.5), (0.2, 0.9)])
            .is_err()
    );
    assert!(bare.clone().brokerage(1.5).is_err());
    let tower = two_sections(deal_scale());
    let base = Tower::new(tower.layers.clone()).unwrap();
    // No scale, a taken name, an unknown section, a section twice.
    assert!(base.clone().deal_scale(bare.clone()).is_err());
    assert!(
        base.clone()
            .deal_scale(
                DealScale::new("A", ["A", "B"])
                    .unwrap()
                    .brokerage(0.01)
                    .unwrap()
            )
            .is_err()
    );
    assert!(
        base.clone()
            .deal_scale(
                DealScale::new("d", ["A", "C"])
                    .unwrap()
                    .brokerage(0.01)
                    .unwrap()
            )
            .is_err()
    );
    assert!(
        tower
            .clone()
            .deal_scale(
                DealScale::new("e", ["A", "B"])
                    .unwrap()
                    .brokerage(0.01)
                    .unwrap()
            )
            .is_err()
    );
    // A section without a premium.
    let unpriced = Tower::new(vec![
        Layer::xol("A", 10.0, 0.0).unwrap(),
        Layer::xol("B", 10.0, 10.0)
            .unwrap()
            .premium_rate(1.0, 10.0)
            .unwrap(),
    ])
    .unwrap();
    assert!(unpriced.deal_scale(bare.brokerage(0.01).unwrap()).is_err());
    // A label inside NMBF in one section and outside in the other.
    let mut mixed = tower.clone();
    mixed.layers[1].deductions[0].inside_nmbf = false;
    let result = mixed.apply_annual(&[100.0]).unwrap();
    assert!(mixed.deal_ratios(&result, "deal").is_err());
    assert!(mixed.deal_ratios(&result, "A").is_ok());
}

#[test]
fn the_ledger_books_a_deal_atom_by_atom_as_its_ratios() {
    let tower = two_sections(deal_scale().brokerage(0.01).unwrap());
    let values = vec![20.0, 60.0, 160.0, 240.0];
    let gross = Source::new(values.clone(), vec![0.4, 0.3, 0.2, 0.1]).unwrap();
    let ledger = tower.deal_ledger(&gross, "deal").unwrap();
    let ratios = tower
        .deal_ratios(&tower.apply_annual(&values).unwrap(), "deal")
        .unwrap();
    // The result at each atom is the margin times the deals' premium.
    let margin = draws(&ratios, &["margin"]);
    for (r, m) in ledger.result().values().iter().zip(&margin) {
        assert!((r - 200.0 * m).abs() < 1e-9, "{r} {m}");
    }
    // Each leg is the ratio times the premium, with the sign of an
    // obligation.
    for (row, ratio) in [
        ("Loss", "loss_ratio"),
        ("Ceding commission", "ceding_commission"),
        ("Brokerage", "brokerage"),
        ("tax", "tax"),
    ] {
        let leg = ledger.row(row).unwrap().values();
        for (l, x) in leg.iter().zip(draws(&ratios, &[ratio])) {
            assert!((l + 200.0 * x).abs() < 1e-9, "{row}");
        }
    }
    assert_eq!(ledger.row("Premium").unwrap().values(), [200.0; 4]);
    // A section alone books its own terms.
    let a = tower.deal_ledger(&gross, "A").unwrap();
    assert!(a.row("Brokerage").is_none() && a.row("tax").is_some());
}

#[test]
fn a_no_claims_bonus_is_a_deduction_in_the_ratios_and_a_leg_in_the_ledger() {
    let tower = Tower::new(vec![
        Layer::quota_share("deal", 1.0)
            .unwrap()
            .premium_rate(1.0, 100.0)
            .unwrap()
            .no_claims_bonus(0.1, 1, 50.0)
            .unwrap(),
    ])
    .unwrap();
    let values = vec![20.0, 50.0, 60.0, 150.0];
    let r = tower
        .deal_ratios(&tower.apply_annual(&values).unwrap(), "deal")
        .unwrap();
    let (ncb, td, margin) = (
        draws(&r, &["no_claims_bonus"]),
        draws(&r, &["total_deductions"]),
        draws(&r, &["margin"]),
    );
    assert_eq!(ncb, vec![0.1, 0.1, 0.0, 0.0]);
    assert_eq!(td, ncb);
    for (i, v) in values.iter().enumerate() {
        assert!((margin[i] - (1.0 - v / 100.0 - ncb[i])).abs() < 1e-12);
    }
    let gross = Source::new(values, vec![0.25; 4]).unwrap();
    let ledger = tower.deal_ledger(&gross, "deal").unwrap();
    let leg = ledger.row("No-claims bonus").unwrap().values();
    for (l, x) in leg.iter().zip(&ncb) {
        assert!((l + 100.0 * x).abs() < 1e-9);
    }
}

#[test]
fn the_ledger_refuses_what_an_annual_loss_cannot_carry() {
    let tower = two_sections(deal_scale());
    let joint = Source::joint(vec![10.0, 20.0], vec![5.0, 10.0], vec![0.5, 0.5]).unwrap();
    assert!(tower.deal_ledger(&joint, "deal").is_err());
    let gross = Source::new(vec![10.0, 20.0], vec![0.5, 0.5]).unwrap();
    assert!(tower.deal_ledger(&gross, "nope").is_err());
    let reinstated = Tower::new(vec![
        Layer::xol("L", 10.0, 5.0)
            .unwrap()
            .paid_reinstatements(2.0, vec![1.0])
            .unwrap(),
    ])
    .unwrap();
    assert!(reinstated.deal_ledger(&gross, "L").is_err());
}
