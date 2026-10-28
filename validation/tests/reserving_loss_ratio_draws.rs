//! Reserving draws into a tower (`reserving-v02.md` decision 26, issue
//! #602): the ultimates and loss ratios of `UltimateDraws`, simulation by
//! simulation, through `Tower::apply_annual`, against the same tower on the
//! fit's own distributions with `Tower::apply_aggregate`.
//!
//! * The total: `apply_annual` on `total_ultimate_draws` cedes, simulation
//!   by simulation, what `apply_aggregate` cedes on the joint `ultimates`
//!   (each simulation's total one aggregate loss), and what it cedes on the
//!   fit's `reserves` with the attachment lowered by the latest values. So
//!   the expected ceded losses agree too.
//! * One origin, a contract on the loss ratio (a cap of 120% and an
//!   attachment of 70% on the origin's premium): the layer on
//!   `loss_ratio_draws` times the premium, against the same layer on the
//!   origin's reserves.
//! * Over segments, each line is named by its key: a cover on one line's
//!   ultimates cedes, simulation by simulation, what it cedes on that line's
//!   part of the joint reserves, and the lines' draws add up to the
//!   total's, index by index.

use prospicio_aggregate::{Layer, Tower};
use prospicio_prob::{Distribution, Empirical, KeyValue, PredictiveDistribution};
use prospicio_reserving::{
    DevelopmentColumn, Grain, Long, MackBootstrap, Month, OdpBootstrap, Triangle, UltimateDraws,
};
use prospicio_validation::triangle;

fn ceded(result: &PredictiveDistribution, layer: &str) -> Vec<f64> {
    let key: Vec<KeyValue> = vec!["ceded".into(), layer.into()];
    Empirical::draws(&result.marginal(&key).expect("a ceded component")).to_vec()
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn assert_close(a: &[f64], b: &[f64], scale: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= 1e-9 * scale, "simulation {i}: {x} vs {y}");
    }
}

fn raa_odp() -> prospicio_reserving::OdpBootstrapFit {
    OdpBootstrap {
        n_sims: 5_000,
        seed: 602,
        ..Default::default()
    }
    .fit(&triangle("raa"), "values")
    .unwrap()
}

#[test]
fn a_tower_on_the_total_draws_cedes_what_it_cedes_on_the_fits_distribution() {
    let fit = raa_odp();
    let latest: f64 = fit.latest_values().iter().sum();
    let total = fit.total_ultimate_draws().unwrap();
    let ultimates = fit.ultimates().unwrap();
    // An aggregate cover of 30,000 excess of the 220,000 total ultimate.
    let (limit, attachment) = (30_000.0, 220_000.0);
    let tower = |a: f64| Tower::new(vec![Layer::stop_loss("adc", limit, a).unwrap()]).unwrap();
    let on_draws = ceded(&tower(attachment).apply_annual(&total).unwrap(), "adc");
    let on_ultimates = ceded(
        &tower(attachment).apply_aggregate(&ultimates).unwrap(),
        "adc",
    );
    let on_reserves = ceded(
        &tower(attachment - latest)
            .apply_aggregate(&fit.reserves)
            .unwrap(),
        "adc",
    );
    assert_close(&on_draws, &on_ultimates, attachment);
    assert_close(&on_draws, &on_reserves, attachment);
    let expected = mean(&on_draws);
    assert!(expected > 0.0 && expected < limit);
    assert!((expected - mean(&on_reserves)).abs() <= 1e-9 * attachment);
}

#[test]
fn a_contract_on_one_origins_loss_ratio_reads_the_same_simulations() {
    let fit = raa_odp();
    let origin: Vec<KeyValue> = vec![1988.into()];
    let premium = 40_000.0;
    let ratio = fit.loss_ratio_draws(&origin, premium).unwrap();
    // A cap of 120% and an attachment of 70% of the premium: on the loss
    // ratio's money amounts and, shifted by the latest value, on the
    // reserves of 1988 alone.
    let limit = (1.2 - 0.7) * premium;
    let attachment = 0.7 * premium;
    let cover = |a: f64| {
        Tower::new(vec![
            Layer::stop_loss("cover", limit, a)
                .unwrap()
                .premium_rate(1.0, premium)
                .unwrap(),
        ])
        .unwrap()
    };
    let amounts: Vec<f64> = ratio.iter().map(|q| q * premium).collect();
    let on_draws = ceded(&cover(attachment).apply_annual(&amounts).unwrap(), "cover");
    let j = fit
        .reserves
        .component_index(&origin)
        .expect("1988 is an origin");
    let latest = fit.latest_values()[j];
    let reserve_1988 = fit
        .reserves
        .select("origin", &[KeyValue::from(1988)])
        .unwrap();
    let on_reserves = ceded(
        &cover(attachment - latest)
            .apply_aggregate(&reserve_1988)
            .unwrap(),
        "cover",
    );
    assert_close(&on_draws, &on_reserves, premium);
    let expected = mean(&on_draws);
    assert!(expected > 0.0, "the cover is reached: {expected}");
    // The mean ultimate loss ratio is the reserves' mean plus the latest,
    // over the premium.
    let reserve = fit.reserves.marginal(&origin).unwrap();
    assert!((mean(&ratio) - (latest + reserve.mean()) / premium).abs() < 1e-12);
}

/// RAA under `lob` values `A` and `B`, B scaled by 3.
fn two_lines() -> Triangle {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/raa.csv");
    let text = std::fs::read_to_string(path).expect("data/raa.csv");
    let rows: Vec<(i32, u32, f64)> = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (
                f[0].parse().unwrap(),
                f[1].parse().unwrap(),
                f[2].parse().unwrap(),
            )
        })
        .collect();
    let (mut keys, mut origin, mut ages, mut values) = (vec![], vec![], vec![], vec![]);
    for (name, scale) in [("A", 1.0), ("B", 3.0)] {
        for &(o, age, v) in &rows {
            keys.push(name);
            origin.push(Month::january(o));
            ages.push(age);
            values.push(scale * v);
        }
    }
    Triangle::from_long(&Long {
        keys: &[("lob", &keys)],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("paid", &values)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

#[test]
fn each_lines_cover_on_its_own_draws_reads_its_part_of_the_joint_reserves() {
    let fits = MackBootstrap {
        n_sims: 4_000,
        seed: 6,
        ..Default::default()
    }
    .fit_segments(&two_lines(), "paid")
    .unwrap();
    let ultimates = fits.ultimates().unwrap();
    let latest = fits.latest_values();
    let total = fits.total_ultimate_draws().unwrap();
    let mut sum = vec![0.0; total.len()];
    for (s, lob) in ["A", "B"].into_iter().enumerate() {
        let part = ultimates.select("lob", &[KeyValue::from(lob)]).unwrap();
        let draws = Empirical::draws(part.total()).to_vec();
        let line_latest: f64 = latest[10 * s..10 * (s + 1)].iter().sum();
        // A cover of 20% of the line's mean ultimate excess of that mean.
        let attachment = mean(&draws);
        let cover =
            |a: f64| Tower::new(vec![Layer::stop_loss(lob, 0.2 * attachment, a).unwrap()]).unwrap();
        let on_draws = ceded(&cover(attachment).apply_annual(&draws).unwrap(), lob);
        let reserves = fits.reserves.select("lob", &[KeyValue::from(lob)]).unwrap();
        let on_reserves = ceded(
            &cover(attachment - line_latest)
                .apply_aggregate(&reserves)
                .unwrap(),
            lob,
        );
        assert_close(&on_draws, &on_reserves, attachment);
        assert!(mean(&on_draws) > 0.0);
        for (t, d) in sum.iter_mut().zip(&draws) {
            *t += d;
        }
    }
    // The lines' draws add up, index by index, to the total's.
    assert_close(&sum, &total, total[0]);
}
