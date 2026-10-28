//! The Bayesian fits' ultimates and loss ratios per simulation
//! (`UltimateDraws`, `reserving-v02.md` decision 26, issue #602), on short
//! samplers: `BayesOdp` on RAA and over two segments, `BayesBf` and
//! `BayesBfNormalised` on GenIns with its premium, Meyers' CRC on group
//! 353 Commercial Auto alone and over two segments, and compartmental
//! model 1 on company 337 alone and over two segments.
//!
//! * Each origin's ultimate less its reserve is its latest value in every
//!   simulation, the origin found by its key, not its position (the latest
//!   paid for Compartmental, whose ultimate is on the paid basis); for
//!   Meyers' models, whose ultimates are their own simulated draws, to
//!   rounding.
//! * The mean of each origin's ultimate draws is the long table's
//!   `ultimate_mean` for Meyers, and the latest plus the table's mean
//!   reserve for the others; the mean loss ratio on Meyers' own premium is
//!   the table's `ulr_mean`.
//! * Fitting again with the same seed gives the same draws in the same
//!   order: NUTS and the predictive simulation are reproducible on one
//!   machine.

use std::sync::OnceLock;

use prospicio_bayes::nuts::Sampler;
use prospicio_prob::{Distribution, KeyValue, PredictiveDistribution};
use prospicio_reserving::bayes::{
    BayesBf, BayesBfNormalised, BayesOdp, Compartmental, CompartmentalPriors, Meyers, MeyersModel,
    Outstanding,
};
use prospicio_reserving::{
    DevelopmentColumn, FitTable, Grain, Lag, Long, Month, Period, Triangle, UltimateDraws,
};
use prospicio_validation::{triangle, triangle_columns};

/// Checks that `fit`'s ultimates less its reserves are `latest` for the
/// component of each key, and that the mean ultimate of each is `mean(j)`.
fn check_layout(fit: &impl UltimateDraws, keys: &[Vec<KeyValue>], latest: &[f64], rel: f64) {
    assert_eq!(keys.len(), latest.len());
    assert_eq!(fit.reserves().n_components(), keys.len());
    for (key, &l) in keys.iter().zip(latest) {
        let ultimate = fit.ultimate_draws(key).unwrap();
        let reserve = fit.reserves().marginal(key).unwrap();
        let reserve = prospicio_prob::Empirical::draws(&reserve);
        for (u, r) in ultimate.iter().zip(reserve) {
            assert!(
                (u - r - l).abs() <= rel * u.abs().max(1.0),
                "{key:?}: {u} - {r} != {l}"
            );
        }
    }
}

/// The mean of each origin's ultimate draws against the long table's
/// `latest` plus `mean` (the mean reserve), row by row.
fn check_means(fit: &impl UltimateDraws, table: &FitTable) {
    let latest = table.column("latest").expect("a latest column");
    let reserve = table.column("mean").expect("a mean column");
    let ultimates = fit.ultimates().unwrap();
    for j in 0..ultimates.n_components() {
        let key = ultimates.components()[j].clone();
        let m = ultimates.marginal(&key).unwrap().mean();
        let target = latest[j] + reserve[j];
        assert!((m - target).abs() <= 1e-9 * target.abs(), "{key:?}");
    }
}

fn same_draws(a: &PredictiveDistribution, b: &PredictiveDistribution) {
    assert_eq!(a.components(), b.components());
    assert_eq!(a.draw_matrix(), b.draw_matrix());
}

/// `latest` for segment `A` followed by twice it for segment `B`, as
/// `twins` builds them.
fn twin_latest(latest: &[f64]) -> Vec<f64> {
    latest
        .iter()
        .copied()
        .chain(latest.iter().map(|l| 2.0 * l))
        .collect()
}

fn origin_keys(origins: &[Period]) -> Vec<Vec<KeyValue>> {
    origins.iter().map(|&p| vec![KeyValue::Period(p)]).collect()
}

fn segment_keys(lobs: &[&str], origins: &[Period]) -> Vec<Vec<KeyValue>> {
    lobs.iter()
        .flat_map(|&lob| {
            origins
                .iter()
                .map(move |&p| vec![KeyValue::from(lob), KeyValue::Period(p)])
        })
        .collect()
}

fn short(chains: usize, n: usize) -> Sampler {
    Sampler {
        chains,
        tune: n,
        draws: n,
        ..Sampler::default()
    }
}

/// `tri`'s `columns` twice, under the `lob` values `A` and `B`, with every
/// value of `B` doubled (exactly, in floating point) so that a swap or
/// misalignment of the segments' latest values shows.
fn twins(tri: &Triangle, columns: &[&str]) -> Triangle {
    let long = tri.to_long();
    let n = long.origin.len();
    let column = |name: &str| -> &[f64] { &long.values.iter().find(|(c, _)| c == name).unwrap().1 };
    let (mut keys, mut origin, mut ages) = (vec![], vec![], vec![]);
    let mut values: Vec<Vec<f64>> = vec![vec![]; columns.len()];
    for (lob, scale) in [("A", 1.0), ("B", 2.0)] {
        for i in 0..n {
            keys.push(lob);
            origin.push(long.origin[i]);
            ages.push(long.development[i]);
            for (c, name) in columns.iter().enumerate() {
                values[c].push(scale * column(name)[i]);
            }
        }
    }
    let named: Vec<(&str, &[f64])> = columns
        .iter()
        .zip(&values)
        .map(|(n, v)| (*n, v.as_slice()))
        .collect();
    Triangle::from_long(&Long {
        keys: &[("lob", &keys)],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &named,
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

#[test]
fn bayes_odp_alone_and_over_segments() {
    let raa = triangle("raa");
    let model = BayesOdp {
        sampler: short(2, 300),
        n_sims: 1_000,
        seed: 3,
        ..BayesOdp::default()
    };
    let fit = model.fit(&raa, "values").unwrap();
    let cl = &fit.chain_ladder;
    check_layout(&fit, &origin_keys(&cl.origins), &cl.latest, 1e-12);
    check_means(&fit, &fit.to_long());
    same_draws(
        &fit.ultimates().unwrap(),
        &model.fit(&raa, "values").unwrap().ultimates().unwrap(),
    );

    let fits = model
        .fit_segments(&twins(&raa, &["values"]), "values")
        .unwrap();
    let latest: Vec<f64> = twin_latest(&cl.latest[..]);
    check_layout(
        &fits,
        &segment_keys(&["A", "B"], &cl.origins),
        &latest,
        1e-12,
    );
    check_means(&fits, &fits.to_long());
}

#[test]
fn the_bayesian_bornhuetter_fergusons() {
    let genins = triangle_columns("genins_premium", &["paid", "premium"]);
    let bf = BayesBf {
        apriori: 0.5,
        cv: 0.2,
        sampler: short(2, 300),
        n_sims: 1_000,
        seed: 4,
        ..BayesBf::default()
    };
    let fit = bf.fit(&genins, "paid", "premium").unwrap();
    let cl = &fit.chain_ladder;
    check_layout(&fit, &origin_keys(&cl.origins), &cl.latest, 1e-12);
    check_means(&fit, &fit.to_long());
    same_draws(
        &fit.ultimates().unwrap(),
        &bf.fit(&genins, "paid", "premium")
            .unwrap()
            .ultimates()
            .unwrap(),
    );

    let normalised = BayesBfNormalised {
        apriori: 0.5,
        cv: 0.2,
        sampler: short(2, 300),
        n_sims: 1_000,
        seed: 4,
        ..BayesBfNormalised::default()
    };
    let fit = normalised.fit(&genins, "paid", "premium").unwrap();
    let cl = &fit.chain_ladder;
    check_layout(&fit, &origin_keys(&cl.origins), &cl.latest, 1e-12);
    check_means(&fit, &fit.to_long());
}

/// Group 353 Commercial Auto at calendar years up to 1997: paid and
/// premium.
fn comauto() -> &'static Triangle {
    static TRIANGLE: OnceLock<Triangle> = OnceLock::new();
    TRIANGLE.get_or_init(|| {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/clrd_comauto_353.csv");
        let text = std::fs::read_to_string(path).expect("data/clrd_comauto_353.csv");
        let rows: Vec<(i32, Lag, f64, f64)> = text
            .lines()
            .filter(|l| !l.starts_with('#'))
            .skip(1)
            .map(|l| {
                let f: Vec<&str> = l.split(',').collect();
                (
                    f[0].parse().unwrap(),
                    f[1].parse().unwrap(),
                    f[2].parse().unwrap(),
                    f[4].parse().unwrap(),
                )
            })
            .filter(|r: &(i32, Lag, f64, f64)| r.0 + (r.1 / 12) as i32 - 1 <= 1997)
            .collect();
        assert_eq!(rows.len(), 55);
        let origin: Vec<Month> = rows.iter().map(|r| Month::january(r.0)).collect();
        let ages: Vec<Lag> = rows.iter().map(|r| r.1).collect();
        let paid: Vec<f64> = rows.iter().map(|r| r.2).collect();
        let premium: Vec<f64> = rows.iter().map(|r| r.3).collect();
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid), ("premium", &premium)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    })
}

#[test]
fn meyers_crc_keeps_its_own_ultimates() {
    let model = Meyers {
        model: MeyersModel::Crc,
        sampler: Sampler {
            tune: 300,
            draws: 300,
            ..Meyers::default().sampler
        },
        n_sims: 1_000,
        seed: 5,
        ..Meyers::default()
    };
    let fit = model.fit(comauto(), "paid", "premium").unwrap();
    let cl = &fit.chain_ladder;
    // The trait's ultimates are the fit's own draws.
    same_draws(&fit.ultimates().unwrap(), &fit.ultimates);
    check_layout(&fit, &origin_keys(&cl.origins), &cl.latest, 1e-12);
    let table = fit.to_long();
    let ultimate_mean = table.column("ultimate_mean").unwrap();
    let ulr_mean = table.column("ulr_mean").unwrap();
    for (j, key) in origin_keys(&cl.origins).iter().enumerate() {
        let u = fit.ultimate_draws(key).unwrap();
        let m = u.iter().sum::<f64>() / u.len() as f64;
        assert!((m - ultimate_mean[j]).abs() <= 1e-9 * m);
        let q = fit.loss_ratio_draws(key, fit.premium[j]).unwrap();
        let mq = q.iter().sum::<f64>() / q.len() as f64;
        assert!((mq - ulr_mean[j]).abs() <= 1e-9 * mq);
    }
    same_draws(
        &fit.ultimates,
        &model.fit(comauto(), "paid", "premium").unwrap().ultimates,
    );

    let fits = model
        .fit_segments(&twins(comauto(), &["paid", "premium"]), "paid", "premium")
        .unwrap();
    same_draws(&fits.ultimates().unwrap(), &fits.ultimates);
    let latest: Vec<f64> = twin_latest(&cl.latest[..]);
    check_layout(
        &fits,
        &segment_keys(&["A", "B"], &cl.origins),
        &latest,
        1e-12,
    );
}

/// Company 337 at calendar years up to 1996, with accident year 1997's
/// premium alone (the papers' training triangle).
fn company_337() -> &'static Triangle {
    static TRIANGLE: OnceLock<Triangle> = OnceLock::new();
    TRIANGLE.get_or_init(|| {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/clrd_wkcomp_337.csv");
        let text = std::fs::read_to_string(path).expect("data/clrd_wkcomp_337.csv");
        let rows: Vec<(i32, Lag, f64, f64, f64)> = text
            .lines()
            .filter(|l| !l.starts_with('#'))
            .skip(1)
            .map(|l| {
                let f: Vec<&str> = l.split(',').collect();
                (
                    f[0].parse().unwrap(),
                    f[1].parse().unwrap(),
                    f[2].parse().unwrap(),
                    f[3].parse().unwrap(),
                    f[5].parse().unwrap(),
                )
            })
            .collect();
        let premium_1997 = rows.iter().find(|r| r.0 == 1997).unwrap().4;
        let mut kept: Vec<_> = rows
            .into_iter()
            .filter(|r| r.0 + (r.1 / 12) as i32 - 1 <= 1996)
            .collect();
        kept.push((1997, 12, f64::NAN, f64::NAN, premium_1997));
        let origin: Vec<Month> = kept.iter().map(|r| Month::january(r.0)).collect();
        let ages: Vec<Lag> = kept.iter().map(|r| r.1).collect();
        let paid: Vec<f64> = kept.iter().map(|r| r.2).collect();
        let incurred: Vec<f64> = kept.iter().map(|r| r.3).collect();
        let premium: Vec<f64> = kept.iter().map(|r| r.4).collect();
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[
                ("paid", &paid),
                ("incurred", &incurred),
                ("premium", &premium),
            ],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    })
}

#[test]
fn compartmental_model_1_is_on_the_paid_basis() {
    let preset = Compartmental::new(CompartmentalPriors::gesmann_morris());
    let model = Compartmental {
        sampler: Sampler {
            chains: 2,
            tune: 200,
            draws: 200,
            ..preset.sampler
        },
        n_sims: 500,
        seed: 6,
        ..preset
    };
    let fit = model
        .fit(
            company_337(),
            "paid",
            Outstanding::FromIncurred("incurred"),
            "premium",
        )
        .unwrap();
    // The 1997 origin has no paid: its ultimate is its reserve.
    assert_eq!(*fit.latest_paid.last().unwrap(), 0.0);
    check_layout(&fit, &origin_keys(&fit.origins), &fit.latest_paid, 1e-12);
    // The ultimate is the paid at the square's last age in the same
    // simulation: the last future paid cell of each origin.
    let last_age = fit
        .paid
        .components()
        .iter()
        .filter_map(|k| match k[1] {
            KeyValue::Int(a) => Some(a),
            _ => None,
        })
        .max()
        .unwrap();
    for &origin in &fit.origins[1..] {
        let key = vec![KeyValue::Period(origin), KeyValue::Int(last_age)];
        let paid = fit.paid.marginal(&key).unwrap();
        let paid = prospicio_prob::Empirical::draws(&paid);
        let ultimate = fit.ultimate_draws(&[KeyValue::Period(origin)]).unwrap();
        for (u, p) in ultimate.iter().zip(paid) {
            assert!(
                (u - p).abs() <= 1e-9 * u.abs().max(1.0),
                "{origin}: {u} vs {p}"
            );
        }
    }
    let again = model
        .fit(
            company_337(),
            "paid",
            Outstanding::FromIncurred("incurred"),
            "premium",
        )
        .unwrap();
    same_draws(&fit.ultimates().unwrap(), &again.ultimates().unwrap());

    let fits = model
        .fit_segments(
            &twins(company_337(), &["paid", "incurred", "premium"]),
            "paid",
            Outstanding::FromIncurred("incurred"),
            "premium",
        )
        .unwrap();
    let latest: Vec<f64> = twin_latest(&fit.latest_paid[..]);
    check_layout(
        &fits,
        &segment_keys(&["A", "B"], &fit.origins),
        &latest,
        1e-12,
    );
}
