//! Frozen files of every saved-artifact format and version (#840).
//!
//! Each file under `validation/reference/artifacts/` was written once by the
//! build that added it and is never rewritten. The test reads every file
//! with today's reader and checks that
//!
//! - it loads, and its `format_version` is the one in its name;
//! - today's writer gives the same document back (as JSON values), so a key
//!   renamed or dropped in either the reader or the writer fails here,
//!   naming the file;
//! - what it holds is what the file was made from: towers and distributions
//!   equal the builders below, and fitted models predict as a refit does.
//!
//! A new format version adds a builder to the matching list and runs
//!
//! ```sh
//! cargo test -p prospicio-validation --test validation artifacts::write_missing_artifacts -- --ignored
//! ```
//!
//! which writes only the files that do not exist yet. Never delete and
//! rewrite an existing file: that would hide the change it exists to catch.

use std::path::PathBuf;

use prospicio_aggregate::{Layer, LimitAllocation, LossSensitivePremium, Tower};
use prospicio_glm::gam::{Gam, GamFit, PSpline};
use prospicio_glm::net::{ElasticNet, ElasticNetFit};
use prospicio_glm::{Glm, GlmFit};
use prospicio_models::{Design, Family, Fitted, Link, Model};
use prospicio_nn::{
    AttentionCann, AttentionCannFit, Cann, CannFit, Head, TabularNet, TabularNetFit, Target,
};
use prospicio_prob::{Dist, Distribution, Gamma, Lognormal, Mixture, Pareto, SeverityDist};
use serde_json::Value;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("reference/artifacts")
}

fn version_of(text: &str) -> u64 {
    let doc: Value = serde_json::from_str(text).unwrap();
    doc["format_version"].as_u64().unwrap()
}

/// One tower per format version, each needing exactly that version.
fn towers() -> Vec<(u64, Tower)> {
    let xol = |name: &str| Layer::xol(name, 10.0, 5.0).unwrap();
    let rated = |name: &str, attachment: f64| {
        Layer::xol(name, 50.0, attachment)
            .unwrap()
            .premium_rate(1.0, 100.0)
            .unwrap()
    };
    vec![
        (
            1,
            Tower::inuring(vec![
                vec![
                    Layer::quota_share("QS", 0.3)
                        .unwrap()
                        .loss_corridor(1e6, 3e6, 0.5)
                        .unwrap()
                        .premium_rate(1.0, 2e7)
                        .unwrap()
                        .sliding_scale(vec![(0.35, 0.55), (0.3, 0.6), (0.2, 0.75)])
                        .unwrap()
                        .profit_commission(0.15, 0.35)
                        .unwrap(),
                    Layer::surplus("surplus", 1e6, 4.0)
                        .unwrap()
                        .aggregate_limit(2e7)
                        .unwrap(),
                ],
                vec![
                    Layer::xol("2x1", 2e6, 1e6)
                        .unwrap()
                        .share(0.6)
                        .unwrap()
                        .aggregate_deductible(5e5)
                        .unwrap()
                        .paid_reinstatements(3e5, vec![1.0, 0.5])
                        .unwrap()
                        .pro_rata_as_to_time()
                        .unwrap()
                        .ceding_commission(0.1)
                        .unwrap(),
                    Layer::xol("free", 5e6, 3e6)
                        .unwrap()
                        .reinstatements(2)
                        .unwrap()
                        .swing_rated(
                            LossSensitivePremium::new(1e5, 1.25, Some(2e5), Some(2e6)).unwrap(),
                        )
                        .unwrap(),
                    Layer::xol("pro rata", 10.0, 5.0)
                        .unwrap()
                        .reinstatements(1)
                        .unwrap()
                        .allocation(LimitAllocation::ProRata)
                        .unwrap(),
                ],
                vec![Layer::stop_loss("SL", f64::INFINITY, 2e7).unwrap()],
            ])
            .unwrap(),
        ),
        (
            2,
            Tower::new(vec![
                Layer::quota_share("QS", 0.3)
                    .unwrap()
                    .premium_rate(1.0, 2e7)
                    .unwrap()
                    .sliding_scale(vec![(0.35, 0.55), (0.2, 0.75)])
                    .unwrap()
                    .profit_commission(0.15, 0.35)
                    .unwrap()
                    .sliding_scale_carry_forward(true, true, Some(3))
                    .unwrap()
                    .profit_commission_carry_forward(None)
                    .unwrap(),
            ])
            .unwrap(),
        ),
        (
            3,
            Tower::new(vec![
                Layer::quota_share("QS", 1.0)
                    .unwrap()
                    .aggregate_limit(120.0)
                    .unwrap()
                    .premium_rate(1.0, 100.0)
                    .unwrap()
                    .sliding_scale(vec![(0.3, 0.5), (0.1, 0.9)])
                    .unwrap()
                    .brokerage_scale(vec![(0.01, 0.5), (0.03, 0.9)])
                    .unwrap()
                    .deduction(prospicio_aggregate::Deduction::premium("tax", 0.03).unwrap())
                    .unwrap()
                    .deduction(prospicio_aggregate::Deduction::loss("tpa", 0.02).unwrap())
                    .unwrap()
                    .deduction(
                        prospicio_aggregate::Deduction::fixed("fronting", 1.5)
                            .unwrap()
                            .outside_nmbf(),
                    )
                    .unwrap()
                    .scales_before_annual_terms()
                    .brokerage_within_commission_range(),
                Layer::xol("XL", 50.0, 50.0)
                    .unwrap()
                    .premium_rate(1.0, 20.0)
                    .unwrap()
                    .brokerage(0.1)
                    .unwrap(),
            ])
            .unwrap(),
        ),
        (
            4,
            Tower::new(vec![rated("A", 0.0), rated("B", 50.0)])
                .unwrap()
                .deal_scale(
                    prospicio_aggregate::DealScale::new("deal", ["A", "B"])
                        .unwrap()
                        .sliding_scale(vec![(0.3, 0.5), (0.1, 0.9)])
                        .unwrap()
                        .carry_forward(true, true, Some(3))
                        .unwrap()
                        .brokerage_scale(vec![(0.01, 0.5), (0.03, 0.9)])
                        .unwrap()
                        .brokerage_within_commission_range(),
                )
                .unwrap(),
        ),
        (
            5,
            Tower::new(vec![xol("L")])
                .unwrap()
                .cover_basis(prospicio_aggregate::timeline::CoverBasis::LossesOccurring),
        ),
        (
            6,
            Tower::new(vec![xol("L")])
                .unwrap()
                .cover_basis(prospicio_aggregate::timeline::CoverBasis::RisksAttaching)
                .interlocking(true),
        ),
        (
            7,
            Tower::new(vec![
                xol("L").scope("peril", &["ws", "fl"]).unwrap(),
                Layer::xol("M", 5.0, 20.0).unwrap(),
            ])
            .unwrap(),
        ),
        (
            8,
            Tower::new(vec![
                Layer::xol("L", 10.0, 5.0)
                    .unwrap()
                    .premium_rate(0.1, 100.0)
                    .unwrap()
                    .no_claims_bonus(0.2, 3, 1.5)
                    .unwrap(),
            ])
            .unwrap(),
        ),
        (
            9,
            Tower::new(vec![
                Layer::xol("lower", 10.0, 10.0)
                    .unwrap()
                    .aggregate_limit(20.0)
                    .unwrap(),
                Layer::xol("top", 10.0, 30.0)
                    .unwrap()
                    .top_and_drop("lower", 10.0)
                    .unwrap(),
                Layer::xol("f", 5.0, 2.0).unwrap().franchise().unwrap(),
                Layer::xol("2nd", 10.0, 5.0)
                    .unwrap()
                    .nth_event(2, 5.0)
                    .unwrap(),
            ])
            .unwrap(),
        ),
        (
            10,
            Tower::new(vec![
                Layer::xol("d", 300.0, 50.0)
                    .unwrap()
                    .disappearing(100.0)
                    .unwrap(),
            ])
            .unwrap(),
        ),
    ]
}

/// Distributions of `risk_rs.distribution` version 1: a family, a
/// spliced and a plain mixture of families.
fn distributions() -> Vec<(&'static str, Dist)> {
    let ln = Lognormal::from_mean_cv(1000.0, 0.5).unwrap();
    let sev = |d: Dist| SeverityDist::try_from(d).unwrap();
    vec![
        ("gamma", Gamma::new(2.0, 500.0).unwrap().into()),
        (
            "mixture",
            Dist::Mixture(std::sync::Arc::new(
                Mixture::from_dists(vec![
                    (0.7, sev(ln.into())),
                    (0.3, sev(Pareto::new(1e5, 2.0).unwrap().into())),
                ])
                .unwrap(),
            )),
        ),
    ]
}

/// A small design and responses shared by the fitted models.
fn data() -> (Design, Vec<f64>) {
    let n = 60;
    let x: Vec<f64> = (0..n).map(|i| (i % 12) as f64).collect();
    let b: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 3 == 1))).collect();
    let y: Vec<f64> = x
        .iter()
        .zip(&b)
        .enumerate()
        .map(|(i, (v, b))| 2.0 + v.sin() + 0.5 * b + 0.1 * ((i * 7) % 5) as f64)
        .collect();
    let d = Design::new(
        vec!["(Intercept)".into(), "x".into(), "b".into()],
        vec![vec![1.0; n], x, b],
    )
    .unwrap();
    (d, y)
}

/// A fitted model saved as a frozen file: its name, its document, and how
/// to load one and predict on [`data`]'s design.
struct Fit {
    name: &'static str,
    version: u64,
    make: fn() -> String,
    predict: fn(&str, &Design) -> Vec<f64>,
}

fn net_spec() -> ElasticNet {
    let mut spec = ElasticNet::new(Family::Gaussian, Link::Identity, 0.5, 0.05);
    spec.penalty_factor = Some(vec![0.0, 1.0, 2.0]);
    spec
}

fn cann() -> Cann {
    let mut spec = Cann::new(Family::Gamma, Link::Log);
    spec.hidden = vec![4];
    spec.epochs = 3;
    spec
}

fn attention() -> AttentionCann {
    let mut spec = AttentionCann::new(Family::Gamma, Link::Log);
    spec.d_model = 4;
    spec.heads = 2;
    spec.layers = 1;
    spec.feed_forward = 6;
    spec.epochs = 2;
    spec
}

fn tabular() -> TabularNet {
    let mut spec = TabularNet::new()
        .hidden(&[4])
        .head(Head::new("y", Family::Gamma));
    spec.epochs = 3;
    spec
}

fn fits() -> Vec<Fit> {
    vec![
        Fit {
            name: "glm_fit",
            version: 1,
            make: || {
                let (d, y) = data();
                Glm::new(Family::Gamma, Link::Log)
                    .fit(&d, &y)
                    .unwrap()
                    .to_json()
            },
            predict: |text, d| GlmFit::from_json(text).unwrap().predict(d).unwrap(),
        },
        Fit {
            name: "gam_fit",
            version: 1,
            make: || {
                let (d, y) = data();
                Gam::new(
                    Glm::new(Family::Gamma, Link::Log),
                    vec![PSpline::new("x").n_basis(6)],
                )
                .fit(&d, &y)
                .unwrap()
                .to_json()
            },
            predict: |text, d| GamFit::from_json(text).unwrap().predict(d).unwrap(),
        },
        Fit {
            name: "gam_fit",
            version: 2,
            make: || {
                let (d, y) = data();
                Gam::new(
                    Glm::new(Family::Gamma, Link::Log),
                    vec![PSpline::cyclic("x").n_basis(6).period(0.0, 12.0)],
                )
                .fit(&d, &y)
                .unwrap()
                .to_json()
            },
            predict: |text, d| GamFit::from_json(text).unwrap().predict(d).unwrap(),
        },
        Fit {
            name: "elastic_net_fit",
            version: 1,
            make: || {
                let (d, y) = data();
                net_spec().fit(&d, &y).unwrap().to_json()
            },
            predict: |text, d| ElasticNetFit::from_json(text).unwrap().predict(d).unwrap(),
        },
        Fit {
            name: "elastic_net_fit",
            version: 2,
            make: || {
                let (d, y) = data();
                net_spec()
                    .lower_limits(vec![f64::NEG_INFINITY, 0.0, -1.0])
                    .upper_limits(vec![f64::INFINITY, f64::INFINITY, 0.5])
                    .fit(&d, &y)
                    .unwrap()
                    .relax(&d, &y, 0.25)
                    .unwrap()
                    .to_json()
            },
            predict: |text, d| ElasticNetFit::from_json(text).unwrap().predict(d).unwrap(),
        },
        Fit {
            name: "cann_fit",
            version: 1,
            make: || {
                let (d, y) = data();
                cann().fit(&d, &y).unwrap().to_json()
            },
            predict: |text, d| CannFit::from_json(text).unwrap().predict(d).unwrap(),
        },
        Fit {
            name: "attention_cann_fit",
            version: 1,
            make: || {
                let (d, y) = data();
                attention().fit(&d, &y).unwrap().to_json()
            },
            predict: |text, d| {
                AttentionCannFit::from_json(text)
                    .unwrap()
                    .predict(d)
                    .unwrap()
            },
        },
        Fit {
            name: "tabular_net_fit",
            version: 1,
            make: || {
                let (d, y) = data();
                tabular()
                    .fit_heads(&d, &[Target::new(&y)])
                    .unwrap()
                    .to_json()
            },
            predict: |text, d| {
                TabularNetFit::from_json(text)
                    .unwrap()
                    .predict_head(0, d, None)
                    .unwrap()
            },
        },
    ]
}

/// Today's writer gives the frozen document back, value for value, except
/// the version of the writing crate in a model's provenance: a writer
/// stamps its own (`prospicio_models::artifact::provenance`), so it moves
/// with every release while the crate names and everything else stay.
fn same_document(file: &str, frozen: &str, rewritten: &str) {
    let (mut a, mut b): (Value, Value) = (
        serde_json::from_str(frozen).unwrap(),
        serde_json::from_str(rewritten).unwrap(),
    );
    for doc in [&mut a, &mut b] {
        if let Some(Value::Array(versions)) = doc.pointer_mut("/provenance/versions") {
            for pair in versions.iter_mut() {
                if let Some(version) = pair.get_mut(1) {
                    *version = Value::Null;
                }
            }
        }
    }
    assert_eq!(
        a, b,
        "{file}: today's reader and writer do not give the frozen document back"
    );
}

fn read(file: &str) -> String {
    let path = dir().join(file);
    std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}; run the ignored write_missing_artifacts once",
            path.display()
        )
    })
}

#[test]
fn every_frozen_tower_loads_as_it_was_written() {
    let towers = towers();
    assert_eq!(towers.len(), 10, "one tower per format version, 1 to 10");
    for (version, built) in towers {
        let file = format!("tower_v{version}.json");
        let text = read(&file);
        assert_eq!(version_of(&text), version, "{file}");
        let tower = Tower::from_json(&text).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(tower, built, "{file}: not the tower it was written from");
        same_document(&file, &text, &tower.to_json());
    }
}

#[test]
fn every_frozen_distribution_loads_as_it_was_written() {
    for (name, built) in distributions() {
        let file = format!("distribution_v1_{name}.json");
        let text = read(&file);
        assert_eq!(version_of(&text), 1, "{file}");
        let d = Dist::from_json(&text).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(d.family(), built.family(), "{file}");
        assert_eq!(d.mean().to_bits(), built.mean().to_bits(), "{file}");
        for p in [0.1, 0.5, 0.9] {
            let (a, b) = (d.quantile(p).unwrap(), built.quantile(p).unwrap());
            assert_eq!(a.to_bits(), b.to_bits(), "{file} at {p}");
        }
        same_document(&file, &text, &d.to_json().unwrap());
    }
}

#[test]
fn every_frozen_model_loads_and_predicts_as_a_refit() {
    let (d, _) = data();
    for fit in fits() {
        let file = format!("{}_v{}.json", fit.name, fit.version);
        let text = read(&file);
        assert_eq!(version_of(&text), fit.version, "{file}");
        let frozen = (fit.predict)(&text, &d);
        assert_eq!(frozen.len(), d.n_rows(), "{file}");
        assert!(frozen.iter().all(|v| v.is_finite()), "{file}");
        // A refit today predicts as the frozen fit did, within the
        // round-off of another platform (the networks train in f32).
        let refit = (fit.predict)(&(fit.make)(), &d);
        for (i, (a, b)) in frozen.iter().zip(&refit).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                "{file} row {i}: frozen {a}, refit {b}"
            );
        }
        // Read back and written again, the document is unchanged.
        let rewritten = match fit.name {
            "glm_fit" => GlmFit::from_json(&text).unwrap().to_json(),
            "gam_fit" => GamFit::from_json(&text).unwrap().to_json(),
            "elastic_net_fit" => ElasticNetFit::from_json(&text).unwrap().to_json(),
            "cann_fit" => CannFit::from_json(&text).unwrap().to_json(),
            "attention_cann_fit" => AttentionCannFit::from_json(&text).unwrap().to_json(),
            "tabular_net_fit" => TabularNetFit::from_json(&text).unwrap().to_json(),
            other => panic!("no reader for {other}"),
        };
        same_document(&file, &text, &rewritten);
    }
}

#[test]
fn format_version_0_is_refused_by_every_reader() {
    let zero = |file: &str| {
        let mut doc: Value = serde_json::from_str(&read(file)).unwrap();
        doc["format_version"] = Value::from(0);
        doc.to_string()
    };
    assert!(Tower::from_json(&zero("tower_v1.json")).is_err());
    assert!(Dist::from_json(&zero("distribution_v1_gamma.json")).is_err());
    assert!(GlmFit::from_json(&zero("glm_fit_v1.json")).is_err());
    assert!(CannFit::from_json(&zero("cann_fit_v1.json")).is_err());
}

/// Writes the frozen files that do not exist yet; never rewrites one.
#[test]
#[ignore = "run by hand to add a new format version's file"]
fn write_missing_artifacts() {
    std::fs::create_dir_all(dir()).unwrap();
    let write = |file: String, text: String, version: u64| {
        assert_eq!(
            version_of(&text),
            version,
            "{file}: the writer stamps another version"
        );
        let path = dir().join(&file);
        if !path.exists() {
            let pretty =
                serde_json::to_string_pretty(&serde_json::from_str::<Value>(&text).unwrap())
                    .unwrap();
            std::fs::write(&path, pretty + "\n").unwrap();
            println!("wrote {file}");
        }
    };
    for (version, tower) in towers() {
        write(format!("tower_v{version}.json"), tower.to_json(), version);
    }
    for (name, d) in distributions() {
        write(
            format!("distribution_v1_{name}.json"),
            d.to_json().unwrap(),
            1,
        );
    }
    for fit in fits() {
        write(
            format!("{}_v{}.json", fit.name, fit.version),
            (fit.make)(),
            fit.version,
        );
    }
}
