//! `Distribution::sample_one` (#972): `k` calls give the same numbers, bit
//! for bit, as one `sample(rng, k)` from the same stream, for the families
//! that override the draw (Gamma, InverseGamma, Affine, Dist, SeverityDist,
//! PredictiveDistribution and the `Box` forward) and for families on the
//! default inverse transform. The bootstraps and Bayesian models draw one
//! value at a time through it, so their seeded numbers rest on this.

use prospicio_core::StreamRng;
use prospicio_prob::{
    Affine, Dist, Distribution, Gamma, InverseGamma, KeyValue, Lognormal, Normal, Pareto,
    PredictiveDistribution, Provenance, Sampled, SeverityDist, Tweedie,
};

const K: usize = 257;

/// `sample_one` `K` times and `sample(K)` from the same seed and stream
/// agree bit for bit, and leave the stream in the same place.
fn agrees(name: &str, d: &dyn Distribution) {
    for (seed, stream) in [(972, 0), (1, 7), (20_261_123, 3)] {
        let mut a = StreamRng::new(seed, stream);
        let mut b = StreamRng::new(seed, stream);
        let one: Vec<u64> = (0..K).map(|_| d.sample_one(&mut a).to_bits()).collect();
        let many: Vec<u64> = d.sample(&mut b, K).iter().map(|x| x.to_bits()).collect();
        assert_eq!(one, many, "{name}: seed {seed}, stream {stream}");
        assert_eq!(
            a.next_open01().to_bits(),
            b.next_open01().to_bits(),
            "{name}: the streams moved by different amounts"
        );
    }
}

fn severity(d: impl Into<Dist>) -> SeverityDist {
    SeverityDist::try_from(d.into()).expect("a parametric severity")
}

#[test]
fn sample_one_matches_sample_for_every_override() {
    // Marsaglia and Tsang, either side of shape 1 (the boost below it).
    for shape in [0.3, 1.0, 2.5, 4_000.0] {
        agrees(&format!("gamma {shape}"), &Gamma::new(shape, 3.0).unwrap());
        agrees(
            &format!("inverse gamma {shape}"),
            &InverseGamma::new(shape, 3.0).unwrap(),
        );
    }
    let gamma = Gamma::new(2.5, 3.0).unwrap();
    agrees("dist gamma", &Dist::from(gamma));
    agrees("severity gamma", &severity(gamma));
    agrees(
        "affine gamma",
        &Affine::new(severity(gamma), 2.0, 5.0).unwrap(),
    );
    let boxed: Box<dyn Distribution> = Box::new(gamma);
    agrees("box dyn gamma", &boxed);
    let draws: Vec<f64> = (1..=40).map(f64::from).collect();
    let predictive = PredictiveDistribution::from_draws(
        vec!["lob".into()],
        vec![vec![KeyValue::from("a")], vec![KeyValue::from("b")]],
        draws.clone(),
        Provenance::new("test"),
    )
    .unwrap();
    agrees("predictive", &predictive);
    agrees("sampled", &Sampled::new(draws).unwrap());
}

#[test]
fn sample_one_matches_sample_on_the_default_inverse_transform() {
    let lognormal = Lognormal::from_mean_cv(100.0, 0.3).unwrap();
    agrees("lognormal", &lognormal);
    agrees("dist lognormal", &Dist::from(lognormal));
    agrees(
        "affine lognormal",
        &Affine::new(severity(lognormal), 0.5, 1.0).unwrap(),
    );
    agrees("normal", &Normal::new(1.0, 2.0).unwrap());
    agrees("pareto", &Pareto::new(10.0, 2.0).unwrap());
    agrees("tweedie", &Tweedie::new(100.0, 2.0, 1.5).unwrap());
}
