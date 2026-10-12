//! Aggregate parity: compound distributions against brute-force
//! convolution (numpy), which is independent of Panjer's recursion and FFT.

use prospicio_aggregate::{fft, panjer};
use prospicio_prob::{Counting, Grid, NegativeBinomial, Poisson};
use prospicio_validation::{check, reference};

fn check_method(method: &str) {
    let cases = reference("compound_convolution.csv");
    check(&cases, |c| {
        let n: Box<dyn Counting> = match c.get("frequency") {
            "poisson" => Box::new(Poisson::new(c.param("params", "lambda")).ok()?),
            "negative_binomial" => Box::new(
                NegativeBinomial::new(c.param("params", "r"), c.param("params", "beta")).ok()?,
            ),
            _ => return None,
        };
        let severity: Vec<f64> = c
            .get("severity")
            .split(';')
            .map(|p| p.parse().unwrap())
            .collect();
        let sev = Grid::new(1.0, severity).ok()?;
        // Compare the interior points; the last point carries the lumped tail.
        let points = c.number("points")? as usize;
        let index = c.number("index")? as usize;
        let (agg, _) = match method {
            "panjer" => panjer(n.as_ref(), &sev, points + 1),
            _ => fft(n.as_ref(), &sev, points + 1),
        }
        .ok()?;
        agg.probs().get(index).copied()
    });
}

#[test]
fn panjer_matches_convolution() {
    check_method("panjer");
}

#[test]
fn fft_matches_convolution() {
    check_method("fft");
}

/// The fields of `params`, such as `model=ppp;FQ=2;t=1000|2000;...`.
fn fields(c: &prospicio_validation::Case) -> std::collections::HashMap<&str, &str> {
    c.get("params")
        .split(';')
        .filter_map(|kv| kv.split_once('='))
        .collect()
}

#[test]
fn collective_model_matches_r() {
    use prospicio_aggregate::CollectiveModel;
    use prospicio_prob::{PanjerClass, PiecewisePareto, Severity, evt::Gpd};
    let cases = reference("collective_r.csv");
    check(&cases, |c| {
        let f = fields(c);
        let num = |k: &str| -> Option<f64> { f.get(k)?.parse().ok() };
        let list = |k: &str| -> Option<Vec<f64>> {
            f.get(k)?.split('|').map(|v| v.parse().ok()).collect()
        };
        let freq = PanjerClass::from_mean_dispersion(num("FQ")?, num("dispersion")?).ok()?;
        let severity: Box<dyn Severity> = match *f.get("model")? {
            "ppp" => Box::new(PiecewisePareto::new(list("t")?, list("alpha")?).ok()?),
            "pgp" => Box::new(Gpd::riegel(num("t")?, num("alpha_ini")?, num("alpha_tail")?).ok()?),
            _ => return None,
        };
        let model = CollectiveModel::new(freq, severity);
        let arg = |k: &str| -> Option<f64> {
            match c.get(k) {
                "inf" => Some(f64::INFINITY),
                s => s.parse().ok(),
            }
        };
        match c.get("quantity") {
            "excess_frequency" => Some(model.excess_frequency(arg("arg")?)),
            "layer_mean" => Some(model.layer_mean(arg("arg")?, arg("arg2")?)),
            "layer_variance" => Some(model.layer_variance(arg("arg")?, arg("arg2")?)),
            _ => None,
        }
    });
}

/// The roadmap's gate: a reserve bootstrap and a reinsurance tower feed
/// capital allocation end to end. RAA reserves by origin (ODP bootstrap)
/// with an adverse development cover, premium risk from a collective model
/// net of an excess-of-loss layer, joined into one portfolio with a rank
/// correlation between the two risks, then measured and allocated.
#[test]
fn reserve_and_tower_feed_capital_end_to_end() {
    use prospicio_aggregate::{Layer, Tower, simulate_events};
    use prospicio_prob::capital::AllocationMethod;
    use prospicio_prob::portfolio::Pairing;
    use prospicio_prob::{
        Distortion, Distribution, Empirical, KeyValue, Lognormal, PredictiveDistribution,
    };
    use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, OdpBootstrap, Triangle};

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/raa.csv");
    let text = std::fs::read_to_string(path).unwrap();
    let rows: Vec<Vec<f64>> = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("origin"))
        .map(|l| l.split(',').map(|v| v.parse().unwrap()).collect())
        .collect();
    let origin: Vec<Month> = rows.iter().map(|r| Month::january(r[0] as i32)).collect();
    let age: Vec<u32> = rows.iter().map(|r| r[1] as u32).collect();
    let value: Vec<f64> = rows.iter().map(|r| r[2]).collect();
    let tri = Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("paid", &value)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    let n = 20_000;
    let boot = OdpBootstrap {
        n_sims: n,
        seed: 11,
        ..Default::default()
    }
    .fit(&tri, "paid")
    .unwrap();
    let reserve = &boot.reserves;

    // An adverse development cover: 10,000 xs the reserve's 75th percentile.
    let retention = reserve.quantile(0.75).unwrap();
    let adc = Tower::new(vec![Layer::stop_loss("adc", 10_000.0, retention).unwrap()])
        .unwrap()
        .apply_aggregate(reserve)
        .unwrap()
        .aggregate(&["kind"])
        .unwrap();
    let ceded = adc.marginal(&vec![KeyValue::from("ceded")]).unwrap();
    let net = adc.marginal(&vec![KeyValue::from("net")]).unwrap();
    assert!((ceded.mean() + net.mean() - reserve.mean()).abs() < 1e-9 * reserve.mean());
    // The cover caps the reserve's tail.
    assert!(net.quantile(0.995).unwrap() < reserve.quantile(0.995).unwrap());

    // Premium risk for next year, net of a 2,000 xs 1,000 per-risk layer.
    let events = simulate_events(
        &prospicio_prob::Poisson::new(8.0).unwrap(),
        &Lognormal::from_mean_cv(800.0, 1.5).unwrap(),
        n,
        12,
    )
    .unwrap();
    let premium = Tower::new(vec![Layer::xol("2x1", 2_000.0, 1_000.0).unwrap()])
        .unwrap()
        .apply(&events)
        .unwrap();
    let premium_net = PredictiveDistribution::from_draws(
        vec!["lob".into()],
        vec![vec![KeyValue::from("property")]],
        premium
            .aggregate(&["kind"])
            .unwrap()
            .marginal(&vec![KeyValue::from("net")])
            .unwrap()
            .draws()
            .to_vec(),
        premium.provenance().clone(),
    )
    .unwrap();

    // The reserve by origin, gross, and the cover as a negative line, so
    // capital can still be allocated to origins. The cover is a function
    // of the reserve simulations: refused as an independent part, joined
    // to the reserve scenario by scenario.
    let cover = PredictiveDistribution::from_draws(
        vec!["layer".into()],
        vec![vec![KeyValue::from("adc")]],
        ceded.draws().iter().map(|c| -c).collect(),
        adc.provenance().clone(),
    )
    .unwrap();
    let reserve_parts = [("reserve", reserve), ("adc", &cover)];
    assert!(PredictiveDistribution::join(&reserve_parts, "part", Pairing::Independent).is_err());
    let reserve_with_cover =
        PredictiveDistribution::join(&reserve_parts, "part", Pairing::SameSimulations).unwrap();
    let parts = [("reserve", &reserve_with_cover), ("premium", &premium_net)];
    let portfolio = PredictiveDistribution::join(&parts, "risk", Pairing::Independent)
        .unwrap()
        .reorder_groups("risk", &[1.0, 0.5, 0.5, 1.0], 13)
        .unwrap();
    assert_eq!(portfolio.dims(), ["risk", "part", "origin", "layer", "lob"]);

    // The total is the reserve net of the cover plus premium net of the layer.
    let expected_mean = net.mean() + premium_net.mean();
    assert!((portfolio.mean() - expected_mean).abs() < 1e-9 * expected_mean);

    // Capital: TVaR 99% of the total, allocated to risks and to origins.
    let tvar = Distortion::tvar(0.99).unwrap();
    let by_risk = portfolio.aggregate(&["risk"]).unwrap();
    let alloc = by_risk.capital(&tvar, AllocationMethod::Euler).unwrap();
    assert!((alloc.allocated.iter().sum::<f64>() - alloc.total).abs() < 1e-6 * alloc.total);
    assert!(alloc.diversification_benefit() > 0.0);
    let by_component = portfolio.allocate(&tvar);
    assert_eq!(by_component.len(), portfolio.n_components());
    assert!((by_component.iter().sum::<f64>() - alloc.total).abs() < 1e-6 * alloc.total);
    // Positive dependence between the risks costs capital.
    let independent = PredictiveDistribution::join(&parts, "risk", Pairing::Independent)
        .unwrap()
        .reorder_groups("risk", &[1.0, 0.0, 0.0, 1.0], 13)
        .unwrap();
    assert!(portfolio.distortion(&tvar) > independent.distortion(&tvar));
}

/// FFT grid sizing against `aggregate` 1.0.1: its `round_bucket` exactly,
/// and on five books sized with no bucket given, the exact mean and
/// aggregate's quantiles within two of its buckets
/// (`validation/scripts/aggregate_sizing.py`).
#[test]
fn grid_sizing_matches_aggregate() {
    use prospicio_aggregate::{Sizing, fft_auto, round_bucket};
    use prospicio_prob::count_families::{MixedPoisson, Mixing};
    use prospicio_prob::{
        Burr, Counting, Dist, Distribution, Gamma, Lognormal, Poisson, SeverityDist, Weibull,
    };

    // One FFT per book, shared by its rows.
    let books = std::cell::RefCell::new(std::collections::HashMap::new());
    let cases = prospicio_validation::reference("sizing_aggregate.csv");
    prospicio_validation::check(&cases, |c| {
        if c.get("distribution") == "round_bucket" {
            return round_bucket(c.number("arg")?).ok();
        }
        let v = |k: &str| c.param("params", k);
        let params = c.get("params");
        if !books.borrow().contains_key(params) {
            let n: Box<dyn Counting> = if params.contains("mixing_cv") {
                Box::new(
                    MixedPoisson::new(v("lambda"), Mixing::Gamma { cv: v("mixing_cv") }, 0.0)
                        .ok()?,
                )
            } else {
                Box::new(Poisson::new(v("lambda")).ok()?)
            };
            let sev: Dist = if params.contains("severity=gamma") {
                Gamma::from_mean_cv(v("mean"), v("cv")).ok()?.into()
            } else if params.contains("severity=lognormal") {
                Lognormal::from_mean_cv(v("mean"), v("cv")).ok()?.into()
            } else if params.contains("severity=weibull") {
                Weibull::new(v("shape"), v("scale")).ok()?.into()
            } else {
                Burr::new(v("alpha"), 1.0, v("scale")).ok()?.into()
            };
            let sev = SeverityDist::try_from(sev).ok()?;
            let (agg, report, size) = fft_auto(n.as_ref(), &sev, &Sizing::default()).ok()?;
            assert!(report.aliasing_error < 1e-12, "{params}");
            assert!(size.tail_estimate < 1e-5, "{params}");
            // Never coarser than aggregate's bucket on these books.
            assert!(size.step <= v("agg_bs"), "{params}: {}", size.step);
            books.borrow_mut().insert(params.to_string(), agg);
        }
        let books = books.borrow();
        let agg = &books[params];
        match c.get("quantity") {
            "mean" => Some(agg.mean()),
            "quantile" => agg.quantile(c.number("arg")?).ok(),
            _ => None,
        }
    });
}

/// Swing and retro premiums, sliding-scale and profit commissions against
/// `aggregate` 1.0.1's `contract_terms` at losses through every kink
/// (`validation/scripts/aggregate_contract_terms.py`).
#[test]
fn contract_terms_match_aggregate() {
    use prospicio_aggregate::{Layer, LossSensitivePremium};

    let cases = prospicio_validation::reference("contract_terms_aggregate.csv");
    prospicio_validation::check(&cases, |c| {
        let v = |k: &str| c.param("params", k);
        let has = |k: &str| c.get("params").contains(&format!("{k}="));
        let x = c.number("arg")?;
        let collar = || {
            LossSensitivePremium::new(
                v("basic"),
                v("lcm"),
                has("minimum").then(|| v("minimum")),
                has("maximum").then(|| v("maximum")),
            )
            .ok()
        };
        // A layer whose premium is `premium` for the placed share.
        let priced = || {
            Layer::quota_share("L", 1.0)
                .ok()?
                .deposit_premium(v("premium"))
                .ok()
        };
        match c.get("quantity") {
            "swing_premium" => {
                let layer = Layer::xol("L", f64::INFINITY, 0.0)
                    .ok()?
                    .share(v("share"))
                    .ok()?
                    .swing_rated(collar()?)
                    .ok()?;
                Some(layer.premium_for(x))
            }
            "retro_premium" => Some(collar()?.premium(x)),
            "sliding_commission" => {
                let anchors = (0..v("n") as usize)
                    .map(|i| (v(&format!("c{i}")), v(&format!("lr{i}"))))
                    .collect();
                Some(
                    priced()?
                        .sliding_scale(anchors)
                        .ok()?
                        .ceding_commission_for(x),
                )
            }
            "profit_commission" => Some(
                priced()?
                    .profit_commission(v("share"), v("allowance"))
                    .ok()?
                    .profit_commission_for(x),
            ),
            _ => None,
        }
    });
}
