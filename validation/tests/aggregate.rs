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

/// The occurrence PML against `aggregate` 1.0.1's `oep` on three Poisson
/// books: the loss, the severity's exceedance and the occurrence return
/// period at eight annual probabilities, at `1e-10`
/// (`validation/scripts/aggregate_oep.py`).
#[test]
fn occurrence_pml_matches_aggregate() {
    use prospicio_aggregate::pml::occurrence_pml;
    use prospicio_prob::{Dist, Gamma, Lognormal, Poisson, Weibull};

    let cases = prospicio_validation::reference("oep_aggregate.csv");
    prospicio_validation::check(&cases, |c| {
        let v = |k: &str| c.param("params", k);
        let params = c.get("params");
        let n = Poisson::new(v("lambda")).ok()?;
        let sev: Dist = if params.contains("severity=gamma") {
            Gamma::from_mean_cv(v("mean"), v("cv")).ok()?.into()
        } else if params.contains("severity=lognormal") {
            Lognormal::from_mean_cv(v("mean"), v("cv")).ok()?.into()
        } else {
            Weibull::new(v("shape"), v("scale")).ok()?.into()
        };
        let pml = occurrence_pml(&n, &sev, c.number("arg")?).ok()?;
        match c.get("quantity") {
            "loss" => Some(pml.loss),
            "severity_survival" => Some(pml.severity_survival),
            "occurrence_return_period" => Some(pml.occurrence_return_period),
            _ => None,
        }
    });
}

/// The simulated OEP, each year's largest event, against the exact curve
/// for a negative binomial count (which `aggregate`'s `oep` does not
/// cover): 200,000 years, within four standard errors at five losses.
#[test]
fn simulated_maxima_match_the_occurrence_exceedance() {
    use prospicio_aggregate::pml::{occurrence_exceedance, occurrence_pml};
    use prospicio_aggregate::simulate_events;
    use prospicio_prob::{Distribution, Lognormal, NegativeBinomial};

    let n = NegativeBinomial::new(1.5, 2.0).unwrap();
    let x = Lognormal::from_mean_cv(100.0, 2.0).unwrap();
    let sims = 200_000;
    let maxima = simulate_events(&n, &x, sims, 20_261_012)
        .unwrap()
        .maxima()
        .unwrap();
    let simulated = maxima.total();
    for p in [0.3, 0.1, 0.03, 0.01, 0.003] {
        let loss = occurrence_pml(&n, &x, p).unwrap().loss;
        let exact = occurrence_exceedance(&n, &x, loss);
        assert!((exact / p - 1.0).abs() < 1e-9);
        let se = (p * (1.0 - p) / sims as f64).sqrt();
        let observed = 1.0 - simulated.cdf(loss);
        assert!((observed - p).abs() < 4.0 * se, "p {p}: {observed} ({se})");
    }
}

/// Compound mean, cv and skewness against `aggregate` 1.0.1's analytic
/// statistics, and the five moment approximations against its own fitting
/// code on those moments, on five books (`validation/scripts/aggregate_moments.py`).
#[test]
fn compound_moments_and_approximations_match_aggregate() {
    use prospicio_aggregate::{Approximation, ApproximationKind, compound_moments};
    use prospicio_prob::{
        Dist, Gamma, Lognormal, NegativeBinomial, Poisson, SeverityDist, Weibull,
    };

    let cases = prospicio_validation::reference("moments_aggregate.csv");
    prospicio_validation::check(&cases, |c| {
        let v = |k: &str| c.param("params", k);
        let params = c.get("params");
        let sev: Dist = if params.contains("severity=gamma") {
            Gamma::from_mean_cv(v("mean"), v("cv")).ok()?.into()
        } else if params.contains("severity=lognormal") {
            Lognormal::from_mean_cv(v("mean"), v("cv")).ok()?.into()
        } else {
            Weibull::new(v("shape"), v("scale")).ok()?.into()
        };
        let sev = SeverityDist::try_from(sev).ok()?;
        // aggregate's `mixed gamma c` count is the negative binomial with
        // r = 1/c² and β = λc².
        let m = if params.contains("mixing_cv") {
            let c2 = v("mixing_cv").powi(2);
            compound_moments(
                &NegativeBinomial::new(1.0 / c2, v("lambda") * c2).ok()?,
                &sev,
            )
        } else {
            compound_moments(&Poisson::new(v("lambda")).ok()?, &sev)
        }
        .ok()?;
        let q = c.get("quantity");
        match q {
            "mean" => return Some(m.mean),
            "cv" => return Some(m.cv()),
            "skew" => return Some(m.skew()),
            _ => {}
        }
        let (fit, key) = q.split_once('_')?;
        let kind = match fit {
            "norm" => ApproximationKind::Normal,
            "gamma" => ApproximationKind::Gamma,
            "lognorm" => ApproximationKind::Lognormal,
            "sgamma" => ApproximationKind::ShiftedGamma,
            _ => ApproximationKind::ShiftedLognormal,
        };
        // scipy's (shape, loc, scale) for each family.
        let (shape, loc, scale) = match m.approximate(kind).ok()? {
            Approximation::Normal { mean, sd } => (f64::NAN, mean, sd),
            Approximation::Gamma { shape, scale } => (shape, 0.0, scale),
            Approximation::Lognormal { meanlog, sdlog } => (sdlog, 0.0, meanlog.exp()),
            Approximation::ShiftedGamma {
                shape,
                scale,
                shift,
            } => (shape, shift, scale),
            Approximation::ShiftedLognormal {
                meanlog,
                sdlog,
                shift,
            } => (sdlog, shift, meanlog.exp()),
        };
        match key {
            "shape" => Some(shape),
            "loc" => Some(loc),
            _ => Some(scale),
        }
    });
}

/// Reinsurance on the grid against `aggregate`'s `reins_stats_df`,
/// `reins_price_df` and `reins_density_df`
/// (`validation/scripts/aggregate_reinsurance.py`): the occurrence layers'
/// statistics, each view priced by a proportional hazard distortion, and
/// an inuring tower's densities point by point.
#[test]
fn reinsurance_statistics_and_prices_match_aggregate() {
    use prospicio_aggregate::{
        CompoundMoments, Layer, Tower, aggregate_layer_statistics, compound_moments,
        occurrence_cession_statistics, occurrence_layer_statistics,
    };
    use prospicio_prob::{Distortion, Distribution, Gamma, Lognormal, Severity};

    let cases = reference("reinsurance_aggregate.csv");
    // One tower's grids per book.
    let towers = std::cell::RefCell::new(std::collections::HashMap::new());
    check(&cases, |c| {
        let v = |k: &str| c.param("params", k);
        let params = c.get("params");
        let text = |k: &str| {
            params
                .split(';')
                .find_map(|p| p.strip_prefix(k)?.strip_prefix('='))
        };
        let n: Box<dyn Counting> = if params.contains("mixing_cv") {
            let c2 = v("mixing_cv").powi(2);
            Box::new(NegativeBinomial::new(1.0 / c2, v("lambda") * c2).ok()?)
        } else {
            Box::new(Poisson::new(v("lambda")).ok()?)
        };
        // `aggregate` rounds the severity limited to `cap` onto the grid.
        let points = v("cap") as usize + 1;
        let (sev, _) = if params.contains("severity=gamma") {
            Grid::rounding(&Gamma::from_mean_cv(v("mean"), v("cv")).ok()?, 1.0, points)
        } else {
            Grid::rounding(
                &Lognormal::from_mean_cv(v("mean"), v("cv")).ok()?,
                1.0,
                points,
            )
        }
        .ok()?;
        // `share:limit:attachment`, `+` between layers.
        let occ: Vec<[f64; 3]> = text("occ")?
            .split('+')
            .map(|l| {
                let f: Vec<f64> = l.split(':').map(|x| x.parse().unwrap()).collect();
                [f[0], f[1], f[2]]
            })
            .collect();
        let pick = |m: &CompoundMoments, s: &str| match s {
            "mean" => Some(m.mean),
            "cv" => Some(m.cv()),
            "skew" => Some(m.skew()),
            _ => None,
        };

        let q = c.get("quantity");
        if q == "reinstatement_premium" {
            // The deposit is the rate on line times the limit; each
            // reinstatement is charged at its rate times the deposit.
            let [_, limit, attach] = occ[0];
            let rates: Vec<f64> = text("rates")?
                .split('+')
                .map(|r| r.parse().unwrap())
                .collect();
            let layer = Layer::xol("ri", limit, attach)
                .ok()?
                .paid_reinstatements(v("rol") * limit, rates)
                .ok()?;
            let grids = Tower::new(vec![layer])
                .ok()?
                .on_grid(&*n, &sev, 1 << 14)
                .ok()?;
            return Some(grids.expected_reinstatement_premium[0]);
        }
        if let Some((col, stat)) = q.split_once(':') {
            if let Some(i) = col.strip_prefix("layer.") {
                let [share, limit, attach] = occ[i.parse::<usize>().ok()? - 1];
                let s = occurrence_layer_statistics(&*n, &sev, share, limit, attach).ok()?;
                return match stat.split_once('_') {
                    Some(("pr", "attach")) => Some(s.pr_attach),
                    Some(("pr", "detach")) => Some(s.pr_detach),
                    Some(("pr", "loss")) => Some(s.pr_loss),
                    Some(("freq", m)) => pick(&s.count, m),
                    Some(("sev", m)) => pick(&s.severity, m),
                    Some(("agg", m)) => pick(&s.aggregate, m),
                    _ if stat == "lol" => Some(s.loss_on_line),
                    _ => None,
                };
            }
            if col == "Ceded" {
                let layers: Vec<(f64, f64, f64)> = occ.iter().map(|l| (l[0], l[1], l[2])).collect();
                let s = occurrence_cession_statistics(&*n, &sev, &layers).ok()?;
                return match stat.split_once('_') {
                    Some(("pr", "attach")) => Some(s.pr_attach),
                    Some(("pr", "loss")) => Some(s.pr_loss),
                    Some(("sev", m)) => pick(&s.severity, m),
                    Some(("agg", m)) => pick(&s.aggregate, m),
                    _ if stat == "lol" => Some(s.loss_on_line),
                    _ => None,
                };
            }
            if col == "agglayer" {
                // The cover sees the annual loss net of the occurrence layers.
                let (limit, retention) = text("agg")?.split_once(':')?;
                let layers = occ
                    .iter()
                    .enumerate()
                    .map(|(i, &[share, limit, attach])| {
                        Layer::xol(format!("occ{i}"), limit, attach).and_then(|l| l.share(share))
                    })
                    .collect::<prospicio_core::Result<Vec<_>>>()
                    .ok()?;
                let subject = Tower::new(layers)
                    .ok()?
                    .on_grid(&*n, &sev, 1 << 14)
                    .ok()?
                    .net?;
                let s = aggregate_layer_statistics(
                    &subject,
                    1.0,
                    limit.parse().ok()?,
                    retention.parse().ok()?,
                )
                .ok()?;
                return match stat.split_once('_') {
                    Some(("pr", "attach")) => Some(s.pr_attach),
                    Some(("pr", "detach")) => Some(s.pr_detach),
                    Some(("agg", m)) => pick(&s.aggregate, m),
                    _ if stat == "lol" => Some(s.loss_on_line),
                    _ => None,
                };
            }
            if col == "gross" {
                let (block, m) = stat.split_once('_')?;
                let moments = if block == "agg" {
                    compound_moments(&*n, &sev).ok()?
                } else {
                    let (m1, var) = (sev.mean(), sev.variance());
                    let m3 = sev.third_moment()?;
                    CompoundMoments {
                        mean: m1,
                        variance: var,
                        third_cumulant: m3 - 3.0 * m1 * (var + m1 * m1) + 2.0 * m1.powi(3),
                    }
                };
                return pick(&moments, m);
            }
        }

        let mut stages = vec![
            occ.iter()
                .enumerate()
                .map(|(i, &[share, limit, attach])| {
                    Layer::xol(format!("occ{i}"), limit, attach)
                        .and_then(|l| l.share(share))
                        .unwrap()
                })
                .collect::<Vec<_>>(),
        ];
        if let Some(agg) = text("agg") {
            let (limit, retention) = agg.split_once(':')?;
            stages.push(vec![
                Layer::stop_loss("agg", limit.parse().ok()?, retention.parse().ok()?).ok()?,
            ]);
        }
        let mut towers = towers.borrow_mut();
        if !towers.contains_key(params) {
            let grids = Tower::inuring(stages)
                .ok()?
                .on_grid(&*n, &sev, 1 << 14)
                .ok()?;
            towers.insert(params.to_string(), grids);
        }
        let grids = &towers[params];
        // The occurrence stage's total cession, and the aggregate cover.
        let view = |name: &str| match name {
            "gross" => Some(&grids.gross),
            "ceded_occ" | "stage0" => grids.stage_ceded[0].as_ref(),
            "ceded" => grids.ceded.last(),
            "net" => grids.net.as_ref(),
            _ => None,
        };
        if let Some(rest) = q.strip_prefix("price:") {
            let (name, stat) = rest.split_once(':')?;
            let price = view(name)?.distortion(&Distortion::proportional_hazard(0.7).ok()?);
            return match stat {
                "L" => Some(view(name)?.mean()),
                "P" => Some(price),
                _ => None,
            };
        }
        let name = q.strip_prefix("density_")?;
        let x = c.number("arg")? as usize;
        Some(view(name)?.probs().get(x).copied().unwrap_or(0.0))
    });
}

#[test]
fn eventual_ruin_matches_aggregate() {
    use prospicio_aggregate::ruin::{SeverityPart, pollaczeck_khinchine};
    use prospicio_prob::Gamma;
    // validation/scripts/aggregate_ruin.py: a gamma severity on 2^16 points
    // and claims of 1, 2 or 3 on 2^10, both on a step of 1.
    let (gamma, _) =
        Grid::rounding(&Gamma::from_mean_cv(50.0, 1.5).unwrap(), 1.0, 1 << 16).unwrap();
    let mut small = vec![0.0; 1 << 10];
    small[1..4].copy_from_slice(&[0.5, 0.3, 0.2]);
    let small = Grid::new(1.0, small).unwrap();
    let cases = reference("ruin_aggregate.csv");
    check(&cases, |c| {
        let params: std::collections::HashMap<&str, f64> = c
            .get("params")
            .split(';')
            .filter_map(|kv| kv.split_once('='))
            .filter_map(|(k, v)| Some((k, v.parse().ok()?)))
            .collect();
        let sev = match c.get("distribution") {
            "G" => &gamma,
            "D" => &small,
            _ => return None,
        };
        let part = if let Some(&v) = params.get("cap") {
            SeverityPart::AtMost(v)
        } else if let Some(&v) = params.get("excess") {
            SeverityPart::Above(v)
        } else if let Some(&v) = params.get("stop_loss") {
            SeverityPart::StopLoss(v)
        } else {
            SeverityPart::All
        };
        let ruin = pollaczeck_khinchine(sev, *params.get("rho")?, sev.len(), part).ok()?;
        let arg = c.number("arg")?;
        match c.get("quantity") {
            "mean" => Some(ruin.severity_mean()),
            "psi" => Some(ruin.psi_at(arg)),
            "u_index" => ruin.capital(arg),
            "u_interp" => ruin.capital_interpolated(arg),
            "lundberg" => ruin.lundberg_exponent(),
            _ => None,
        }
    });
}

#[test]
fn layer_loss_picks_match_aggregate() {
    use prospicio_aggregate::picks::layer_loss_picks;
    use prospicio_prob::Gamma;
    // validation/scripts/aggregate_picks.py: a gamma severity on 2^14
    // points, 10 claims, layers to 100, 500 and 1,000.
    let (sev, _) = Grid::rounding(&Gamma::from_mean_cv(100.0, 1.5).unwrap(), 1.0, 1 << 14).unwrap();
    let attachments = [100.0, 500.0, 1000.0];
    let cases = reference("picks_aggregate.csv");
    check(&cases, |c| {
        let picks: Vec<f64> = c
            .get("params")
            .strip_prefix("picks=")?
            .split(':')
            .map(|p| p.parse().ok())
            .collect::<Option<_>>()?;
        let picked = layer_loss_picks(&sev, &attachments, &picks, 10.0).ok()?;
        assert_eq!(picked.straight_bottom, c.get("distribution") == "straight");
        let arg = c.number("arg")?;
        match c.get("quantity") {
            "p" => picked.severity.probs().get(arg as usize).copied(),
            "layer" => {
                let g = &picked.severity;
                let i = attachments.iter().position(|&a| a == arg)?;
                let bottom = if i == 0 { 0.0 } else { attachments[i - 1] };
                let lev = |a: f64| {
                    (0..g.len())
                        .map(|j| g.x(j).min(a) * g.probs()[j])
                        .sum::<f64>()
                };
                Some(10.0 * (lev(arg) - lev(bottom)))
            }
            _ => None,
        }
    });
    // The straight-line bottom layer meets its pick, where aggregate's
    // rounded line reaches 704.90.
    let picked = layer_loss_picks(&sev, &attachments, &[700.0, 250.0, 50.0], 10.0).unwrap();
    let g = &picked.severity;
    let lev100: f64 = (0..g.len()).map(|j| g.x(j).min(100.0) * g.probs()[j]).sum();
    assert!((10.0 * lev100 - 700.0).abs() < 1e-9);
}

// An event loss table's simulated years against its analytic occurrence
// exceedance curve, `1 − exp(−Σ rate_i P(L_i > x))`, within four binomial
// standard errors, and its counts and mean loss within four standard errors.
mod event_loss_table {
    use prospicio_aggregate::cat::EventLossTable;

    const YEARS: usize = 200_000;

    fn year_max(events: &prospicio_aggregate::EventSet, y: usize) -> f64 {
        events.events(y).iter().copied().fold(0.0, f64::max)
    }

    fn check(elt: &EventLossTable, thresholds: &[f64], seed: u64) {
        let (events, ids) = elt.simulate(YEARS, seed).unwrap();
        let n = YEARS as f64;
        for &x in thresholds {
            let p = elt.oep(x);
            let hit = (0..YEARS).filter(|&y| year_max(&events, y) > x).count() as f64 / n;
            let se = (p * (1.0 - p) / n).sqrt();
            assert!(
                (hit - p).abs() < 4.0 * se,
                "x = {x}: {hit} vs {p} (se {se})"
            );
        }
        let counts: f64 = events.counts().iter().sum::<usize>() as f64 / n;
        let lambda = elt.rate();
        assert!(
            (counts - lambda).abs() < 4.0 * (lambda / n).sqrt(),
            "{counts} vs {lambda}"
        );
        assert_eq!(
            ids.iter().map(Vec::len).collect::<Vec<_>>(),
            events.counts()
        );
        let totals: Vec<f64> = (0..YEARS).map(|y| events.events(y).iter().sum()).collect();
        let mean = totals.iter().sum::<f64>() / n;
        let var = totals.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let eal = elt.expected_annual_loss();
        assert!(
            (mean - eal).abs() < 4.0 * (var / n).sqrt(),
            "{mean} vs {eal}"
        );
    }

    #[test]
    fn points_without_secondary_uncertainty() {
        let elt = EventLossTable::new(
            &[0.01, 0.04, 0.2, 0.5],
            &[10_000.0, 3000.0, 800.0, 100.0],
            &[0.0; 4],
            &[0.0; 4],
            &[10_000.0, 3000.0, 800.0, 100.0],
        )
        .unwrap();
        // Between the steps the curve is flat.
        check(&elt, &[50.0, 500.0, 2000.0, 5000.0], 11);
        assert!((elt.oep(5000.0) - (1.0 - (-0.01f64).exp())).abs() < 1e-15);
    }

    #[test]
    fn betas_at_return_periods() {
        let elt = EventLossTable::new(
            &[0.005, 0.02, 0.1, 0.4],
            &[20_000.0, 6000.0, 1500.0, 200.0],
            &[6000.0, 2000.0, 600.0, 100.0],
            &[2000.0, 1000.0, 200.0, 0.0],
            &[60_000.0, 25_000.0, 8000.0, 2000.0],
        )
        .unwrap();
        let at: Vec<f64> = [0.1, 0.02, 0.01, 0.004]
            .iter()
            .map(|&p| elt.occurrence_pml(p).unwrap())
            .collect();
        check(&elt, &at, 23);
    }
}

#[test]
fn clash_matches_aggregate() {
    use prospicio_aggregate::bivariate::{Clash, JointGrid};
    use prospicio_prob::{Distribution, Gamma, Lognormal};
    use std::collections::HashMap;
    // validation/scripts/aggregate_clash.py: A only 2, B only 3, both 1;
    // gamma and lognormal losses; a Poisson or negative binomial count.
    let clash = Clash::new(2.0, 3.0, 1.0).unwrap();
    let mut fits: HashMap<String, (JointGrid, Grid)> = HashMap::new();
    let cases = reference("clash_aggregate.csv");
    check(&cases, |c| {
        let key = format!("{};{}", c.get("distribution"), c.get("params"));
        if !fits.contains_key(&key) {
            let n: Box<dyn Counting> = match c.get("distribution") {
                "poisson" => Box::new(Poisson::new(clash.events()).ok()?),
                "mixed" => Box::new(NegativeBinomial::from_mean_variance(12.0, 48.0).ok()?),
                _ => return None,
            };
            let points = ["points_a", "points_b"].map(|p| c.param("params", p) as usize);
            let (a, _) = Grid::rounding(
                &Gamma::from_mean_cv(10.0, 1.5).unwrap(),
                c.param("params", "step_a"),
                points[0],
            )
            .ok()?;
            let (b, _) = Grid::rounding(
                &Lognormal::from_mean_cv(5.0, 0.5).unwrap(),
                c.param("params", "step_b"),
                points[1],
            )
            .ok()?;
            let (joint, _) = clash.aggregate(n.as_ref(), &a, &b, points).ok()?;
            let total = joint.total().ok()?;
            fits.insert(key.clone(), (joint, total));
        }
        let (joint, total) = &fits[&key];
        let at = |col: &str| c.number(col).map(|x| x as usize);
        match c.get("quantity") {
            "density" => Some(joint.prob(at("arg")?, at("arg2")?)),
            "marginal_a" => Some(joint.marginal(0).ok()?.probs()[at("arg")?]),
            "marginal_b" => Some(joint.marginal(1).ok()?.probs()[at("arg")?]),
            "total" => Some(total.probs()[at("arg")?]),
            "mean_a" => Some(joint.mean(0)),
            "mean_b" => Some(joint.mean(1)),
            "var_a" => Some(joint.variance(0)),
            "var_b" => Some(joint.variance(1)),
            "moment_ab" => Some(joint.moment(1, 1)),
            "correlation" => Some(joint.correlation()),
            "total_mean" => Some(total.mean()),
            "total_step" => Some(total.step()),
            _ => None,
        }
    });
}
