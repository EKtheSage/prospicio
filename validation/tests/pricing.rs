//! Pricing parity: Pareto layer rating and implied alphas against the R
//! package Pareto (reference values only; see `docs/design/pareto.md`).

use prospicio_pricing::layer::{
    XsLayer, alpha_between_frequencies, alpha_between_frequency_and_layer, alpha_between_layers,
    pareto_extrapolation,
};
use prospicio_validation::{check, reference};

#[test]
fn pareto_rating_matches_r() {
    let cases = reference("pareto_rating_r.csv");
    check(&cases, |c| {
        let fields: std::collections::HashMap<&str, &str> = c
            .get("params")
            .split(';')
            .filter_map(|kv| kv.split_once('='))
            .collect();
        let num = |k: &str| -> Option<f64> {
            match *fields.get(k)? {
                "Inf" => Some(f64::INFINITY),
                v => v.parse().ok(),
            }
        };
        let truncation = num("truncation");
        let layer = |c: &str, a: &str| XsLayer::new(num(c)?, num(a)?).ok();
        match c.get("quantity") {
            "extrapolation" => pareto_extrapolation(
                layer("c1", "a1")?,
                layer("c2", "a2")?,
                num("alpha")?,
                truncation,
            )
            .ok(),
            "alpha_between_layers" => alpha_between_layers(
                (layer("c1", "a1")?, num("e1")?),
                (layer("c2", "a2")?, num("e2")?),
                truncation,
            )
            .ok(),
            "alpha_between_frequency_and_layer" => alpha_between_frequency_and_layer(
                num("threshold")?,
                num("frequency")?,
                layer("c", "a")?,
                num("e")?,
                truncation,
            )
            .ok(),
            "alpha_between_frequencies" => alpha_between_frequencies(
                num("t1")?,
                num("f1")?,
                num("t2")?,
                num("f2")?,
                truncation,
            )
            .ok(),
            _ => None,
        }
    });
}

#[test]
fn tower_matching_matches_r() {
    use prospicio_pricing::tower::{SelectionRule, match_tower};
    let cases = reference("tower_matching_r.csv");
    check(&cases, |c| {
        let fields: std::collections::HashMap<&str, &str> = c
            .get("params")
            .split(';')
            .filter_map(|kv| kv.split_once('='))
            .collect();
        let list = |k: &str| -> Option<Vec<Option<f64>>> {
            fields
                .get(k)?
                .split('|')
                .map(|v| match v {
                    "NA" => Some(None),
                    v => v.parse().ok().map(Some),
                })
                .collect()
        };
        let attachments: Vec<f64> = list("a")?.into_iter().collect::<Option<_>>()?;
        let losses: Vec<f64> = list("e")?.into_iter().collect::<Option<_>>()?;
        let model = match_tower(
            &attachments,
            &losses,
            &list("f")?,
            SelectionRule::MinimizeAlphaRatio,
        )
        .ok()?;
        let j = c.number("index")? as usize;
        match c.get("quantity") {
            "frequency" => Some(model.frequency),
            "threshold" => model.severity.thresholds().get(j).copied(),
            "alpha" => model.severity.alphas().get(j).copied(),
            _ => None,
        }
    });
}

#[test]
fn pml_curve_fit_matches_r() {
    use prospicio_pricing::tower::fit_pml_curve;
    let cases = reference("pml_curve_r.csv");
    check(&cases, |c| {
        let fields: std::collections::HashMap<&str, &str> = c
            .get("params")
            .split(';')
            .filter_map(|kv| kv.split_once('='))
            .collect();
        let list = |k: &str| -> Option<Vec<f64>> {
            fields.get(k)?.split('|').map(|v| v.parse().ok()).collect()
        };
        let truncation = fields.get("truncation").and_then(|v| v.parse().ok());
        let model = fit_pml_curve(
            &list("rp")?,
            &list("x")?,
            fields.get("tail")?.parse().ok()?,
            truncation,
        )
        .ok()?;
        let j = c.number("index")? as usize;
        match c.get("quantity") {
            "frequency" => Some(model.frequency),
            "threshold" => model.severity.thresholds().get(j).copied(),
            "alpha" => model.severity.alphas().get(j).copied(),
            _ => None,
        }
    });
}

#[test]
fn simulated_layers_are_priced_and_allocated() {
    use prospicio_aggregate::{CollectiveModel, Layer, Tower, simulate_events};
    use prospicio_pricing::risk_load::{PremiumRule, price_portfolio};
    use prospicio_prob::{Distortion, KeyValue, Lognormal, Poisson, PredictiveDistribution};

    // A per-risk programme of two XOL layers on 10,000 simulated years.
    let n = 10_000;
    let freq = Poisson::new(6.0).unwrap();
    let sev = Lognormal::from_mean_cv(1_000.0, 2.0).unwrap();
    let events = simulate_events(&freq, &sev, n, 21).unwrap();
    let layers = [("1x1", 1_000.0, 1_000.0), ("8x2", 8_000.0, 2_000.0)];
    let result = Tower::new(
        layers
            .iter()
            .map(|&(name, l, a)| Layer::xol(name, l, a).unwrap())
            .collect(),
    )
    .unwrap()
    .apply(&events)
    .unwrap();

    // Keep only the ceded losses, one component per layer.
    let ceded: Vec<Vec<f64>> = layers
        .iter()
        .map(|&(name, _, _)| {
            let key = vec![KeyValue::from("ceded"), KeyValue::from(name)];
            result.marginal(&key).unwrap().into_draws()
        })
        .collect();
    let draws: Vec<f64> = (0..n)
        .flat_map(|i| ceded.iter().map(move |c| c[i]))
        .collect();
    let pd = PredictiveDistribution::from_draws(
        vec!["layer".into()],
        layers
            .iter()
            .map(|&(name, _, _)| vec![KeyValue::from(name)])
            .collect(),
        draws,
        result.provenance().clone(),
    )
    .unwrap();

    let rule = PremiumRule::cost_of_capital(0.10).unwrap();
    let p = price_portfolio(&pd, &rule, &Distortion::tvar(0.99).unwrap()).unwrap();

    // Expected ceded losses agree with the collective model in closed form.
    let model = CollectiveModel::new(freq, sev);
    for (price, &(_, l, a)) in p.allocated.iter().zip(&layers) {
        let mean = model.layer_mean(l, a);
        let se = (model.layer_variance(l, a) / n as f64).sqrt();
        assert!((price.expected_loss - mean).abs() < 4.0 * se, "{l} xs {a}");
    }
    // Allocated prices add up to the programme's price.
    let premium: f64 = p.allocated.iter().map(|c| c.premium).sum();
    assert!((premium - p.total.premium).abs() < 1e-9 * p.total.premium);
    // Every allocated layer earns the cost of capital on its capital.
    for c in &p.allocated {
        assert!((c.return_on_capital() - 0.10).abs() < 1e-9);
    }
    // The high layer carries more capital per unit of loss, so it prices
    // at a lower loss ratio; writing both saves premium.
    assert!(p.allocated[1].loss_ratio() < p.allocated[0].loss_ratio());
    assert!(p.diversification() > 0.0);
    assert!(p.total.expected_loss < p.total.premium && p.total.premium < p.total.assets);
}

#[test]
fn mbbefd_exposure_curves_match_mpmath() {
    use prospicio_pricing::exposure::{ExposureCurve, Mbbefd};
    let cases = reference("mbbefd_mpmath.csv");
    check(&cases, |c| {
        let curve = c.get("curve");
        let m = if let Some(v) = curve.strip_prefix("c=") {
            Mbbefd::swiss_re(v.parse().unwrap()).unwrap()
        } else {
            Mbbefd::new(c.param("curve", "b"), c.param("curve", "g")).unwrap()
        };
        match c.get("quantity") {
            "mean" => Some(m.mean()),
            "G" => Some(m.g(c.number("x")?)),
            _ => None,
        }
    });
}

#[test]
fn natural_allocation_matches_aggregate() {
    use prospicio_pricing::natural::{Allocation, Portfolio};
    use prospicio_prob::Distortion;
    // Monograph 15's InsCo (validation/scripts/aggregate_natural.py).
    let rows: Vec<Vec<f64>> = [
        [15.0, 7.0, 0.0],
        [15.0, 13.0, 0.0],
        [5.0, 20.0, 11.0],
        [7.0, 33.0, 0.0],
        [13.0, 20.0, 7.0],
        [5.0, 27.0, 8.0],
        [15.0, 16.0, 9.0],
        [26.0, 19.0, 10.0],
        [17.0, 8.0, 40.0],
        [16.0, 20.0, 64.0],
    ]
    .iter()
    .map(|r| r.to_vec())
    .collect();
    let units = vec!["X1".to_string(), "X2".into(), "X3".into()];
    let insco = Portfolio::from_rows(units, &rows, None).unwrap();
    // Pricing Insurance Risk's Discrete case: two independent units.
    let grid = |atoms: &[(usize, f64)]| {
        let mut p = vec![0.0; atoms.iter().map(|a| a.0).max().unwrap() + 1];
        atoms.iter().for_each(|&(x, q)| p[x] = q);
        prospicio_prob::Grid::new(1.0, p).unwrap()
    };
    let discrete = Portfolio::from_independent(
        vec!["X1".into(), "X2".into()],
        &[
            grid(&[(0, 0.5), (8, 0.25), (10, 0.25)]),
            grid(&[(0, 0.5), (1, 0.25), (90, 0.25)]),
        ],
    )
    .unwrap();
    let cases = reference("natural_aggregate.csv");
    check(&cases, |c| {
        let params: std::collections::HashMap<&str, &str> = c
            .get("params")
            .split(';')
            .filter_map(|kv| kv.split_once('='))
            .collect();
        let assets: f64 = params.get("assets")?.parse().ok()?;
        let port = match *params.get("case")? {
            "insco" => &insco,
            "discrete" => &discrete,
            _ => return None,
        };
        match c.get("distribution") {
            "price" => {
                let param: f64 = params.get("param")?.parse().ok()?;
                let g = match *params.get("family")? {
                    "ccoc" => Distortion::ccoc(param),
                    "ph" => Distortion::proportional_hazard(param),
                    "wang" => Distortion::wang(param),
                    "dual" => Distortion::dual_power(param),
                    "tvar" => Distortion::tvar(param),
                    _ => return None,
                }
                .ok()?;
                let method = match *params.get("allocation")? {
                    "linear" => Allocation::Linear,
                    "lifted" => Allocation::Lifted,
                    _ => return None,
                };
                let price = port.price(&g, assets, method).ok()?;
                let (unit, col) = c.get("quantity").split_once('.')?;
                let p = match unit {
                    "total" => price.total,
                    u => price.allocated[price.units.iter().position(|n| n == u)?],
                };
                Some(match col {
                    "L" => p.loss,
                    "M" => p.margin,
                    "P" => p.premium,
                    "Q" => p.capital,
                    "a" => p.assets,
                    _ => return None,
                })
            }
            "bodoff" => {
                let i = port.units().iter().position(|u| u == c.get("quantity"))?;
                Some(port.bodoff(assets)[i])
            }
            "bounds" => {
                let premium: f64 = params.get("premium")?.parse().ok()?;
                let (unit, side) = c.get("quantity").split_once('.')?;
                let i = port.units().iter().position(|u| u == unit)?;
                let b = &port.premium_bounds(premium, assets).ok()?[i];
                Some(if side == "lower" { b.lower } else { b.upper })
            }
            "classical" => {
                use prospicio_pricing::classical::{Kind, Principle, calibrate};
                let premium: f64 = params.get("premium")?.parse().ok()?;
                let kind = match c.get("quantity") {
                    "Expected Value" => Kind::ExpectedValue,
                    "VaR" => Kind::Var,
                    "Variance" => Kind::Variance,
                    "Standard Deviation" => Kind::StandardDeviation,
                    "Semi-Variance" => Kind::SemiVariance,
                    "Exponential" => Kind::Exponential,
                    "Esscher" => Kind::Esscher,
                    "Dutch" => Kind::Dutch,
                    "Fischer" => Kind::Fischer { q: 2.0 },
                    _ => return None,
                };
                Some(
                    match calibrate(kind, port.totals(), port.probs(), premium).ok()? {
                        Principle::ExpectedValue(t)
                        | Principle::Variance(t)
                        | Principle::StandardDeviation(t)
                        | Principle::SemiVariance(t)
                        | Principle::Exponential(t)
                        | Principle::Esscher(t)
                        | Principle::Dutch(t)
                        | Principle::Var(t) => t,
                        Principle::Fischer { theta, .. } => theta,
                    },
                )
            }
            _ => None,
        }
    });
}
