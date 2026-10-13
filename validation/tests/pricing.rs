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
    let discrete3 = Portfolio::from_independent(
        vec!["X1".into(), "X2".into(), "X3".into()],
        &[
            grid(&[(0, 0.5), (8, 0.25), (10, 0.25)]),
            grid(&[(0, 0.5), (1, 0.25), (90, 0.25)]),
            grid(&[(0, 0.4), (2, 0.3), (5, 0.2), (30, 0.1)]),
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
        let port = match *params.get("case")? {
            "insco" => &insco,
            "discrete" => &discrete,
            "discrete3" => &discrete3,
            _ => return None,
        };
        match c.get("distribution") {
            "tvar_threshold" => return port.tvar_threshold(c.number("arg")?).ok(),
            "stand_alone" => {
                let shape: f64 = params.get("shape")?.parse().ok()?;
                let g = match *params.get("distortion")? {
                    "ph" => Distortion::proportional_hazard(shape),
                    "wang" => Distortion::wang(shape),
                    _ => return None,
                }
                .ok()?;
                let s = port
                    .price_stand_alone(&g, params.get("p")?.parse().ok()?)
                    .ok()?;
                let (unit, stat) = c.get("quantity").split_once('.')?;
                let pent = match unit {
                    "total" => s.total,
                    "sum" => s.sum,
                    u => s.stand_alone[s.units.iter().position(|v| v == u)?],
                };
                return match stat {
                    "L" => Some(pent.loss),
                    "P" => Some(pent.premium),
                    "a" => Some(pent.assets),
                    _ => None,
                };
            }
            "comonotonic" => {
                let i = port.units().iter().position(|u| u == c.get("quantity"))?;
                let x = c.number("arg")?;
                let k = port.totals().iter().position(|&t| t == x)?;
                return Some(port.comonotonic_kappa()[i][k]);
            }
            _ => {}
        }
        let assets: f64 = params.get("assets")?.parse().ok()?;
        match c.get("distribution") {
            "priority" => {
                use prospicio_pricing::natural::Rule;
                let ladder = port.priority(assets).ok()?;
                let (unit, rule) = c.get("quantity").split_once('.')?;
                if unit == "total" {
                    return Some(ladder.total);
                }
                let i = ladder.units.iter().position(|u| u == unit)?;
                let rule = match rule {
                    "senior" => Rule::Senior,
                    "equal" => Rule::Equal,
                    "junior" => Rule::Junior,
                    _ => return None,
                };
                Some(ladder.recovery(rule)[i])
            }
            "priority_kappa" => {
                let (unit, side) = c.get("quantity").split_once('.')?;
                let i = port.units().iter().position(|u| u == unit)?;
                let x: f64 = c.get("arg").parse().ok()?;
                let k = port.totals().iter().position(|&t| t == x)?;
                let (junior, senior) = port.priority_kappa(i, assets).ok()?;
                Some(if side == "junior" {
                    junior[k]
                } else {
                    senior[k]
                })
            }
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

#[test]
fn priority_ladder_identities() {
    use prospicio_pricing::natural::{Portfolio, Rule};
    // Three independent units (the discrete3 case of
    // aggregate_natural.py), as grids and as their 36 joint scenarios.
    let atoms: [&[(usize, f64)]; 3] = [
        &[(0, 0.5), (8, 0.25), (10, 0.25)],
        &[(0, 0.5), (1, 0.25), (90, 0.25)],
        &[(0, 0.4), (2, 0.3), (5, 0.2), (30, 0.1)],
    ];
    let grids: Vec<_> = atoms
        .iter()
        .map(|a| {
            let mut p = vec![0.0; a.iter().map(|x| x.0).max().unwrap() + 1];
            a.iter().for_each(|&(x, q)| p[x] = q);
            prospicio_prob::Grid::new(1.0, p).unwrap()
        })
        .collect();
    let units: Vec<String> = vec!["X1".into(), "X2".into(), "X3".into()];
    let (mut rows, mut probs) = (Vec::new(), Vec::new());
    for &(x1, p1) in atoms[0] {
        for &(x2, p2) in atoms[1] {
            for &(x3, p3) in atoms[2] {
                rows.push(vec![x1 as f64, x2 as f64, x3 as f64]);
                probs.push(p1 * p2 * p3);
            }
        }
    }
    let independent = Portfolio::from_independent(units.clone(), &grids).unwrap();
    let scenarios = Portfolio::from_rows(units.clone(), &rows, Some(&probs)).unwrap();
    // Off the grid too: E[X ∧ t] is linear between grid points.
    for a in [0.0, 7.5, 12.0, 40.0, 97.25, 130.0] {
        let g = independent.priority(a).unwrap();
        let r = scenarios.priority(a).unwrap();
        for rule in [Rule::Senior, Rule::Equal, Rule::Junior] {
            for (x, y) in g.recovery(rule).iter().zip(r.recovery(rule)) {
                assert!((x - y).abs() < 1e-12, "a = {a}, {rule:?}: {x} vs {y}");
            }
        }
        for i in 0..3 {
            assert!(r.epd(Rule::Senior)[i] <= r.epd(Rule::Equal)[i] + 1e-15);
            assert!(r.epd(Rule::Equal)[i] <= r.epd(Rule::Junior)[i] + 1e-15);
        }
        let equal: f64 = r.equal.iter().sum();
        assert!((equal - r.total).abs() < 1e-12);
        // The conditional ladder integrates to the junior shortfall.
        for i in 0..3 {
            for port in [&independent, &scenarios] {
                let (junior, senior) = port.priority_kappa(i, a).unwrap();
                let lost: f64 = junior.iter().zip(port.probs()).map(|(j, p)| j * p).sum();
                assert!((lost - r.shortfall(Rule::Junior)[i]).abs() < 1e-12);
                for ((j, s), x) in junior.iter().zip(&senior).zip(port.totals()) {
                    assert!((j + s - (x - a).max(0.0)).abs() < 1e-12);
                }
            }
        }
    }
    // Two units, dependent: senior one plus junior other is the pot.
    let rows = [
        [15.0, 7.0],
        [5.0, 20.0],
        [26.0, 19.0],
        [17.0, 48.0],
        [16.0, 84.0],
    ];
    let rows: Vec<Vec<f64>> = rows.iter().map(|r| r.to_vec()).collect();
    let two = Portfolio::from_rows(vec!["A".into(), "B".into()], &rows, None).unwrap();
    for a in [20.0, 45.0, 64.5, 100.0] {
        let l = two.priority(a).unwrap();
        assert!((l.senior[0] + l.junior[1] - l.total).abs() < 1e-12);
        assert!((l.senior[1] + l.junior[0] - l.total).abs() < 1e-12);
    }
    assert!(two.priority(-1.0).is_err());
    assert!(two.priority_kappa(2, 10.0).is_err());
}

/// Classical credibility against R's `actuar` 3.3.7
/// (`validation/scripts/r_credibility.R`): the Bühlmann–Straub model on
/// Hachemeister's data with the unbiased and the iterative estimator,
/// Bühlmann's model (equal weights), a fit with missing cells, and the
/// limited fluctuation standards.
#[test]
fn credibility_matches_actuar() {
    use prospicio_pricing::credibility::{BuhlmannStraub, LimitedFluctuation, StructureEstimator};

    let data = reference("hachemeister.csv");
    let mut ratios = vec![vec![0.0; 12]; 5];
    let mut weights = vec![vec![0.0; 12]; 5];
    for row in &data {
        let i = row.number("state").unwrap() as usize - 1;
        let j = row.number("quarter").unwrap() as usize - 1;
        ratios[i][j] = row.number("ratio").unwrap();
        weights[i][j] = row.number("weight").unwrap();
    }
    let cases = reference("credibility_actuar.csv");
    check(&cases, |c| {
        if c.get("distribution") == "limited_fluctuation" {
            let lf =
                LimitedFluctuation::new(c.param("params", "p"), c.param("params", "k")).ok()?;
            return Some(lf.frequency_standard());
        }
        let params = c.get("params");
        let (mut r, mut w) = (ratios.clone(), weights.clone());
        if let Some(cells) = params.split(';').find_map(|kv| kv.strip_prefix("missing=")) {
            for cell in cells.split('|') {
                let (i, j) = cell.split_once('.')?;
                let (i, j) = (i.parse::<usize>().ok()? - 1, j.parse::<usize>().ok()? - 1);
                r[i][j] = f64::NAN;
                w[i][j] = f64::NAN;
            }
        }
        let estimator = if params.contains("estimator=iterative") {
            StructureEstimator::iterative()
        } else {
            StructureEstimator::Unbiased
        };
        let weights = params.contains("weights=yes").then_some(w.as_slice());
        let fit = BuhlmannStraub::new(estimator).fit(&r, weights).ok()?;
        let group = || Some(c.number("arg")? as usize - 1);
        match c.get("quantity") {
            "collective_mean" => Some(fit.collective_mean),
            "between_variance" => Some(fit.between_variance),
            "within_variance" => Some(fit.within_variance),
            "credibility" => Some(fit.credibility[group()?]),
            "premium" => Some(fit.premiums[group()?]),
            _ => None,
        }
    });
}

// Ratemaking against Werner and Modlin, *Basic Ratemaking* (5th ed., CAS,
// 2016). The book's exhibits round at each step; each test checks the exact
// value where the book's arithmetic is exact, and reproduces a printed
// figure by rounding where the book does (`knowledge/references/
// werner-modlin-ratemaking.md`).

mod ratemaking {
    use prospicio_pricing::ratemaking::{
        Applies, Period, RateChange, RateHistory, TrendFit, trend_factor, year_fraction,
    };

    fn ymd(y: i32, m: u8, d: u8) -> f64 {
        year_fraction(y, m, d).unwrap()
    }

    fn round(x: f64, dp: i32) -> f64 {
        let s = 10f64.powi(dp);
        (x * s).round() / s
    }

    /// Within half a unit of the last printed digit.
    fn printed(x: f64, book: f64, dp: i32) {
        let tol = 0.5 * 10f64.powi(-dp) + 1e-12;
        assert!((x - book).abs() <= tol, "{x} is not {book} at {dp} dp");
    }

    fn calendar(y: i32) -> Period {
        Period::calendar(f64::from(y), f64::from(y + 1)).unwrap()
    }

    /// Tables 5.15 to 5.21: +5% on 7/1/10, +10% on 1/1/11, -1% on 4/1/12.
    fn chapter_5() -> RateHistory {
        RateHistory::new(&[
            RateChange::written(ymd(2010, 7, 1), 0.05),
            RateChange::written(ymd(2011, 1, 1), 0.10),
            RateChange::written(ymd(2012, 4, 1), -0.01),
        ])
        .unwrap()
    }

    fn shares(h: &RateHistory, p: &Period, term: f64) -> Vec<f64> {
        h.rate_level(p, term)
            .unwrap()
            .portions
            .iter()
            .map(|p| p.share)
            .collect()
    }

    fn close_all(got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= tol, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn parallelogram_chapter_5() {
        let h = chapter_5();
        // Figures 5.17 to 5.21: (period, term, shares, average, factor).
        let py2012 = Period::policy(2012.0, 2013.0).unwrap();
        let cases = [
            (
                calendar(2011),
                1.0,
                vec![0.125, 0.375, 0.5],
                1.0963,
                1.0430559,
                1.0431,
            ),
            (
                calendar(2011),
                0.5,
                vec![0.25, 0.75],
                1.1288,
                1.0130233,
                1.0130,
            ),
            (py2012, 1.0, vec![0.25, 0.75], 1.1464, 0.9974811, 0.9975),
        ];
        // Table 5.19 prints the levels at 4 dp (1.1435 for 1.14345).
        let printed_levels = RateHistory::from_levels(
            &[ymd(2010, 7, 1), ymd(2011, 1, 1), ymd(2012, 4, 1)],
            &[1.05, 1.155, 1.1435],
            &[Applies::Written; 3],
        )
        .unwrap();
        for (p, term, want, average, factor, book) in cases {
            close_all(&shares(&h, &p, term), &want, 1e-14);
            assert!((h.rate_level(&p, term).unwrap().on_level_factor() - factor).abs() < 1e-7);
            let l = printed_levels.rate_level(&p, term).unwrap();
            printed(l.average_level, average, 4);
            printed(l.current_level / average, book, 4);
        }
        printed(h.current_level(), 1.1435, 4);
    }

    #[test]
    fn parallelogram_appendix_a_six_month() {
        // Current Rate Level-1 and -2: six-month policies, calendar years.
        let effective = [
            ymd(2011, 4, 1),
            ymd(2012, 7, 1),
            ymd(2013, 10, 1),
            ymd(2014, 7, 1),
            ymd(2015, 10, 1),
            ymd(2016, 1, 1),
        ];
        let changes = [-0.05, 0.10, 0.05, -0.02, 0.05, 0.05];
        let book_levels = [0.9500, 1.0450, 1.0973, 1.0754, 1.1292, 1.1857];
        let rows: [(i32, &[f64], f64, f64, f64); 5] = [
            (2011, &[0.5, 0.5], 0.9750, 1.2161, 1.2159),
            (2012, &[0.75, 0.25], 0.9738, 1.2176, 1.2175),
            (2013, &[0.9375, 0.0625], 1.0483, 1.1311, 1.1309),
            (2014, &[0.0625, 0.6875, 0.25], 1.0886, 1.0892, 1.0891),
            (2015, &[0.9375, 0.0625], 1.0788, 1.0991, 1.0991),
        ];
        let written = vec![Applies::Written; 6];
        let book = RateHistory::from_levels(&effective, &book_levels, &written).unwrap();
        let exact = RateHistory::new(
            &effective
                .iter()
                .zip(changes)
                .map(|(&e, c)| RateChange::written(e, c))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        for (y, want, average, factor, exact_factor) in rows {
            close_all(&shares(&book, &calendar(y), 0.5), want, 1e-12);
            // The book rounds the average to 4 dp, then the factor.
            let l = book.rate_level(&calendar(y), 0.5).unwrap();
            printed(l.average_level, average, 4);
            // 0.97375 does not round half up reliably in binary, so divide
            // by the printed average once it is checked.
            printed(round(l.current_level, 4) / average, factor, 4);
            // The exact chain of changes moves the factor by up to 2e-4.
            printed(
                exact
                    .rate_level(&calendar(y), 0.5)
                    .unwrap()
                    .on_level_factor(),
                exact_factor,
                4,
            );
        }
    }

    #[test]
    fn parallelogram_appendix_c_annual() {
        // Current Rate Level-1 and -2: annual policies, calendar years.
        let effective = [ymd(2011, 10, 1), ymd(2014, 3, 1), ymd(2015, 7, 1)];
        let levels = [0.9240, 1.0589, 1.2029];
        let h = RateHistory::from_levels(&effective, &levels, &[Applies::Written; 3]).unwrap();
        let rows: [(i32, &[f64], f64); 6] = [
            (2010, &[1.0], 1.2029),
            (2011, &[0.96875, 0.03125], 1.2058),
            (2012, &[0.28125, 0.71875], 1.2724),
            (2013, &[1.0], 1.3018),
            (2014, &[0.652_777_777_777_8, 0.347_222_222_222_2], 1.2391),
            (
                2015,
                &[0.013_888_888_888_9, 0.861_111_111_111_1, 0.125],
                1.1190,
            ),
        ];
        for (y, want, factor) in rows {
            close_all(&shares(&h, &calendar(y), 1.0), want, 1e-12);
            let l = h.rate_level(&calendar(y), 1.0).unwrap();
            printed(
                round(l.current_level, 4) / round(l.average_level, 4),
                factor,
                4,
            );
        }
    }

    #[test]
    fn trend_lengths() {
        // Section 0.4 of the spec: (from, to, years) for each of the book's
        // examples. Forecast policies are written while the rates are in
        // effect.
        let (annual, six) = (1.0, 0.5);
        let cy11 = calendar(2011);
        let py11 = Period::policy(2011.0, 2012.0).unwrap();
        let rates_2013 = Period::policy(2013.0, 2014.0).unwrap();
        let rates_2013_2y = Period::policy(2013.0, 2015.0).unwrap();
        let rates_2015 = Period::policy(2015.0, 2016.0).unwrap();
        let cases = [
            // Premium (chapter 5): average written dates.
            (
                cy11.average_written_date(annual),
                rates_2013.average_written_date(annual),
                2.5,
            ),
            (
                cy11.average_written_date(six),
                rates_2013.average_written_date(six),
                2.25,
            ),
            (
                py11.average_written_date(annual),
                rates_2013.average_written_date(annual),
                2.0,
            ),
            (
                cy11.average_written_date(annual),
                rates_2013_2y.average_written_date(annual),
                3.0,
            ),
            (
                cy11.average_earned_date(annual),
                rates_2013.average_earned_date(annual),
                2.5,
            ),
            (cy11.average_written_date(annual), ymd(2011, 11, 15), 0.875),
            (
                ymd(2011, 11, 15),
                rates_2013.average_written_date(annual),
                1.625,
            ),
            // Losses (chapter 6): average accident dates.
            (
                cy11.average_earned_date(annual),
                rates_2015.average_earned_date(annual),
                4.5,
            ),
            (
                cy11.average_earned_date(six),
                rates_2015.average_earned_date(six),
                4.25,
            ),
            (
                py11.average_earned_date(annual),
                rates_2015.average_earned_date(annual),
                4.0,
            ),
            (
                py11.average_earned_date(six),
                rates_2015.average_earned_date(six),
                4.0,
            ),
            (
                calendar(2010).average_earned_date(annual),
                Period::policy(2012.0, 2013.0)
                    .unwrap()
                    .average_earned_date(annual),
                2.5,
            ),
            // Appendix C: annual policies, rates effective 5/1/2016.
            (
                calendar(2010).average_earned_date(annual),
                Period::policy(ymd(2016, 5, 1), ymd(2017, 5, 1))
                    .unwrap()
                    .average_earned_date(annual),
                6.0 + 10.0 / 12.0,
            ),
            // Appendix A: six-month policies, rates effective 1/1/2017,
            // from the 12 months ending 12/31/2015.
            (
                Period::policy(2015.0, 2016.0)
                    .unwrap()
                    .average_written_date(six),
                Period::policy(2017.0, 2018.0)
                    .unwrap()
                    .average_written_date(six),
                2.0,
            ),
            (
                calendar(2015).average_earned_date(six),
                Period::policy(2017.0, 2018.0)
                    .unwrap()
                    .average_earned_date(six),
                2.25,
            ),
        ];
        for (i, (from, to, years)) in cases.into_iter().enumerate() {
            let years: f64 = years;
            assert!(
                (to - from - years).abs() < 1e-12,
                "case {i}: {} not {years}",
                to - from
            );
        }
    }

    #[test]
    fn premium_trend() {
        // One step, Table 5.24's 2% over 2.5 years.
        printed(trend_factor(0.02, 2.5), 1.0508, 4);
        // Table 5.29, two-step: CY2011 earned at current rate level.
        let average_earned: f64 = 1_440_788.0 / 1_947.0;
        // The book divides by the average rounded to 740.00.
        let current = 753.00 / round(average_earned, 2);
        let projected = trend_factor(-0.01, 1.625);
        assert!((current - 1.017568).abs() < 1e-6);
        assert!((projected - 0.983801).abs() < 1e-6);
        // The book multiplies the factors rounded to 4 dp, and rounds the
        // total to 4 dp before applying it.
        let total = round(round(current, 4) * round(projected, 4), 4);
        printed(total, 1.0011, 4);
        printed(1_440_788.0 * total, 1_442_373.0, 0);
        printed(trend_factor(0.02, 0.875), 1.0175, 4);
        // Appendix A, Premium Trend-3: (EP at current rate level, earned
        // exposures) by calendar year, to the latest 115.35 and +2% for 2
        // years.
        let rows = [
            (1_364_916.59, 12_900.0, 1.1342),
            (1_405_728.94, 13_020.0, 1.1116),
            (1_448_424.45, 13_130.0, 1.0879),
            (1_492_177.86, 13_258.0, 1.0663),
            (1_536_267.03, 13_380.0, 1.0452),
        ];
        for (ep, exposure, total) in rows {
            let current = 115.35 / round(ep / exposure, 2);
            printed(
                round(round(current, 4) * round(trend_factor(0.02, 2.0), 4), 4),
                total,
                4,
            );
        }
    }

    fn quarters(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / 4.0).collect()
    }

    /// Exponential fits to the last `n` points against printed annual
    /// changes in percent, within 0.05 points of the printed tenth.
    fn fits(values: &[f64], step: f64, book: &[(usize, f64)]) {
        for &(n, pct) in book {
            let v = &values[values.len() - n..];
            let t: Vec<f64> = (0..n).map(|i| i as f64 * step).collect();
            let fit = TrendFit::exponential(&t, v).unwrap();
            let got = 100.0 * fit.annual_change();
            assert!(
                (got - pct).abs() <= 0.05 + 1e-9,
                "{n} points: {got} vs {pct}"
            );
        }
    }

    #[test]
    fn loss_trend_table_6_14() {
        // Frequency as printed (4 dp), severity and pure premium (2 dp).
        let freq = [
            0.0587, 0.0587, 0.0593, 0.0587, 0.0582, 0.0578, 0.0565, 0.0555, 0.0552, 0.0553, 0.0556,
            0.0560, 0.0559, 0.0554, 0.0551, 0.0549, 0.0550, 0.0549, 0.0547, 0.0544,
        ];
        let sev = [
            1061.45, 1076.56, 1085.56, 1098.02, 1102.46, 1107.52, 1117.71, 1123.70, 1127.63,
            1124.01, 1116.42, 1109.48, 1105.38, 1102.61, 1096.67, 1093.77, 1100.67, 1108.79,
            1122.13, 1126.46,
        ];
        let pp = [
            62.32, 63.16, 64.33, 64.44, 64.17, 64.04, 63.20, 62.41, 62.24, 62.11, 62.09, 62.19,
            61.82, 61.13, 60.48, 60.03, 60.49, 60.86, 61.37, 61.28,
        ];
        fits(
            &freq,
            0.25,
            &[
                (20, -1.7),
                (16, -1.3),
                (12, -0.7),
                (8, -1.2),
                (6, -0.9),
                (4, -1.5),
            ],
        );
        fits(
            &sev,
            0.25,
            &[
                (20, 0.5),
                (16, -0.1),
                (12, -0.2),
                (8, 1.2),
                (6, 2.5),
                (4, 3.3),
            ],
        );
        fits(
            &pp,
            0.25,
            &[
                (20, -1.2),
                (16, -1.4),
                (12, -0.9),
                (8, -0.1),
                (6, 1.6),
                (4, 1.9),
            ],
        );
        // The 20-point linear fit: -$0.75 a year.
        let linear = TrendFit::linear(&quarters(20), &pp).unwrap();
        printed(linear.annual_change(), -0.75, 2);
        // Trend factors: 1% for 4 years, and the two-step example.
        printed(trend_factor(0.01, 4.0), 1.0406, 4);
        printed(trend_factor(-0.01, 2.0), 0.98, 2);
        printed(trend_factor(0.02, 2.5), 1.05, 2);
    }

    #[test]
    fn loss_trend_appendices() {
        // Appendix A, Premium Trend-1: average written premium at current
        // rate level, 12 months ending each quarter.
        let premium = [
            103.05, 103.58, 104.15, 104.72, 105.29, 105.84, 106.39, 106.92, 107.42, 107.96, 108.55,
            109.16, 109.75, 110.32, 110.87, 111.40, 111.95, 112.54, 113.10, 113.69, 114.25, 114.79,
            115.35,
        ];
        fits(
            &premium,
            0.25,
            &[
                (20, 2.1),
                (16, 2.1),
                (12, 2.0),
                (8, 2.0),
                (6, 2.0),
                (4, 1.9),
            ],
        );
        // Appendix A, Loss Trend-5: -0.5% current, +0.5% projected for 2.25
        // years, accident years 2011 to 2015 (4 to 0 years to 6/30/2015).
        let totals = [0.9912, 0.9962, 1.0012, 1.0062, 1.0113];
        for (k, total) in totals.into_iter().enumerate() {
            let current = round(trend_factor(-0.005, 4.0 - k as f64), 4);
            let projected = round(trend_factor(0.005, 2.25), 4);
            printed(round(current * projected, 4), total, 4);
        }
        // Appendix C, Net Trend-1 and -2: severity and adjusted frequency
        // by accident year 2005 to 2014, fitted on annual points.
        let severity = [
            91_032.0, 56_271.0, 49_372.0, 100_052.0, 69_625.0, 166_764.0, 169_710.0, 78_630.0,
            217_319.0, 299_634.0,
        ];
        let frequency = [2.92, 2.22, 4.31, 3.91, 5.87, 3.90, 3.78, 4.35, 4.35, 3.45];
        fits(&severity[..7], 1.0, &[(7, 17.0)]);
        fits(&frequency[..7], 1.0, &[(7, 8.2)]);
        fits(&severity, 1.0, &[(10, 16.3), (5, 15.2)]);
        fits(&frequency, 1.0, &[(10, 3.5), (5, -1.0)]);
        // Net Trend-3: 8.9% from each accident year to 5/1/2017, with the
        // book's lengths at 2 dp.
        let factors = [
            (6.83, 1.7902),
            (5.83, 1.6439),
            (4.83, 1.5095),
            (3.83, 1.3862),
            (2.83, 1.2729),
        ];
        for (years, factor) in factors {
            printed(trend_factor(0.089, years), factor, 4);
        }
    }
}

// The overall indication against Werner and Modlin, chapters 7, 8 and 12
// and Appendices A to C; the book's rounding is reproduced where it changes
// a printed figure.
mod indication {
    use prospicio_pricing::credibility::LimitedFluctuation;
    use prospicio_pricing::ratemaking::{
        ExpenseCategory, LossRatioExperience, LossRatioYear, Provisions, PurePremiumExperience,
        PurePremiumYear, credibility_weighted, premium_based_expenses, trend_factor,
        trended_present_rates,
    };

    fn round(x: f64, dp: i32) -> f64 {
        let s = 10f64.powi(dp);
        (x * s).round() / s
    }

    fn printed(x: f64, book: f64, dp: i32) {
        let tol = 0.5 * 10f64.powi(-dp) + 1e-9;
        assert!((x - book).abs() <= tol, "{x} is not {book} at {dp} dp");
    }

    #[test]
    fn chapters_7_8_and_12() {
        // Chapter 7: $180 of loss and LAE, $20 fixed, V 15%, Q 5%: $250.
        let p = Provisions::new(0.15, 0.05).unwrap();
        printed(p.pure_premium_rate(180.0, 20.0), 250.0, 2);
        // Table 7.3: the right premium against the all-variable one.
        let all_variable = Provisions::new(0.23, 0.05).unwrap();
        for (loss, right, wrong) in [
            (135.0, 193.75, 187.50),
            (180.0, 250.0, 250.0),
            (225.0, 306.25, 312.50),
        ] {
            printed(p.pure_premium_rate(loss, 20.0), right, 2);
            printed(all_variable.pure_premium_rate(loss, 0.0), wrong, 2);
        }
        // Premium-based projection with a 10% fixed ratio on $250: $256.25.
        printed(p.pure_premium_rate(180.0, 0.10 * 250.0), 256.25, 2);
        // Table 7.4: 5.9% general expenses, 75% fixed: 4.4% and 1.5%.
        let (fixed, variable) = premium_based_expenses(&[ExpenseCategory {
            ratio: 0.059,
            fixed_share: 0.75,
        }])
        .unwrap();
        printed(100.0 * fixed, 4.4, 1);
        printed(100.0 * variable, 1.5, 1);
        // Chapter 8: $500 and +10%.
        let p = Provisions::new(0.25, 0.10).unwrap();
        printed(p.pure_premium_rate(300.0, 25.0), 500.0, 2);
        printed(p.loss_ratio_change(0.65, 0.065), 0.10, 4);
        // Chapter 12: 100 claims against 1,082, Z 0.30; $270.
        let lf = LimitedFluctuation::new(0.90, 0.05).unwrap();
        let z = lf.credibility(100.0, lf.frequency_standard().round());
        printed(z, 0.30, 2);
        printed(credibility_weighted(200.0, round(z, 2), 300.0), 270.0, 2);
        // Trended present rates: $229; the larger group's change: $220.
        let c = trended_present_rates(0.10, 0.06, trend_factor(0.05, 2.0));
        printed(200.0 * (1.0 + c), 229.0, 0);
        printed(200.0 * 330.0 / 300.0, 220.0, 2);
    }

    #[test]
    fn methods_agree_on_consistent_data() {
        // Chapter 8: with premium at current rates P_C = X × current rate,
        // the loss ratio method's factor times the current rate is the pure
        // premium method's rate.
        let (exposure, current_rate, losses, fixed_dollars) = (1_000.0, 480.0, 310_000.0, 26_000.0);
        let p = Provisions::new(0.22, 0.04).unwrap();
        let premium = exposure * current_rate;
        let factor = 1.0 + p.loss_ratio_change(losses / premium, fixed_dollars / premium);
        let rate = p.pure_premium_rate(losses / exposure, fixed_dollars / exposure);
        assert!((factor * current_rate - rate).abs() < 1e-10);
    }

    #[test]
    fn appendix_a_loss_ratio() {
        // LR Indication-1: (EP, CRL, premium trend, losses, LDF, loss trend)
        // with ULAE 1.143.
        let rows = [
            (
                1_122_372.0,
                1.2161,
                1.1342,
                856_495.0,
                1.0000,
                0.9912,
                1_548_088.0,
                970_359.0,
            ),
            (
                1_154_508.0,
                1.2176,
                1.1116,
                867_184.0,
                0.9799,
                0.9962,
                1_562_608.0,
                967_578.0,
            ),
            (
                1_280_545.0,
                1.1311,
                1.0879,
                835_120.0,
                1.0003,
                1.0012,
                1_575_741.0,
                955_974.0,
            ),
            (
                1_369_976.0,
                1.0892,
                1.0663,
                821_509.0,
                1.0282,
                1.0062,
                1_591_109.0,
                971_450.0,
            ),
            (
                1_397_750.0,
                1.0991,
                1.0452,
                797_866.0,
                1.0966,
                1.0113,
                1_605_706.0,
                1_011_357.0,
            ),
        ];
        let years: Vec<LossRatioYear> = rows
            .iter()
            .map(|&(ep, crl, pt, l, ldf, lt, _, _)| LossRatioYear {
                earned_premium: ep,
                on_level_factor: crl,
                premium_trend: pt,
                losses: l,
                development: ldf,
                loss_trend: lt,
                lae_factor: 1.143,
            })
            .collect();
        for (y, r) in years.iter().zip(&rows) {
            printed(y.projected_premium(), r.6, 0);
            printed(y.projected_losses(), r.7, 0);
        }
        let e = LossRatioExperience::new(&years).unwrap();
        // The book sums rounded rows: 7,883,252 against 7,883,253.19.
        assert!((e.projected_premium() - 7_883_253.19).abs() < 0.01);
        printed(e.projected_losses(), 4_876_718.0, 0);
        printed(100.0 * e.loss_ratio(), 61.9, 1);
        // Expense-1: premium-based, the selected ratios split and rounded
        // to 0.1%: 11.3% fixed, 17.0% variable.
        let split = |ratio: f64, share: f64| {
            let (f, v) = premium_based_expenses(&[ExpenseCategory {
                ratio,
                fixed_share: share,
            }])
            .unwrap();
            (round(100.0 * f, 1), round(100.0 * v, 1))
        };
        let parts = [
            split(0.063, 0.75),
            split(0.085, 0.75),
            split(0.002, 1.0),
            split(0.112, 0.0),
            split(0.021, 0.0),
        ];
        let (fixed, variable): (f64, f64) =
            parts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
        printed(fixed, 11.3, 1);
        printed(variable, 17.0, 1);
        // -6.2% from the selected 61.9%, 11.3%, 17.0% and 5% profit.
        let p = Provisions::new(0.170, 0.05).unwrap();
        printed(100.0 * p.variable_permissible_loss_ratio(), 78.0, 1);
        printed(100.0 * p.loss_ratio_change(0.619, 0.113), -6.2, 1);
        // Credibility-1: 3,612 claims are fully credible; the complement
        // is 13.2% indicated, 5.0% taken, net trend 1.005 / 1.02 for a year.
        let lf = LimitedFluctuation::new(0.90, 0.05).unwrap();
        assert_eq!(lf.credibility(3_612.0, 1_082.0), 1.0);
        let complement = trended_present_rates(0.132, 0.05, (1.005f64 / 1.02).powf(1.0));
        printed(100.0 * complement, 6.2, 1);
        printed(
            100.0 * credibility_weighted(p.loss_ratio_change(0.619, 0.113), 1.0, complement),
            -6.2,
            1,
        );
    }

    #[test]
    fn appendix_b_pure_premium() {
        // PP Indication-1: (exposure, non-cat losses, LDF, loss trend) with
        // ULAE 1.012, and the projected losses.
        let rows = [
            (12_760.0, 5_161_624.0, 1.0000, 1.1939, 6_236_412.0),
            (12_766.0, 4_820_968.0, 1.0012, 1.1705, 5_717_511.0),
            (12_805.0, 4_112_172.0, 1.0055, 1.1476, 4_802_025.0),
            (12_834.0, 5_052_052.0, 1.0186, 1.1251, 5_859_265.0),
            (13_411.0, 6_559_224.0, 1.0555, 1.1030, 7_727_993.0),
        ];
        let years: Vec<PurePremiumYear> = rows
            .iter()
            .map(|&(x, l, ldf, lt, _)| PurePremiumYear {
                exposure: x,
                losses: l,
                development: ldf,
                loss_trend: lt,
                lae_factor: 1.012,
            })
            .collect();
        for (y, r) in years.iter().zip(&rows) {
            printed(y.projected_losses(), r.4, 0);
        }
        let e = PurePremiumExperience::new(&years).unwrap();
        printed(e.projected_losses(), 30_343_206.0, 0);
        printed(e.pure_premium(), 469.88, 2);
        // 683 claims against 1,082: Z 79.5%, rounded before weighting.
        let z = LimitedFluctuation::new(0.90, 0.05)
            .unwrap()
            .credibility(683.0, 1_082.0);
        printed(z, 0.795, 3);
        let weighted = credibility_weighted(469.88, round(z, 3), 585.75);
        printed(weighted, 493.63, 2);
        // Reinsurance-1: $214,575 over 13,411 exposures grown 1% for 2 years.
        let reinsurance = 214_575.0 / round(13_411.0 * trend_factor(0.01, 2.0), 0);
        printed(reinsurance, 15.68, 2);
        // Expense-1: $72.79 fixed per exposure trended 3.4% for 2 years.
        let fixed = 72.79 * round(trend_factor(0.034, 2.0), 4);
        printed(fixed, 77.83, 2);
        // Variable 13.8%, profit 5%: (493.63 + 29.11 + 74.57 + 15.68 +
        // 77.83) / 81.2% = $850.76.
        let p = Provisions::new(0.138, 0.05).unwrap();
        let pure_premium = 493.63 + 29.11 + 74.57;
        printed(pure_premium, 597.31, 2);
        printed(p.pure_premium_rate(pure_premium + 15.68, 77.83), 850.76, 2);
    }

    #[test]
    fn appendix_c_loss_ratio_net_trend() {
        // LR Indication-1: (EP, CRL, ultimate loss and ALAE, net trend), no
        // premium trend, no ULAE in the losses.
        let rows = [
            (
                14_904_664.0,
                1.2029,
                11_673_500.0,
                1.7902,
                17_928_820.0,
                20_897_900.0,
            ),
            (
                14_494_543.0,
                1.2058,
                11_200_835.0,
                1.6439,
                17_477_520.0,
                18_413_053.0,
            ),
            (
                14_442_449.0,
                1.2724,
                6_290_368.0,
                1.5095,
                18_376_572.0,
                9_495_310.0,
            ),
            (
                14_834_605.0,
                1.3018,
                18_254_793.0,
                1.3862,
                19_311_689.0,
                25_304_794.0,
            ),
            (
                18_265_093.0,
                1.2391,
                23_371_444.0,
                1.2729,
                22_632_277.0,
                29_749_511.0,
            ),
        ];
        let years: Vec<LossRatioYear> = rows
            .iter()
            .map(|&(ep, crl, l, nt, _, _)| LossRatioYear {
                earned_premium: ep,
                on_level_factor: crl,
                premium_trend: 1.0,
                losses: l,
                development: 1.0,
                loss_trend: nt,
                lae_factor: 1.0,
            })
            .collect();
        for (y, r) in years.iter().zip(&rows) {
            printed(y.projected_premium(), r.4, 0);
            printed(y.projected_losses(), r.5, 0);
        }
        let e = LossRatioExperience::new(&years).unwrap();
        printed(e.projected_premium(), 95_726_878.0, 0);
        printed(e.projected_losses(), 103_860_568.0, 0);
        printed(100.0 * e.loss_ratio(), 108.5, 1);
        // Expense-1: all variable, 32.3% underwriting and 2.4% ULAE.
        let categories: Vec<ExpenseCategory> = [0.064, 0.026, 0.021, 0.212, 0.024]
            .iter()
            .map(|&ratio| ExpenseCategory {
                ratio,
                fixed_share: 0.0,
            })
            .collect();
        let (fixed, variable) = premium_based_expenses(&categories).unwrap();
        assert_eq!(fixed, 0.0);
        printed(100.0 * variable, 34.7, 1);
        // Profit -5%: permissible 70.3%, indicated +54.3%.
        let p = Provisions::new(variable, -0.05).unwrap();
        printed(100.0 * p.permissible_loss_ratio(0.0), 70.3, 1);
        let indicated = p.loss_ratio_change(1.085, 0.0);
        printed(100.0 * indicated, 54.3, 1);
        // 283 claims against 683 (k 7.5%, p 95%): Z 64.4%; with the
        // countrywide +18.5%, +41.6%.
        let lf = LimitedFluctuation::new(0.95, 0.075).unwrap();
        printed(lf.frequency_standard(), 683.0, 0);
        let z = lf.credibility(283.0, 683.0);
        printed(z, 0.644, 3);
        printed(
            100.0 * credibility_weighted(round(indicated, 3), round(z, 3), 0.185),
            41.6,
            1,
        );
    }
}

#[test]
fn pricing_bounds_match_aggregate() {
    use prospicio_pricing::bounds::{PricingBounds, distortion_envelope};
    // validation/scripts/aggregate_bounds.py: InsCo's total as X.
    let xv = [22.0, 28.0, 36.0, 40.0, 55.0, 65.0, 100.0];
    let xp = [0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1];
    let ys = [
        (
            "X3".to_string(),
            vec![0.0, 7.0, 8.0, 9.0, 10.0, 11.0, 40.0, 64.0],
            vec![0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ),
        (
            "Z".to_string(),
            vec![0.0, 30.0, 60.0, 90.0],
            vec![0.4, 0.3, 0.2, 0.1],
        ),
    ];
    let uncapped = PricingBounds::new((&xv, &xp), &ys, f64::INFINITY).unwrap();
    let capped = PricingBounds::new((&xv, &xp), &ys, 65.0).unwrap();
    let cases = reference("bounds_aggregate.csv");
    let (grid, exact): (Vec<_>, Vec<_>) = cases
        .into_iter()
        .partition(|c| c.get("distribution").starts_with("bounds_"));
    check(&exact, |c| {
        let arg = c.number("arg")?;
        match c.get("distribution") {
            "price" => {
                let pb = if c.param("params", "a").is_finite() {
                    &capped
                } else {
                    &uncapped
                };
                let (risk, side) = c.get("quantity").split_once('.')?;
                let r = pb.bounds(arg).ok()?.into_iter().find(|r| r.risk == risk)?;
                Some(if side == "lower" { r.lower } else { r.upper })
            }
            "envelope" => {
                let premium = c.param("params", "premium");
                let (lo, hi) =
                    distortion_envelope((&xv, &xp), premium, f64::INFINITY, &[arg]).ok()?[0];
                Some(if c.get("quantity") == "lower" { lo } else { hi })
            }
            _ => None,
        }
    });
    // aggregate's Bounds searches a grid of levels, nearly a subset of the
    // distortions: the exact envelope lies outside its envelope, close by.
    // Its members price X at P to its root tolerance and its upper envelope
    // interpolates linearly in s, so it may poke out by 2.4e-4 (at s = 0.1).
    for c in &grid {
        let s = c.number("arg").unwrap();
        let theirs = c.number("expected").unwrap();
        let premium = c.param("params", "premium");
        let (lo, hi) = distortion_envelope((&xv, &xp), premium, f64::INFINITY, &[s]).unwrap()[0];
        if c.get("distribution") == "bounds_min" {
            assert!(
                lo <= theirs + 1e-3 && theirs - lo < 0.05,
                "s = {s}: {lo} vs {theirs}"
            );
        } else {
            assert!(
                hi >= theirs - 1e-3 && hi - theirs < 0.05,
                "s = {s}: {hi} vs {theirs}"
            );
        }
    }
}

// Experience rating against Clark, *Basics of Reinsurance Pricing* (CAS
// study note, 2014), the per-risk example of section A.4.a: 400,000 xs
// 100,000 effective 1/1/97, losses trended at 4% to 7/1/97. The note's
// trend factors use actual days over 365.25; it rounds each factor to
// three decimals and each amount to the dollar before the next step
// (`knowledge/references/clark-reinsurance-pricing.md`).

mod clark_experience {
    use prospicio_aggregate::{Layer, Tower};
    use prospicio_pricing::experience::{Experience, LargeLoss, Trend, YearBasis};

    /// Days from 1/1/1900 by the proleptic Gregorian calendar.
    fn days(y: i32, m: u32, d: u32) -> i64 {
        let (y, m) = if m <= 2 { (y - 1, m + 12) } else { (y, m) };
        let (y, m, d) = (i64::from(y), i64::from(m), i64::from(d));
        365 * y + y / 4 - y / 100 + y / 400 + (153 * (m - 3) + 2) / 5 + d
    }

    /// The accident date as a decimal year on the note's basis: actual days
    /// to 7/1/1997 over 365.25.
    fn decimal(y: i32, m: u32, d: u32) -> f64 {
        1997.5 - (days(1997, 7, 1) - days(y, m, d)) as f64 / 365.25
    }

    /// (accident date, untrended, factor, trended, loss in layer), as printed.
    const LOSSES: [((i32, u32, u32), f64, f64, f64, f64); 14] = [
        ((1988, 9, 20), 240_946.0, 1.411, 339_975.0, 239_975.0),
        ((1988, 10, 11), 821_499.0, 1.408, 1_156_671.0, 400_000.0),
        ((1989, 3, 15), 158_129.0, 1.385, 219_009.0, 119_009.0),
        ((1990, 6, 21), 114_051.0, 1.317, 150_205.0, 50_205.0),
        ((1990, 10, 24), 78_043.0, 1.300, 101_456.0, 1_456.0),
        ((1991, 1, 10), 162_533.0, 1.289, 209_505.0, 109_505.0),
        ((1992, 2, 23), 324_298.0, 1.234, 400_184.0, 300_184.0),
        ((1992, 4, 30), 100_549.0, 1.225, 123_173.0, 23_173.0),
        ((1992, 9, 22), 75_476.0, 1.206, 91_024.0, 0.0),
        ((1993, 1, 1), 171_885.0, 1.193, 205_059.0, 105_059.0),
        ((1993, 5, 18), 94_218.0, 1.175, 110_706.0, 10_706.0),
        ((1993, 8, 1), 170_297.0, 1.166, 198_566.0, 98_566.0),
        ((1994, 8, 15), 87_133.0, 1.119, 97_502.0, 0.0),
        ((1995, 7, 12), 771_249.0, 1.080, 832_949.0, 400_000.0),
    ];

    /// (year, on-level premium, trended in layer, LDF, ultimate in layer,
    /// loss cost), as printed.
    const YEARS: [(i32, f64, f64, f64, f64, f64); 8] = [
        (1988, 1_422_554.0, 639_975.0, 1.000, 639_975.0, 0.450),
        (1989, 1_823_103.0, 119_009.0, 1.000, 119_009.0, 0.065),
        (1990, 2_054_034.0, 51_661.0, 1.000, 51_661.0, 0.025),
        (1991, 2_147_147.0, 109_505.0, 1.000, 109_505.0, 0.051),
        (1992, 2_151_541.0, 323_357.0, 1.010, 326_591.0, 0.152),
        (1993, 2_159_198.0, 214_331.0, 1.050, 225_048.0, 0.104),
        (1994, 2_167_158.0, 0.0, 1.150, 0.0, 0.000),
        (1995, 2_187_654.0, 400_000.0, 1.300, 520_000.0, 0.238),
    ];

    fn experience() -> Experience {
        Experience::new(
            LOSSES
                .iter()
                .enumerate()
                .map(|(i, &((y, m, d), amount, ..))| LargeLoss {
                    id: format!("{}", i + 1),
                    accident: decimal(y, m, d),
                    amount,
                })
                .collect(),
        )
        .unwrap()
    }

    fn treaty() -> Tower {
        Tower::new(vec![
            Layer::xol("400 xs 100", 400_000.0, 100_000.0).unwrap(),
        ])
        .unwrap()
    }

    /// The trend factors at 4% to 7/1/97, to the note's three decimals.
    #[test]
    fn trend_factors_match_the_note() {
        let p = experience()
            .prepare(&Trend::Rate {
                rate: 0.04,
                to: 1997.5,
            })
            .unwrap();
        for (l, &(_, _, factor, ..)) in p.losses().iter().zip(&LOSSES) {
            assert!(
                (l.factor - factor).abs() <= 0.0005 + 1e-12,
                "{} vs {factor}",
                l.factor
            );
        }
    }

    /// From the printed factors: each trended loss and its loss in the
    /// layer to the dollar, the layer losses by year, developed, and the
    /// loss costs to the printed 0.1%.
    #[test]
    fn burning_cost_matches_the_note() {
        let factors = LOSSES.iter().map(|l| l.2).collect();
        let p = experience().prepare(&Trend::Factors(factors)).unwrap();
        let layer = treaty();
        for (l, &(_, _, _, trended, in_layer)) in p.losses().iter().zip(&LOSSES) {
            assert!(
                (l.trended - trended).abs() <= 0.5,
                "{} vs {trended}",
                l.trended
            );
            // The note takes the layer from the rounded trended loss.
            assert!((layer.ceded(&[trended])[0] - in_layer).abs() < 1e-9);
        }
        let years: Vec<YearBasis> = YEARS
            .iter()
            .map(|&(year, premium, _, development, ..)| YearBasis {
                year,
                premium,
                development,
            })
            .collect();
        let towers = vec![treaty(); years.len()];
        let b = p.burning_cost(&towers, &years).unwrap();
        for (row, &(year, _, in_layer, _, ultimate, cost)) in b.years.iter().zip(&YEARS) {
            assert_eq!(row.year, year);
            // At most three losses a year, each half a dollar from the note's.
            assert!(
                (row.layer_loss[0] - in_layer).abs() <= 1.5,
                "{year}: {:?}",
                row.layer_loss
            );
            assert!(
                (row.developed[0] - ultimate).abs() <= 2.0,
                "{year}: {:?}",
                row.developed
            );
            assert!(
                (row.loss_cost[0] - cost).abs() <= 0.0005 + 1e-12,
                "{year}: {:?}",
                row.loss_cost
            );
        }
        let ultimate: f64 = b.years.iter().map(|r| r.developed[0]).sum();
        assert!((ultimate - 1_991_789.0).abs() <= 6.0, "{ultimate}");
        // The note's total is the developed losses over the premium, summed.
        assert!(
            (b.loss_cost[0] - 0.124).abs() <= 0.0005,
            "{:?}",
            b.loss_cost
        );
        assert!((ultimate / 16_112_389.0 - b.loss_cost[0]).abs() < 1e-15);
    }
}
