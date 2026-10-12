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
