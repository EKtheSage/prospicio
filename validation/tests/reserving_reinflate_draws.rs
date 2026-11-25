//! Stochastic re-inflation of simulated future cells
//! (`reserving-inflation.md`, PR 8, decision 11, #979):
//! `Indexation::reinflate_draws` against the expected values that
//! `validation/scripts/reserving_reinflate_draws.py` computes from the same
//! stated inputs with its own calendar arithmetic. The draws are built
//! with `PredictiveDistribution::from_draws` from the file's `draw` rows,
//! the index and the paths from its `index`, `base`, `future_rate` and
//! `path` rows, and every re-inflated cell, reserve by origin and total
//! is checked to 1e-12 relative.

use std::collections::{BTreeMap, HashMap};

use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};
use prospicio_reserving::{CalendarIndex, Grain, IndexPaths, Indexation, Month, Period};
use prospicio_validation::{Case, check, reference};

/// `name=value;name=value` as a map.
fn args(case: &Case) -> BTreeMap<String, String> {
    case.get("arg")
        .split(';')
        .filter(|s| !s.is_empty())
        .map(|pair| {
            let (k, v) = pair.split_once('=').expect("name=value");
            (k.to_string(), v.to_string())
        })
        .collect()
}

/// A period label: `2023`, `2023Q4` or `2023-12`.
fn period(label: &str) -> Period {
    if let Some((year, month)) = label.split_once('-') {
        return Period::containing(
            Month::new(year.parse().unwrap(), month.parse().unwrap()).unwrap(),
            Grain::Month,
        );
    }
    match label.split_once('Q') {
        Some((year, q)) => {
            let q: u8 = q.parse().unwrap();
            Period::containing(
                Month::new(year.parse().unwrap(), 3 * (q - 1) + 1).unwrap(),
                Grain::Quarter,
            )
        }
        None => Period::year(label.parse().unwrap()),
    }
}

/// The objects one method's rows build, and its results.
struct Built {
    ix: Indexation,
    paths: Option<IndexPaths>,
    draws: PredictiveDistribution,
    out: PredictiveDistribution,
    by_origin: PredictiveDistribution,
}

fn cell(origin: &str, age: &str) -> Vec<KeyValue> {
    vec![
        KeyValue::Period(period(origin)),
        KeyValue::Int(age.parse().unwrap()),
    ]
}

fn build(cases: &[&Case]) -> Built {
    let value = |c: &Case| c.number("expected").unwrap();
    let of = |q: &'static str| cases.iter().filter(move |c| c.get("quantity") == q);
    let index_rows: Vec<_> = of("index").collect();
    let first = period(&args(index_rows[0])["period"]);
    let mut index =
        CalendarIndex::new(first, index_rows.iter().map(|c| value(c)).collect()).unwrap();
    if let Some(rate) = of("future_rate").next() {
        index = index.with_future_rate(value(rate)).unwrap();
    }
    let base = of("base").next().expect("a base row");
    let (year, month) = args(base)["month"]
        .split_once('-')
        .map(|(y, m)| (y.parse().unwrap(), m.parse().unwrap()))
        .unwrap();
    let ix = Indexation {
        index,
        base: Month::new(year, month).unwrap(),
    };

    let draw_rows: Vec<_> = of("draw").collect();
    let n_sims = draw_rows
        .iter()
        .map(|c| args(c)["sim"].parse::<usize>().unwrap())
        .max()
        .unwrap()
        + 1;
    let components: Vec<Vec<KeyValue>> = draw_rows
        .iter()
        .filter(|c| args(c)["sim"] == "0")
        .map(|c| {
            let a = args(c);
            cell(&a["origin"], &a["age"])
        })
        .collect();
    // The rows are simulation-major, cells in the same order in each.
    for (k, c) in draw_rows.iter().enumerate() {
        let a = args(c);
        assert_eq!(a["sim"], (k / components.len()).to_string());
        assert_eq!(
            cell(&a["origin"], &a["age"]),
            components[k % components.len()]
        );
    }
    let draws = PredictiveDistribution::from_draws(
        vec!["origin".into(), "development".into()],
        components,
        draw_rows.iter().map(|c| value(c)).collect(),
        Provenance::new("reserving_reinflate_draws.csv"),
    )
    .unwrap();
    assert_eq!(draws.n_sims(), n_sims);

    let path_rows: Vec<_> = of("path").collect();
    let paths = (!path_rows.is_empty()).then(|| {
        IndexPaths::new(
            period(&args(path_rows[0])["period"]),
            n_sims,
            path_rows.iter().map(|c| value(c)).collect(),
        )
        .unwrap()
    });
    let out = ix.reinflate_draws(&draws, paths.as_ref()).unwrap();
    assert_eq!(out.dims(), draws.dims());
    assert_eq!(out.components(), draws.components());
    let by_origin = out.aggregate(&["origin"]).unwrap();
    Built {
        ix,
        paths,
        draws,
        out,
        by_origin,
    }
}

#[test]
fn reinflated_draws_match_the_closed_form() {
    let cases = reference("reserving_reinflate_draws.csv");
    let mut by_method: BTreeMap<String, Vec<&Case>> = BTreeMap::new();
    for case in &cases {
        by_method
            .entry(case.get("method").into())
            .or_default()
            .push(case);
    }
    assert_eq!(
        by_method.keys().map(String::as_str).collect::<Vec<_>>(),
        ["annual_paths", "annual_rate", "monthly_paths", "quarterly_paths"]
    );
    let built: HashMap<String, Built> = by_method
        .iter()
        .map(|(m, rows)| (m.clone(), build(rows)))
        .collect();
    assert!(built["annual_rate"].paths.is_none());
    check(&cases, |case| {
        let b = &built[case.get("method")];
        let a = args(case);
        let sim = || a["sim"].parse::<usize>().unwrap();
        let column = |pd: &PredictiveDistribution, key: Vec<KeyValue>| {
            pd.row(sim())?
                .get(pd.component_index(&key)?)
                .copied()
        };
        match case.get("quantity") {
            "index" => {
                let k = b.ix.index.periods().iter().position(|&p| p == period(&a["period"]))?;
                Some(b.ix.index.values()[k])
            }
            "base" => b.ix.index.value(b.ix.base).ok(),
            "future_rate" => b.ix.index.future_rate(),
            "path" => {
                let paths = b.paths.as_ref()?;
                let k = paths.periods().iter().position(|&p| p == period(&a["period"]))?;
                Some(paths.path(sim())?[k])
            }
            "draw" => column(&b.draws, cell(&a["origin"], &a["age"])),
            "reinflated" => column(&b.out, cell(&a["origin"], &a["age"])),
            "reserve" => column(&b.by_origin, vec![KeyValue::Period(period(&a["origin"]))]),
            "total" => Some(b.out.row(sim())?.iter().sum()),
            _ => None,
        }
    });
}
