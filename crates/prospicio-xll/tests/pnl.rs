//! The P&L ledger on a sheet (`PRO.PNL`, `docs/design/xll.md` decision
//! 43), reached through the catalogue as Excel reaches it:
//!
//! * every `aggregate` 1.0.1 case of
//!   `validation/reference/pnl_builders_aggregate.csv` the function can
//!   state (a book with premium, loss-basis, fixed and mixed expenses, one
//!   or two annual covers with commissions; the consolidated face, the
//!   walk and both peels): the sheet's statistics and scenario columns and
//!   the card, at the file's tolerances;
//! * the ratios against the crate's ledger, equally likely atoms when
//!   `probs` is omitted, a profit commission, and the errors.

mod common;

use common::{n, num, t, text, validation};
use prospicio_aggregate::pnl::{Book, Expense, Face, Source};
use prospicio_aggregate::{Layer, Tower};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn none() -> Grid {
    Grid::missing()
}

/// A block of rows, the short ones padded with blanks.
fn rows(mut cells: Vec<Vec<Value>>) -> Grid {
    let width = cells.iter().map(Vec::len).max().unwrap_or(0);
    for row in &mut cells {
        row.resize(width, Value::Empty);
    }
    Grid::from_rows(cells)
}

fn s(x: &str) -> Value {
    Value::from(x)
}

fn v(x: f64) -> Value {
    Value::Number(x)
}

/// A source of `pnl_builder_sources.csv`: its values and probabilities.
fn source(name: &str) -> (Grid, Grid) {
    let (x, p): (Vec<f64>, Vec<f64>) = validation("reference/pnl_builder_sources.csv")
        .lines()
        .skip(1)
        .map(|l| l.split(',').collect::<Vec<_>>())
        .filter(|f| f[0] == name)
        .map(|f| (f[1].parse::<f64>().unwrap(), f[2].parse::<f64>().unwrap()))
        .unzip();
    (Grid::column(&x), Grid::column(&p))
}

fn one_cover() -> Grid {
    rows(vec![
        vec![
            s("name"),
            s("limit"),
            s("attachment"),
            s("premium"),
            s("commission"),
        ],
        vec![s("agg 3 xs 5"), v(3.0), v(5.0), v(1.0), v(0.1)],
    ])
}

fn two_covers() -> Grid {
    rows(vec![
        vec![
            s("name"),
            s("limit"),
            s("attachment"),
            s("premium"),
            s("commission"),
        ],
        vec![s("Low"), v(2.0), v(3.0), v(0.5)],
        vec![s("High"), v(3.0), v(5.0), v(1.0), v(0.1)],
    ])
}

fn premium_expense() -> Grid {
    rows(vec![vec![Value::Empty, v(0.1)]])
}

/// A reference case's arguments: values, probs, premium, expenses,
/// covers, face.
fn case(name: &str) -> Option<[Grid; 6]> {
    let (small_x, small_p) = source("ex3a");
    let small = |premium: f64, expenses: Grid, covers: Grid, face: &str| {
        [
            small_x.clone(),
            small_p.clone(),
            n(premium),
            expenses,
            covers,
            t(face),
        ]
    };
    Some(match name {
        "ex3a" => small(10.0, premium_expense(), one_cover(), "walk"),
        "ex3b" => small(10.0, premium_expense(), one_cover(), "consolidated"),
        "ex6w" => small(10.0, premium_expense(), two_covers(), "walk"),
        "ex6td" => small(10.0, premium_expense(), two_covers(), "top-down"),
        "ex6bu" => small(10.0, premium_expense(), two_covers(), "bottom-up"),
        "ex1" => small(10.0, premium_expense(), none(), "consolidated"),
        "ex1w" => small(10.0, premium_expense(), none(), "walk"),
        "exp1" => small(
            10.0,
            rows(vec![
                vec![Value::Empty, v(0.1)],
                vec![Value::Empty, Value::Empty, v(0.05)],
                vec![Value::Empty, Value::Empty, Value::Empty, v(1.0)],
            ]),
            none(),
            "consolidated",
        ),
        "exp2" => small(
            10.0,
            rows(vec![vec![Value::Empty, v(0.1), v(0.05)]]),
            none(),
            "consolidated",
        ),
        "exp3" => small(
            10.0,
            rows(vec![
                vec![s("label"), s("premium"), s("loss"), s("fixed")],
                vec![s("Acq"), v(0.1)],
                vec![s("LAE"), Value::Empty, v(0.05)],
            ]),
            none(),
            "consolidated",
        ),
        "ex5" => {
            let (x, p) = source("ex5");
            [
                x,
                p,
                n(1000.0),
                rows(vec![
                    vec![s("LAE"), Value::Empty, v(0.05)],
                    vec![s("Acq"), v(0.25)],
                ]),
                none(),
                t("consolidated"),
            ]
        }
        // Relabelled legs and the derived premium are not arguments.
        _ => return None,
    })
}

fn pnl(args: &[Grid; 6], output: &str) -> Grid {
    let mut a = args.to_vec();
    a.push(t(output));
    let out = call("pro_pnl", a);
    assert!(out.rows > 1, "{out:?}");
    out
}

fn cell_text(v: &Value) -> &str {
    match v {
        Value::Text(s) => s,
        Value::Empty => "",
        other => panic!("{other:?}"),
    }
}

#[test]
fn ledgers_match_aggregate() {
    let levels = [
        "k01", "k05", "k10", "k25", "k50", "k75", "k90", "k95", "k99",
    ];
    let mut outputs: std::collections::HashMap<(String, &str), Grid> = Default::default();
    let (mut checked, mut cases) = (0, std::collections::BTreeSet::new());
    let reference = validation("reference/pnl_builders_aggregate.csv");
    for line in reference.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let (name, quantity, key, col) = (f[0], f[1], f[2], f[3]);
        // On the consolidated face aggregate's κ75 is the next atom by a
        // last bit (the Rust parity test checks that tie instead).
        if matches!(name, "ex3b" | "lblc") && col == "k75" {
            continue;
        }
        let Some(args) = case(name) else { continue };
        if !matches!(quantity, "sheet" | "card") {
            continue;
        }
        let out = outputs
            .entry((name.to_owned(), quantity))
            .or_insert_with(|| pnl(&args, quantity));
        let parts: Vec<&str> = key.split('|').collect();
        let row = (1..out.rows)
            .find(|&r| {
                let step = cell_text(out.get(r, 0));
                let side = cell_text(out.get(r, 1));
                match (quantity, parts.as_slice()) {
                    ("sheet", [st, sd, line]) => {
                        step == *st && side == *sd && cell_text(out.get(r, 2)) == *line
                    }
                    ("sheet", [sd, line]) => side == *sd && cell_text(out.get(r, 2)) == *line,
                    ("card", [st, sd]) => step == *st && side == *sd,
                    ("card", [sd]) => side == *sd,
                    _ => false,
                }
            })
            .unwrap_or_else(|| panic!("{name} {quantity}: no row {key}"));
        let column = match (quantity, col) {
            ("sheet", "EX") | ("card", "EX") => 3 - usize::from(quantity == "card"),
            ("sheet", "SD") => 4,
            ("sheet", "CV") => 5,
            ("sheet", "Skew") => 6,
            ("sheet", k) => 7 + levels.iter().position(|l| *l == k).expect("a level"),
            ("card", "SD") => 3,
            ("card", "CV") => 4,
            ("card", "Skew") => 5,
            ("card", "P01") => 6,
            ("card", "Median") => 7,
            ("card", "P99") => 8,
            other => panic!("{other:?}"),
        };
        let got = out.get(row, column);
        let (abs, rel): (f64, f64) = (f[5].parse().unwrap(), f[6].parse().unwrap());
        match f[4].parse::<f64>() {
            Ok(expected) if expected.is_finite() => {
                let x = num(got);
                assert!(
                    (x - expected).abs() <= abs + rel * expected.abs(),
                    "{name} {quantity} {key} {col}: {x} vs {expected}"
                );
            }
            _ => assert!(got.is_blank(), "{name} {key} {col}: {got:?}"),
        }
        checked += 1;
        cases.insert(name);
    }
    assert_eq!(cases.len(), 11, "{cases:?}");
    assert!(checked > 1500, "{checked}");
}

#[test]
fn ratios_are_the_crate_s() {
    let (x, p) = source("ex3a");
    let out = call(
        "pro_pnl",
        vec![
            x.clone(),
            p.clone(),
            n(10.0),
            premium_expense(),
            two_covers(),
            none(),
            t("ratios"),
        ],
    );
    let (xs, ps): (Vec<f64>, Vec<f64>) = (
        x.cells.iter().map(num).collect(),
        p.cells.iter().map(num).collect(),
    );
    let tower = Tower::new(vec![
        Layer::stop_loss("Low", 2.0, 3.0)
            .unwrap()
            .deposit_premium(0.5)
            .unwrap(),
        Layer::stop_loss("High", 3.0, 5.0)
            .unwrap()
            .deposit_premium(1.0)
            .unwrap()
            .ceding_commission(0.1)
            .unwrap(),
    ])
    .unwrap();
    let ledger = Book::new("Book", 10.0)
        .expense(Expense::premium(0.1))
        .ledger(&Source::new(xs, ps).unwrap(), Some(&tower), Face::Walk)
        .unwrap();
    let want = ledger.ratios();
    assert_eq!(out.rows, want.len() + 1, "{out:?}");
    assert_eq!(out.get(0, 6), &Value::from("LR"));
    for (r, w) in want.iter().enumerate() {
        assert_eq!(cell_text(out.get(r + 1, 0)), w.step);
        assert_eq!(num(out.get(r + 1, 1)), w.p);
        assert_eq!(num(out.get(r + 1, 4)), w.m);
        assert_eq!(out.get(r + 1, 8).is_blank(), w.cr.is_none());
    }
}

#[test]
fn simulated_years_are_equally_likely() {
    let years = Grid::column(&[2.0, 3.0, 8.0, 4.0]);
    let quarter = Grid::column(&[0.25; 4]);
    let a = call("pro_pnl", vec![years.clone(), none(), n(10.0)]);
    let b = call("pro_pnl", vec![years, quarter, n(10.0)]);
    assert_eq!(a, b);
    // Premium 10 less a mean loss of 4.25.
    let result = (1..a.rows)
        .find(|&r| a.get(r, 1) == &Value::from("Margin"))
        .unwrap();
    assert!((num(a.get(result, 3)) - 5.75).abs() < 1e-12, "{a:?}");
    // A scenario level of one's own.
    let c = call(
        "pro_pnl",
        vec![
            Grid::column(&[2.0, 3.0, 8.0, 4.0]),
            none(),
            n(10.0),
            none(),
            none(),
            none(),
            none(),
            Grid::row(&[0.2, 0.8]),
        ],
    );
    assert_eq!(c.columns, 9);
    assert_eq!(c.get(0, 8), &Value::from("Scenario 80%"));
}

#[test]
fn a_profit_commission_is_booked() {
    let (x, p) = source("ex3a");
    let covers = rows(vec![
        vec![
            s("limit"),
            s("attachment"),
            s("premium"),
            s("profit_share"),
            s("profit_allowance"),
        ],
        vec![v(3.0), v(5.0), v(1.0), v(0.5), v(0.2)],
    ]);
    let out = call("pro_pnl", vec![x, p, n(10.0), none(), covers, t("walk")]);
    let pc = (1..out.rows)
        .find(|&r| cell_text(out.get(r, 2)).contains("profit"))
        .unwrap_or_else(|| panic!("no profit commission row: {out:?}"));
    assert_eq!(cell_text(out.get(pc, 0)), "agg 3 xs 5");
    assert!(num(out.get(pc, 3)) > 0.0);
}

#[test]
fn errors() {
    let (x, p) = source("ex3a");
    let run = |args: Vec<Grid>| text(&call("pro_pnl", args)).to_owned();
    assert!(
        run(vec![x.clone(), Grid::column(&[0.5]), n(10.0)]).contains("one probability per value")
    );
    assert!(run(vec![x.clone(), p.clone()]).contains("premium is required"));
    let bad = rows(vec![vec![s("limit"), s("retention")], vec![v(1.0), v(2.0)]]);
    assert!(
        run(vec![x.clone(), p.clone(), n(10.0), none(), bad])
            .contains("unknown column \"retention\"")
    );
    let no_attachment = rows(vec![vec![s("limit")], vec![v(1.0)]]);
    assert!(
        run(vec![x.clone(), p.clone(), n(10.0), none(), no_attachment])
            .contains("needs a attachment column")
    );
    assert!(
        run(vec![
            x.clone(),
            p.clone(),
            n(10.0),
            none(),
            none(),
            t("sideways")
        ])
        .contains("face must be")
    );
    assert!(
        run(vec![
            x.clone(),
            p.clone(),
            n(10.0),
            none(),
            none(),
            t("top-down")
        ])
        .contains("tower")
    );
    assert!(
        run(vec![
            x.clone(),
            p.clone(),
            n(10.0),
            rows(vec![vec![s("Acq")]])
        ])
        .contains("no premium rate")
    );
    assert!(
        run(vec![x, p, n(10.0), none(), none(), none(), t("graph")]).contains("output must be")
    );
}
