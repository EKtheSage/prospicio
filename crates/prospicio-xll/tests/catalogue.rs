//! The catalogue Excel registers: names, help texts within Excel's limits,
//! and every symbol reachable through `call`.

use std::collections::HashSet;

use prospicio_xll::sheet::Grid;
use prospicio_xll::{FUNCTIONS, call};

#[test]
fn names_and_help_fit_excel() {
    let mut names = HashSet::new();
    let mut symbols = HashSet::new();
    for f in FUNCTIONS {
        assert!(f.name.starts_with("PRO."), "{}", f.name);
        assert!(
            names.insert(f.name) && symbols.insert(f.symbol),
            "{} twice",
            f.name
        );
        // xlfRegister takes strings of at most 255 characters; the last
        // argument's help is padded by two (xll.rs).
        assert!(f.help.len() < 256, "{} help", f.name);
        assert!(
            f.args.iter().all(|a| a.help.len() < 254),
            "{} argument help",
            f.name
        );
        let arg_names: Vec<_> = f.args.iter().map(|a| a.name).collect();
        assert!(arg_names.join(",").len() < 256);
    }
}

#[test]
fn every_symbol_dispatches() {
    for f in FUNCTIONS {
        // Missing arguments reach the function, which answers with a value
        // or an error; never "no function".
        let out = call(f.symbol, vec![Grid::missing(); f.args.len()]);
        let text = format!("{:?}", out.cells);
        assert!(!text.contains("no function"), "{}: {text}", f.name);
    }
    assert!(format!("{:?}", call("pro_nothing", vec![]).cells).contains("no function"));
}

/// Every catalogue entry has an export in `src/xll.rs` taking exactly its
/// arguments, and every export is in the catalogue: a mismatch would call
/// a function with the wrong number of arguments, which only Excel (and
/// the fake-Excel harness) would see.
#[test]
fn exports_match_the_catalogue() {
    let src = include_str!("../src/xll.rs");
    let exports: Vec<(String, usize)> = src
        .lines()
        .filter_map(|l| {
            l.trim()
                .strip_prefix("udf!(")?
                .strip_suffix(");")
                .map(str::to_owned)
        })
        .map(|inner| {
            let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
            (parts[0].to_owned(), parts.len() - 1)
        })
        .collect();
    assert_eq!(exports.len(), FUNCTIONS.len(), "one export per function");
    for f in FUNCTIONS {
        let found = exports.iter().find(|(s, _)| s == f.symbol);
        assert_eq!(
            found.map(|e| e.1),
            Some(f.args.len()),
            "{} ({})",
            f.name,
            f.symbol
        );
    }
}
