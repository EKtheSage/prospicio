//! The catalogue Excel registers: names, help texts within Excel's limits,
//! and every symbol reachable through `call`.

use std::collections::HashSet;

use prospicio_xll::sheet::Grid;
use prospicio_xll::{ALIASES, COMMON, COMMON_PREFIXES, FUNCTIONS, GROUP_CATEGORIES, call};

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
    assert_eq!(
        exports.len(),
        FUNCTIONS.len() + ALIASES.len(),
        "one export per function and per old name"
    );
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

/// Each old name is exported with the arguments of the function it now runs.
#[test]
fn alias_exports_take_their_targets_arguments() {
    let src = include_str!("../src/xll.rs");
    for a in ALIASES {
        let line = src
            .lines()
            .map(str::trim)
            .find(|l| {
                l.strip_prefix("udf!(")
                    .is_some_and(|r| r.split([',', ')']).next() == Some(a.symbol))
            })
            .unwrap_or_else(|| panic!("no export for {}", a.name));
        let f = FUNCTIONS.iter().find(|f| f.symbol == a.target).unwrap();
        assert_eq!(line.matches(',').count(), f.args.len(), "{}", a.name);
    }
}

/// The old names: each runs a function of the catalogue under its old
/// name's symbol, none is a current name, and only the group was added.
#[test]
fn aliases_point_at_the_catalogue() {
    let mut seen = HashSet::new();
    for a in ALIASES {
        assert!(seen.insert(a.name), "{} twice", a.name);
        assert!(
            FUNCTIONS
                .iter()
                .all(|f| f.name != a.name && f.symbol != a.symbol)
        );
        let f = FUNCTIONS
            .iter()
            .find(|f| f.symbol == a.target)
            .unwrap_or_else(|| panic!("{} runs nothing", a.name));
        assert_eq!(a.symbol, a.name.to_lowercase().replace('.', "_"));
        let (group, rest) = f.name["PRO.".len()..].split_once('.').unwrap();
        assert!(
            GROUP_CATEGORIES.iter().any(|(g, _)| *g == group),
            "{}",
            f.name
        );
        assert_eq!(a.name, format!("PRO.{rest}"), "{}", f.name);
        // An alias calls through to its function.
        let out = call(a.symbol, vec![Grid::missing(); f.args.len()]);
        assert!(!format!("{:?}", out.cells).contains("no function"));
    }
}

/// Every name is in a group, or is one of the common functions.
#[test]
fn every_function_has_a_group() {
    for f in FUNCTIONS {
        let first = f.name.split('.').nth(1).unwrap();
        let grouped = GROUP_CATEGORIES.iter().any(|(g, _)| *g == first);
        assert!(
            grouped || COMMON_PREFIXES.contains(&first),
            "{} is in no group",
            f.name
        );
        assert_eq!(f.category() == COMMON, !grouped, "{}", f.name);
        assert_eq!(f.symbol, f.name.to_lowercase().replace('.', "_"));
    }
}

/// The table of old and new names in `docs/design/xll.md` is the alias table.
#[test]
fn the_design_note_lists_every_rename() {
    let note = include_str!("../../../docs/design/xll.md");
    let mut rows: Vec<(String, String)> = note
        .lines()
        .filter_map(|l| {
            let cells: Vec<&str> = l.split('|').map(str::trim).collect();
            // | `PRO.OLD` | `PRO.GRP.OLD` | Group |
            let old = cells.get(1)?.strip_prefix("`PRO.")?.strip_suffix('`')?;
            let new = cells.get(2)?.strip_prefix('`')?.strip_suffix('`')?;
            Some((format!("PRO.{old}"), new.to_owned()))
        })
        .filter(|(_, new)| new.starts_with("PRO."))
        .collect();
    let mut expected: Vec<(String, String)> = ALIASES
        .iter()
        .map(|a| {
            let f = FUNCTIONS.iter().find(|f| f.symbol == a.target).unwrap();
            (a.name.to_owned(), f.name.to_owned())
        })
        .collect();
    rows.sort();
    expected.sort();
    assert_eq!(rows, expected);
}
