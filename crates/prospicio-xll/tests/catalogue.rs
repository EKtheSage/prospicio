//! The catalogue Excel registers: names, help texts within Excel's limits,
//! and every symbol reachable through `call`.

use std::collections::HashSet;

use prospicio_xll::jobs::ASYNC;
use prospicio_xll::sheet::Grid;
use prospicio_xll::{COMMON, COMMON_PREFIXES, FUNCTIONS, GROUP_CATEGORIES, call};

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
    let exports: Vec<(String, usize, bool)> = src
        .lines()
        .filter_map(|l| {
            let l = l.trim();
            let (inner, is_async) = match l.strip_prefix("udf!(") {
                Some(rest) => (rest, false),
                None => (l.strip_prefix("udf_async!(")?, true),
            };
            Some((inner.strip_suffix(");")?.to_owned(), is_async))
        })
        .map(|(inner, is_async)| {
            let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
            (parts[0].to_owned(), parts.len() - 1, is_async)
        })
        .collect();
    assert_eq!(exports.len(), FUNCTIONS.len(), "one export per function");
    for f in FUNCTIONS {
        let found = exports.iter().find(|(s, _, _)| s == f.symbol);
        assert_eq!(
            found.map(|e| e.1),
            Some(f.args.len()),
            "{} ({})",
            f.name,
            f.symbol
        );
        // Asynchronous exactly when listed in `jobs::ASYNC` (decision 48).
        assert_eq!(
            found.map(|e| e.2),
            Some(ASYNC.contains(&f.symbol)),
            "{} async",
            f.name
        );
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

/// Grouped functions added after the rename of decision 47, so with no old
/// name in its table.
const ADDED_AFTER_THE_RENAME: &[&str] = &[
    "PRO.RES.CURVE",
    "PRO.RES.DEV.EXHIBIT",
    "PRO.RES.SELECT",
    "PRO.RES.CURVE.STATS",
    "PRO.RES.MUNICH",
    "PRO.RES.MUNICH.PARAMS",
    "PRO.RES.PIC",
    "PRO.RES.PIC.PARAMS",
    "PRO.RES.FREQSEV",
    "PRO.RES.FREQSEV.TRENDED",
    "PRO.RES.DISPOSAL",
    "PRO.RES.CASE.OS",
    "PRO.RES.TREND.BY.AGE",
    "PRO.RES.BS.CASE",
    "PRO.RES.BS.SETTLEMENT",
    "PRO.RES.BS",
    "PRO.RES.RATIO",
    "PRO.RES.ULAE",
    "PRO.RES.INDUSTRY.PRIOR",
    "PRO.PRC.COUNTS",
    "PRO.PRC.COUNTS.PREDICT",
    "PRO.PRC.QUANTREG",
    "PRO.PRC.QUANTREG.PREDICT",
    "PRO.RES.CY.TEST",
    "PRO.RES.DEVCOR.TEST",
    "PRO.RES.BACKTEST",
    "PRO.RES.EXCESS.PATTERN",
    "PRO.RES.EXCESS.SHAPE",
    "PRO.RES.BENCHMARK.CURVE",
    "PRO.PRC.TREND.FACTORS",
    "PRO.PRC.LR.CUMULATIVE",
    "PRO.CAP.NESTED.CDF",
    "PRO.CAP.NESTED.SIM",
];

/// The table of old and new names in `docs/design/xll.md` (decision 47),
/// which a user's find-and-replace reads: every new name is in the
/// catalogue, its old name is it without the group, no old name is still a
/// name, and every grouped function is in the table but those added after
/// the rename, which never had a name without a group.
#[test]
fn the_design_note_lists_every_rename() {
    let note = include_str!("../../../docs/design/xll.md");
    let rows: Vec<(String, String)> = note
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
    for (old, new) in &rows {
        assert!(FUNCTIONS.iter().any(|f| f.name == new), "{new}");
        assert!(
            FUNCTIONS.iter().all(|f| f.name != old),
            "{old} is still a name"
        );
        let (_, rest) = new["PRO.".len()..].split_once('.').unwrap();
        assert_eq!(old, &format!("PRO.{rest}"));
    }
    let mut grouped: Vec<&str> = FUNCTIONS
        .iter()
        .filter(|f| f.category() != COMMON && !ADDED_AFTER_THE_RENAME.contains(&f.name))
        .map(|f| f.name)
        .collect();
    let mut listed: Vec<&str> = rows.iter().map(|(_, new)| new.as_str()).collect();
    grouped.sort_unstable();
    listed.sort_unstable();
    assert_eq!(grouped, listed);
}
