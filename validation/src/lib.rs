//! Parity harness: checks Rust results against reference implementations on
//! standard datasets, as the release gate in `docs/architecture.md` requires.
//!
//! - `reference/*.csv`: expected values, one case per row, each with its own
//!   `abs_tol` and `rel_tol` (a case passes if it meets either) and the
//!   `source` that produced it. Generator scripts live in `scripts/`.
//! - `data/*.csv`: reference datasets in long format.
//!
//! Parity suites are the integration tests in `tests/`.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use act_reserving::{DevelopmentColumn, Grain, Lag, Long, Month, Triangle};

/// One row of a reference file: column name to raw value.
#[derive(Debug, Clone)]
pub struct Case {
    fields: BTreeMap<String, String>,
}

impl Case {
    /// Raw value of `column`; panics if the file has no such column.
    pub fn get(&self, column: &str) -> &str {
        self.fields
            .get(column)
            .unwrap_or_else(|| panic!("reference file has no column {column:?}"))
    }

    /// `column` parsed as a number, or `None` if it is empty.
    pub fn number(&self, column: &str) -> Option<f64> {
        let raw = self.get(column);
        (!raw.is_empty()).then(|| {
            raw.parse()
                .unwrap_or_else(|_| panic!("{column} = {raw:?} is not a number"))
        })
    }

    /// A parameter from a `name=value;name=value` column.
    pub fn param(&self, column: &str, name: &str) -> f64 {
        self.get(column)
            .split(';')
            .find_map(|pair| pair.strip_prefix(name)?.strip_prefix('='))
            .unwrap_or_else(|| panic!("{column} has no parameter {name}"))
            .parse()
            .unwrap_or_else(|_| panic!("parameter {name} is not a number"))
    }

    fn expected(&self) -> f64 {
        self.number("expected").expect("expected is required")
    }

    fn passes(&self, got: f64) -> bool {
        let want = self.expected();
        let err = (got - want).abs();
        let abs_tol = self.number("abs_tol").unwrap_or(0.0);
        let rel_tol = self.number("rel_tol").unwrap_or(0.0);
        got == want || err <= abs_tol || err <= rel_tol * want.abs()
    }

    fn describe(&self) -> String {
        let mut out = String::new();
        for (k, v) in &self.fields {
            if !v.is_empty() && k != "source" {
                let _ = write!(out, "{k}={v} ");
            }
        }
        out
    }
}

/// Reads `validation/<relative>` as comma-separated rows, skipping `#` lines.
fn read_rows(relative: &str) -> Vec<BTreeMap<String, String>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut lines = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let header: Vec<&str> = lines
        .next()
        .expect("file has no header")
        .split(',')
        .collect();
    lines
        .map(|line| {
            let values: Vec<&str> = line.splitn(header.len(), ',').collect();
            assert_eq!(values.len(), header.len(), "bad row in {relative}: {line}");
            header
                .iter()
                .zip(values)
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        })
        .collect()
}

/// Loads a reference file from `validation/reference/`.
pub fn reference(name: &str) -> Vec<Case> {
    read_rows(&format!("reference/{name}"))
        .into_iter()
        .map(|fields| Case { fields })
        .collect()
}

/// Evaluates every case and panics with a list of all failures.
///
/// `eval` returns the Rust result for a case, or `None` if the case is not
/// covered yet; uncovered cases are reported as failures so that nothing in
/// a reference file is silently skipped.
pub fn check(cases: &[Case], mut eval: impl FnMut(&Case) -> Option<f64>) {
    assert!(!cases.is_empty(), "no parity cases supplied");
    let mut failures = Vec::new();
    for case in cases {
        match eval(case) {
            Some(got) if case.passes(got) => {}
            Some(got) => failures.push(format!(
                "{}: got {got:e}, want {:e} ({})",
                case.describe(),
                case.expected(),
                case.get("source")
            )),
            None => failures.push(format!("{}: not evaluated", case.describe())),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} parity cases failed:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

/// Loads a cumulative annual triangle from `validation/data/<name>.csv`
/// (columns `origin,development,value`: origin year, age in months, value)
/// with one column, `values`.
pub fn triangle(name: &str) -> Triangle {
    let rows = read_rows(&format!("data/{name}.csv"));
    let parse = |row: &BTreeMap<String, String>, k: &str| -> f64 {
        row[k]
            .parse()
            .unwrap_or_else(|_| panic!("{name}: {k} = {:?} is not a number", row[k]))
    };
    let origin: Vec<Month> = rows
        .iter()
        .map(|r| Month::january(parse(r, "origin") as i32))
        .collect();
    let ages: Vec<Lag> = rows
        .iter()
        .map(|r| parse(r, "development") as Lag)
        .collect();
    let values: Vec<f64> = rows.iter().map(|r| parse(r, "value")).collect();
    Triangle::from_long(&Long {
        index: None,
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("values", &values)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(expected: &str, abs_tol: &str, rel_tol: &str) -> Case {
        let fields = [
            ("expected", expected),
            ("abs_tol", abs_tol),
            ("rel_tol", rel_tol),
            ("source", "test"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        Case { fields }
    }

    #[test]
    fn passes_within_either_tolerance() {
        check(&[case("100", "0.5", "0")], |_| Some(100.4));
        check(&[case("100", "0", "1e-3")], |_| Some(100.09));
    }

    #[test]
    #[should_panic(expected = "1 of 1 parity cases failed")]
    fn fails_outside_tolerance() {
        check(&[case("100", "0.5", "1e-3")], |_| Some(100.6));
    }

    #[test]
    #[should_panic(expected = "not evaluated")]
    fn uncovered_case_fails() {
        check(&[case("1", "0", "0")], |_| None);
    }
}
