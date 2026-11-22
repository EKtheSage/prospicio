//! The defaults the help texts state are the defaults the code has (#955).
//!
//! A help text says "10000 if omitted" or "\"gamma\" (default)"; nothing
//! tied that to the code, which is how `PRO.RES.BAYES.BF`'s `apriori`
//! help came to describe a range the function refuses. Here every
//! function that answers a triangle alone (RAA, every other argument
//! omitted) is called again with each stated default given explicitly,
//! one argument at a time, and must answer the same.
//!
//! Not covered: a function that needs more than a triangle, a default
//! that is not one number, one word or TRUE/FALSE ("the latest if
//! omitted"), and the asynchronous functions (bootstraps and MCMC fits,
//! minutes in a debug build).

use prospicio_xll::jobs::ASYNC;
use prospicio_xll::sheet::{Grid, Value};
use prospicio_xll::{FUNCTIONS, call};

use crate::common::raa;

/// The default a help text states, if it is one value.
fn stated_default(help: &str) -> Option<Value> {
    if let Some(at) = help.find(" if omitted") {
        let words: Vec<&str> = help[..at].split_whitespace().collect();
        let last = words.last()?.trim_start_matches(['(', ';']);
        let before = words.len().checked_sub(2).map(|i| words[i]);
        // "2 to 1000 if omitted" and "0.99, 0.995, 0.999 if omitted" are
        // not one value.
        let listed = before.is_some_and(|b| {
            b == "to" || (b.ends_with(',') && b.trim_end_matches(',').parse::<f64>().is_ok())
        });
        if listed {
            return None;
        }
        return match last {
            "TRUE" => Some(Value::Bool(true)),
            "FALSE" => Some(Value::Bool(false)),
            _ => last.replace(',', "").parse().ok().map(Value::Number),
        };
    }
    let at = help.find(" (default)")?;
    let words: Vec<&str> = help[..at].split_whitespace().collect();
    let last = words.last()?.trim_matches(['"', '(', ',']);
    // "1 Poisson (default)": the value is the number, the word its name.
    if let Some(x) = words.len().checked_sub(2).and_then(|i| words[i].parse().ok()) {
        return Some(Value::Number(x));
    }
    (!last.is_empty()).then(|| Value::from(last))
}

/// Arguments whose stated default cannot be given on its own, with the
/// reason.
const NOT_ALONE: &[(&str, &str)] = &[(
    "weight",
    "the weight on the selected factors: an error without selected factors",
)];

fn is_error(g: &Grid) -> bool {
    matches!(g.as_scalar(), Some(Value::Text(s)) if s.starts_with("#PROSPICIO!"))
}

#[test]
fn the_parser_reads_the_forms_the_catalogue_uses() {
    let d = stated_default;
    assert_eq!(d("Simulations (rows), 1 to 1,000,000 (10,000 if omitted)"), Some(Value::Number(10000.0)));
    assert_eq!(d("Annual severity trend, 0 if omitted"), Some(Value::Number(0.0)));
    assert_eq!(d("one value or one per loss (FALSE if omitted)"), Some(Value::Bool(false)));
    assert_eq!(d("\"gamma\" (default), \"lognormal\" or \"none\""), Some(Value::from("gamma")));
    assert_eq!(d("Var N / E N: 1 Poisson (default), above 1 negative binomial"), Some(Value::Number(1.0)));
    assert_eq!(d("Return periods in years; 2 to 1000 if omitted"), None);
    assert_eq!(d("Probabilities; 0.99, 0.995, 0.999 if omitted"), None);
    assert_eq!(d("The origin the selections are at; the latest if omitted"), None);
    assert_eq!(d("A triangle"), None);
}

#[test]
fn a_stated_default_given_explicitly_changes_nothing() {
    let (mut functions, mut checked) = (0, 0);
    let mut wrong = Vec::new();
    for f in FUNCTIONS {
        if ASYNC.contains(&f.symbol) || f.args.first().is_none_or(|a| a.name != "triangle") {
            continue;
        }
        let base_args = || {
            let mut args = vec![Grid::missing(); f.args.len()];
            args[0] = raa();
            args
        };
        let base = call(f.symbol, base_args());
        if is_error(&base) {
            continue;
        }
        functions += 1;
        for (i, arg) in f.args.iter().enumerate().skip(1) {
            let Some(default) = stated_default(arg.help) else {
                continue;
            };
            if NOT_ALONE.iter().any(|(name, _)| *name == arg.name) {
                continue;
            }
            let mut args = base_args();
            args[i] = Grid::scalar(default.clone());
            let out = call(f.symbol, args);
            checked += 1;
            if out != base {
                let shown = match out.as_scalar() {
                    Some(Value::Text(s)) => s.clone(),
                    _ => "another result".into(),
                };
                wrong.push(format!("{} {} = {default:?}: {shown}", f.name, arg.name));
            }
        }
    }
    assert!(wrong.is_empty(), "{} of {checked}:\n{}", wrong.len(), wrong.join("\n"));
    // The check must not thin out unnoticed.
    assert!(functions >= 20 && checked >= 25, "{functions} functions, {checked} defaults");
    println!("{functions} functions, {checked} stated defaults");
}
