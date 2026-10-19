//! Distortions on a sheet (`docs/design/xll.md`, decision 29): a
//! distortion's `g(s)`, the risk measure it gives a discrete or simulated
//! distribution, and the member of each family that gives a target premium
//! (`prospicio_prob::distortion`), as `aggregate`'s `Distortion` and its
//! calibration to a price.

use prospicio_prob::Distortion;
use prospicio_prob::distortion::{Family, calibrate};

use crate::pricing::{map1, numbers, scalar};
use crate::sheet::{Grid, Value};

type Result<T> = std::result::Result<T, String>;

const NAMES: &str = "\"ccoc\", \"ph\", \"wang\", \"dual\", \"tvar\", \"exp\", or with r0 \
\"clin\", \"cll\", \"lep\" or \"ly\"";

/// A family by `aggregate`'s short name (or its long one); the capped
/// families take the intercept `r0`.
fn family(name: &str, r0_arg: &Grid) -> Result<Family> {
    let r0 = || scalar(r0_arg, "r0");
    let none = |f: Family| {
        if r0_arg.is_blank() {
            Ok(f)
        } else {
            Err(format!(
                "r0 is taken only by \"clin\", \"cll\", \"lep\" and \"ly\", not {name:?}"
            ))
        }
    };
    match name
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-'], "_")
        .as_str()
    {
        "ccoc" => none(Family::Ccoc),
        "ph" | "proportional_hazard" => none(Family::ProportionalHazard),
        "wang" => none(Family::Wang),
        "dual" | "dual_power" => none(Family::DualPower),
        "tvar" => none(Family::Tvar),
        "exp" | "exponential" => none(Family::Exponential),
        "clin" | "capped_linear" => Ok(Family::CappedLinear { r0: r0()? }),
        "cll" | "capped_log_linear" => Ok(Family::CappedLogLinear { r0: r0()? }),
        "lep" => Ok(Family::Lep { r0: r0()? }),
        "ly" | "linear_yield" => Ok(Family::LinearYield { r0: r0()? }),
        other => Err(format!("family must be {NAMES}, got {other:?}")),
    }
}

fn text(grid: &Grid, name: &str) -> Result<String> {
    match grid.as_scalar() {
        Some(Value::Text(s)) if !s.trim().is_empty() => Ok(s.clone()),
        _ => Err(format!("{name} must be one of {NAMES}")),
    }
}

fn distortion(name: &Grid, parameter: &Grid, r0: &Grid) -> Result<Distortion> {
    family(&text(name, "family")?, r0)?
        .with(scalar(parameter, "parameter")?)
        .map_err(|e| e.to_string())
}

/// `PRO.DISTORTION(s, family, parameter, [r0])`: `g(s)` at each
/// exceedance probability `s` in `[0, 1]`.
pub fn g(args: &[Grid]) -> Result<Grid> {
    let d = distortion(&args[1], &args[2], &args[3])?;
    map1(&args[0], "s", false, |s| {
        if (0.0..=1.0).contains(&s) {
            Ok(d.g(s))
        } else {
            Err(format!("s must be a probability in [0, 1], got {s}"))
        }
    })
}

/// The outcomes ascending with their probabilities: `probs` one per value
/// (rescaled to sum to 1), or equally likely draws when omitted.
fn outcomes(values: &Grid, probs: &Grid) -> Result<(Vec<f64>, Vec<f64>)> {
    let x = numbers(values, "values")?;
    let p = if probs.is_blank() {
        vec![1.0 / x.len() as f64; x.len()]
    } else {
        let p = numbers(probs, "probs")?;
        if p.len() != x.len() {
            return Err(format!(
                "{} values and {} probs: give one probability per value",
                x.len(),
                p.len()
            ));
        }
        if p.iter().any(|q| !(q.is_finite() && *q >= 0.0)) {
            return Err("probs must be non-negative".into());
        }
        let total: f64 = p.iter().sum();
        if (total - 1.0).abs() > 1e-9 {
            return Err(format!("probs must sum to 1, not {total}"));
        }
        p.iter().map(|q| q / total).collect()
    };
    let mut pairs: Vec<(f64, f64)> = x.into_iter().zip(p).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(pairs.into_iter().unzip())
}

/// `PRO.DISTORTION.PRICE(values, [probs], family, parameter, [r0])`: the
/// distortion risk measure `ρ_g(X) = Σ x (g(P(X ≥ x)) − g(P(X > x)))` of
/// the outcomes (draws when `probs` is omitted), with the mean and margin.
pub fn price(args: &[Grid]) -> Result<Grid> {
    let (x, p) = outcomes(&args[0], &args[1])?;
    let d = distortion(&args[2], &args[3], &args[4])?;
    let mean: f64 = x.iter().zip(&p).map(|(x, p)| x * p).sum();
    let rho = d.apply_discrete(&x, &p);
    let row = |label: &str, v: f64| vec![Value::from(label), Value::number(v)];
    Ok(Grid::from_rows(vec![
        row("Mean", mean),
        row("Price", rho),
        row("Margin", rho - mean),
        row("Loss ratio", mean / rho),
    ]))
}

/// `PRO.DISTORTION.CALIBRATE(values, [probs], premium, [families], [r0])`:
/// the member of each family whose price of the outcomes is `premium`,
/// for the five standard families (CCoC, PH, Wang, dual, TVaR) unless
/// `families` names others. To price a loss backed by assets `a`, give
/// the outcomes capped at `a`.
pub fn calibrate_families(args: &[Grid]) -> Result<Grid> {
    let (x, p) = outcomes(&args[0], &args[1])?;
    let premium = scalar(&args[2], "premium")?;
    let names: Vec<String> = if args[3].is_blank() {
        ["ccoc", "ph", "wang", "dual", "tvar"]
            .map(String::from)
            .to_vec()
    } else {
        args[3]
            .cells
            .iter()
            .filter(|v| !v.is_blank())
            .map(|v| match v {
                Value::Text(s) => Ok(s.clone()),
                other => Err(format!(
                    "families must be names such as \"ph\", got {other:?}"
                )),
            })
            .collect::<Result<_>>()?
    };
    let mut rows = vec![["Family", "Parameter", "Price"].map(Value::from).to_vec()];
    for name in &names {
        let f = family(name, &args[4])?;
        rows.push(match calibrate(f, &x, &p, premium) {
            Ok(d) => {
                let parameter = d.parameters().last().map_or(f64::NAN, |(_, v)| *v);
                vec![
                    Value::from(f.name()),
                    Value::number(parameter),
                    Value::number(d.apply_discrete(&x, &p)),
                ]
            }
            Err(e) => vec![
                Value::from(f.name()),
                Value::Text(format!("#PROSPICIO! {e}")),
                Value::Empty,
            ],
        });
    }
    Ok(Grid::from_rows(rows))
}
