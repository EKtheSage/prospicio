//! Distributions from their characteristic function on a sheet
//! (`docs/design/xll.md`, decision 67): `prospicio_prob::chf`'s engine,
//! `aggregate`'s `AggregateCHF` (#779), over the laws of #780, and the
//! alpha-stable law's own figures.
//!
//! `PRO.CAP.CHF(law, params, ...)` names a law and gives its parameters
//! as one row or column, in the order the law's table below lists them;
//! trailing parameters with a default may be left blank. It answers the
//! window's moments, VaR and TVaR and the inversion's diagnostics, or the
//! grid's points and bucket masses. `PRO.CAP.STABLE` and
//! `PRO.CAP.STABLE.PARETO` give a stable law's parameters in both
//! parameterizations, its mean, tail constants and `P(X > loc)`: the
//! first four values (alpha, beta, scale, the S1 location) are
//! `PRO.CAP.CHF`'s `params` for `"stable"`.

use prospicio_prob::chf::{ChfDistribution, ChfLaw, ChfOptions, Tilt, invert};
use prospicio_prob::levy::{NormalInverseGaussian, TemperedStable, TweediePower, VarianceGamma};
use prospicio_prob::{
    Distribution, Gamma, InverseGaussian, Normal, Stable, StableParameterization,
};

use crate::aggregate::{MAX_POINTS, percent};
use crate::cat::probabilities;
use crate::pricing::Result;
use crate::reserving::{count, number, number_or, text_option};
use crate::sheet::{Grid, Value};
use crate::unknown_name;

fn err(e: prospicio_core::Error) -> String {
    e.to_string()
}

/// Each law's name, its parameters in order, and how many are required
/// (the rest default: a stable law's scale 1 and location 0, the Lévy
/// laws' `theta` and `mu` 0).
const LAWS: &[(&str, &[&str], usize)] = &[
    ("normal", &["mu", "sigma"], 2),
    ("gamma", &["shape", "scale"], 2),
    ("inverse_gaussian", &["mean", "shape"], 2),
    ("tweedie", &["power", "mean", "dispersion"], 3),
    ("stable", &["alpha", "beta", "scale", "loc"], 2),
    ("stable_s0", &["alpha", "beta", "scale", "loc"], 2),
    (
        "normal_inverse_gaussian",
        &["alpha", "beta", "delta", "mu"],
        3,
    ),
    ("variance_gamma", &["sigma", "nu", "theta", "mu"], 2),
    ("tempered_stable", &["C", "G", "M", "Y", "mu"], 4),
];

/// The `params` range's cells in reading order, `None` for a blank one.
fn cells(g: &Grid) -> Result<Vec<Option<f64>>> {
    if g.rows != 1 && g.columns != 1 {
        return Err("params must be a row or a column".into());
    }
    g.cells
        .iter()
        .enumerate()
        .map(|(k, v)| match v {
            Value::Number(x) => Ok(Some(*x)),
            v if v.is_blank() => Ok(None),
            other => Err(format!(
                "params value {} must be a number, got {other:?}",
                k + 1
            )),
        })
        .collect()
}

/// The law `name` with the parameters in `params`.
fn law(name: &Grid, params: &Grid) -> Result<ChfLaw> {
    let name = text_option(name, "law")?.ok_or("law is required")?;
    let Some(&(name, names, required)) = LAWS.iter().find(|(n, _, _)| *n == name) else {
        let table: Vec<(&str, ())> = LAWS.iter().map(|(n, _, _)| (*n, ())).collect();
        return Err(unknown_name("law", &name, &table));
    };
    let given = cells(params)?;
    if given.len() > names.len() {
        return Err(format!(
            "params: {name} takes {} values ({}), got {}",
            names.len(),
            names.join(", "),
            given.len()
        ));
    }
    let mut v = Vec::with_capacity(names.len());
    for (k, p) in names.iter().enumerate() {
        match given.get(k).copied().flatten() {
            Some(x) => v.push(x),
            None if k < required => {
                return Err(format!(
                    "params: {name} takes {} ({p} is missing)",
                    names[..required].join(", ")
                ));
            }
            None => v.push(if name.starts_with("stable") && *p == "scale" {
                1.0
            } else {
                0.0
            }),
        }
    }
    Ok(match name {
        "normal" => ChfLaw::Normal(Normal::new(v[0], v[1]).map_err(err)?),
        "gamma" => ChfLaw::Gamma(Gamma::new(v[0], v[1]).map_err(err)?),
        "inverse_gaussian" => {
            ChfLaw::InverseGaussian(InverseGaussian::new(v[0], v[1]).map_err(err)?)
        }
        "tweedie" => ChfLaw::TweediePower(TweediePower::new(v[0], v[1], v[2]).map_err(err)?),
        "stable" | "stable_s0" => {
            let p = if name == "stable" {
                StableParameterization::S1
            } else {
                StableParameterization::S0
            };
            ChfLaw::Stable(Stable::new(v[0], v[1], v[2], v[3], p).map_err(err)?)
        }
        "normal_inverse_gaussian" => ChfLaw::NormalInverseGaussian(
            NormalInverseGaussian::new(v[0], v[1], v[2], v[3]).map_err(err)?,
        ),
        "variance_gamma" => {
            ChfLaw::VarianceGamma(VarianceGamma::new(v[0], v[1], v[2], v[3]).map_err(err)?)
        }
        "tempered_stable" => {
            ChfLaw::TemperedStable(TemperedStable::new(v[0], v[1], v[2], v[3], v[4]).map_err(err)?)
        }
        _ => unreachable!("{name} is in LAWS"),
    })
}

/// The `tilt` argument: blank or TRUE the automatic tilt, FALSE none, a
/// number its e-folds (`aggregate`'s `tilt`).
fn tilt(g: &Grid) -> Result<Tilt> {
    match g.as_scalar() {
        Some(v) if v.is_blank() => Ok(Tilt::Auto),
        Some(Value::Bool(true)) => Ok(Tilt::Auto),
        Some(Value::Bool(false)) => Ok(Tilt::Off),
        Some(Value::Number(x)) if x.is_finite() && *x >= 0.0 => Ok(Tilt::Fixed(*x)),
        Some(Value::Text(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Tilt::Auto),
            "off" => Ok(Tilt::Off),
            _ => Err(format!(
                "tilt must be TRUE, FALSE or a number of e-folds from 0, got {s:?}"
            )),
        },
        _ => Err("tilt must be TRUE, FALSE or a number of e-folds from 0".into()),
    }
}

/// An optional number argument, `None` when blank.
fn optional(g: &Grid, name: &str) -> Result<Option<f64>> {
    match g.as_scalar() {
        Some(v) if v.is_blank() => Ok(None),
        _ => number(g, name).map(Some),
    }
}

fn row(label: &str, x: f64) -> Vec<Value> {
    vec![Value::from(label), Value::number(x)]
}

fn flag(label: &str, x: bool) -> Vec<Value> {
    vec![Value::from(label), Value::Bool(x)]
}

/// The window's mean, SD, VaR and TVaR at each probability, then the
/// inversion's report.
fn summary(d: &ChfDistribution, ps: &[f64]) -> Result<Grid> {
    let r = d.report();
    let mut rows = vec![row("Mean", d.mean()), row("SD", d.std_dev())];
    for &p in ps {
        rows.push(vec![
            Value::Text(format!("VaR {}", percent(p))),
            Value::number(d.quantile(p).map_err(err)?),
        ]);
    }
    for &p in ps {
        rows.push(vec![
            Value::Text(format!("TVaR {}", percent(p))),
            Value::number(d.tvar(p).map_err(err)?),
        ]);
    }
    rows.extend([
        row("Law mean", r.law_mean),
        row("Law SD", r.law_variance.sqrt()),
        row("Step", r.step),
        row("Left edge", r.x_min),
        row("Points", d.masses().len() as f64),
        flag("Step given", r.step_given),
        row("Truncation error", r.truncation_error),
        flag("Truncated", r.is_truncated()),
        row("Mass below", r.outside_below),
        row("Mass above", r.outside_above),
        row("Tilt left", r.tilt.0),
        row("Tilt right", r.tilt.1),
        row("Edge error", r.edge_error),
        row("Total mass", r.total_mass),
    ]);
    Ok(Grid::from_rows(rows))
}

/// `PRO.CAP.CHF(law, params, [probabilities], [output], [log2], [step],
/// [x_min], [tilt])`: the law's distribution on `2^log2` buckets from its
/// characteristic function. `output` `"summary"` (the default) or
/// `"masses"`, the grid's points and their masses under a header row.
pub fn chf(args: &[Grid]) -> Result<Grid> {
    let law = law(&args[0], &args[1])?;
    let ps = probabilities(&args[2])?;
    let output = text_option(&args[3], "output")?.unwrap_or_else(|| "summary".into());
    if output != "summary" && output != "masses" {
        return Err(unknown_name(
            "output",
            &output,
            &[("summary", ()), ("masses", ())],
        ));
    }
    let max_log2 = MAX_POINTS.trailing_zeros() as usize - usize::from(output == "masses");
    let options = ChfOptions {
        log2: count(&args[4], "log2", 16, max_log2)? as u32,
        step: optional(&args[5], "step")?,
        x_min: optional(&args[6], "x_min")?,
        tilt: tilt(&args[7])?,
        ..ChfOptions::default()
    };
    let d = invert(&law, &options).map_err(err)?;
    if output == "masses" {
        let mut rows = vec![vec![Value::from("x"), Value::from("Mass")]];
        rows.extend(
            d.xs()
                .into_iter()
                .zip(d.masses())
                .map(|(x, &m)| vec![Value::number(x), Value::number(m)]),
        );
        return Ok(Grid::from_rows(rows));
    }
    summary(&d, &ps)
}

/// A stable law's figures: its parameters in S1 and its S0 location, its
/// mean, the constants of its two power tails and `P(X > loc)`.
fn stable_table(st: &Stable) -> Grid {
    Grid::from_rows(vec![
        row("Alpha", st.alpha()),
        row("Beta", st.beta()),
        row("Scale", st.scale()),
        row("Location (S1)", st.loc()),
        row("Location (S0)", st.loc0()),
        row("Mean", st.mean()),
        row("Right tail constant", st.tail_constant()),
        row("Left tail constant", st.left_tail_constant()),
        row("P(X > location)", st.prob_above_loc()),
    ])
}

/// `PRO.CAP.STABLE(alpha, beta, [scale], [loc], [parameterization])`:
/// `loc` read as `"S1"` (the default, scipy's) or `"S0"` (Nolan's).
pub fn stable(args: &[Grid]) -> Result<Grid> {
    let p = match text_option(&args[4], "parameterization")?.as_deref() {
        None | Some("s1") => StableParameterization::S1,
        Some("s0") => StableParameterization::S0,
        Some(other) => {
            return Err(unknown_name(
                "parameterization",
                other,
                &[("S1", ()), ("S0", ())],
            ));
        }
    };
    let st = Stable::new(
        number(&args[0], "alpha")?,
        number(&args[1], "beta")?,
        number_or(&args[2], "scale", 1.0)?,
        number_or(&args[3], "loc", 0.0)?,
        p,
    )
    .map_err(err)?;
    Ok(stable_table(&st))
}

/// `PRO.CAP.STABLE.PARETO(frequency, alpha, theta, sev_mean)`: the stable
/// limit of a Poisson compound whose severity has the Pareto tail
/// `(theta / x)^alpha`, `1 < alpha < 2` (`Stable::from_compound_pareto`,
/// the generalized central limit theorem), in `PRO.CAP.STABLE`'s layout.
pub fn stable_pareto(args: &[Grid]) -> Result<Grid> {
    let st = Stable::from_compound_pareto(
        number(&args[0], "frequency")?,
        number(&args[1], "alpha")?,
        number(&args[2], "theta")?,
        number(&args[3], "sev_mean")?,
    )
    .map_err(err)?;
    Ok(stable_table(&st))
}
