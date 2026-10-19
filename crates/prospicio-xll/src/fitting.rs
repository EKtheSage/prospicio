//! Severity fits to large losses (`docs/design/xll.md`, decision 25): a
//! column of losses, with each loss's reporting threshold, policy-limit
//! censoring and weight, fitted by maximum likelihood to a body family
//! (`prospicio_prob::severity_fit`) or a Pareto family (the R package
//! Pareto's estimators, `prospicio_prob::LargeLosses`). The parameters come
//! out in the order the severity functions take them, so a fit's cells
//! feed `PRO.SEV.CDF`, `PRO.LAYER` and the others directly.

use prospicio_prob::evt::Gpd;
use prospicio_prob::severity_fit::{Family, SeverityFit, fit_severity, goodness_of_fit};
use prospicio_prob::{Dist, LargeLosses, Pareto, PiecewisePareto, Truncation};

use crate::pricing::{numbers, piecewise_truncation, scalar};
use crate::sheet::{Grid, Value, position};

type Result<T> = std::result::Result<T, String>;

const FAMILIES: &str = "family must be \"lognormal\", \"gamma\", \"weibull\", \"burr\", \
\"pareto\", \"piecewise_pareto\" or \"gen_pareto\"";

const OUTPUTS: &str = "output must be \"params\", \"gof\" or \"compare\"";

/// The body families `"compare"` fits, in its rows' order.
const BODY: [Family; 4] = [
    Family::Lognormal,
    Family::Gamma,
    Family::Weibull,
    Family::Burr,
];

/// The losses and, per loss, the reporting threshold, censoring flag and
/// weight. A row whose loss cell is blank is skipped; the other columns
/// are one value for every loss or a range of the losses' shape.
fn large_losses(
    losses: &Grid,
    thresholds: &Grid,
    censored: &Grid,
    weights: &Grid,
) -> Result<LargeLosses> {
    let shaped = |g: &Grid, name: &str| -> Result<()> {
        if g.cells.len() == 1 || (g.rows == losses.rows && g.columns == losses.columns) {
            Ok(())
        } else {
            Err(format!(
                "{name} ({} x {}) must be one value or a range the size of losses ({} x {})",
                g.rows, g.columns, losses.rows, losses.columns
            ))
        }
    };
    for (g, name) in [
        (thresholds, "thresholds"),
        (censored, "censored"),
        (weights, "weights"),
    ] {
        shaped(g, name)?;
    }
    let cell = |g: &'_ Grid, i: usize| -> Value {
        if g.cells.len() == 1 {
            g.cells[0].clone()
        } else {
            g.cells[i].clone()
        }
    };
    let (mut y, mut r, mut c, mut w) = (vec![], vec![], vec![], vec![]);
    for i in 0..losses.cells.len() {
        let at = position(i / losses.columns, i % losses.columns);
        match &losses.cells[i] {
            Value::Number(x) => y.push(*x),
            v if v.is_blank() => continue,
            other => return Err(format!("losses {at} must be a number, got {other:?}")),
        }
        r.push(match cell(thresholds, i) {
            Value::Number(x) => x,
            v if v.is_blank() => 0.0,
            other => return Err(format!("thresholds {at} must be a number, got {other:?}")),
        });
        c.push(match cell(censored, i) {
            Value::Bool(b) => b,
            Value::Number(x) if x == 0.0 || x == 1.0 => x == 1.0,
            v if v.is_blank() => false,
            other => {
                return Err(format!(
                    "censored {at} must be TRUE or FALSE (or 1 or 0), got {other:?}"
                ));
            }
        });
        w.push(match cell(weights, i) {
            Value::Number(x) => x,
            v if v.is_blank() => 1.0,
            other => return Err(format!("weights {at} must be a number, got {other:?}")),
        });
    }
    if y.is_empty() {
        return Err("losses is required".into());
    }
    let e = |e: prospicio_core::Error| e.to_string();
    LargeLosses::new(y)
        .map_err(e)?
        .reporting_thresholds(r)
        .map_err(e)?
        .censored(c)
        .map_err(e)?
        .weights(w)
        .map_err(e)
}

/// A fit: its distribution, the named parameters with their standard
/// errors (NaN where there is none), and the body fits' likelihood.
struct Fitted {
    dist: Dist,
    parameters: Vec<(String, f64, f64)>,
    body: Option<SeverityFit>,
}

fn fit(family: &str, data: &LargeLosses, t: &Grid, truncation: &Grid) -> Result<Fitted> {
    let e = |e: prospicio_core::Error| e.to_string();
    let none = |g: &Grid, name: &str| -> Result<()> {
        if g.is_blank() {
            Ok(())
        } else {
            Err(format!(
                "{name} is taken only by the Pareto families, not {family:?}"
            ))
        }
    };
    if let Ok(body) = Family::from_name(family) {
        none(t, "t")?;
        none(truncation, "truncation")?;
        let f = fit_severity(body, data).map_err(e)?;
        let se = f.standard_errors();
        let parameters = body
            .parameter_names()
            .iter()
            .zip(f.parameters())
            .zip(se)
            .map(|((n, &p), s)| ((*n).to_owned(), p, s))
            .collect();
        return Ok(Fitted {
            dist: f.dist().clone(),
            parameters,
            body: Some(f),
        });
    }
    let nan = f64::NAN;
    match family {
        "pareto" => {
            let t = scalar(t, "t")?;
            let tr = match truncation.as_scalar() {
                Some(v) if v.is_blank() => None,
                _ => Some(scalar(truncation, "truncation")?),
            };
            let (p, se) = Pareto::fit_with_standard_error(t, data, tr).map_err(e)?;
            let mut parameters = vec![
                ("t".to_owned(), t, nan),
                ("alpha".to_owned(), p.alpha(), se),
            ];
            if let Some(tr) = tr {
                parameters.push(("truncation".to_owned(), tr, nan));
            }
            Ok(Fitted {
                dist: p.into(),
                parameters,
                body: None,
            })
        }
        "piecewise_pareto" => {
            let t = numbers(t, "t (thresholds)")?;
            let tr = if truncation.is_blank() {
                None
            } else {
                Some(piecewise_truncation(truncation)?)
            };
            let (p, se) = match tr {
                Some((_, Truncation::WholeDistribution)) => {
                    let p = PiecewisePareto::fit(t.clone(), data, tr).map_err(e)?;
                    let n = p.alphas().len();
                    (p, vec![nan; n])
                }
                _ => PiecewisePareto::fit_with_standard_errors(t.clone(), data, tr).map_err(e)?,
            };
            let parameters = p
                .alphas()
                .iter()
                .zip(se)
                .enumerate()
                .map(|(k, (&a, s))| (format!("alpha above {}", t[k]), a, s))
                .collect();
            Ok(Fitted {
                dist: p.into(),
                parameters,
                body: None,
            })
        }
        "gen_pareto" => {
            none(truncation, "truncation")?;
            let t = scalar(t, "t")?;
            let g = Gpd::fit_riegel(t, data).map_err(e)?;
            // Riegel's form: xi = 1 / alpha_tail, beta = t / alpha_ini.
            let parameters = vec![
                ("t".to_owned(), t, nan),
                ("alpha_ini".to_owned(), t / g.beta(), nan),
                ("alpha_tail".to_owned(), 1.0 / g.xi(), nan),
            ];
            Ok(Fitted {
                dist: g.into(),
                parameters,
                body: None,
            })
        }
        _ => Err(FAMILIES.into()),
    }
}

/// `PRO.SEV.FIT(losses, family, [output], [t], [truncation], [thresholds],
/// [censored], [weights])`: a maximum likelihood fit to large losses.
/// `output` is `"params"` (default: each parameter with its standard
/// error, then the fit's statistics), `"gof"` (the fit against the
/// product-limit estimate at each loss) or `"compare"` (the four body
/// families side by side).
pub fn severity_fit(args: &[Grid]) -> Result<Grid> {
    let output = match args[2].as_scalar() {
        Some(v) if v.is_blank() => "params".to_owned(),
        Some(Value::Text(s)) => s.trim().to_ascii_lowercase(),
        _ => return Err(OUTPUTS.into()),
    };
    let data = large_losses(&args[0], &args[5], &args[6], &args[7])?;
    if output == "compare" {
        if !args[1].is_blank() || !args[3].is_blank() || !args[4].is_blank() {
            return Err(
                "\"compare\" fits every body family: leave family, t and truncation blank".into(),
            );
        }
        return Ok(compare(&data));
    }
    let family = match args[1].as_scalar() {
        Some(Value::Text(s)) => s.trim().to_ascii_lowercase().replace([' ', '-'], "_"),
        _ => return Err(FAMILIES.into()),
    };
    match output.as_str() {
        "params" | "parameters" => params(&fit(&family, &data, &args[3], &args[4])?, &data),
        "gof" => gof(&fit(&family, &data, &args[3], &args[4])?, &data),
        _ => Err(OUTPUTS.into()),
    }
}

/// Why a body fit should not be read as it stands: a singular information
/// matrix (no standard errors) means the simplex stopped at a boundary of
/// the family, where its likelihood can still beat the others'. A Burr
/// with `alpha → 0` and `gamma → ∞` at a finite product is a Pareto from
/// its scale with alpha `alpha · gamma`.
fn note(f: &SeverityFit) -> Option<String> {
    let p = f.parameters();
    if f.family() == Family::Burr && p[0] < 1e-4 && p[1] > 1e4 {
        return Some(format!(
            "a Burr at its Pareto limit (alpha -> 0, gamma -> infinity): a Pareto from {:.6} with alpha {:.6}; fit \"pareto\" instead",
            p[2],
            p[0] * p[1]
        ));
    }
    if f.standard_errors().iter().any(|s| !s.is_finite()) {
        return Some("no standard errors: the fit is at a boundary of the family, so its parameters are not identified".into());
    }
    if !f.converged() {
        return Some("the simplex did not meet its tolerance".into());
    }
    None
}

/// The largest distance between the fit's survival and the product-limit
/// estimate, or NaN where the fit gives the data no probability.
fn distance(dist: &Dist, data: &LargeLosses) -> f64 {
    goodness_of_fit(dist, data).map_or(f64::NAN, |g| g.max_distance)
}

fn params(f: &Fitted, data: &LargeLosses) -> Result<Grid> {
    let row = |label: &str, x: Value| vec![Value::from(label), x, Value::Empty];
    let mut rows = vec![["Parameter", "Estimate", "SE"].map(Value::from).to_vec()];
    for (name, p, se) in &f.parameters {
        rows.push(vec![
            Value::Text(name.clone()),
            Value::number(*p),
            Value::number(*se),
        ]);
    }
    let body = f.body.as_ref();
    let stat = |g: fn(&SeverityFit) -> f64| Value::number(body.map_or(f64::NAN, g));
    rows.push(row("Losses", Value::number(data.len() as f64)));
    rows.push(row("Log-likelihood", stat(SeverityFit::log_likelihood)));
    rows.push(row("AIC", stat(SeverityFit::aic)));
    rows.push(row("BIC", stat(SeverityFit::bic)));
    rows.push(row("KS distance", Value::number(distance(&f.dist, data))));
    rows.push(row(
        "Converged",
        body.map_or(Value::Empty, |b| Value::Bool(b.converged())),
    ));
    let note = body.and_then(note).unwrap_or_else(|| "none".into());
    rows.push(row("Note", Value::Text(note)));
    Ok(Grid::from_rows(rows))
}

fn gof(f: &Fitted, data: &LargeLosses) -> Result<Grid> {
    let g = goodness_of_fit(&f.dist, data).map_err(|e| e.to_string())?;
    let mut rows = vec![
        [
            "Loss",
            "At risk",
            "Events",
            "Empirical survival",
            "Model survival",
            "Model quantile",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for (i, s) in g.steps.iter().enumerate() {
        rows.push(
            [
                s.x,
                s.at_risk,
                s.events,
                s.survival,
                g.model_survival[i],
                g.model_quantile[i],
            ]
            .map(Value::number)
            .to_vec(),
        );
    }
    Ok(Grid::from_rows(rows))
}

fn compare(data: &LargeLosses) -> Grid {
    let mut rows = vec![
        [
            "Family",
            "p1",
            "p2",
            "p3",
            "Log-likelihood",
            "AIC",
            "BIC",
            "KS distance",
            "Converged",
            "Note",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for family in BODY {
        let mut row = vec![Value::from(family.name())];
        match fit_severity(family, data) {
            Ok(f) => {
                let p = f.parameters();
                row.extend((0..3).map(|i| p.get(i).map_or(Value::Empty, |&x| Value::number(x))));
                row.extend(
                    [
                        f.log_likelihood(),
                        f.aic(),
                        f.bic(),
                        distance(f.dist(), data),
                    ]
                    .map(Value::number),
                );
                row.push(Value::Bool(f.converged()));
                row.push(Value::Text(note(&f).unwrap_or_else(|| "none".into())));
            }
            Err(e) => {
                row.push(Value::Text(format!("#PROSPICIO! {e}")));
                row.extend(std::iter::repeat_n(Value::Empty, 8));
            }
        }
        rows.push(row);
    }
    Grid::from_rows(rows)
}
