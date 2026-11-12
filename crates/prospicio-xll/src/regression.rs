//! Zero-inflated and hurdle counts, and quantile and expectile regression,
//! on a sheet (`docs/design/xll.md`, decision 55): the 0.0.8 Models fits
//! beside `PRO.PRC.GLM` (`prospicio_glm::zero`, `prospicio_glm::quantile`).
//!
//! They take a table as the GLM functions do (`data, response, terms`,
//! decision 41) and refit from the cells on every call (decision 4).

use prospicio_glm::Glm;
use prospicio_glm::negbin::NegativeBinomialGlm;
use prospicio_glm::quantile::{Bandwidth, ExpectileRegression, QuantileRegression};
use prospicio_glm::zero::{
    CountDistribution, Hurdle, TwoPartFit, ZeroInflated, count_log_likelihoods, vuong,
};
use prospicio_math::special::norm_cdf;
use prospicio_models::{Coding, Design, Family, Fitted, Link, Model};

use crate::glm::{
    Predictor, Table, coding, column_name, design, number_text, predictors, term_level,
};
use crate::reserving::{count, text_option};
use crate::sheet::{Grid, Value};

type Result<T> = std::result::Result<T, String>;

fn err(e: impl ToString) -> String {
    e.to_string()
}

/// A two-sided normal p-value.
fn p_value(z: f64) -> f64 {
    2.0 * norm_cdf(-z.abs())
}

/// A part of a model: its predictors' coding, or the intercept alone.
struct Part {
    coding: Option<(Coding, Vec<Predictor>)>,
}

impl Part {
    /// The part's design on `table`, with `exposure`'s log as offset.
    fn design(&self, table: &Table, exposure: Option<&str>, what: &str) -> Result<Design> {
        match &self.coding {
            Some((c, p)) => design(c, table, p, exposure, what),
            None => {
                let n = table.rows();
                let d = Design::new(vec!["(Intercept)".into()], vec![vec![1.0; n]]).map_err(err)?;
                match exposure {
                    None => Ok(d),
                    Some(name) => d
                        .with_offset(table.numbers(name, what)?.iter().map(|x| x.ln()).collect())
                        .map_err(err),
                }
            }
        }
    }
}

/// A fitted two-part model with what it was fitted from.
struct TwoPartModel {
    fit: TwoPartFit,
    count: Part,
    zero: Part,
    exposure: Option<String>,
    /// The count part's design and the counts, for Vuong's test; `None`
    /// with prior weights, which the test does not take.
    plain: Option<(Design, Vec<f64>)>,
}

impl TwoPartModel {
    fn designs(&self, table: &Table, what: &str) -> Result<(Design, Design)> {
        Ok((
            self.count.design(table, self.exposure.as_deref(), what)?,
            self.zero.design(table, None, what)?,
        ))
    }
}

/// The `zero_terms` argument: blank for the count part's terms, `1` for an
/// intercept alone, else predictors as `terms` reads them.
fn zero_terms(grid: &Grid) -> Result<Option<Option<Vec<Predictor>>>> {
    if grid.is_blank() {
        return Ok(None);
    }
    let one = match grid.as_scalar() {
        Some(Value::Number(x)) => *x == 1.0,
        Some(Value::Text(s)) => s.trim() == "1",
        _ => false,
    };
    if one {
        return Ok(Some(None));
    }
    predictors(grid)
        .map(|p| Some(Some(p)))
        .map_err(|e| e.replacen("terms", "zero_terms", 1))
}

/// The model arguments at `args[0..9]`: `data, response, terms, [model],
/// [count], [zero_terms], [zero_link], [exposure], [weights]`, fitted.
fn two_part_fit(args: &[Grid]) -> Result<(TwoPartModel, Table)> {
    let table = Table::new(&args[0], "data")?;
    let response = column_name(&args[1], "response")?
        .ok_or("response is required: the name of the column of counts")?;
    let terms = predictors(&args[2])?;
    let hurdle = match text_option(&args[3], "model")?.as_deref() {
        None | Some("zero_inflated" | "zeroinfl") => false,
        Some("hurdle") => true,
        Some(other) => {
            return Err(format!(
                "model must be \"zero_inflated\" or \"hurdle\", got {other:?}"
            ));
        }
    };
    let dist = match text_option(&args[4], "count")?.as_deref() {
        None => CountDistribution::Poisson,
        Some(name) => CountDistribution::from_name(name)
            .ok_or_else(|| crate::unknown_name("count", name, CountDistribution::NAMES))?,
    };
    let link = match text_option(&args[6], "zero_link")?.as_deref() {
        None | Some("logit") => Link::Logit,
        Some("probit") => Link::Probit,
        Some("cloglog") => Link::Cloglog,
        Some(other) => {
            return Err(format!(
                "zero_link must be \"logit\", \"probit\" or \"cloglog\", got {other:?}"
            ));
        }
    };
    let exposure = column_name(&args[7], "exposure")?;
    let weights = column_name(&args[8], "weights")?;
    let count_part = Part {
        coding: Some((coding(&table, &terms)?, terms)),
    };
    let zero_part = match zero_terms(&args[5])? {
        None => {
            let p = predictors(&args[2])?;
            Part {
                coding: Some((coding(&table, &p)?, p)),
            }
        }
        Some(None) => Part { coding: None },
        Some(Some(p)) => Part {
            coding: Some((coding(&table, &p)?, p)),
        },
    };
    let mut c = count_part.design(&table, exposure.as_deref(), "data")?;
    let mut z = zero_part.design(&table, None, "data")?;
    if let Some(w) = &weights {
        let w = table.numbers(w, "data")?;
        c = c.with_weights(w.clone()).map_err(err)?;
        z = z.with_weights(w).map_err(err)?;
    }
    let y = table.numbers(&response, "data")?;
    if let Some(r) = y.iter().position(|&v| !(v >= 0.0 && v.fract() == 0.0)) {
        return Err(format!(
            "data's column {response:?} must be counts, whole numbers from 0; row {} is {}",
            r + 1,
            y[r]
        ));
    }
    let plain = weights.is_none().then(|| (c.clone(), y.clone()));
    let fit = if hurdle {
        Hurdle::new(dist).zero_link(link).fit_parts(&c, &z, &y)
    } else {
        ZeroInflated::new(dist)
            .zero_link(link)
            .fit_parts(&c, &z, &y)
    }
    .map_err(err)?;
    Ok((
        TwoPartModel {
            fit,
            count: count_part,
            zero: zero_part,
            exposure,
            plain,
        },
        table,
    ))
}

/// `PRO.PRC.COUNTS(data, response, terms, [model], [count], [zero_terms],
/// [zero_link], [exposure], [weights])`: each coefficient of the count and
/// zero parts with its SE, z, p-value and exp(estimate), then theta, the
/// log-likelihood, AIC and BIC, then (without prior weights) Vuong's test
/// against the plain Poisson or negative binomial GLM on the count terms,
/// positive favouring the two-part model.
pub fn two_part(args: &[Grid]) -> Result<Grid> {
    let (m, _) = two_part_fit(args)?;
    let fit = &m.fit;
    let header = [
        "Part",
        "Term",
        "Level",
        "Estimate",
        "SE",
        "z",
        "p-value",
        "exp(Estimate)",
    ];
    let mut rows = vec![header.map(Value::from).to_vec()];
    let parts = [
        (
            "Count",
            fit.count_names(),
            fit.count_coefficients(),
            fit.count_standard_errors(),
        ),
        (
            "Zero",
            fit.zero_names(),
            fit.zero_coefficients(),
            fit.zero_standard_errors(),
        ),
    ];
    for (part, names, b, se) in parts {
        for (j, name) in names.iter().enumerate() {
            let (term, level) = term_level(name);
            let z = b[j] / se[j];
            rows.push(vec![
                Value::from(part),
                term,
                level,
                Value::number(b[j]),
                Value::number(se[j]),
                Value::number(z),
                Value::number(p_value(z)),
                Value::number(b[j].exp()),
            ]);
        }
    }
    let width = header.len();
    rows.push(vec![Value::Empty; width]);
    let mut stats: Vec<(&str, f64)> = Vec::new();
    if let (Some(theta), Some(se)) = (fit.theta(), fit.se_log_theta()) {
        stats.push(("Theta", theta));
        stats.push(("SE log(theta)", se));
    }
    stats.extend([
        ("Log-likelihood", fit.log_likelihood()),
        ("AIC", fit.aic()),
        ("BIC", fit.bic()),
        ("Observations", fit.log_likelihoods().len() as f64),
        ("Parameters", fit.n_parameters() as f64),
    ]);
    for (label, x) in stats {
        let mut row = vec![Value::from(label), Value::number(x)];
        row.resize(width, Value::Empty);
        rows.push(row);
    }
    if let Some((design, y)) = &m.plain {
        rows.push(vec![Value::Empty; width]);
        rows.push(
            ["Vuong against the plain model", "Statistic", "p-value"]
                .map(Value::from)
                .into_iter()
                .chain(std::iter::repeat_n(Value::Empty, width - 3))
                .collect(),
        );
        let (ll, k) = match fit.count_distribution() {
            CountDistribution::Poisson => {
                let g = Glm::new(Family::Poisson, Link::Log)
                    .fit(design, y)
                    .map_err(err)?;
                (
                    count_log_likelihoods(y, g.fitted(), None).map_err(err)?,
                    g.coefficients().len(),
                )
            }
            CountDistribution::NegativeBinomial => {
                let nb = NegativeBinomialGlm::new(Link::Log)
                    .fit(design, y)
                    .map_err(err)?;
                (
                    count_log_likelihoods(y, nb.glm().fitted(), Some(nb.theta())).map_err(err)?,
                    nb.glm().coefficients().len(),
                )
            }
        };
        // pscl counts the coefficients alone, not theta.
        let k_fit = fit.count_coefficients().len() + fit.zero_coefficients().len();
        let v = vuong(fit.log_likelihoods(), &ll, k_fit, k).map_err(err)?;
        for (label, s) in [
            ("Raw", v.raw),
            ("AIC-corrected", v.aic),
            ("BIC-corrected", v.bic),
        ] {
            let mut row = vec![
                Value::from(label),
                Value::number(s.statistic),
                Value::number(s.p_value),
            ];
            row.resize(width, Value::Empty);
            rows.push(row);
        }
    }
    Ok(Grid::from_rows(rows))
}

/// `PRO.PRC.COUNTS.PREDICT(data, response, terms, newdata, [model],
/// [count], [zero_terms], [zero_link], [exposure], [weights], [output],
/// [max_count])`: each row of `newdata`'s expected count (`"mean"`), its
/// probability of a zero (`"zero"`), or the probabilities of 0 to
/// `max_count` under a header of the counts (`"probabilities"`).
pub fn two_part_predict(args: &[Grid]) -> Result<Grid> {
    let model_args = [
        &args[0], &args[1], &args[2], &args[4], &args[5], &args[6], &args[7], &args[8], &args[9],
    ]
    .map(Clone::clone);
    let (m, _) = two_part_fit(&model_args)?;
    let new = Table::new(&args[3], "newdata")?;
    let (c, z) = m.designs(&new, "newdata")?;
    match text_option(&args[10], "output")?.as_deref() {
        None | Some("mean") => Ok(Grid::column(&m.fit.predict_parts(&c, &z).map_err(err)?)),
        Some("zero") => Ok(Grid::column(&m.fit.prob_zero_parts(&c, &z).map_err(err)?)),
        Some("probabilities") => {
            let max = count(&args[11], "max_count", 10, 1000)?;
            let p = m.fit.probabilities_parts(&c, &z, max).map_err(err)?;
            let mut rows = vec![(0..=max).map(|k| Value::number(k as f64)).collect()];
            rows.extend(
                p.into_iter()
                    .map(|r| r.into_iter().map(Value::number).collect()),
            );
            Ok(Grid::from_rows(rows))
        }
        Some(other) => Err(format!(
            "output must be \"mean\", \"zero\" or \"probabilities\", got {other:?}"
        )),
    }
}

/// The `tau` argument: one or more probabilities, as a row or a column.
fn taus(grid: &Grid) -> Result<Vec<f64>> {
    if grid.rows != 1 && grid.columns != 1 {
        return Err("tau must be a number or a row or column of numbers".into());
    }
    let taus = grid
        .cells
        .iter()
        .filter(|v| !v.is_blank())
        .map(|v| match v {
            Value::Number(t) if *t > 0.0 && *t < 1.0 => Ok(*t),
            other => Err(format!(
                "tau must be between 0 and 1, such as 0.9, got {other:?}"
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    if taus.is_empty() {
        return Err("tau is required: the quantile or expectile, such as 0.9".into());
    }
    Ok(taus)
}

/// A fit's predictions on a design.
type Predict = Box<dyn Fn(&Design) -> prospicio_core::Result<Vec<f64>>>;

/// One of a fit's statistics, if it has it.
type Statistic = fn(&LinearFit) -> Option<f64>;

/// One fit at one tau: coefficients, SEs and statistics.
struct LinearFit {
    names: Vec<String>,
    coefficients: Vec<f64>,
    se: Vec<f64>,
    objective: f64,
    iterations: usize,
    bandwidth: Option<f64>,
    fitted: Predict,
}

/// The fits at each tau, with the coding to predict with.
struct LinearModel {
    taus: Vec<f64>,
    fits: Vec<LinearFit>,
    coding: Coding,
    terms: Vec<Predictor>,
}

/// The model arguments `data, response, terms, tau, [method], [weights],
/// [bandwidth]`, one fit per tau, and the coding to predict with.
fn linear_fits(args: &[Grid]) -> Result<LinearModel> {
    let table = Table::new(&args[0], "data")?;
    let response = column_name(&args[1], "response")?
        .ok_or("response is required: the name of the column the model explains")?;
    let terms = predictors(&args[2])?;
    let taus = taus(&args[3])?;
    let expectile = match text_option(&args[4], "method")?.as_deref() {
        None | Some("quantile") => false,
        Some("expectile") => true,
        Some(other) => {
            return Err(format!(
                "method must be \"quantile\" or \"expectile\", got {other:?}"
            ));
        }
    };
    let bandwidth = match text_option(&args[6], "bandwidth")?.as_deref() {
        None | Some("hall_sheather") => Bandwidth::HallSheather,
        Some("bofinger") => Bandwidth::Bofinger,
        Some(other) => {
            return Err(format!(
                "bandwidth must be \"hall_sheather\" or \"bofinger\", got {other:?}"
            ));
        }
    };
    if expectile && !args[6].is_blank() {
        return Err("bandwidth is the quantile regression's kernel standard errors'; leave it empty for expectiles".into());
    }
    let coding = coding(&table, &terms)?;
    let mut d = design(&coding, &table, &terms, None, "data")?;
    if let Some(w) = column_name(&args[5], "weights")? {
        d = d.with_weights(table.numbers(&w, "data")?).map_err(err)?;
    }
    let y = table.numbers(&response, "data")?;
    let fits = taus
        .iter()
        .map(|&tau| -> Result<LinearFit> {
            if expectile {
                let f = ExpectileRegression::new(tau).fit(&d, &y).map_err(err)?;
                Ok(LinearFit {
                    names: f.names().to_vec(),
                    coefficients: f.coefficients().to_vec(),
                    se: f.standard_errors(),
                    objective: f.objective(),
                    iterations: f.iterations(),
                    bandwidth: None,
                    fitted: Box::new(move |d| f.predict(d)),
                })
            } else {
                let f = QuantileRegression::new(tau)
                    .bandwidth(bandwidth)
                    .fit(&d, &y)
                    .map_err(err)?;
                Ok(LinearFit {
                    names: f.names().to_vec(),
                    coefficients: f.coefficients().to_vec(),
                    se: f.standard_errors(),
                    objective: f.objective(),
                    iterations: f.iterations(),
                    bandwidth: Some(f.bandwidth()),
                    fitted: Box::new(move |d| f.predict(d)),
                })
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(LinearModel {
        taus,
        fits,
        coding,
        terms,
    })
}

/// `PRO.PRC.QUANTREG(data, response, terms, tau, [method], [weights],
/// [bandwidth])`: per coefficient its estimate and SE at each tau, then
/// each fit's objective, iterations and (quantiles) bandwidth.
pub fn linear(args: &[Grid]) -> Result<Grid> {
    let LinearModel { taus, fits, .. } = linear_fits(args)?;
    let mut header = vec![Value::from("Term"), Value::from("Level")];
    for t in &taus {
        header.push(Value::Text(format!("Estimate {}", number_text(*t))));
        header.push(Value::Text(format!("SE {}", number_text(*t))));
    }
    let width = header.len();
    let mut rows = vec![header];
    for (j, name) in fits[0].names.iter().enumerate() {
        let (term, level) = term_level(name);
        let mut row = vec![term, level];
        for f in &fits {
            row.push(Value::number(f.coefficients[j]));
            row.push(Value::number(f.se[j]));
        }
        rows.push(row);
    }
    rows.push(vec![Value::Empty; width]);
    let stats: [(&str, Statistic); 3] = [
        ("Objective", |f| Some(f.objective)),
        ("Iterations", |f| Some(f.iterations as f64)),
        ("Bandwidth", |f| f.bandwidth),
    ];
    for (label, get) in stats {
        if fits.iter().all(|f| get(f).is_none()) {
            continue;
        }
        let mut row = vec![Value::from(label), Value::Empty];
        for f in &fits {
            row.push(get(f).map_or(Value::Empty, Value::number));
            row.push(Value::Empty);
        }
        rows.push(row);
    }
    Ok(Grid::from_rows(rows))
}

/// `PRO.PRC.QUANTREG.PREDICT(data, response, terms, tau, newdata,
/// [method], [weights], [bandwidth], [rearrange])`: each row of `newdata`'s
/// fitted quantile or expectile, a column per tau; with `rearrange` TRUE
/// the taus in increasing order and each row sorted, so the quantiles do
/// not cross.
pub fn linear_predict(args: &[Grid]) -> Result<Grid> {
    let model_args = [
        &args[0], &args[1], &args[2], &args[3], &args[5], &args[6], &args[7],
    ]
    .map(Clone::clone);
    let LinearModel {
        taus,
        fits,
        coding,
        terms,
    } = linear_fits(&model_args)?;
    let rearrange = match args[8].as_scalar() {
        Some(v) if v.is_blank() => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(x)) => *x != 0.0,
        _ => return Err("rearrange must be TRUE or FALSE".into()),
    };
    let new = Table::new(&args[4], "newdata")?;
    let d = design(&coding, &new, &terms, None, "newdata")?;
    let mut order: Vec<usize> = (0..taus.len()).collect();
    if rearrange {
        order.sort_by(|&a, &b| taus[a].total_cmp(&taus[b]));
    }
    let columns = order
        .iter()
        .map(|&k| (fits[k].fitted)(&d).map_err(err))
        .collect::<Result<Vec<_>>>()?;
    let rows = (0..d.n_rows())
        .map(|i| {
            let mut row: Vec<f64> = columns.iter().map(|c| c[i]).collect();
            if rearrange {
                row.sort_by(f64::total_cmp);
            }
            row.into_iter().map(Value::number).collect()
        })
        .collect();
    Ok(Grid::from_rows(rows))
}
