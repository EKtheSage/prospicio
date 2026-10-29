//! Generalized linear models on a sheet (`docs/design/xll.md`, decision
//! 41): a fit from a table with a header row, its predictions on new rows
//! and its relativities (`prospicio_glm::Glm` on a
//! `prospicio_models::Terms` coding).
//!
//! Every function takes the same model arguments first, `data, response,
//! terms, [family], [link], [exposure], [weights], [p]`: the table, the
//! response's column name, the predictors' names, and the model. A fit is
//! not kept between calls (decision 4): each function refits from the
//! cells, so a change to the data recalculates everything that uses it.

use prospicio_glm::{Dispersion, Glm, GlmFit};
use prospicio_math::special::norm_quantile;
use prospicio_models::{Coding, Column, Design, Family, Fitted, Frame, Link, Model, Terms};

use crate::reserving::{number_or, text_option};
use crate::sheet::{Grid, Value, position};

type Result<T> = std::result::Result<T, String>;

/// The most rows a table may have: a recalculation should not hang Excel
/// for minutes.
const MAX_ROWS: usize = 1_048_576;

/// A table argument: its column names (the header row) and each column's
/// cells, with blank rows at the end dropped.
struct Table {
    names: Vec<String>,
    columns: Vec<Vec<Value>>,
}

impl Table {
    fn new(grid: &Grid, what: &str) -> Result<Self> {
        if grid.rows < 2 {
            return Err(format!(
                "{what} needs a header row of column names and at least one row of values"
            ));
        }
        let names = (0..grid.columns)
            .map(|c| match grid.get(0, c) {
                Value::Text(s) if !s.trim().is_empty() => Ok(s.trim().to_owned()),
                Value::Number(x) => Ok(number_text(*x)),
                _ => Err(format!(
                    "{what}'s header {} must be a column name",
                    position(0, c)
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some(i) = (1..names.len()).find(|&i| names[..i].contains(&names[i])) {
            return Err(format!("{what} has two columns named {:?}", names[i]));
        }
        // Rows after the last one with a value: a range drawn longer than
        // the data.
        let last = (1..grid.rows)
            .rev()
            .find(|&r| (0..grid.columns).any(|c| !grid.get(r, c).is_blank()))
            .ok_or_else(|| format!("{what} has no rows of values under its header"))?;
        if last > MAX_ROWS {
            return Err(format!("{what} has more than {MAX_ROWS} rows"));
        }
        let columns = (0..grid.columns)
            .map(|c| (1..=last).map(|r| grid.get(r, c).clone()).collect())
            .collect();
        Ok(Self { names, columns })
    }

    fn index(&self, name: &str, what: &str) -> Result<usize> {
        self.names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| format!("{what} has no column named {name:?}"))
    }

    /// The column `name` as numbers: every cell a number.
    fn numbers(&self, name: &str, what: &str) -> Result<Vec<f64>> {
        let j = self.index(name, what)?;
        self.columns[j]
            .iter()
            .enumerate()
            .map(|(r, v)| match v {
                Value::Number(x) => Ok(*x),
                other => Err(format!(
                    "{what}'s column {name:?} must be numbers; row {} is {}",
                    r + 1,
                    shown(other)
                )),
            })
            .collect()
    }

    /// The column `name` as a model column: numbers when every cell is a
    /// number and `factor` is false, else the cells' text.
    fn column(&self, name: &str, factor: bool, what: &str) -> Result<Column> {
        let j = self.index(name, what)?;
        let cells = &self.columns[j];
        if let Some(r) = cells.iter().position(Value::is_blank) {
            return Err(format!(
                "{what}'s column {name:?} is blank in row {}",
                r + 1
            ));
        }
        if !factor && cells.iter().all(|v| matches!(v, Value::Number(_))) {
            return Ok(Column::Numeric(
                cells
                    .iter()
                    .map(|v| match v {
                        Value::Number(x) => *x,
                        _ => unreachable!("checked above"),
                    })
                    .collect(),
            ));
        }
        cells
            .iter()
            .enumerate()
            .map(|(r, v)| match v {
                Value::Number(x) => Ok(number_text(*x)),
                Value::Text(s) => Ok(s.trim().to_owned()),
                Value::Bool(b) => Ok(if *b { "TRUE" } else { "FALSE" }.to_owned()),
                other => Err(format!(
                    "{what}'s column {name:?} row {} is {}",
                    r + 1,
                    shown(other)
                )),
            })
            .collect::<Result<_>>()
            .map(Column::Categorical)
    }
}

/// A number as a factor level or a column name: `3`, not `3.0`.
pub(crate) fn number_text(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        x.to_string()
    }
}

/// A cell as an error message shows it.
fn shown(v: &Value) -> String {
    match v {
        Value::Number(x) => x.to_string(),
        Value::Text(s) => format!("{s:?}"),
        Value::Empty => "a blank".into(),
        Value::Bool(b) => b.to_string().to_uppercase(),
        Value::Error(e) => e.to_string(),
    }
}

/// One predictor: a column, a factor with its reference level, or the
/// interaction of two columns.
enum Predictor {
    /// Numeric when every cell is a number, else a factor whose reference
    /// is the first level in sorted order.
    Auto(String),
    /// A factor with this reference level (`name=level`).
    Factor { name: String, reference: String },
    /// `a:b`.
    Interaction(String, String),
}

impl Predictor {
    fn columns(&self) -> Vec<&str> {
        match self {
            Self::Auto(n) | Self::Factor { name: n, .. } => vec![n],
            Self::Interaction(a, b) => vec![a, b],
        }
    }
}

/// The `terms` argument: a row or column of predictor names, each a
/// column of `data` (`age`), a factor with its reference level
/// (`region=A`) or an interaction (`age:region`).
fn predictors(grid: &Grid) -> Result<Vec<Predictor>> {
    if grid.rows != 1 && grid.columns != 1 {
        return Err("terms must be a row or a column of column names".into());
    }
    let mut terms = Vec::new();
    for (i, v) in grid.cells.iter().enumerate() {
        let s = match v {
            v if v.is_blank() => continue,
            Value::Text(s) => s.trim().to_owned(),
            Value::Number(x) => number_text(*x),
            other => {
                return Err(format!(
                    "terms {} must be a column name, got {}",
                    i + 1,
                    shown(other)
                ));
            }
        };
        let bad = || {
            format!(
                "terms {}: {s:?} must be a column name, name=level (a factor with its base level) or a:b (an interaction)",
                i + 1
            )
        };
        let term = if let Some((name, level)) = s.split_once('=') {
            let (name, level) = (name.trim(), level.trim());
            if name.is_empty() || level.is_empty() || name.contains(':') {
                return Err(bad());
            }
            Predictor::Factor {
                name: name.into(),
                reference: level.into(),
            }
        } else if let Some((a, b)) = s.split_once(':') {
            let (a, b) = (a.trim(), b.trim());
            if a.is_empty() || b.is_empty() || b.contains(':') {
                return Err(bad());
            }
            Predictor::Interaction(a.into(), b.into())
        } else {
            Predictor::Auto(s)
        };
        terms.push(term);
    }
    if terms.is_empty() {
        return Err(
            "terms is required: the predictors' column names, such as {\"age\",\"region\"}".into(),
        );
    }
    Ok(terms)
}

/// The `family` argument and its parameter `p` (the Tweedie power, the
/// negative binomial's theta): Poisson if omitted.
fn family(grid: &Grid, p: &Grid) -> Result<(Family, bool)> {
    let name = text_option(grid, "family")?;
    let needs = |what: &str| -> Result<f64> {
        match p.as_scalar() {
            Some(Value::Number(x)) => Ok(*x),
            _ => Err(format!(
                "family {:?} needs p, {what}",
                name.as_deref().unwrap_or("")
            )),
        }
    };
    if !matches!(
        name.as_deref(),
        Some("tweedie" | "negative_binomial" | "negbin")
    ) && !p.is_blank()
    {
        return Err("p is the tweedie power or the negative_binomial theta; leave it empty for other families".into());
    }
    let quasi = matches!(name.as_deref(), Some("quasipoisson" | "quasi_poisson"));
    let family = match name.as_deref() {
        None | Some("poisson" | "quasipoisson" | "quasi_poisson") => Family::Poisson,
        Some("gamma") => Family::Gamma,
        Some("gaussian" | "normal") => Family::Gaussian,
        Some("inverse_gaussian") => Family::InverseGaussian,
        Some("binomial") => Family::Binomial,
        Some("tweedie") => Family::Tweedie {
            power: needs("the power, such as 1.5")?,
        },
        Some("negative_binomial" | "negbin") => Family::NegativeBinomial {
            theta: needs("theta, such as 2")?,
        },
        Some(other) => {
            return Err(format!(
                "family must be \"poisson\", \"quasipoisson\", \"gamma\", \"tweedie\", \"gaussian\", \"inverse_gaussian\", \"binomial\" or \"negative_binomial\", got {other:?}"
            ));
        }
    };
    Ok((family, quasi))
}

/// The `link` argument: the log link if omitted, except logit for the
/// binomial and identity for the gaussian.
fn link(grid: &Grid, family: Family) -> Result<Link> {
    match text_option(grid, "link")?.as_deref() {
        None => Ok(match family {
            Family::Binomial => Link::Logit,
            Family::Gaussian => Link::Identity,
            _ => Link::Log,
        }),
        Some("log") => Ok(Link::Log),
        Some("identity") => Ok(Link::Identity),
        Some("logit") => Ok(Link::Logit),
        Some("probit") => Ok(Link::Probit),
        Some("cloglog") => Ok(Link::Cloglog),
        Some("inverse") => Ok(Link::Inverse),
        Some("inverse_squared") => Ok(Link::InverseSquared),
        Some(other) => Err(format!(
            "link must be \"log\", \"identity\", \"logit\", \"probit\", \"cloglog\", \"inverse\" or \"inverse_squared\", got {other:?}"
        )),
    }
}

/// A fitted GLM with what it was fitted from.
struct Fit {
    fit: GlmFit,
    coding: Coding,
    predictors: Vec<Predictor>,
    exposure: Option<String>,
}

/// The frame of `table` holding every column the predictors use.
fn frame(table: &Table, predictors: &[Predictor], what: &str) -> Result<Frame> {
    let mut columns: Vec<(String, Column)> = Vec::new();
    // A column is a factor when a term names its reference level, or when
    // an interaction uses it and it holds text.
    let forced: Vec<&str> = predictors
        .iter()
        .filter_map(|p| match p {
            Predictor::Factor { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    for p in predictors {
        for name in p.columns() {
            if columns.iter().any(|(n, _)| n == name) {
                continue;
            }
            let c = table.column(name, forced.contains(&name), what)?;
            columns.push((name.to_owned(), c));
        }
    }
    Frame::new(columns).map_err(|e| e.to_string())
}

/// An optional column-name argument.
fn column_name(grid: &Grid, name: &str) -> Result<Option<String>> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(None),
        Some(Value::Text(s)) => Ok(Some(s.trim().to_owned())),
        _ => Err(format!("{name} must be a column name of data")),
    }
}

/// The design of `table` under `coding`, with ln(exposure) as offset
/// when an exposure column is named.
fn design(
    coding: &Coding,
    table: &Table,
    predictors: &[Predictor],
    exposure: Option<&str>,
    what: &str,
) -> Result<Design> {
    let d = coding
        .design(&frame(table, predictors, what)?)
        .map_err(|e| e.to_string())?;
    match exposure {
        None => Ok(d),
        Some(name) => {
            let e = table.numbers(name, what)?;
            if let Some(r) = e.iter().position(|&x| !(x > 0.0 && x.is_finite())) {
                return Err(format!(
                    "{what}'s exposure column {name:?} must be positive; row {} is {}",
                    r + 1,
                    e[r]
                ));
            }
            d.with_offset(e.iter().map(|x| x.ln()).collect())
                .map_err(|e| e.to_string())
        }
    }
}

/// The model arguments at `args[0..8]`: `data, response, terms, [family],
/// [link], [exposure], [weights], [p]`, fitted.
fn fit(args: &[Grid]) -> Result<Fit> {
    let table = Table::new(&args[0], "data")?;
    let response = column_name(&args[1], "response")?
        .ok_or("response is required: the name of the column the model explains")?;
    let predictors = predictors(&args[2])?;
    let (family, quasi) = family(&args[3], &args[7])?;
    let link = link(&args[4], family)?;
    let exposure = column_name(&args[5], "exposure")?;
    let weights = column_name(&args[6], "weights")?;
    if exposure.is_some() && link != Link::Log {
        return Err("exposure is an offset ln(exposure) on the log link: use link \"log\", or give the exposure as weights".into());
    }
    let mut terms = Terms::new().intercept();
    for p in &predictors {
        terms = match p {
            Predictor::Auto(n) => match table.column(n, false, "data")? {
                Column::Numeric(_) => terms.numeric(n),
                Column::Categorical(_) => terms.factor(n),
            },
            Predictor::Factor { name, reference } => terms.factor_with_reference(name, reference),
            Predictor::Interaction(a, b) => terms.interaction(a, b),
        };
    }
    let coding = terms
        .fit(&frame(&table, &predictors, "data")?)
        .map_err(|e| e.to_string())?;
    let mut d = design(&coding, &table, &predictors, exposure.as_deref(), "data")?;
    if let Some(w) = &weights {
        d = d
            .with_weights(table.numbers(w, "data")?)
            .map_err(|e| e.to_string())?;
    }
    let y = table.numbers(&response, "data")?;
    let mut glm = Glm::new(family, link);
    if quasi {
        glm = glm.dispersion(Dispersion::Pearson);
    }
    let fit = glm.fit(&d, &y).map_err(|e| e.to_string())?;
    Ok(Fit {
        fit,
        coding,
        predictors,
        exposure,
    })
}

/// A design column's term and level: `region[B]` is `region`, `B`; `age`
/// is `age` and blank; an interaction such as `age:region[B]` is the
/// whole name and blank.
fn term_level(name: &str) -> (Value, Value) {
    match name.split_once('[') {
        Some((term, rest)) if !name.contains(':') => (
            Value::from(term),
            Value::from(rest.strip_suffix(']').unwrap_or(rest)),
        ),
        _ => (Value::from(name), Value::Empty),
    }
}

/// `PRO.GLM(data, response, terms, [family], [link], [exposure],
/// [weights], [p])`: per coefficient its term, level, estimate, standard
/// error, statistic, p-value and exp(estimate), then the fit's statistics
/// as label and value.
pub fn summary(args: &[Grid]) -> Result<Grid> {
    let m = fit(args)?;
    let fit = &m.fit;
    let mut rows = vec![
        [
            "Term",
            "Level",
            "Estimate",
            "SE",
            "Statistic",
            "p-value",
            "exp(Estimate)",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    let (se, p) = (fit.std_errors(), fit.p_values());
    for (j, name) in fit.names().iter().enumerate() {
        let b = fit.coefficients()[j];
        let (term, level) = term_level(name);
        rows.push(vec![
            term,
            level,
            Value::number(b),
            Value::number(se[j]),
            Value::number(b / se[j]),
            Value::number(p[j]),
            Value::number(b.exp()),
        ]);
    }
    let width = rows[0].len();
    rows.push(vec![Value::Empty; width]);
    for (label, x) in [
        ("Deviance", fit.deviance()),
        ("Null deviance", fit.null_deviance()),
        ("Dispersion", fit.dispersion()),
        ("Log-likelihood", fit.log_likelihood()),
        ("AIC", fit.aic()),
        ("BIC", fit.bic()),
        ("Observations", fit.n_obs() as f64),
        ("Residual df", fit.df_resid()),
        ("Iterations", fit.iterations() as f64),
    ] {
        let mut row = vec![Value::from(label), Value::number(x)];
        row.resize(width, Value::Empty);
        rows.push(row);
    }
    Ok(Grid::from_rows(rows))
}

/// `PRO.GLM.PREDICT(data, response, terms, newdata, [family], [link],
/// [exposure], [weights], [p])`: the fitted mean of each row of `newdata`,
/// a table with the same column names (its exposure too, when one is
/// named), as a column.
pub fn predict(args: &[Grid]) -> Result<Grid> {
    let model_args = [
        &args[0], &args[1], &args[2], &args[4], &args[5], &args[6], &args[7], &args[8],
    ]
    .map(Clone::clone);
    let m = fit(&model_args)?;
    let new = Table::new(&args[3], "newdata")?;
    let d = design(
        &m.coding,
        &new,
        &m.predictors,
        m.exposure.as_deref(),
        "newdata",
    )?;
    let mean = m.fit.predict(&d).map_err(|e| e.to_string())?;
    Ok(Grid::column(&mean))
}

/// `PRO.GLM.RELATIVITIES(data, response, terms, [family], [link],
/// [exposure], [weights], [p], [level])`: on the log link, each term's
/// relativity exp(estimate) with its confidence interval at `level` (95%
/// if omitted), every factor's base level first at 1, and a numeric term's
/// relativity per unit.
pub fn relativities(args: &[Grid]) -> Result<Grid> {
    let m = fit(args)?;
    if m.fit.spec().link != Link::Log {
        return Err("relativities are exp(estimate) on the log link: use link \"log\"".into());
    }
    let level = number_or(&args[8], "level", 0.95)?;
    if !(level > 0.0 && level < 1.0) {
        return Err(format!("level must be between 0 and 1, got {level}"));
    }
    let z = norm_quantile(0.5 + level / 2.0);
    let fit = &m.fit;
    let se = fit.std_errors();
    let mut rows = vec![
        [
            "Term",
            "Level",
            "Relativity",
            &format!("Lower {}", percent(level)),
            &format!("Upper {}", percent(level)),
        ]
        .map(Value::from)
        .to_vec(),
    ];
    let references = m.coding.references();
    let mut based: Vec<String> = Vec::new();
    for (j, name) in fit.names().iter().enumerate() {
        if name == "(Intercept)" {
            continue;
        }
        let (term, level) = term_level(name);
        if let (Value::Text(t), Value::Text(_)) = (&term, &level) {
            if !based.contains(t) {
                if let Some((_, base)) = references.iter().find(|(n, _)| n == t) {
                    rows.push(vec![
                        term.clone(),
                        Value::Text(format!("{base} (base)")),
                        Value::Number(1.0),
                        Value::Empty,
                        Value::Empty,
                    ]);
                }
                based.push(t.clone());
            }
        }
        let b = fit.coefficients()[j];
        rows.push(vec![
            term,
            match level {
                Value::Empty if name.contains(':') => Value::from("interaction"),
                Value::Empty => Value::from("per unit"),
                l => l,
            },
            Value::number(b.exp()),
            Value::number((b - z * se[j]).exp()),
            Value::number((b + z * se[j]).exp()),
        ]);
    }
    Ok(Grid::from_rows(rows))
}

/// A probability as a percent label: `95%`, `99.5%`.
fn percent(p: f64) -> String {
    let x = (p * 1000.0).round() / 10.0;
    format!("{x}%")
}
