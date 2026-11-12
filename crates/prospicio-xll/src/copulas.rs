//! Copulas on a sheet (`docs/design/xll.md`, decisions 32 and 64): a
//! copula's distribution function at points, uniforms simulated from one,
//! nested Archimedean copulas given as a table of nodes, and Iman–Conover's
//! reordering of simulated columns to a rank correlation
//! (`prospicio_prob::copula`). Each call builds its copula from the
//! arguments; nothing is kept between calls.

use prospicio_core::StreamRng;
use prospicio_prob::copula::{
    Archimedean, ArchimedeanCopula, Copula, GaussianCopula, ImanConover, NestedArchimedeanCopula,
    NestedNode, Scores, StudentTCopula, iman_conover_with,
};
use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};

use crate::pricing::scalar;
use crate::reserving::{count, seed, text_option};
use crate::sheet::{Grid, Value, position};

type Result<T> = std::result::Result<T, String>;

const NAMES: &str = "\"gaussian\", \"t\", \"clayton\", \"gumbel\", \"frank\" or \"joe\"";

/// The most rows a call simulates or reorders, under Excel's 1,048,576.
const MAX_ROWS: usize = 1_000_000;

/// The numbers of a range, row by row, every cell a number.
fn matrix(grid: &Grid, name: &str) -> Result<Vec<f64>> {
    grid.cells
        .iter()
        .enumerate()
        .map(|(k, v)| match v {
            Value::Number(x) => Ok(*x),
            _ => Err(format!(
                "{name} must be all numbers; {} is not",
                position(k / grid.columns, k % grid.columns)
            )),
        })
        .collect()
}

/// A correlation matrix from `parameter`: a `d × d` range, or a single
/// correlation for every pair of `dim` dimensions (2 when omitted).
fn correlation(parameter: &Grid, dim: Option<usize>) -> Result<(Vec<f64>, usize)> {
    if let Some(Value::Number(rho)) = parameter.as_scalar() {
        let d = dim.unwrap_or(2);
        let r = (0..d * d)
            .map(|k| if k / d == k % d { 1.0 } else { *rho })
            .collect();
        return Ok((r, d));
    }
    if parameter.rows != parameter.columns || parameter.rows < 2 {
        return Err(format!(
            "parameter must be a correlation or a square correlation matrix, got {} x {}",
            parameter.rows, parameter.columns
        ));
    }
    let d = parameter.rows;
    if let Some(dim) = dim
        && dim != d
    {
        return Err(format!(
            "dim is {dim} but the correlation matrix is {d} x {d}"
        ));
    }
    Ok((matrix(parameter, "the correlation matrix")?, d))
}

/// A copula by name: the Gaussian and t take a correlation (or matrix),
/// the t also `nu`; the Archimedean families take `theta` and `dim`.
fn copula(name: &Grid, parameter: &Grid, nu: &Grid, dim: Option<usize>) -> Result<Box<dyn Copula>> {
    let name = text_option(name, "family")?.ok_or_else(|| format!("family must be {NAMES}"))?;
    let no_nu = || {
        if nu.is_blank() {
            Ok(())
        } else {
            Err(format!("nu is taken only by the t copula, not {name:?}"))
        }
    };
    let archimedean = |family: Archimedean| -> Result<Box<dyn Copula>> {
        no_nu()?;
        let theta = scalar(parameter, "theta")?;
        Ok(Box::new(
            ArchimedeanCopula::new(family, theta, dim.unwrap_or(2)).map_err(|e| e.to_string())?,
        ))
    };
    match name.replace([' ', '-'], "_").as_str() {
        "gaussian" | "normal" => {
            no_nu()?;
            let (r, d) = correlation(parameter, dim)?;
            Ok(Box::new(
                GaussianCopula::new(&r, d).map_err(|e| e.to_string())?,
            ))
        }
        "t" | "student_t" => {
            let (r, d) = correlation(parameter, dim)?;
            let nu = scalar(nu, "nu")?;
            Ok(Box::new(
                StudentTCopula::new(&r, d, nu).map_err(|e| e.to_string())?,
            ))
        }
        "clayton" => archimedean(Archimedean::Clayton),
        "gumbel" => archimedean(Archimedean::Gumbel),
        "frank" => archimedean(Archimedean::Frank),
        "joe" => archimedean(Archimedean::Joe),
        other => Err(format!("family must be {NAMES}, got {other:?}")),
    }
}

/// `PRO.CAP.COPULA.CDF(points, family, parameter, [nu])`: `C(u)` at each
/// point, one point a row of `points` (its columns the dimensions). The
/// Gaussian's is exact in two dimensions and by Genz's quasi-Monte Carlo
/// method (about `1e-5`) in more; the t has none in closed form.
pub fn cdf(args: &[Grid]) -> Result<Grid> {
    let points = &args[0];
    let u = matrix(points, "points")?;
    let c = copula(&args[1], &args[2], &args[3], Some(points.columns))?;
    let values = u
        .chunks(points.columns)
        .enumerate()
        .map(|(i, point)| {
            c.cdf(point)
                .map(Value::number)
                .map_err(|e| format!("point {}: {e}", i + 1))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Grid::from_rows(
        values.into_iter().map(|v| vec![v]).collect(),
    ))
}

/// `PRO.CAP.COPULA.SIM(family, parameter, sims, [dim], [seed], [nu])`:
/// `sims` rows of uniforms in `(0, 1)`, one column a dimension. Row `i`
/// is drawn from stream `i` of `seed`, as `prospicio_prob::copula::simulate`
/// draws it, so a sheet and the Python and R packages agree; turn the
/// uniforms into losses with `PRO.SEV.QUANTILE`.
pub fn sim(args: &[Grid]) -> Result<Grid> {
    let sims = count(&args[2], "sims", 10_000, MAX_ROWS)?;
    let dim = if args[3].is_blank() {
        None
    } else {
        Some(count(&args[3], "dim", 2, 64)?)
    };
    let c = copula(&args[0], &args[1], &args[5], dim)?;
    let seed = seed(&args[4])?;
    let d = c.dim();
    let mut u = vec![0.0; d];
    let rows = (0..sims)
        .map(|i| {
            c.sample(&mut StreamRng::new(seed, i as u64), &mut u);
            u.iter().map(|&x| Value::Number(x)).collect()
        })
        .collect();
    Ok(Grid::from_rows(rows))
}

/// A node's name: a number or a text, as typed.
fn label(v: &Value) -> Option<String> {
    match v {
        Value::Number(x) => Some(format!("{x}")),
        Value::Text(t) if !t.trim().is_empty() => Some(t.trim().to_owned()),
        _ => None,
    }
}

/// The nested copula of `family` (Clayton or Gumbel) given by `nodes`, a
/// table of three columns (node, parent, theta; the root's parent blank),
/// and `members`, a row or column with the node each dimension joins.
/// A node's children are taken in the table's order.
fn nested(family: &Grid, nodes: &Grid, members: &Grid) -> Result<NestedArchimedeanCopula> {
    let family = match text_option(family, "family")?
        .map(|f| f.to_lowercase())
        .as_deref()
    {
        Some("clayton") => Archimedean::Clayton,
        Some("gumbel") => Archimedean::Gumbel,
        _ => return Err("family must be \"clayton\" or \"gumbel\"".into()),
    };
    if nodes.columns != 3 {
        return Err(format!(
            "nodes must have three columns (node, parent, theta), got {}",
            nodes.columns
        ));
    }
    let mut rows: Vec<(String, Option<String>, f64)> = Vec::with_capacity(nodes.rows);
    for i in 0..nodes.rows {
        let name = label(nodes.get(i, 0)).ok_or_else(|| {
            format!(
                "nodes {}: the node must be a name or a number",
                position(i, 0)
            )
        })?;
        if rows.iter().any(|r| r.0 == name) {
            return Err(format!("node {name} appears twice in nodes"));
        }
        let parent = match nodes.get(i, 1) {
            v if v.is_blank() => None,
            v => Some(label(v).ok_or_else(|| {
                format!(
                    "nodes {}: the parent must be a name, a number or blank",
                    position(i, 1)
                )
            })?),
        };
        let theta = match nodes.get(i, 2) {
            Value::Number(x) => *x,
            _ => return Err(format!("nodes {}: theta must be a number", position(i, 2))),
        };
        rows.push((name, parent, theta));
    }
    let roots: Vec<&str> = rows
        .iter()
        .filter(|r| r.1.is_none())
        .map(|r| r.0.as_str())
        .collect();
    let root = match roots[..] {
        [root] => root.to_owned(),
        [] => return Err("nodes must have one root, a node whose parent is blank".into()),
        _ => {
            return Err(format!(
                "nodes has {} roots; leave only one parent blank",
                roots.len()
            ));
        }
    };
    for (name, parent, _) in &rows {
        if let Some(p) = parent
            && !rows.iter().any(|r| &r.0 == p)
        {
            return Err(format!("node {name}'s parent {p} is not in nodes"));
        }
    }
    if members.rows != 1 && members.columns != 1 {
        return Err("members must be a row or a column, one cell a dimension".into());
    }
    let mut leaves: Vec<Vec<usize>> = vec![Vec::new(); rows.len()];
    for (j, v) in members.cells.iter().enumerate() {
        let name =
            label(v).ok_or_else(|| format!("members: dimension {} must name a node", j + 1))?;
        let k = rows.iter().position(|r| r.0 == name).ok_or_else(|| {
            format!(
                "members: dimension {} joins node {name}, which is not in nodes",
                j + 1
            )
        })?;
        leaves[k].push(j);
    }
    let mut visited = 0;
    let tree = build(&rows, &leaves, &root, &mut visited)?;
    if visited != rows.len() {
        return Err("every node must lie below the root; a parent chain loops".into());
    }
    NestedArchimedeanCopula::new(family, tree).map_err(|e| e.to_string())
}

fn build(
    rows: &[(String, Option<String>, f64)],
    leaves: &[Vec<usize>],
    name: &str,
    visited: &mut usize,
) -> Result<NestedNode> {
    *visited += 1;
    let k = rows.iter().position(|r| r.0 == name).expect("a known node");
    let children = rows
        .iter()
        .filter(|r| r.1.as_deref() == Some(name))
        .map(|r| build(rows, leaves, &r.0, visited))
        .collect::<Result<Vec<_>>>()?;
    let joined = leaves[k].len() + children.len();
    if joined < 2 {
        return Err(format!(
            "node {name} joins {joined} member; each node joins at least two dimensions or nodes"
        ));
    }
    Ok(NestedNode::new(rows[k].2, leaves[k].clone(), children))
}

/// `PRO.CAP.NESTED.CDF(points, family, nodes, members)`: a nested
/// Archimedean copula's `C(u)` at each point, one point a row of
/// `points`, one column a dimension in `members`' order.
pub fn nested_cdf(args: &[Grid]) -> Result<Grid> {
    let points = &args[0];
    let c = nested(&args[1], &args[2], &args[3])?;
    if points.columns != c.dim() {
        return Err(format!(
            "points has {} columns but members has {} dimensions",
            points.columns,
            c.dim()
        ));
    }
    let u = matrix(points, "points")?;
    let values = u
        .chunks(points.columns)
        .enumerate()
        .map(|(i, point)| {
            c.cdf(point)
                .map(Value::number)
                .map_err(|e| format!("point {}: {e}", i + 1))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Grid::from_rows(
        values.into_iter().map(|v| vec![v]).collect(),
    ))
}

/// `PRO.CAP.NESTED.SIM(family, nodes, members, [sims], [seed])`: `sims`
/// rows of uniforms from a nested Archimedean copula, one column a
/// dimension in `members`' order; row `i` from stream `i` of `seed`, as
/// `PRO.CAP.COPULA.SIM` and `prospicio_prob::copula::simulate` draw it.
pub fn nested_sim(args: &[Grid]) -> Result<Grid> {
    let c = nested(&args[0], &args[1], &args[2])?;
    let sims = count(&args[3], "sims", 10_000, MAX_ROWS)?;
    let seed = seed(&args[4])?;
    let mut u = vec![0.0; c.dim()];
    let rows = (0..sims)
        .map(|i| {
            c.sample(&mut StreamRng::new(seed, i as u64), &mut u);
            u.iter().map(|&x| Value::Number(x)).collect()
        })
        .collect();
    Ok(Grid::from_rows(rows))
}

/// `PRO.CAP.IMAN.CONOVER(draws, correlation, [seed], [dof], [shuffle])`: the
/// columns of `draws` (one a variable, one row a simulation) reordered so
/// that their rank correlation is close to `correlation`, each column
/// keeping its own values. Normal scores, or Student t scores with `dof`
/// degrees of freedom for more joint extremes; `shuffle` TRUE shuffles
/// the rows of the result.
pub fn iman_conover(args: &[Grid]) -> Result<Grid> {
    let draws = &args[0];
    let m = draws.columns;
    if m < 2 {
        return Err("draws must have at least two columns, one a variable".into());
    }
    if draws.rows > MAX_ROWS {
        return Err(format!("draws may have at most {MAX_ROWS} rows"));
    }
    let x = matrix(draws, "draws")?;
    let (r, _) = correlation(&args[1], Some(m))?;
    let scores = if args[3].is_blank() {
        Scores::Normal
    } else {
        Scores::StudentT(scalar(&args[3], "dof")?)
    };
    let shuffle_rows = match args[4].as_scalar() {
        Some(v) if v.is_blank() => false,
        Some(Value::Bool(b)) => *b,
        _ => return Err("shuffle must be TRUE or FALSE".into()),
    };
    let pd = PredictiveDistribution::from_draws(
        vec!["column".into()],
        (0..m).map(|j| vec![KeyValue::Int(j as i64)]).collect(),
        x,
        Provenance::new("PRO.CAP.IMAN.CONOVER"),
    )
    .map_err(|e| e.to_string())?;
    let options = ImanConover {
        scores,
        shuffle_rows,
    };
    let joined =
        iman_conover_with(&pd, &r, seed(&args[2])?, &options).map_err(|e| e.to_string())?;
    let rows = (0..joined.n_sims())
        .map(|i| {
            joined
                .row(i)
                .expect("a row per simulation")
                .iter()
                .map(|&v| Value::number(v))
                .collect()
        })
        .collect();
    Ok(Grid::from_rows(rows))
}
