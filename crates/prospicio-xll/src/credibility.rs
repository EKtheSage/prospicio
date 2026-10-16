//! Classical credibility (`docs/design/xll.md`): the limited fluctuation
//! standards and the square root rule, and the Bühlmann–Straub model with
//! its credibility premiums, over `prospicio_pricing::credibility`.

use prospicio_pricing::credibility::{BuhlmannStraub, LimitedFluctuation, StructureEstimator};

use crate::exposure::broadcast;
use crate::pricing::{Result, scalar};
use crate::sheet::{Grid, Value, position};

/// The full credibility standard in expected claims for probability `p`
/// of lying within `k` of the mean: `PRO.CRED.STANDARD(p, k, [kind],
/// [cv])`, `kind` `"frequency"` (default), `"severity"` or
/// `"pure_premium"`, the last two with the severity's `cv`. `p` and `k`
/// broadcast, so a column of `p` against a row of `k` is the textbook
/// table.
pub fn standard(args: &[Grid]) -> Result<Grid> {
    let kind = match args[2].as_scalar() {
        Some(v) if v.is_blank() => "frequency".to_owned(),
        Some(Value::Text(s)) => s.trim().to_ascii_lowercase().replace([' ', '-'], "_"),
        _ => return Err(KINDS.into()),
    };
    let cv = if args[3].is_blank() {
        None
    } else {
        Some(scalar(&args[3], "cv")?)
    };
    let cv_for = |what: &str| cv.ok_or(format!("a {what} standard needs the severity's cv"));
    let kind = match kind.as_str() {
        "frequency" => {
            if cv.is_some() {
                return Err("cv is taken only by the severity and pure premium standards".into());
            }
            Kind::Frequency
        }
        "severity" => Kind::Severity(cv_for("severity")?),
        "pure_premium" => Kind::PurePremium(cv_for("pure premium")?),
        _ => return Err(KINDS.into()),
    };
    broadcast(&[&args[0], &args[1]], &["p", "k"], &[false, false], |v| {
        let lf = LimitedFluctuation::new(v[0], v[1]).map_err(|e| e.to_string())?;
        Ok(match kind {
            Kind::Frequency => lf.frequency_standard(),
            Kind::Severity(cv) => lf.severity_standard(cv),
            Kind::PurePremium(cv) => lf.pure_premium_standard(cv),
        })
    })
}

const KINDS: &str = "kind must be \"frequency\", \"severity\" or \"pure_premium\"";

#[derive(Clone, Copy)]
enum Kind {
    Frequency,
    Severity(f64),
    PurePremium(f64),
}

/// Partial credibility by the square root rule, `min(1, sqrt(n /
/// standard))`: `PRO.CRED.Z(n, standard)`, broadcast.
pub fn z(args: &[Grid]) -> Result<Grid> {
    broadcast(
        &[&args[0], &args[1]],
        &["n", "standard"],
        &[false, false],
        |v| {
            if v[1] > 0.0 {
                // The rule does not depend on p and k.
                let lf = LimitedFluctuation::new(0.9, 0.05).map_err(|e| e.to_string())?;
                Ok(lf.credibility(v[0], v[1]))
            } else {
                Err(format!("standard must be positive, got {}", v[1]))
            }
        },
    )
}

/// A range as rows of numbers, a blank cell as NaN (missing).
fn matrix(g: &Grid, name: &str) -> Result<Vec<Vec<f64>>> {
    (0..g.rows)
        .map(|r| {
            (0..g.columns)
                .map(|c| match g.get(r, c) {
                    Value::Number(x) => Ok(*x),
                    v if v.is_blank() => Ok(f64::NAN),
                    other => Err(format!(
                        "{name} {} must be a number or blank, got {other:?}",
                        position(r, c)
                    )),
                })
                .collect()
        })
        .collect()
}

fn fit(args: &[Grid]) -> Result<prospicio_pricing::credibility::BuhlmannStraubFit> {
    let ratios = matrix(&args[0], "ratios")?;
    let weights = if args[1].is_blank() {
        None
    } else {
        if (args[1].rows, args[1].columns) != (args[0].rows, args[0].columns) {
            return Err(format!(
                "weights ({} x {}) must have the shape of ratios ({} x {})",
                args[1].rows, args[1].columns, args[0].rows, args[0].columns
            ));
        }
        Some(matrix(&args[1], "weights")?)
    };
    let estimator = match args[2].as_scalar() {
        Some(v) if v.is_blank() => StructureEstimator::Unbiased,
        Some(Value::Text(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "unbiased" => StructureEstimator::Unbiased,
            "iterative" => StructureEstimator::iterative(),
            other => {
                return Err(format!(
                    "estimator must be \"unbiased\" or \"iterative\", got {other:?}"
                ));
            }
        },
        _ => return Err("estimator must be \"unbiased\" or \"iterative\"".into()),
    };
    BuhlmannStraub::new(estimator)
        .fit(&ratios, weights.as_deref())
        .map_err(|e| e.to_string())
}

/// The Bühlmann–Straub model's exhibit, one row per group (a row of
/// `ratios`): its weight, weighted mean, credibility factor and
/// credibility premium: `PRO.BUHLMANN(ratios, [weights], [estimator],
/// [groups])`. Without weights it is Bühlmann's model.
pub fn buhlmann(args: &[Grid]) -> Result<Grid> {
    let f = fit(args)?;
    let n = f.premiums.len();
    let labels: Vec<Value> = if args[3].is_blank() {
        (1..=n).map(|i| Value::Number(i as f64)).collect()
    } else if args[3].cells.len() == n {
        args[3].cells.clone()
    } else {
        return Err(format!(
            "groups must have one label per row of ratios ({n}), got {}",
            args[3].cells.len()
        ));
    };
    let mut rows = vec![
        ["Group", "Weight", "Mean", "Credibility", "Premium"]
            .map(Value::from)
            .to_vec(),
    ];
    for (i, label) in labels.into_iter().enumerate() {
        rows.push(vec![
            label,
            Value::number(f.group_weights[i]),
            Value::number(f.group_means[i]),
            Value::number(f.credibility[i]),
            Value::number(f.premiums[i]),
        ]);
    }
    Ok(Grid::from_rows(rows))
}

/// The Bühlmann–Straub structure parameters as a two-column table: the
/// collective mean, the between and within variances, `k = s² / a` (blank
/// when `a <= 0`) and the iterations:
/// `PRO.BUHLMANN.PARAMS(ratios, [weights], [estimator])`.
pub fn buhlmann_params(args: &[Grid]) -> Result<Grid> {
    let f = fit(args)?;
    Ok(Grid::from_rows(vec![
        vec![
            Value::from("Collective mean"),
            Value::number(f.collective_mean),
        ],
        vec![
            Value::from("Between variance"),
            Value::number(f.between_variance),
        ],
        vec![
            Value::from("Within variance"),
            Value::number(f.within_variance),
        ],
        vec![Value::from("k"), Value::number(f.k())],
        vec![
            Value::from("Iterations"),
            Value::Number(f.iterations as f64),
        ],
    ]))
}
