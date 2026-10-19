//! Extreme value theory for large losses (`docs/design/xll.md`, decision
//! 26): the threshold diagnostics that choose where a generalized Pareto
//! tail starts (R `ismev`'s `mrl.plot` and `gpd.fitrange`), the
//! peaks-over-threshold fit there with its VaR and TVaR, and Hill's
//! estimator of the tail index.

use prospicio_prob::evt::{Gpd, PotTail, hill, threshold_diagnostics};

use crate::pricing::{numbers, scalar};
use crate::sheet::{Grid, Value};

type Result<T> = std::result::Result<T, String>;

/// The data, sorted, with at least `min` values.
fn sorted(grid: &Grid, min: usize) -> Result<Vec<f64>> {
    let mut x = numbers(grid, "data")?;
    if x.len() < min {
        return Err(format!("data needs at least {min} values, got {}", x.len()));
    }
    x.sort_by(f64::total_cmp);
    Ok(x)
}

/// The empirical quantile of sorted `x` at `p`, interpolated between order
/// statistics (R's default, type 7).
fn quantile(x: &[f64], p: f64) -> f64 {
    let h = p * (x.len() - 1) as f64;
    let (lo, frac) = (h.floor() as usize, h.fract());
    if lo + 1 >= x.len() {
        x[x.len() - 1]
    } else {
        x[lo] + frac * (x[lo + 1] - x[lo])
    }
}

/// `PRO.EVT.THRESHOLDS(data, [thresholds])`: at each threshold the number
/// of values above it, the mean excess with its standard error, and the
/// GPD fitted to the exceedances (shape, scale and the modified scale
/// `beta - xi u`, with standard errors). Above a threshold where a GPD
/// fits, the mean excess is linear and the shape and modified scale are
/// constant. Thresholds default to the data's 50%, 55%, ..., 95%
/// quantiles.
pub fn thresholds(args: &[Grid]) -> Result<Grid> {
    let x = sorted(&args[0], 3)?;
    let us = if args[1].is_blank() {
        (0..10)
            .map(|i| quantile(&x, 0.5 + 0.05 * i as f64))
            .collect()
    } else {
        numbers(&args[1], "thresholds")?
    };
    let mut rows = vec![
        [
            "Threshold",
            "Above",
            "Mean excess",
            "Mean excess SE",
            "xi",
            "xi SE",
            "beta",
            "Modified scale",
            "Modified scale SE",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for d in threshold_diagnostics(&x, &us) {
        rows.push(
            [
                d.threshold,
                d.n_above as f64,
                d.mean_excess,
                d.mean_excess_se,
                d.xi,
                d.xi_se,
                d.beta,
                d.modified_scale,
                d.modified_scale_se,
            ]
            .map(Value::number)
            .to_vec(),
        );
    }
    Ok(Grid::from_rows(rows))
}

/// The tail probabilities `PRO.EVT.FIT` reports if none are given.
const LEVELS: [f64; 3] = [0.99, 0.995, 0.999];

/// `PRO.EVT.FIT(data, threshold, [probabilities])`: the GPD fitted to the
/// exceedances over `threshold` by maximum likelihood, with standard
/// errors, and the tail's VaR and TVaR at each probability (0.99, 0.995
/// and 0.999 if omitted). The shape and scale are `PRO.SEV`'s `"gpd"`
/// family for the exceedances.
pub fn fit(args: &[Grid]) -> Result<Grid> {
    let x = sorted(&args[0], 3)?;
    let u = scalar(&args[1], "threshold")?;
    let y: Vec<f64> = x.iter().filter(|&&v| v > u).map(|&v| v - u).collect();
    if y.len() < 3 {
        return Err(format!(
            "threshold {u} leaves {} values above it; a GPD needs at least 3",
            y.len()
        ));
    }
    let levels = if args[2].is_blank() {
        LEVELS.to_vec()
    } else {
        crate::reserving::percentiles(&args[2])
            .map_err(|_| "probabilities must be between 0 and 1, such as 0.995".to_owned())?
    };
    let e = |e: prospicio_core::Error| e.to_string();
    let (g, cov): (Gpd, _) = Gpd::fit_with_covariance(&y).map_err(e)?;
    let p_exceed = y.len() as f64 / x.len() as f64;
    let tail = PotTail::new(u, p_exceed, g).map_err(e)?;
    let row = |label: String, v: f64, se: f64| {
        vec![Value::Text(label), Value::number(v), Value::number(se)]
    };
    let mut rows = vec![
        ["Quantity", "Value", "SE"].map(Value::from).to_vec(),
        row("Threshold".into(), u, f64::NAN),
        row("Exceedances".into(), y.len() as f64, f64::NAN),
        row("P(exceed)".into(), p_exceed, f64::NAN),
        row("xi".into(), g.xi(), cov[0][0].sqrt()),
        row("beta".into(), g.beta(), cov[1][1].sqrt()),
    ];
    for (name, f) in [
        ("VaR", PotTail::var as fn(&PotTail, f64) -> _),
        ("TVaR", PotTail::tvar),
    ] {
        for &p in &levels {
            let label = format!("{name} {p}");
            rows.push(match f(&tail, p) {
                Ok(v) => row(label, v, f64::NAN),
                Err(_) => vec![
                    Value::Text(label),
                    Value::Text(format!(
                        "#PROSPICIO! below the tail: {p} is under 1 - P(exceed) = {}",
                        1.0 - p_exceed
                    )),
                    Value::Empty,
                ],
            });
        }
    }
    Ok(Grid::from_rows(rows))
}

/// `PRO.EVT.HILL(data, [k])`: Hill's estimate of the tail index from the
/// `k` largest values, for each `k` (ten values from 5% to 50% of the
/// data if omitted), with the threshold it sits on (the `k + 1`-th
/// largest value) and the Pareto alpha `1 / xi`.
pub fn hill_estimates(args: &[Grid]) -> Result<Grid> {
    let x = sorted(&args[0], 2)?;
    let n = x.len();
    let ks: Vec<usize> = if args[1].is_blank() {
        let mut ks: Vec<usize> = (1..=10)
            .map(|i| ((0.05 * i as f64 * n as f64).round() as usize).clamp(1, n - 1))
            .collect();
        ks.dedup();
        ks
    } else {
        numbers(&args[1], "k")?
            .into_iter()
            .map(|k| {
                if k.fract() == 0.0 && k >= 1.0 && (k as usize) < n {
                    Ok(k as usize)
                } else {
                    Err(format!(
                        "k must be whole numbers from 1 to {}, got {k}",
                        n - 1
                    ))
                }
            })
            .collect::<Result<_>>()?
    };
    let xi = hill(&x, &ks).map_err(|e| e.to_string())?;
    let mut rows = vec![["k", "Threshold", "xi", "alpha"].map(Value::from).to_vec()];
    for (&k, &xi) in ks.iter().zip(&xi) {
        rows.push(
            [k as f64, x[n - 1 - k], xi, 1.0 / xi]
                .map(Value::number)
                .to_vec(),
        );
    }
    Ok(Grid::from_rows(rows))
}
