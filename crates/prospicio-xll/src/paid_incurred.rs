//! Paid and incurred developed together (`docs/design/xll.md`, decision
//! 49): the Munich chain ladder (`prospicio_reserving::MunichChainLadder`,
//! Quarg and Mack 2004) and Merz and Wüthrich's paid-incurred chain
//! (`prospicio_reserving::PaidIncurredChain`, 2010).
//!
//! Each function takes two triangles of the same shape, paid and
//! incurred, laid out as every other triangle argument is (one row per
//! origin, one column per development period, blank where unobserved),
//! and both must be observed at the same cells. Development periods are
//! numbered from 1, as the triangle's columns are.

use prospicio_reserving::{
    Development, DevelopmentColumn, Grain, Long, MunichChainLadder, MunichFit, PaidIncurredChain,
    PicConvention, PicFit, PicPriors, SigmaInterpolation, Triangle, VarianceExtrapolation,
};

use crate::pricing::numbers;
use crate::reserving::{COLUMN, origin_labels, tail, text_option, triangle};
use crate::sheet::{Grid, Value};

type Result<T> = std::result::Result<T, String>;

const PAID: &str = "paid";
const INCURRED: &str = "incurred";

/// The paid and incurred triangles as one triangle with a `paid` and an
/// `incurred` column, after checking that they are observed at the same
/// cells.
fn pair(paid: &Grid, incurred: &Grid) -> Result<Triangle> {
    if (paid.rows, paid.columns) != (incurred.rows, incurred.columns) {
        return Err(format!(
            "paid and incurred must be the same size, got {} x {} and {} x {}",
            paid.rows, paid.columns, incurred.rows, incurred.columns
        ));
    }
    let read = |grid: &Grid, name: &str| {
        triangle(grid)
            .map(|t| t.to_long())
            .map_err(|e| format!("{name}: {e}"))
    };
    let (p, i) = (read(paid, PAID)?, read(incurred, INCURRED)?);
    if p.origin != i.origin || p.development != i.development {
        return Err("paid and incurred must be observed at the same cells".into());
    }
    let values = |long: &prospicio_reserving::LongTable| {
        long.values
            .iter()
            .find(|(n, _)| n == COLUMN)
            .map(|(_, v)| v.clone())
            .expect("a sheet triangle has its one column")
    };
    let (pv, iv) = (values(&p), values(&i));
    Triangle::from_long(&Long {
        keys: &[],
        origin: &p.origin,
        development: DevelopmentColumn::Age(&p.development),
        values: &[(PAID, &pv), (INCURRED, &iv)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .map_err(|e| e.to_string())
}

/// A ratio, blank where the denominator is 0.
fn ratio(a: f64, b: f64) -> Value {
    if b == 0.0 {
        Value::Empty
    } else {
        Value::number(a / b)
    }
}

/// A cell of `v`, blank past its end.
fn at(v: &[f64], k: usize) -> Value {
    v.get(k).map_or(Value::Empty, |x| Value::number(*x))
}

// ---------------------------------------------------------------------------
// The Munich chain ladder.

/// One `sigma` setting, as `PRO.RES.MACK` reads it.
fn sigma_one(v: &Value) -> Result<SigmaInterpolation> {
    match v {
        v if v.is_blank() => Ok(SigmaInterpolation::LogLinear),
        Value::Text(s) => match s.trim().to_ascii_lowercase().as_str() {
            "log-linear" | "loglinear" | "log_linear" => Ok(SigmaInterpolation::LogLinear),
            "mack" => Ok(SigmaInterpolation::Mack),
            other => Err(format!(
                "sigma must be \"log-linear\" or \"mack\", got {other:?}"
            )),
        },
        _ => Err("sigma must be \"log-linear\" or \"mack\"".into()),
    }
}

/// The `sigma` argument: one setting for both columns, or two cells,
/// paid then incurred (R's `est.sigmaP` and `est.sigmaI`).
fn sigmas(grid: &Grid) -> Result<(SigmaInterpolation, SigmaInterpolation)> {
    match grid.cells.as_slice() {
        [one] => {
            let s = sigma_one(one)?;
            Ok((s, s))
        }
        [p, i] => Ok((sigma_one(p)?, sigma_one(i)?)),
        _ => Err("sigma must be one setting, or two cells: paid then incurred".into()),
    }
}

fn munich_fit(args: &[Grid]) -> Result<MunichFit> {
    let tri = pair(&args[0], &args[1])?;
    let (sp, si) = sigmas(&args[2])?;
    let dev = |sigma_interpolation| Development {
        sigma_interpolation,
        ..Development::default()
    };
    MunichChainLadder {
        paid: dev(sp),
        incurred: dev(si),
        tail_paid: tail(&args[3]).map_err(|e| format!("tail_paid: {e}"))?,
        tail_incurred: tail(&args[4]).map_err(|e| format!("tail_incurred: {e}"))?,
    }
    .fit(&tri, PAID, INCURRED)
    .map_err(|e| e.to_string())
}

/// `PRO.RES.MUNICH(paid, incurred, [sigma], [tail_paid], [tail_incurred],
/// [origins])`: per origin and in total, the latest paid and incurred and
/// their ratio, the Munich ultimates and their ratio, the paid reserve and
/// incurred IBNR, and the separate chain ladders' ultimates.
pub fn summary(args: &[Grid]) -> Result<Grid> {
    let fit = munich_fit(args)?;
    let labels = origin_labels(&args[5], fit.origins.len())?;
    let (cp, ci) = (
        &fit.paid.chain_ladder.ultimate,
        &fit.incurred.chain_ladder.ultimate,
    );
    let row = |label: Value, lp: f64, li: f64, up: f64, ui: f64, clp: f64, cli: f64| {
        vec![
            label,
            Value::number(lp),
            Value::number(li),
            ratio(lp, li),
            Value::number(up),
            Value::number(ui),
            ratio(up, ui),
            Value::number(up - lp),
            Value::number(ui - li),
            Value::number(clp),
            Value::number(cli),
        ]
    };
    let mut rows = vec![
        [
            "Origin",
            "Latest paid",
            "Latest incurred",
            "Latest P/I",
            "Ultimate paid",
            "Ultimate incurred",
            "Ultimate P/I",
            "Paid reserve",
            "Incurred IBNR",
            "CL paid",
            "CL incurred",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for (o, label) in labels.into_iter().enumerate() {
        rows.push(row(
            label,
            fit.latest_paid[o],
            fit.latest_incurred[o],
            fit.ultimate_paid[o],
            fit.ultimate_incurred[o],
            cp[o],
            ci[o],
        ));
    }
    let sum = |v: &[f64]| v.iter().sum::<f64>();
    rows.push(row(
        Value::from("Total"),
        sum(&fit.latest_paid),
        sum(&fit.latest_incurred),
        sum(&fit.ultimate_paid),
        sum(&fit.ultimate_incurred),
        sum(cp),
        sum(ci),
    ));
    Ok(Grid::from_rows(rows))
}

/// `PRO.RES.MUNICH.PARAMS(paid, incurred, [sigma], [tail_paid],
/// [tail_incurred])`: a row per development period, its ratios `q` (P/I)
/// and `q^-1` (I/P), their deviations `rho`, and each column's age-to-age
/// factor and Mack sigma from that period (the tail's on the last, blank
/// without one); then `lambda` for paid and for incurred, each with the
/// number of residual pairs behind it.
pub fn params(args: &[Grid]) -> Result<Grid> {
    let fit = munich_fit(args)?;
    let (dp, di) = (
        &fit.paid.chain_ladder.development,
        &fit.incurred.chain_ladder.development,
    );
    let (tp, ti) = (&fit.paid.chain_ladder.tail, &fit.incurred.chain_ladder.tail);
    let n = fit.development.len();
    // The factor and sigma from period k: the estimated link, or the
    // tail's on the oldest age when there is one.
    let link = |ldf: &[f64], sigma: &[f64], t: &prospicio_reserving::TailFit, k: usize| {
        if k + 1 < n {
            (at(ldf, k), at(sigma, k))
        } else if t.factor != 1.0 {
            (Value::number(t.factor), Value::number(t.sigma))
        } else {
            (Value::Empty, Value::Empty)
        }
    };
    let mut rows = vec![
        [
            "Development",
            "q (P/I)",
            "q^-1 (I/P)",
            "rho paid",
            "rho incurred",
            "f paid",
            "sigma paid",
            "f incurred",
            "sigma incurred",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for k in 0..n {
        let (fp, sp) = link(&dp.ldf, &dp.sigma, tp, k);
        let (fi, si) = link(&di.ldf, &di.sigma, ti, k);
        rows.push(vec![
            Value::Number((k + 1) as f64),
            at(&fit.q, k),
            at(&fit.q_inverse, k),
            at(&fit.rho_paid, k),
            at(&fit.rho_incurred, k),
            fp,
            sp,
            fi,
            si,
        ]);
    }
    let blank = || std::iter::repeat_n(Value::Empty, 6);
    let lambda = |name: &str, value: f64, count: usize| {
        let mut row = vec![
            Value::from(name),
            Value::number(value),
            Value::Number(count as f64),
        ];
        row.extend(blank());
        row
    };
    rows.push(lambda("Lambda paid", fit.lambda_paid, fit.n_residuals_paid));
    rows.push(lambda(
        "Lambda incurred",
        fit.lambda_incurred,
        fit.n_residuals_incurred,
    ));
    Ok(Grid::from_rows(rows))
}

// ---------------------------------------------------------------------------
// The paid-incurred chain.

/// An optional range of numbers, one per development period (or one
/// fewer); `None` when blank.
fn optional_numbers(grid: &Grid, name: &str) -> Result<Option<Vec<f64>>> {
    if grid.is_blank() {
        Ok(None)
    } else {
        numbers(grid, name).map(Some)
    }
}

fn pic_fit(args: &[Grid]) -> Result<PicFit> {
    let tri = pair(&args[0], &args[1])?;
    let extrapolation = match text_option(&args[4], "extrapolation")?.as_deref() {
        None | Some("log-linear" | "loglinear" | "log_linear") => VarianceExtrapolation::LogLinear,
        Some("mack") => VarianceExtrapolation::Mack,
        Some(other) => {
            return Err(format!(
                "extrapolation must be \"log-linear\" or \"mack\", got {other:?}"
            ));
        }
    };
    let convention = match text_option(&args[5], "convention")?.as_deref() {
        None | Some("model") => PicConvention::Model,
        Some("r" | "chainladder" | "r-chainladder") => PicConvention::RChainLadder,
        Some(other) => {
            return Err(format!(
                "convention must be \"model\" or \"r\", got {other:?}"
            ));
        }
    };
    // A side of the priors is given as a mean and a standard deviation
    // together, or not at all.
    let side = |mean: &Grid, sd: &Grid, m: &str, s: &str| -> Result<(Vec<f64>, Vec<f64>)> {
        match (optional_numbers(mean, m)?, optional_numbers(sd, s)?) {
            (None, None) => Ok((Vec::new(), Vec::new())),
            (Some(mean), Some(sd)) => Ok((mean, sd)),
            _ => Err(format!("{m} and {s} are given together or not at all")),
        }
    };
    let (phi_mean, phi_sd) = side(&args[6], &args[7], "phi_mean", "phi_sd")?;
    let (psi_mean, psi_sd) = side(&args[8], &args[9], "psi_mean", "psi_sd")?;
    PaidIncurredChain {
        priors: PicPriors {
            phi_mean,
            phi_sd,
            psi_mean,
            psi_sd,
        },
        sigma: optional_numbers(&args[2], "sigma")?,
        tau: optional_numbers(&args[3], "tau")?,
        extrapolation,
        convention,
    }
    .fit(&tri, PAID, INCURRED)
    .map_err(|e| e.to_string())
}

/// `PRO.RES.PIC(paid, incurred, [sigma], [tau], [extrapolation],
/// [convention], [phi_mean], [phi_sd], [psi_mean], [psi_sd], [origins])`:
/// per origin and in total, the latest paid and incurred, the weight on
/// incurred, the ultimate, the reserve, its standard error (the root of
/// the mean squared error of prediction) and CV.
pub fn summary_pic(args: &[Grid]) -> Result<Grid> {
    let fit = pic_fit(args)?;
    let labels = origin_labels(&args[10], fit.origins.len())?;
    let cv = |se: f64, reserve: f64| {
        if reserve > 0.0 {
            Value::number(se / reserve)
        } else {
            Value::Empty
        }
    };
    let mut rows = vec![
        [
            "Origin",
            "Latest paid",
            "Latest incurred",
            "Incurred weight",
            "Ultimate",
            "Reserve",
            "SE",
            "CV",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for (o, label) in labels.into_iter().enumerate() {
        rows.push(vec![
            label,
            Value::number(fit.latest_paid[o]),
            Value::number(fit.latest_incurred[o]),
            Value::number(fit.incurred_weight[o]),
            Value::number(fit.ultimate[o]),
            Value::number(fit.reserve[o]),
            Value::number(fit.std_err[o]),
            cv(fit.std_err[o], fit.reserve[o]),
        ]);
    }
    let reserve = fit.total_reserve();
    rows.push(vec![
        Value::from("Total"),
        Value::number(fit.latest_paid.iter().sum()),
        Value::number(fit.latest_incurred.iter().sum()),
        Value::Empty,
        Value::number(fit.total_ultimate()),
        Value::number(reserve),
        Value::number(fit.total_std_err),
        cv(fit.total_std_err, reserve),
    ]);
    Ok(Grid::from_rows(rows))
}

/// `PRO.RES.PIC.PARAMS(paid, incurred, [sigma], [tau], [extrapolation],
/// [convention], [phi_mean], [phi_sd], [psi_mean], [psi_sd])`: a row per
/// development period, its paid and incurred standard deviations `sigma`
/// and `tau` (given or estimated) and the posterior mean and standard
/// deviation of the log link parameters `Phi` and `Psi` (`tau` and `Psi`
/// blank on the last period).
pub fn params_pic(args: &[Grid]) -> Result<Grid> {
    let fit = pic_fit(args)?;
    let n = fit.ages.len();
    let sd = fit.posterior_sd();
    let (phi, psi) = (fit.phi(), fit.psi());
    let mut rows = vec![
        [
            "Development",
            "sigma",
            "tau",
            "Phi",
            "Phi SD",
            "Psi",
            "Psi SD",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for k in 0..n {
        rows.push(vec![
            Value::Number((k + 1) as f64),
            at(&fit.sigma, k),
            at(&fit.tau, k),
            at(phi, k),
            at(&sd, k),
            at(psi, k),
            if k < psi.len() {
                at(&sd, n + k)
            } else {
                Value::Empty
            },
        ]);
    }
    Ok(Grid::from_rows(rows))
}
