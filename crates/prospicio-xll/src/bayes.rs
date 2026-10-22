//! Bayesian reserving by MCMC (`docs/design/xll.md`, decisions 21 and 38):
//! the Bayesian over-dispersed Poisson chain ladder of England and Verrall
//! (`prospicio_reserving::bayes::BayesOdp`) and the Bayesian
//! Bornhuetter-Ferguson with a normalised pattern
//! (`prospicio_reserving::bayes::BayesBfNormalised`), sampled with NUTS,
//! each as one worksheet function with views of the same fit: the
//! reserves, the posterior of the parameters, and the sampler's
//! diagnostics (and for the BF, the prior and posterior ultimates).
//!
//! A fit takes seconds, and a sheet usually shows all three views, so the
//! last few fits are kept by their inputs. The fit is a pure function of
//! its inputs and seed, so a kept fit is the one a refit would give: the
//! cache saves time and changes no result.

use std::sync::{Arc, Mutex};

use prospicio_bayes::nuts::{ParamSummary, PosteriorDraws, Sampler};
use prospicio_reserving::bayes::{
    BayesBfNormalised, BayesBfNormalisedFit, BayesOdp, BayesOdpFit, Diagnostics,
};

use crate::reserving::{
    COLUMN, EXPOSURE, MAX_SIMS, bootstrap_exhibit, count, exposed_triangle, origin_labels,
    percentiles, process, seed, triangle,
};
use crate::sheet::{Grid, Value};

type Result<T> = std::result::Result<T, String>;

/// The most kept draws a chain may take, and the most chains.
pub const MAX_DRAWS: usize = 20_000;
pub const MAX_CHAINS: usize = 8;

/// How many fits of each model are kept.
const KEPT: usize = 8;

/// The last [`KEPT`] fits of one model, keyed on every input.
type Kept<T> = Mutex<Vec<(String, Arc<T>)>>;

static ODP_FITS: Kept<BayesOdpFit> = Mutex::new(Vec::new());
static BF_FITS: Kept<BayesBfNormalisedFit> = Mutex::new(Vec::new());

/// The fit kept under `key`, or `fit`'s, kept from now on.
fn kept<T>(cache: &Kept<T>, key: String, fit: impl FnOnce() -> Result<T>) -> Result<Arc<T>> {
    if let Some((_, f)) = cache
        .lock()
        .map_err(|_| "the fit cache is poisoned")?
        .iter()
        .find(|(k, _)| *k == key)
    {
        return Ok(Arc::clone(f));
    }
    let f = Arc::new(fit()?);
    let mut kept = cache.lock().map_err(|_| "the fit cache is poisoned")?;
    if kept.len() == KEPT {
        kept.remove(0);
    }
    kept.push((key, Arc::clone(&f)));
    Ok(f)
}

/// The sampler's arguments, `draws`, `chains`, `seed` and `sims`, from
/// `args[first..]`.
fn sampling(args: &[Grid], first: usize) -> Result<(usize, usize, u64, usize)> {
    Ok((
        count(&args[first], "draws", 1_000, MAX_DRAWS)?,
        count(&args[first + 1], "chains", 4, MAX_CHAINS)?,
        seed(&args[first + 2])?,
        count(&args[first + 3], "sims", 10_000, MAX_SIMS)?,
    ))
}

/// The `prior_sd` argument: a positive number, 10 if omitted.
fn prior_sd(grid: &Grid) -> Result<f64> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(10.0),
        Some(Value::Number(x)) if *x > 0.0 && x.is_finite() => Ok(*x),
        _ => Err("prior_sd must be a positive number (10 if omitted)".into()),
    }
}

/// The ODP fit for these inputs, from the kept fits or fitted now.
fn fit(args: &[Grid]) -> Result<Arc<BayesOdpFit>> {
    let (draws, chains, seed, n_sims) = sampling(args, 2)?;
    let prior_sd = prior_sd(&args[7])?;
    let process = process(&args[8])?;
    let key = format!(
        "{:?}|{draws}|{chains}|{seed}|{n_sims}|{prior_sd:?}|{process:?}",
        args[0]
    );
    kept(&ODP_FITS, key, || {
        let tri = triangle(&args[0])?;
        let model = BayesOdp {
            intercept_sd: prior_sd,
            origin_sd: prior_sd,
            development_sd: prior_sd,
            process,
            sampler: Sampler {
                chains,
                tune: draws,
                draws,
                ..BayesOdp::default().sampler
            },
            n_sims,
            seed,
            ..BayesOdp::default()
        };
        model.fit(&tri, COLUMN).map_err(|e| e.to_string())
    })
}

/// The Bayesian ODP chain ladder: `PRO.BAYES.ODP(triangle, [output],
/// [draws], [chains], [seed], [sims], [percentiles], [prior_sd],
/// [process], [origins])`. `output` is `"reserves"` (default: the reserve
/// exhibit, as `PRO.ODP`'s), `"params"` (each parameter's posterior
/// summary with R-hat and ESS) or `"diagnostics"` (the sampler's checks
/// and any warnings).
pub fn bayes_odp(args: &[Grid]) -> Result<Grid> {
    let output = match args[1].as_scalar() {
        Some(v) if v.is_blank() => "reserves".to_owned(),
        Some(Value::Text(s)) => s.trim().to_ascii_lowercase(),
        _ => return Err(OUTPUTS.into()),
    };
    // Check the view's own arguments before sampling.
    let probabilities = percentiles(&args[6])?;
    match output.as_str() {
        "reserves" => {
            let f = fit(args)?;
            let labels = origin_labels(&args[9], f.chain_ladder.origins.len())?;
            bootstrap_exhibit(&f.reserves, &f.chain_ladder, labels, &probabilities)
        }
        "params" | "parameters" => params(fit(args)?.summary().map_err(|e| e.to_string())?),
        "diagnostics" => {
            let f = fit(args)?;
            Ok(diagnostics(&f.diagnostics, &f.posterior, f.scale))
        }
        _ => Err(OUTPUTS.into()),
    }
}

const OUTPUTS: &str = "output must be \"reserves\", \"params\" or \"diagnostics\"";

/// The `cv` argument: a positive number, 0.1 if omitted, or `"flat"` for
/// a flat prior on the log ultimate (the Bayesian chain ladder).
fn cv(grid: &Grid) -> Result<f64> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(0.1),
        Some(Value::Number(x)) if *x > 0.0 && x.is_finite() => Ok(*x),
        Some(Value::Text(s)) if s.trim().eq_ignore_ascii_case("flat") => Ok(f64::INFINITY),
        _ => Err("cv must be a positive number (0.1 if omitted) or \"flat\"".into()),
    }
}

/// The `common_cv` argument: none if omitted, else a positive number.
fn common_cv(grid: &Grid) -> Result<Option<f64>> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(None),
        Some(Value::Number(x)) if *x > 0.0 && x.is_finite() => Ok(Some(*x)),
        _ => Err("common_cv must be a positive number, or omitted for none".into()),
    }
}

/// The normalised Bayesian BF fit for these inputs, from the kept fits or
/// fitted now.
fn bf_fit(args: &[Grid]) -> Result<Arc<BayesBfNormalisedFit>> {
    let apriori = match args[2].as_scalar() {
        Some(Value::Number(x)) if *x > 0.0 && x.is_finite() => *x,
        Some(v) if v.is_blank() => return Err("apriori is required".into()),
        _ => return Err("apriori must be a positive number".into()),
    };
    let cv = cv(&args[3])?;
    let (draws, chains, seed, n_sims) = sampling(args, 5)?;
    let development_sd = prior_sd(&args[10])?;
    let process = process(&args[11])?;
    let common_cv = common_cv(&args[13])?;
    if common_cv.is_some() && cv.is_infinite() {
        return Err("common_cv needs a finite cv, not \"flat\"".into());
    }
    let key = format!(
        "{:?}|{:?}|{apriori:?}|{cv:?}|{draws}|{chains}|{seed}|{n_sims}|{development_sd:?}|{process:?}|{common_cv:?}",
        args[0], args[1]
    );
    kept(&BF_FITS, key, || {
        let tri = exposed_triangle(&args[0], &args[1])?;
        let model = BayesBfNormalised {
            apriori,
            cv,
            development_sd,
            process,
            sampler: Sampler {
                chains,
                tune: draws,
                draws,
                ..BayesBfNormalised::default().sampler
            },
            n_sims,
            seed,
            common_cv,
        };
        model.fit(&tri, COLUMN, EXPOSURE).map_err(|e| e.to_string())
    })
}

/// The Bayesian Bornhuetter-Ferguson with a normalised pattern and a
/// gamma prior on each origin's ultimate, mean `apriori` times exposure:
/// `PRO.BAYES.BF(triangle, exposure, apriori, [cv], [output], [draws],
/// [chains], [seed], [sims], [percentiles], [prior_sd], [process],
/// [origins], [common_cv])`. `output` is `"reserves"` (default),
/// `"ultimates"` (each origin's exposure, prior and posterior expected
/// ultimate), `"params"` or `"diagnostics"`.
pub fn bayes_bf(args: &[Grid]) -> Result<Grid> {
    let output = match args[4].as_scalar() {
        Some(v) if v.is_blank() => "reserves".to_owned(),
        Some(Value::Text(s)) => s.trim().to_ascii_lowercase(),
        _ => return Err(BF_OUTPUTS.into()),
    };
    let probabilities = percentiles(&args[9])?;
    match output.as_str() {
        "reserves" => {
            let f = bf_fit(args)?;
            let labels = origin_labels(&args[12], f.chain_ladder.origins.len())?;
            bootstrap_exhibit(&f.reserves, &f.chain_ladder, labels, &probabilities)
        }
        "ultimates" => {
            let f = bf_fit(args)?;
            let labels = origin_labels(&args[12], f.chain_ladder.origins.len())?;
            Ok(ultimates(&f, labels))
        }
        "params" | "parameters" => params(bf_fit(args)?.summary().map_err(|e| e.to_string())?),
        "diagnostics" => {
            let f = bf_fit(args)?;
            Ok(diagnostics(&f.diagnostics, &f.posterior, f.scale))
        }
        _ => Err(BF_OUTPUTS.into()),
    }
}

const BF_OUTPUTS: &str =
    "output must be \"reserves\", \"ultimates\", \"params\" or \"diagnostics\"";

/// Per origin: exposure, latest, the prior mean of the expected ultimate
/// (apriori times exposure), its posterior mean, and the posterior mean
/// loss ratio; then the total.
fn ultimates(f: &BayesBfNormalisedFit, labels: Vec<Value>) -> Grid {
    let posterior = f.expected_ultimate();
    let latest = &f.chain_ladder.latest;
    let mut rows = vec![
        [
            "Origin",
            "Exposure",
            "Latest",
            "Prior ultimate",
            "Posterior ultimate",
            "Posterior LR",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for (i, label) in labels.into_iter().enumerate() {
        rows.push(vec![
            label,
            Value::number(f.exposure[i]),
            Value::number(latest[i]),
            Value::number(f.prior_ultimate[i]),
            Value::number(posterior[i]),
            Value::number(posterior[i] / f.exposure[i]),
        ]);
    }
    let (exposure, ultimate): (f64, f64) = (f.exposure.iter().sum(), posterior.iter().sum());
    rows.push(vec![
        Value::from("Total"),
        Value::number(exposure),
        Value::number(latest.iter().sum()),
        Value::number(f.prior_ultimate.iter().sum()),
        Value::number(ultimate),
        Value::number(ultimate / exposure),
    ]);
    Grid::from_rows(rows)
}

fn params(summary: Vec<ParamSummary>) -> Result<Grid> {
    let mut rows = vec![
        [
            "Parameter",
            "Mean",
            "SD",
            "Q5",
            "Median",
            "Q95",
            "R-hat",
            "Bulk ESS",
            "Tail ESS",
        ]
        .map(Value::from)
        .to_vec(),
    ];
    for p in summary {
        rows.push(vec![
            Value::Text(p.name),
            Value::number(p.mean),
            Value::number(p.sd),
            Value::number(p.q05),
            Value::number(p.q50),
            Value::number(p.q95),
            Value::number(p.rhat),
            Value::number(p.ess_bulk),
            Value::number(p.ess_tail),
        ]);
    }
    Ok(Grid::from_rows(rows))
}

fn diagnostics(d: &Diagnostics, posterior: &PosteriorDraws, scale: f64) -> Grid {
    let n = |label: &str, x: f64| vec![Value::from(label), Value::number(x)];
    let chains = posterior.chains();
    let mut rows = vec![
        n("Max R-hat", d.max_rhat),
        n("Min bulk ESS", d.min_ess_bulk),
        n("Min tail ESS", d.min_ess_tail),
        n("Divergences", d.divergences as f64),
        n("Chains", chains as f64),
        n("Draws per chain", posterior.draws_per_chain() as f64),
        n("Scale (phi)", scale),
    ];
    if d.warnings.is_empty() {
        rows.push(vec![Value::from("Warnings"), Value::from("none")]);
    } else {
        for w in &d.warnings {
            rows.push(vec![Value::from("Warning"), Value::Text(w.clone())]);
        }
    }
    Grid::from_rows(rows)
}
