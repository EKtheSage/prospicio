//! Bayesian reserving by MCMC (`docs/design/xll.md`, decision 21): the
//! Bayesian over-dispersed Poisson chain ladder of England and Verrall
//! (`prospicio_reserving::bayes::BayesOdp`), sampled with NUTS, as one
//! worksheet function with three views of the same fit: the reserves, the
//! posterior of the parameters, and the sampler's diagnostics.
//!
//! A fit takes seconds, and a sheet usually shows all three views, so the
//! last few fits are kept by their inputs. The fit is a pure function of
//! its inputs and seed, so a kept fit is the one a refit would give: the
//! cache saves time and changes no result.

use std::sync::{Arc, Mutex};

use prospicio_bayes::nuts::Sampler;
use prospicio_reserving::bayes::{BayesOdp, BayesOdpFit};

use crate::reserving::{
    MAX_SIMS, bootstrap_exhibit, count, origin_labels, percentiles, process, seed, triangle,
};
use crate::sheet::{Grid, Value};

type Result<T> = std::result::Result<T, String>;

/// The most kept draws a chain may take, and the most chains.
pub const MAX_DRAWS: usize = 20_000;
pub const MAX_CHAINS: usize = 8;

/// How many fits are kept.
const KEPT: usize = 8;

static FITS: Mutex<Vec<(String, Arc<BayesOdpFit>)>> = Mutex::new(Vec::new());

/// The fit for these inputs, from the kept fits or fitted now.
fn fit(args: &[Grid]) -> Result<Arc<BayesOdpFit>> {
    let draws = count(&args[2], "draws", 1_000, MAX_DRAWS)?;
    let chains = count(&args[3], "chains", 4, MAX_CHAINS)?;
    let seed = seed(&args[4])?;
    let n_sims = count(&args[5], "sims", 10_000, MAX_SIMS)?;
    let prior_sd = match args[7].as_scalar() {
        Some(v) if v.is_blank() => 10.0,
        Some(Value::Number(x)) if *x > 0.0 && x.is_finite() => *x,
        _ => return Err("prior_sd must be a positive number (10 if omitted)".into()),
    };
    let process = process(&args[8])?;
    let key = format!(
        "{:?}|{draws}|{chains}|{seed}|{n_sims}|{prior_sd:?}|{process:?}",
        args[0]
    );
    if let Some((_, f)) = FITS
        .lock()
        .map_err(|_| "the fit cache is poisoned")?
        .iter()
        .find(|(k, _)| *k == key)
    {
        return Ok(Arc::clone(f));
    }
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
    let f = Arc::new(model.fit(&tri, "values").map_err(|e| e.to_string())?);
    let mut kept = FITS.lock().map_err(|_| "the fit cache is poisoned")?;
    if kept.len() == KEPT {
        kept.remove(0);
    }
    kept.push((key, Arc::clone(&f)));
    Ok(f)
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
        "params" | "parameters" => params(&*fit(args)?),
        "diagnostics" => Ok(diagnostics(&*fit(args)?)),
        _ => Err(OUTPUTS.into()),
    }
}

const OUTPUTS: &str = "output must be \"reserves\", \"params\" or \"diagnostics\"";

fn params(f: &BayesOdpFit) -> Result<Grid> {
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
    for p in f.summary().map_err(|e| e.to_string())? {
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

fn diagnostics(f: &BayesOdpFit) -> Grid {
    let d = &f.diagnostics;
    let n = |label: &str, x: f64| vec![Value::from(label), Value::number(x)];
    let chains = f.posterior.chains();
    let mut rows = vec![
        n("Max R-hat", d.max_rhat),
        n("Min bulk ESS", d.min_ess_bulk),
        n("Min tail ESS", d.min_ess_tail),
        n("Divergences", d.divergences as f64),
        n("Chains", chains as f64),
        n("Draws per chain", f.posterior.draws_per_chain() as f64),
        n("Scale (phi)", f.scale),
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
