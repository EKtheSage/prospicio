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
use prospicio_prob::Distribution;
use prospicio_reserving::bayes::{
    BayesBfNormalised, BayesBfNormalisedFit, BayesOdp, BayesOdpFit, Diagnostics,
};
use prospicio_reserving::{IndustryPrior, IndustryPriorSettings, PatternPrior, Triangle};

use crate::reserving::{
    COLUMN, EXPOSURE, MAX_SIMS, bootstrap_exhibit, count, exposed_triangle, origin_labels,
    percentile_label, percentiles, process, seed, triangle,
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

/// The Bayesian ODP chain ladder: `PRO.RES.BAYES.ODP(triangle, [output],
/// [draws], [chains], [seed], [sims], [percentiles], [prior_sd],
/// [process], [origins])`. `output` is `"reserves"` (default: the reserve
/// exhibit, as `PRO.RES.ODP`'s), `"params"` (each parameter's posterior
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
    let pattern = (args.len() > 16).then(|| pattern_values(&args[15], &args[16]));
    let pattern = pattern.transpose()?.flatten();
    let key = format!(
        "{:?}|{:?}|{apriori:?}|{cv:?}|{draws}|{chains}|{seed}|{n_sims}|{development_sd:?}|{process:?}|{common_cv:?}|{pattern:?}",
        args[0], args[1]
    );
    kept(&BF_FITS, key, || {
        let tri = exposed_triangle(&args[0], &args[1])?;
        let prior = match &pattern {
            Some((mean, sd)) => Some(pattern_prior(&tri, mean, sd)?),
            None => None,
        };
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
        match prior {
            Some(prior) => model.fit_with_pattern_prior(&tri, COLUMN, EXPOSURE, &prior),
            None => model.fit(&tri, COLUMN, EXPOSURE),
        }
        .map_err(|e| e.to_string())
    })
}

/// The Bayesian Bornhuetter-Ferguson with a normalised pattern and a
/// gamma prior on each origin's ultimate, mean `apriori` times exposure:
/// `PRO.RES.BAYES.BF(triangle, exposure, apriori, [cv], [output], [draws],
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
        "new_year" | "new year" => {
            let premium = match args[14].as_scalar() {
                Some(Value::Number(x)) if *x > 0.0 && x.is_finite() => *x,
                Some(v) if v.is_blank() => {
                    return Err(
                        "the new_year view needs new_premium, the new year's premium".into(),
                    );
                }
                _ => return Err("new_premium must be a positive number".into()),
            };
            let f = bf_fit(args)?;
            let (_, _, seed, n_sims) = sampling(args, 5)?;
            let year = f
                .new_year(premium, n_sims, seed)
                .map_err(|e| e.to_string())?;
            let mut header = vec![
                Value::Text(year.origin.to_string()),
                Value::from("Mean"),
                Value::from("SD"),
            ];
            header.extend(
                probabilities
                    .iter()
                    .map(|&p| Value::Text(percentile_label(p))),
            );
            let mut rows = vec![header];
            for (label, d) in [
                ("Loss ratio", &year.loss_ratio),
                ("Ultimate", &year.ultimate),
            ] {
                let mut row = vec![
                    Value::from(label),
                    Value::number(d.mean()),
                    Value::number(d.std_dev()),
                ];
                for &p in &probabilities {
                    row.push(Value::number(d.quantile(p).map_err(|e| e.to_string())?));
                }
                rows.push(row);
            }
            Ok(Grid::from_rows(rows))
        }
        _ => Err(BF_OUTPUTS.into()),
    }
}

const BF_OUTPUTS: &str =
    "output must be \"reserves\", \"ultimates\", \"params\", \"diagnostics\" or \"new_year\"";

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

// ---------------------------------------------------------------------------
// A pattern prior and the industry prior (decision 54).

/// The `pattern_mean` and `pattern_sd` arguments: each a row or column of
/// the prior on the log share ratio to the first age at each later age,
/// one cell per triangle column with the first blank, or one per age after
/// the first; `None` when both are omitted.
fn pattern_values(mean: &Grid, sd: &Grid) -> Result<Option<(Vec<f64>, Vec<f64>)>> {
    match (mean.is_blank(), sd.is_blank()) {
        (true, true) => return Ok(None),
        (false, false) => {}
        _ => return Err("pattern_mean and pattern_sd go together".into()),
    }
    let read = |g: &Grid, name: &str| -> Result<Vec<f64>> {
        if g.rows != 1 && g.columns != 1 {
            return Err(format!("{name} must be a row or a column"));
        }
        let mut cells = g.cells.as_slice();
        if cells.first().is_some_and(Value::is_blank) {
            cells = &cells[1..];
        }
        cells
            .iter()
            .enumerate()
            .map(|(i, v)| match v {
                Value::Number(x) => Ok(*x),
                other => Err(format!("{name} {} must be a number, got {other:?}", i + 1)),
            })
            .collect()
    };
    Ok(Some((read(mean, "pattern_mean")?, read(sd, "pattern_sd")?)))
}

/// The pattern prior at the triangle's ages.
fn pattern_prior(tri: &Triangle, mean: &[f64], sd: &[f64]) -> Result<PatternPrior> {
    PatternPrior::new(tri.development(), mean, sd).map_err(|e| e.to_string())
}

/// The companies' triangles stacked in one range, a company label per
/// row (each company's rows together, oldest origin first) and a premium
/// per row: the triangles and premiums in the order the companies first
/// appear.
fn companies(
    stacked: &Grid,
    labels: &Grid,
    premiums: &Grid,
) -> Result<(Vec<Triangle>, Vec<Vec<f64>>)> {
    let n = stacked.rows;
    if labels.cells.len() != n || (labels.rows != 1 && labels.columns != 1) {
        return Err(format!(
            "companies must be a column of {n} labels, one per row of triangles"
        ));
    }
    if premiums.cells.len() != n || (premiums.rows != 1 && premiums.columns != 1) {
        return Err(format!(
            "premiums must be a column of {n} values, one per row of triangles"
        ));
    }
    let mut names: Vec<Value> = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (r, label) in labels.cells.iter().enumerate() {
        if label.is_blank() {
            return Err(format!("companies {} is blank", r + 1));
        }
        match names.iter().position(|l| l == label) {
            Some(k) if k + 1 == names.len() => groups[k].push(r),
            Some(_) => {
                return Err(format!(
                    "company {label:?}'s rows must be together; row {} is apart from the others",
                    r + 1
                ));
            }
            None => {
                names.push(label.clone());
                groups.push(vec![r]);
            }
        }
    }
    let mut triangles = Vec::new();
    let mut money = Vec::new();
    for (k, rows) in groups.iter().enumerate() {
        let sub = Grid::from_rows(
            rows.iter()
                .map(|&r| {
                    (0..stacked.columns)
                        .map(|c| stacked.get(r, c).clone())
                        .collect()
                })
                .collect(),
        );
        triangles.push(triangle(&sub).map_err(|e| format!("company {:?}: {e}", names[k]))?);
        money.push(
            rows.iter()
                .map(|&r| match &premiums.cells[r] {
                    Value::Number(x) => Ok(*x),
                    other => Err(format!(
                        "premiums {} must be a number, got {other:?}",
                        r + 1
                    )),
                })
                .collect::<Result<Vec<f64>>>()?,
        );
    }
    Ok((triangles, money))
}

/// `PRO.RES.INDUSTRY.PRIOR(triangles, companies, premiums, [min_developed],
/// [subtract_process], [age], [output])`: an industry prior for the
/// Bayesian BF from other companies' triangles (Schedule P, say): the mean
/// loss ratio and its cvs between and within companies, and a prior on the
/// pattern. `age` gives the prior for a triangle that stops at that age.
/// `output` is `"summary"` (default: the numbers `PRO.RES.BAYES.BF` takes
/// as `apriori`, `cv` and `common_cv`), `"pattern"` (by age, the prior
/// mean and SD of the log share ratio, which it takes as `pattern_mean`
/// and `pattern_sd`, and the percent developed) or `"report"` (what was
/// left out, and warnings).
pub fn industry_prior(args: &[Grid]) -> Result<Grid> {
    let (triangles, premiums) = companies(&args[0], &args[1], &args[2])?;
    let defaults = IndustryPriorSettings::default();
    let min_developed = match args[3].as_scalar() {
        Some(v) if v.is_blank() => defaults.min_developed,
        Some(Value::Number(x)) if (0.0..=1.0).contains(x) => *x,
        _ => return Err("min_developed must be a share from 0 to 1 (0.5 if omitted)".into()),
    };
    let subtract_process = match args[4].as_scalar() {
        Some(v) if v.is_blank() => false,
        Some(Value::Bool(b)) => *b,
        _ => return Err("subtract_process must be TRUE or FALSE (FALSE if omitted)".into()),
    };
    let mut prior: IndustryPrior = IndustryPriorSettings {
        min_developed,
        subtract_process,
        ..defaults
    }
    .from_triangles(&triangles, COLUMN, &premiums)
    .map_err(|e| e.to_string())?;
    match args[5].as_scalar() {
        Some(v) if v.is_blank() => {}
        Some(Value::Number(x)) if *x > 0.0 && x.fract() == 0.0 => {
            prior = prior.at_age(*x as u32).map_err(|e| e.to_string())?;
        }
        _ => return Err("age must be one of the triangles' ages in months, such as 60".into()),
    }
    let output = match args[6].as_scalar() {
        Some(v) if v.is_blank() => "summary".to_owned(),
        Some(Value::Text(s)) => s.trim().to_ascii_lowercase(),
        _ => return Err(PRIOR_OUTPUTS.into()),
    };
    let n = |label: &str, x: f64| vec![Value::from(label), Value::number(x)];
    match output.as_str() {
        "summary" => {
            let r = &prior.report;
            Ok(Grid::from_rows(vec![
                n("Apriori", prior.apriori),
                n("Apriori SE", prior.apriori_se),
                n("CV", prior.cv),
                vec![
                    Value::from("Common CV"),
                    prior.common_cv().map_or(Value::Empty, Value::number),
                ],
                n("Between-company CV", prior.between_cv),
                n("Within-company CV", prior.within_cv),
                n("Process CV", prior.process_cv),
                n("Tail factor", prior.tail_factor),
                n("Companies", r.n_companies as f64),
                n("Companies used", r.n_companies_used as f64),
                n("Origins used", r.n_origins_used as f64),
            ]))
        }
        "pattern" => {
            let p = &prior.pattern;
            let mut rows = vec![
                [
                    "Age",
                    "Pattern mean",
                    "Pattern SD",
                    "Percent developed",
                    "Percent developed SD",
                ]
                .map(Value::from)
                .to_vec(),
            ];
            for (k, &age) in p.ages.iter().enumerate() {
                let beta = |v: &[f64]| {
                    k.checked_sub(1)
                        .map_or(Value::Empty, |j| Value::number(v[j]))
                };
                rows.push(vec![
                    Value::number(f64::from(age)),
                    beta(&p.mean),
                    beta(&p.sd),
                    Value::number(prior.percent_developed[k]),
                    Value::number(prior.percent_developed_sd[k]),
                ]);
            }
            Ok(Grid::from_rows(rows))
        }
        "report" => {
            let r = &prior.report;
            let mut rows: Vec<Vec<Value>> = r
                .excluded
                .iter()
                .map(|e| vec![Value::from("Excluded"), Value::Text(e.clone())])
                .chain(
                    r.warnings
                        .iter()
                        .map(|w| vec![Value::from("Warning"), Value::Text(w.clone())]),
                )
                .collect();
            if rows.is_empty() {
                rows.push(vec![Value::from("Report"), Value::from("none")]);
            }
            Ok(Grid::from_rows(rows))
        }
        _ => Err(PRIOR_OUTPUTS.into()),
    }
}

const PRIOR_OUTPUTS: &str = "output must be \"summary\", \"pattern\" or \"report\"";
