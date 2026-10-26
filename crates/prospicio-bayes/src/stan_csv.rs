//! CmdStan's CSV output read into [`PosteriorDraws`], so the diagnostics,
//! summary and PSIS-LOO of this crate run on a Stan fit as on our own
//! (#395). The format, as CmdStan writes it for `method=sample`:
//!
//! - lines starting with `#` are comments: the configuration first (with
//!   `num_warmup`, `save_warmup` and `thin`), the adaptation block (step
//!   size and inverse metric) after the header, timing at the end;
//! - one header line of column names: the sampler's columns, which end
//!   in `__` (`lp__`, `accept_stat__`, `stepsize__`, `treedepth__`,
//!   `n_leapfrog__`, `divergent__`, `energy__`), then the parameters,
//!   transformed parameters and generated quantities, a container's
//!   elements joined to its name by `.` in column-major order
//!   (`theta.1.2`);
//! - one line per draw, the warmup draws first when `save_warmup` is on.
//!
//! [`read`] takes one file per chain (CmdStan writes one per chain, or
//! one per chain with `num_chains`), checks that they share a header and
//! a length, drops saved warmup draws, renames `theta.1.2` to `theta[1,2]`
//! as cmdstanr and `posterior` do, and splits the columns: `lp__` and the
//! model's variables become the draws, the other `__` columns the
//! sampler's diagnostics, with each chain's count of `divergent__` draws
//! as its divergences.
//!
//! ```
//! use prospicio_bayes::stan_csv;
//!
//! let chain = "# method = sample (Default)\n\
//!              lp__,accept_stat__,divergent__,mu,theta.1,theta.2\n\
//!              ## Adaptation terminated\n\
//!              -1.5,0.9,0,0.1,1.0,2.0\n\
//!              -1.2,0.8,1,0.2,1.1,2.1\n";
//! let fit = stan_csv::parse(&[chain, chain]).unwrap();
//! assert_eq!(fit.draws.names(), ["lp__", "mu", "theta[1]", "theta[2]"]);
//! assert_eq!(fit.draws.divergences(), [1, 1]);
//! assert_eq!(fit.sampler.unwrap().names(), ["accept_stat__", "divergent__"]);
//! ```

use std::path::Path;

use prospicio_core::{Error, Result};

use crate::nuts::PosteriorDraws;

/// A Stan fit read from its CSV files.
#[derive(Debug, Clone, PartialEq)]
pub struct StanFit {
    /// `lp__` and the model's variables, one chain per file.
    pub draws: PosteriorDraws,
    /// The sampler's other `__` columns (`accept_stat__`, `stepsize__`,
    /// `treedepth__`, `n_leapfrog__`, `divergent__`, `energy__`), in the
    /// same chains; `None` for a file without them (standalone generated
    /// quantities).
    pub sampler: Option<PosteriorDraws>,
    /// Warmup draws dropped from each chain (saved with `save_warmup`).
    pub warmup_dropped: usize,
}

/// Reads CmdStan CSV files, one per chain. See the [module](self) docs.
pub fn read<P: AsRef<Path>>(paths: &[P]) -> Result<StanFit> {
    let texts = paths
        .iter()
        .map(|p| {
            std::fs::read_to_string(p.as_ref())
                .map_err(|e| Error::Data(format!("cannot read {}: {e}", p.as_ref().display())))
        })
        .collect::<Result<Vec<String>>>()?;
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    parse(&refs)
}

/// Parses the text of CmdStan CSV files, one per chain.
pub fn parse(chains: &[&str]) -> Result<StanFit> {
    if chains.is_empty() {
        return Err(Error::Data("no Stan CSV chains given".into()));
    }
    let parsed = chains
        .iter()
        .enumerate()
        .map(|(c, text)| {
            parse_chain(text).map_err(|e| Error::Data(format!("chain {}: {e}", c + 1)))
        })
        .collect::<Result<Vec<Chain>>>()?;
    let header = &parsed[0].header;
    let n = parsed[0].rows.len();
    for (c, p) in parsed.iter().enumerate().skip(1) {
        if p.header != *header {
            return Err(Error::Data(format!(
                "chain {} has other columns than chain 1",
                c + 1
            )));
        }
        if p.rows.len() != n {
            return Err(Error::Data(format!(
                "chain {} has {} draws and chain 1 has {n}",
                c + 1,
                p.rows.len()
            )));
        }
    }
    if n == 0 {
        return Err(Error::Data("the chains have no draws after warmup".into()));
    }
    let is_sampler = |name: &str| name.ends_with("__") && name != "lp__";
    let model: Vec<usize> = (0..header.len())
        .filter(|&j| !is_sampler(&header[j]))
        .collect();
    let sampler: Vec<usize> = (0..header.len())
        .filter(|&j| is_sampler(&header[j]))
        .collect();
    let divergent = header.iter().position(|h| h == "divergent__");
    let pick = |cols: &[usize], p: &Chain| -> Vec<f64> {
        p.rows
            .iter()
            .flat_map(|r| cols.iter().map(move |&j| r[j]))
            .collect()
    };
    let divergences: Vec<usize> = match divergent {
        Some(j) => parsed
            .iter()
            .map(|p| p.rows.iter().filter(|r| r[j] != 0.0).count())
            .collect(),
        None => vec![],
    };
    let names = |cols: &[usize]| cols.iter().map(|&j| rename(&header[j])).collect();
    let draws = PosteriorDraws::from_chains(
        names(&model),
        parsed.iter().map(|p| pick(&model, p)).collect(),
        divergences.clone(),
    )?;
    let sampler = if sampler.is_empty() {
        None
    } else {
        Some(PosteriorDraws::from_chains(
            names(&sampler),
            parsed.iter().map(|p| pick(&sampler, p)).collect(),
            divergences,
        )?)
    };
    Ok(StanFit {
        draws,
        sampler,
        warmup_dropped: parsed[0].warmup_dropped,
    })
}

struct Chain {
    header: Vec<String>,
    rows: Vec<Vec<f64>>,
    warmup_dropped: usize,
}

/// `theta.1.2` as `theta[1,2]`; names without an index unchanged.
fn rename(name: &str) -> String {
    match name.split_once('.') {
        Some((base, idx)) => format!("{base}[{}]", idx.replace('.', ",")),
        None => name.to_string(),
    }
}

/// The value of a configuration comment `#   key = value (Default)`.
fn config<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let rest = line.trim_start_matches('#').trim();
    let (k, v) = rest.split_once('=')?;
    if k.trim() != key {
        return None;
    }
    Some(v.split_whitespace().next().unwrap_or(""))
}

fn parse_chain(text: &str) -> Result<Chain> {
    let mut header: Option<Vec<String>> = None;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut num_warmup = 1000usize;
    let mut save_warmup = false;
    let mut thin = 1usize;
    let mut method: Option<String> = None;
    // Rows before CmdStan's "Adaptation terminated" line are warmup.
    let mut adaptation_at: Option<usize> = None;
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') {
            if header.is_none() {
                if let Some(v) = config(line, "num_warmup") {
                    num_warmup = v.parse().unwrap_or(num_warmup);
                } else if let Some(v) = config(line, "save_warmup") {
                    save_warmup = matches!(v, "1" | "true");
                } else if let Some(v) = config(line, "thin") {
                    thin = v.parse::<usize>().unwrap_or(1).max(1);
                } else if let Some(v) = config(line, "method") {
                    method = Some(v.to_string());
                }
            } else if line.contains("Adaptation terminated") {
                adaptation_at = Some(rows.len());
            }
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        match &header {
            None => header = Some(fields.iter().map(|f| f.to_string()).collect()),
            Some(h) => {
                if fields.len() != h.len() {
                    return Err(Error::Data(format!(
                        "line {} has {} values for {} columns",
                        i + 1,
                        fields.len(),
                        h.len()
                    )));
                }
                let row = fields
                    .iter()
                    .map(|f| {
                        f.parse::<f64>().map_err(|_| {
                            Error::Data(format!("line {}: {f:?} is not a number", i + 1))
                        })
                    })
                    .collect::<Result<Vec<f64>>>()?;
                rows.push(row);
            }
        }
    }
    if let Some(m) = method.filter(|m| m != "sample") {
        return Err(Error::Data(format!(
            "the file holds method={m}, not draws from method=sample"
        )));
    }
    let header = header.ok_or_else(|| Error::Data("no header line".into()))?;
    let warmup = if !save_warmup {
        0
    } else {
        adaptation_at.unwrap_or_else(|| num_warmup.div_ceil(thin))
    };
    if warmup > rows.len() {
        return Err(Error::Data(format!(
            "{warmup} warmup draws expected and {} draws found",
            rows.len()
        )));
    }
    rows.drain(..warmup);
    Ok(Chain {
        header,
        rows,
        warmup_dropped: warmup,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD: &str =
        "# model = m\n# method = sample (Default)\n#   sample\n#     num_warmup = 3\n";

    fn chain(save_warmup: bool, adaptation_line: bool) -> String {
        let mut s = String::from(HEAD);
        s.push_str(&format!("#     save_warmup = {save_warmup}\n"));
        s.push_str("lp__,divergent__,a,b.1.1,b.2.1\n");
        if save_warmup {
            for i in 0..3 {
                s.push_str(&format!("-9,0,{i}.5,9,9\n"));
            }
        }
        if adaptation_line {
            s.push_str("# Adaptation terminated\n# Step size = 0.8\n");
        }
        for i in 0..5 {
            s.push_str(&format!("-1,{},{i},1,2\n", i % 2));
        }
        s.push_str("# \n#  Elapsed Time: 0.01 seconds (Warm-up)\n");
        s
    }

    #[test]
    fn warmup_is_dropped_by_the_adaptation_line_or_the_count() {
        for (save, line) in [(false, true), (true, true), (true, false)] {
            let c = chain(save, line);
            let fit = parse(&[&c, &c]).unwrap();
            assert_eq!(fit.draws.draws_per_chain(), 5, "save {save} line {line}");
            assert_eq!(fit.warmup_dropped, if save { 3 } else { 0 });
            assert_eq!(fit.draws.parameter(1)[0], vec![0.0, 1.0, 2.0, 3.0, 4.0]);
            assert_eq!(fit.draws.divergences(), [2, 2]);
        }
    }

    #[test]
    fn containers_are_renamed_in_column_major_order() {
        let c = chain(false, true);
        let fit = parse(&[&c]).unwrap();
        assert_eq!(fit.draws.names(), ["lp__", "a", "b[1,1]", "b[2,1]"]);
        let (ll, n) = fit.draws.log_likelihood("b").unwrap();
        assert_eq!(n, 2);
        assert_eq!(&ll[..2], [1.0, 2.0]);
    }

    #[test]
    fn mismatched_chains_and_bad_values_are_errors() {
        let c = chain(false, true);
        let short = c.replacen("-1,0,0,1,2\n", "", 1);
        assert!(parse(&[&c, &short]).is_err());
        let other = c.replace("b.2.1", "c");
        assert!(parse(&[&c, &other]).is_err());
        let bad = c.replacen("-1,0,0,1,2", "-1,0,x,1,2", 1);
        assert!(parse(&[&bad]).is_err());
        let opt = c.replace("method = sample", "method = optimize");
        assert!(parse(&[&opt]).is_err());
        assert!(parse(&[]).is_err());
    }

    #[test]
    fn nan_and_infinite_values_parse() {
        let c = chain(false, true).replacen("-1,0,0,1,2", "-1,0,nan,inf,-inf", 1);
        let fit = parse(&[&c]).unwrap();
        assert!(fit.draws.draw(0)[1].is_nan());
        assert_eq!(fit.draws.draw(0)[2], f64::INFINITY);
    }
}
