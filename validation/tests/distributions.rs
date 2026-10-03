//! Distribution parity: moments and quantiles against SciPy; limited
//! expected values and grid masses against 30-digit mpmath references.

use act_prob::{Distribution, Lognormal, Severity};
use act_validation::{check, reference};

#[test]
fn lognormal_matches_scipy() {
    let cases: Vec<_> = reference("distributions_scipy.csv")
        .into_iter()
        .filter(|c| c.get("distribution") == "lognormal")
        .collect();
    check(&cases, |c| {
        let d = Lognormal::new(c.param("params", "meanlog"), c.param("params", "sdlog")).ok()?;
        let arg = c.number("arg");
        match c.get("quantity") {
            "mean" => Some(d.mean()),
            "variance" => Some(d.variance()),
            "cdf" => Some(d.cdf(arg?)),
            "quantile" => d.quantile(arg?).ok(),
            _ => None,
        }
    });
}

#[test]
fn gamma_matches_scipy() {
    use act_prob::Gamma;
    let cases: Vec<_> = reference("distributions_scipy.csv")
        .into_iter()
        .filter(|c| c.get("distribution") == "gamma")
        .collect();
    check(&cases, |c| {
        let d = Gamma::new(c.param("params", "shape"), c.param("params", "scale")).ok()?;
        let arg = c.number("arg");
        match c.get("quantity") {
            "mean" => Some(d.mean()),
            "variance" => Some(d.variance()),
            "cdf" => Some(d.cdf(arg?)),
            "survival" => Some(d.survival(arg?)),
            "quantile" => d.quantile(arg?).ok(),
            _ => None,
        }
    });
}

#[test]
fn severities_match_integration() {
    use act_prob::Gamma;
    let cases = reference("severity_mpmath.csv");
    check(&cases, |c| {
        let d: Box<dyn Severity> = match c.get("distribution") {
            "lognormal" => Box::new(
                Lognormal::new(c.param("params", "meanlog"), c.param("params", "sdlog")).ok()?,
            ),
            "gamma" => {
                Box::new(Gamma::new(c.param("params", "shape"), c.param("params", "scale")).ok()?)
            }
            _ => return None,
        };
        let arg = c.number("arg")?;
        match c.get("quantity") {
            "lev" => Some(d.lev(arg)),
            "stop_loss" => Some(d.stop_loss(arg)),
            "layer" => Some(d.layer(arg, c.number("arg2")?)),
            _ => None,
        }
    });
}

#[test]
fn lognormal_grids_match_textbook_masses() {
    use act_prob::Grid;
    let cases = reference("grid_mpmath.csv");
    check(&cases, |c| {
        let d = Lognormal::new(c.param("params", "meanlog"), c.param("params", "sdlog")).ok()?;
        let step = c.number("step")?;
        let points = c.number("points")? as usize;
        let (grid, _) = match c.get("method") {
            "local_moment" => Grid::local_moment(&d, step, points),
            "rounding" => Grid::rounding(&d, step, points),
            "lower" => Grid::lower(&d, step, points),
            _ => return None,
        }
        .ok()?;
        grid.probs().get(c.number("index")? as usize).copied()
    });
}

#[test]
fn claim_counts_match_scipy() {
    use act_prob::{Binomial, Counting, NegativeBinomial, Poisson};
    let cases: Vec<_> = reference("distributions_scipy.csv")
        .into_iter()
        .filter(|c| {
            matches!(
                c.get("distribution"),
                "poisson" | "negative_binomial" | "binomial"
            )
        })
        .collect();
    check(&cases, |c| {
        let n: Box<dyn Counting> = match c.get("distribution") {
            "poisson" => Box::new(Poisson::new(c.param("params", "lambda")).ok()?),
            "binomial" => {
                Box::new(Binomial::new(c.param("params", "n") as u64, c.param("params", "p")).ok()?)
            }
            _ => Box::new(
                NegativeBinomial::new(c.param("params", "r"), c.param("params", "beta")).ok()?,
            ),
        };
        let arg = c.number("arg");
        match c.get("quantity") {
            "mean" => Some(n.mean()),
            "variance" => Some(n.variance()),
            "pmf" => Some(n.pmf(arg? as u64)),
            "cdf" => Some(n.cdf(arg? as u64)),
            "quantile" => n.quantile(arg?).ok().map(|k| k as f64),
            _ => None,
        }
    });
}

#[test]
fn distortions_match_integration() {
    use act_prob::{Distortion, Grid};
    // The discrete distribution in validation/scripts/mpmath_distortion.py.
    let values = [0.0, 1.0, 2.0, 5.0, 10.0, 100.0];
    let probs = [0.5, 0.2, 0.15, 0.1, 0.0499999999, 1e-10];
    let cases = reference("distortion_mpmath.csv");
    check(&cases, |c| {
        let a = c.number("arg")?;
        let d = match c.get("quantity") {
            "tvar" => Distortion::tvar(a),
            "wang" => Distortion::wang(a),
            "proportional_hazard" => Distortion::proportional_hazard(a),
            "dual_power" => Distortion::dual_power(a),
            _ => return None,
        }
        .ok()?;
        match c.get("distribution") {
            "discrete" => Some(d.apply_discrete(&values, &probs)),
            "lognormal" => {
                let ln = Lognormal::new(c.param("params", "meanlog"), c.param("params", "sdlog"))
                    .ok()?;
                let step = c.param("params", "step");
                let points = c.param("params", "points") as usize;
                let (grid, _) = Grid::local_moment(&ln, step, points).ok()?;
                Some(grid.distortion(&d))
            }
            _ => None,
        }
    });
}

#[test]
fn special_functions_match_scipy() {
    use act_math::special::{beta_inc, gamma_inc, student_t_cdf};
    let cases = reference("special_scipy.csv");
    check(&cases, |c| {
        let x = c.number("arg")?;
        match c.get("distribution") {
            "beta_inc" => Some(beta_inc(c.param("params", "a"), c.param("params", "b"), x)),
            "student_t" => Some(student_t_cdf(x, c.param("params", "nu"))),
            "gamma_p" => Some(gamma_inc(c.param("params", "a"), x).0),
            "gamma_q" => Some(gamma_inc(c.param("params", "a"), x).1),
            _ => None,
        }
    });
}

#[test]
fn gpd_fits_match_exact_likelihood() {
    use act_prob::evt::Gpd;
    let cases = reference("gpd_mpmath.csv");
    check(&cases, |c| {
        let (xi0, beta0) = (c.param("params", "xi0"), c.param("params", "beta0"));
        let n = c.param("params", "n") as usize;
        // The data set in validation/scripts/mpmath_gpd.py.
        let truth = Gpd::new(xi0, beta0).ok()?;
        let y: Vec<f64> = (1..=n)
            .map(|i| truth.quantile((i as f64 - 0.5) / n as f64))
            .collect::<Result<_, _>>()
            .ok()?;
        let fit = Gpd::fit(&y).ok()?;
        match c.get("quantity") {
            "xi" => Some(fit.xi()),
            "beta" => Some(fit.beta()),
            _ => None,
        }
    });
}

/// A Pareto (optionally truncated) from `t=..;alpha=..[;truncation=..]`.
fn pareto_from(c: &act_validation::Case) -> Option<act_prob::Pareto> {
    let p = act_prob::Pareto::new(c.param("params", "t"), c.param("params", "alpha")).ok()?;
    if c.get("params").contains("truncation") {
        p.truncated(c.param("params", "truncation")).ok()
    } else {
        Some(p)
    }
}

/// A piecewise Pareto from `params` such as
/// `t=1000|2000;alpha=1.0|2.0;truncation=5000;type=lp`.
fn piecewise_pareto_from(c: &act_validation::Case) -> Option<act_prob::PiecewisePareto> {
    let mut fields = std::collections::HashMap::new();
    for kv in c.get("params").split(';') {
        let (k, v) = kv.split_once('=')?;
        fields.insert(k, v);
    }
    let list = |k: &str| -> Option<Vec<f64>> {
        fields.get(k)?.split('|').map(|v| v.parse().ok()).collect()
    };
    let pp = act_prob::PiecewisePareto::new(list("t")?, list("alpha")?).ok()?;
    match fields.get("truncation") {
        None => Some(pp),
        Some(tr) => {
            let kind = match *fields.get("type")? {
                "lp" => act_prob::Truncation::LastPiece,
                "wd" => act_prob::Truncation::WholeDistribution,
                _ => return None,
            };
            pp.truncated(tr.parse().ok()?, kind).ok()
        }
    }
}

/// Cover and attachment from `arg` ("inf" for unlimited) and `arg2`.
fn layer_args(c: &act_validation::Case) -> Option<(f64, f64)> {
    let cover = match c.get("arg") {
        "inf" => f64::INFINITY,
        s => s.parse().ok()?,
    };
    Some((cover, c.get("arg2").parse().ok()?))
}

#[test]
fn layer_moments_match_integration() {
    let cases = reference("layer_moments_mpmath.csv");
    check(&cases, |c| {
        let (cover, att) = layer_args(c)?;
        let sev: Box<dyn Severity> = match c.get("distribution") {
            "pareto" => Box::new(pareto_from(c)?),
            "piecewise_pareto" => Box::new(piecewise_pareto_from(c)?),
            "log_affine_pareto" => Box::new(
                act_prob::LogAffinePareto::new(
                    c.param("params", "t"),
                    c.param("params", "alpha_0"),
                    c.param("params", "gamma"),
                )
                .ok()?,
            ),
            "gpd" => Box::new(
                act_prob::evt::Gpd::new(c.param("params", "xi"), c.param("params", "beta"))
                    .ok()?
                    .shifted(c.param("params", "location"))
                    .ok()?,
            ),
            "lognormal" => Box::new(
                Lognormal::new(c.param("params", "meanlog"), c.param("params", "sdlog")).ok()?,
            ),
            _ => return None,
        };
        match c.get("quantity") {
            "layer" => Some(sev.layer(cover, att)),
            "layer_second_moment" => Some(sev.layer_second_moment(cover, att)),
            _ => None,
        }
    });
}

#[test]
fn pareto_matches_r() {
    let cases = reference("pareto_r.csv");
    check(&cases, |c| {
        let p = pareto_from(c)?;
        match c.get("quantity") {
            "cdf" => Some(p.cdf(c.number("arg")?)),
            "quantile" => p.quantile(c.number("arg")?).ok(),
            q => {
                let (cover, att) = layer_args(c)?;
                match q {
                    "layer" => Some(p.layer(cover, att)),
                    "layer_second_moment" => Some(p.layer_second_moment(cover, att)),
                    "layer_variance" => Some(p.layer_variance(cover, att)),
                    _ => None,
                }
            }
        }
    });
}

#[test]
fn piecewise_pareto_matches_r() {
    let cases = reference("piecewise_pareto_r.csv");
    check(&cases, |c| {
        let p = piecewise_pareto_from(c)?;
        match c.get("quantity") {
            "cdf" => Some(p.cdf(c.number("arg")?)),
            "quantile" => p.quantile(c.number("arg")?).ok(),
            q => {
                let (cover, att) = layer_args(c)?;
                match q {
                    "layer" => Some(p.layer(cover, att)),
                    "layer_second_moment" => Some(p.layer_second_moment(cover, att)),
                    "layer_variance" => Some(p.layer_variance(cover, att)),
                    _ => None,
                }
            }
        }
    });
}

#[test]
fn gen_pareto_matches_r() {
    let cases = reference("gen_pareto_r.csv");
    check(&cases, |c| {
        let g = act_prob::evt::Gpd::riegel(
            c.param("params", "t"),
            c.param("params", "alpha_ini"),
            c.param("params", "alpha_tail"),
        )
        .ok()?;
        match c.get("quantity") {
            "cdf" => Some(g.cdf(c.number("arg")?)),
            "quantile" => g.quantile(c.number("arg")?).ok(),
            q => {
                let (cover, att) = layer_args(c)?;
                match q {
                    "layer" => Some(g.layer(cover, att)),
                    "layer_second_moment" => Some(g.layer_second_moment(cover, att)),
                    "layer_variance" => Some(g.layer_variance(cover, att)),
                    _ => None,
                }
            }
        }
    });
}

#[test]
fn pareto_fits_match_r() {
    use act_prob::{LargeLosses, Pareto, PiecewisePareto, Truncation};
    // The data in validation/scripts/r_pareto_fit.R.
    let losses = vec![
        1100.0, 1300.0, 1750.0, 2000.0, 2600.0, 3500.0, 4100.0, 5200.0, 7000.0, 9000.0, 12000.0,
        18000.0, 25000.0, 40000.0,
    ];
    let cases = reference("pareto_fit_r.csv");
    check(&cases, |c| {
        let fields: std::collections::HashMap<&str, &str> = c
            .get("params")
            .split(';')
            .filter_map(|kv| kv.split_once('='))
            .collect();
        let list = |k: &str| -> Option<Vec<f64>> {
            match *fields.get(k)? {
                "" => Some(vec![]),
                v => v.split('|').map(|x| x.parse().ok()).collect(),
            }
        };
        let mut data = LargeLosses::new(losses.clone()).ok()?;
        let (r, cens, w) = (list("r")?, list("c")?, list("w")?);
        if !r.is_empty() {
            data = data.reporting_thresholds(r).ok()?;
        }
        if !cens.is_empty() {
            data = data
                .censored(cens.iter().map(|&x| x == 1.0).collect())
                .ok()?;
        }
        if !w.is_empty() {
            data = data.weights(w).ok()?;
        }
        let truncation: Option<f64> = fields.get("truncation").and_then(|v| v.parse().ok());
        let t = list("t")?;
        let index = c.number("index")? as usize;
        match c.get("model") {
            "pareto" => Some(Pareto::fit(t[0], &data, truncation).ok()?.alpha()),
            "gen_pareto" => {
                let g = act_prob::evt::Gpd::fit_riegel(t[0], &data).ok()?;
                // ξ = 1/α_tail, β = t/α_ini.
                let alphas = [t[0] / g.beta(), 1.0 / g.xi()];
                alphas.get(index).copied()
            }
            "piecewise_pareto" => {
                let fit = PiecewisePareto::fit(
                    t,
                    &data,
                    truncation.map(|tr| {
                        let kind = match fields.get("type") {
                            Some(&"wd") => Truncation::WholeDistribution,
                            _ => Truncation::LastPiece,
                        };
                        (tr, kind)
                    }),
                )
                .ok()?;
                fit.alphas().get(index).copied()
            }
            _ => None,
        }
    });
}

#[test]
fn log_affine_pareto_matches_r() {
    use act_prob::LogAffinePareto;
    let cases = reference("local_pareto_r.csv");
    check(&cases, |c| {
        let params = c.get("params");
        let d = if params.contains("delta") {
            LogAffinePareto::from_delta(
                c.param("params", "t"),
                c.param("params", "alpha_0"),
                c.param("params", "delta"),
            )
        } else {
            LogAffinePareto::new(
                c.param("params", "t"),
                c.param("params", "alpha_0"),
                c.param("params", "gamma"),
            )
        }
        .ok()?;
        match c.get("quantity") {
            "cdf" => Some(d.cdf(c.number("arg")?)),
            "quantile" => d.quantile(c.number("arg")?).ok(),
            q => {
                let (cover, att) = layer_args(c)?;
                match q {
                    "layer" => Some(d.layer(cover, att)),
                    "layer_second_moment" => Some(d.layer_second_moment(cover, att)),
                    "layer_variance" => Some(d.layer_variance(cover, att)),
                    _ => None,
                }
            }
        }
    });
}

#[test]
fn allocations_match_numpy() {
    use act_prob::capital::AllocationMethod;
    use act_prob::{Distortion, KeyValue, PredictiveDistribution, Provenance};
    // The draws in validation/scripts/numpy_allocation.py.
    let (n, m) = (400usize, 4usize);
    let mut draws = Vec::with_capacity(n * m);
    for i in 0..n as u64 {
        let a = (i * 37) % 101;
        draws.extend(
            [
                a,
                (i * 53) % 97 + a / 2,
                (i * i) % 89,
                3 * ((i * 29) % 41) + (a / 10).pow(2),
            ]
            .map(|v| v as f64),
        );
    }
    let pd = PredictiveDistribution::from_draws(
        vec!["line".into()],
        (0..m).map(|j| vec![KeyValue::from(j as i64)]).collect(),
        draws,
        Provenance::new("validation"),
    )
    .unwrap();
    let cases = reference("allocation_numpy.csv");
    check(&cases, |c| {
        let arg = c.number("arg")?;
        let d = match c.get("distortion") {
            "tvar" => Distortion::tvar(arg),
            "proportional_hazard" => Distortion::proportional_hazard(arg),
            "dual_power" => Distortion::dual_power(arg),
            _ => return None,
        }
        .ok()?;
        let method = match c.get("method") {
            "covariance" => AllocationMethod::Covariance,
            "proportional" => AllocationMethod::Proportional,
            "marginal" => AllocationMethod::Marginal,
            "shapley" => AllocationMethod::Shapley,
            _ => AllocationMethod::Euler,
        };
        let a = pd.capital(&d, method).ok()?;
        match c.get("method") {
            "total" => Some(a.total),
            "standalone" => Some(a.standalone[c.number("component")? as usize]),
            _ => Some(a.allocated[c.number("component")? as usize]),
        }
    });
}

#[test]
fn tweedie_matches_mpmath() {
    use act_prob::Tweedie;
    let cases = reference("tweedie_mpmath.csv");
    check(&cases, |c| {
        let y = Tweedie::new(
            c.param("params", "mu"),
            c.param("params", "phi"),
            c.param("params", "p"),
        )
        .ok()?;
        let arg = c.number("arg")?;
        match c.get("quantity") {
            "ln_pdf" => Some(y.ln_pdf(arg)),
            "cdf" => Some(y.cdf(arg)),
            "survival" => Some(y.survival(arg)),
            "lev" => Some(y.lev(arg)),
            _ => None,
        }
    });
}
