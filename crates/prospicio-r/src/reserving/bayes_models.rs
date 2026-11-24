//! The Bayesian models built from R's arguments, their one-year settings and their diagnostics.

use super::*;

/// The `BayesOdp` of the one-year views: `priors` are `intercept_sd`,
/// `origin_sd` and `development_sd`; `run` is `chains`, `tune`, `draws`,
/// `n_sims`, `seed`, `target_accept` and `max_depth`.
pub(crate) fn bayes_odp_model(
    priors: &[f64],
    process: &str,
    run: &[f64],
    dependence: SegmentDependence,
) -> Result<BayesOdp> {
    let (&[intercept_sd, origin_sd, development_sd], &[chains, tune, draws, n_sims, seed, ta, md]) =
        (priors, run)
    else {
        return Err(Error::Other(
            "bayes_odp_model takes 3 priors and 7 run settings".into(),
        ));
    };
    let boot = bootstrap(n_sims, seed, process, SegmentDependence::Independent)?;
    Ok(BayesOdp {
        intercept_sd,
        origin_sd,
        development_sd,
        process: boot.process,
        sampler: BayesSampler {
            chains: whole(chains, "chains")? as usize,
            tune: whole(tune, "tune")? as usize,
            draws: whole(draws, "draws")? as usize,
            seed: 0,
            target_accept: ta,
            max_depth: whole(md, "max_depth")?,
        },
        n_sims: boot.n_sims,
        seed: boot.seed,
        dependence,
    })
}

/// The `BayesBf` of `bayes_bf` and `bayes_bf_one_year`: `priors` are
/// `apriori`, `cv` and `development_sd`, `run` is `chains`, `tune`,
/// `draws`, `n_sims`, `seed`, `target_accept` and `max_depth`.
pub(crate) fn bayes_bf_model(priors: &[f64], process: &str, run: &[f64]) -> Result<BayesBf> {
    let (&[apriori, cv, development_sd], &[chains, tune, draws, n_sims, seed, ta, md]) =
        (priors, run)
    else {
        return Err(Error::Other(
            "bayes_bf_model takes 3 priors and 7 run settings".into(),
        ));
    };
    let boot = bootstrap(n_sims, seed, process, SegmentDependence::Independent)?;
    Ok(BayesBf {
        apriori,
        cv,
        development_sd,
        process: boot.process,
        sampler: BayesSampler {
            chains: whole(chains, "chains")? as usize,
            tune: whole(tune, "tune")? as usize,
            draws: whole(draws, "draws")? as usize,
            seed: 0,
            target_accept: ta,
            max_depth: whole(md, "max_depth")?,
        },
        n_sims: boot.n_sims,
        seed: boot.seed,
    })
}

/// The `BayesBfNormalised` of `bayes_bf_normalised` and
/// `bayes_bf_normalised_one_year`: `priors` are
/// `apriori`, `cv`, `development_sd` and `common_cv` (0 for no multiplier
/// common to all origins), `run` is `chains`, `tune`,
/// `draws`, `n_sims`, `seed`, `target_accept` and `max_depth`.
pub(crate) fn bayes_bf_normalised_model(
    priors: &[f64],
    process: &str,
    run: &[f64],
) -> Result<BayesBfNormalised> {
    let (&[apriori, cv, development_sd, common_cv], &[chains, tune, draws, n_sims, seed, ta, md]) =
        (priors, run)
    else {
        return Err(Error::Other(
            "bayes_bf_normalised_model takes 4 priors and 7 run settings".into(),
        ));
    };
    let boot = bootstrap(n_sims, seed, process, SegmentDependence::Independent)?;
    Ok(BayesBfNormalised {
        apriori,
        cv,
        development_sd,
        process: boot.process,
        sampler: BayesSampler {
            chains: whole(chains, "chains")? as usize,
            tune: whole(tune, "tune")? as usize,
            draws: whole(draws, "draws")? as usize,
            seed: 0,
            target_accept: ta,
            max_depth: whole(md, "max_depth")?,
        },
        n_sims: boot.n_sims,
        seed: boot.seed,
        common_cv: (common_cv != 0.0).then_some(common_cv),
    })
}

/// The `HierarchicalBf` of `hierarchical_bf`: `priors` are `apriori`, `cv`,
/// `development_sd`, `tau_scale` and `omega_scale`, `run` is `chains`,
/// `tune`, `draws`, `n_sims`, `seed`, `target_accept` and `max_depth`.
pub(crate) fn hierarchical_bf_model(
    priors: &[f64],
    level_cv: Nullable<f64>,
    process: &str,
    run: &[f64],
) -> Result<HierarchicalBf> {
    let (
        &[apriori, cv, development_sd, tau_scale, omega_scale],
        &[chains, tune, draws, n_sims, seed, ta, md],
    ) = (priors, run)
    else {
        return Err(Error::Other(
            "hierarchical_bf_model takes 5 priors and 7 run settings".into(),
        ));
    };
    let boot = bootstrap(n_sims, seed, process, SegmentDependence::Independent)?;
    Ok(HierarchicalBf {
        apriori,
        level_cv: match level_cv {
            Nullable::NotNull(c) => Some(c),
            Nullable::Null => None,
        },
        cv,
        development_sd,
        tau_scale,
        omega_scale,
        process: boot.process,
        sampler: BayesSampler {
            chains: whole(chains, "chains")? as usize,
            tune: whole(tune, "tune")? as usize,
            draws: whole(draws, "draws")? as usize,
            seed: 0,
            target_accept: ta,
            max_depth: whole(md, "max_depth")?,
        },
        n_sims: boot.n_sims,
        seed: boot.seed,
    })
}

/// The settings of a Bayesian one-year view: `k_threshold`, the `refit`
/// policy by name (`None` for the model's one-year default), the most refits
/// (`NULL` for no limit) and the number
/// of simulations (`NULL` for the model's one-year default).
pub(crate) fn one_year_settings(
    k_threshold: f64,
    refit: Option<&str>,
    max_refits: Nullable<f64>,
    one_year_sims: Nullable<f64>,
) -> Result<BayesOneYear> {
    let refit = match refit {
        None => None,
        Some("refit_flagged") => Some(OneYearRefit::RefitFlagged),
        Some("reweight_only") => Some(OneYearRefit::ReweightOnly),
        Some("refit_all") => Some(OneYearRefit::RefitAll),
        Some("moment_match_then_refit") => Some(OneYearRefit::MomentMatchThenRefit),
        Some(refit) => {
            return Err(Error::Other(format!(
                "refit must be \"moment_match_then_refit\", \"refit_flagged\", \
                 \"reweight_only\" or \"refit_all\", got \"{refit}\""
            )));
        }
    };
    let max_refits = match max_refits {
        Nullable::NotNull(m) => Some(whole(m, "max_refits")? as usize),
        Nullable::Null => None,
    };
    let n_sims = match one_year_sims {
        Nullable::NotNull(n) => Some(whole(n, "one_year_sims")? as usize),
        Nullable::Null => None,
    };
    Ok(BayesOneYear {
        k_threshold,
        refit,
        max_refits,
        n_sims,
        light_refit: None,
        no_light_refit: false,
    })
}

/// A compartmental or hierarchical one-year view's `light_refit`: `NULL`
/// for the model's default (model 2's light refit), the vector `0` to turn
/// it off (the R wrapper sends it for `FALSE`), an empty vector for
/// [`LightRefit::default`] (4 chains of 150 warm-up and 250 kept draws), or
/// `c(chains, tune, draws)`. Returns the sampler and whether the default is
/// turned off.
pub(crate) fn light_refit_arg(
    light_refit: Nullable<Vec<f64>>,
) -> Result<(Option<LightRefit>, bool)> {
    match light_refit {
        Nullable::Null => Ok((None, false)),
        Nullable::NotNull(v) if v.is_empty() => Ok((Some(LightRefit::default()), false)),
        Nullable::NotNull(v) if v.len() == 1 && v[0] == 0.0 => Ok((None, true)),
        Nullable::NotNull(v) if v.len() == 3 => Ok((
            Some(LightRefit {
                chains: whole(v[0], "light_refit chains")? as usize,
                tune: whole(v[1], "light_refit tune")? as usize,
                draws: whole(v[2], "light_refit draws")? as usize,
            }),
            false,
        )),
        Nullable::NotNull(_) => Err(Error::Other(
            "light_refit must be NULL, TRUE, FALSE or c(chains, tune, draws)".into(),
        )),
    }
}

/// The `Meyers` of `meyers` and `meyers_segments`: `priors` are
/// `logelr_mean`, `logelr_sd`, `origin_sd`, `development_sd`, `gamma_sd` and
/// `rho_shape`; `run` is `chains`, `tune`, `draws`, `n_sims`, `seed`,
/// `target_accept` and `max_depth`.
pub(crate) fn meyers_settings(
    model: &str,
    priors: [f64; 6],
    run: [f64; 7],
    dependence: SegmentDependence,
) -> Result<Meyers> {
    let model = MeyersModel::from_name(model)
        .ok_or_else(|| crate::unknown_name("model", model, MeyersModel::NAMES))?;
    let [
        logelr_mean,
        logelr_sd,
        origin_sd,
        development_sd,
        gamma_sd,
        rho_shape,
    ] = priors;
    let [chains, tune, draws, n_sims, seed, target_accept, max_depth] = run;
    let n_sims = whole(n_sims, "n_sims")? as usize;
    if n_sims == 0 {
        return Err(Error::Other("n_sims must be positive".into()));
    }
    Ok(Meyers {
        model,
        logelr_mean,
        logelr_sd,
        origin_sd,
        development_sd,
        gamma_sd,
        rho_shape,
        sampler: BayesSampler {
            chains: whole(chains, "chains")? as usize,
            tune: whole(tune, "tune")? as usize,
            draws: whole(draws, "draws")? as usize,
            seed: 0,
            target_accept,
            max_depth: whole(max_depth, "max_depth")?,
        },
        n_sims,
        seed: whole(seed, "seed")?,
        dependence,
    })
}

/// One row per segment: `list(table, warnings)`, `table` a fit table with
/// the key columns of `totals` and `max_rhat`, `min_ess_bulk`,
/// `min_ess_tail` and `divergences`, `warnings` each segment's messages
/// joined by "; ".
pub(crate) fn segment_diagnostics<T>(
    segments: &SegmentFits<T>,
    totals: FitTable,
    diag: fn(&T) -> &Diagnostics,
) -> List {
    let fits = &segments.fits;
    let col = |g: fn(&Diagnostics) -> f64| fits.iter().map(|f| g(diag(f))).collect::<Vec<f64>>();
    let table = FitTable {
        keys: totals.keys,
        origin: None,
        age: None,
        values: vec![
            ("max_rhat".into(), col(|d| d.max_rhat)),
            ("min_ess_bulk".into(), col(|d| d.min_ess_bulk)),
            ("min_ess_tail".into(), col(|d| d.min_ess_tail)),
            ("divergences".into(), col(|d| d.divergences as f64)),
        ],
    };
    let warnings: Vec<String> = fits.iter().map(|f| diag(f).warnings.join("; ")).collect();
    list!(table = fit_table(table), warnings = warnings)
}

/// A posterior summary as a list of columns, as `bayes_glm_model@summary`.
pub(crate) fn summary_list(s: &[ParamSummary]) -> List {
    let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
    list!(
        parameter = s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
        mean = col(|p| p.mean),
        sd = col(|p| p.sd),
        q05 = col(|p| p.q05),
        q50 = col(|p| p.q50),
        q95 = col(|p| p.q95),
        rhat = col(|p| p.rhat),
        ess_bulk = col(|p| p.ess_bulk),
        ess_tail = col(|p| p.ess_tail)
    )
}
