//! Shared glue: error mapping, the draws of a fit, option names and their parsers, and the arguments of the bootstraps and one-year views.

use super::*;

/// The prefix that marks a setting the method does not support
/// ([`prospicio_reserving::Error::is_not_supported`]) in an error message.
/// extendr's error carries only a message, so `reserving_result()` in
/// `R/prospicio/R/reserving.R` reads the prefix, removes it and signals the
/// condition class `prospicio_not_supported` (#580, `reserving-v02.md`
/// decision 29).
pub(crate) const NOT_SUPPORTED_PREFIX: &str = "[prospicio_not_supported] ";

/// A reserving error as an R error message, marked when the setting is not
/// supported.
pub(crate) fn to_r(e: prospicio_reserving::Error) -> Error {
    if e.is_not_supported() {
        Error::Other(format!("{NOT_SUPPORTED_PREFIX}{e}"))
    } else {
        Error::Other(e.to_string())
    }
}

/// The key an R list or vector names, or `None` for NULL.
fn origin_key(origin: Nullable<List>) -> Result<Option<ComponentKey>> {
    match origin {
        Nullable::NotNull(key) => Ok(Some(key_from_list(key)?)),
        Nullable::Null => Ok(None),
    }
}

/// [`UltimateDraws::ultimates`] for R.
pub(crate) fn ultimates_of(fit: &impl UltimateDraws) -> Result<PredictiveDistribution> {
    Ok(PredictiveDistribution {
        inner: fit.ultimates().map_err(to_r)?,
    })
}

/// One origin's ultimate draws, or the total's with no origin.
pub(crate) fn ultimate_draws_of(
    fit: &impl UltimateDraws,
    origin: Nullable<List>,
) -> Result<Vec<f64>> {
    match origin_key(origin)? {
        None => fit.total_ultimate_draws(),
        Some(key) => fit.ultimate_draws(&key),
    }
    .map_err(to_r)
}

/// One origin's loss ratio draws, or the total's with no origin.
pub(crate) fn loss_ratio_draws_of(
    fit: &impl UltimateDraws,
    origin: Nullable<List>,
    premium: f64,
) -> Result<Vec<f64>> {
    match origin_key(origin)? {
        None => fit.total_loss_ratio_draws(premium),
        Some(key) => fit.loss_ratio_draws(&key, premium),
    }
    .map_err(to_r)
}

/// The pattern draws' segment label as the binding takes it.
fn label_refs(segment: &[String]) -> Vec<&str> {
    segment.iter().map(String::as_str).collect()
}

/// The ages of a segment's pattern draws in months.
pub(crate) fn pattern_ages_of(
    patterns: &DevelopmentPatterns,
    segment: &[String],
) -> Result<Vec<f64>> {
    let seg = patterns.segment(&label_refs(segment)).map_err(to_r)?;
    Ok(seg.ages().iter().map(|&a| f64::from(a)).collect())
}

/// [`DevelopmentPatterns::pattern_draws`] flattened row-major.
pub(crate) fn pattern_draws_of(
    patterns: &DevelopmentPatterns,
    segment: &[String],
) -> Result<Vec<f64>> {
    let rows = patterns.pattern_draws(&label_refs(segment)).map_err(to_r)?;
    Ok(rows.into_iter().flatten().collect())
}

/// The monthly basis R names: NULL or `"origin"` for the origin's own
/// basis with the default reading, `"origin_ln_linear"`, or an
/// `underwriting_basis()`.
fn monthly_basis(value: &Robj) -> Result<MonthlyBasis> {
    let bad = |what: &str| {
        Error::Other(format!(
            "basis must be NULL, \"origin\", \"origin_ln_linear\" or an underwriting_basis(){what}"
        ))
    };
    if value.is_null() {
        return Ok(MonthlyBasis::default());
    }
    if let Some(name) = value.as_str() {
        return match name {
            "origin" => Ok(MonthlyBasis::default()),
            "origin_ln_linear" => Ok(MonthlyBasis::Origin(AnnualReading::LnLinear)),
            _ => Err(bad(&format!(", got {name:?}"))),
        };
    }
    underwriting_basis(value)
        .map(MonthlyBasis::Underwriting)
        .map_err(|_| bad(""))
}

/// [`DevelopmentPatterns::monthly`] flattened row-major.
pub(crate) fn monthly_pattern_draws_of(
    patterns: &DevelopmentPatterns,
    segment: &[String],
    months: f64,
    basis: &Robj,
    monotone: &str,
) -> Result<Vec<f64>> {
    if !(months >= 1.0 && months.fract() == 0.0 && months <= f64::from(u32::MAX)) {
        return Err(Error::Other(
            "months must be a whole number, at least 1".into(),
        ));
    }
    let monotone = match monotone {
        "refuse" => Monotone::Refuse,
        "running_max" => Monotone::RunningMax,
        _ => {
            return Err(Error::Other(format!(
                "monotone must be \"refuse\" or \"running_max\", got {monotone:?}"
            )));
        }
    };
    let rows = patterns
        .monthly(
            &label_refs(segment),
            &monthly_basis(basis)?,
            months as usize,
            monotone,
        )
        .map_err(to_r)?;
    Ok(rows.into_iter().flatten().collect())
}

pub(crate) fn grain(name: &str, arg: &str) -> Result<Grain> {
    Grain::from_name(&name.trim().to_ascii_lowercase())
        .ok_or_else(|| crate::unknown_name(arg, name, Grain::NAMES))
}

pub(crate) fn development(
    average: &str,
    sigma_interpolation: &str,
    excl: &Robj,
    pattern: Option<SelectedPattern>,
) -> Result<Development> {
    let average = Average::from_name(average)
        .ok_or_else(|| crate::unknown_name("average", average, Average::NAMES))?;
    let sigma_interpolation =
        SigmaInterpolation::from_name(sigma_interpolation).ok_or_else(|| {
            crate::unknown_name(
                "sigma_interpolation",
                sigma_interpolation,
                SigmaInterpolation::NAMES,
            )
        })?;
    let (n_periods, drop_high, drop_low) = exclusions(excl)?;
    Ok(Development {
        average,
        sigma_interpolation,
        pattern,
        n_periods,
        drop_high,
        drop_low,
    })
}

/// The link ratio exclusions as `development_args()` in R encodes them: a
/// list of `n_periods` (`NULL` or a whole number), `drop_high` and
/// `drop_low` (whole numbers from 0). Whether the combination is allowed
/// for a method is the library's to say.
pub(crate) fn exclusions(excl: &Robj) -> Result<(Option<usize>, usize, usize)> {
    let count = |value: &Robj, name: &str| -> Result<usize> {
        let x = value
            .as_real()
            .ok_or_else(|| Error::Other(format!("{name} must be a single number")))?;
        usize::try_from(whole(x, name)?)
            .map_err(|_| Error::Other(format!("{name} must be a whole number from 0")))
    };
    let list = List::try_from(excl)
        .map_err(|_| Error::Other("the link ratio exclusions must be a list".into()))?;
    let (mut n_periods, mut drop_high, mut drop_low) = (None, 0, 0);
    for (name, value) in list.iter() {
        match name {
            "n_periods" if !value.is_null() => n_periods = Some(count(&value, name)?),
            "drop_high" => drop_high = count(&value, name)?,
            "drop_low" => drop_low = count(&value, name)?,
            _ => {}
        }
    }
    Ok((n_periods, drop_high, drop_low))
}

/// A selected pattern as `ldf_pattern()` in R encodes it: `NULL`, or a
/// list of `kind` ("ldf", "cdf" or "percent_developed"), `ages` and
/// `values`, and `weights` with `weight_ages` (`NULL` for one weight on
/// every selected factor), `origin` (`NULL`, or the grain of the origin
/// periods the pattern was measured on, averaged over the periods of
/// `target` by `SelectedPattern::averaged_over_origins`), and
/// `underwriting` (`NULL`, or a list of `writing`, `policy_months` and
/// `reading` (`"weibull"` when missing): the
/// underwriting basis the pattern is converted to from accident periods of
/// `origin`, or of `target` without it, by
/// `SelectedPattern::to_underwriting_basis`, its writing no longer than a
/// period of `target`, `UnderwritingBasis::check_within`).
pub(crate) fn selected_pattern(pattern: &Robj, target: Grain) -> Result<Option<SelectedPattern>> {
    if pattern.is_null() {
        return Ok(None);
    }
    let list = List::try_from(pattern)
        .map_err(|_| Error::Other("the selected pattern must be a list".into()))?;
    let mut kind = None;
    let (mut ages, mut values) = (Vec::new(), Vec::new());
    let (mut weight_ages, mut weights) = (None, vec![1.0]);
    let mut origin = None;
    let mut underwriting = None;
    for (name, value) in list.iter() {
        match name {
            "kind" => kind = value.as_str().map(str::to_owned),
            "origin" if !value.is_null() => {
                origin = Some(
                    value
                        .as_str()
                        .ok_or_else(|| Error::Other("ldf_origin must be a single string".into()))?
                        .to_owned(),
                )
            }
            "underwriting" if !value.is_null() => underwriting = Some(underwriting_basis(&value)?),
            "ages" => ages = value.as_real_vector().unwrap_or_default(),
            "values" => values = value.as_real_vector().unwrap_or_default(),
            "weight_ages" => weight_ages = value.as_real_vector(),
            "weights" => weights = value.as_real_vector().unwrap_or_default(),
            _ => {}
        }
    }
    let kind = kind
        .as_deref()
        .and_then(PatternKind::from_name)
        .ok_or_else(|| {
            crate::unknown_name(
                "ldf_kind",
                kind.as_deref().unwrap_or(""),
                PatternKind::NAMES,
            )
        })?;
    let pairs = |ages: &[f64], values: &[f64], what: &str| -> Result<Vec<(Lag, f64)>> {
        if ages.len() != values.len() {
            return Err(Error::Other(format!(
                "{what}: {} ages but {} values",
                ages.len(),
                values.len()
            )));
        }
        ages.iter()
            .zip(values)
            .map(|(&a, &v)| Ok((age(a)?, v)))
            .collect()
    };
    let weight = match weight_ages {
        None => match weights.as_slice() {
            [w] => PatternWeight::All(*w),
            _ => return Err(Error::Other("ldf_weight must be a single number".into())),
        },
        Some(weight_ages) => PatternWeight::ByAge(pairs(&weight_ages, &weights, "ldf_weight")?),
    };
    let selected = SelectedPattern::new(kind, pairs(&ages, &values, "ldf")?).with_weight(weight);
    let origin = origin.map(|g| grain(&g, "ldf_origin")).transpose()?;
    Ok(Some(match (underwriting, origin) {
        (Some(basis), from) => {
            basis.check_within(target).map_err(to_r)?;
            selected
                .to_underwriting_basis(from.unwrap_or(target), &basis)
                .map_err(to_r)?
        }
        (None, Some(g)) => selected.averaged_over_origins(g, target).map_err(to_r)?,
        (None, None) => selected,
    }))
}

/// The chain ladder that gives a method its development pattern, with the
/// selected pattern `ldf` (as `selected_pattern` reads it, for origins of
/// `target`).
pub(crate) fn pattern(
    average: &str,
    sigma_interpolation: &str,
    excl: &Robj,
    tail: Robj,
    ldf: &Robj,
    target: Grain,
) -> Result<ChainLadder> {
    Ok(ChainLadder {
        development: development(
            average,
            sigma_interpolation,
            excl,
            selected_pattern(ldf, target)?,
        )?,
        tail: tail_arg(&tail)?,
    })
}

/// An `apriori` argument from R: one number for every origin, an unnamed
/// vector of one per origin in origin order (oldest first), or a vector
/// named by origin label (`"2021"`, `"2021Q3"`), as `apriori_arg()` in
/// `reserving.R` passes it.
pub(crate) fn apriori_arg(apriori: &Robj) -> Result<Apriori> {
    let values = apriori
        .as_real_vector()
        .ok_or_else(|| Error::Other("apriori must be a number or a numeric vector".into()))?;
    match apriori.names() {
        Some(names) => {
            let names: Vec<String> = names.map(str::to_string).collect();
            Apriori::from_labels(names.iter().zip(values)).map_err(to_r)
        }
        None if values.len() == 1 => Ok(Apriori::All(values[0])),
        None => Ok(Apriori::InOrder(values)),
    }
}

/// The method of `odp_one_year()` and `mack_one_year()`, by name, with
/// the settings it reads and `chain_ladder` its development pattern.
pub(crate) fn one_year_method(
    method: &str,
    exposure: Nullable<String>,
    apriori: Robj,
    n_iters: f64,
    trend: f64,
    decay: f64,
    chain_ladder: ChainLadder,
) -> Result<OneYearMethod> {
    let exposure = match exposure {
        Nullable::NotNull(e) => Some(e),
        Nullable::Null => None,
    };
    let needs = |e: Option<String>| {
        e.ok_or_else(|| {
            Error::Other(format!(
                "method \"{method}\" needs an exposure column: pass exposure = ..."
            ))
        })
    };
    Ok(match method {
        "chain_ladder" => {
            if exposure.is_some() {
                return Err(Error::Other(
                    "method \"chain_ladder\" takes no exposure column".into(),
                ));
            }
            OneYearMethod::ChainLadder(chain_ladder)
        }
        "expected_loss" => OneYearMethod::ExpectedLoss(
            ExpectedLoss {
                apriori: apriori_arg(&apriori)?,
                chain_ladder,
            },
            needs(exposure)?,
        ),
        "bornhuetter_ferguson" => OneYearMethod::BornhuetterFerguson(
            BornhuetterFerguson {
                apriori: apriori_arg(&apriori)?,
                chain_ladder,
            },
            needs(exposure)?,
        ),
        "benktander" => {
            let n_iters = usize::try_from(whole(n_iters, "n_iters")?)
                .map_err(|_| Error::Other(format!("n_iters {n_iters} is too large")))?;
            OneYearMethod::Benktander(
                Benktander {
                    apriori: apriori_arg(&apriori)?,
                    n_iters,
                    chain_ladder,
                },
                needs(exposure)?,
            )
        }
        "cape_cod" => OneYearMethod::CapeCod(
            CapeCod {
                trend,
                decay,
                chain_ladder,
            },
            needs(exposure)?,
        ),
        _ => {
            return Err(Error::Other(format!(
                "method must be \"chain_ladder\", \"expected_loss\", \
                 \"bornhuetter_ferguson\", \"benktander\" or \"cape_cod\", got \"{method}\""
            )));
        }
    })
}

/// The dependence between segments named `dependence` ("independent",
/// "synchronized" or "rank_correlation"), with the row-major Spearman
/// matrix that "rank_correlation" needs and the others refuse.
pub(crate) fn segment_dependence(
    dependence: &str,
    spearman: Nullable<Vec<f64>>,
) -> Result<SegmentDependence> {
    let named = SegmentDependence::from_name(dependence)
        .ok_or_else(|| crate::unknown_name("dependence", dependence, SegmentDependence::NAMES))?;
    match (named, spearman) {
        (SegmentDependence::RankCorrelation { .. }, Nullable::NotNull(spearman)) => {
            Ok(SegmentDependence::RankCorrelation { spearman })
        }
        (SegmentDependence::RankCorrelation { .. }, Nullable::Null) => Err(Error::Other(
            "dependence = \"rank_correlation\" needs a spearman matrix".into(),
        )),
        (_, Nullable::NotNull(_)) => Err(Error::Other(
            "spearman is taken only with dependence = \"rank_correlation\"".into(),
        )),
        (d, Nullable::Null) => Ok(d),
    }
}

/// Mack's bootstrap of `mack_bootstrap()` and `mack_one_year()`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mack_bootstrap(
    n_sims: f64,
    seed: f64,
    process: &str,
    average: &str,
    sigma_interpolation: &str,
    excl: &Robj,
    centre_residuals: bool,
    dependence: SegmentDependence,
    pattern: Option<SelectedPattern>,
) -> Result<MackBootstrap> {
    let n_sims = whole(n_sims, "n_sims")? as usize;
    if n_sims == 0 {
        return Err(Error::Other("n_sims must be positive".into()));
    }
    let process = MackProcess::from_name(process)
        .ok_or_else(|| crate::unknown_name("process", process, MackProcess::NAMES))?;
    Ok(MackBootstrap {
        n_sims,
        seed: whole(seed, "seed")?,
        process,
        development: development(average, sigma_interpolation, excl, pattern)?,
        centre_residuals,
        dependence,
        ..Default::default()
    })
}

/// The ODP bootstrap of `odp_bootstrap()` and `odp_one_year()`.
pub(crate) fn bootstrap(
    n_sims: f64,
    seed: f64,
    process: &str,
    dependence: SegmentDependence,
) -> Result<OdpBootstrap> {
    let n_sims = whole(n_sims, "n_sims")? as usize;
    if n_sims == 0 {
        return Err(Error::Other("n_sims must be positive".into()));
    }
    let process = ProcessDistribution::from_name(process)
        .ok_or_else(|| crate::unknown_name("process", process, ProcessDistribution::NAMES))?;
    Ok(OdpBootstrap {
        n_sims,
        seed: whole(seed, "seed")?,
        process,
        dependence,
        ..Default::default()
    })
}

pub(crate) fn months(years: &[i32], months: &[i32], arg: &str) -> Result<Vec<Month>> {
    if years.len() != months.len() {
        return Err(Error::Other(format!(
            "{arg}: {} years but {} months",
            years.len(),
            months.len()
        )));
    }
    years
        .iter()
        .zip(months)
        .map(|(&y, &m)| {
            let m = u8::try_from(m)
                .map_err(|_| Error::Other(format!("{arg}: month {m} is not in 1..12")))?;
            Month::new(y, m).map_err(crate::to_r)
        })
        .collect()
}

pub(crate) fn growth_curve(name: &str) -> Result<GrowthCurve> {
    GrowthCurve::from_name(name)
        .ok_or_else(|| crate::unknown_name("curve", name, GrowthCurve::NAMES))
}

pub(crate) fn growth_curve_name(curve: GrowthCurve) -> &'static str {
    curve.name()
}

/// Clark's `max_age`: R passes `Inf` to develop to infinity.
pub(crate) fn clark_max_age(x: f64) -> Option<f64> {
    (x != f64::INFINITY).then_some(x)
}

pub(crate) fn age(x: f64) -> Result<Lag> {
    Lag::try_from(whole(x, "development")?)
        .map_err(|_| Error::Other(format!("development age {x} is too large")))
}

/// An optional age in months named `name`; `NULL` is not given.
pub(crate) fn optional_age(x: Nullable<f64>, name: &str) -> Result<Option<Lag>> {
    match x {
        Nullable::NotNull(a) => Lag::try_from(whole(a, name)?)
            .map(Some)
            .map_err(|_| Error::Other(format!("{name} {a} is too large"))),
        Nullable::Null => Ok(None),
    }
}

pub(crate) fn optional(x: Nullable<f64>) -> Option<f64> {
    match x {
        Nullable::NotNull(v) => Some(v),
        Nullable::Null => None,
    }
}

/// Key columns as a named list of character vectors.
pub(crate) fn key_columns(keys: Vec<(String, Vec<String>)>) -> List {
    let (names, values): (Vec<String>, Vec<Vec<String>>) = keys.into_iter().unzip();
    List::from_names_and_values(names, values).expect("one name per key")
}

/// Which factors a one-year bootstrap simulates: with `bootstrap_ldf` the
/// method's pattern (the Rust default), without it the triangle's own.
pub(crate) fn one_year_pattern(bootstrap_ldf: bool) -> OneYearPattern {
    if bootstrap_ldf {
        OneYearPattern::Method
    } else {
        OneYearPattern::Estimated
    }
}

/// `values` as an array of `N`, or an error naming `name`.
pub(crate) fn fixed<const N: usize>(values: &[f64], name: &str) -> Result<[f64; N]> {
    values
        .try_into()
        .map_err(|_| Error::Other(format!("{name} must have {N} values, got {}", values.len())))
}
