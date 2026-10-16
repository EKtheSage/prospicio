//! prospicio as an Excel add-in: an XLL for 64-bit Excel on Windows
//! (`docs/design/xll.md`).
//!
//! Thin wrappers only, as in the Python and R bindings. The crate has two
//! halves:
//!
//! - a portable core: worksheet values ([`sheet`]), the functions on them
//!   ([`reserving`], [`pricing`]) and the catalogue Excel registers ([`FUNCTIONS`]),
//!   tested with plain `cargo test` on any host;
//! - the Excel C API layer (`xll`, Windows only): XLOPER12 conversion, the
//!   exported functions and `xlAutoOpen`, which registers the catalogue.
//!
//! Build the add-in with `cargo xtask xll` (or `cargo build --release -p
//! prospicio-xll --target x86_64-pc-windows-msvc`, or `-gnu` from Linux)
//! and rename `prospicio_xll.dll` to `prospicio.xll`.

pub mod credibility;
pub mod exposure;
pub mod pricing;
pub mod reserving;
pub mod sheet;
pub mod triangles;
#[cfg(windows)]
mod xll;

use sheet::{Grid, Value};

/// One worksheet argument.
#[derive(Debug)]
pub struct Arg {
    pub name: &'static str,
    pub help: &'static str,
}

/// One worksheet function: its exported symbol, its name in Excel, its
/// help and arguments, and the call on worksheet values.
#[derive(Debug)]
pub struct Function {
    pub symbol: &'static str,
    pub name: &'static str,
    pub help: &'static str,
    pub args: &'static [Arg],
    pub run: fn(&[Grid]) -> Result<Grid, String>,
}

/// The function category in Excel's Insert Function dialog.
pub const CATEGORY: &str = "prospicio";

/// The add-in's name in Excel's Add-ins dialog.
pub const ADDIN_NAME: &str = "prospicio";

const TRIANGLE: Arg = Arg {
    name: "triangle",
    help: "Cumulative values: one row per origin (oldest first), one column per age; blank where unobserved",
};
const SELECTED: Arg = Arg {
    name: "selected",
    help: "Selected factors, one cell per triangle column; a blank keeps the estimate",
};
const KIND: Arg = Arg {
    name: "kind",
    help: "How selected is expressed: \"ldf\" (age-to-age, default), \"cdf\" (to ultimate) or \"percent\" (developed)",
};
const WEIGHT: Arg = Arg {
    name: "weight",
    help: "Weight on the selected factors against the estimates, 0 to 1; 1 if omitted",
};
const SIGMA: Arg = Arg {
    name: "sigma",
    help: "How Mack's last sigma is filled in: \"log-linear\" (default) or \"mack\" (Mack 1993)",
};
const AVERAGE: Arg = Arg {
    name: "average",
    help: "\"volume\" (default), \"simple\" or \"regression\"",
};
const TAIL: Arg = Arg {
    name: "tail",
    help: "A tail factor to ultimate, or \"exponential\", \"inverse_power\" or \"bondy\"; omitted for none",
};
const ORIGINS: Arg = Arg {
    name: "origins",
    help: "Origin labels, one per triangle row; omitted for 1, 2, ...",
};

const SIMS: Arg = Arg {
    name: "sims",
    help: "Number of simulations, 10000 if omitted (at most 1000000)",
};
const SEED: Arg = Arg {
    name: "seed",
    help: "Random seed, 0 if omitted; the same seed gives the same result",
};
const PERCENTILES: Arg = Arg {
    name: "percentiles",
    help: "Probabilities to report, a row or column such as {0.75,0.95,0.995}; omitted for 75%, 90%, 95%, 99.5%",
};
const PROCESS: Arg = Arg {
    name: "process",
    help: "\"gamma\" (default) for process error on future values, or \"none\" for parameter error only",
};

const SIM_ROWS: Arg = Arg {
    name: "sims",
    help: "Number of simulations, 10000 if omitted (at most 100000, one row each)",
};
const MACK_PROCESS: Arg = Arg {
    name: "process",
    help: "\"gamma\" (default), \"lognormal\", \"normal\", \"residuals\", or \"none\" for parameter error only",
};

const EXPOSURE: Arg = Arg {
    name: "exposure",
    help: "Exposure (premium) per origin, a row or column, one value per triangle row",
};
const APRIORI: Arg = Arg {
    name: "apriori",
    help: "The a priori expected loss ratio, ultimate per unit of exposure, such as 0.75",
};

const CDR_PERCENTILES: Arg = Arg {
    name: "percentiles",
    help: "Probabilities to report, a row or column; omitted for 0.5%, 5%, 95%, 99.5% (a negative CDR is adverse)",
};

const CLARK_EXPOSURE: Arg = Arg {
    name: "exposure",
    help: "Premium per origin for Clark's Cape Cod; omitted for Clark's LDF method",
};
const CURVE: Arg = Arg {
    name: "curve",
    help: "Growth curve: \"loglogistic\" (default) or \"weibull\"",
};
const MAX_AGE: Arg = Arg {
    name: "max_age",
    help: "Age in months development stops at, such as 240; omitted (or \"inf\") for none",
};

const TRUNCATION: Arg = Arg {
    name: "truncation",
    help: "Loss size the Pareto is truncated at; omitted (or \"unlimited\") for none",
};
const fn arg(name: &'static str, help: &'static str) -> Arg {
    Arg { name, help }
}

const LIMIT: Arg = Arg {
    name: "limit",
    help: "Limit(s): a number, a range, or \"unlimited\"",
};
const ATTACHMENT: Arg = Arg {
    name: "attachment",
    help: "Attachment(s): a number or a range; a column against a row of limits gives a table",
};
const FAMILY: Arg = Arg {
    name: "family",
    help: "pareto, lognormal, lognormal_cv, gamma, gamma_cv, weibull, loglogistic, inverse_gamma, inverse_gaussian, burr, gpd, piecewise_pareto or gen_pareto (see the README for p1, p2, p3)",
};
const P1: Arg = Arg {
    name: "p1",
    help: "First parameter: Pareto t, lognormal meanlog, the mean (_cv families, inverse Gaussian), shape, Burr alpha, GPD xi, piecewise thresholds (a range) or gen_pareto t",
};
const P2: Arg = Arg {
    name: "p2",
    help: "Second parameter: Pareto alpha, lognormal sdlog, the CV (_cv families), scale, inverse Gaussian shape, Burr gamma, GPD beta, piecewise alphas (a range) or gen_pareto alpha_ini",
};
const P3: Arg = Arg {
    name: "p3",
    help: "Third parameter: the Burr scale, gen_pareto alpha_tail, a Pareto truncation, or a piecewise truncation and \"lp\" or \"wd\"; omitted otherwise",
};
const EXPOSURE_CURVE: Arg = Arg {
    name: "curve",
    help: "swiss_re (p1 = c), mbbefd (p1 = b, p2 = g) or tabulated (p1 = x, p2 = G(x), p3 = \"linear\" or \"log-log\")",
};
const CP1: Arg = Arg {
    name: "p1",
    help: "Swiss Re c, MBBEFD b, or a tabulated curve's x values (a range from 0 to 1)",
};
const CP2: Arg = Arg {
    name: "p2",
    help: "MBBEFD g, or a tabulated curve's G(x) values (a range from 0 to 1); omitted for swiss_re",
};
const CP3: Arg = Arg {
    name: "p3",
    help: "A tabulated curve's interpolation, \"linear\" (default) or \"log-log\"; omitted otherwise",
};
const MPL: Arg = Arg {
    name: "mpl",
    help: "Maximum possible loss (or sum insured) of the risk(s): a number or a range",
};
const RATIOS: Arg = Arg {
    name: "ratios",
    help: "Loss per unit of exposure, one row per group and one column per period; blank where missing",
};
const WEIGHTS: Arg = Arg {
    name: "weights",
    help: "Exposures of the same shape as ratios, blank where missing; omitted for Buhlmann's model (equal weights)",
};
const ESTIMATOR: Arg = Arg {
    name: "estimator",
    help: "Between-group variance: \"unbiased\" (default) or \"iterative\" (Bichsel-Straub)",
};
const GROUPS: Arg = Arg {
    name: "groups",
    help: "Labels for the groups, one per row of ratios; omitted for 1, 2, ...",
};
const X: Arg = Arg {
    name: "x",
    help: "Loss size(s): a number or a range",
};
const PROBABILITY: Arg = Arg {
    name: "p",
    help: "Probability or probabilities, between 0 and 1: a number or a range",
};

fn version(_: &[Grid]) -> Result<Grid, String> {
    Ok(Grid::scalar(Value::Text(format!(
        "prospicio {}",
        env!("CARGO_PKG_VERSION")
    ))))
}

/// Every worksheet function, in registration order. Each `symbol` is
/// exported by `xll.rs` with `args.len()` arguments.
pub static FUNCTIONS: &[Function] = &[
    Function {
        symbol: "pro_version",
        name: "PRO.VERSION",
        help: "The prospicio version this add-in was built from.",
        args: &[],
        run: version,
    },
    Function {
        symbol: "pro_cl",
        name: "PRO.CL",
        help: "Chain ladder exhibit: latest, CDF, ultimate and reserve per origin, with a total.",
        args: &[TRIANGLE, AVERAGE, TAIL, ORIGINS, SELECTED, KIND, WEIGHT],
        run: reserving::cl_summary,
    },
    Function {
        symbol: "pro_cl_ldf",
        name: "PRO.CL.LDF",
        help: "Selected age-to-age factors, one per triangle column; the last is the tail.",
        args: &[TRIANGLE, AVERAGE, TAIL, SELECTED, KIND, WEIGHT],
        run: reserving::cl_ldf,
    },
    Function {
        symbol: "pro_cl_cdf",
        name: "PRO.CL.CDF",
        help: "Age-to-ultimate factors, one per triangle column.",
        args: &[TRIANGLE, AVERAGE, TAIL, SELECTED, KIND, WEIGHT],
        run: reserving::cl_cdf,
    },
    Function {
        symbol: "pro_cl_ultimate",
        name: "PRO.CL.ULTIMATE",
        help: "Chain ladder ultimate per origin.",
        args: &[TRIANGLE, AVERAGE, TAIL, SELECTED, KIND, WEIGHT],
        run: reserving::cl_ultimate,
    },
    Function {
        symbol: "pro_cl_reserve",
        name: "PRO.CL.RESERVE",
        help: "Chain ladder reserve (ultimate less latest) per origin.",
        args: &[TRIANGLE, AVERAGE, TAIL, SELECTED, KIND, WEIGHT],
        run: reserving::cl_reserve,
    },
    Function {
        symbol: "pro_cl_factors",
        name: "PRO.CL.FACTORS",
        help: "Development exhibit: LDF, CDF, Mack sigma and standard error per link, and the tail.",
        args: &[TRIANGLE, AVERAGE, TAIL, SELECTED, KIND, WEIGHT, SIGMA],
        run: reserving::cl_factors,
    },
    Function {
        symbol: "pro_mack",
        name: "PRO.MACK",
        help: "Mack exhibit: ultimate, reserve, process, parameter and total standard error and CV per origin, with a total.",
        args: &[
            TRIANGLE, AVERAGE, TAIL, ORIGINS, SELECTED, KIND, WEIGHT, SIGMA,
        ],
        run: reserving::mack_summary,
    },
    Function {
        symbol: "pro_odp",
        name: "PRO.ODP",
        help: "ODP bootstrap exhibit: CL reserve, simulated mean, SD, CV and percentiles per origin, with a total.",
        args: &[TRIANGLE, SIMS, SEED, PERCENTILES, TAIL, PROCESS, ORIGINS],
        run: reserving::odp_summary,
    },
    Function {
        symbol: "pro_odp_sims",
        name: "PRO.ODP.SIMS",
        help: "ODP bootstrap's simulated reserves: one row per simulation, a column per origin and the total.",
        args: &[
            TRIANGLE,
            Arg {
                name: "sims",
                help: "Number of simulations, 10000 if omitted (at most 100000, one row each)",
            },
            SEED,
            TAIL,
            PROCESS,
            ORIGINS,
        ],
        run: reserving::odp_sims,
    },
    Function {
        symbol: "pro_mack_boot",
        name: "PRO.MACK.BOOT",
        help: "Mack bootstrap exhibit: CL reserve, simulated mean, SD, CV and percentiles per origin, with a total.",
        args: &[
            TRIANGLE,
            SIMS,
            SEED,
            PERCENTILES,
            AVERAGE,
            TAIL,
            MACK_PROCESS,
            ORIGINS,
            SIGMA,
        ],
        run: reserving::mack_bootstrap_summary,
    },
    Function {
        symbol: "pro_mack_boot_sims",
        name: "PRO.MACK.BOOT.SIMS",
        help: "Mack bootstrap's simulated reserves: one row per simulation, a column per origin and the total.",
        args: &[
            TRIANGLE,
            SIM_ROWS,
            SEED,
            AVERAGE,
            TAIL,
            MACK_PROCESS,
            ORIGINS,
            SIGMA,
        ],
        run: reserving::mack_bootstrap_sims,
    },
    Function {
        symbol: "pro_mack_cdr",
        name: "PRO.MACK.CDR",
        help: "Merz-Wuthrich one-year view: reserve, CDR(1) SE and run-off (Mack) SE per origin, with a total.",
        args: &[
            TRIANGLE, AVERAGE, TAIL, ORIGINS, SELECTED, KIND, WEIGHT, SIGMA,
        ],
        run: reserving::mack_cdr,
    },
    Function {
        symbol: "pro_mack_cdr_years",
        name: "PRO.MACK.CDR.YEARS",
        help: "Merz-Wuthrich CDR standard error per origin in each future calendar year, with the total.",
        args: &[
            TRIANGLE, AVERAGE, TAIL, ORIGINS, SELECTED, KIND, WEIGHT, SIGMA,
        ],
        run: reserving::mack_cdr_years,
    },
    Function {
        symbol: "pro_odp_cdr",
        name: "PRO.ODP.CDR",
        help: "ODP bootstrap one-year view: simulated CDR mean, SD and percentiles per origin; negative is adverse.",
        args: &[
            TRIANGLE,
            SIMS,
            SEED,
            CDR_PERCENTILES,
            TAIL,
            PROCESS,
            ORIGINS,
        ],
        run: reserving::odp_cdr,
    },
    Function {
        symbol: "pro_mack_boot_cdr",
        name: "PRO.MACK.BOOT.CDR",
        help: "Mack bootstrap one-year view: simulated CDR mean, SD and percentiles per origin; negative is adverse.",
        args: &[
            TRIANGLE,
            SIMS,
            SEED,
            CDR_PERCENTILES,
            AVERAGE,
            TAIL,
            MACK_PROCESS,
            ORIGINS,
            SIGMA,
        ],
        run: reserving::mack_bootstrap_cdr,
    },
    Function {
        symbol: "pro_el",
        name: "PRO.EL",
        help: "Expected loss method: ultimate is exposure times the a priori loss ratio.",
        args: &[TRIANGLE, EXPOSURE, APRIORI, AVERAGE, TAIL, ORIGINS],
        run: reserving::expected_loss,
    },
    Function {
        symbol: "pro_bf",
        name: "PRO.BF",
        help: "Bornhuetter-Ferguson: latest plus the a priori ultimate's unreported share, per origin, with a total.",
        args: &[TRIANGLE, EXPOSURE, APRIORI, AVERAGE, TAIL, ORIGINS],
        run: reserving::bornhuetter_ferguson,
    },
    Function {
        symbol: "pro_benktander",
        name: "PRO.BENKTANDER",
        help: "Benktander: Bornhuetter-Ferguson iterated on its own ultimate (1 iteration is BF).",
        args: &[
            TRIANGLE,
            EXPOSURE,
            APRIORI,
            Arg {
                name: "iterations",
                help: "Number of iterations, 1 (Bornhuetter-Ferguson) if omitted",
            },
            AVERAGE,
            TAIL,
            ORIGINS,
        ],
        run: reserving::benktander,
    },
    Function {
        symbol: "pro_capecod",
        name: "PRO.CAPECOD",
        help: "Cape Cod (Stanard-Buhlmann): the a priori loss ratio estimated from the triangle and exposure.",
        args: &[
            TRIANGLE,
            EXPOSURE,
            Arg {
                name: "trend",
                help: "Annual trend of the loss ratio between origins, 0 if omitted",
            },
            Arg {
                name: "decay",
                help: "Weight decay by distance between origins, 0 to 1; 1 if omitted",
            },
            AVERAGE,
            TAIL,
            ORIGINS,
        ],
        run: reserving::cape_cod,
    },
    Function {
        symbol: "pro_clark",
        name: "PRO.CLARK",
        help: "Clark's growth curve: LDF method, or Cape Cod with exposure; reserve and its SEs per origin, with a total.",
        args: &[TRIANGLE, CLARK_EXPOSURE, CURVE, MAX_AGE, ORIGINS],
        run: reserving::clark,
    },
    Function {
        symbol: "pro_clark_params",
        name: "PRO.CLARK.PARAMS",
        help: "Clark's fitted parameters: curve, omega, theta, the ELR (Cape Cod), the scale and the observations.",
        args: &[TRIANGLE, CLARK_EXPOSURE, CURVE, MAX_AGE],
        run: reserving::clark_params,
    },
    Function {
        symbol: "pro_tri_incremental",
        name: "PRO.TRI.INCREMENTAL",
        help: "A cumulative triangle's incremental values, laid out as the triangle.",
        args: &[TRIANGLE],
        run: triangles::incremental,
    },
    Function {
        symbol: "pro_tri_cumulative",
        name: "PRO.TRI.CUMULATIVE",
        help: "An incremental triangle's cumulative values, laid out as the triangle.",
        args: &[Arg {
            name: "triangle",
            help: "Incremental values: one row per origin (oldest first), one column per age; blank where unobserved",
        }],
        run: triangles::cumulative,
    },
    Function {
        symbol: "pro_tri_linkratios",
        name: "PRO.TRI.LINKRATIOS",
        help: "A cumulative triangle's age-to-age link ratios, one column fewer than its ages.",
        args: &[TRIANGLE],
        run: triangles::link_ratios,
    },
    Function {
        symbol: "pro_tri_latest",
        name: "PRO.TRI.LATEST",
        help: "Each origin's latest observed value (the latest diagonal), a column.",
        args: &[TRIANGLE],
        run: triangles::latest,
    },
    Function {
        symbol: "pro_sev_cdf",
        name: "PRO.SEV.CDF",
        help: "A severity's distribution function P(X <= x) at each x.",
        args: &[X, FAMILY, P1, P2, P3],
        run: pricing::cdf,
    },
    Function {
        symbol: "pro_sev_survival",
        name: "PRO.SEV.SURVIVAL",
        help: "A severity's survival function P(X > x) at each x.",
        args: &[X, FAMILY, P1, P2, P3],
        run: pricing::survival,
    },
    Function {
        symbol: "pro_sev_quantile",
        name: "PRO.SEV.QUANTILE",
        help: "A severity's quantile (VaR) at each probability p.",
        args: &[PROBABILITY, FAMILY, P1, P2, P3],
        run: pricing::quantile,
    },
    Function {
        symbol: "pro_sev_stats",
        name: "PRO.SEV.STATS",
        help: "A severity's family, mean, SD and CV.",
        args: &[FAMILY, P1, P2, P3],
        run: pricing::stats,
    },
    Function {
        symbol: "pro_ler",
        name: "PRO.LER",
        help: "Loss elimination ratio of a deductible, LEV(d) / E[X]: the credit for a straight deductible.",
        args: &[
            Arg {
                name: "deductible",
                help: "Deductible(s): a number or a range",
            },
            FAMILY,
            P1,
            P2,
            P3,
        ],
        run: pricing::ler,
    },
    Function {
        symbol: "pro_pareto_extrapolate",
        name: "PRO.PARETO.EXTRAPOLATE",
        help: "Pareto extrapolation: expected loss of each to layer per unit of the from layer's, at this alpha.",
        args: &[
            arg("from_limit", "The rated layer's limit, or \"unlimited\""),
            arg("from_attachment", "The rated layer's attachment"),
            arg(
                "to_limit",
                "Limit(s) of the layer(s) to rate: a number, a range or \"unlimited\"",
            ),
            arg(
                "to_attachment",
                "Attachment(s) of the layer(s) to rate; a column against a row of limits gives a table",
            ),
            arg("alpha", "The Pareto alpha"),
            TRUNCATION,
        ],
        run: pricing::pareto_extrapolate,
    },
    Function {
        symbol: "pro_pareto_alpha",
        name: "PRO.PARETO.ALPHA",
        help: "The Pareto alpha at which two layers have the given expected losses.",
        args: &[
            arg("limit1", "First layer's limit, or \"unlimited\""),
            arg("attachment1", "First layer's attachment"),
            arg("loss1", "First layer's expected loss"),
            arg("limit2", "Second layer's limit, or \"unlimited\""),
            arg("attachment2", "Second layer's attachment"),
            arg("loss2", "Second layer's expected loss"),
            TRUNCATION,
        ],
        run: pricing::pareto_alpha_layers,
    },
    Function {
        symbol: "pro_pareto_alpha_freqlayer",
        name: "PRO.PARETO.ALPHA.FREQLAYER",
        help: "The Pareto alpha at which a frequency of losses above a threshold gives a layer its expected loss.",
        args: &[
            arg(
                "threshold",
                "The loss size the frequency counts losses above",
            ),
            arg(
                "frequency",
                "Expected number of losses above the threshold a year",
            ),
            arg("limit", "The layer's limit, or \"unlimited\""),
            arg("attachment", "The layer's attachment"),
            arg("loss", "The layer's expected loss a year"),
            TRUNCATION,
        ],
        run: pricing::pareto_alpha_frequency_layer,
    },
    Function {
        symbol: "pro_pareto_alpha_freqs",
        name: "PRO.PARETO.ALPHA.FREQS",
        help: "The Pareto alpha between two frequencies above two thresholds.",
        args: &[
            arg("threshold1", "First threshold"),
            arg(
                "frequency1",
                "Expected number of losses above the first threshold a year",
            ),
            arg("threshold2", "Second threshold"),
            arg(
                "frequency2",
                "Expected number of losses above the second threshold a year",
            ),
            TRUNCATION,
        ],
        run: pricing::pareto_alpha_frequencies,
    },
    Function {
        symbol: "pro_lev",
        name: "PRO.LEV",
        help: "Limited expected value E[min(X, limit)] of a severity, per limit.",
        args: &[LIMIT, FAMILY, P1, P2, P3],
        run: pricing::lev,
    },
    Function {
        symbol: "pro_layer",
        name: "PRO.LAYER",
        help: "Expected loss to the layer limit xs attachment per ground-up loss.",
        args: &[LIMIT, ATTACHMENT, FAMILY, P1, P2, P3],
        run: pricing::layer,
    },
    Function {
        symbol: "pro_layer_var",
        name: "PRO.LAYER.VAR",
        help: "Variance of the loss to the layer limit xs attachment per ground-up loss.",
        args: &[LIMIT, ATTACHMENT, FAMILY, P1, P2, P3],
        run: pricing::layer_variance,
    },
    Function {
        symbol: "pro_layer_cv",
        name: "PRO.LAYER.CV",
        help: "CV of the loss to the layer, given a loss above its attachment.",
        args: &[LIMIT, ATTACHMENT, FAMILY, P1, P2, P3],
        run: pricing::layer_coefficient_of_variation,
    },
    Function {
        symbol: "pro_ilf",
        name: "PRO.ILF",
        help: "Increased limit factor LEV(limit) / LEV(basic_limit).",
        args: &[
            LIMIT,
            Arg {
                name: "basic_limit",
                help: "The basic limit the factors are relative to",
            },
            FAMILY,
            P1,
            P2,
            P3,
        ],
        run: pricing::increased_limit_factor,
    },
    Function {
        symbol: "pro_exposure_curve",
        name: "PRO.EXPOSURE.CURVE",
        help: "Exposure curve G(x): the share of a risk's expected loss below a fraction x of its MPL.",
        args: &[
            Arg {
                name: "x",
                help: "Fraction(s) of the MPL, between 0 and 1: a number or a range",
            },
            EXPOSURE_CURVE,
            CP1,
            CP2,
            CP3,
        ],
        run: exposure::exposure_curve,
    },
    Function {
        symbol: "pro_exposure_share",
        name: "PRO.EXPOSURE.SHARE",
        help: "Share of a risk's expected loss in the layer limit xs attachment, for a risk of the given MPL.",
        args: &[LIMIT, ATTACHMENT, MPL, EXPOSURE_CURVE, CP1, CP2, CP3],
        run: exposure::share,
    },
    Function {
        symbol: "pro_exposure_quantile",
        name: "PRO.EXPOSURE.QUANTILE",
        help: "Destruction rate (loss over MPL) at each probability, for simulating losses from the curve.",
        args: &[PROBABILITY, EXPOSURE_CURVE, CP1, CP2, CP3],
        run: exposure::quantile,
    },
    Function {
        symbol: "pro_exposure_stats",
        name: "PRO.EXPOSURE.STATS",
        help: "The curve's b, g, mean destruction rate and probability of a total loss.",
        args: &[EXPOSURE_CURVE, CP1, CP2, CP3],
        run: exposure::stats,
    },
    Function {
        symbol: "pro_swissre_c",
        name: "PRO.SWISSRE.C",
        help: "The Swiss Re c whose mean destruction rate (expected loss over MPL) is rate.",
        args: &[Arg {
            name: "rate",
            help: "Mean destruction rate(s), in (0, 1]: a number or a range",
        }],
        run: exposure::swiss_re_c,
    },
    Function {
        symbol: "pro_swissre_c_layer",
        name: "PRO.SWISSRE.C.LAYER",
        help: "The Swiss Re c whose share of a risk's expected loss in limit xs attachment is share.",
        args: &[
            LIMIT,
            ATTACHMENT,
            MPL,
            Arg {
                name: "share",
                help: "The layer's share(s) of the risk's expected loss, between 0 and 1",
            },
        ],
        run: exposure::swiss_re_c_layer,
    },
    Function {
        symbol: "pro_profile",
        name: "PRO.PROFILE",
        help: "Exposure-rated risk profile: each band's expected claims, surplus cession and loss to a per-risk layer, with a total.",
        args: &[
            Arg {
                name: "sums_insured",
                help: "Each band's sum insured (its representative risk, taken as the MPL): a row or column",
            },
            Arg {
                name: "expected_losses",
                help: "Each band's expected annual loss (premium times loss ratio): one per band",
            },
            Arg {
                name: "limit",
                help: "The per-risk layer's limit, or \"unlimited\"",
            },
            Arg {
                name: "attachment",
                help: "The per-risk layer's attachment; 0 if omitted",
            },
            EXPOSURE_CURVE,
            Arg {
                name: "p1",
                help: "Swiss Re c or MBBEFD b (one, or one per band), or a tabulated curve's x values",
            },
            Arg {
                name: "p2",
                help: "MBBEFD g (one, or one per band), or a tabulated curve's G(x) values",
            },
            CP3,
            Arg {
                name: "lower",
                help: "Each band's lowest sum insured, a blank for a band without bounds; omitted for none",
            },
            Arg {
                name: "upper",
                help: "Each band's highest sum insured, a blank for a band without bounds",
            },
            Arg {
                name: "spread",
                help: "How a band spreads its sums insured between its bounds: \"uniform\" (default), \"tilted\" or \"auto\"",
            },
            Arg {
                name: "retention",
                help: "A surplus treaty's retention line, which the layer inures to; omitted for none",
            },
            Arg {
                name: "lines",
                help: "The surplus treaty's number of lines",
            },
        ],
        run: exposure::profile,
    },
    Function {
        symbol: "pro_cred_standard",
        name: "PRO.CRED.STANDARD",
        help: "Full credibility standard in expected claims (limited fluctuation): within k of the mean with probability p.",
        args: &[
            Arg {
                name: "p",
                help: "Probability, such as 0.90: a number or a range",
            },
            Arg {
                name: "k",
                help: "Tolerance, such as 0.05: a number or a range; a row against a column of p gives a table",
            },
            Arg {
                name: "kind",
                help: "\"frequency\" (default), \"severity\" or \"pure_premium\"",
            },
            Arg {
                name: "cv",
                help: "The severity's coefficient of variation, for the severity and pure premium standards",
            },
        ],
        run: credibility::standard,
    },
    Function {
        symbol: "pro_cred_z",
        name: "PRO.CRED.Z",
        help: "Partial credibility by the square root rule, min(1, sqrt(n / standard)).",
        args: &[
            Arg {
                name: "n",
                help: "Observed claims: a number or a range",
            },
            Arg {
                name: "standard",
                help: "The full credibility standard in claims: a number or a range",
            },
        ],
        run: credibility::z,
    },
    Function {
        symbol: "pro_buhlmann",
        name: "PRO.BUHLMANN",
        help: "Buhlmann-Straub credibility: each group's weight, mean, credibility factor and premium.",
        args: &[RATIOS, WEIGHTS, ESTIMATOR, GROUPS],
        run: credibility::buhlmann,
    },
    Function {
        symbol: "pro_buhlmann_params",
        name: "PRO.BUHLMANN.PARAMS",
        help: "Buhlmann-Straub structure parameters: collective mean, between and within variances, k.",
        args: &[RATIOS, WEIGHTS, ESTIMATOR],
        run: credibility::buhlmann_params,
    },
];

/// Runs the function exported as `symbol` on `args`, catching a panic so
/// that it never unwinds into Excel. An error comes back as a single text
/// cell starting with `#PROSPICIO!`, which shows the user why.
///
/// Arguments past those given are omitted ones, as Excel passes for an
/// optional argument left out.
pub fn call(symbol: &str, mut args: Vec<Grid>) -> Grid {
    let Some(f) = FUNCTIONS.iter().find(|f| f.symbol == symbol) else {
        return error_cell(&format!("no function {symbol}"));
    };
    if args.len() > f.args.len() {
        return error_cell(&format!(
            "{} takes {} arguments, got {}",
            f.name,
            f.args.len(),
            args.len()
        ));
    }
    args.resize_with(f.args.len(), Grid::missing);
    match std::panic::catch_unwind(|| (f.run)(&args)) {
        Ok(Ok(grid)) => grid,
        Ok(Err(message)) => error_cell(&message),
        Err(_) => error_cell("internal error (a panic); please report it"),
    }
}

fn error_cell(message: &str) -> Grid {
    Grid::scalar(Value::Text(format!("#PROSPICIO! {message}")))
}
