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

pub mod aggregate;
pub mod bayes;
pub mod copulas;
pub mod credibility;
pub mod diagonal;
pub mod distortions;
pub mod evt;
pub mod experience;
pub mod exposure;
pub mod fitting;
pub mod odp_glm;
pub mod payout;
pub mod pricing;
pub mod quote;
pub mod ratemaking;
pub mod reserving;
pub mod sheet;
pub mod terms;
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

const TERM_LOSSES: Arg = Arg {
    name: "losses",
    help: "Ceded losses: one row a term (a simulated path), one column a year in order",
};
const CARRY_YEARS: Arg = Arg {
    name: "carry_years",
    help: "Later years a carried amount may be used in, then dropped (until used up if omitted)",
};
const COPULA_FAMILY: Arg = Arg {
    name: "family",
    help: "gaussian, t, clayton, gumbel, frank or joe",
};
const COPULA_PARAMETER: Arg = Arg {
    name: "parameter",
    help: "gaussian and t: a correlation for every pair, or a square correlation matrix; clayton, gumbel, frank, joe: theta",
};
const DATA: Arg = Arg {
    name: "data",
    help: "The losses: a range; blank cells are skipped",
};
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
    help: "volume (default), simple, regression or geometric; or a row: kind, n_periods (latest calendar periods), drop_high, drop_low (link ratios left out at each age)",
};
const ODP_AVERAGE: Arg = Arg {
    name: "average",
    help: "Link ratios kept: a row \"volume\", n_periods (latest calendar periods), drop_high, drop_low; all of them if omitted",
};
const TAIL: Arg = Arg {
    name: "tail",
    help: "A tail factor to ultimate, or \"exponential\", \"inverse_power\" or \"bondy\"; omitted for none",
};
const ORIGINS: Arg = Arg {
    name: "origins",
    help: "Origin labels, one per triangle row; omitted for 1, 2, ...",
};
const RESIDUALS_BY: Arg = Arg {
    name: "by",
    help: "cell (default: a row per observed increment), or origin, development or calendar for the count, mean and SD of each group's standardized deviance residuals",
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

const DIAG_ORIGINS: Arg = Arg {
    name: "origins",
    help: "Each origin's start, a row or column: a year such as 2024, or an Excel date (floored to grain); in any order, every period from the first to the last",
};
const DIAG_AGES: Arg = Arg {
    name: "ages",
    help: "Each origin's age in months (an accident year at its first year-end is 12); leave empty when valuation is given",
};
const DIAG_LATEST: Arg = Arg {
    name: "latest",
    help: "Each origin's latest cumulative value; 0 for an origin with nothing reported",
};
const DIAG_EXPOSURE: Arg = Arg {
    name: "exposure",
    help: "Exposure (premium) per origin, a row or column, one positive value per origin",
};
const DIAG_PATTERN: Arg = Arg {
    name: "pattern",
    help: "The selected pattern's values, a row or column, such as PRO.CL.LDF's; a blank value is skipped, so an age-to-age factor before it runs to the next age given (give 1 for no development)",
};
const PATTERN_AGES: Arg = Arg {
    name: "pattern_ages",
    help: "Each pattern value's age in months, such as 3, 6, 9, ...; omitted for 12, 24, ... by cell",
};
const PATTERN_KIND: Arg = Arg {
    name: "kind",
    help: "How pattern is expressed: \"ldf\" (age-to-age to the next age given, default), \"cdf\" (to ultimate) or \"percent\" (developed)",
};
const DIAG_TAIL: Arg = Arg {
    name: "tail",
    help: "A tail factor past the pattern's end, or \"exponential\", \"inverse_power\" or \"bondy\" fitted on its factors; omitted for none",
};
const DIAG_GRAIN: Arg = Arg {
    name: "grain",
    help: "The origins' grain: \"year\" (default), \"semiannual\", \"quarter\" or \"month\"",
};
const DIAG_VALUATION: Arg = Arg {
    name: "valuation",
    help: "The Excel date all origins are valued at, in place of ages: an accident year 2024 at 31 December 2024 is 12 months",
};
const PAST_END: Arg = Arg {
    name: "past_end",
    help: "An origin older than the pattern's end: \"reject\" (default, an error), \"none\" (a factor of 1; rescales a cumulative level to 100% for every origin) or \"tail\" (the tail spread on)",
};
const PATTERN_ORIGIN: Arg = Arg {
    name: "pattern_origin",
    help: "The origin basis pattern was measured on, such as \"month\", averaged to grain; omitted to read it as given. An estimated tail is then fitted on monthly factors: a constant tail is safer",
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
    help: "pareto, lognormal, lognormal_cv, gamma, gamma_cv, weibull, loglogistic, inverse_gamma, inverse_gaussian, burr, lomax, gpd, piecewise_pareto or gen_pareto (see the README for p1, p2, p3)",
};
const BODY_FAMILY: Arg = Arg {
    name: "family",
    help: "lognormal, gamma, weibull, burr or lomax",
};
const P1: Arg = Arg {
    name: "p1",
    help: "First parameter: Pareto t, lognormal meanlog, the mean (_cv families, inverse Gaussian), shape, Burr or Lomax alpha, GPD xi, piecewise thresholds (a range) or gen_pareto t",
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
const EFFECTIVE: Arg = Arg {
    name: "effective",
    help: "Each rate change's effective date (a year fraction such as 2011.5, or an Excel date), in order",
};
const CHANGES: Arg = Arg {
    name: "changes",
    help: "Each rate change, 0.05 for +5% (or the cumulative levels, with given \"levels\")",
};
const FROM: Arg = Arg {
    name: "from",
    help: "Each period's start (a year fraction or an Excel date): a number or a range",
};
const TO: Arg = Arg {
    name: "to",
    help: "Each period's end, exclusive (2012 for calendar year 2011)",
};
const PERIOD: Arg = Arg {
    name: "period",
    help: "\"calendar\" (default: premium earned, losses occurring in the window) or \"policy\" (policies written in it)",
};
const TERM: Arg = Arg {
    name: "term",
    help: "Policy term in years, 1 if omitted (0.5 for six-month policies)",
};
const APPLIES: Arg = Arg {
    name: "applies",
    help: "\"written\" (default: new and renewal policies) or \"in_force\" (all premium earned after the date), one or one per change",
};
const GIVEN: Arg = Arg {
    name: "given",
    help: "\"changes\" (default) or \"levels\": whether changes holds each change or the cumulative level after it",
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
        args: &[
            TRIANGLE,
            SIMS,
            SEED,
            PERCENTILES,
            TAIL,
            PROCESS,
            ORIGINS,
            ODP_AVERAGE,
        ],
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
            ODP_AVERAGE,
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
            ODP_AVERAGE,
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
        symbol: "pro_diag_cl",
        name: "PRO.DIAG.CL",
        help: "Chain ladder on a latest diagonal (a bordereau) with a selected pattern: each origin's age, latest, CDF, ultimate and reserve, oldest first, with a total.",
        args: &[
            DIAG_ORIGINS,
            DIAG_AGES,
            DIAG_LATEST,
            DIAG_PATTERN,
            PATTERN_AGES,
            PATTERN_KIND,
            DIAG_TAIL,
            DIAG_GRAIN,
            DIAG_VALUATION,
            PAST_END,
            PATTERN_ORIGIN,
        ],
        run: diagonal::chain_ladder,
    },
    Function {
        symbol: "pro_diag_el",
        name: "PRO.DIAG.EL",
        help: "Expected loss method on a latest diagonal with a selected pattern: ultimate is exposure times the a priori loss ratio.",
        args: &[
            DIAG_ORIGINS,
            DIAG_AGES,
            DIAG_LATEST,
            DIAG_EXPOSURE,
            APRIORI,
            DIAG_PATTERN,
            PATTERN_AGES,
            PATTERN_KIND,
            DIAG_TAIL,
            DIAG_GRAIN,
            DIAG_VALUATION,
            PAST_END,
            PATTERN_ORIGIN,
        ],
        run: diagonal::expected_loss,
    },
    Function {
        symbol: "pro_diag_bf",
        name: "PRO.DIAG.BF",
        help: "Bornhuetter-Ferguson on a latest diagonal with a selected pattern: latest plus the a priori ultimate's unreported share, per origin, with a total.",
        args: &[
            DIAG_ORIGINS,
            DIAG_AGES,
            DIAG_LATEST,
            DIAG_EXPOSURE,
            APRIORI,
            DIAG_PATTERN,
            PATTERN_AGES,
            PATTERN_KIND,
            DIAG_TAIL,
            DIAG_GRAIN,
            DIAG_VALUATION,
            PAST_END,
            PATTERN_ORIGIN,
        ],
        run: diagonal::bornhuetter_ferguson,
    },
    Function {
        symbol: "pro_diag_benktander",
        name: "PRO.DIAG.BENKTANDER",
        help: "Benktander on a latest diagonal with a selected pattern: Bornhuetter-Ferguson iterated on its own ultimate (1 iteration is BF).",
        args: &[
            DIAG_ORIGINS,
            DIAG_AGES,
            DIAG_LATEST,
            DIAG_EXPOSURE,
            APRIORI,
            DIAG_PATTERN,
            Arg {
                name: "iterations",
                help: "Number of iterations, 1 (Bornhuetter-Ferguson) if omitted",
            },
            PATTERN_AGES,
            PATTERN_KIND,
            DIAG_TAIL,
            DIAG_GRAIN,
            DIAG_VALUATION,
            PAST_END,
            PATTERN_ORIGIN,
        ],
        run: diagonal::benktander,
    },
    Function {
        symbol: "pro_diag_capecod",
        name: "PRO.DIAG.CAPECOD",
        help: "Cape Cod on a latest diagonal with a selected pattern: the a priori loss ratio estimated from the diagonal and exposure, trended to the latest valuation.",
        args: &[
            DIAG_ORIGINS,
            DIAG_AGES,
            DIAG_LATEST,
            DIAG_EXPOSURE,
            DIAG_PATTERN,
            Arg {
                name: "trend",
                help: "Annual trend of the loss ratio between origins, 0 if omitted",
            },
            Arg {
                name: "decay",
                help: "Weight decay by distance between origins, 0 to 1; 1 if omitted",
            },
            PATTERN_AGES,
            PATTERN_KIND,
            DIAG_TAIL,
            DIAG_GRAIN,
            DIAG_VALUATION,
            PAST_END,
            PATTERN_ORIGIN,
        ],
        run: diagonal::cape_cod,
    },
    Function {
        symbol: "pro_pattern_average",
        name: "PRO.PATTERN.AVERAGE",
        help: "A pattern measured on one origin basis (such as accident months) averaged to a coarser one: age, percent developed and CDF for every month.",
        args: &[
            Arg {
                name: "pattern_ages",
                help: "Each pattern value's age in months; omitted for 12, 24, ... by cell",
            },
            DIAG_PATTERN,
            Arg {
                name: "pattern_origin",
                help: "The origin basis pattern was measured on: \"month\", \"quarter\" or \"semiannual\"",
            },
            Arg {
                name: "grain",
                help: "The origin basis to average to, a whole multiple of pattern_origin: \"year\" (default), \"semiannual\" or \"quarter\"",
            },
            PATTERN_KIND,
        ],
        run: diagonal::pattern_average,
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
        symbol: "pro_odp_glm",
        name: "PRO.ODP.GLM",
        help: "ODP GLM exhibit: latest, ultimate, reserve, process and parameter SE, RMSEP and CV per origin, with a total (Taylor and McGuire's Table 5-4).",
        args: &[TRIANGLE, ORIGINS],
        run: odp_glm::summary,
    },
    Function {
        symbol: "pro_odp_glm_params",
        name: "PRO.ODP.GLM.PARAMS",
        help: "The ODP GLM's coefficients: term, level, estimate and SE (intercept, origins and development periods from the second), then Pearson's dispersion.",
        args: &[TRIANGLE, ORIGINS],
        run: odp_glm::params,
    },
    Function {
        symbol: "pro_odp_glm_residuals",
        name: "PRO.ODP.GLM.RESIDUALS",
        help: "The ODP GLM's residuals: a row per observed increment (actual, fitted, Pearson, deviance, leverage, standardized), or grouped by origin, development or calendar period.",
        args: &[TRIANGLE, RESIDUALS_BY, ORIGINS],
        run: odp_glm::residuals,
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
        help: "Exposure curve G(x): the share of a risk's expected loss below a fraction x of its MPL. A column of x against a row of c (or b, g) gives one column per curve.",
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
        help: "Destruction rate (loss over MPL) at each probability, for simulating losses from the curve. A column of u against a row of c (or b, g) gives one column per curve.",
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
    Function {
        symbol: "pro_yearfraction",
        name: "PRO.YEARFRACTION",
        help: "An Excel date on Werner and Modlin's half-month grid, as a year fraction (30 June 2013 is 2013.5).",
        args: &[Arg {
            name: "date",
            help: "Excel date(s): a date or a range",
        }],
        run: ratemaking::year_fraction_of,
    },
    Function {
        symbol: "pro_onlevel",
        name: "PRO.ONLEVEL",
        help: "On-level factor of each period's premium by the parallelogram method: current over average rate level.",
        args: &[EFFECTIVE, CHANGES, FROM, TO, PERIOD, TERM, APPLIES, GIVEN],
        run: ratemaking::on_level,
    },
    Function {
        symbol: "pro_ratelevel",
        name: "PRO.RATELEVEL",
        help: "One period's premium by rate level (the parallelogram's cells), average and current levels, on-level factor.",
        args: &[EFFECTIVE, CHANGES, FROM, TO, PERIOD, TERM, APPLIES, GIVEN],
        run: ratemaking::rate_level,
    },
    Function {
        symbol: "pro_avgdate",
        name: "PRO.AVGDATE",
        help: "Average written or earned (accident) date of each period, for trend lengths.",
        args: &[
            FROM,
            TO,
            PERIOD,
            TERM,
            Arg {
                name: "basis",
                help: "\"earned\" (default; the average accident date of losses) or \"written\"",
            },
        ],
        run: ratemaking::average_date,
    },
    Function {
        symbol: "pro_trend_fit",
        name: "PRO.TREND.FIT",
        help: "Least-squares trend of values over time: annual change, intercept, slope, R-squared.",
        args: &[
            Arg {
                name: "times",
                help: "Times in years (2011.25 for a quarter) or Excel dates",
            },
            Arg {
                name: "values",
                help: "Values at those times (frequencies, severities, average premiums), the same number of cells",
            },
            Arg {
                name: "kind",
                help: "\"exponential\" (default; ln y on time) or \"linear\"",
            },
        ],
        run: ratemaking::trend_fit,
    },
    Function {
        symbol: "pro_lr_experience",
        name: "PRO.LR.EXPERIENCE",
        help: "Loss ratio method: each year's projected premium, losses and loss ratio, and the total.",
        args: &[
            Arg {
                name: "earned_premium",
                help: "Earned premium per experience year",
            },
            Arg {
                name: "losses",
                help: "Reported (or paid) losses per year, the same count",
            },
            Arg {
                name: "on_level",
                help: "Factor to the current rate level: one value for every year, or one per year; 1 if omitted",
            },
            Arg {
                name: "premium_trend",
                help: "Premium trend factor: one value for every year, or one per year; 1 if omitted",
            },
            Arg {
                name: "development",
                help: "Loss development factor to ultimate: one value for every year, or one per year; 1 if omitted",
            },
            Arg {
                name: "loss_trend",
                help: "Loss trend factor: one value for every year, or one per year; 1 if omitted",
            },
            Arg {
                name: "lae",
                help: "1 + the LAE ratio not in losses: one value for every year, or one per year; 1 if omitted",
            },
        ],
        run: ratemaking::lr_experience,
    },
    Function {
        symbol: "pro_pp_experience",
        name: "PRO.PP.EXPERIENCE",
        help: "Pure premium method: each year's exposure, projected losses and pure premium, and the total.",
        args: &[
            Arg {
                name: "exposure",
                help: "Earned exposure per experience year",
            },
            Arg {
                name: "losses",
                help: "Reported (or paid) losses per year, the same count",
            },
            Arg {
                name: "development",
                help: "Loss development factor to ultimate: one value for every year, or one per year; 1 if omitted",
            },
            Arg {
                name: "loss_trend",
                help: "Loss trend factor: one value for every year, or one per year; 1 if omitted",
            },
            Arg {
                name: "lae",
                help: "1 + the LAE ratio not in losses: one value for every year, or one per year; 1 if omitted",
            },
        ],
        run: ratemaking::pp_experience,
    },
    Function {
        symbol: "pro_indicated_change",
        name: "PRO.INDICATED.CHANGE",
        help: "Loss ratio method's indicated change, (LR + F) / (1 - V - Q) - 1.",
        args: &[
            Arg {
                name: "loss_ratio",
                help: "Projected loss and LAE ratio at current rates: a number or a range",
            },
            Arg {
                name: "variable",
                help: "Variable expense provision, a fraction of premium (0.25 for 25%)",
            },
            Arg {
                name: "profit",
                help: "Underwriting profit provision, a fraction of premium (may be negative)",
            },
            Arg {
                name: "fixed_ratio",
                help: "Fixed expense ratio to premium; 0 if omitted",
            },
        ],
        run: ratemaking::indicated_change,
    },
    Function {
        symbol: "pro_indicated_rate",
        name: "PRO.INDICATED.RATE",
        help: "Pure premium method's indicated rate, (PP + fixed) / (1 - V - Q).",
        args: &[
            Arg {
                name: "pure_premium",
                help: "Projected pure premium with LAE (and other costs per exposure): a number or a range",
            },
            Arg {
                name: "variable",
                help: "Variable expense provision, a fraction of premium (0.25 for 25%)",
            },
            Arg {
                name: "profit",
                help: "Underwriting profit provision, a fraction of premium (may be negative)",
            },
            Arg {
                name: "fixed_per_exposure",
                help: "Fixed expense per exposure; 0 if omitted",
            },
        ],
        run: ratemaking::indicated_rate,
    },
    Function {
        symbol: "pro_permissible_lr",
        name: "PRO.PERMISSIBLE.LR",
        help: "Permissible loss ratio, 1 - F - V - Q.",
        args: &[
            Arg {
                name: "variable",
                help: "Variable expense provision, a fraction of premium (0.25 for 25%)",
            },
            Arg {
                name: "profit",
                help: "Underwriting profit provision, a fraction of premium (may be negative)",
            },
            Arg {
                name: "fixed_ratio",
                help: "Fixed expense ratio to premium; 0 if omitted",
            },
        ],
        run: ratemaking::permissible_lr,
    },
    Function {
        symbol: "pro_trended_present",
        name: "PRO.TRENDED.PRESENT",
        help: "Trended present rates complement as a change: (1 + indicated) / (1 + implemented) * trend - 1.",
        args: &[
            Arg {
                name: "prior_indicated",
                help: "The last review's indicated change",
            },
            Arg {
                name: "prior_implemented",
                help: "The change taken then",
            },
            Arg {
                name: "trend",
                help: "Trend factor from the last target effective date to the new one",
            },
        ],
        run: ratemaking::trended_present,
    },
    Function {
        symbol: "pro_cred_weighted",
        name: "PRO.CRED.WEIGHTED",
        help: "Credibility-weighted estimate, Z * estimate + (1 - Z) * complement.",
        args: &[
            Arg {
                name: "estimate",
                help: "The indication: a number or a range",
            },
            Arg {
                name: "credibility",
                help: "Z, between 0 and 1",
            },
            Arg {
                name: "complement",
                help: "The complement of credibility",
            },
        ],
        run: ratemaking::cred_weighted,
    },
    Function {
        symbol: "pro_expenses",
        name: "PRO.EXPENSES",
        help: "Premium-based projection: fixed and variable expense ratios from each category's ratio and fixed share.",
        args: &[
            Arg {
                name: "ratios",
                help: "Each expense category's ratio to premium",
            },
            Arg {
                name: "fixed_shares",
                help: "Each category's fixed share, 0 to 1; all variable if omitted",
            },
        ],
        run: ratemaking::expenses,
    },
    Function {
        symbol: "pro_burningcost",
        name: "PRO.BURNINGCOST",
        help: "Burning cost from large losses: each accident year's trended layer losses, developed, over its premium.",
        args: &[
            Arg {
                name: "accident",
                help: "Each loss's accident date (an Excel date or a decimal year): a column",
            },
            Arg {
                name: "amounts",
                help: "Each loss's amount (incurred at the latest valuation), the same count",
            },
            Arg {
                name: "limits",
                help: "Each layer's limit, a row (or \"unlimited\")",
            },
            Arg {
                name: "attachments",
                help: "Each layer's attachment: one, or one per limit",
            },
            Arg {
                name: "years",
                help: "The accident years to rate, a column",
            },
            Arg {
                name: "premiums",
                help: "Each year's on-level subject premium (or exposure)",
            },
            Arg {
                name: "development",
                help: "Each year's excess development factor for its layer losses: one, or one per year; 1 if omitted",
            },
            Arg {
                name: "trend",
                help: "Annual severity trend (0.05 for 5%) to trend_to, or a trend factor per loss; none if omitted",
            },
            Arg {
                name: "trend_to",
                help: "The prospective period's average accident date the losses are trended to",
            },
            Arg {
                name: "threshold",
                help: "The reporting threshold: losses trended below threshold times the largest trend factor are left out",
            },
            Arg {
                name: "agg_deductible",
                help: "Each layer's annual aggregate deductible: one, or one per layer; none if omitted",
            },
            Arg {
                name: "agg_limit",
                help: "Each layer's annual aggregate limit: one, or one per layer; none if omitted",
            },
        ],
        run: experience::burning_cost,
    },
    Function {
        symbol: "pro_agg_stats",
        name: "PRO.AGG.STATS",
        help: "Aggregate (compound) annual loss, ground up or to a per-occurrence layer: mean, SD, CV, skewness, VaR, TVaR.",
        args: &[
            Arg {
                name: "count_mean",
                help: "Expected number of claims a year",
            },
            Arg {
                name: "dispersion",
                help: "Var N / E N: 1 Poisson (default), above 1 negative binomial, below 1 binomial",
            },
            FAMILY,
            P1,
            P2,
            P3,
            Arg {
                name: "limit",
                help: "A per-occurrence layer's limit (or \"unlimited\"); ground up if limit and attachment are omitted",
            },
            Arg {
                name: "attachment",
                help: "The per-occurrence layer's attachment; 0 if omitted",
            },
            Arg {
                name: "probabilities",
                help: "Probabilities for VaR and TVaR: a row or column; 0.99, 0.995, 0.999 if omitted",
            },
            Arg {
                name: "step",
                help: "The grid's step, with points; sized automatically if both are omitted",
            },
            Arg {
                name: "points",
                help: "The grid's number of points: the severity is rounded onto it, its mass above lumped at the top",
            },
        ],
        run: aggregate::stats,
    },
    Function {
        symbol: "pro_agg_layers",
        name: "PRO.AGG.LAYERS",
        help: "Per-occurrence layer statistics, one column per layer: chances of attaching and exhausting, annual moments.",
        args: &[
            Arg {
                name: "count_mean",
                help: "Expected number of claims a year",
            },
            Arg {
                name: "dispersion",
                help: "Var N / E N: 1 Poisson (default), above 1 negative binomial, below 1 binomial",
            },
            FAMILY,
            P1,
            P2,
            P3,
            Arg {
                name: "limits",
                help: "Each layer's limit (or \"unlimited\"): a row",
            },
            Arg {
                name: "attachments",
                help: "Each layer's attachment: one, or one per limit",
            },
            Arg {
                name: "shares",
                help: "Each layer's placed share: one, or one per limit; 1 if omitted",
            },
            Arg {
                name: "step",
                help: "The grid's step, with points; sized automatically if both are omitted",
            },
            Arg {
                name: "points",
                help: "The grid's number of points: the severity is rounded onto it, its mass above lumped at the top",
            },
        ],
        run: aggregate::layers,
    },
    Function {
        symbol: "pro_agg_cover",
        name: "PRO.AGG.COVER",
        help: "Aggregate covers (a stop loss) on the annual total, one column per cover: chances of attaching and exhausting, annual moments.",
        args: &[
            Arg {
                name: "count_mean",
                help: "Expected number of claims a year",
            },
            Arg {
                name: "dispersion",
                help: "Var N / E N: 1 Poisson (default), above 1 negative binomial, below 1 binomial",
            },
            FAMILY,
            P1,
            P2,
            P3,
            Arg {
                name: "limits",
                help: "Each cover's annual limit (or \"unlimited\"): a row",
            },
            Arg {
                name: "retentions",
                help: "Each cover's annual retention: one, or one per limit",
            },
            Arg {
                name: "shares",
                help: "Each cover's placed share: one, or one per limit; 1 if omitted",
            },
            Arg {
                name: "occ_limits",
                help: "Per-occurrence layers inuring to the covers: their limits (a row); none if omitted",
            },
            Arg {
                name: "occ_attachments",
                help: "The per-occurrence layers' attachments: one, or one per occurrence limit",
            },
            Arg {
                name: "occ_shares",
                help: "The per-occurrence layers' placed shares: one, or one per occurrence limit; 1 if omitted",
            },
            Arg {
                name: "subject",
                help: "With occurrence layers, the covers sit on the total \"net\" of them (default) or on their \"ceded\" total",
            },
            Arg {
                name: "step",
                help: "The grid's step, with points; sized automatically if both are omitted",
            },
            Arg {
                name: "points",
                help: "The grid's number of points: the severity is rounded onto it, its mass above lumped at the top",
            },
        ],
        run: aggregate::cover,
    },
    Function {
        symbol: "pro_tower_price",
        name: "PRO.TOWER.PRICE",
        help: "A reinsurance tower on the annual grid, each view (gross, layers, cover, net) priced by a distortion: loss, SD, premium, margin, loss ratio, rate on line.",
        args: &[
            Arg {
                name: "count_mean",
                help: "Expected number of claims a year",
            },
            Arg {
                name: "dispersion",
                help: "Var N / E N: 1 Poisson (default), above 1 negative binomial, below 1 binomial",
            },
            FAMILY,
            P1,
            P2,
            P3,
            Arg {
                name: "limit",
                help: "The gross per-claim limit (a policy limit) or \"unlimited\"; none if omitted",
            },
            Arg {
                name: "attachment",
                help: "The gross per-claim deductible; 0 if omitted",
            },
            Arg {
                name: "limits",
                help: "Each per-occurrence layer's limit (or \"unlimited\"): a row",
            },
            Arg {
                name: "attachments",
                help: "Each layer's attachment: one, or one per limit",
            },
            Arg {
                name: "shares",
                help: "Each layer's placed share: one, or one per limit; 1 if omitted",
            },
            Arg {
                name: "agg_deductibles",
                help: "Each layer's annual aggregate deductible: one, or one per limit; 0 if omitted",
            },
            Arg {
                name: "agg_limits",
                help: "Each layer's annual aggregate limit (or \"unlimited\"): one, or one per limit; unlimited if omitted",
            },
            Arg {
                name: "cover_limit",
                help: "A stop loss on the annual total net of the layers: its limit (or \"unlimited\"); none if omitted",
            },
            Arg {
                name: "cover_retention",
                help: "The stop loss's annual retention",
            },
            Arg {
                name: "distortion",
                help: "ccoc, ph, wang, dual, tvar, exp, bitvar, beta, wtdtvar, clin, cll, lep or ly (aggregate's names)",
            },
            Arg {
                name: "parameter",
                help: "A number (ccoc r, ph rho, wang lambda, dual beta, tvar p, exp k), or a range: bitvar p0, p1, w; beta a, b; wtdtvar levels | weights; clin, cll, lep, ly r0 and the parameter",
            },
            Arg {
                name: "step",
                help: "The grid's step, with points; sized automatically if both are omitted",
            },
            Arg {
                name: "points",
                help: "The grid's number of points: the severity is rounded onto it, its mass above lumped at the top",
            },
        ],
        run: aggregate::tower_price,
    },
    Function {
        symbol: "pro_agg_pml",
        name: "PRO.AGG.PML",
        help: "Occurrence and aggregate PML curves by return period: the year's largest occurrence and its total.",
        args: &[
            Arg {
                name: "count_mean",
                help: "Expected number of occurrences a year",
            },
            Arg {
                name: "dispersion",
                help: "Var N / E N: 1 Poisson (default), above 1 negative binomial, below 1 binomial",
            },
            FAMILY,
            P1,
            P2,
            P3,
            Arg {
                name: "return_periods",
                help: "Return periods in years (a row or column; below 1 read as probabilities); 2 to 1000 if omitted",
            },
            Arg {
                name: "step",
                help: "The aggregate grid's step, with points; sized automatically if both are omitted",
            },
            Arg {
                name: "points",
                help: "The aggregate grid's number of points: the severity is rounded onto it, its mass above lumped at the top",
            },
        ],
        run: aggregate::pml,
    },
    Function {
        symbol: "pro_bayes_odp",
        name: "PRO.BAYES.ODP",
        help: "Bayesian ODP chain ladder sampled by MCMC (NUTS): reserves, parameters' posterior, or diagnostics.",
        args: &[
            TRIANGLE,
            Arg {
                name: "output",
                help: "\"reserves\" (default), \"params\" (posterior with R-hat and ESS) or \"diagnostics\"",
            },
            Arg {
                name: "draws",
                help: "Kept draws per chain, after as many warm-up draws (1000 if omitted, at most 20000)",
            },
            Arg {
                name: "chains",
                help: "Number of chains, run in parallel (4 if omitted, at most 8)",
            },
            SEED,
            SIMS,
            PERCENTILES,
            Arg {
                name: "prior_sd",
                help: "SD of the normal priors on the log-scale intercept, origin and development effects (10 if omitted). Keep it wide: a tight one pulls late columns toward the first column's level and raises the reserve",
            },
            PROCESS,
            ORIGINS,
        ],
        run: bayes::bayes_odp,
    },
    Function {
        symbol: "pro_bayes_bf",
        name: "PRO.BAYES.BF",
        help: "Bayesian Bornhuetter-Ferguson by MCMC (NUTS): a gamma prior on each ultimate, a normalised pattern; reserves, ultimates, posterior or diagnostics.",
        args: &[
            TRIANGLE,
            EXPOSURE,
            APRIORI,
            Arg {
                name: "cv",
                help: "CV of the gamma prior on each origin's ultimate (0.1 if omitted); small holds the a priori firmly (classical BF), \"flat\" is the Bayesian chain ladder",
            },
            Arg {
                name: "output",
                help: "\"reserves\" (default), \"ultimates\" (prior and posterior), \"params\" (posterior with R-hat and ESS) or \"diagnostics\"",
            },
            Arg {
                name: "draws",
                help: "Kept draws per chain, after as many warm-up draws (1000 if omitted, at most 20000)",
            },
            Arg {
                name: "chains",
                help: "Number of chains, run in parallel (4 if omitted, at most 8)",
            },
            SEED,
            SIMS,
            PERCENTILES,
            Arg {
                name: "prior_sd",
                help: "SD of the normal priors on the pattern's log share ratios to the first age (10 if omitted)",
            },
            PROCESS,
            ORIGINS,
            Arg {
                name: "common_cv",
                help: "CV of one multiplier common to all origins' prior ultimates (omitted for none); needs a numeric cv",
            },
        ],
        run: bayes::bayes_bf,
    },
    Function {
        symbol: "pro_sev_fit",
        name: "PRO.SEV.FIT",
        help: "Maximum likelihood severity fit to large losses with reporting thresholds and censoring: parameters, fit against the data, or the body families compared.",
        args: &[
            Arg {
                name: "losses",
                help: "The losses: a range; blank cells are skipped",
            },
            Arg {
                name: "family",
                help: "lognormal, gamma, weibull, burr, lomax, pareto, piecewise_pareto or gen_pareto (blank for \"compare\")",
            },
            Arg {
                name: "output",
                help: "\"params\" (default: parameters in PRO.SEV's order, SEs, AIC), \"gof\" (survival and Q-Q at each loss) or \"compare\"",
            },
            Arg {
                name: "t",
                help: "Pareto families: the threshold (a row of thresholds for piecewise_pareto)",
            },
            Arg {
                name: "truncation",
                help: "Pareto and piecewise_pareto: a truncation point, then optionally \"lp\" or \"wd\"",
            },
            Arg {
                name: "thresholds",
                help: "Reporting thresholds: one value or one per loss (0 if omitted)",
            },
            Arg {
                name: "censored",
                help: "TRUE where a policy limit capped the loss: one value or one per loss (FALSE if omitted)",
            },
            Arg {
                name: "weights",
                help: "Weights: one value or one per loss (1 if omitted)",
            },
            Arg {
                name: "upper",
                help: "Body families: truncation from above, the most a loss could be and still be kept; one value or one per loss (none if omitted)",
            },
        ],
        run: fitting::severity_fit,
    },
    Function {
        symbol: "pro_sev_fit_grouped",
        name: "PRO.SEV.FIT.GROUPED",
        help: "A body family fitted to counts of losses in size bands, by maximum likelihood or minimum chi-square: parameters, SEs, chi-square and p-value, or each band's expected count.",
        args: &[
            Arg {
                name: "bounds",
                help: "Band bounds, one more than counts: a row or column; a last cell blank or \"inf\" opens the top band",
            },
            Arg {
                name: "counts",
                help: "The number of losses in each band (b_k, b_k+1]",
            },
            BODY_FAMILY,
            Arg {
                name: "method",
                help: "\"ml\" (default, maximum likelihood) or \"chi_square\" (minimum chi-square)",
            },
            Arg {
                name: "output",
                help: "\"params\" (default) or \"table\" (observed and expected count per band)",
            },
        ],
        run: fitting::grouped,
    },
    Function {
        symbol: "pro_sev_fit_lev",
        name: "PRO.SEV.FIT.LEV",
        help: "A body family fitted by the minimum distance between its limited expected values and empirical ones (Bahnemann): parameters and distance, or each size's fitted limited mean.",
        args: &[
            Arg {
                name: "sizes",
                help: "The sizes x at which the limited means are known: a row or column, increasing",
            },
            Arg {
                name: "lev",
                help: "The empirical limited expected value E[X; x] at each size",
            },
            BODY_FAMILY,
            Arg {
                name: "weights",
                help: "A weight per size for the squared distances (1 if omitted)",
            },
            Arg {
                name: "output",
                help: "\"params\" (default) or \"table\" (empirical and fitted limited mean per size)",
            },
        ],
        run: fitting::lev_distance,
    },
    Function {
        symbol: "pro_evt_thresholds",
        name: "PRO.EVT.THRESHOLDS",
        help: "Threshold choice for a GPD tail: at each threshold the mean excess and the fitted shape and modified scale, with SEs (R ismev).",
        args: &[
            DATA,
            Arg {
                name: "thresholds",
                help: "Thresholds: a row or column (the data's 50%, 55%, ..., 95% quantiles if omitted)",
            },
        ],
        run: evt::thresholds,
    },
    Function {
        symbol: "pro_evt_fit",
        name: "PRO.EVT.FIT",
        help: "Peaks over threshold: the GPD fitted to the exceedances with SEs, and the tail's VaR and TVaR.",
        args: &[
            DATA,
            Arg {
                name: "threshold",
                help: "The threshold u; the GPD fits the values above it, minus u",
            },
            Arg {
                name: "probabilities",
                help: "Probabilities for VaR and TVaR (0.99, 0.995, 0.999 if omitted)",
            },
        ],
        run: evt::fit,
    },
    Function {
        symbol: "pro_evt_hill",
        name: "PRO.EVT.HILL",
        help: "Hill's estimate of the tail index from the k largest values, with its threshold and Pareto alpha.",
        args: &[
            DATA,
            Arg {
                name: "k",
                help: "Numbers of largest values: a row or column (ten from 5% to 50% of the data if omitted)",
            },
        ],
        run: evt::hill_estimates,
    },
    Function {
        symbol: "pro_payout_pattern",
        name: "PRO.PAYOUT.PATTERN",
        help: "A payment pattern by time: share paid, paid to date, outstanding and the discount at a rate.",
        args: &[
            Arg {
                name: "times",
                help: "Payment times in years from inception: a row or column",
            },
            Arg {
                name: "values",
                help: "One per time: shares paid then, the share paid to date, or factors to ultimate (see kind)",
            },
            Arg {
                name: "kind",
                help: "\"shares\" (default, summing to 1), \"paid\" (share paid to date, the last 1) or \"factors\" (to ultimate, such as PRO.CL.CDF, the last 1)",
            },
            Arg {
                name: "shift",
                help: "Years to move the pattern later (0 if omitted; 0.5 for losses occurring evenly over a year)",
            },
            Arg {
                name: "rate",
                help: "Annual effective discount rate (0 if omitted)",
            },
        ],
        run: payout::pattern_table,
    },
    Function {
        symbol: "pro_payout_stats",
        name: "PRO.PAYOUT.STATS",
        help: "A payment pattern's discount factor, duration and discounted capital years at a rate.",
        args: &[
            Arg {
                name: "times",
                help: "Payment times in years from inception: a row or column",
            },
            Arg {
                name: "values",
                help: "One per time: shares paid then, the share paid to date, or factors to ultimate (see kind)",
            },
            Arg {
                name: "rate",
                help: "Annual effective discount rate, such as 0.04",
            },
            Arg {
                name: "kind",
                help: "\"shares\" (default, summing to 1), \"paid\" (share paid to date, the last 1) or \"factors\" (to ultimate, such as PRO.CL.CDF, the last 1)",
            },
            Arg {
                name: "shift",
                help: "Years to move the pattern later (0 if omitted; 0.5 for losses occurring evenly over a year)",
            },
        ],
        run: payout::stats,
    },
    Function {
        symbol: "pro_payout_price",
        name: "PRO.PAYOUT.PRICE",
        help: "A cost-of-capital price over the payout: P = E d + coc (assets - P) A, capital held while losses are outstanding.",
        args: &[
            Arg {
                name: "expected_loss",
                help: "Expected loss E[X], undiscounted",
            },
            Arg {
                name: "assets",
                help: "Assets backing the loss: the one-year risk measure, such as a VaR from PRO.AGG.STATS",
            },
            Arg {
                name: "coc",
                help: "Cost of capital, a spread over the discount rate, such as 0.1",
            },
            Arg {
                name: "times",
                help: "Payment times in years from inception: a row or column",
            },
            Arg {
                name: "values",
                help: "One per time: shares paid then, the share paid to date, or factors to ultimate (see kind)",
            },
            Arg {
                name: "rate",
                help: "Annual effective discount rate, such as 0.04",
            },
            Arg {
                name: "kind",
                help: "\"shares\" (default, summing to 1), \"paid\" (share paid to date, the last 1) or \"factors\" (to ultimate, such as PRO.CL.CDF, the last 1)",
            },
            Arg {
                name: "shift",
                help: "Years to move the pattern later (0 if omitted; 0.5 for losses occurring evenly over a year)",
            },
        ],
        run: payout::price,
    },
    Function {
        symbol: "pro_quote",
        name: "PRO.QUOTE",
        help: "A cover's technical premium (R + f) / (1 - e - b) with its build-up and market ratios, and the cover at each quoted premium.",
        args: &[
            Arg {
                name: "expected_loss",
                help: "Expected loss E[X]",
            },
            Arg {
                name: "risk_premium",
                help: "The risk-loaded premium R, such as PRO.TOWER.PRICE's premium",
            },
            Arg {
                name: "assets",
                help: "Assets backing the loss, for capital and its return (blank leaves them out)",
            },
            Arg {
                name: "expenses",
                help: "Internal expenses as a share of the premium (0 if omitted)",
            },
            Arg {
                name: "brokerage",
                help: "Brokerage as a share of the premium (0 if omitted)",
            },
            Arg {
                name: "fixed",
                help: "A fixed expense per contract, an amount (0 if omitted)",
            },
            Arg {
                name: "limit",
                help: "The limit at the placed share, for the rate on line and payback",
            },
            Arg {
                name: "quoted",
                help: "Quoted premiums: a value or a row, one column each",
            },
        ],
        run: quote::quote,
    },
    Function {
        symbol: "pro_distortion",
        name: "PRO.DISTORTION",
        help: "A distortion's g(s) at each exceedance probability s.",
        args: &[
            Arg {
                name: "s",
                help: "Exceedance probabilities in [0, 1]: a number or a range",
            },
            Arg {
                name: "family",
                help: "ccoc, ph, wang, dual, tvar, exp, bitvar, beta, wtdtvar, or with r0 clin, cll, lep, ly (aggregate's names)",
            },
            Arg {
                name: "parameter",
                help: "r (ccoc), rho (ph), lambda (wang), beta (dual), p (tvar), k (exp), slope (clin), b (cll), r (lep, ly); a range for bitvar (p0, p1, w), beta (a, b) and wtdtvar (levels | weights)",
            },
            Arg {
                name: "r0",
                help: "clin, cll, lep and ly only: the intercept r0",
            },
        ],
        run: distortions::g,
    },
    Function {
        symbol: "pro_distortion_price",
        name: "PRO.DISTORTION.PRICE",
        help: "A distortion's risk measure of a discrete or simulated distribution, with its mean and margin.",
        args: &[
            Arg {
                name: "values",
                help: "Outcomes: a range, any order (simulated draws when probs is omitted)",
            },
            Arg {
                name: "probs",
                help: "One probability per value, summing to 1 (equally likely if omitted)",
            },
            Arg {
                name: "family",
                help: "ccoc, ph, wang, dual, tvar, exp, bitvar, beta, wtdtvar, or with r0 clin, cll, lep, ly (aggregate's names)",
            },
            Arg {
                name: "parameter",
                help: "r (ccoc), rho (ph), lambda (wang), beta (dual), p (tvar), k (exp), slope (clin), b (cll), r (lep, ly); a range for bitvar (p0, p1, w), beta (a, b) and wtdtvar (levels | weights)",
            },
            Arg {
                name: "r0",
                help: "clin, cll, lep and ly only: the intercept r0",
            },
        ],
        run: distortions::price,
    },
    Function {
        symbol: "pro_distortion_calibrate",
        name: "PRO.DISTORTION.CALIBRATE",
        help: "Each family's parameter that prices the outcomes at a target premium (CCoC, PH, Wang, dual, TVaR by default).",
        args: &[
            Arg {
                name: "values",
                help: "Outcomes: a range, any order (simulated draws when probs is omitted)",
            },
            Arg {
                name: "probs",
                help: "One probability per value, summing to 1 (equally likely if omitted)",
            },
            Arg {
                name: "premium",
                help: "The target premium, between the mean and the largest outcome (cap the outcomes at the assets first)",
            },
            Arg {
                name: "families",
                help: "Family names: a row or column (ccoc, ph, wang, dual, tvar if omitted)",
            },
            Arg {
                name: "r0",
                help: "clin, cll, lep and ly only: the intercept r0",
            },
        ],
        run: distortions::calibrate_families,
    },
    Function {
        symbol: "pro_copula_cdf",
        name: "PRO.COPULA.CDF",
        help: "A copula's distribution function C(u) at each point, one point a row (Gaussian in two dimensions, Archimedean in any).",
        args: &[
            Arg {
                name: "points",
                help: "Points in [0, 1]^d, one a row, one column a dimension",
            },
            COPULA_FAMILY,
            COPULA_PARAMETER,
            Arg {
                name: "nu",
                help: "t only: the degrees of freedom (the t has no closed-form cdf)",
            },
        ],
        run: copulas::cdf,
    },
    Function {
        symbol: "pro_copula_sim",
        name: "PRO.COPULA.SIM",
        help: "Uniforms simulated from a copula, one row a simulation and one column a dimension; reproducible by seed. Turn them into losses with PRO.SEV.QUANTILE.",
        args: &[
            COPULA_FAMILY,
            COPULA_PARAMETER,
            Arg {
                name: "sims",
                help: "Simulations (rows), 1 to 1,000,000 (10,000 if omitted)",
            },
            Arg {
                name: "dim",
                help: "Dimensions (columns): 2 if omitted, or the correlation matrix's size",
            },
            Arg {
                name: "seed",
                help: "A whole number from 0 (0 if omitted); row i is stream i of the seed",
            },
            Arg {
                name: "nu",
                help: "t only: the degrees of freedom",
            },
        ],
        run: copulas::sim,
    },
    Function {
        symbol: "pro_iman_conover",
        name: "PRO.IMAN.CONOVER",
        help: "Simulated columns reordered to a target rank correlation by Iman and Conover's method, each column keeping its own values.",
        args: &[
            Arg {
                name: "draws",
                help: "Simulations: one column a variable, one row a simulation",
            },
            Arg {
                name: "correlation",
                help: "The target correlation: a square matrix, one row and column a variable, or one number for every pair",
            },
            Arg {
                name: "seed",
                help: "A whole number from 0 (0 if omitted)",
            },
            Arg {
                name: "dof",
                help: "Student t scores with these degrees of freedom, for more joint extremes (normal scores if omitted)",
            },
            Arg {
                name: "shuffle",
                help: "TRUE shuffles the result's rows (FALSE if omitted: the first column comes back sorted)",
            },
        ],
        run: copulas::iman_conover,
    },
    Function {
        symbol: "pro_swing",
        name: "PRO.SWING",
        help: "A swing-rated or retro premium at each loss: clip(basic + lcm x loss, minimum, maximum) (aggregate's contract_terms).",
        args: &[
            Arg {
                name: "loss",
                help: "Losses: a number or a range (a swing layer's ceded loss at its share)",
            },
            Arg {
                name: "basic",
                help: "Basic premium at 100%",
            },
            Arg {
                name: "lcm",
                help: "Loss conversion factor, such as 100/80 (times the tax multiplier for a retro plan)",
            },
            Arg {
                name: "minimum",
                help: "Minimum premium at 100% (basic if omitted)",
            },
            Arg {
                name: "maximum",
                help: "Maximum premium at 100% (no cap if omitted)",
            },
            Arg {
                name: "share",
                help: "Placed share of a swing-rated layer (1 if omitted)",
            },
        ],
        run: terms::swing,
    },
    Function {
        symbol: "pro_sliding_scale",
        name: "PRO.SLIDING.SCALE",
        help: "A sliding-scale ceding commission at each ceded loss: the rate at loss / premium, linear between anchors, times the premium.",
        args: &[
            Arg {
                name: "loss",
                help: "Ceded losses: a number or a range",
            },
            Arg {
                name: "premium",
                help: "The premium at 100%, for the loss ratio",
            },
            Arg {
                name: "commissions",
                help: "Commission rates at the anchors: a row or column",
            },
            Arg {
                name: "loss_ratios",
                help: "The anchors' loss ratios, increasing: one per commission",
            },
        ],
        run: terms::sliding_scale,
    },
    Function {
        symbol: "pro_profit_commission",
        name: "PRO.PROFIT.COMMISSION",
        help: "A profit commission at each ceded loss: share x max(premium x (1 - allowance) - loss, 0).",
        args: &[
            Arg {
                name: "loss",
                help: "Ceded losses: a number or a range",
            },
            Arg {
                name: "premium",
                help: "The premium at 100%, for the loss ratio",
            },
            Arg {
                name: "share",
                help: "Share of the profit returned to the cedant",
            },
            Arg {
                name: "allowance",
                help: "The reinsurer's expense and margin allowance, a share of premium",
            },
        ],
        run: terms::profit_commission,
    },
    Function {
        symbol: "pro_profit_commission_term",
        name: "PRO.PROFIT.COMMISSION.TERM",
        help: "A profit commission in each year of a term, deficits carried forward and used oldest first; one row a term, one column a year.",
        args: &[
            TERM_LOSSES,
            Arg {
                name: "premium",
                help: "The premium a year at 100%",
            },
            Arg {
                name: "share",
                help: "Share of the profit returned to the cedant",
            },
            Arg {
                name: "allowance",
                help: "The reinsurer's expense and margin allowance, a share of premium",
            },
            CARRY_YEARS,
        ],
        run: terms::profit_commission_term,
    },
    Function {
        symbol: "pro_sliding_scale_term",
        name: "PRO.SLIDING.SCALE.TERM",
        help: "A sliding-scale ceding commission in each year of a term, deficits and credits carried forward; one row a term, one column a year.",
        args: &[
            TERM_LOSSES,
            Arg {
                name: "premium",
                help: "The premium a year at 100%, for the loss ratio",
            },
            Arg {
                name: "commissions",
                help: "Commission rates at the anchors: a row or column",
            },
            Arg {
                name: "loss_ratios",
                help: "The anchors' loss ratios, increasing, one per commission",
            },
            Arg {
                name: "carry",
                help: "deficit (loss above the last anchor), credit (below the first) or both (if omitted)",
            },
            CARRY_YEARS,
        ],
        run: terms::sliding_scale_term,
    },
    Function {
        symbol: "pro_layer_terms",
        name: "PRO.LAYER.TERMS",
        help: "A layer's contract terms in expectation over its annual loss: premium, reinstatement premium, ceding and profit commissions, margin.",
        args: &[
            Arg {
                name: "count_mean",
                help: "Expected number of claims a year",
            },
            Arg {
                name: "dispersion",
                help: "Var N / E N: 1 Poisson (default), above 1 negative binomial, below 1 binomial",
            },
            FAMILY,
            P1,
            P2,
            P3,
            Arg {
                name: "layer_limit",
                help: "The layer's per-occurrence limit",
            },
            Arg {
                name: "layer_attachment",
                help: "The layer's attachment",
            },
            Arg {
                name: "premium",
                help: "Deposit premium at 100%",
            },
            Arg {
                name: "reinstatements",
                help: "Paid reinstatements: a row of rates, each a share of the deposit (pro rata as to amount)",
            },
            Arg {
                name: "commissions",
                help: "A flat ceding commission rate, or a range of rates with loss_ratios for a sliding scale",
            },
            Arg {
                name: "loss_ratios",
                help: "A sliding scale's anchor loss ratios, one per commission",
            },
            Arg {
                name: "profit_share",
                help: "Profit commission: the share of profit returned",
            },
            Arg {
                name: "profit_allowance",
                help: "Profit commission: the reinsurer's allowance, a share of premium",
            },
            Arg {
                name: "share",
                help: "Placed share (1 if omitted)",
            },
            Arg {
                name: "policy_limit",
                help: "A per-claim limit on the gross loss (none if omitted)",
            },
            Arg {
                name: "step",
                help: "Grid step; give with points, or omit both to size the grid automatically",
            },
            Arg {
                name: "points",
                help: "Grid points; give with step",
            },
        ],
        run: aggregate::layer_terms,
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
