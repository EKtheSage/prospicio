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

pub mod pricing;
pub mod reserving;
pub mod sheet;
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
    help: "\"pareto\" (p1 = threshold t, p2 = alpha), \"lognormal\" (meanlog, sdlog) or \"lognormal_cv\" (mean, CV)",
};
const P1: Arg = Arg {
    name: "p1",
    help: "First parameter: the Pareto threshold t, the lognormal meanlog, or the mean",
};
const P2: Arg = Arg {
    name: "p2",
    help: "Second parameter: the Pareto alpha, the lognormal sdlog, or the CV",
};
const TRUNCATION: Arg = Arg {
    name: "truncation",
    help: "Pareto only: the loss size the Pareto is truncated at; omitted for none",
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
        args: &[TRIANGLE, AVERAGE, TAIL, ORIGINS],
        run: reserving::cl_summary,
    },
    Function {
        symbol: "pro_cl_ldf",
        name: "PRO.CL.LDF",
        help: "Selected age-to-age factors, one per triangle column; the last is the tail.",
        args: &[TRIANGLE, AVERAGE, TAIL],
        run: reserving::cl_ldf,
    },
    Function {
        symbol: "pro_cl_cdf",
        name: "PRO.CL.CDF",
        help: "Age-to-ultimate factors, one per triangle column.",
        args: &[TRIANGLE, AVERAGE, TAIL],
        run: reserving::cl_cdf,
    },
    Function {
        symbol: "pro_cl_ultimate",
        name: "PRO.CL.ULTIMATE",
        help: "Chain ladder ultimate per origin.",
        args: &[TRIANGLE, AVERAGE, TAIL],
        run: reserving::cl_ultimate,
    },
    Function {
        symbol: "pro_cl_reserve",
        name: "PRO.CL.RESERVE",
        help: "Chain ladder reserve (ultimate less latest) per origin.",
        args: &[TRIANGLE, AVERAGE, TAIL],
        run: reserving::cl_reserve,
    },
    Function {
        symbol: "pro_cl_factors",
        name: "PRO.CL.FACTORS",
        help: "Development exhibit: LDF, CDF, Mack sigma and standard error per link, and the tail.",
        args: &[TRIANGLE, AVERAGE, TAIL],
        run: reserving::cl_factors,
    },
    Function {
        symbol: "pro_mack",
        name: "PRO.MACK",
        help: "Mack exhibit: ultimate, reserve, process, parameter and total standard error and CV per origin, with a total.",
        args: &[TRIANGLE, AVERAGE, TAIL, ORIGINS],
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
        ],
        run: reserving::mack_bootstrap_sims,
    },
    Function {
        symbol: "pro_lev",
        name: "PRO.LEV",
        help: "Limited expected value E[min(X, limit)] of a Pareto or lognormal severity, per limit.",
        args: &[LIMIT, FAMILY, P1, P2, TRUNCATION],
        run: pricing::lev,
    },
    Function {
        symbol: "pro_layer",
        name: "PRO.LAYER",
        help: "Expected loss to the layer limit xs attachment per ground-up loss, Pareto or lognormal.",
        args: &[LIMIT, ATTACHMENT, FAMILY, P1, P2, TRUNCATION],
        run: pricing::layer,
    },
    Function {
        symbol: "pro_layer_cv",
        name: "PRO.LAYER.CV",
        help: "CV of the loss to the layer, given a loss above its attachment.",
        args: &[LIMIT, ATTACHMENT, FAMILY, P1, P2, TRUNCATION],
        run: pricing::layer_coefficient_of_variation,
    },
    Function {
        symbol: "pro_ilf",
        name: "PRO.ILF",
        help: "Increased limit factor LEV(limit) / LEV(basic_limit), Pareto or lognormal.",
        args: &[
            LIMIT,
            Arg {
                name: "basic_limit",
                help: "The basic limit the factors are relative to",
            },
            FAMILY,
            P1,
            P2,
            TRUNCATION,
        ],
        run: pricing::increased_limit_factor,
    },
];

/// Runs the function exported as `symbol` on `args`, catching a panic so
/// that it never unwinds into Excel. An error comes back as a single text
/// cell starting with `#PROSPICIO!`, which shows the user why.
pub fn call(symbol: &str, args: Vec<Grid>) -> Grid {
    let Some(f) = FUNCTIONS.iter().find(|f| f.symbol == symbol) else {
        return error_cell(&format!("no function {symbol}"));
    };
    match std::panic::catch_unwind(|| (f.run)(&args)) {
        Ok(Ok(grid)) => grid,
        Ok(Err(message)) => error_cell(&message),
        Err(_) => error_cell("internal error (a panic); please report it"),
    }
}

fn error_cell(message: &str) -> Grid {
    Grid::scalar(Value::Text(format!("#PROSPICIO! {message}")))
}
