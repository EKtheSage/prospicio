//! The Excel C API layer: XLOPER12 values, the callback into Excel, the
//! add-in entry points Excel calls (`xlAutoOpen` and the others) and one
//! exported function per entry of [`FUNCTIONS`].
//!
//! Every function is registered with arguments and result of type `Q`
//! (an XLOPER12 holding values, never a reference) and as thread safe
//! (`$`), so Excel recalculates them on all its threads. A result is
//! allocated here and marked `xlbitDLLFree`; Excel hands it back to
//! [`xlAutoFree12`] once it has copied it.

use std::ffi::c_void;
use std::sync::OnceLock;

use crate::sheet::{CellError, Grid, Value};
use crate::{ADDIN_NAME, CATEGORY, FUNCTIONS, Function, call};

const XLTYPE_NUM: u32 = 0x0001;
const XLTYPE_STR: u32 = 0x0002;
const XLTYPE_BOOL: u32 = 0x0004;
const XLTYPE_ERR: u32 = 0x0010;
const XLTYPE_MULTI: u32 = 0x0040;
const XLTYPE_MISSING: u32 = 0x0080;
const XLTYPE_NIL: u32 = 0x0100;
const XLTYPE_INT: u32 = 0x0800;
const XLBIT_DLL_FREE: u32 = 0x4000;

const XLERR_NULL: i32 = 0;
const XLERR_DIV0: i32 = 7;
const XLERR_VALUE: i32 = 15;
const XLERR_REF: i32 = 23;
const XLERR_NAME: i32 = 29;
const XLERR_NUM: i32 = 36;
const XLERR_NA: i32 = 42;

const XL_SPECIAL: i32 = 0x4000;
const XL_FREE: i32 = XL_SPECIAL;
const XL_GET_NAME: i32 = 9 | XL_SPECIAL;
const XLF_REGISTER: i32 = 149;
const XLRET_SUCCESS: i32 = 0;

/// Excel's longest string, in UTF-16 units.
const MAX_STRING: usize = 32767;

/// The value part of an XLOPER12: 24 bytes on 64-bit Windows, the size of
/// its largest member (`flow`).
#[repr(C)]
#[derive(Clone, Copy)]
union XlVal {
    num: f64,
    str_: *mut u16,
    xbool: i32,
    err: i32,
    w: i32,
    array: XlArray,
    _size: [u64; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XlArray {
    lparray: *mut XlOper,
    rows: i32,
    columns: i32,
}

/// Excel's XLOPER12.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct XlOper {
    val: XlVal,
    xltype: u32,
}

const _: () = assert!(std::mem::size_of::<XlOper>() == 32);

impl XlOper {
    fn missing() -> Self {
        Self {
            val: XlVal { w: 0 },
            xltype: XLTYPE_MISSING,
        }
    }

    fn num(x: f64) -> Self {
        Self {
            val: XlVal { num: x },
            xltype: XLTYPE_NUM,
        }
    }

    fn int(w: i32) -> Self {
        Self {
            val: XlVal { w },
            xltype: XLTYPE_INT,
        }
    }

    /// A string we allocated, freed by [`free_owned`].
    fn str(s: &str) -> Self {
        let mut units: Vec<u16> = Vec::with_capacity(s.len() + 1);
        units.push(0);
        units.extend(s.encode_utf16().take(MAX_STRING));
        units[0] = (units.len() - 1) as u16;
        let ptr = Box::into_raw(units.into_boxed_slice()) as *mut u16;
        Self {
            val: XlVal { str_: ptr },
            xltype: XLTYPE_STR,
        }
    }

    fn from_value(v: &Value) -> Self {
        match v {
            // An empty string, not Nil: Excel shows Nil in an array as 0.
            Value::Empty => Self::str(""),
            Value::Number(x) => Self::num(*x),
            Value::Text(s) => Self::str(s),
            Value::Bool(b) => Self {
                val: XlVal { xbool: *b as i32 },
                xltype: XLTYPE_BOOL,
            },
            Value::Error(e) => Self {
                val: XlVal {
                    err: match e {
                        CellError::Null => XLERR_NULL,
                        CellError::Div0 => XLERR_DIV0,
                        CellError::Value => XLERR_VALUE,
                        CellError::Ref => XLERR_REF,
                        CellError::Name => XLERR_NAME,
                        CellError::Num => XLERR_NUM,
                        CellError::NA => XLERR_NA,
                    },
                },
                xltype: XLTYPE_ERR,
            },
        }
    }

    /// A result for Excel: a scalar for a 1 x 1 grid, else an array.
    fn from_grid(grid: &Grid) -> Self {
        if let Some(v) = grid.as_scalar() {
            return Self::from_value(v);
        }
        let cells: Vec<XlOper> = grid.cells.iter().map(Self::from_value).collect();
        let ptr = Box::into_raw(cells.into_boxed_slice()) as *mut XlOper;
        Self {
            val: XlVal {
                array: XlArray {
                    lparray: ptr,
                    rows: grid.rows as i32,
                    columns: grid.columns as i32,
                },
            },
            xltype: XLTYPE_MULTI,
        }
    }
}

/// Reads one Excel-owned value.
///
/// # Safety
/// `x` is a valid XLOPER12 of a scalar type.
unsafe fn read_value(x: &XlOper) -> Value {
    unsafe {
        match x.xltype & 0x0FFF {
            XLTYPE_NUM => Value::Number(x.val.num),
            XLTYPE_INT => Value::Number(f64::from(x.val.w)),
            XLTYPE_BOOL => Value::Bool(x.val.xbool != 0),
            XLTYPE_STR => {
                let p = x.val.str_;
                if p.is_null() {
                    return Value::Empty;
                }
                let len = usize::from(*p);
                let units = std::slice::from_raw_parts(p.add(1), len);
                Value::Text(String::from_utf16_lossy(units))
            }
            XLTYPE_ERR => Value::Error(match x.val.err {
                XLERR_NULL => CellError::Null,
                XLERR_DIV0 => CellError::Div0,
                XLERR_REF => CellError::Ref,
                XLERR_NAME => CellError::Name,
                XLERR_NUM => CellError::Num,
                XLERR_NA => CellError::NA,
                _ => CellError::Value,
            }),
            XLTYPE_MISSING | XLTYPE_NIL => Value::Empty,
            _ => Value::Error(CellError::Value),
        }
    }
}

/// Reads an argument of type `Q`: a scalar, an array, or missing.
///
/// # Safety
/// `x` is null or points to a valid XLOPER12 that Excel owns for the call.
unsafe fn read_grid(x: *const XlOper) -> Grid {
    if x.is_null() {
        return Grid::missing();
    }
    unsafe {
        let x = &*x;
        if x.xltype & 0x0FFF != XLTYPE_MULTI {
            return Grid::scalar(read_value(x));
        }
        let a = x.val.array;
        let (rows, columns) = (a.rows.max(0) as usize, a.columns.max(0) as usize);
        let cells = std::slice::from_raw_parts(a.lparray, rows * columns)
            .iter()
            .map(|c| read_value(c))
            .collect();
        Grid {
            rows,
            columns,
            cells,
        }
    }
}

/// Frees what [`XlOper::str`] and [`XlOper::from_grid`] allocated inside
/// `x`, not `x` itself.
///
/// # Safety
/// `x` was built by those constructors and not freed before.
unsafe fn free_owned(x: &XlOper) {
    unsafe {
        match x.xltype & 0x0FFF {
            XLTYPE_STR => {
                let p = x.val.str_;
                let len = usize::from(*p) + 1;
                drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(p, len)));
            }
            XLTYPE_MULTI => {
                let a = x.val.array;
                let n = (a.rows * a.columns) as usize;
                let cells = Box::from_raw(std::ptr::slice_from_raw_parts_mut(a.lparray, n));
                for c in cells.iter() {
                    free_owned(c);
                }
            }
            _ => {}
        }
    }
}

/// A result Excel frees through [`xlAutoFree12`].
fn result(grid: &Grid) -> *mut XlOper {
    let mut x = XlOper::from_grid(grid);
    x.xltype |= XLBIT_DLL_FREE;
    Box::into_raw(Box::new(x))
}

/// Called by Excel with each result marked `xlbitDLLFree`.
///
/// # Safety
/// `x` is a pointer [`result`] returned.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn xlAutoFree12(x: *mut XlOper) {
    if x.is_null() {
        return;
    }
    unsafe {
        let x = Box::from_raw(x);
        free_owned(&x);
    }
}

type MdCallBack12 = unsafe extern "system" fn(i32, i32, *const *mut XlOper, *mut XlOper) -> i32;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetModuleHandleW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
}

/// Excel's `MdCallBack12`, exported by the Excel process that loaded us.
fn callback() -> Option<MdCallBack12> {
    static CALLBACK: OnceLock<usize> = OnceLock::new();
    let p = *CALLBACK.get_or_init(|| unsafe {
        let exe = GetModuleHandleW(std::ptr::null());
        GetProcAddress(exe, c"MdCallBack12".as_ptr().cast()) as usize
    });
    (p != 0).then(|| unsafe { std::mem::transmute::<usize, MdCallBack12>(p) })
}

/// Calls Excel's function `xlfn` (as `Excel12v`), returning its result,
/// which the caller frees with `xlFree`.
fn excel(xlfn: i32, args: &mut [XlOper]) -> Option<XlOper> {
    let cb = callback()?;
    let ptrs: Vec<*mut XlOper> = args.iter_mut().map(|a| a as *mut XlOper).collect();
    let mut res = XlOper::missing();
    let rc = unsafe { cb(xlfn, ptrs.len() as i32, ptrs.as_ptr(), &mut res) };
    (rc == XLRET_SUCCESS).then_some(res)
}

/// Frees a result Excel allocated (`xlFree`; harmless on a number).
fn excel_free(mut x: XlOper) {
    excel(XL_FREE, std::slice::from_mut(&mut x));
}

/// Registers `f` with Excel under the add-in's path `dll`.
fn register(dll: &XlOper, f: &Function) -> bool {
    // Type text: the result, then each argument, all `Q`; `$` thread safe.
    let types = format!("Q{}$", "Q".repeat(f.args.len()));
    let arg_names = f.args.iter().map(|a| a.name).collect::<Vec<_>>().join(",");
    let mut args = vec![
        *dll,
        XlOper::str(f.symbol),
        XlOper::str(&types),
        XlOper::str(f.name),
        XlOper::str(&arg_names),
        XlOper::int(1),
        XlOper::str(CATEGORY),
        XlOper::str(""),
        XlOper::str(""),
        XlOper::str(f.help),
    ];
    // The Insert Function dialog drops the last two characters of the last
    // argument's help, a long-standing Excel quirk; pad it.
    args.extend(f.args.iter().enumerate().map(|(i, a)| {
        if i + 1 == f.args.len() {
            XlOper::str(&format!("{}  ", a.help))
        } else {
            XlOper::str(a.help)
        }
    }));
    let res = excel(XLF_REGISTER, &mut args);
    // Free our strings, not the add-in path, which belongs to Excel.
    for a in &args[1..] {
        unsafe { free_owned(a) };
    }
    match res {
        Some(r) => {
            let ok = r.xltype & 0x0FFF == XLTYPE_NUM;
            excel_free(r);
            ok
        }
        None => false,
    }
}

/// Called by Excel when the add-in loads: registers every function.
#[unsafe(no_mangle)]
pub extern "system" fn xlAutoOpen() -> i32 {
    let Some(dll) = excel(XL_GET_NAME, &mut []) else {
        return 0;
    };
    for f in FUNCTIONS {
        register(&dll, f);
    }
    excel_free(dll);
    1
}

/// Called by Excel when the add-in unloads.
#[unsafe(no_mangle)]
pub extern "system" fn xlAutoClose() -> i32 {
    1
}

#[unsafe(no_mangle)]
pub extern "system" fn xlAutoAdd() -> i32 {
    1
}

#[unsafe(no_mangle)]
pub extern "system" fn xlAutoRemove() -> i32 {
    1
}

/// The add-in's name in the Add-ins dialog.
///
/// # Safety
/// `action` is null or a valid XLOPER12.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn xlAddInManagerInfo12(action: *const XlOper) -> *mut XlOper {
    let name = match unsafe { read_grid(action) }.as_scalar() {
        Some(Value::Number(n)) if *n == 1.0 => Value::Text(ADDIN_NAME.into()),
        _ => Value::Error(CellError::Value),
    };
    result(&Grid::scalar(name))
}

/// Exports `$symbol` taking one `Q` argument per name, dispatching to the
/// catalogue entry with that symbol.
macro_rules! udf {
    ($symbol:ident $(, $arg:ident)*) => {
        /// # Safety
        /// Called by Excel with valid XLOPER12 arguments.
        #[unsafe(no_mangle)]
        pub unsafe extern "system" fn $symbol($($arg: *const XlOper),*) -> *mut XlOper {
            let args = vec![$(unsafe { read_grid($arg) }),*];
            result(&call(stringify!($symbol), args))
        }
    };
}

// One line per export; tests/catalogue.rs checks each against FUNCTIONS.
#[rustfmt::skip]
mod exports {
    use super::*;

    udf!(pro_version);
    udf!(pro_cl, triangle, average, tail, origins, selected, kind, weight, pattern_ages);
    udf!(pro_cl_ldf, triangle, average, tail, selected, kind, weight, pattern_ages);
    udf!(pro_cl_cdf, triangle, average, tail, selected, kind, weight, pattern_ages);
    udf!(pro_cl_ultimate, triangle, average, tail, selected, kind, weight, pattern_ages);
    udf!(pro_cl_reserve, triangle, average, tail, selected, kind, weight, pattern_ages);
    udf!(pro_cl_factors, triangle, average, tail, selected, kind, weight, sigma, pattern_ages);
    udf!(pro_mack, triangle, average, tail, origins, selected, kind, weight, sigma, pattern_ages);
    udf!(pro_odp, triangle, sims, seed, percentiles, tail, process, origins, average);
    udf!(pro_odp_sims, triangle, sims, seed, tail, process, origins, average);
    udf!(pro_mack_boot, triangle, sims, seed, percentiles, average, tail, process, origins, sigma);
    udf!(pro_mack_boot_sims, triangle, sims, seed, average, tail, process, origins, sigma);
    udf!(pro_mack_cdr, triangle, average, tail, origins, selected, kind, weight, sigma, pattern_ages);
    udf!(pro_mack_cdr_years, triangle, average, tail, origins, selected, kind, weight, sigma, pattern_ages);
    udf!(pro_odp_cdr, triangle, sims, seed, percentiles, tail, process, origins, average);
    udf!(pro_mack_boot_cdr, triangle, sims, seed, percentiles, average, tail, process, origins, sigma);
    udf!(pro_el, triangle, exposure, apriori, average, tail, origins, selected, kind, weight, pattern_ages);
    udf!(pro_bf, triangle, exposure, apriori, average, tail, origins, selected, kind, weight, pattern_ages);
    udf!(pro_benktander, triangle, exposure, apriori, iterations, average, tail, origins, selected, kind, weight, pattern_ages);
    udf!(pro_capecod, triangle, exposure, trend, decay, average, tail, origins, selected, kind, weight, pattern_ages);
    udf!(pro_diag_cl, origins, ages, latest, pattern, pattern_ages, kind, tail, grain, valuation, past_end, pattern_origin);
    udf!(pro_diag_el, origins, ages, latest, exposure, apriori, pattern, pattern_ages, kind, tail, grain, valuation, past_end, pattern_origin);
    udf!(pro_diag_bf, origins, ages, latest, exposure, apriori, pattern, pattern_ages, kind, tail, grain, valuation, past_end, pattern_origin);
    udf!(pro_diag_benktander, origins, ages, latest, exposure, apriori, pattern, iterations, pattern_ages, kind, tail, grain, valuation, past_end, pattern_origin);
    udf!(pro_diag_capecod, origins, ages, latest, exposure, pattern, trend, decay, pattern_ages, kind, tail, grain, valuation, past_end, pattern_origin);
    udf!(pro_pattern_average, pattern_ages, pattern, pattern_origin, grain, kind);
    udf!(pro_clark, triangle, exposure, curve, max_age, origins);
    udf!(pro_clark_params, triangle, exposure, curve, max_age);
    udf!(pro_odp_glm, triangle, origins);
    udf!(pro_odp_glm_params, triangle, origins);
    udf!(pro_odp_glm_residuals, triangle, by, origins);
    udf!(pro_tri_incremental, triangle);
    udf!(pro_tri_cumulative, triangle);
    udf!(pro_tri_linkratios, triangle);
    udf!(pro_tri_latest, triangle);
    udf!(pro_sev_cdf, x, family, p1, p2, p3);
    udf!(pro_sev_survival, x, family, p1, p2, p3);
    udf!(pro_sev_quantile, p, family, p1, p2, p3);
    udf!(pro_sev_stats, family, p1, p2, p3);
    udf!(pro_ler, deductible, family, p1, p2, p3);
    udf!(pro_pareto_extrapolate, from_limit, from_attachment, to_limit, to_attachment, alpha, truncation);
    udf!(pro_pareto_alpha, limit1, attachment1, loss1, limit2, attachment2, loss2, truncation);
    udf!(pro_pareto_alpha_freqlayer, threshold, frequency, limit, attachment, loss, truncation);
    udf!(pro_pareto_alpha_freqs, threshold1, frequency1, threshold2, frequency2, truncation);
    udf!(pro_lev, limit, family, p1, p2, p3);
    udf!(pro_layer, limit, attachment, family, p1, p2, p3);
    udf!(pro_layer_var, limit, attachment, family, p1, p2, p3);
    udf!(pro_layer_cv, limit, attachment, family, p1, p2, p3);
    udf!(pro_ilf, limit, basic_limit, family, p1, p2, p3);
    udf!(pro_exposure_curve, x, curve, p1, p2, p3);
    udf!(pro_exposure_share, limit, attachment, mpl, curve, p1, p2, p3);
    udf!(pro_exposure_quantile, p, curve, p1, p2, p3);
    udf!(pro_exposure_stats, curve, p1, p2, p3);
    udf!(pro_swissre_c, rate);
    udf!(pro_swissre_c_layer, limit, attachment, mpl, share);
    udf!(pro_profile, sums_insured, expected_losses, limit, attachment, curve, p1, p2, p3, lower, upper, spread, retention, lines);
    udf!(pro_cred_standard, p, k, kind, cv);
    udf!(pro_cred_z, n, standard);
    udf!(pro_buhlmann, ratios, weights, estimator, groups);
    udf!(pro_buhlmann_params, ratios, weights, estimator);
    udf!(pro_yearfraction, date);
    udf!(pro_onlevel, effective, changes, from, to, period, term, applies, given);
    udf!(pro_ratelevel, effective, changes, from, to, period, term, applies, given);
    udf!(pro_avgdate, from, to, period, term, basis);
    udf!(pro_trend_fit, times, values, kind);
    udf!(pro_lr_experience, earned_premium, losses, on_level, premium_trend, development, loss_trend, lae);
    udf!(pro_pp_experience, exposure, losses, development, loss_trend, lae);
    udf!(pro_indicated_change, loss_ratio, variable, profit, fixed_ratio);
    udf!(pro_indicated_rate, pure_premium, variable, profit, fixed_per_exposure);
    udf!(pro_permissible_lr, variable, profit, fixed_ratio);
    udf!(pro_trended_present, prior_indicated, prior_implemented, trend);
    udf!(pro_cred_weighted, estimate, credibility, complement);
    udf!(pro_expenses, ratios, fixed_shares);
    udf!(pro_burningcost, accident, amounts, limits, attachments, years, premiums, development, trend, trend_to, threshold, agg_deductible, agg_limit);
    udf!(pro_agg_stats, count_mean, dispersion, family, p1, p2, p3, limit, attachment, probabilities, step, points);
    udf!(pro_agg_layers, count_mean, dispersion, family, p1, p2, p3, limits, attachments, shares, step, points);
    udf!(pro_agg_cover, count_mean, dispersion, family, p1, p2, p3, limits, retentions, shares, occ_limits, occ_attachments, occ_shares, subject, step, points);
    udf!(pro_tower_price, count_mean, dispersion, family, p1, p2, p3, limit, attachment, limits, attachments, shares, agg_deductibles, agg_limits, cover_limit, cover_retention, distortion, parameter, step, points);
    udf!(pro_agg_pml, count_mean, dispersion, family, p1, p2, p3, return_periods, step, points);
    udf!(pro_bayes_odp, triangle, output, draws, chains, seed, sims, percentiles, prior_sd, process, origins);
    udf!(pro_bayes_bf, triangle, exposure, apriori, cv, output, draws, chains, seed, sims, percentiles, prior_sd, process, origins, common_cv);
    udf!(pro_sev_fit, losses, family, output, t, truncation, thresholds, censored, weights, upper);
    udf!(pro_sev_fit_grouped, bounds, counts, family, method, output);
    udf!(pro_sev_fit_lev, sizes, lev, family, weights, output);
    udf!(pro_evt_thresholds, data, thresholds);
    udf!(pro_evt_fit, data, threshold, probabilities);
    udf!(pro_evt_hill, data, k);
    udf!(pro_payout_pattern, times, values, kind, shift, rate);
    udf!(pro_payout_stats, times, values, rate, kind, shift);
    udf!(pro_payout_price, expected_loss, assets, coc, times, values, rate, kind, shift);
    udf!(pro_quote, expected_loss, risk_premium, assets, expenses, brokerage, fixed, limit, quoted);
    udf!(pro_distortion, s, family, parameter, r0);
    udf!(pro_distortion_price, values, probs, family, parameter, r0);
    udf!(pro_distortion_calibrate, values, probs, premium, families, r0);
    udf!(pro_copula_cdf, points, family, parameter, nu);
    udf!(pro_copula_sim, family, parameter, sims, dim, seed, nu);
    udf!(pro_iman_conover, draws, correlation, seed, dof, shuffle);
    udf!(pro_swing, loss, basic, lcm, minimum, maximum, share);
    udf!(pro_profit_commission_term, losses, premium, share, allowance, carry_years);
    udf!(pro_sliding_scale_term, losses, premium, commissions, loss_ratios, carry, carry_years);
    udf!(pro_sliding_scale, loss, premium, commissions, loss_ratios);
    udf!(pro_profit_commission, loss, premium, share, allowance);
    udf!(pro_layer_terms, count_mean, dispersion, family, p1, p2, p3, layer_limit, layer_attachment, premium, reinstatements, commissions, loss_ratios, profit_share, profit_allowance, share, policy_limit, step, points);
    udf!(pro_glm, data, response, terms, family, link, exposure, weights, p);
    udf!(pro_glm_predict, data, response, terms, newdata, family, link, exposure, weights, p);
    udf!(pro_glm_relativities, data, response, terms, family, link, exposure, weights, p, level);
    udf!(pro_npv, rate, flows, times, unit);
    udf!(pro_irr, flows, times, unit, lower, upper, extend, detail);
    udf!(pro_moic, released, funded, opening);
    udf!(pro_cashflow, u, terms, patterns, requirement, commission, output, rate, commutation_requirement);
    udf!(pro_pnl, values, probs, premium, expenses, covers, face, output, levels, expense_at_mean);
    udf!(pro_ylt_stats, years, losses, n_years, probabilities, quantile, times, hours, negative);
    udf!(pro_ylt_oep, years, losses, return_periods, amounts, n_years, quantile, times, hours, negative);
    udf!(pro_elt_stats, rates, means, sd_independent, sd_correlated, exposures);
    udf!(pro_elt_oep, rates, means, sd_independent, sd_correlated, exposures, return_periods, amounts);
    udf!(pro_allocate, draws, family, parameter, method, r0);
    udf!(pro_allocate_total, draws, total, method, measure, level);
}
