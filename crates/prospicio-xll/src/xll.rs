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
    udf!(pro_cl, triangle, average, tail, origins, selected, kind, weight);
    udf!(pro_cl_ldf, triangle, average, tail, selected, kind, weight);
    udf!(pro_cl_cdf, triangle, average, tail, selected, kind, weight);
    udf!(pro_cl_ultimate, triangle, average, tail, selected, kind, weight);
    udf!(pro_cl_reserve, triangle, average, tail, selected, kind, weight);
    udf!(pro_cl_factors, triangle, average, tail, selected, kind, weight, sigma);
    udf!(pro_mack, triangle, average, tail, origins, selected, kind, weight, sigma);
    udf!(pro_odp, triangle, sims, seed, percentiles, tail, process, origins);
    udf!(pro_odp_sims, triangle, sims, seed, tail, process, origins);
    udf!(pro_mack_boot, triangle, sims, seed, percentiles, average, tail, process, origins, sigma);
    udf!(pro_mack_boot_sims, triangle, sims, seed, average, tail, process, origins, sigma);
    udf!(pro_mack_cdr, triangle, average, tail, origins, selected, kind, weight, sigma);
    udf!(pro_mack_cdr_years, triangle, average, tail, origins, selected, kind, weight, sigma);
    udf!(pro_odp_cdr, triangle, sims, seed, percentiles, tail, process, origins);
    udf!(pro_mack_boot_cdr, triangle, sims, seed, percentiles, average, tail, process, origins, sigma);
    udf!(pro_el, triangle, exposure, apriori, average, tail, origins);
    udf!(pro_bf, triangle, exposure, apriori, average, tail, origins);
    udf!(pro_benktander, triangle, exposure, apriori, iterations, average, tail, origins);
    udf!(pro_capecod, triangle, exposure, trend, decay, average, tail, origins);
    udf!(pro_lev, limit, family, p1, p2, truncation);
    udf!(pro_layer, limit, attachment, family, p1, p2, truncation);
    udf!(pro_layer_cv, limit, attachment, family, p1, p2, truncation);
    udf!(pro_ilf, limit, basic_limit, family, p1, p2, truncation);
}
