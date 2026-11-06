---
type: Reference
title: Excel's asynchronous worksheet functions, as Excel-DNA implements them
description: An Excel 2010+ asynchronous UDF is registered with the type text `>`, its argument types and a last `X`; Excel passes the handle as an XLOPER12 of type xltypeBigData (xltypeStr | xltypeInt), and the XLL answers later, from any thread, with xlAsyncReturn (16 | xlSpecial = 16400) on that handle and the result; a dropped handle is refused with InvalidAsyncContext. Calculation events go through xlEventRegister (16401) with xleventCalculationEnded 1 and xleventCalculationCanceled 2. Macro type 0 registers a hidden function. Taken from Excel-DNA's source, Microsoft's pages being unreachable from the cloud session.
resource: https://github.com/Excel-DNA/ExcelDna
tags: [excel, xll, c-api, async, environment]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-05T12:00:00Z }
sources:
  - id: xlcall
    resource: https://github.com/Excel-DNA/ExcelDna/blob/master/Source/ExcelDna.Integration/XlCall.cs
    title: Excel-DNA, XlCall.cs (xlSpecial, xlAsyncReturn, xlEventRegister, the calculation events)
  - id: registration
    resource: https://github.com/Excel-DNA/ExcelDna/blob/master/Source/ExcelDna.Loader/XlRegistration.cs
    title: Excel-DNA, XlRegistration.cs (the type text of a native async function; macro type 0 for a hidden one)
  - id: types
    resource: https://github.com/Excel-DNA/ExcelDna/blob/master/Source/ExcelDna.Loader/XlTypes.cs
    title: Excel-DNA, XlTypes.cs (xltypeBigData for the async handle; "X")
  - id: handle
    resource: https://github.com/Excel-DNA/ExcelDna/blob/master/Source/ExcelDna.Integration/ExcelAsyncHandle.cs
    title: Excel-DNA, ExcelAsyncHandle.cs (xlAsyncReturn, InvalidAsyncContext treated as a warning)
  - id: design
    resource: ../docs/design/xll.md
    title: xll.md, decisions 47 and 48
---

# Facts

* The type text of a native asynchronous function is `>` (no result),
  the argument types, then `X`; Excel-DNA appends `$` only when the
  function is also marked thread safe, and never `&` (cluster safe) on an
  asynchronous one.[^registration] prospicio registers `>`, `Q` per
  argument and `X`, without `$`.[^design]
* The handle arrives as an XLOPER12 of type `xltypeBigData`
  (`xltypeStr | xltypeInt`, 0x0802).[^types] The XLL keeps a copy and
  later calls `xlAsyncReturn` (`16 | xlSpecial`, 16400) with the handle and
  the result.[^xlcall]
* Excel-DNA treats `InvalidAsyncContext` from `xlAsyncReturn` as a
  warning, not an error:[^handle] a handle Excel has dropped (an
  interrupted recalculation) is refused, not a crash. prospicio relies on
  that to register no calculation events.[^design]
* `xlEventRegister` is `17 | xlSpecial` (16401), with
  `xleventCalculationEnded` 1 and `xleventCalculationCanceled` 2; Excel-DNA
  passes the name of a DLL export (`CalculationCanceled`), not a
  registered function.[^xlcall]
* The sixth argument of `xlfRegister`, the macro type, is 2 for a command,
  0 for a hidden function and 1 for a listed one.[^registration]
* Not confirmed in Excel itself: these facts are Excel-DNA's reading of
  the C API, checked only against prospicio's fake Excel
  (`crates/prospicio-xll/harness/fake_excel.c`).[^design]

[^xlcall]: Excel-DNA, XlCall.cs.
[^registration]: Excel-DNA, XlRegistration.cs, GetRegisterParameters.
[^types]: Excel-DNA, XlTypes.cs.
[^handle]: Excel-DNA, ExcelAsyncHandle.cs.
[^design]: docs/design/xll.md, decisions 47 and 48.
