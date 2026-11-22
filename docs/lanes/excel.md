# Excel lane

**Session:** Cloud (the Excel session). **Owns:** `crates/prospicio-xll/`, `docs/design/xll.md`, the `xll` task in `xtask/src/main.rs`, `.github/workflows/xll.yml`

This lane wraps other lanes' crates and changes none of them. Excel is only a front end to Rust: a calculation the crates lack goes to the owning lane first (`xll.md` decision 65)

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/excel.md](done/excel.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next

- [excel-confirm-groups] in Excel 365: the four categories in Insert Function
- [excel-confirm-async] in Excel 365: `#GETTING_DATA` then the result for the bootstraps and Bayesian fits, and an edit during a long run (decision 48)
- [excel-more-functions] further functions as the user asks, each in its group
- [excel-multi-year-commission] parked, to be discussed with the user: the expected multi-year profit commission or sliding scale, deficits carried, from a layer's annual loss on the grid; needs a calculation in `prospicio-aggregate` first (decision 65)
- [code-signing-certificate] a code-signing certificate is the user's to buy
- [excel-help-defaults-test] #955 a test that compares each help text's stated defaults with the code: each function called with its required arguments alone and with each documented default given explicitly
- [excel-harness-concurrent-async] #955 the fake Excel with two asynchronous calls outstanding and a refused handle, so the join, cache-hit-burst and dropped-handle paths run outside the unit tests of `src/jobs.rs`
- [excel-close-during-run] #955 in Excel 365, with `[excel-confirm-async]`: quit during a long `PRO.RES.BAYES.BF`; the shutdown flag is in place (xlAutoClose), the crash was never confirmed
