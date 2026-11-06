---
type: Environment Fact
title: Building and testing the Excel XLL without Excel
description: How the 64-bit XLL cross-builds from Linux with MinGW, what it imports, and how a fake Excel under Wine exercises the C API.
tags: [environment, excel, xll, windows, wine]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-05T12:00:00Z }
sources:
  - id: design
    resource: ../docs/design/xll.md
    title: xll.md, the XLL design note
  - id: harness
    resource: ../crates/prospicio-xll/harness/fake_excel.c
    title: The fake-Excel smoke test
---

# Facts

* The cloud container has no MinGW or Wine by default; `apt-get install
  gcc-mingw-w64-x86-64 wine64` installs both (Ubuntu 24.04, Wine 9.0;
  checked 2026-10-15), and `rustup target add x86_64-pc-windows-gnu` adds
  the Rust target. `cargo xtask xll` sets
  `CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc`
  itself.[^design]
* The GNU-target XLL imports only system DLLs (KERNEL32, msvcrt, ntdll,
  USERENV, WS2_32, bcryptprimitives, api-ms-win-core-synch): Rust's std
  links libgcc statically, so nothing from MinGW ships with it. Release
  and stripped it was about 3.6 MB at eight functions (2026-10-15).
* Sizes on the GNU target, stripped: 11.0 MB at 68 functions
  (2026-11-04), 9.7 MB of it with `lto = "fat"` and one codegen unit
  (`opt-level = "s"` on top saved nothing more). At 114 functions
  (2026-11-05) 16.2 MB; with their 98 old names (xll.md decision 47)
  16.3 MB, 24.6 MB unstripped, 14.0 MB with LTO and 5.8 MB of that
  gzipped. Of the 13.3 MB of code the wrappers are 5% (`prospicio_xll`,
  671 KB); `rustfft` is 16%, its f64 kernels three times over (scalar
  731 KB, SSE 631 KB, AVX 828 KB); generic `core` and `alloc` code 26%;
  `prospicio-reserving` 12%; the Bayesian stack with the GLMs and the
  thread pool (`nuts-rs`, `prospicio-bayes`, `-glm`, `-models`, `faer`,
  `rayon`) about 17%. So removing functions barely shrinks it unless a
  whole engine goes. Measured with `x86_64-w64-mingw32-nm -S
  --size-sort -C` on the unstripped DLL, grouped by the first crate path
  in each symbol. `cargo xtask xll` with LTO took 5 minutes on a cloud
  container, Wine smoke test included.
* xlfRegister's macro type (its sixth argument) 0 registers a function
  that calculates on a sheet but is not listed in the Insert Function
  dialog; 1 is a listed worksheet function, 2 a command. That is
  Microsoft's xlfRegister table as remembered; `learn.microsoft.com` was
  not reachable from the cloud session on 2026-11-05, and Excel 365 has
  not yet confirmed it, nor whether formula autocomplete also leaves a
  macro type 0 name out. The add-in registered its old names that way
  in #746 and dropped them before a release (xll.md decision 47).
* An XLL finds Excel through `GetProcAddress(GetModuleHandle(NULL),
  "MdCallBack12")`, so any host executable that exports `MdCallBack12`
  can load and drive it. `harness/fake_excel.c` does that: built with
  MinGW and run under Wine (`WINEDEBUG=-all wine fake_excel.exe
  prospicio.xll`), it checks every `xlfRegister` call and RAA's Mack
  totals through real XLOPER12 values.[^harness] Wine's first run prints a
  harmless `syswow64\rundll32.exe` warning while it creates `~/.wine`.
* Under Wine the release XLL runs `PRO.RES.ODP` on RAA with 20,000
  simulations in about 0.2 s (177 ms, checked 2026-10-16), so the
  bootstrap's 1,000,000 cap is about 10 s, the longest one recalculation
  can take.
* On GitHub's `ubuntu-24.04` runner the XLL workflow's build job (apt
  install of MinGW and Wine, the release build, the Wine smoke test, the
  workbook and the artifact) takes about two minutes (checked 2026-10-16,
  PR #284). Wine runs headless there with no display. `python3 -m pip
  install` puts packages in the user site on that runner, which
  `python3 -I` leaves out, so the workbook script runs without `-I`.
* A cloud session cannot download Actions logs or artifacts: both are
  served from `*.blob.core.windows.net`, which its egress proxy refuses.
  The job's steps and annotations come from the API
  (`actions/jobs/<id>`, `check-runs/<id>/annotations`).
* `harness/fake_excel.c` calls each export through a function pointer
  of a fixed arity (`Fn7`, `Fn10`, ...), so appending an argument to a
  worksheet function needs its call sites there widened too. A call with
  fewer arguments than the export takes is undefined behaviour under the
  x64 calling convention: the add-in reads stack garbage as the missing
  `XLOPER12 *` arguments. On 2026-10-29 the smoke test still passed with
  `pro_res_cl` and `pro_res_bf` called two and four arguments short, so a green
  smoke test does not prove the arities agree; `tests/catalogue.rs`
  checks `xll.rs` against `FUNCTIONS`, not the harness. Real Excel always
  passes every registered argument.[^harness]
* XLOPER12 is 32 bytes on 64-bit Windows: a 24-byte value union (its
  largest member is `flow`) and the 4-byte `xltype`, padded.
