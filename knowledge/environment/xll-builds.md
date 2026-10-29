---
type: Environment Fact
title: Building and testing the Excel XLL without Excel
description: How the 64-bit XLL cross-builds from Linux with MinGW, what it imports, and how a fake Excel under Wine exercises the C API.
tags: [environment, excel, xll, windows, wine]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-29T12:00:00Z }
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
  and stripped it is about 3.6 MB.
* An XLL finds Excel through `GetProcAddress(GetModuleHandle(NULL),
  "MdCallBack12")`, so any host executable that exports `MdCallBack12`
  can load and drive it. `harness/fake_excel.c` does that: built with
  MinGW and run under Wine (`WINEDEBUG=-all wine fake_excel.exe
  prospicio.xll`), it checks every `xlfRegister` call and RAA's Mack
  totals through real XLOPER12 values.[^harness] Wine's first run prints a
  harmless `syswow64\rundll32.exe` warning while it creates `~/.wine`.
* Under Wine the release XLL runs `PRO.ODP` on RAA with 20,000
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
  `pro_cl` and `pro_bf` called two and four arguments short, so a green
  smoke test does not prove the arities agree; `tests/catalogue.rs`
  checks `xll.rs` against `FUNCTIONS`, not the harness. Real Excel always
  passes every registered argument.[^harness]
* XLOPER12 is 32 bytes on 64-bit Windows: a 24-byte value union (its
  largest member is `flow`) and the 4-byte `xltype`, padded.
