---
type: Environment Fact
title: Building and testing the Excel XLL without Excel
description: How the 64-bit XLL cross-builds from Linux with MinGW, what it imports, and how a fake Excel under Wine exercises the C API.
tags: [environment, excel, xll, windows, wine]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-15T22:30:00Z }
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
* XLOPER12 is 32 bytes on 64-bit Windows: a 24-byte value union (its
  largest member is `flow`) and the 4-byte `xltype`, padded.
