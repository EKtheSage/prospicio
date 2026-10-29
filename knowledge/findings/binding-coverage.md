---
type: Finding
title: On 2026-10-28 the bindings reached 76% (Python) and 71% (R) of the library crates' 789 public types and functions
description: "check_bindings.py (#572) counts 789 public module-level types and free functions in the ten library crates; Python names 603 of them, R 559 and the Excel add-in 159. The 416 unbound items seeded the allow-list: 79 are prospicio-math's numerical building blocks, which no binding depends on, and prospicio-models is the least covered crate after math (about half). Coverage resolves the binding crates' use trees and paths through the library's modules and re-exports, by crate, from source text."
tags: [bindings, python, r, excel, ci, coverage, tooling]
status: draft
generated: { by: claude-code/local-session, at: 2026-10-28T12:00:00Z }
sources:
  - id: script
    resource: ../validation/scripts/check_bindings.py
    title: The binding coverage check, its scan and its limits
  - id: allowlist
    resource: ../validation/bindings-allowlist.toml
    title: The allow-list seeded on 2026-10-28
---

# Coverage on 2026-10-28

The first run of `validation/scripts/check_bindings.py` on `main` (after
#699) gave, per crate, the public module-level types and free functions,
how many the Python and R binding crates name, and how many the Excel
add-in names (report only)[^script]:

| crate | items | Python | R | Excel |
|---|---|---|---|---|
| prospicio_aggregate | 121 | 99 (82%) | 85 (70%) | 15 (12%) |
| prospicio_bayes | 43 | 34 (79%) | 31 (72%) | 3 (7%) |
| prospicio_core | 7 | 7 (100%) | 7 (100%) | 7 (100%) |
| prospicio_glm | 40 | 34 (85%) | 29 (72%) | 1 (2%) |
| prospicio_math | 44 | 5 (11%) | 4 (9%) | 0 (0%) |
| prospicio_models | 77 | 39 (51%) | 37 (48%) | 0 (0%) |
| prospicio_nn | 11 | 10 (91%) | 9 (82%) | 0 (0%) |
| prospicio_pricing | 168 | 146 (87%) | 130 (77%) | 44 (26%) |
| prospicio_prob | 133 | 108 (81%) | 106 (80%) | 45 (34%) |
| prospicio_reserving | 145 | 121 (83%) | 121 (83%) | 44 (30%) |
| total | 789 | 603 (76%) | 559 (71%) | 159 (20%) |

The 186 Python and 230 R gaps seeded the allow-list[^allowlist]: 79
entries are `prospicio-math` (the bindings do not depend on it), 4 are
error types, 2 are traits, 2 are reserving's `Result` alias, 4 are the
layer terms `NthEvent` and `DropDown` (#591), which the bindings reach
through `Layer`'s methods, and 325 are marked "(seed); review" for
their lanes to bind or give a reason. R trails Python most in
aggregate (36 against 22) and pricing (38 against 22).

A first version of the scan matched names, not paths, and counted 607
and 566: a name in one crate (`Basis`, `Diagnostics`, `Distribution`,
`Result`, `Term`, `Allocation`) bound another crate's item of that name,
and a module or method segment (`stack`, `new`, `fit`) bound a function
of that name. Eleven such Python and R gaps were hidden that way, among
them R's `pnl::stack`, which Python binds.

# What the numbers mean

- Methods are not counted: a type's binding stands in for its methods, so
  a bound type with half its methods unbound still counts as bound.
- An item is bound when a `use` tree or a `prospicio_*::` path in
  `crates/prospicio-python/src` or `crates/prospicio-r/src` resolves to
  it, through the library's modules and re-exports, by crate and
  declaring module. A path stops at its first item, so `Type::new` binds
  the type only. A trait imported only to call its methods counts as
  bound; an item reached through a glob import of the binding crate is
  missed.
- The whole scan, ten crates and three binding crates from source text,
  takes about 3 seconds on the local Windows machine, mostly Python start
  and file reads; no nightly toolchain or rustdoc JSON is needed.
