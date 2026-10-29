---
type: Finding
title: On 2026-10-28 the bindings reached 77% (Python) and 72% (R) of the library crates' 787 public types and functions
description: "check_bindings.py (#572) counts 787 public module-level types and free functions in the ten library crates; Python names 607 of them, R 566 and the Excel add-in 168. The 401 unbound items seeded the allow-list: 79 are prospicio-math's numerical building blocks, which no binding depends on, and prospicio-models is the least covered crate after math (about half). Coverage is a name match on the binding crates' use trees and paths, so it is an approximation in both directions."
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
#696) gave, per crate, the public module-level types and free functions,
how many the Python and R binding crates name, and how many the Excel
add-in names (report only)[^script]:

| crate | items | Python | R | Excel |
|---|---|---|---|---|
| prospicio_aggregate | 119 | 100 (84%) | 87 (73%) | 15 (13%) |
| prospicio_bayes | 43 | 34 (79%) | 31 (72%) | 3 (7%) |
| prospicio_core | 7 | 7 (100%) | 7 (100%) | 7 (100%) |
| prospicio_glm | 40 | 35 (88%) | 30 (75%) | 2 (5%) |
| prospicio_math | 44 | 5 (11%) | 4 (9%) | 0 (0%) |
| prospicio_models | 77 | 39 (51%) | 38 (49%) | 1 (1%) |
| prospicio_nn | 11 | 10 (91%) | 9 (82%) | 0 (0%) |
| prospicio_pricing | 168 | 147 (88%) | 131 (78%) | 49 (29%) |
| prospicio_prob | 133 | 108 (81%) | 107 (80%) | 45 (34%) |
| prospicio_reserving | 145 | 122 (84%) | 122 (84%) | 46 (32%) |
| total | 787 | 607 (77%) | 566 (72%) | 168 (21%) |

The 180 Python and 221 R gaps seeded the allow-list[^allowlist]: 79
entries are `prospicio-math` (the bindings do not depend on it), 4 are
error types, 2 are traits and 316 are marked "(seed); review" for their
lanes to bind or give a reason. R trails Python most in aggregate (32
against 19) and pricing (37 against 21).

# What the numbers mean

- Methods are not counted: a type's binding stands in for its methods, so
  a bound type with half its methods unbound still counts as bound.
- The match is by name over `use` trees and `prospicio_*::` paths in
  `crates/prospicio-python/src` and `crates/prospicio-r/src`. Two crates'
  items of one name are bound together, and a trait imported only to call
  its methods counts as bound; an item reached through a glob import is
  missed.
- The whole scan, ten crates and three binding crates from source text,
  takes about 3 seconds on the local Windows machine, mostly Python start
  and file reads; no nightly toolchain or rustdoc JSON is needed.
