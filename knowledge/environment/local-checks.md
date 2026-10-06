---
type: Environment Fact
title: Local checks in the cloud container and their noise
description: What runs locally before a push, what cannot, and the generated-file churn cargo xtask r leaves to revert.
tags: [environment, ci, bindings, r, python]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-06T18:30:00Z }
stale_after: 2027-01-05T00:00:00Z
sources:
  - id: agents
    resource: ../AGENTS.md
    title: AGENTS.md, Checks
---

# What runs

* `cargo fmt --all --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test`, and clippy on `act-python` and `act-r`.[^agents]
* `cargo xtask python`: builds the extension and runs pytest; its final
  docs render fails because Quarto is not installed (CI's `docs` job
  covers it).
* `cargo xtask r`: roxygen, install, every `R/actuarialrs/tests/*.R`; the
  pkgdown step fails on the network (see
  [cloud network](/environment/cloud-network.md)). The "boom" errors in
  its output are expected test errors.

# Noise to revert after `cargo xtask r`

The container's roxygen2 and R differ from the committed output. Undo
only these:

* the `RoxygenNote: 7.3.1` line it adds to `R/actuarialrs/DESCRIPTION`
  (keep a real `Collate` or `Suggests` change);
* the link targets it rewrites in `man/gamma_distribution.Rd`,
  `man/lognormal.Rd` and `man/weibull_distribution.Rd`: this R resolves
  `[stats::dgamma()]` to `\link[stats:GammaDist]`, `dlnorm` to
  `stats:Lognormal` and `dweibull` to `stats:Weibull`, where the committed
  files (and CI) have `stats:dgamma`, `stats:dlnorm`, `stats:dweibull`.
  When one of these files changes for real (a new argument), keep that
  change and restore only the link; reverting the whole file loses it;
* `NAMESPACE`'s multi-line `importFrom(stats, aggregate, coef, predict,
  quantile)`, which it splits into four lines.

Keep the other regenerated `man/` and `NAMESPACE` changes: they come from
the code change.

# Also

* `R/actuarialrs/R/extendr-wrappers.R` is edited by hand for a new or
  changed extendr function; match the existing line format.
* The freMTPL2 parity runs only in release builds
  (`cargo test --release -p act-validation --test fremtpl2`).

[^agents]: AGENTS.md, Checks
