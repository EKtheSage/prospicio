---
type: Finding
title: PanjerClass forwards its third cumulant (it did not before October 2026)
description: impl Counting for PanjerClass forwarded the pmf, mean, variance and pgf but not third_moment or third_cumulant until branch claude/stoic-hawking-qc3qby, so the defaults ran and the cumulant from the raw moment cancelled at large means; compound_moments with PanjerClass::from_mean_dispersion(1000, 1) and a lognormal of mean 10, CV 0.5 gives a third cumulant of 1,953,124.76 against the exact 1,953,125 (1.2e-7), the skewness as far off; it now forwards every method and is exact.
tags: [counting, compound, moments, panjer, precision, excel]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-16T14:00:00Z }
verified: { by: process:ci, at: 2026-10-16T14:30:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-prob/src/counting.rs
    title: impl Counting for PanjerClass
  - id: reference
    resource: ../validation/reference/moments_aggregate.csv
    title: aggregate 1.0.1's compound moments on mpmath (book L)
---

# Facts

* `PanjerClass` (the count chosen by mean and dispersion) implements
  `Counting` by forwarding to the concrete family. Until branch
  claude/stoic-hawking-qc3qby it forwarded only `pmf`, `mean`,
  `variance`, `panjer_ab`, `pgf` and `pgf_complex`; `third_moment` and
  `third_cumulant` fell to the trait's defaults: the raw third moment
  summed from the pmf, and the cumulant converted from it,
  `E[N^3] - 3 m E[N^2] + 2 m^3`, which cancels when the mean is large.
* With 1,000 claims (dispersion 1, a Poisson) and a lognormal of mean 10
  and CV 0.5, `compound_moments` gave a third cumulant of 1,953,124.76;
  the exact value is `1000 x 10^3 x 1.25^3 = 1,953,125`. The skewness was
  off by the same 1.2e-7, against `aggregate`'s 1e-12 (book L of
  `moments_aggregate.csv`).
* It now forwards `third_moment`, `third_cumulant`, `cdf`, `quantile` and
  `sample` as well, so it agrees bit for bit with its family; the compound
  third cumulant above is exact. The families do not override `cdf`,
  `quantile` or `sample`, so forwarding them changes no draw.
* Any type that wraps a `Counting` must forward the two third-order
  methods: the defaults are correct but lose digits at large means.
  `CountDist`, `Box<T>` and the bindings' `AnyCount` already do.
* The Excel add-in's aggregate functions unwrap the `PanjerClass` into its
  concrete count, a workaround from before the fix; it is still exact and
  can now be dropped (the Excel lane's file).
