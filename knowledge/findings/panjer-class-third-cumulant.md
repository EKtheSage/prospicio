---
type: Finding
title: PanjerClass does not forward its third cumulant
description: impl Counting for PanjerClass forwards the pmf, mean, variance and pgf but not third_moment or third_cumulant, so the defaults run, and the cumulant from the raw moment cancels at large means; compound_moments with PanjerClass::from_mean_dispersion(1000, 1) and a lognormal of mean 10, CV 0.5 gives a third cumulant of 1,953,124.76 against the exact 1,953,125 (1.2e-7), the skewness as far off; the concrete Poisson gives it exactly.
tags: [counting, compound, moments, panjer, precision, excel]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-16T14:00:00Z }
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
  `Counting` by forwarding `pmf`, `mean`, `variance`, `panjer_ab`, `pgf`
  and `pgf_complex` to the concrete family. `third_moment` and
  `third_cumulant` are not forwarded, so the trait's defaults run: the raw
  third moment summed from the pmf, and the cumulant converted from it,
  `E[N³] - 3 m E[N²] + 2 m³`, which cancels when the mean is large.
* With 1,000 claims (dispersion 1, a Poisson) and a lognormal of mean 10
  and CV 0.5, `compound_moments` gives a third cumulant of 1,953,124.76;
  the exact value is `1000 × 10³ × 1.25³ = 1,953,125`. The skewness is
  off by the same 1.2e-7, against `aggregate`'s 1e-12 (book L of
  `moments_aggregate.csv`). The concrete `Poisson` gives it exactly.
* The Excel add-in's aggregate functions unwrap the `PanjerClass` into
  its concrete count, so they are exact; any other caller of
  `PanjerClass` with a large mean loses the digits until the impl
  forwards both methods (the Probability lane's crate).
