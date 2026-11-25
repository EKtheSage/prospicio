---
type: Finding
title: The compartmental model's paid draws are cumulative, under incremental paid too
description: CompartmentalFit paid and CompartmentalPaths paid hold posterior predictive cumulative paid of each future cell (after each origin's latest paid), whether the model is fitted to cumulative or incremental paid, so a calendar-period operation on them (re-inflation) needs them differenced by origin first.
tags: [reserving, compartmental, inflation, draws]
status: draft
generated: { by: claude-code/local-session, at: 2026-11-24T12:00:00-08:00 }
sources:
  - id: results
    resource: ../crates/prospicio-reserving/src/bayes/compartmental/results.rs
    title: The doc of the paid draws
  - id: draws
    resource: ../crates/prospicio-reserving/src/inflation_draws.rs
    title: reinflate_draws, whose doc names the case
---

# Compartmental paid draws are cumulative

`CompartmentalFit::paid` (and the paths' `paid`) are the posterior
predictive cumulative paid of each future cell of the square, with
dimensions `origin` (a period) and `age` (months). The simulation adds
each increment onto the latest paid, so the values are cumulative even
when the model is fitted to incremental paid.

The keys have the shape that `Indexation::reinflate_draws` reads (an
`origin` period and an `age` in months), so the draws pass its checks,
but re-inflating cumulative amounts by their cell's payment period is
wrong. Difference them by origin first: the first future cell less the
origin's latest paid, then each cell less the one before.

Found while building PR 8 of #979 (`reserving-inflation.md` decision 20).
