---
type: Reference Implementation
title: R ChainLadder CDR (one-year view reference)
description: R ChainLadder 0.2.21's CDR.MackChainLadder, the reference for Merz-Wuthrich's claims development result; its conventions, and where it and the 2008 paper differ.
resource: https://cran.r-project.org/package=ChainLadder
tags: [reserving, merz-wuthrich, cdr, one-year, parity, r]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-05T12:00:00+00:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 4
  - id: script
    resource: ../validation/scripts/reserving_cdr_r.R
    title: Generator of validation/reference/reserving_cdr_r.csv
  - id: paper
    resource: https://www.casact.org/sites/default/files/database/forum_08fforum_21merz_wuetrich.pdf
    title: Merz and Wuthrich (2008), Modelling the claims development result for solvency purposes, CAS E-Forum Fall 2008
---

# Use

`CDR(MackChainLadder(tri), dev = "all")` gives the standard error of the
claims development result per origin and in total for every future
calendar year, and Mack's standard error. Its internal `CL_MSEPs` is the
linear approximation of Merz and Wuthrich (2008), with the covariance
between two origins carried by the older origin's parameter term.[^script]

# Conventions

* It reports one calendar year per development age, so the last column,
  `CDR(n)S.E.`, is past the run-off and always zero. act-reserving reports
  one year per age-to-age factor; the parity test reads R's extra year as
  zero.[^design]
* Its `Mack.S.E.` column is the square root of the summed yearly MSEPs.
  It equals `MackChainLadder`'s `Mack.S.E` per origin and in total to
  rounding, on RAA, GenIns, ABC, MW2008 and MW2014.[^script]
* With `alpha != 1` it only warns ("formulae hold only for alpha=1") and
  returns numbers; with a tail it stops. act-reserving errors in both
  cases.[^design]
* It reads the latest diagonal positionally (row `i`'s latest at column
  `I - i + 1`), so it assumes a full trapezoid; act-reserving checks that
  shape and errors otherwise.

# The 2008 paper's Table 4

R with `est.sigma = "Mack"` (the paper's sigma rule (4.1)) reproduces the
paper's totals to the unit: reserves 2,237,826, one-year 81,080, Mack
108,401. The two oldest open origins do not: the paper prints 567 and
1,488 for the one-year view (Mack 567 and 1,566), where R and
act-reserving give 566.17 and 1,486.56 (Mack 566.17 and 1,563.81). Those
two depend most on the last, extrapolated sigma; a rounded sigma in the
paper is a likely cause (not verified). Unit tests therefore check the
paper's totals and youngest origin only.[^paper]

[^script]: Generator of validation/reference/reserving_cdr_r.csv
[^design]: Design note, reserving v0.2, decision 4
[^paper]: Merz and Wuthrich (2008), Modelling the claims development result for solvency purposes, CAS E-Forum Fall 2008
