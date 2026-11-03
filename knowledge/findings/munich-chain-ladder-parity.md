---
type: Finding
title: The Munich chain ladder against R ChainLadder and chainladder-python
description: prospicio's MunichChainLadder matches R ChainLadder 0.2.21 to 1e-12 relative with no tail, a log-linear tail and constant tails, and chainladder-python 0.10.1's MunichAdjustment under both sigma interpolations; R's default log-linear sigma silently becomes Mack's rule on MCL, q_inverse is exactly 1/q, the oldest rho differs between the libraries, and R's lambda is NA where the ratios carry no information.
tags: [reserving, munich, paid-incurred, parity, chainladder-r, chainladder-python]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-02T18:00:00-08:00 }
verified: { by: process:ci, at: 2026-11-02T18:00:00-08:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 29
  - id: test
    resource: ../validation/tests/reserving_munich.rs
    title: Validation test of the Munich chain ladder
  - id: rscript
    resource: ../validation/scripts/reserving_munich_r.R
    title: Script writing R ChainLadder's reference values
  - id: pyscript
    resource: ../validation/scripts/reserving_munich_python.py
    title: Script writing chainladder-python's reference values
---

# Facts

* R's `MunichChainLadder` defaults to `est.sigmaP = est.sigmaI =
  "log-linear"`, but on MCL its log-linear model of sigma has a p-value
  above 5%, so R warns and uses Mack's rule instead. prospicio's Mack does
  not fall back: R's default output equals prospicio with
  `SigmaInterpolation::Mack`, and prospicio's default (log-linear) equals
  chainladder-python's default.[^rscript][^test]
* With R's weights, `q = sum P / sum I` and `q_inverse = sum I / sum P`
  over the same origins, so `q_inverse = 1/q` exactly at every age. R and
  chainladder-python centre the paid residuals on `1/q` and the projection
  on `q_inverse`, which are therefore the same.[^design]
* The residual pairs are those of links whose sigma was estimated (two or
  more link ratios in both columns): 20 on MCL. R blanks its residual
  matrices' last two columns and chainladder-python masks exact zeros;
  both give the same 20 pairs on a square triangle. Within them R's two
  `lm` fits drop missing values separately: where incurred's sigma is 0 at
  a link (every ratio the same) its residuals there are 0/0 and leave the
  incurred regression only, so on MCL with incurred flat from 60 to 72
  months in 2001 and 2002 R keeps 20 paid and 18 incurred pairs and
  `lambda_paid` stays 0.636021.[^design][^test]
* The oldest age's rho (one origin) is extrapolated log-linearly by R
  whatever `est.sigma` says (0.1966 on MCL's paid), while
  chainladder-python fills it by the column's sigma interpolation (0.0311
  under Mack's rule). It enters only the tail step, so the ultimates agree
  without a tail.[^pyscript][^design]
* When incurred is a fixed multiple of paid every rho is 0 and R's lambda
  is `NA`; prospicio's lambda is 0 and its projections are the separate
  chain ladders exactly.[^test]
* chainladder-python's `MunichAdjustment` has no tail step; R's
  `tailP`/`tailI` (TRUE or a constant) are reproduced through each
  column's Mack tail.[^test]
