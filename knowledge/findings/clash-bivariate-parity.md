---
type: Finding
title: Clash and the bivariate compound FFT against aggregate
description: aggregate's clash model fixes the events hitting neither line by independence, n0 = na nb / nc, so a Poisson count gives Cov(A, B) = nc E[X] E[Y]; prospicio's 2-D FFT matches aggregate 1.0.1 at 1e-12 once aggregate's 1e-15 fuzz clip, which drops about 1e-11 of real tail mass and moves second moments by 3e-9, is turned off.
tags: [aggregate, clash, bivariate, fft, parity, catastrophe]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-13T22:00:00Z }
sources:
  - id: bivariate
    resource: ../crates/prospicio-aggregate/src/bivariate.rs
    title: prospicio_aggregate::bivariate (fft_bivariate, JointGrid, Clash, tests)
  - id: script
    resource: ../validation/scripts/aggregate_clash.py
    title: validation/scripts/aggregate_clash.py (targets from aggregate 1.0.1's clash)
  - id: aggregate
    resource: https://github.com/mynl/aggregate
    title: mynl/aggregate 1.0.1, aggregate/bivariate.py (solve_clash_model, BivariateAggregate.update_work, _clip_density_fuzz)
---

# Finding

* `aggregate`'s `clash NAME na nb nc claims <A> <B> [freq]` takes the
  expected events a year hitting line A only, B only and both. Each event
  hits A with probability `pa` and B with `pb`, independently, so the
  2 × 2 table of expected counts has independent rows and columns, which
  fixes the cell hitting neither: `n0 = na nb / nc`. Then
  `n = na + nb + nc + n0`, `pa = (na + nc) / n`, `pb = (nb + nc) / n`, and
  the shared count has mean `n` (Poisson unless a frequency clause is
  given).[^aggregate]
* Because `(na + nc)(nb + nc) = nc n`, a Poisson count gives
  `Cov(A, B) = Var(N) pa pb E[X] E[Y] = nc E[X] E[Y]`: with independent
  triggers the clash count alone carries the covariance. A mixed count
  adds `Var(N) − n` times `pa pb E[X] E[Y]`.[^bivariate]
* `aggregate` zeroes every joint cell below `1e-15` as FFT fuzz
  (`_clip_density_fuzz`). On 2^11 × 2^10 grids that removes most cells
  (1.9 million of 2.1 million) and about `1e-11` of real tail mass, which
  moves the variances and `E[AB]` by about `3e-9` relative. With the clip
  turned off its deficit is round-off (`3e-14`) and prospicio agrees at
  `4e-11` relative on the moments and `1e-12` on every probability.[^script]
* In prospicio, setting each buffer cell's negative round-off to 0 before
  summing biased each marginal by up to `1e-11` (the clamp adds about
  `1e-17` per cell over millions of cells); clamping only after the fold to
  the output grid keeps the marginals equal to the 1-D compound.
  Putting what the probabilities lack of 1 (about `1e-12`, from a
  severity grid summing to 1 within round-off) on the far corner moved the
  covariance by `4e-9`; it is left missing instead.[^bivariate]
* Aliasing on one axis does not move the other, so the mass wrapped
  around the 2-D buffer is the sum of each marginal's, which a 1-D
  compound on twice the buffer measures cheaply. A negative binomial
  shared count (variance 48 on a mean of 12) wrapped `1e-11` on a
  1,024-point buffer of step 0.25 where the Poisson wrapped nothing
  measurable.[^bivariate]

[^aggregate]: aggregate/bivariate.py, solve_clash_model, and parser.py, _clash_spec
[^bivariate]: crates/prospicio-aggregate/src/bivariate.rs, tests
[^script]: validation/scripts/aggregate_clash.py
