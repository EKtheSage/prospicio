---
type: Finding
title: A count proposal and a defensive tail mixture cut the standard error of a compound tail TVaR by 6 to 14 times, with a slight low shift at 0.995
description: "On a compound Poisson (mean 2) with a discrete severity p_k proportional to k^-3 on 1 to 200, 200,000 years drawn from a Poisson(4) count and a 30% tail mixture above 20 give the TVaR at 0.995 and 0.999 with batch standard errors of 0.137 and 0.185 against plain Monte Carlo's 0.777 and 2.60 (5.7 and 14 times smaller), effective sample size 48,000; over 20 seeds the standardised error has mean -0.04 at 0.999 but -0.5 at 0.995, about half a standard error low."
tags: [aggregate, importance-sampling, tail, tvar, monte-carlo, prospicio-aggregate, prospicio-prob]
status: draft
verified: { by: process:ci, at: 2026-10-29T12:00:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-29T12:00:00Z }
sources:
  - id: design
    resource: ../docs/design/aggregate.md
    title: aggregate.md, "Importance sampling of the tail"
  - id: test
    resource: ../validation/tests/aggregate.rs
    title: importance_sampling_matches_the_fft_tail_with_a_smaller_error
---

# Importance sampling of a compound tail

The proposal raises the claim count (Poisson 4 for 2) and draws 30% of the
losses from the severity above 20. The years' weights average 1 within
0.02 and their effective sample size is about a quarter of the years, yet
the tail is far better covered: the TVaR's batch standard error is 5.7
times smaller at 0.995 and 14 times at 0.999 than plain Monte Carlo's on
the same number of years.

Run over 20 seeds, the standardised error of the weighted TVaR at 0.999
has mean −0.04 and spread 1.20; at 0.995 the mean is −0.5 with spread
0.93, a small low shift (about 0.07 on 27.3) that plain Monte Carlo does
not show (mean +0.22). The self-normalised estimator is biased at order
`1/n`, which may account for it; it is well inside the parity test's four
standard errors.
