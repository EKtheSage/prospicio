---
type: Finding
title: The one-year view with a bootstrap tail, against Merz-Wuthrich and Mack's tail risk
description: Valuing the bootstraps' tail step one development period after the oldest age puts it in the coming year for an annual origin at the oldest age, which it closes. With a constant tail 1.05, Mack's bootstrap then gives every open origin 1.05 times R's CDR(1)S.E., the oldest origin R's Mack.S.E with the tail (C^2 se^2 + tail_sigma^2 C) and mean zero, and the total their combination, on RAA, GenIns and ABC within 2.2 Monte Carlo standard errors; with R's refitted log-linear tail the oldest origin's process error is R's tail sigma and its one-year SD its lifetime one. R ChainLadder 0.2.21's CDR rejects any tail, so these targets are built from R outputs under the chosen semantics, not from a one-year reference.
tags: [reserving, bootstrap, mack, odp, tail, one-year, cdr, merz-wuthrich]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-11T12:00:00-07:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 11, the one-year view with a bootstrap tail
  - id: test
    resource: ../validation/tests/reserving_one_year_tail.rs
    title: Validation test of Mack's one-year view with a constant and a refitted tail
  - id: cdr
    resource: ../validation/reference/reserving_cdr_r.csv
    title: R ChainLadder 0.2.21 CDR(MackChainLadder(tri)) one-year standard errors
  - id: tails
    resource: ../validation/reference/reserving_tails_r.csv
    title: R ChainLadder 0.2.21 MackChainLadder with tail = TRUE and tail = 1.05
  - id: unit
    resource: ../crates/prospicio-reserving/src/one_year_bootstrap.rs
    title: Unit tests of the one-year view with a tail (and mack_bootstrap.rs)
  - id: dependence
    resource: ../validation/tests/reserving_dependence.rs
    title: Validation test of one-year tails with dependence between segments (one_year_tails_with_dependence)
---

# Finding

No implementation computes a one-year view with a tail. R ChainLadder
0.2.21's `CDR(MackChainLadder(RAA, tail = 1.05))` stops with "Sorry, tail
factors are currently not supported." (re-checked 2026-10-11, R 4.5.3), and
chainladder-python has no CDR. Mack (1999) gives the tail a sigma and a
standard error but no calendar time. Decision 11 gives it one: the step
from the oldest age to ultimate is valued one development period after the
oldest age, so with an annual grain it falls in the coming year for the
origin already at the oldest age, which it closes at its simulated
ultimate.[^design]

Under that choice the one-year view with a constant tail `T` has closed
forms built from R outputs already committed, which `MackBootstrap::one_year`
reproduces (20,000 simulations, seed 20,261,011, Gamma process, centred
residuals, both rules for the last sigma):[^test]

* **Open origins** have opening and closing ultimates both `T` times those
  without a tail (for a tail attached at the oldest age), so their CDR is
  `T` times the untailed CDR in distribution (exactly, per simulation,
  only when the tail factor is fixed, which the unit tests check; a drawn
  factor shifts the random stream), and their standard deviation `T`
  times Merz and Wüthrich's `CDR(1)S.E.` A tail attached before the
  oldest age replaces estimated factors inside the triangle, which move
  with the pseudo factors, and does not scale this way.
* **The oldest origin** has CDR `U0 - C*`, `C*` drawn from its observed
  value `C` with a lognormal factor of mean `T` and Mack's standard error
  `se` and process variance `tail_sigma^2 C`. Its variance,
  `C^2 se^2 + tail_sigma^2 C`, is exactly R's `Mack.S.E` of that origin
  under `MackChainLadder(tail = 1.05)` (RAA 686.9; simulated 679.5, -2.17
  standard errors), and its mean is zero.
* **The total** is `sqrt(T^2 total^2 + var_oldest)`, the oldest origin's
  CDR being independent of the others (RAA 26,518 against 26,434, +0.29
  standard errors; GenIns 1,875,659 against 1,864,797, +1.15; ABC 122,595
  against 123,066, -0.76).

Every origin and total of the three triangles under both rules is within
2.2 Monte Carlo standard errors of its target, and every mean CDR within
0.014 standard deviations of zero.

With R's log-linear rule refitted on each simulation's pseudo factors, the
oldest origin's tail-step process error (the Gamma run less the run
without process error on one seed) has R's `tail_sigma sqrt(C)` (RAA
124.9 against 124.3, GenIns 51,968 against 51,663, ABC 1,086 against
1,082), and its one-year standard deviation is its lifetime one (RAA 236
against 204 on another seed, +1.18 combined standard errors; GenIns
+0.48; ABC +0.60). Its mean is not zero: the refit is convex in the
factors ([bootstrap tail](/findings/bootstrap-tail-parameter-error.md)),
and on the ODP's pseudo factors it inherits the 78% high mean of RAA's
oldest origin.

Consequences of the semantics, not of the data:

* An annual origin at the oldest age carries its whole tail uncertainty in
  the coming year: its one-year SD is its lifetime SD. That is
  conservative if the true tail runs off over several years.
* The step is the whole tail whatever its shape. A curve (`TailCurve`,
  100 extrapolated periods by default) or a decayed constant selects one
  factor per period for a year past the oldest age and a remainder, yet
  the step takes their product within one development period (three
  months on a quarterly grain). For such tails the one-year SD of an
  origin that takes the step is an upper bound, not the tail's own
  calendar pattern.
* A constant tail's parameter error enters the one-year view only through
  the closed origins: the open origins keep the method's constant at the
  end of the year.
* With quarterly development, origins that reach the oldest age within
  nine months take the step too; annual origins on a December valuation
  reach it only at the year's end, so only the oldest takes it.

Synchronized segments share residual positions only, so an estimated tail
refitted on two identical lines without process error keeps their CDRs
equal; a constant tail's lognormal draw is each line's own (correlation
0.92 for the ODP, 0.86 for Mack on two copies of CLRD's wkcomp with
`tail_std_err` 0.02, 2,000 simulations, the CDR totals); rank correlation
gives Spearman 0.494 against 0.5.[^dependence]

[^design]: `docs/design/reserving-v02.md`, decision 11.
[^test]: `validation/tests/reserving_one_year_tail.rs` against
    `validation/reference/reserving_cdr_r.csv` and
    `validation/reference/reserving_tails_r.csv`. The quoted standard
    errors are printed by the test with `--nocapture`.
[^dependence]: `validation/tests/reserving_dependence.rs`,
    `one_year_tails_with_dependence`, checks the equal copies, a constant
    tail's draws differing by line with correlation in (0.1, 1), and the
    rank-correlated Spearman.
