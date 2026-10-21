---
type: Reference
title: Excess-layer loss development, the literature behind reserving-excess.md
description: What Pinto and Gogol, Levine, Bear, Siewert, Sahasrabuddhe, Bloom and Patrik and Korn print on excess loss development, read in full on 2026-10-21, with the exact examples (Bloom and Patrik pp. 4-8 is a pure scale shift and reproduces to the digit) and the sources that find shape drift; no source prints a capped-data solve.
resource: https://www.casact.org/abstract/analysis-excess-loss-development
tags: [reserving, excess-development, layers, literature]
status: draft
generated: { by: claude-code/local-session, at: 2026-10-21T18:00:00Z }
sources:
  - id: pinto-gogol
    resource: https://www.casact.org/abstract/analysis-excess-loss-development
    title: Pinto and Gogol, An Analysis of Excess Loss Development, PCAS 74 (1987), 227-255 (Levine's discussion 256-271)
  - id: bear
    resource: https://www.casact.org/abstract/analysis-excess-loss-development
    title: Bear, Discussion of Pinto and Gogol, PCAS 79 (1992), 134-148
  - id: siewert
    resource: https://casact.org/abstract/model-reserving-workers-compensation-high-deductibles
    title: Siewert, A Model for Reserving Workers Compensation High Deductibles, CAS Forum Summer 1996, 217-244
  - id: sahasrabuddhe
    resource: https://www.casact.org/abstract/claims-development-layer-relationship-between-claims-development-patterns-trend-and-claim
    title: Sahasrabuddhe, Claims Development by Layer, CAS E-Forum Fall 2010 (revised 2013, with errata)
  - id: bloom-patrik
    resource: https://casact.org/abstract/methodology-avoiding-pitfalls-excess-loss-development
    title: Bloom and Patrik, A Methodology for Avoiding the Pitfalls of Excess Loss Development, CAS E-Forum Fall 2013
  - id: korn
    resource: https://www.casact.org/abstract/analysis-excess-loss-development
    title: Korn, Strategies for Modeling Loss Development, Variance 11(1), 95-117 (2017), section 6
  - id: verrall
    resource: https://ideas.repec.org/a/taf/uaajxx/v8y2004i3p67-89.html
    title: Verrall, A Bayesian GLM for the Bornhuetter-Ferguson method, NAAJ 8(3), 67-89
  - id: code
    resource: ../validation/tests/reserving_excess.rs
    title: The parity tests of the examples below
---

# What it is

The papers that `docs/design/reserving-excess.md` (issue 192) builds on, read
in full on 2026-10-21 for the note's gate: Pinto and Gogol with Levine's and
Bear's discussions, Siewert (Appendix I skimmed), Sahasrabuddhe (equation
images did not extract, so formulas are read off the exhibit columns), Bloom
and Patrik (Appendix 2 skimmed) and Korn's section 6. Boor and Mahler were
skimmed (claim-level development with a random ratio per claim; no layer
factors); Gillam and Couret were not found; Clark 2003 and Verrall 2004 are
not relevant to the gate. The R `ChainLadder` package and chainladder-python
have no excess LDF function. Where a paper's number is reproduced by a CI test
it is checked in [the parity finding](/findings/excess-ldf-scale-shift-parity.md).

# Methods

* **Pinto and Gogol (1987)** fit excess factors by retention as `a x^b` to ISO
  data. Their section 4 is a **shape change**: a single-parameter Pareto above
  k with shape `q_i` at report `i`, `LDF(i to j, xs ck) = (n_j / n_i)
  ((q_i - 1) / (q_j - 1)) c^(q_i - q_j)`; their p. 247 example is 2.5 x 4^0.5 =
  5.0 and Appendix B (pp. 253-254) is 10^0.3 = 1.996 (exactly 1.99526; both
  reproduced by `ParetoShape` in PR 2, see
  [the shape-drift finding](/findings/excess-shape-drift-parity.md)). Section
  5 (p. 248) gives the layer identity `LDF(c-d) = (f(c) - f(d)) / (f(c) /
  CDF_c - f(d) / CDF_d)`, `f` the ultimate excess over the ultimate ground-up
  loss. Paid layer factors (pp. 241-245) come from the incurred ones times a
  ratio of paid-to-reported ratios, which hardly varies by retention.
* **Levine** tests the fit and restates the identity. **Bear** (1992)
  generalizes to bounded layers with equal limit-to-attachment ratios,
  `f' = d c^(q_i - q_j)`; the p. 138 example is 3.5 x 2^0.15 = 3.88 (3.8835,
  reproduced in PR 2).
* **Siewert (1996)** splits development into counts times severity with
  limited severity relativities: `XSLDF = LDF (1 - R_(t+1)) / (1 - R_t)`, and
  the unlimited factor is the relativity-weighted average of the limited and
  excess factors. With counts this is the note's formula. Its AY 1993 example
  prints 1.1684 and the rounded relativities give 1.1669. Table 6 fits Weibulls
  at 48 months and at ultimate (scale and shape both move; 3.2338 at 750k,
  2.7539 at 500k), approximate because the parameters are rounded.
* **Sahasrabuddhe (2010)** rebases a limited factor by limited means at the
  claim size model's scale by age (exponential here, trended by cost level
  through indices): `F_X(i,j) = F_B(10,j) [LEV_X(theta_i,inf) /
  LEV_B(theta_10,inf)] / [LEV_X(theta_i,j) / LEV_B(theta_10,j)]`. Example 1
  (pp. 15-18) uses GenIns limited to 1m, a basic limit of 500k and a policy
  limit of 2m, `theta_10,j` from 28,138 to 280,000; E4 at row 10, j = 1 is
  12.291 x [LEV_280000(1m) / LEV_280000(500k)] / [LEV_28138(1m) /
  LEV_28138(500k)] = 14.352. Example 3, column 9 is 0.856 (an ad hoc decay in
  the selected ratios). Errata: the subscripts of equation 3.10 and Example 3's
  column references.
* **Bloom and Patrik (2013)** restate Pinto and Gogol with ELFs: `Excess CDF =
  ELF / (1 / CDF_unlimited - (1 - ELF) / CDF_limited)`; section 4 prints 17.54.
  Their section 7 values are approximate (8.0835 recomputed against 8.093).
* **Korn (2017)**, section 6, is the note's model: `LDF = CCDF x SDF`, the
  scale factor backed out of a capped LEV (equations 6.3 to 6.9), counts moved
  with survival functions for retentions (6.11); the shape could vary
  instead. No numbers are printed.

# The worked example (Bloom and Patrik, pp. 4-8)

Four claims of 100, 375, 250 and 500 at 12 months for Year One (1,225 in
total), ground-up factors 12:24 3.60 and 24:36 1.25 (ultimate 5,513, exactly
5,512.5), counts fixed, 5% inflation (Year Two 5,788, Year Three 6,078) and a
deductible of 350. A pure scale shift. The excess of 350 triangle prints
175, 3,010 and 4,113 (Year One), 219, 3,231 and 4,388 (Year Two), 265, 3,462
and 4,678 (Year Three), with factors 17.20, 1.37 and 23.50, then 14.77, 1.36
and 20.06, then 13.08, 1.35 and 17.67. Limited to 350: 1,050, 1,400 and 1,400,
factors 1.33 and 1.00. Applying the weighted factors to Year Three gives an
ultimate of 5,732, 22.5% over its true 4,678: why the severity is trended and
the pattern is not (decision 5). The ELFs are .746, .758 and .764, and the
excess CDF of Year Three is 17.54 (-0.77% error).

# What the sources say that the note had not

* **Shape drift is found in data.** Pinto and Gogol, Bear, Siewert (the
  Weibull shape) and Korn all find the shape moving with maturity, so a pure
  scale shift may understate a high layer's development. PR 2's simulation
  confirms it: under shape drift the scale shift understates high layers by
  14% to 29% ([the study](/findings/excess-simulation-study.md)).
* **Paid is derived from incurred.** Pinto and Gogol and Bear get paid layer
  factors from incurred ones and ratios of paid-to-reported ratios, and
  Sahasrabuddhe says his method covers paid or reported losses, so decision 9
  keeps paid out of PR 1 only as a later option.
* **The solve for the shifts is trivial when uncapped** (`c_j` is the
  ground-up factor over the count factor) and matters only for a capped
  ground-up column, where Korn notes the factor exceeds the severity
  development factor. No source prints a numeric example of that solve.
* **The note's lognormal illustration** (1.275, 1.545, 1.761, 1.982) takes the
  mean as 50,000 at the earlier age and 65,000 at ultimate, not 50,000 at
  ultimate.
