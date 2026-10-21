---
type: Reference
title: Excess-layer loss development, the literature behind reserving-excess.md
description: Which papers on excess loss development exist, what their abstracts say, and what was and was not read, as checked on 2026-10-21 for the design note of issue 192; none is reproduced in prospicio yet.
resource: https://www.casact.org/abstract/analysis-excess-loss-development
tags: [reserving, excess-development, layers, literature]
status: draft
generated: { by: claude-code/local-session, at: 2026-10-21T12:00:00Z }
sources:
  - id: pinto-gogol
    resource: https://www.casact.org/abstract/analysis-excess-loss-development
    title: Pinto and Gogol, An Analysis of Excess Loss Development, PCAS 74 (1987), from p. 227
  - id: siewert
    resource: https://casact.org/abstract/model-reserving-workers-compensation-high-deductibles
    title: Siewert, A Model for Reserving Workers Compensation High Deductibles, CAS Forum Summer 1996, 217-244
  - id: sahasrabuddhe
    resource: https://www.casact.org/abstract/claims-development-layer-relationship-between-claims-development-patterns-trend-and-claim
    title: Sahasrabuddhe, Claims Development by Layer, CAS E-Forum Fall 2010
  - id: bloom-patrik
    resource: https://casact.org/abstract/methodology-avoiding-pitfalls-excess-loss-development
    title: A Methodology for Avoiding the Pitfalls of Excess Loss Development, CAS E-Forum Fall 2013 (authors Bloom and Patrik per a search result, not the abstract page)
  - id: verrall
    resource: https://ideas.repec.org/a/taf/uaajxx/v8y2004i3p67-89.html
    title: Verrall, A Bayesian GLM for the Bornhuetter-Ferguson method, NAAJ 8(3), 67-89
---

# What it is

The papers that `docs/design/reserving-excess.md` (issue 192) builds on.
Only abstracts and catalogue records were read on 2026-10-21; no formula in
the design note is quoted from them. The concept stays a draft until the
note's PR 1 reads the papers in full: the Fall 2013 paper's authors (Bloom
and Patrik) come from a search result, not its abstract page, and none of
the papers is yet known to print a worked example of a layer pattern
derived from a ground-up triangle.

* Pinto and Gogol (1987): Pareto distributions fitted to casualty losses by
  maturity, giving paid and reported development factors by layer, with the
  effects of allocated expenses and policy limits.
* Siewert (1996): development factors consistent across deductible levels,
  from a loss distribution model.
* Sahasrabuddhe (2010): development patterns, trend and claim size models
  (increased limit factors) kept consistent across layers.
* Bloom and Patrik (2013): excess triangles are thin and miss inflation's
  leverage, so excess development factors are derived from ground-up data
  through excess loss factors, with retentions that change by year.
* Mahler wrote no standalone excess development paper that a search found;
  his 1998 discussion of Gillam and Couret (PCAS 85, 316-344) is the item.
* Verrall (2004): the Bayesian BF whose normalised development pattern the
  user chose for the new Bayesian BF model (issue 410).

The `ChainLadder` R package and chainladder-python have no excess LDF
function, so there is no software parity target for the derived pattern.
