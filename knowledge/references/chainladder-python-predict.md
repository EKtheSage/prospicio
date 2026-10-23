---
type: Reference Implementation
title: chainladder-python's predict, a development fitted on one book and applied to another
description: In chainladder-python 0.10.1, fit(source).predict(target) gives each origin of the target its latest value times the source's cdf_ at its age, the tail carried as fitted, Bornhuetter-Ferguson and Benktander their apriori on the target's exposure and Cape Cod its apriori estimated again on the target. It is the parity reference for prospicio_reserving's ChainLadderFit::to_chain_ladder with LatestDiagonal::from_triangle. A hand probe found three traps; a missing latest cell and an earlier valuation give NaN, and Cape Cod on a latest_diagonal aligns every origin to the first factor to ultimate unless val_to_dev() is called first.
resource: https://github.com/casact/chainladder-python
tags: [reserving, development, chain-ladder, expected-loss, cape-cod, latest-diagonal, parity, python]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-23T12:00:00-07:00 }
verified: { by: process:ci, at: 2026-10-23T12:00:00-07:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 17, a fitted development applied to another book
  - id: script
    resource: ../validation/scripts/reserving_fit_apply_python.py
    title: chainladder-python reference generator for a fitted development applied to another book
  - id: test
    resource: ../validation/tests/reserving_fit_apply.rs
    title: Validation test of to_chain_ladder and LatestDiagonal::from_triangle against chainladder-python
---

# Use

`Chainladder().fit(source).predict(target)`, and the same on a
`Pipeline` with a tail or on an expected-loss estimator with
`sample_weight`, applies the development fitted on one triangle to
another triangle or to a `latest_diagonal`. It is the reference for
`ChainLadderFit::to_chain_ladder` applied through `fit_diagonal` to
`LatestDiagonal::from_triangle(target, ...)` (decision 17 of
`reserving-v02.md`).[^design] The generator records it with
chainladder-python 0.10.1 on GenIns (the chain ladder with no tail,
`TailConstant(1.05)`, `TailCurve()` and `TailCurve(curve='inverse_power')`)
and on the CAS workers' compensation triangle (Bornhuetter-Ferguson,
Benktander and two Cape Cods, exposure the latest premium), each applied
to the source's own diagonal and to books made from it: perturbed by
origin, doubled, shifted ten years, cut to some of its origins and, on
GenIns, valued before 2008.[^script] The test reproduces every origin and
total at 1e-9 relative in two arms: prospicio's own source fit through
`to_chain_ladder`, and the reference's own `cdf_` given as a pattern of
factors to ultimate.[^test]

# Behaviour

These are backed by the generator and the test:

* **Latest times the source's factors to ultimate.** The second arm, the
  source's `cdf_` with nothing estimated on the target, gives predict's
  ultimates on every target, so predict re-estimates no development on the
  target: each origin's latest value times the source's `cdf_` at its age.
  A book shifted ten years later gives the same ultimates; one cut to the
  origins from 2006 (at most 60 months old) or to 2004 gives those
  origins' ultimates.
* **The tail is carried, not refitted.** With `TailConstant(1.05)` or a
  `TailCurve` in the source's pipeline, the source's `cdf_` (the tail
  factor at the oldest age, 1.0294991710529173 for the exponential curve
  on GenIns and 1.2924303115436933 for the inverse power) gives predict's
  ultimates on a perturbed GenIns, whose own curve would differ.
* **Expected-loss methods.** Bornhuetter-Ferguson and Benktander apply
  their apriori to the target's exposure (`sample_weight` in predict).
  Cape Cod estimates its apriori again on the target, with the source's
  pattern: 0.6404940442771272 on the workers' compensation diagonal and
  0.5805161561534887 on the perturbed book with 10% more premium, with no
  trend and no decay; on the origins from 1993 alone it is estimated on
  those origins only.
* **The earlier-valued book.** The generator does not use predict for
  GenIns valued before 2008 (see below) but `DevelopmentConstant` with the
  source's `cdf_` on that book, which prospicio's `LatestDiagonal` reads at
  each origin's own age.

A hand probe of chainladder-python 0.10.1 on 2026-10-23, not kept in a
script, found three traps, which the generator steps around and prospicio
does not share:

* **A missing latest cell.** With GenIns' 2005 cell at 72 months removed,
  predict gives that origin NaN. prospicio projects it from its last
  observed cell (60 months), checked by hand in the test.
* **An earlier valuation.** On GenIns valued before 2008, predict values
  the target at the source's valuation and gives NaN. prospicio projects
  each origin at its own age.
* **Cape Cod on a latest diagonal.** `CapeCod.predict(latest_diagonal)`
  aligns every origin to the first factor to ultimate (on the workers'
  compensation diagonal its apriori was 2.063321) unless the diagonal is
  given with `.val_to_dev()`, which the generator does. prospicio reads
  each origin's own age.

The same probe found that GenIns perturbed by origin, fitted on its own,
has an exponential curve tail of 1.0294941502734942, not the source's, so
the perturbed target does tell a carried tail from a refitted one.

chainladder-python cannot read an age between the source's (18 months
against an annual pattern) or a quarterly target, so prospicio's
log-linear reading of them on the diagonal has no reference and is checked
by hand.

[^design]: Design note, reserving v0.2, decision 17, a fitted development applied to another book
[^script]: chainladder-python reference generator for a fitted development applied to another book
[^test]: Validation test of to_chain_ladder and LatestDiagonal::from_triangle against chainladder-python
