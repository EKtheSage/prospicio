---
type: Reference Implementation
title: chainladder-python's predict, a development fitted on one book and applied to another
description: In chainladder-python 0.10.1, fit(source).predict(target) gives each origin of the target its latest value times the source's cdf_ at its age, the tail carried as fitted (a curve refitted on the target would differ), Bornhuetter-Ferguson and Benktander their apriori on the target's exposure and Cape Cod its apriori estimated again on the target; on a raw latest_diagonal Cape Cod's apriori reads every origin at the first factor to ultimate. It is the parity reference for prospicio_reserving's ChainLadderFit::to_chain_ladder with LatestDiagonal::from_triangle. Its NaN traps are in chainladder-python-predict-traps.md.
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
chainladder-python 0.10.1:[^script]

* on GenIns, the chain ladder with no tail, `TailConstant(1.05)`,
  `TailCurve()` and `TailCurve(curve='inverse_power')`, each applied to
  the source's own diagonal and to GenIns perturbed by origin; the source
  with no tail also applied to GenIns doubled, shifted ten years and cut to
  the origins from 2006 and to 2004;
* on the CAS workers' compensation triangle, Bornhuetter-Ferguson,
  Benktander and two Cape Cods (exposure the latest premium), each applied
  to the source's own diagonal, a perturbed book with 10% more premium and
  the book shifted ten years, and the Cape Cods also to the origins from
  1993 and to the raw `latest_diagonal`. There is no doubled book here.

The test reproduces every origin and total at 1e-9 relative in two arms:
prospicio's own source fit through `to_chain_ladder`, and the reference's
own `cdf_` given as a pattern of factors to ultimate. The GenIns targets
whose development ages are the source's (the diagonal, the perturbed,
doubled and shifted books and the origins to 2004) also go through the
plain triangle path, `to_chain_ladder().fit(...)`, against the same
rows.[^test]

# Behaviour

These are backed by the generator and the test:

* **Latest times the source's factors to ultimate.** The second arm, the
  source's `cdf_` with nothing estimated on the target, gives predict's
  ultimates on every predict target, so predict re-estimates no
  development on the target: each origin's latest value times the source's
  `cdf_` at its age. A book shifted ten years later gives the same
  ultimates; one cut to the origins from 2006 (at most 60 months old) or
  to 2004 gives those origins' ultimates.
* **The tail is carried, not refitted.** With `TailConstant(1.05)` or a
  `TailCurve` in the source's pipeline, the source's `cdf_` (the tail
  factor at the oldest age, 1.0294991710529173 for the exponential curve
  on GenIns and 1.2924303115436933 for the inverse power) gives predict's
  ultimates on the perturbed GenIns. The same pipelines fitted on the
  perturbed GenIns itself have tails of 1.0294941502734942 and
  1.2923593825173993 (rows `refitted_tail_perturbed`, which prospicio's own
  refit reproduces), each more than the tolerance from the source's, so
  this target tells a carried tail from a refitted one.
* **Expected-loss methods.** Bornhuetter-Ferguson and Benktander apply
  their apriori to the target's exposure (`sample_weight` in predict).
  Cape Cod estimates its apriori again on the target, with the source's
  pattern: 0.6404940442771272 on the workers' compensation diagonal and
  0.5805161561534887 on the perturbed book with 10% more premium, with no
  trend and no decay; on the origins from 1993 alone it is estimated on
  those origins only.
* **Cape Cod on a raw latest diagonal.** Given `paid.latest_diagonal` as it
  is, without `.val_to_dev()`, `CapeCod.predict`'s apriori is the one of
  every origin read at the first factor to ultimate: 2.0633214031618854 on
  every origin with no trend and no decay, against 0.6404940442771272 with
  `.val_to_dev()`. The rows of target `diagonal_raw` record it, for both
  Cape Cods, and the test reproduces it with every origin at 12 months.
  Only the apriori is recorded, not those ultimates. prospicio's
  `from_triangle` reads each origin at its own age.
* **The earlier-valued book.** For GenIns valued before 2008 the generator
  records not predict but `DevelopmentConstant` with the source's `cdf_`
  on that book, which prospicio's `LatestDiagonal` reproduces at each
  origin's own age.

Predict has no rows for two targets, GenIns valued before 2008 and GenIns
with its 2005 cell at 72 months removed, nor for ages between the
source's; [the traps](chainladder-python-predict-traps.md) say why.
prospicio projects the missing cell's origin from its last observed cell
(60 months) and the earlier book at each origin's own age, both checked by
hand in the test.

[^design]: Design note, reserving v0.2, decision 17, a fitted development applied to another book
[^script]: chainladder-python reference generator for a fitted development applied to another book
[^test]: Validation test of to_chain_ladder and LatestDiagonal::from_triangle against chainladder-python
