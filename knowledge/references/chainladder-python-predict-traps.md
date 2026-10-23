---
type: Reference Implementation
title: Traps of chainladder-python's predict on another book
description: In chainladder-python 0.10.1, fit(source).predict(target) gives NaN for an origin whose latest cell is missing and for every origin of a book valued earlier than the source, and CapeCod.predict on a raw latest_diagonal reads every origin's apriori at the first factor to ultimate unless val_to_dev() is called first. The reference generator for prospicio_reserving's to_chain_ladder asserts each and raises if chainladder-python stops behaving so; no CI test runs those assertions.
resource: https://github.com/casact/chainladder-python
tags: [reserving, development, chain-ladder, cape-cod, latest-diagonal, parity, python, pitfalls]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-23T12:00:00-07:00 }
sources:
  - id: script
    resource: ../validation/scripts/reserving_fit_apply_python.py
    title: chainladder-python reference generator for a fitted development applied to another book
  - id: predict
    resource: chainladder-python-predict.md
    title: chainladder-python's predict, a development fitted on one book and applied to another
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 17, item 8, where prospicio departs from chainladder-python
---

# Traps

chainladder-python 0.10.1's `fit(source).predict(target)` is the
reference for a fitted development applied to another book,[^predict] but
three of its results are not what an actuary wants. The reference
generator asserts each of them when it runs and raises if
chainladder-python stops behaving so;[^script] the generator is run by
hand, not in CI, so these facts are not marked verified. prospicio departs
from all three (decision 17, item 8).[^design]

* **A missing latest cell.** With GenIns' 2005 cell at 72 months removed,
  predict gives NaN for that origin alone.
* **An earlier valuation.** On GenIns valued before 2008, predict gives
  NaN for every origin; the hand probe that found it read this as predict
  valuing the target at the source's valuation. The generator records
  `DevelopmentConstant` with the source's `cdf_` for that book instead.
* **Cape Cod on a raw latest diagonal.** `CapeCod.predict(latest_diagonal)`
  reads every origin's apriori at the first factor to ultimate unless the
  diagonal is given with `.val_to_dev()`. The generator asserts that the
  two aprioris differ; the value itself (2.0633214031618854 on the
  workers' compensation diagonal with no trend and no decay) is a parity
  row that CI checks, in the predict note.

A hand probe on 2026-10-23, not asserted by the generator, found that
predict cannot read an age between the source's (18 months against an
annual pattern) or a quarterly target, so prospicio's log-linear reading
of them on the diagonal has no reference and is checked by hand.

[^predict]: chainladder-python's predict, a development fitted on one book and applied to another
[^script]: chainladder-python reference generator for a fitted development applied to another book
[^design]: Design note, reserving v0.2, decision 17, item 8, where prospicio departs from chainladder-python
