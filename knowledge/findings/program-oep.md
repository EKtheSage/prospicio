---
type: Finding
title: The occurrence PML of a program against aggregate 2.1.0
description: A program's ceded (or net) function of an occurrence is non-decreasing, so its occurrence PML is that function of the gross PML; where the function is flat the view has an atom, the survival is read at the flat stretch's right end and the achieved annual probability falls below the asked one; aggregate 2.1.0's oep on occurrence programs is matched on 216 values to 1e-9, atoms included.
tags: [oep, pml, reinsurance, aggregate, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-09T22:00:00Z }
verified: { by: process:ci, at: 2026-11-09T22:00:00Z }
sources:
  - id: script
    resource: ../validation/scripts/aggregate_program_oep.py
    title: validation/scripts/aggregate_program_oep.py (aggregate 2.1.0's utilities.oep on occurrence programs)
  - id: code
    resource: ../crates/prospicio-aggregate/src/pml.rs
    title: prospicio_aggregate::pml (program_occurrence_pml, program_occurrence_exceedance, OccurrenceView)
---

# Finding

* With shares at most 1 and no overlap within a stage, the program's ceded
  loss `g(x)` of an occurrence is non-decreasing in the gross loss, and so
  is the net `x - g(x)`. The year's largest ceded occurrence is then `g` of
  its largest gross one, and the ceded PML at `p` is `g` of the gross PML.
  Annual terms act on a year's losses and do not enter (`aggregate` says
  the same).[^code]
* Where the view is flat (the net across a layer placed in full; the
  ceded below the first attachment, across a gap or above the top) the
  view has an atom. The survival of the view at its PML is the gross
  survival at the right end of the flat stretch, so the achieved OEP,
  `1 - G_N(1 - S)`, falls below `p`: on a lognormal (mean 100, cv 2) with a
  Poisson(3) count, the net of `50% po 200 xs 100 and 500 xs 400` is 300 for
  gross losses from 400 to 900, and at `p = 0.05` the achieved OEP is 0.0266.
  Above the program's top the ceded loss is its total limit and the
  achieved OEP is 0.
* `aggregate`'s `oep` on `occurrence ceded to` and `net of` clauses, two
  severities, two programs and nine probabilities, 216 values (loss,
  survival, achieved OEP), is matched to 1e-9.[^script]

[^code]: crates/prospicio-aggregate/src/pml.rs.
[^script]: validation/scripts/aggregate_program_oep.py, run 2026-11-09.
