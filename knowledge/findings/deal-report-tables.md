---
type: Finding
title: A deal's best simulations fall outside the default IRR search, and the report templates need papermill for Jupyter parameters
description: "DealBook.cashflows() and irr() search for the IRR between -95% and 100% by default, so a collateralised quota share whose best years return more than 100% a year reports those simulations as having no IRR; on the sample submission's 20% quota share 116 of 2,000 flat-commission simulations (loss ratios 0.27 to 0.37, MOIC 1.72 to 2.30) had no root, and upper=5.0 finds all of them (the largest is 237%). Quarto's -P on a document with the Jupyter engine needs papermill; the knitr engine does not. The sample submission's development is not monotone (incurred falls in 3 triangle cells and on 255 listing rows) and five cost-index dates are a day off a quarter start, so neither is a check a submission must pass."
tags: [guides, submission, deal-report, irr, quarto, prospicio-pricing]
status: draft
generated: { by: claude-code/local-session, at: 2026-10-28T12:00:00Z }
sources:
  - id: guide
    resource: ../docs/guides/submission-schema.md
    title: the submission schema and deal report guide
  - id: templates
    resource: ../examples/templates/README.md
    title: the deal report templates
  - id: notebook
    resource: ../examples/ils_deal_analysis.ipynb
    title: the ILS deal analysis notebook
---

# The deal report's tables and the IRR's search range

Found while writing `docs/guides/submission-schema.md` and the report
templates in `examples/templates/` (#605).

* `irr()` and `DealBook.cashflows()` search between `lower = -0.95` and
  `upper = 1` by default and return NaN, with `found` false, when no sign
  change lies in that range. A flow of -13.1m at month 0 followed by
  positive months has one sign change, but its root is above 100% a year,
  so it is reported as no root.
* On the guide's 20% quota share (ODP bootstrap loss ratios, a 15% level
  cv, collateral on 30% of unearned premium and 125% of outstanding loss),
  116 of 2,000 flat-commission simulations and 42 of the sliding scale's
  had no root under the default range. They were the best years (loss
  ratios 0.27 to 0.37, MOIC 1.72 to 2.30), so leaving them out of the
  IRR's statistics biases them down. `cashflows(upper=5.0)` finds every
  root; the largest is 237%.
* `examples/ils_deal_analysis.ipynb` calls `cashflows()` with the default
  range and reports a "no root" count beside its IRR percentiles, so its
  IRR statistics leave out the deal's best years in the same way.
* Since #670 an IRR result says when a root lies outside the range
  (`root_above`, `root_below`), so such a simulation is told from flows
  with no root at all, and `extend=True` (R `extend = TRUE`) searches past
  the range to find it; the default search is unchanged.
* Quarto renders a document's `params` with `-P name:value`. With the knitr
  engine they are `params$name`; with the Jupyter engine they replace the
  cell tagged `parameters`, and Quarto needs `papermill` in the Python
  environment for that (without it the render fails with "The papermill
  package is required for processing --execute-params").
* A Quarto document runs one Jupyter kernel, so a template with both
  Python and R chunks needs knitr with `reticulate`, which no CI job
  installs. The templates are therefore twins, one per language, reading
  the same tidy tables.
* The sample submission is not monotone: incurred falls in three triangle
  cells (2016 at 120 months, 2017 and 2018 at 96) and paid in one, and on
  255 rows of the large-loss listing incurred falls from one valuation to
  the next. Five dates of `cost_index.csv` are a day off a quarter's first
  day, from the generator's truncation of a year fraction. A submission
  check that required either would reject the sample, and real data.
