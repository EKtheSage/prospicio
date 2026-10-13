---
type: Dataset
title: Synthetic reinsurance submission, commercial property per risk
description: A seeded, generated submission for a 2026 per-risk excess of loss program (premium and rate changes, 168 listed large claims, a loss triangle, a risk profile, treaty terms, and an answer key of the true parameters); what examples/submission_evaluation.ipynb estimates from it and how close it gets.
resource: ../examples/data/submission
tags: [reinsurance, pricing, synthetic, large-losses, experience-rating, exposure-rating, notebook]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-13T12:00:00Z }
sources:
  - id: generator
    resource: ../examples/data/make_submission.py
    title: make_submission.py, the seeded generator that holds the true parameters
  - id: key
    resource: ../examples/data/answer_key.json
    title: answer_key.json, the true parameters and layer losses, outside the submission's folder
  - id: notebook
    resource: ../examples/submission_evaluation.ipynb
    title: submission_evaluation.ipynb, the evaluation that reads the files
  - id: tests
    resource: ../python/tests/test_examples.py
    title: test_submission_evaluation, test_submission_files_match_generator
  - id: map
    resource: ../knowledge/findings/program-evaluation-api-map.md
    title: What the API covers, the steps the notebook follows
---

# Facts

* Eight CSV files in `examples/data/submission/` and the answer key in
  `examples/data/answer_key.json`, outside the submission's folder so that
  nothing reading the submission sees it; both written by
  `examples/data/make_submission.py` (seed 20261013, numpy only), with LF line
  endings on every platform. The files are committed (about 100 kB);
  `test_submission_files_match_generator` fails if they drift from the
  generator: the CSV files byte for byte, the answer key's full-precision
  floats to a relative 1e-9.[^tests]
* The world: accident years 2016 to 2025 valued at 31 December 2025; exposure
  grows 3% a year and the cedant's rate changes (nine, from 2017) come on top;
  claims inflate at 4.5% a year; losses above 1m at 2026 prices are a gamma
  mixed Poisson with 10 a year at 2026 exposure (mixing cv 0.25), each a
  Pareto with alpha 1.6 cut at the risk's sum insured; the sum insured of a
  claim is drawn from the portfolio's bands in proportion to premium times
  (mid / 1m) to the 1.25; everything else is attritional at 40% of on-level
  premium, with its noise drawn once per accident year so that the
  triangle's paid never exceeds its incurred. Incurred falls with age in three
  cells, all from listed claims whose case reserves were revised
  down.[^generator]
* Output sizes: 246 claims generated, 168 listed (incurred over 500,000 at the
  latest valuation), 842 listing rows, 100 claims over 750,000 once developed,
  1 of them at its policy limit. The three layers (5m xs 5m, 10m xs 10m,
  20m xs 20m) have true expected losses of 1.37m, 0.68m and 0.28m a year under
  occurrence terms.[^key]
* What the notebook recovers (local run, `PROSPICIO_EXAMPLE_SIMS` at its
  default): on-level premium for 2025, the exposure growth and the cost trend
  to 0.1%; the Pareto alpha 1.720 (standard error about 0.17) against 1.6
  with truncation and censoring, and 1.24 when the per-claim reporting
  thresholds are ignored; 17.7 claims a year over 750,000 against 15.6 for
  claims on risks insured for more; a Pearson dispersion of 0.80 against a
  true 1.97; layer losses from the credibility blend 1.39m, 0.58m and 0.23m
  (errors +1%, -14%, -17%), from the experience model -21%, -33%, -39%, from
  the burning cost 1.19m, 0.49m and 0 (the top layer has no claim in ten
  years). The technical premiums from the true layer losses are within 1% to
  8% of the notebook's, and their rates on line fall inside the 5% to 95%
  range of its 200 parameter draws.[^notebook]
* `test_submission_evaluation` runs the notebook on every PR (4,000
  simulations, 20 draws) and asserts the looser forms of these: every
  reconciliation check passes; on-level premium within 0.5%, exposure growth
  within 0.2 points and cost trend within 0.5 points of the truth; alpha
  within two standard errors of 1.6 and at least 0.2 above the fit that
  ignores the thresholds; the blended layer losses within 10% of the truth
  for the working layer and 25% above it; each technical premium within 10%
  of the one from the true layer losses; and the verdicts quote, counter and
  refer.[^tests]

# Consequences

* The experience of ten years is nearly silent about the layers: with the
  limited-fluctuation standard of 20% at 90%, the credibility of the
  experience is 0.16, 0.07 and 0.03 for the three layers, so the exposure
  curve (Swiss Re c = 3 here) decides the price, and moving c from 2 to 4
  moves the technical rate on line of the working layer from 47.1% to 35.1%,
  while the tail alpha moving 0.25 either way moves it by under two points.
  A count dispersion of 3 moves it as much as the curve (47.5%), and ten
  yearly counts cannot settle the dispersion.
* The broker's quotes in `treaty_terms.csv` are hand-set (43%, 15% and 6%
  rate on line), not derived from the model, so that the decision differs by
  layer: quote the working layer, counter the middle one, and refer the top
  one to the portfolio view (short of the return on stand-alone capital, but
  29% above the premium under one price of risk).
* One seed is one realisation: the errors above are the size of error to
  expect, not evidence of a method's bias.

[^generator]: make_submission.py, the TRUE dictionary
[^key]: answer_key.json
[^tests]: test_examples.py
[^notebook]: submission_evaluation.ipynb, run locally on 2026-10-13
