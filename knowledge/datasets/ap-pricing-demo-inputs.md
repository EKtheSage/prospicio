---
type: Dataset
title: The AP pricing demo's inputs
description: The inputs.json of the AP pricing demo (the requester's pricing-functions project, examples/data), copied verbatim; 14 accident years 2012-2025 of earned premium and loss ratios, a calendar-year trend schedule 2012-2027, a development pattern and layer and Vasicek settings, all illustrative round numbers; the running loss ratio's four printed values are checked on it.
resource: ../validation/data/ap_pricing_demo_inputs.json
tags: [pricing, ratemaking, loss-ratio, trend, synthetic, excel, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-14T12:00:00Z }
verified: { by: process:ci, at: 2026-11-14T12:00:00Z }
sources:
  - id: file
    resource: ../validation/data/ap_pricing_demo_inputs.json
    title: validation/data/ap_pricing_demo_inputs.json, sha256 5047fb2477924fb28d0decfdcfe819ae2812fb54d3226b8f5b28b6a4de51b6d7
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/806
    title: "#806, the demo's printed cumulative loss ratios"
  - id: tests
    resource: ../validation/tests/pricing_ap_cumulative_lr.rs
    title: validation/tests/pricing_ap_cumulative_lr.rs and its Python and R twins
---

# Facts

* The user supplied the file on 2026-11-14 from the demo's
  `pricing-functions/examples/data/inputs.json`; it is committed unchanged
  so its hash identifies the demo's version.
* `earned_premium` (2.4M in 2012 rising to 3.996M in 2025) and
  `loss_ratios` (0.55 to 0.74) are one value per accident year 2012 to
  2025; the running premium-weighted loss ratio on them is 0.61,
  0.594705882353, 0.638120662040, ..., 0.637034303676 in row 14, the
  values #806 quotes from the demo, reproduced to 5e-13.[^tests]
* `trend_years` and `trend_rates` (2012 to 2027, 3.5% to 7%), the RAD
  treaty incepting 2026-01-01 and the `dev_pattern` (0.35 to 1.0, padded
  to `dev_target_length` 10) are the demo's trend and `REVERSE_TRANSPOSE`
  inputs; #801's and #806's own vectors already test those functions, so
  this file is not read for them.
* The `layer` and `vasicek` blocks configure the demo's simulations and
  are not used by any prospicio test.

[^tests]: validation/tests/pricing_ap_cumulative_lr.rs, run 2026-11-14.
