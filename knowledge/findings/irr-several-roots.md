---
type: Finding
title: An IRR on simulated cash flows can have several roots or none, so the search reports which
description: "Flows that change sign more than once can have several internal rates of return: -100, +230, -132 at 0, 1 and 2 years are worth 0 at both 10% and 20%, since -100 + 230 v - 132 v^2 = 0 at v = 1/1.1 and 1/1.2. Flows of one sign have none. A grid search over -0.95 to 1 keeps the root nearest 0 and counts both (sign_changes = 2); a search from 15% finds 20%. The day count matters too: -100 now and 103 three months later return 1.03^4 - 1 = 12.55% a year in twelfths and 1.03^(365/90) - 1 = 12.74% on actual/365 day offsets."
tags: [prospicio-pricing, irr, cash-flows, time-value]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-21T12:00:00Z }
verified: { by: process:ci, at: 2026-10-21T12:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-pricing/src/returns.rs
    title: prospicio_pricing::returns::irr, irr_rows and their unit tests
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/480
    title: "Issue #480: return metrics on simulated cash flows"
---

# Facts

* By Descartes' rule of signs, flows with one change of sign have at most one IRR; each further change can add one. Reserve releases after capital calls give a second change, so simulated cash flows do have several roots.
* `irr` evaluates the flows on a grid of rates, takes each sign change or exact zero as a bracket, keeps the bracket whose midpoint is nearest 0, bisects it and refines with Newton steps clipped to it (the reference model's defaults: 400 points from -0.95 to 1, 8 bisections, 3 Newton steps).
* `sign_changes` counts the brackets: 0 with `found = false` and NaN, more than 1 when the reported rate is one of several.
* A root outside the searched range is not found: -100 now and 400 in a year return 300%, outside [-0.95, 1].
* Month offsets as twelfths of a year and day offsets on actual/365 give different annual rates on the same flows (`TimeUnit`).
