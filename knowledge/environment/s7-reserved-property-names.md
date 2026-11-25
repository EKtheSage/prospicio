---
type: Environment Fact
title: S7 refuses some property names
description: S7 0.2.2's new_class() refuses a property named names, class or dim, so an R binding's S7 class carries such a field under another name or as the names of another property; the APC fit's canonical coefficient names are the names of its coefficients vector.
tags: [environment, r, bindings, s7]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-24T23:00:00Z }
stale_after: 2027-05-24T00:00:00Z
sources:
  - id: check
    resource: process:local session claude/apc-glm-bindings, 2026-11-24
    title: S7::new_class("t", properties = list(<name> = S7::class_any)) for names, class, dim, levels and x under S7 0.2.2 and R 4.5.3
---

# Facts (2026-11-24)

* S7 0.2.2 (R 4.5.3) stops `new_class()` with "property can't be named:
  names" for a property called `names`, and the same for `class` and `dim`.
  `levels` and ordinary names are accepted.[^check]
* An R binding's S7 class therefore cannot mirror a Rust field or Python
  attribute called `names`. `apc_glm_fit` (#979, PR 7) gives the canonical
  coefficient names as the names of its `coefficients` vector, as
  `odp_glm_fit` does, where Python's `ApcGlmFit` has `names`.[^check]

[^check]: The check above, run while binding the APC GLM in R.
