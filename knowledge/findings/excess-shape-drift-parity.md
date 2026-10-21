---
type: Finding
title: Pinto and Gogol's and Bear's shape-drift factors are one ratio of Pareto layer expectations, and ParetoShape reproduces their examples
description: With a base development D (counts above the threshold, or a base layer's losses) and Pareto shapes by maturity, the layer factor is D times the base's per-claim expectation ratio over the layer's; it gives Pinto and Gogol's 2.5 (4)^0.5 = 5.0 and 10^0.3 = 1.99526 (printed 1.996 from rounded amounts) and Bear's 3.5 (2.0)^0.15 = 3.8835 (printed 3.88).
tags: [reserving, excess-development, shape-drift, pareto, parity, pinto-gogol, bear]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-21T22:00:00Z }
verified: { by: process:ci, at: 2026-10-21T22:00:00Z }
sources:
  - id: code
    resource: ../validation/tests/reserving_excess.rs
    title: The parity tests of ParetoShape against Pinto and Gogol and Bear
  - id: module
    resource: ../crates/prospicio-reserving/src/excess.rs
    title: SeverityByMaturity::ParetoShape and ShapeBase
  - id: pinto-gogol
    resource: https://www.casact.org/abstract/analysis-excess-loss-development
    title: Pinto and Gogol, PCAS 74 (1987), pp. 245-247 and Appendix B, pp. 253-254
  - id: bear
    resource: https://www.casact.org/abstract/analysis-excess-loss-development
    title: Bear, Discussion of Pinto and Gogol, PCAS 79 (1992), pp. 137-138
---

# What was found

Above a threshold `k` the claims follow a single-parameter Pareto with
shape `q_j` at report `j`. With `g_q(L)` the expected loss to layer `L` of
one claim from `Pareto(k, q)` (1 for a count) and `D_j` the base's
development to ultimate, the layer `A` develops by

    CDF_A(j) = D_j [g_q_j(B) / g_q_ult(B)] [g_q_ult(A) / g_q_j(A)]

with `B` the base (the counts above `k`, the losses excess of `k`, or any
layer at or above `k`). For `A` excess of `c k` and `B` excess of `k` the
bracket is `c^(q_j - q_ult)`, Pinto and Gogol's term; for layers with the
same limit-to-attachment ratio it is Bear's `c^(q_i - q_j)` with `d` the
lower layer's development; with counts as the base it is Pinto and
Gogol's full `(n_j / n_i) ((q_i - 1) / (q_j - 1)) c^(q_i - q_j)`.

| Target (source) | Printed | Computed |
|---|---|---|
| Pinto and Gogol p. 247: k = 25,000, retention 100,000, q 1.75 to 1.25, d = 2.5 | 5.0 | 5.0 (1e-12), from the losses excess of k or from the counts above k |
| Appendix B: excess of 100,000 over excess of 10,000, q 1.6 (27 months) to 1.3 (ultimate) | 1.996 a | 10^0.3 = 1.99526 a |
| Appendix B: excess of 100k, 300k, 500k, 1m and the layers 100-300k, 300-500k, 500k-1m | amounts to three decimals | each factor within the rounding of the printed ratio |
| Bear p. 138: layer 200,000 to 600,000 from 100,000 to 300,000, q 1.25 to 1.10, d = 3.5 | 3.88 | 3.8835 (1e-12) |

The printed 1.996 is .501 / .251, a ratio of amounts rounded to three
decimals; the exact factor is 1.99526. With an unbounded base in Bear's
example the factor differs by more than 0.05, so his equal-ratio condition
is needed for his closed form, not for the general ratio above.

Not pinned: Siewert's Table 6, whose Weibull fits move scale and shape
together (not a Pareto shape drift, and its parameters are rounded), and
Sahasrabuddhe's Example 3, which needs the example's trend indices.
