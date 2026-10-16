---
type: Finding
title: Profit commission deficits carried to extinction telescope over a term
description: With M = premium (1 - allowance) and deficits carried to extinction, a term's profit commissions sum to share x (sum of (M - ceded_t) + D_T), D_T the deficit left at the end; it holds to 1e-9 on all 2,000 simulated paths of a four-year term, and a limit in years only ever lowers what is carried.
tags: [aggregate, reinsurance, contract-terms, multi-year]
status: stable
verified: { by: process:ci, at: 2026-10-16T09:00:00Z }
generated: { by: claude-code/cloud-session, at: 2026-10-16T09:00:00Z }
sources:
  - id: tests
    resource: ../crates/prospicio-aggregate/src/term.rs
    title: the deficits_to_extinction_telescope_on_every_path unit test
  - id: design
    resource: ../docs/design/aggregate.md
    title: Contract terms over several years
---

A profit commission with deficits carried forward pays, in year `t`,
`share × max(M − ceded_t − D_{t−1}, 0)` and carries `D_t = max(D_{t−1} +
ceded_t − M, 0)`, with `M = premium × (1 − allowance)`. Summing over a
term of `T` years, each year's result `M − ceded_t` either pays out or
moves the deficit, so `Σ PC_t = share × (Σ_t (M − ceded_t) + D_T)`: the
term pays its total result, plus back the deficit it still carries at the
end.[^design]

`Tower::apply_term` meets the identity to `1e-9` on every one of 2,000
simulated four-year paths (Poisson 3 claims a year, lognormal mean 10 and
CV 1.5, premium 30, allowance 25%, share 30%), and the commissions over
the term never exceed those paid year by year without the carry; the
carry lowers them on a good share of paths.[^tests]

The identity is a check that needs no outside reference: `aggregate` has
no multi-year term. It holds only to extinction; with a limit in years a
dropped deficit is gone, so the sum is larger than the identity's right
side by `share` times what was dropped.

[^tests]: the deficits_to_extinction_telescope_on_every_path unit test
[^design]: Contract terms over several years
