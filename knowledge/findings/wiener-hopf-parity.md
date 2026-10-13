---
type: Finding
title: Wiener-Hopf ruin for renewal claims against aggregate
description: The cepstral Wiener-Hopf ruin matches aggregate 1.0.1 at 3e-12 once the severity is put on the grid as aggregate does it; Grid::rounding piles the tail past the grid on the last point, which moves a lognormal's mean by 1e-9 and psi by 2e-9.
tags: [aggregate, ruin, renewal, wiener-hopf, fft, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-14T00:10:00Z }
sources:
  - id: ruin
    resource: ../crates/prospicio-aggregate/src/ruin.rs
    title: prospicio_aggregate::ruin (wiener_hopf, RenewalRuin, ruin_cepstral, unwrap)
  - id: script
    resource: ../validation/scripts/aggregate_wiener_hopf.py
    title: validation/scripts/aggregate_wiener_hopf.py (targets from aggregate 1.0.1's wiener_hopf)
  - id: aggregate
    resource: https://github.com/mynl/aggregate
    title: mynl/aggregate 1.0.1, aggregate/_aggregate.py (Aggregate.wiener_hopf, _discretize_wait_pmf) and aggregate/_renewal.py (ruin_cepstral)
---

# Finding

`aggregate` 1.0.1's `wiener_hopf` finds eventual ruin for renewal claims
as the all-time maximum of the walk with steps `X - cW`, by the cepstrum
of `1 - φ_Y` with the zero at `z = 1` divided out [aggregate]. prospicio's
port matches it on four cases (light and heavy severities, exponential,
gamma, mixed and very irregular waits, ρ = 0.1 and 0.3) with ψ within
3e-12 and the capital on the grid exact [ruin] [script].

- **The severity's discretization decides the last digits.** `aggregate`
  puts the severity on the grid by rounding from survival differences and
  rescales to 1, dropping the mass past the grid; `Grid::rounding` instead
  puts that mass on the last point. For a lognormal (mean 10, cv 1) on
  `2^16` points of 1/32 the two means differ by 1e-9 and ψ by up to
  2.3e-9. The parity test builds the severity `aggregate`'s way; for a use
  where the grid covers the tail the two agree.
- **The phase unwrap must be numpy's.** Each jump of at least `π`
  between neighbouring phases is brought back by `2π`, with a jump of
  exactly `-π` read as `+π` when the raw difference is positive.
- **Exponential waits** give Pollaczeck–Khinchine within the unit test's bound (5e-3
  at a step of 1/16 against a mean of 10), and `ψ(0)` near `1 / (1 + ρ)`.
- **Heavy tails need room.** With the lognormal at ρ = 0.1, ψ at the top
  of a grid of 2048 is 3.7e-6, above `aggregate`'s warning level of 1e-6;
  the values still agree, but more points are needed for the far tail.
