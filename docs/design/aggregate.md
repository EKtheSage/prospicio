# Design note: aggregate loss and reinsurance

Status: **In progress** · v0.3 · Depends on: `distributions.md` (`Grid`, `Counting`, `Severity`), `predictive-distribution.md` · Lane: Aggregate

## Goal

The distribution of a period's total loss `S = X_1 + … + X_N` from a claim
count `N` and a severity `X`, three ways, and the split of losses between
an insurer and its reinsurers. All numerics that are not aggregation itself
(distributions, grids, risk measures) come from `act-prob`.

| Method | Input | Output | Exact for | Use when |
|---|---|---|---|---|
| Panjer | `Counting` in the `(a, b, 0)` class, severity `Grid` | aggregate `Grid` | the discretized problem | moderate claim counts, `O(n · m)` |
| FFT | any count with a pgf, severity `Grid` | aggregate `Grid` | the discretized problem, up to aliasing | large grids or claim counts, `O(n log n)` |
| Monte Carlo | any `Counting`, any severity | `PredictiveDistribution` | nothing (sampling error) | per-event losses, reinsurance terms, dependence |

## What exists

- `act_aggregate::panjer(&frequency, &severity_grid, points)` returns the
  aggregate `Grid` and a `CompoundReport`.

## Decisions

- **Error is reported, never hidden.** A compound grid lumps its mass above
  the last point onto that point (as `Grid` discretization does) and
  `CompoundReport` records `tail_mass = P(S > (n - 1)h)`, the untruncated
  mean `E[N] E[X]` on the severity grid, the grid mean and `mean_error()`.
  The severity grid's own `DiscretizationReport` stays with the caller.
- **Panjer underflow is an error.** For a Poisson mean above about 700 with
  no mass at zero severity, `P(S = 0)` underflows to 0 and the recursion
  would return all zeros. `panjer` fails and points to FFT instead of
  scaling tricks.
- **The aggregate grid uses the severity grid's step.**

## Validation

`validation/tests/aggregate.rs` checks Panjer against a brute-force
compound sum, `Σ_n P(N = n) f^{*n}`, with numpy convolutions and SciPy
pmfs (`validation/scripts/compound_convolution.py`): 80 points for a
Poisson and a negative binomial at `1e-14`. FFT will be checked against the
same file. Unit tests check `E[S] = E[N] E[X]`, the compound variance
`E[N] Var[X] + Var[N] E[X]^2`, and `S = N` for a unit severity.

## Next

1. FFT aggregation (needs complex FFT: `rustfft`, pure Rust, WASM-safe), with
   an aliasing check from the padded grid.
2. Monte Carlo frequency-severity into a `PredictiveDistribution`, one RNG
   stream per simulation, keeping event-level losses for reinsurance.
3. Reinsurance: per-occurrence and aggregate layers, reinstatements and
   towers as data, applied to simulated events, giving gross, ceded and
   net `PredictiveDistribution`s.
