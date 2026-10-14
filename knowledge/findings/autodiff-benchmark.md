---
type: Finding
title: Automatic differentiation for small Bayesian reserving models, benchmarked
description: On Meyers' CRC log posterior (29 parameters, 55 cells) every autodiff option matched the hand-written gradient to about 1e-15; in the repository the in-house dual type with 32 tangents a pass costs 2.3 to 2.5 times the hand-written gradient (num-dual cost 10 to 90 times, candle about 100, Burn about 190), matches every hand-written gradient to 1e-12, and needs 3 passes for the 90 parameters of the compartmental model 2; Burn 0.21's sigmoid rounds to f32 on an f64 backend.
tags: [bayes, autodiff, performance, burn, candle, num-dual, ode, dual-numbers]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-13T22:00:00-07:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: reserving-bayes.md, decision 15
  - id: compartmental
    resource: ../docs/design/reserving-compartmental.md
    title: reserving-compartmental.md, decision 4 (PR A)
  - id: code
    resource: ../crates/prospicio-reserving/src/bayes/autodiff.rs
    title: The oracle tests and the benchmark (autodiff_benchmark, ignored)
  - id: numdual
    resource: https://crates.io/crates/num-dual
    title: num-dual on crates.io
  - id: burn
    resource: https://crates.io/crates/burn
    title: Burn on crates.io
---

# Automatic differentiation for small Bayesian reserving models

## The evaluation that chose the dual type

A throwaway benchmark (outside the repository, 2026-10-13; rustc 1.98.1,
Windows MSVC, Intel Core Ultra 9 285H) implemented the Meyers CRC log
posterior on the sampler's scale, with priors, from group 353's paid column.
Not run by CI; it informed decision 15 of the design note.[^design]

* **Correctness.** Every method matched the hand-written gradient to about
  1e-15 relative at six random points, once Burn's `sigmoid` was written
  out: Burn 0.21's `activation::sigmoid` converts to f32 internally even on
  an f64 backend, which gave a gradient error of 1.4e-7.[^burn]
* **Time per value and gradient** (median, release): hand-written 0.57 µs;
  in-house dual (fixed size 29) 2.5 to 2.8 µs; num-dual 0.15 fixed size 17
  µs (6.5 to 7.3 µs with fat LTO), runtime size 50 µs;[^numdual] candle
  0.10.2 55 to 61 µs; Burn 0.21 with `Autodiff<NdArray<f64>>` 99 to 118 µs.
* **Hessians.** Central differences of the exact gradient took 33 to 53 µs
  and agreed with num-dual's exact Hessian to 5.5e-11; num-dual's own took
  370 to 610 µs.
* **Footprint** (clean release build, crates pulled): none 4.5 s, 0 crates;
  num-dual 27 s, 19; Burn 75 s, 126 (a 20 s rebuild); candle 107 s, 114,
  with C compiled through `onig_sys`.
* **ODEs.** A one-compartment model (3 parameters, fixed-step RK4, 100
  steps a year for 10 years) differentiated with duals through the solver
  matched the closed-form model's gradient to 9e-10 (the RK4 error) at 1.6
  to 2.2 times the cost of a plain solve.
* Not measured there: nuts-rs runs per second.

## Chunked passes in the repository

`prospicio-reserving`'s `bayes/dual.rs` (`Dual<N>` and `gradient::<N>`,
`⌈p / N⌉` passes for a parameter count known at run time) against the
crate's own hand-written gradients (2026-10-13, same machine and compiler;
the compartmental note, decision 4).[^compartmental] The benchmark is the
ignored test `autodiff_benchmark`.[^code]

* **Times** (release, median µs per value and gradient, one run of three;
  runs differ by about 15%). `p` = 27 and 90 are the one-OS compartmental
  models 1 and 2 on a synthetic triangle shaped like company 337's training
  triangle, model 2's 87 parameters padded with three standard normals;
  they have no hand-written gradient.

  | `p` | f64 value | hand-written | N = 4 | N = 8 | N = 16 | N = 32 | fixed size `N = p` |
  |---|---|---|---|---|---|---|---|
  | 27 | 0.35 | | 5.23 | 4.31 | 3.73 | 3.25 | 2.84 |
  | 29 (Meyers CRC) | 0.52 | 1.11 | 5.60 | 3.54 | 3.04 | 2.77 | 3.55 |
  | 90 | 2.42 | | 100.3 | 63.9 | 48.5 | 43.5 | 58.4 |

* **The chunk.** 32 tangents was the fastest or within noise of it in every
  row and run; at 90 parameters three passes of 32 beat one pass of 90,
  whose 728-byte duals cost more to copy than the two extra passes. Fewer
  tangents cost more: at 29 parameters, 2.9 to 3.2 times the hand-written
  gradient at 8 and 4.7 to 5.8 at 4, against 2.3 to 2.5 at 32.
* **The repository's hand-written CRC gradient** takes 1.1 to 1.4 µs, about
  twice the throwaway's 0.57 µs: `Model::log_posterior` allocates its
  variances and residuals on each call. So the ratio to it (2.5) is smaller
  than the throwaway's dual-to-hand ratio (4 to 5), with the dual itself
  about as fast (2.8 against 2.5 to 2.8 µs).
* **Correctness.** Over 20 points per model on its whitening scale, the
  dual gradient matched the hand-written one to at most 1.1e-15 relative
  to its largest component and 2.5e-13 componentwise (Meyers CSR), and the
  generic value the hand-written value to 1.6e-14, for `BayesOdp` (GenIns,
  RAA), `BayesBf` (cv 0.2, flat) and `Meyers` (CRC, CSR, CAY).
* **Two numerical details.** A dual `tanh` with derivative `1 - tanh²`
  loses about `ε / (1 - tanh²)` relative (2e-14 at `x = -4`); `1 / cosh²`
  does not. The kernel `(1 - e^(-x)) / x` of the one-compartment closed
  form, written as a quotient, has a dual derivative that loses about
  `2 ε / |x|` relative (3e-12 at `x = 1e-5`), so its series branch reaches
  `|x| < 1e-2` with seven terms rather than `1e-5` with three; its dual
  gradient then matches central differences to 7.1e-9 through
  `k_er = k_p`.

[^design]: reserving-bayes.md, decision 15
[^compartmental]: reserving-compartmental.md, decision 4 (PR A)
[^code]: The oracle tests and the benchmark (autodiff_benchmark, ignored)
[^numdual]: num-dual on crates.io
[^burn]: Burn on crates.io
