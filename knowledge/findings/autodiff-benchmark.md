---
type: Finding
title: Automatic differentiation for small Bayesian reserving models, benchmarked
description: On Meyers' CRC log posterior (29 parameters, 55 cells) every autodiff option matched the hand-written gradient to about 1e-15, but an in-house forward-mode dual type cost 4 to 5 times the hand-written gradient while num-dual cost 10 to 90 times, candle about 100 and Burn about 190; Burn 0.21's sigmoid rounds to f32 on an f64 backend.
tags: [bayes, autodiff, performance, burn, candle, num-dual, ode]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-13T15:00:00-07:00 }
sources:
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: reserving-bayes.md, decision 15
  - id: numdual
    resource: https://crates.io/crates/num-dual
    title: num-dual on crates.io
  - id: burn
    resource: https://crates.io/crates/burn
    title: Burn on crates.io
---

# Automatic differentiation for small Bayesian reserving models

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
* Not measured: nuts-rs runs per second, and chunked dual passes at a
  runtime parameter count.

[^design]: reserving-bayes.md, decision 15
[^numdual]: num-dual on crates.io
[^burn]: Burn on crates.io
