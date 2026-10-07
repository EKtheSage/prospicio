---
type: Finding
title: R callbacks must stay on R's main thread
description: An R function held inside a Rust object may be called only from the thread R runs on, so parallel code that meets one must run in order on the calling thread.
tags: [r, extendr, threads, rayon, custom-distribution]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-06T07:40:00Z }
sources:
  - id: custom
    resource: ../crates/prospicio-prob/src/custom.rs
    title: prospicio_prob::Custom
  - id: binding
    resource: ../crates/prospicio-r/src/pareto.rs
    title: R CustomDist and RFunction
  - id: tests
    resource: ../R/prospicio/tests/test-custom.R
    title: R custom_distribution tests
---

# Finding

* extendr's `Function` (an `Robj`) is neither `Send` nor `Sync`, and for
  good reason: R's interpreter may only be entered from its main thread.
  Calling an R function from a rayon worker crashes R or corrupts its
  state.[^binding]
* A Python callable is different: `Python::attach` from any thread is
  safe, but every call holds the GIL, so threads only queue on it.

# Resolution

* `Distribution::is_parallel_safe()` defaults to true; a `Custom` built
  with `parallel_safe = false` (both bindings do) returns false, and a
  `Mixture` returns false if any component does.[^custom]
* `simulate_events` and copula simulation check it and then run every
  simulation in order on the calling thread, which for R is the main
  thread. Each simulation still owns its stream, so the draws are
  identical to the parallel run (a Rust test checks this).[^custom]
* The R wrapper holds the function in a newtype with `unsafe impl
  Send/Sync`, justified only by that check.[^binding] Note that Rust 2021
  closures capture fields, not the whole value: a closure that uses
  `f.0` captures the non-`Send` `Function`, so the call must go through a
  method on the wrapper.
* An R error inside the callback reaches the Rust side as an `Err`
  through extendr's `Function::call`; R still prints its own
  "Error in ..." line to stderr.[^tests]

[^custom]: prospicio_prob::Custom
[^binding]: R CustomDist and RFunction
[^tests]: R custom_distribution tests
