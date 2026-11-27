---
type: Finding
title: R closures sent to worker processes carry unforced promises
description: A function shipped to a PSOCK worker serializes its enclosing frame, and an argument that was never evaluated travels as a promise that the worker then evaluates in its own empty global environment.
tags: [r, parallel, serialization, tuning]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-27T05:20:00Z }
sources:
  - id: tuning
    resource: ../R/prospicio/R/models.R
    title: R cross_validate, grid_search and run_tasks
  - id: tests
    resource: ../R/prospicio/tests/test-models-parallel.R
    title: R parallel tuning tests
---

# Finding

* `parallel::parLapply(cl, tasks, f)` serializes `f` with its enclosing
  frame. In `grid_search()`, that frame holds the arguments `data`, `fit`
  and `score`, and an argument the function has not used yet is still a
  promise for the caller's expression (`fit`, say). The worker then
  evaluates it in its own global environment, where the caller's objects
  do not exist: "object 'fit' not found".[^tuning]
* Forcing the arguments (`force(data); force(fit); force(score)`) before
  building the tasks sends their values instead.[^tuning]
* The global environment is serialized by name, not by contents: a
  function made at the top level (as `function(train) fit(0.05, train)`
  typed at the console) reaches the worker without the globals it calls,
  and fails with "could not find function". Making it inside `local()`
  (or exporting the globals with `parallel::clusterExport()`) sends
  them; checked with a one-worker PSOCK cluster in R 4.3.[^tests]
* A forked worker (`parallel::mclapply`) does not copy the parent's
  threads, so a rayon pool the parent has already started has no workers
  in the child, and a parallel call there can wait forever. Fresh
  processes (a PSOCK cluster, `future`'s multisession plan, `mirai`)
  avoid it.

# Resolution

`cross_validate()` and `grid_search()` force their arguments before they
make tasks, take a `map` that runs the tasks, and document PSOCK workers
with `RAYON_NUM_THREADS=1` set before the package loads (#761). The
tests compare a PSOCK run with the serial one.[^tests]

[^tuning]: R cross_validate, grid_search and run_tasks
[^tests]: R parallel tuning tests
