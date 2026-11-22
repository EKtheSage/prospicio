---
type: Environment Fact
title: The abi3 wheel on Python 3.9
description: The package's Python files import and its light tests pass on Python 3.9 with the test dependencies' last releases for it; the one difference found was the built-in sum, which adds floats with compensation from Python 3.12 only, so a mean taken with it moved in the last bit between interpreters.
tags: [environment, python, wheel, abi3, ci, release]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-22T00:00:00Z }
stale_after: 2027-05-22T00:00:00Z
sources:
  - id: run
    resource: process:cloud-session, issue #962 (branch claude/cool-shannon-fqpoy8-ci-962-python39)
    title: The wheel installed into CPython 3.9.25 and 3.12.3, 2026-11-22
  - id: workflow
    resource: ../.github/workflows/release.yml
    title: The wheels job's step "Test the wheel on Python 3.9, the oldest it serves"
---

# Facts (2026-11-22)

* The wheel is abi3 from Python 3.9 (`abi3-py39` in
  `crates/prospicio-python/Cargo.toml`, `requires-python = ">=3.9"`), and
  until #962 every test ran on 3.12. On CPython 3.9.25 every submodule of
  `prospicio` imports, and the light test files (all but the NUTS-heavy
  files, the boosting tests, the guides and the notebooks) pass: 3,765
  tests.
* `uv pip install --group dev` resolves on 3.9, to older releases: SciPy
  1.13.1 and xgboost-cpu 2.1.4 among them. Python 3.9 reached its end of
  life in October 2025, so a test dependency may drop it at any release;
  the workflow's comment says to cut the step back to the import then.
* The built-in `sum` adds floats with compensated (Neumaier) summation
  from Python 3.12 and plainly before. `prospicio.models._mean_se`, the
  mean and standard error behind `Comparison`, `PermutationImportance`,
  `InteractionScreen` and `PartialDependence`, used it, so
  `test_explanations_average_over_folds` failed on 3.9 by one unit in the
  last place (the mean of eight values that are four values twice). It
  now uses `math.fsum`, which is correctly rounded on every version.
* A check of an installed wheel runs from outside `python/`: there
  `import prospicio` finds the checkout's source, which has no extension
  module. pytest is not affected, since it puts `tests/` on the path and
  not `python/`.
