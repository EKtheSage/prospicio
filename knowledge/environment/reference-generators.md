---
type: Environment Fact
title: Rebuilding the parity references with uv
description: Every Python generator under validation/scripts carries PEP 723 metadata (the package versions its output records, the rest resolved by exclude-newer at the day after the file was committed), so uv run rebuilds its file; on 2026-11-12, 61 of the 61 weekly generators reproduced their committed files within each row's tolerance once three gaps were fixed (eltq missing from the ELT generator, SciPy missing from a script that loads another by importlib, the stacking weights needing NumPy 2.4.6), and under NumPy 2.5 the stacking weights move by 2e-12 against a tolerance of 1e-12.
tags: [environment, validation, reference, uv, reproducibility]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-11-12T00:00:00Z }
stale_after: 2027-05-12T00:00:00Z
sources:
  - id: runner
    resource: ../validation/scripts/rerun_references.py
    title: validation/scripts/rerun_references.py (the weekly set and the comparison rules)
  - id: manifest
    resource: ../validation/scripts/check_reference_manifest.py
    title: validation/scripts/check_reference_manifest.py (refuses a Python generator without metadata)
  - id: pep723
    resource: https://peps.python.org/pep-0723/
    title: PEP 723, inline script metadata
---

# Fact

* A Python generator's `# /// script` block lists each package it imports
  (or loads by `importlib`, or needs at run time, as `aggregate` needs
  `eltq` for an ELT with secondary uncertainty), pinned `==` where the file
  it writes or its docstring records the version, and
  `[tool.uv] exclude-newer` at the day after the file's last commit, so an
  unpinned package resolves as it could have then.[^manifest]
  `uv run validation/scripts/<name>.py` rebuilds the file from the
  repository root; `check_reference_manifest.py` fails on a generator
  without the block.
* `rerun_references.py` reruns the cheap pure-Python generators (the
  `aggregate`, SciPy, mpmath, statsmodels and NumPy families, 61 scripts,
  about 9 minutes, most of it `aggregate_bivariate_copula.py` and the
  mpmath series) and compares each file with the committed copy: rows with
  `expected`, `abs_tol` and `rel_tol` within the row's own tolerance, other
  CSVs to 1e-9, other files byte for byte, and every file under
  `validation/data/` unchanged; it puts every file back.[^runner]
  `--latest` drops the pins and the date.
* On 2026-11-12 all 61 reproduced under the pins after three fixes:
  `aggregate_elt_units.py` needs `eltq==2.15.0` (named only in its
  docstring's `uv run --with`), `scipy_fit_curves.py` needs SciPy through
  the `scipy_severity_fits.py` it loads by `importlib`, and
  `stacking_weights.py` reproduces its weights only under NumPy 2.4.6
  (with 2.5.4 two pseudo-BMA weights move by 2.1e-12 and 2.9e-12 against
  an `abs_tol` of 1e-12, and the log densities it writes to
  `validation/data/stacking_lpd.csv` move too).[^runner]
* `scipy_special.py` records SciPy 1.17.1 and gives the same values under
  1.18.1 (the `source` column moves; the runner lists that and does not
  count it as a difference).
* Not in the weekly set (#844 starts with the pure-Python families): chainladder-python,
  the Stan generators (CmdStan is not a pip package), ONNX,
  pyvinecopulib, scikit-learn, MAPIE, SHAP, `statsmodels_fremtpl2.py`
  (it needs the downloaded freMTPL2 data) and the R scripts.

# Citations

[^manifest]: validation/scripts/check_reference_manifest.py, `pinned`.
[^runner]: validation/scripts/rerun_references.py; the run of 2026-11-12 in a cloud session with uv 0.11.32.
