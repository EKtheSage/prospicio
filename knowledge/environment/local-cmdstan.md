---
type: Environment Fact
title: CmdStan on the local Windows machine
description: CmdStan 2.39.0 is built at ~/.cmdstan and compiles models with RTools 4.5's make and g++, from PowerShell or cmd only; cmdstanpy 1.3.0 under uv needs MAKE set because it looks for mingw32-make; the Meyers reference also needs arviz 0.23.4 for PSIS-LOO.
tags: [environment, stan, cmdstan, windows, bayes, validation]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-12T00:00:00Z }
stale_after: 2027-01-12T00:00:00Z
sources:
  - id: script
    resource: ../validation/scripts/reserving_bayes_stan.py
    title: The Stan reference generator for BayesOdp
  - id: design
    resource: ../docs/design/reserving-bayes.md
    title: Design note, Bayesian reserving (decision 10)
  - id: compartmental
    resource: ../validation/scripts/reserving_compartmental_stan.py
    title: The Stan reference generator for Compartmental
  - id: meyers
    resource: ../validation/scripts/reserving_meyers_stan.py
    title: The Stan reference generator for Meyers (CRC and CSR)
  - id: gm2
    resource: ../validation/scripts/reserving_compartmental_gm2_stan.py
    title: The Stan reference generator for Compartmental model 2
---

# Facts (2026-10-12)

* CmdStan 2.39.0 is built at `~/.cmdstan/cmdstan-2.39.0`: `bin/stanc.exe`
  and `stan/lib/stan_math/lib/tbb/tbb.dll` are there, and the prebuilt
  `examples/bernoulli/bernoulli.exe` samples.[^script]
* The toolchain is RTools 4.5 at `C:/rtools45`: `usr/bin/make.exe` and
  `x86_64-w64-mingw32.static.posix/bin/g++.exe` (GCC 14.3.0). Neither is
  on PATH. With both directories and CmdStan's `tbb` directory prepended
  to PATH, a model compiles in about 55 s.[^script]
* Run the generator from PowerShell or cmd. From Git Bash the same `make`
  stops with "make/stanc:39: Cannot detect OS properly".
* cmdstanpy 1.3.0 calls `mingw32-make` on Windows, which RTools 4.5 does
  not have (`[WinError 2]`). It reads the `MAKE` environment variable at
  compile time, so the generator sets `MAKE` to RTools' `make.exe`.[^script]
* cmdstanpy compiles next to the `.stan` file; the generator compiles a
  copy under `target/stan/`, so the executable stays out of the tree, and
  rewrites the copy only when the model changes, so a rerun skips the
  compile.[^script]
* `uv run --no-project --with cmdstanpy==1.3.0 --with pandas python
  validation/scripts/reserving_bayes_stan.py` resolves cmdstanpy 1.3.0
  with numpy 2 and pandas 3 from uv's cache. numpy 2 prints a scalar as
  `np.float64(...)`, so the CSV writer converts to `float` first.
* The full reference (GenIns and RAA, 4 chains of 2,000 warm-up and 25,000
  draws each, dense metric) takes about 60 s once the model is
  compiled.[^design]
* Meyers' CRC and CSR (2026-10-12): both programs compile in about 55 s
  each; the four cases (paid and incurred, 4 x 25,000 draws, dense metric,
  `adapt_delta` 0.99) take about 5 minutes in all, about half of it
  arviz 0.23.4's PSIS-LOO on 100,000 draws x 55 cells. arviz resolves
  under Python 3.14 with cmdstanpy 1.3.0, pandas 3.0.6 and numpy 2.5.4
  (`--with arviz==0.23.4`); it warns that it is pre-1.0 and, with Pareto k
  near 1, about `exp` overflow in its smoothing, which does not change the
  estimate. A rerun with the same seed reproduced every summary bit for
  bit.[^meyers]
* Meyers' CAY (2026-10-13): `validation/scripts/reserving_meyers_cay_stan.py`
  imports `reserving_meyers_stan.py` (the toolchain, the compile under
  `target/stan/` and the data handling), so it runs from the repository
  root with the same `uv` command; one case (incurred) takes a few
  minutes, and three runs with the same seed reproduced every
  summary.[^meyers]
* Gesmann and Morris' model 1 (2026-10-13):
  `validation/scripts/reserving_compartmental_stan.py` imports the
  toolchain and the compile from `reserving_meyers_stan.py` and runs with
  the same `uv` command; the program compiles in about 2 minutes and 4
  chains of 2,000 warm-up and 25,000 draws at `adapt_delta` 0.99 and
  `max_treedepth` 15 take about 20 minutes (one chain finished 6 minutes
  after the others). Stan prints rejections during warm-up (a NaN mean,
  an LKJ Cholesky factor at 0); none in the kept draws, which have no
  divergence and no draw at the maximum depth.[^compartmental]
* Gesmann and Morris' model 2 (2026-10-14):
  `validation/scripts/reserving_compartmental_gm2_stan.py` imports the
  same helpers and runs with the same `uv` command. The program compiles
  in about 90 s; 4 chains of 2,000 warm-up and 25,000 draws at
  `max_treedepth` 15 took about 30 minutes at `adapt_delta` 0.99 (5
  divergent transitions, so the reference tightens to 0.999) and about 92
  minutes at 0.999 (the slowest chain; the four finished 28 to 92 minutes
  in, with seed sweeps of our own fits loading the machine). Warm-up again
  prints rejections (an infinite or NaN location); arviz warns that
  Pareto k is above 0.7 for some cells (max 1.04).[^gm2]
