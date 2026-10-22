---
type: Environment Fact
title: R on CI's Windows runner starts with no CRAN mirror
description: The weekly and manual CI runs failed in the r (windows-latest) job from 2026-10-12 because install.packages() was called with no repos and R on the Windows runner had no mirror set; Ubuntu got setup-r's RSPM repos. The install steps now fall back to the cloud mirror when CRAN is unset.
tags: [environment, ci, r, windows]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-22T12:00:00Z }
stale_after: 2027-04-22T00:00:00Z
sources:
  - id: runs
    resource: process:GitHub Actions scheduled runs of ci.yml 38215829780 (2026-10-12) and 39212765062 (2026-10-19)
    title: Failed r (windows-latest) jobs
  - id: ci
    resource: ../.github/workflows/ci.yml
    title: CI workflow
---

# What failed

The `r` job runs on Windows only in the weekly run and in a manual
(`workflow_dispatch`) run, so pull requests never saw this. From
2026-10-12 the Windows leg failed one second into the step that installs
S7, lightgbm and xgboost:[^runs]

```
Error in contrib.url(repos, "source") :
  trying to use CRAN without setting a mirror
```

The step had passed on 2026-10-05, when it still called
`install.packages('S7', repos = 'https://cloud.r-project.org')`; #128
dropped the `repos` argument on the assumption that setup-r's
`use-public-rspm: true` sets the repos everywhere. It does on Ubuntu, not
for `Rscript` under bash on the Windows runner, where `getOption('repos')`
keeps CRAN at `@CRAN@`.

# The fix

Both `install.packages()` calls in the `r` job (the engines and
`remotes`/`pkgbuild` for the install-from-tarball check) first run

```r
r <- getOption('repos'); if (is.null(r) || r['CRAN'] %in% c('@CRAN@', NA)) { r['CRAN'] <- 'https://cloud.r-project.org'; options(repos = r) }
```

which sets CRAN only when it is unset and keeps any other repos, so Ubuntu
still installs RSPM's binaries.[^ci] A later `install.packages()` without
`repos` in a job that runs on Windows needs the same guard.

# The cancelled bayes job

The `bayes` job of the 2026-10-19 run was cancelled at 38.5 minutes with
no test failure; the weekly run then shared a concurrency group with
pushes to `main`, which has since changed (see
[Where CI's time went](/environment/ci-runtime.md)).

[^runs]: GitHub Actions scheduled runs of ci.yml, 2026-10-12 and 2026-10-19.
[^ci]: `.github/workflows/ci.yml`, job `r`.
