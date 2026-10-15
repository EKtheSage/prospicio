---
type: Finding
title: A spliced body is fitted by truncation at the join, not censoring, and a small P-P distance does not clear the tail
description: On the severity guide's synthetic losses (a lognormal body of meanlog -2.00, sdlog 1.20 below 1, a Pareto 1.8 above), fit_severity with the reporting threshold of 0.1 gives -1.93 and 1.04; treating the claims above 1 as censored at 1 moves it further off (-1.84, 0.98), because censoring puts the body's own mass above the join; the largest P-P gap is 0.017 while the three largest losses (11.0, 11.7, 12.6) sit far above the fit's quantiles (5.0, 5.8, 7.8).
tags: [severity, fitting, truncation, censoring, guides]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-15T12:00:00Z }
sources:
  - id: guide
    resource: ../docs/guides/severity-and-frequency.md
    title: Severity and frequency, section 6 (the printed values, checked by python/tests/test_guides.py)
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/194
    title: Severity fitting with truncation and censoring beyond the Pareto family
---

The guide draws 8,000 claims from a splice: a lognormal body (meanlog
-2.00, sdlog 1.20) below 1 and a Pareto with alpha 1.8 above it, the share
above 1 set by the splice and not by the lognormal's own tail. Of them,
4,681 lie above a reporting threshold of 0.1.

`fit_severity("lognormal", reported, reporting_thresholds=[0.1] * n)`
gives meanlog -1.93 (standard error 0.05) and sdlog 1.04 (0.03), the same
as the guide's hand-written truncated likelihood to two decimals. The
sdlog is short because the claims above 1 follow the Pareto.

Two ways to keep the Pareto claims out of the body's fit behave
differently:

- **Censoring at 1** (each claim above 1 known only to exceed 1) gives
  meanlog -1.84 and sdlog 0.98, further from the truth. Censoring says the
  body itself puts that much mass above 1, which in a splice it does not.
- **Truncation at both ends** (only claims between 0.1 and 1, the window's
  probability in the normalizing term) gives -2.21 and 1.33. It is the
  right model for a splice. `fit_severity` has no upper truncation, so the
  guide keeps its hand-written likelihood for this case.

`goodness_of_fit` on the threshold-only fit reports a largest gap of 0.017
between the model's and the product-limit survival functions, since most
claims are in the body. The Q-Q pairs show the misfit: the three largest
losses, 11.0, 11.7 and 12.6, against the fit's quantiles there of 5.0, 5.8
and 7.8. Read the upper Q-Q pairs, not only the distance, before trusting a
body family in the tail.
