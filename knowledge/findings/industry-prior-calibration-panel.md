---
type: Finding
title: "Industry priors per line on the calibration panel: the Pearson scale explains more than the whole within-company spread on incurred"
description: "IndustryPrior on the calibration study's 200 Schedule P triangles (50 companies a line): apriori 0.50 (other liability) to 0.78 (personal auto paid), between-company cv 0.13 to 0.39, within-company cv 0.15 to 0.47, and the chain ladder's ODP process cv above the within cv on every incurred line and on commercial auto paid, so subtracting the process would floor the prior's cv at 0.01; the pattern prior's sd of the log share ratios is 0.28 to 1.09 at 24 months and 1.07 to 2.1 at 120. Simulated companies' parameters are recovered within four standard errors."
tags: [reserving, bayes, bornhuetter-ferguson, industry-prior, schedule-p, calibration]
status: draft
generated: { by: claude-code/local-session, at: 2026-11-03T12:00:00Z }
sources:
  - id: panel-test
    resource: ../validation/tests/reserving_industry_prior_panel.rs
    title: the_industry_prior_per_line_on_the_calibration_panel (ignored; run by hand) with the recorded values
  - id: recovery
    resource: ../validation/tests/reserving_industry_prior.rs
    title: Recovery of simulated companies' parameters (plain cargo test)
  - id: script
    resource: ../validation/scripts/calibration_panel.py
    title: Builds the panel from cas-schedule-p[screens]==2026.6.13
  - id: note
    resource: ../docs/design/reserving-bayes.md
    title: Design note, decision 28 (the industry prior) and open decisions 20 to 25
  - id: code
    resource: ../crates/prospicio-reserving/src/industry_prior.rs
    title: prospicio_reserving::IndustryPrior, IndustryPriorSettings, PatternPrior, PriorReport
---

# What was found

Run on 2026-11-02 (Windows, release build) on the calibration study's
panel, sha256
`075b392525b26c505c251ea1d94f1641b4c6ab4216ee085655e5ec678fff59d2`
(cas-schedule-p 2026.6.13, the hash in
[the calibration panel finding](meyers-calibration-panel.md)), under
`IndustryPriorSettings::default()` (`min_developed` 0.5,
`subtract_process` false). Not marked verified: no CI job runs the panel
test, which needs the panel and asserts these values to 1e-6 relative on
the pinned hash only. The prior is deterministic.

The triangles are accident years 1988 to 1997 at 31 December 1997 (the
upper triangle); the squares add the outcomes to 120 months. All 50
companies of each line are used. "Process" is the chain ladder's ODP
process cv of an origin's ultimate loss ratio (decision 28's
`process_cv`); `cv` is the within cv under the default.

| Line | Basis | apriori (se) | between cv | within cv | process cv | origins |
|---|---|---|---|---|---|---|
| Commercial auto | paid | 0.6662 (0.0262) | 0.2592 | 0.2918 | 0.2960 | 450 |
| Commercial auto | incurred | 0.6612 (0.0251) | 0.2440 | 0.3412 | 0.5179 | 500 |
| Personal auto | paid | 0.7798 (0.0163) | 0.1388 | 0.1540 | 0.1436 | 450 |
| Personal auto | incurred | 0.7644 (0.0150) | 0.1294 | 0.1591 | 0.3249 | 500 |
| Workers' comp | paid | 0.6385 (0.0193) | 0.1890 | 0.2927 | 0.1697 | 450 |
| Workers' comp | incurred | 0.6512 (0.0195) | 0.1890 | 0.2961 | 0.6902 | 500 |
| Other liability | paid | 0.4958 (0.0297) | 0.3841 | 0.4705 | 0.3436 | 400 |
| Other liability | incurred | 0.5014 (0.0301) | 0.3941 | 0.4617 | 0.8898 | 500 |

On the squares (the same companies with every origin at 120 months)
`apriori` moves by at most 0.019 (personal auto paid, 0.7798 to 0.7606),
the between cv falls by 0.004 to 0.038 in every line, and the within cv
moves by -0.015 to +0.035: the outcomes change the prior little.

* **The process part is larger than the whole within spread** on all four
  incurred lines (0.32 to 0.89 against 0.16 to 0.46) and on commercial auto
  paid (0.296 against 0.292). The textbook prior cv, the within spread less
  the process, would then be the floor 0.01, which pins each origin's
  expected ultimate to the company level. A chain ladder's Pearson scale on
  Schedule P carries its misfit (calendar effects, case reserving changes),
  which is not process; this is why `subtract_process` is off by default
  (open decision 20). On paid it leaves 0.06 (personal auto), 0.24
  (workers' compensation) and 0.32 (other liability).
* **The between and within cvs are of the same size** in every line (ratio
  0.64 to 0.90 on the triangles). A program's level is about as uncertain
  as one of its origins about that level, so the normalised BF's common
  multiplier (`common_cv`) matters as much as `cv`.
* **The pattern's spread is wide at late ages.** The sd over companies of
  the log share ratio at 24 months is 0.28 (personal auto paid) to 1.09
  (personal auto incurred), at 60 months 0.74 to 1.62 and at 120 months
  1.07 to 2.1 on the triangles. At 120 months each company's share rests
  on one origin, so the spread is mostly estimation error; the prior leans
  little on the industry there (open decision 24).
* **`min_developed` 0.5 drops the newest origin on paid** (two on other
  liability paid, where 12 months is 16% and 24 months 39% developed) and
  none on incurred (68% to 83% at 12 months, other liability 54%).

Recovery (`reserving_industry_prior.rs`, run by every `cargo test`):
companies drawn as the normalised BF with a common multiplier
(`apriori` 0.7, between cv 0.15, within cv 0.1, pattern sds 0.1 to 0.25)
give, with exact cells and 2,000 companies, `apriori` 0.69958, within cv
0.09942 and between cv 0.15067, each within four standard errors derived
in the test. With ODP cells (process cv 0.05 on an ultimate, 1,500
companies) the realised within cv is 0.1138 and `subtract_process`
recovers 0.1002 against the structural 0.1.
