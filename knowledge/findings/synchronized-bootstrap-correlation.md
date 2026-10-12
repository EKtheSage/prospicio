---
type: Finding
title: Dependence between reserving lines, synchronized bootstrap and rank correlation
description: Resampling the same residual positions in every line (Kirschner, Kerley and Isaacs 2008) gives the lines' parameter error about the correlation of their paired residuals, at most its absolute value; on the six CLRD lines the ODP's is within 0.055 of it, Mack's within 0.145, and process error dilutes it. Iman-Conover on the line totals reproduces a Spearman matrix after converting it to 2 sin(pi rho / 6). Iman-Conover leaves the first line's rows sorted by its total, so the paired rows are shuffled afterwards.
tags: [reserving, bootstrap, dependence, correlation, capital, iman-conover, clrd]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-08T21:00:00-07:00 }
sources:
  - id: kki
    resource: https://casact.org/sites/default/files/2021-07/Two-Approaches-Kirschner-Kerley-Isaacs.pdf
    title: Kirschner, Kerley and Isaacs (2008), Two approaches to calculating correlated reserve indications across multiple lines of business, Variance 2(1), 15-38
  - id: tm
    resource: https://ideas.repec.org/a/taf/uaajxx/v11y2007i3p70-88.html
    title: Taylor and McGuire (2007), A synchronous bootstrap to account for dependencies between lines of business in the estimation of loss reserve prediction error, North American Actuarial Journal 11(3), 70-88
  - id: design
    resource: ../docs/design/reserving-v02.md
    title: Design note, reserving v0.2, decision 10
  - id: test
    resource: ../validation/tests/reserving_dependence.rs
    title: Validation test of SegmentDependence on the six CLRD lines
  - id: data
    resource: /datasets/clrd-lines.md
    title: The six CLRD lines dataset
---

# Finding

* **What Kirschner et al. synchronize.** Their correlated bootstrap draws,
  for each accident and development year, one position, and every line
  takes its own residual at that position; each line keeps its own
  residuals. They say nothing of process error, so prospicio draws it
  independently per line, as each bootstrap draws it.[^kki] Taylor and
  McGuire's synchronous bootstrap is the same idea.[^tm]
* **The induced correlation.** To first order a line's reserve is
  `sum_c a_c r(p_c)`, so two synchronized lines' parameter error has
  correlation `rho a1.a2 / (|a1| |a2|)`, `rho` the correlation of their
  paired residuals over the positions: at most `|rho|`, and near it when
  the lines develop alike. On the 15 pairs of the six CLRD lines[^data]
  (10,000 simulations, `rho` from -0.15 to 0.60) the ODP's is within 0.055
  of `rho` (largest gap prodliab with wkcomp, 0.54 against 0.60) and
  Mack's within 0.145 (ppauto with wkcomp, 0.18 against 0.32: ppauto
  develops much faster, so the two reserves weigh different factors'
  residuals).[^test]
* **Process error dilutes it**, Mack's more than the ODP's: comauto with
  wkcomp is 0.53 parameter only and 0.40 with the Gamma process under the
  ODP, 0.49 and 0.17 under Mack's.[^test]
* **Each line keeps its distribution** when every line has a residual
  wherever any has: the pool and the draw count are the same, only the
  pairing changes. With one segment the synchronized draws are the
  independent ones bit for bit for the ODP, and for Mack's bootstrap when
  no link has a zero cumulative value or a zero sigma; otherwise Mack's
  synchronized path draws a position for such a link where the independent
  one draws nothing, so the stream shifts and the draws are equal in
  distribution only.[^design]
* **Rank correlation.** Iman–Conover on normal scores gives a Spearman rho
  near `(6/pi) asin(r/2)`, so a Spearman target goes in as
  `r = 2 sin(pi rho / 6)`. With targets 0.5, 0.25 and -0.3 on comauto,
  ppauto and wkcomp the totals' Spearman rhos are 0.501, 0.253 and
  -0.304 at 10,000 simulations. A Spearman rho of 1 converts to just below
  1 and is accepted.[^test]
* **Capital.** ODP with the Gamma process on comauto and wkcomp, TVaR at
  99% of the total and its diversification benefit: 4,949,627 and 136,529
  independent, 5,011,717 and 78,069 synchronized, 5,075,655 and 10,501 at
  a Spearman rho of 0.9.[^test]
* **No shared uniforms between lines.** The segments of a simulation read
  separate, consecutive parts of its stream. On the six CLRD lines at 4,000
  simulations, seeds 1 to 3, the mean of the 15 pairwise correlations of
  the independent totals, and of the synchronized process error alone, was
  within 0.005 of zero for both bootstraps.[^design]
* **Iman–Conover sorts the first group.** `reorder_groups` keeps the first
  group's column of target ranks in order, so its result lists the first
  segment's simulations by increasing total: a prefix of the rows is then
  a biased sample, and two such results joined with
  `Pairing::Independent` (accepted, as the seeds differ) are strongly
  dependent through that common order. prospicio shuffles
  the paired rows afterwards; a unit test checks the row index is
  uncorrelated with each segment's total.[^design]

# Consequences

* Synchronize only lines of the same origins, ages and observed cells;
  prospicio refuses others.
* Expect the synchronized correlation of the totals below the residuals'
  and well below it with Mack's process error; use the rank correlation
  when the target is a judgment, not the data.
