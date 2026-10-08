---
type: Reference
title: Property exposure curves beyond MBBEFD
description: The exposure curves used in property per-risk rating (first-loss scales, PSOLD, Lloyd's and reinsurer curves), which of them prospicio_pricing::exposure covers and how.
resource: https://pep.vse.cz/doi/10.18267/j.pep.683.pdf
tags: [pricing, exposure-rating, property, reinsurance, mbbefd]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-06T08:30:00Z }
verified: { by: process:ci, at: 2026-10-06T17:45:37Z }
sources:
  - id: pep
    resource: https://pep.vse.cz/doi/10.18267/j.pep.683.pdf
    title: Exposure modelling in property reinsurance, Prague Economic Papers 28(2), 2019
  - id: cas
    resource: https://www.casact.org/sites/default/files/presentation/reinsure_2014_handouts_paper_3348_handout_2171_0.pdf
    title: CAS Reinsurance Seminar 2014 handout on exposure curves
  - id: verisk
    resource: https://www.verisk.com/siteassets/media/downloads/demystifying-the-origins-and-applications-of-original-loss-curves-.pdf
    title: Verisk, Demystifying the origins and applications of original loss curves
  - id: code
    resource: ../crates/prospicio-pricing/src/exposure.rs
    title: prospicio_pricing::exposure
  - id: profile
    resource: ../crates/prospicio-pricing/src/profile.rs
    title: prospicio_pricing::profile (tests)
---

# The curves in use

* **First-loss scales (Europe)**: an exposure curve `G(d) = E[min(X, d)] / E[X]`
  on the destruction rate. The first research was Salzmann's (1963, INA
  homeowners data); few other curves are public.[^pep]
* **ISO PSOLD (US)**: tables built from ISO's commercial property data,
  by occupancy and size of risk; licensed, not public.[^pep][^verisk]
* **Named curves**: Lloyd's curves, the reinsurers' own (Swiss Re,
  Munich Re), Salzmann (1960 INA homeowners) and Ludwig (1984–1988
  homeowners and small commercial); fitted alternatives use log-log
  interpolation or MBBEFD.[^cas]
* **Riebesell** is a liability increased-limits scale; it is the
  collective model with Pareto claim sizes.[^verisk]

# How prospicio_pricing::exposure covers them

* MBBEFD with Bernegger's `c` family covers the Swiss Re Y1–Y4 and
  Lloyd's curves.
* `TabulatedCurve` takes any published table (Salzmann, Ludwig, PSOLD, a
  reinsurer's): linear interpolation, required concave, which makes the
  destruction rate discrete. Its mean rate is the first chord's, so a
  coarse first point overstates the mean rate: a c = 3 curve tabulated
  every 0.001 still overstates it, because the curve is steepest at
  0.[^code]
* `SeverityCurve` turns any severity into a curve at an MPL, which covers
  Riebesell (a Pareto).[^code]
* Not covered yet: log-log interpolation between table points.

# Surplus and per-risk XL from a profile

* A risk ceding `c` to a surplus keeps `(1 − c)` of every loss, so the
  per-risk XL it inures to sees the same destruction-rate curve at sum
  insured `(1 − c) SI`; its exposure-rated loss is
  `EL (1 − c) [G((a + l)/((1 − c) SI)) − G(a/((1 − c) SI))]`. Simulated
  events agree with this within four standard errors.[^profile]
* A band whose sums insured spread between bounds is not its mean risk:
  a risk's expected loss is proportional to its SI, so the band's
  cession averages over SI weighted by SI. A 1m–5m band with a 2m
  retention cedes 3/8 against 1/3 for its 3m mean risk; spreading moves
  loss into layers the mean risk never reaches.[^profile]
* A uniform spread fixes the band's mean SI at the midpoint, which a
  band's total SI over its risks need not match. The truncated
  exponential `∝ exp(θ s)` on the bounds (the most even spread with a
  given mean) matches both, with `θ` from `1/(1 − e^{−t}) − 1/t = p`,
  `t = θ (U − L)`, `p` the mean's place in the band. A 1m–5m band with
  mean 2m leans to small risks and cedes less than the uniform band's
  3/8 to a 2m surplus. Averaging over probability (`∫₀¹ g(Q(p)) dp`)
  rather than SI keeps a steep tilt accurate; the surplus cession's kink
  at the retention inside a quadrature piece then costs about 3e-8,
  relative.[^profile]

[^pep]: Exposure modelling in property reinsurance
[^cas]: CAS Reinsurance Seminar 2014 handout
[^verisk]: Verisk, original loss curves
[^code]: prospicio_pricing::exposure
[^profile]: prospicio_pricing::profile (tests)
