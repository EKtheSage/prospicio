---
type: Reference
title: Clark's Basics of Reinsurance Pricing
description: The CAS study note behind prospicio_pricing::experience and the planned casualty exposure rater; its per-risk experience rating example rounds trend factors to three decimals and trended losses to the dollar, counts days over 365.25, develops each year's layer total, and totals the loss cost as a ratio of sums; its casualty exposure section prints one numeric example, for workers compensation by excess loss factors, and its liability formulas without numbers.
resource: https://www.casact.org/sites/default/files/2021-03/8_Clark.pdf
tags: [pricing, reinsurance, experience-rating, burning-cost, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-13T11:20:00Z }
verified: { by: process:ci, at: 2026-10-13T17:31:26Z }
sources:
  - id: clark
    resource: https://www.casact.org/sites/default/files/2021-03/8_Clark.pdf
    title: Clark, Basics of Reinsurance Pricing, CAS study note (first version 1996, revised 2014), per-risk experience rating
  - id: tests
    resource: ../validation/tests/pricing.rs
    title: The experience rating parity tests (module clark_experience)
  - id: dev
    resource: ../validation/tests/pricing.rs
    title: Per-claim development against the chain ladder (module claim_development)
---

# What it is

The CAS study note on reinsurance pricing. Its per-risk experience rating
example takes fourteen large losses from 1988 to 1995, trends them at 4% a
year to 7/1/1997, applies a 400 xs 100 layer, develops each accident year's
layer total with an excess factor and divides by the on-level premium: a
total loss cost of 12.4%. It is the reference for
`prospicio_pricing::experience`.[^clark]

# Conventions met reproducing it

- **Day count.** Factors `1.04^t` with `t` the actual days from the accident
  date to 7/1/1997 over 365.25 reproduce all fourteen printed factors to
  three decimals.[^tests]
- **Rounding.** The note multiplies the untrended loss by the factor rounded
  to three decimals and rounds the trended loss to the dollar; the loss in
  the layer comes from that rounded value. The parity test passes the
  printed factors (`Trend::Factors`) and matches trended losses within
  half a dollar, layer losses by year within 1.5 and loss costs to the
  printed 0.1%.[^tests]
- **Development.** Excess development is applied to each year's layer
  total, not per claim.[^clark]
- **Total.** The 12.4% is the developed losses over the premium, each
  summed over the years (1,991,789 over 16,112,389), not the mean of the
  yearly loss costs.[^tests]

# What it does not print

Per-claim development (IBNER) and a count IBNR load appear only in prose:
in the workers compensation section, projecting each claimant into the
layer means "a smaller development factor would then be needed to include
only 'true IBNR' claims". There is no exhibit to reproduce, so
`DevelopmentPattern::from_history` is checked instead against the chain
ladder: with every claim valued at every age up to its latest, pooling the
claims seen at both ages gives the volume-weighted chain ladder factors of
the summed triangle, and each year's developed claims sum to its chain
ladder ultimate (to 1e-13 on the factors).[^clark][^dev]

Nor does it print an index (stability) clause or currency conversion; the
index clause is checked against a tower whose retention and limit are
moved by hand.[^clark]

[^clark]: Clark, Basics of Reinsurance Pricing.
[^tests]: `validation/tests/pricing.rs`, module `clark_experience`.
[^dev]: `validation/tests/pricing.rs`, module `claim_development`.

# The casualty exposure section

Section 3A(b) prints the liability exposure factor for a primary policy,
the same with ALAE included in the treaty's loss (the attachment and
limit divided by `1 + e` before they are capped at the policy limit), and,
in 3B(a), the factor for an umbrella policy over an underlying limit. None
has a numeric example: its truncated Pareto (`B`, `Q`, `P`, `S`, `T`) and
mixed exponential severities have no printed parameters, and its two
illustrations (an umbrella 1M xs 1M under a 500K xs 500K treaty, and a 1M
limit with an ALAE loading of 1.5 leaving a treaty at 1.5M unexposed) are
symbolic.[^clark]

Its one numeric example is workers compensation: four state and hazard
group rows with standard premium, ELR and ELFs at 250,000 and 1,000,000,
for a 750,000 xs 250,000 treaty. Factor `ELF(250K) − ELF(1M)` (0.024,
0.032, 0.050, 0.065), treaty losses premium × ELR × factor (1,680, 2,240,
4,250, 5,525), 13,695 in total over 400,000 of premium, a 3.42% loss cost
(13,695 / 400,000 = 3.4238%). The arithmetic reproduces exactly from the
printed inputs (checked by hand 2026-10-17; a CI test follows with the
rater, `docs/design/casualty-exposure.md`).[^clark]

The umbrella's drop-down uses an aggregate excess factor `ϕ` defined only
in the later section on aggregate distributions.[^clark]
