---
type: Reference
title: Clark's Basics of Reinsurance Pricing
description: The CAS study note behind prospicio_pricing::experience; its per-risk experience rating example rounds trend factors to three decimals and trended losses to the dollar, counts days over 365.25, develops each year's layer total, and totals the loss cost as a ratio of sums.
resource: https://www.casact.org/sites/default/files/2021-03/8_Clark.pdf
tags: [pricing, reinsurance, experience-rating, burning-cost, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-13T12:00:00Z }
sources:
  - id: clark
    resource: https://www.casact.org/sites/default/files/2021-03/8_Clark.pdf
    title: Clark, Basics of Reinsurance Pricing, CAS study note (first version 1996, revised 2014), per-risk experience rating
  - id: tests
    resource: ../validation/tests/pricing.rs
    title: The experience rating parity tests (module clark_experience)
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

[^clark]: Clark, Basics of Reinsurance Pricing.
[^tests]: `validation/tests/pricing.rs`, module `clark_experience`.
