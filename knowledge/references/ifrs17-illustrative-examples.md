---
type: Reference
title: IFRS 17's Illustrative Examples
description: The IFRS Foundation's worked examples accompanying IFRS 17, the parity targets of the measurement of a group of contracts (GMM, PAA, reinsurance held); Examples 1, 2, 3, 6, 7 and 8 were read, round to whole units at each step, and ifrs17::Group matches 1, 2, 3A, 6, 7 and 8 in CI; Example 15's text could not be reached from a cloud session.
resource: https://www.ifrs.org/content/dam/ifrs/publications/html-standards/english/2025/issued/ifrs17-ie.html
tags: [reserving, ifrs17, csm, loss-component, parity]
status: stable
verified: { by: process:ci, at: 2026-11-28T03:00:00Z }
generated: { by: claude-code/cloud-session, at: 2026-11-27T15:00:00Z }
sources:
  - id: ie
    resource: https://www.ifrs.org/content/dam/ifrs/publications/html-standards/english/2025/issued/ifrs17-ie.html
    title: IFRS Foundation, IFRS 17 Insurance Contracts, Illustrative Examples (2025 HTML edition), read on 2026-11-27 up to IE98
  - id: note
    resource: ../docs/design/reserving-ifrs17-measurement.md
    title: The design note that proposes them as parity targets (#982)
---

# What it is

The examples the IASB publishes with IFRS 17, free on ifrs.org. Their
index (paragraph ranges): 1 initial recognition (IE4 to IE11), 2
subsequent measurement (IE12 to IE28; 2A favourable, 2B onerous), 3
presentation in profit or loss (IE29 to IE41), 4 and 5 separating
components, 6 further CSM features (IE56 to IE71), 7 acquisition cash
flows (IE72 to IE80), 8 the reversal of losses in an onerous group (IE81 to
IE98), 9 direct participation (IE99 to IE112), 10 the PAA (IE113 to IE123),
11 reinsurance held at initial recognition (IE124 to IE129), 12A to 12C its
subsequent measurement and onerous underlying groups (IE130 onward), 13 and
14 transfers and business combinations, 15 the systematic allocation of
insurance finance income or expense (IE152 to IE172), 16 underlying items,
17 and 18 the modified retrospective approach.[^ie]

# Figures read

* **Example 1** (three years, premium 900 at once, 5%, RA 120): outflows of
  200 a year have a present value of 545, fulfilment cash flows of -235 and
  a CSM of 235 (1A); outflows of 400 a year, 1,089, fulfilment cash flows of
  309, no CSM and a loss of 309 (1B).
* **Example 2**: the CSM of 235 accretes 12 and releases 82 (247 / 3) in
  year 1, closing at 165; in year 2, 2A accretes 8, adds 68 for future
  service and releases 121 (closing 120), and year 3 accretes 6 and releases
  126; 2B accretes 8 and loses 173 for future service, so the CSM is nil and
  a loss of 113 is recognised. Year 1's liability reconciliation: cash in
  900, finance expense 39 (27 on the present value, 12 on the CSM), current
  service -122, cash out -200, closing 617.
* **Example 3**: insurance revenue 222, 261 and 196 (3A); 3B's revenue
  totals 682 (900 + 82 - 300).
* **Example 6** (IE56 to IE71): 200 contracts, a single premium of 3,000,
  deaths at the end of each year and the account at maturity, discounted
  at the current 10% (7% from the end of year 2), the CSM accreting at the
  4% of nominal cash flows (B72(b)). Recognition: PV (404), RA 30, CSM 374.
  Year 2: the change in financial assumptions (the return falls to 7%)
  goes to finance, the discretion (61 more at maturity) adjusts the CSM at
  the locked-in 10% (55 = 61 / 1.1), and the 2 between 57 at 7% and 55 is
  finance (B97(a)): IFE 197 and 10, CSM release 107 = 213 / 394 × 198,
  closing 3,044 / 10 / 106 / 3,160. `ifrs17::Group` matches it with a
  financial revision of the opening view, the current curve and an
  accretion curve.
* **Example 10** (the premium allocation approach): not read, out of
  reach for the same reason as Example 15; `ifrs17::paa` is tested on
  cases worked by hand until a local session reads it.
* **Examples 11 and 12A to 12C** (reinsurance held): not read, out of
  reach for the same reason; `ifrs17::reinsurance` is tested on cases
  worked by hand until a local session reads them.
* **Example 15** (IE152 to IE172, the allocation of finance income or
  expense under the OCI option): not read. The fetched page's text stops at
  IE70 and `www.ifrs.org` is refused to a cloud session's shell, so its
  parity waits for a local session.
* **Example 7**: acquisition cash flows recovered at 30 a year beside a CSM
  release of 65 a year; revenue 300 a year.
* **Example 8**, year 1 (premium 800, outflows of 400 a year, 5%, RA 240):
  fulfilment cash flows and loss of 529; the loss component's share of the
  changes 529 / 1,329 = 39.8%; finance expense 54 (33 on the LRC excluding
  the loss component, 21 on the loss component); revenue 289; a loss
  component of 359 and a liability of 903 at the end of the year.

Every figure rounds to whole units, and the examples chain on the rounded
values, so a test matches them to the unit. Examples 8's later years, 9 to
18 were not read; a summary of them was not reliable (its Example 10
figures contradicted each other), so read the page itself before using
them.[^note]

[^ie]: IFRS Foundation, *IFRS 17 Illustrative Examples*, 2025 edition.
[^note]: `docs/design/reserving-ifrs17-measurement.md`, decision 11.
