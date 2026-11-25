---
type: Reference
title: The Solvency II counterparty default module for type 1 exposures
description: Delegated Regulation (EU) 2015/35 as amended, Articles 192, 196, 197, 199, 200 and 201; the loss given default of a reinsurance arrangement, the factors F, the probability of default by credit quality step and by solvency ratio, the variance V_inter plus V_intra and SCR_def,1; the reference for prospicio_aggregate::solvency2 (#980), whose reference test checks these formulas.
resource: https://www.eiopa.europa.eu/rulebook/solvency-ii-single-rulebook_en
tags: [solvency-ii, counterparty, reinsurance, credit-risk, capital]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-24T00:00:00Z }
verified: { by: process:ci, at: 2026-11-24T12:00:00Z }
sources:
  - id: eurlex2024
    resource: http://publications.europa.eu/resource/celex/02015R0035-20241114
    title: Delegated Regulation (EU) 2015/35, consolidated text of 2024-11-14 (Publications Office cellar, the text in force on 2026-11-24)
  - id: eurlex2027
    resource: http://publications.europa.eu/resource/celex/02015R0035-20270130
    title: Delegated Regulation (EU) 2015/35, consolidated text of 2027-01-30, after Delegated Regulation (EU) 2026/269 (M13)
  - id: art192
    resource: https://www.eiopa.europa.eu/rulebook/solvency-ii-single-rulebook/article-5845_en
    title: Delegated Regulation (EU) 2015/35, Article 192, loss given default (EIOPA single rulebook)
  - id: art197
    resource: https://legislation.gov.uk/eur/2015/35/article/197/2020-01-31
    title: Article 197 as substituted by Delegated Regulation (EU) 2019/981, the factors F, F', F'' and F'''
  - id: art199
    resource: https://www.legislation.gov.uk/eur/2015/35/article/199/2020-01-31
    title: Article 199, probability of default, as at 2020-01-31 (the EU text then in force, after 2019/981)
  - id: art200
    resource: https://www.eiopa.europa.eu/rulebook/solvency-ii-single-rulebook/article-5853_en
    title: Article 200, SCR on type 1 exposures
  - id: art201
    resource: https://www.eiopa.europa.eu/rulebook/solvency-ii-single-rulebook/article-5854_en
    title: Article 201, the variance of the loss distribution of type 1 exposures
  - id: qa3119
    resource: https://www.eiopa.europa.eu/qa-regulation/questions-and-answers-database/3119_en
    title: EIOPA Q&A 3119 on the factor F' after 2019/981
  - id: qa2530
    resource: https://www.eiopa.europa.eu/qa-regulation/questions-and-answers-database/2530_en
    title: EIOPA Q&A 2530 on V_inter terms with zero probabilities of default
  - id: note
    resource: ../docs/design/reinsurer-default.md
    title: The design note for reinsurer default on a tower (#980)
---

# What it is

The standard formula's capital charge for the default of counterparties
whose exposures are not diversified and are likely rated, such as
reinsurers (type 1 exposures). Read from the EIOPA single rulebook and
legislation.gov.uk on 2026-11-24, and every constant rechecked the same
day for #980 PR 6 against the EU consolidated texts of 2024-11-14 and
2027-01-30 (see "The check" below). `prospicio_aggregate::solvency2`
implements Arts 192(2), 199, 200 and 201 (#980), and
`validation/tests/aggregate_reinsurer_default_solvency2.rs` checks it, with
its Python and R bindings, against `reinsurer_default_solvency2.csv`, which
`validation/scripts/aggregate_reinsurer_default.py` computes from the
formulas below independently of the Rust (every credit quality step, the
ratio table's points, values between them and both clamps, the 50 % and
90 % shares with F of 100 % and 50 % and F' of 18 %, and type 1 portfolios
taking each of Art 200's three branches, with repeated and zero
probabilities), to 1e-12 relative.

# The numbers

- **Loss given default of a reinsurance arrangement (Art 192(2), first
  subparagraph)**:
  `LGD = max(50% (Recoverables + 50% RM_re) - F Collateral, 0)`, where
  Recoverables is the best estimate of the amounts recoverable and the
  corresponding debtors, `RM_re` the risk-mitigating effect on underwriting
  risk (Art 196: the underwriting SCR without the arrangement less the SCR
  with it) and Collateral the risk-adjusted value of the collateral (Art
  197: market value less a market-risk adjustment, 90% of it if only the
  third-party custody condition fails, nil if Art 214 is not met). For a
  counterparty with 60% or more of its assets under collateral
  arrangements the 50% is 90% and F is F' (Art 192(2), second
  subparagraph; that subparagraph is base text, and Delegated Regulation
  (EU) 2016/467 replaced its formula). From 2027-01-30 Delegated
  Regulation (EU) 2026/269 replaces that formula again: the factor in the
  90% case is F''' instead of F'.
- **The factors (Art 197(7) as substituted by 2019/981)**: all 100% when
  the undertaking's share of the insolvency estate above the collateral
  does not take account of receiving the collateral; otherwise F, F', F''
  and F''' are 50%, 18%, 16% and 90%. EIOPA Q&A 3119 notes that F' = 18%
  looks unintended (90% expected) and declined to answer it as a question
  on the legal text. Delegated Regulation (EU) 2026/269 makes the fix:
  from 2027-01-30 the 90% case uses F''' (100%, or 90%), so the factor
  there is F' (100% or 18%) until 2027-01-29 and F''' (100% or 90%) from
  2027-01-30. The code takes the factor as an argument, so it needs no
  change; the caller passes the factor for the date of the calculation.
- **Probability of default by credit quality step (Art 199(2))**: steps 0
  to 6 give 0.002%, 0.01%, 0.05%, 0.24%, 1.2%, 4.2% and 4.2%.
- **Unrated Solvency II undertakings meeting their MCR (Art 199(3))**: by
  solvency ratio, linearly interpolated between 196% (0.01%), 175% (0.05%),
  150% (0.1%), 125% (0.2%), 122% (0.24%), 100% (0.5%), 95% (1.2%) and 75%
  (4.2%); 0.01% at or above 196%; 4.2% at or below 75%; 4.2% if the MCR is not met.
  Before an undertaking's first solvency ratio is disclosed (Art 199(5)),
  the credit quality step applies if an ECAI assessment exists, and
  otherwise a 100% ratio (0.5%).
- **SCR (Art 200)**: with `sigma = sqrt(V)` (Art 200(4)) and the sum of
  the LGDs of all type 1 exposures, `3 sigma` if `sigma` is "lower than or
  equal to" 7% of that sum (Art 200(1)), `5 sigma` if above 7% and at most
  20% (Art 200(2)), else the sum (Art 200(3)). A unit test in
  `solvency2.rs` puts sigma exactly on 7% and on 20% in floating point and
  checks that the lower multiple applies there.
- **Variance (Art 201(1) to (3))**: `V = V_inter + V_intra`, with exposures grouped
  by probability of default `PD_j` and `TLGD_j` the sum of the LGDs in
  group `j`: `V_inter = sum over all ordered pairs (j, k), j = k included,
  of PD_j(1 - PD_j) PD_k(1 - PD_k) /
  (1.25 (PD_j + PD_k) - PD_j PD_k) TLGD_j TLGD_k` and `V_intra = sum over
  j of 1.5 PD_j (1 - PD_j) / (2.5 - PD_j) times the sum of the squared
  LGDs in group j` (Art 201(3); the rulebook's text writes the inner sum
  as "sum PD j (LGD^2 i)", `PD_j` being the sum's subscript, over the
  exposures whose probability is `PD_j`, not a factor). EIOPA Q&A 2530
  takes a term whose two probabilities are both 0 as 0, and from
  2027-01-30 Art 201(2)(a) as amended by 2026/269 says so: the sum
  excludes "the case where PD_j = PD_k = 0". Art 199(1) makes each entry
  a single name exposure (several counterparties of one group take the
  LGD-weighted average of their probabilities), and Art 201(3) squares
  each single name exposure's LGD, so splitting one reinsurer over several
  entries with the same probability understates `V_intra`.

# Recovery rates

A Lloyd's and Institute of Actuaries working party (Reinsurance Bad Debt
Provisions for General Insurance Companies, 2000, with a 2005 note) found
ultimate recoveries from failed reinsurers' estates mostly between 20% and
60%; the standard formula's 50% loss given default sits in that range.

# The check

On 2026-11-24 (#980 PR 6) every constant in
`crates/prospicio-aggregate/src/solvency2.rs` and
`validation/scripts/aggregate_reinsurer_default.py` was compared with the
consolidated text of 2024-11-14 (the text in force), and all match. EUR-Lex's
HTML and PDF pages did not load, so the XHTML of the consolidations
came from the Publications Office cellar (the `eurlex2024` and
`eurlex2027` sources, requested with `Accept: application/xhtml+xml`);
the formulas there are images, which were read one by one.

| Constant | Code | Text | Article |
|---|---|---|---|
| PD by step 0 to 6 | 0.00002, 0.0001, 0.0005, 0.0024, 0.012, 0.042, 0.042 | 0.002%, 0.01%, 0.05%, 0.24%, 1.20%, 4.2%, 4.2% | Art 199(2) |
| PD by ratio | 196% 0.01%, 175% 0.05%, 150% 0.1%, 125% 0.2%, 122% 0.24%, 100% 0.5%, 95% 1.2%, 75% 4.2%, linear between | the same table, linear interpolation, 4.2% below 75%, 0.01% above 196% | Art 199(3) |
| PD when the MCR is not met; before the first disclosure | documented: 4.2%; the step if rated, else a 100% ratio | the same | Art 199(4), (5) |
| LGD share and `RM_re` weight | 0.5 or 0.9; 0.5 | `max[50% (Recoverables + 50% RM_re) - F Collateral; 0]`, and `max[90% (...) - F' Collateral; 0]` when 60% or more of the counterparty's assets are collateralised | Art 192(2) |
| F, F' | caller's argument, documented as 100% or 50%, 100% or 18% (F''' of 100% or 90% from 2027-01-30) | 100% when the estate share ignores the collateral, else F, F', F'', F''' = 50%, 18%, 16%, 90% | Art 197(7) |
| SCR thresholds | `<= 0.07` gives 3 sigma, `<= 0.2` gives 5 sigma, else the total | 3 sigma at "lower than or equal to 7 %", 5 sigma above 7% and "lower or equal to 20 %", else the total; sigma the square root of V | Art 200(1) to (4) |
| `V_inter` | 1.25 in `1.25 (PD_j + PD_k) - PD_j PD_k`, every ordered pair, a pair of zero probabilities 0 | the same, "all possible combinations (j,k)" (from 2027-01-30 excluding PD_j = PD_k = 0) | Art 201(2) |
| `V_intra` | 1.5 and 2.5 in `1.5 PD_j (1 - PD_j) / (2.5 - PD_j)` times the sum of `LGD_i^2` | the same | Art 201(3) |

Amendments after 2020 that exist: Delegated Regulations (EU) 2021/1256,
2024/2765 and 2026/269. Of these only 2026/269 (published 2026-02-18,
applying from 2027-01-30) changes these articles: the 90% case's factor
(Art 192(2)) and the zero-probability exclusion (Art 201(2)(a)), both
described above.
