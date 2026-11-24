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
  - id: art192
    resource: https://www.eiopa.europa.eu/rulebook/solvency-ii-single-rulebook/article-5845_en
    title: Delegated Regulation (EU) 2015/35, Article 192, loss given default (EIOPA single rulebook)
  - id: art197
    resource: https://legislation.gov.uk/eur/2015/35/article/197/2020-01-31
    title: Article 197 as substituted by Delegated Regulation (EU) 2019/981, the factors F, F', F'' and F'''
  - id: art199
    resource: https://www.legislation.gov.uk/eur/2015/35/article/199/adopted
    title: Article 199, probability of default
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
legislation.gov.uk on 2026-11-24. `prospicio_aggregate::solvency2`
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

- **Loss given default of a reinsurance arrangement (Art 192(2))**:
  `LGD = max(50% (Recoverables + 50% RM_re) - F Collateral, 0)`, where
  Recoverables is the best estimate of the amounts recoverable and the
  corresponding debtors, `RM_re` the risk-mitigating effect on underwriting
  risk (Art 196: the underwriting SCR without the arrangement less the SCR
  with it) and Collateral the risk-adjusted value of the collateral (Art
  197: market value less a market-risk adjustment, 90% of it if only the
  third-party custody condition fails, nil if Art 214 is not met). For a
  counterparty with 60% or more of its assets under collateral
  arrangements the 50% is 90% and F is F' (2016/467).
- **The factors (Art 197(7) as substituted by 2019/981)**: all 100% when
  the undertaking's share of the insolvency estate above the collateral
  does not take account of receiving the collateral; otherwise F, F', F''
  and F''' are 50%, 18%, 16% and 90%. EIOPA Q&A 3119 notes that F' = 18%
  looks unintended (90% expected) and declined to answer it as a question
  on the legal text.
- **Probability of default by credit quality step (Art 199(2))**: steps 0
  to 6 give 0.002%, 0.01%, 0.05%, 0.24%, 1.2%, 4.2% and 4.2%.
- **Unrated Solvency II undertakings meeting their MCR (Art 199(3))**: by
  solvency ratio, linearly interpolated between 196% (0.01%), 175% (0.05%),
  150% (0.1%), 125% (0.2%), 122% (0.24%), 100% (0.5%), 95% (1.2%) and 75%
  (4.2%); 0.01% at or above 196%; 4.2% at or below 75%; 4.2% if the MCR is not met.
  Before an undertaking's first solvency ratio is disclosed (Art 199(5)),
  the credit quality step applies if an ECAI assessment exists, and
  otherwise a 100% ratio (0.5%).
- **SCR (Art 200)**: with `sigma = sqrt(V)` and the sum of the LGDs of all
  type 1 exposures, `3 sigma` if `sigma` is at most 7% of that sum, `5 sigma`
  if at most 20%, else the sum.
- **Variance (Art 201)**: `V = V_inter + V_intra`, with exposures grouped
  by probability of default `PD_j` and `TLGD_j` the sum of the LGDs in
  group `j`: `V_inter = sum over all ordered pairs (j, k), j = k included,
  of PD_j(1 - PD_j) PD_k(1 - PD_k) /
  (1.25 (PD_j + PD_k) - PD_j PD_k) TLGD_j TLGD_k` and `V_intra = sum over
  j of 1.5 PD_j (1 - PD_j) / (2.5 - PD_j) times the sum of the squared
  LGDs in group j`. EIOPA Q&A 2530 takes a term whose two probabilities
  are both 0 as 0.

# Recovery rates

A Lloyd's and Institute of Actuaries working party (Reinsurance Bad Debt
Provisions for General Insurance Companies, 2000, with a 2005 note) found
ultimate recoveries from failed reinsurers' estates mostly between 20% and
60%; the standard formula's 50% loss given default sits in that range.
