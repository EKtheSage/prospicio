---
type: Finding
title: A deal's marginal price on a book is at most its Euler price under TVaR, so pricing every deal on its marginal leaves the book short
description: "DealBook::marginal_price prices the book's loss with and without one deal, P = (EL + r dA) / (1 + r) under a cost of capital r with dA the TVaR the deal adds. TVaR(book without the deal) is at least the mean of the rest over the book's tail, so dA is at most the deal's co-TVaR (Euler) assets and the marginal premium at most its allocated one; the same holds for the premium under any concave distortion. On 20,000 simulations of three deals with a shared exponential driver and Pareto(2.5) noise, TVaR 99 assets and r = 10%, the marginal premiums are 2.050, 1.057 and 3.283 against Euler's 2.233, 1.207 and 3.560 and sum to 6.39 against the book's 7.00. The only deal in a book has its stand-alone price exactly."
tags: [prospicio-pricing, book, marginal-price, allocation, tvar]
status: draft
generated: { by: claude-code/cloud-session, at: 2026-10-28T12:00:00Z }
verified: { by: process:ci, at: 2026-10-28T12:00:00Z }
sources:
  - id: code
    resource: ../crates/prospicio-pricing/src/book.rs
    title: DealBook::marginal_price and its tests
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/588
    title: the marginal price of a new contract on a book
---

# The marginal price of a deal on a book

`DealBook::marginal_price(name, rule, assets)` answers what a deal must
charge, written last, so that the book's price still holds: the book's
loss priced with the deal and without it, by the `PremiumRule` and the
capital standard of `price_portfolio`, and the differences.

Under a cost of capital `r` the difference of the two prices is
`(EL + r dA) / (1 + r)`, with `EL` the deal's expected loss and `dA` the
assets it adds. TVaR is subadditive and, on draws, `TVaR(rest)` is at
least the mean of the rest over the book's tail, which is the book's TVaR
less the deal's co-TVaR. So `dA` is at most the deal's Euler assets and the
marginal premium at most its Euler premium. The unit test
`the_marginal_price_is_at_most_the_euler_price_under_tvar` checks this on
2,000 simulations under a cost of capital and under a TVaR pricing
distortion, and `the_only_deal_has_its_standalone_price` the one-deal
book; CI runs both.

On 20,000 simulations of three deals with a shared exponential driver
and Pareto(2.5) noise (TVaR 99 assets, `r = 0.10`):

| deal | marginal P | Euler P | dA | Euler assets |
|---|---|---|---|---|
| a | 2.050 | 2.233 | 5.705 | 7.720 |
| b | 1.057 | 1.207 | 2.976 | 4.624 |
| c | 3.283 | 3.560 | 9.174 | 12.218 |

The marginal premiums sum to 6.39 against the book's 7.00. Price the next
deal on its marginal and the book's return holds; price every deal on its
marginal and the book falls about 9% short.
