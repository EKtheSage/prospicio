---
type: Finding
title: In the one-year view example, the reserve's one-year risk is 80% of its run-off spread, the XL's deposit follows its risk-loaded price in one step, and the reinsurance costs 5.8% of the TVaR it saves
description: "examples/one_year_view.ipynb at 20,000 simulations and its seeds. GenIns re-reserved on the ODP bootstrap (OdpBootstrap.one_year with the chain ladder) has a CDR sd of 2.41m and a 1-in-200 adverse development of 7.44m, against the run-off view's 3.00m and 9.14m above its mean. The per-risk XL's reinstatement premium is linear in its deposit (0.0747 per unit), so the deposit that makes deposit plus expected reinstatements equal the TVaR 99% plus 10% cost-of-capital price (0.81m) is price / 1.0747 = 0.75m, exact without iteration. Net of what it costs (0.46m a year: premiums and reinstatements less commission and expected recoveries), the reinsurance cuts the company's TVaR 99% by 8.06m; leaving the cost out reads 10.67m."
tags: [examples, one-year, cdr, odp, reinsurance, reinstatements, capital, tvar]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-22T18:00:00-07:00 }
verified: { by: process:ci, at: 2026-10-22T18:00:00-07:00 }
sources:
  - id: notebook
    resource: ../examples/one_year_view.ipynb
    title: The one-year view notebook and its R twin examples/one_year_view.qmd
  - id: test
    resource: ../python/tests/test_examples.py
    title: test_one_year_view, which asserts each relation below at 20,000 simulations
  - id: issues
    resource: https://github.com/EKtheSage/prospicio/issues/266
    title: "Issues #266, #267 and #268: one-year reserve risk, reinsurance cost in the net view, the XL's premium"
---

# Facts

* The reserve line is minus the claims development result of `OdpBootstrap(n_sims=20000, seed=1).one_year(tri, "paid", ChainLadder())` on GenIns paid: opening reserve 18.68m, adverse development mean 0.10m, sd 2.41m, 99.5% 7.44m. The run-off view (`fit`, same seed) has sd 3.00m and a 99.5% 9.14m above its mean, so the one-year sd is 0.80 of the run-off sd.
* With the reserve at its one-year change and the premium lines at their full losses, the reserve's CoTVaR 99% is 2.33m gross and 4.62m net of a 59.33m and 51.27m total; motor (31.35m, 34.15m) dominates. Under the run-off reserve the reserve took 22.00m gross.
* A paid reinstatement premium is `rate * premium * (limit used / limit) * (1 - t)`, linear in the deposit, and the ceded losses do not depend on the premium. A run at deposit 1 measures the rate per unit (0.0747), and `deposit = price / (1 + rate)` makes deposit plus expected reinstatements equal the risk-loaded price to rounding, with no iteration. R's `apply_tower` gives the same 0.0747.
* A surplus treaty cedes the same share of each risk's premium as of its losses, so at an original loss ratio `lr` its ceded premium is its exposure-rated loss over `lr` (`Layer.surplus(...).with_premium_rate(1.0, loss / lr)`); the tower reports the commission as `("ceding_commission", "surplus")`.
* A fixed premium adds in full to TVaR, so leaving the reinsurance's cost out of the net view overstates the TVaR saving by about what the cedant pays (2.61m against 2.42m paid). Net of cost the saving is 8.06m and the expected cost 0.46m, 5.8% of it, below the 10% cost of capital the XL is priced at. The 60% loss ratio and 35% commission are the example's assumptions.
* The R twin prints every one of these numbers to the same digits (rendered with R 4.5.3 on Windows on 2026-10-22).
