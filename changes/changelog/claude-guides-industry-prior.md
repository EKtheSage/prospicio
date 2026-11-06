### Guides

- `docs/guides/reserving-industry-prior.md` and its R twin: an industry prior from a Schedule P line fetched at run time, fitted to a thin program and tested on every company of four lines held out in turn (#730).
- `examples/ils_deal_analysis.ipynb` and its R twin build the prior with `IndustryPrior` from synthetic peer triangles in place of the credibility estimate, draw the 2026 loss ratio with `new_year` in place of a lognormal, and pay each simulation on its own pattern (#730).
