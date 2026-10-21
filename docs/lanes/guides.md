# Guides lane

**Session:** Local (the guides session). **Owns:** `examples/`, `docs/guides/`, `python/tests/test_guides.py`, `R/prospicio/tests/test-guides.R`, `test_reinsurance_pricing` and future notebook tests in `python/tests/test_examples.py`, the `*.ipynb` lines in `.gitattributes`

Knowledge, documentation, guides and how-tos for actuaries.

This lane writes about the library and does not change crates; a gap it finds becomes an issue for the lane that owns the code

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

## Next

- [guides-open] open, waiting on the user
- keep the guides and notebooks current as the library changes, and update them when #192 lands

## Done

- `examples/reinsurance_pricing.ipynb` (#186)
- a submission evaluation mapped onto the Python API, with its gaps filed for the owning lanes (#190 to #197)
- `examples/submission_evaluation.ipynb`, that evaluation run end to end on a synthetic submission (`examples/data/make_submission.py` writes its files, `test_submission_evaluation` runs it)
- the guide `docs/guides/reinsurance-price.md`, what a reinsurance price is and how a reinsurer arrives at it, with every Python block run and every printed number checked in CI (`python/tests/test_guides.py`)
- six "when to use what" guides: `choosing-a-method.md`, `risk-loads-and-distortions.md`, `experience-and-exposure-rating.md`, `aggregate-distributions.md`, `severity-and-frequency.md` and `reserving-methods.md`
- notebooks committed without outputs (an nbstripout git filter keeps local runs out of commits), executed by the docs site build and published there with the guides in the Python site's User Guide (`cargo xtask python` stages both; the full run happens only in the Pages build on `main`, so check that run after merging a notebook or guide)
- The guides and notebooks use the features from #190, #191, #195, #196, #197 and #204 to #206 (large-loss preparation and as-if history, payment patterns and discounting, event loss tables, the quote, `price` on a grid, distortion `.family` and `.parameters`, towers on an annual grid), and those from #193, #194 and #211 to #214 (`PredictiveDistribution.select`, `fit_severity` with `goodness_of_fit`, `TowerModel.diagnostics` and `alpha_bounds`, `TabulatedCurve.from_severity`, the Swiss Re `c` calibrations, `layer_cv`, `PolicyLimits`, `Pareto.fit_with_standard_error`, `RiskProfile`'s `spread`, the exponential calibration)
- R twins of the first two notebooks as Quarto documents (`examples/*.qmd`, numbers equal to the notebooks; not run by CI yet)
- The R twin of the guide for `aggregate` users (`docs/guides/prospicio-for-aggregate-users.qmd`, `docs/guides/capital-modeling-monograph-15.qmd`), every printed number checked by `R/prospicio/tests/test-guides.R` in CI's R job
- Done too (2026-10-17): `prospicio-for-aggregate-users.md` (the aggregate 1.2.0 Twenty Minutes session run in prospicio, numbers beside aggregate's) and `capital-modeling-monograph-15.md` (CAS Monograph 15 chapter 9: InsCo, five distortions, the 35 xs 65 stop loss, the Iman-Conover book; premiums match, lifted capital under CCoC and PH differs by aggregate version), each with an R twin
- the coverage map `knowledge/references/aggregate-docs-coverage.md`
- source texts kept locally in `knowledge/references/` (PDFs are gitignored; never commit them)
- Done too (2026-10-17), each as a Python guide with an R twin, source printed values beside prospicio's: `pricing-insurance-risk-case-studies` (the Discrete, Tame, Cat/NonCat and Hurricane/SCS case studies, five distortions, against aggregate 0.30.1), `distributions-for-actuaries-claim-size` and `-counts-aggregates` (CAS Monograph 2, Bahnemann), `ratemaking-werner-modlin` and `-indications` (Basic Ratemaking, all rounding differences tabulated), `reserving-friedland` (Estimating Unpaid Claims), `reserving-taylor-mcguire` (Monograph 3) and `reserving-meyers` (Monograph 8; seeded NUTS digits differ by platform and toolchain, so checked blocks print the book value, a tolerance and agreement, and the heavy fits are `no-run` with recorded numbers)
- Gaps found went to issues for the owning lanes (#328 to #331 handed over; #333, #334, #336, #338, #339, #342 open)
- #322 and #327 were fixed in the guides session with the user's go-ahead
- Then `glm-ratemaking-goldburd` and `-validation` (CAS Monograph 5, GLMs for insurance rating; run on the repo's synthetic policy data because the monograph's data is not in the PDF, so only its closed-form examples are compared) and `glm-to-comprehensive-pricing-chalk` and `-2` (CAS Monograph 16; seeded 20,000-policy portfolio, boosting compared after rounding; the R twins read `examples/data/chalk_portfolio.csv` written by `make_chalk_portfolio.py`), each with an R twin
- their gaps are issues #350 to #353, #355 to #358 and #360 for the Models lane
- Then (2026-10-18) the Pricing Insurance Risk guides (`pir-risk-measures`, `pir-portfolio-pricing`, `pir-price-allocation`, `pir-price-allocation-practice`; Parts I to III of the book) and Monograph 15 chapters 3 to 6 (`capital-modeling-monograph-15-theory`, `-applications`), each with an R twin, and `docs/guides/reading-list.md` mapping every source text to its guides, knowledge file and open issues (keep it current as guides merge; its page has no code, `PROSE_PAGES` in `python/tests/test_guides.py`)
- Gaps went to issues #365 to #369, #373 and #375 for the Aggregate and Probability lanes
- [guides-sync-0.0.4] (2026-10-21) stale gap passages updated now that their issues closed, in `reserving-taylor-mcguire`, `glm-ratemaking-goldburd-validation`, `glm-to-comprehensive-pricing-chalk-2` and `pir-price-allocation`, each with its R twin (`OdpGlm`, `GlmFit` diagnostics and quantile residuals, the explanation helpers, the elastic net bounds, MARS, GLMMs, `bodoff_price`), and the reading list's open-issues column (only #192 is open)
