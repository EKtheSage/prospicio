# Excel lane

**Session:** Cloud (the Excel session). **Owns:** `crates/prospicio-xll/`, `docs/design/xll.md`, the `xll` task in `xtask/src/main.rs`, `.github/workflows/xll.yml`

This lane wraps other lanes' crates and changes none of them

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

## Next

- [excel-more-functions] further functions as the user asks
- [code-signing-certificate] a code-signing certificate is the user's to buy

## Done

- the XLL prototype for 64-bit Excel (`xll.md`): the chain ladder and Mack as worksheet functions (`PRO.CL`, `PRO.CL.LDF`, `PRO.CL.CDF`, `PRO.CL.ULTIMATE`, `PRO.CL.RESERVE`, `PRO.CL.FACTORS`, `PRO.MACK`, `PRO.VERSION`) over `prospicio-reserving`, built by `cargo xtask xll` from Windows or Linux and smoke-tested in a fake Excel under Wine
- handed to the user as a local file and tested in Excel 365
- the `PRO.` prefix and text errors confirmed by the user (#278)
- layer pricing (`PRO.LEV`, `PRO.LAYER`, `PRO.LAYER.CV`, `PRO.ILF`) on a Pareto or lognormal severity, limits and attachments broadcast into tables (`xll.md` decisions 12 and 13; #281)
- the ODP bootstrap exhibit (`PRO.ODP`: mean, SD, CV and percentiles per origin and in total, reproducible by seed; RAA's total within R `BootChainLadder`'s targets; decision 14; #283)
- distribution: `.github/workflows/xll.yml` builds and smoke-tests the add-in on every PR that touches it, keeps it as a CI artifact and attaches it to a draft GitHub release on a version tag, signing it when a certificate secret is set (decisions 15 and 16)
- the bootstrap's draws (`PRO.ODP.SIMS`, the same draws `PRO.ODP` summarizes)
- Mack's bootstrap (`PRO.MACK.BOOT`, `PRO.MACK.BOOT.SIMS`; its total SD reconciles with Mack's on RAA)
- the expected-loss methods (`PRO.EL`, `PRO.BF`, `PRO.BENKTANDER`, `PRO.CAPECOD` on a premium column; chainladder-python parity on GenIns)
- Mack's options (`sigma`, R parity) and selected factors (`selected`, `kind`, `weight` on the chain-ladder functions and Mack; chainladder-python `DevelopmentConstant` parity)
- the one-year view (`PRO.MACK.CDR`, `PRO.MACK.CDR.YEARS`: R `CDR()` parity; `PRO.ODP.CDR`, `PRO.MACK.BOOT.CDR`)
- Clark (`PRO.CLARK`, `PRO.CLARK.PARAMS`: LDF and Cape Cod, R parity)
- triangle helpers (`PRO.TRI.INCREMENTAL`, `.CUMULATIVE`, `.LINKRATIOS`, `.LATEST`). severities over every parametric family (`PRO.SEV.CDF`, `.SURVIVAL`, `.QUANTILE`, `.STATS`, `PRO.LER`, and LEV, layer, CV and ILF for all; mpmath parity)
- the Pareto helpers (`PRO.PARETO.EXTRAPOLATE`, `PRO.PARETO.ALPHA`, `.ALPHA.FREQLAYER`, `.ALPHA.FREQS`; R Pareto parity)
- the piecewise Pareto and Riegel's generalized Pareto as families and `PRO.LAYER.VAR` (R Pareto parity)
- exposure curves (`PRO.EXPOSURE.CURVE`, `.SHARE`, `.QUANTILE`, `.STATS`, `PRO.SWISSRE.C`, `.C.LAYER`: MBBEFD, Swiss Re and tabulated curves, mpmath parity; decision 18)
- an exposure-rated risk profile (`PRO.PROFILE`: per-band curves, spread bounds, a surplus underneath)
- classical credibility (`PRO.CRED.STANDARD`, `PRO.CRED.Z`, `PRO.BUHLMANN`, `.PARAMS`; actuar parity on Hachemeister)
- rate level and trend (`PRO.ONLEVEL`, `PRO.RATELEVEL`, `PRO.AVGDATE`, `PRO.TREND.FIT`, `PRO.YEARFRACTION`; Werner and Modlin parity; Excel dates or year fractions, decision 19)
- the indication (`PRO.LR.EXPERIENCE`, `PRO.PP.EXPERIENCE`, `PRO.INDICATED.CHANGE`, `.RATE`, `PRO.PERMISSIBLE.LR`, `PRO.TRENDED.PRESENT`, `PRO.CRED.WEIGHTED`, `PRO.EXPENSES`; Werner and Modlin Appendices A to C end to end)
- the burning cost (`PRO.BURNINGCOST`: Clark parity, several layers with aggregate terms)
- aggregate distributions (`PRO.AGG.STATS`, `PRO.AGG.LAYERS`: `aggregate` parity; decision 20)
- the Bayesian ODP chain ladder by MCMC (`PRO.BAYES.ODP`: NUTS, reserves, posterior and diagnostics of one kept fit; Stan parity on RAA and GenIns; decision 21)
- the occurrence and aggregate PML curves (`PRO.AGG.PML`: by return period, any count; `aggregate` `oep` parity; decision 22)
- aggregate covers on the annual total (`PRO.AGG.COVER`: a stop loss ground up, net of occurrence layers or on their ceded total; `aggregate` `reins_stats_df` parity; decision 23; #404)
- towers priced on a grid (`PRO.TOWER.PRICE`: gross, layers with annual terms, a stop loss on the net and the net, each view priced by a distortion; `aggregate` `reins_price_df` parity; decision 24; #406)
- The parity plan (user, 2026-10-16) is done
- `PRO.BAYES.ODP`'s `prior_sd` help warns that a tight prior lifts the late columns
- Severity fits to large losses (`PRO.SEV.FIT`: body families by maximum likelihood with reporting thresholds and censoring, the Pareto families by R Pareto's estimators, P-P and Q-Q against the product-limit estimate, the body families compared, a note on a fit at a boundary; R Pareto parity; decision 25; #413)
- EVT (`PRO.EVT.THRESHOLDS`, `PRO.EVT.FIT`, `PRO.EVT.HILL`: threshold diagnostics, the GPD tail with its VaR and TVaR, Hill's index; `ismev` parity; decision 26; #415)
- the time value of money (`PRO.PAYOUT.PATTERN`, `.STATS`, `.PRICE`: a pattern from shares, paid to date or a chain ladder's factors, its discount factor, duration and capital years, a cost-of-capital price over the payout; decision 27; #416)
- a cover's quote (`PRO.QUOTE`: the technical premium grossed up for expenses and brokerage, its build-up and market ratios, the cover at quoted premiums; decision 28; #417)
- distortions (`PRO.DISTORTION`, `.PRICE`, `.CALIBRATE`: g(s), the price of outcomes or draws, each family's parameter for a target premium; `aggregate` parity; decision 29; #418)
- contract terms (`PRO.SWING`, `PRO.SLIDING.SCALE`, `PRO.PROFIT.COMMISSION` at a loss, `PRO.LAYER.TERMS` in expectation on the grid; `aggregate` parity; decision 30)
- the distortions with several parameters (bitvar, beta, wtdtvar: their parameters in one range in `PRO.DISTORTION`, `.PRICE` and `PRO.TOWER.PRICE`; `aggregate` parity; decision 31; #428)
- copulas (`PRO.COPULA.CDF`, `PRO.COPULA.SIM`, `PRO.IMAN.CONOVER`: Gaussian, t and Archimedean cdfs and uniforms, simulated columns reordered to a rank correlation; mpmath parity; decision 32; #429)
- multi-year contract terms (`PRO.PROFIT.COMMISSION.TERM`, `PRO.SLIDING.SCALE.TERM`: one term a row of losses, deficits and credits carried forward oldest first, for a number of years or until used up; checked against `Tower::apply_aggregate_term`; decision 33)
- The parity plan's seven groups are done
- [excel-exclusions] link-ratio exclusions and the geometric average in the `average` argument (a row `kind, n_periods, drop_high, drop_low` in the sixteen chain-ladder, Mack, expected-loss and Mack bootstrap functions; a last `average` on `PRO.ODP`, `.SIMS`, `.CDR`; chainladder-python and R ChainLadder parity; decision 34)
- [excel-odp-glm] the ODP GLM on a sheet: `PRO.ODP.GLM` (Taylor and McGuire's Table 5-4: reserves, process and parameter SE, RMSEP), `PRO.ODP.GLM.PARAMS` (coefficients, SEs, dispersion), `PRO.ODP.GLM.RESIDUALS` (by cell, or grouped by origin, development or calendar period); R `glm` parity on GenIns and ABC (decision 35)
- [excel-severity-fits-2] the newer severity fits: the Lomax in `PRO.SEV.FIT` and the severity functions, truncation from above (`upper`), `PRO.SEV.FIT.GROUPED` (maximum likelihood or minimum chi-square on size bands) and `PRO.SEV.FIT.LEV` (minimum limited-mean distance); Bahnemann parity (decision 36)
- [excel-workbook-coverage] the test workbook calls all 96 functions with a check beside each (the twelve it left out added, the ODP sheet's P95 checks fixed); `PRO.EXPOSURE.CURVE` and `.QUANTILE` tabulate one curve per c (or b, g) in a row or column (decision 37)
- [excel-bayes-bf] the Bayesian Bornhuetter-Ferguson by MCMC (`PRO.BAYES.BF`: `BayesBfNormalised`, a gamma prior on each origin's ultimate at apriori times exposure, `cv` or `"flat"`, `common_cv`; reserves, ultimates, posterior and diagnostics of one kept fit; the flat limit at `PRO.BAYES.ODP`, the tight one at `PRO.BF`; decision 38)
- [excel-diagonal] a latest diagonal on a sheet (#544, decision 39, branch `claude/excel-diagonal`): `PRO.DIAG.CL`, `.EL`, `.BF`, `.BENKTANDER` and `.CAPECOD` project origins (years or Excel dates) at their ages or a valuation date with a selected pattern (values and ages, `kind`, `tail`), the origin grain, `past_end` (reject, none, tail) and `pattern_origin` (a pattern of accident months averaged to the origins' grain), and `PRO.PATTERN.AVERAGE` shows that averaging; a triangle's pattern is frozen through `PRO.CL.LDF`; the hand diagonal, chainladder-python's `DevelopmentConstant` on GenIns and the CAS workers' compensation triangle at 1e-9 and the Reserving lane's past-end and averaging cases reproduced, each exhibit equal to the crate's fit to the bit
- [excel-diagonal-details] decision 39's last three details accepted by the user on 2026-10-25 as built: `PRO.PATTERN.AVERAGE`'s `grain` defaults to `"year"`, a `pattern_origin` equal to `grain` is an error there, and a tail of exactly 1 counts as no tail (#544)
- [excel-triangle-pattern-ages] #635 and #680 (0.0.7, branch `claude/xll-el-selected-ages`, decision 40): `selected`, `kind` and `weight` on `PRO.EL`, `PRO.BF`, `PRO.BENKTANDER` and `PRO.CAPECOD`; `pattern_ages` on the thirteen functions that take `selected`, so a monthly pattern reaches a triangle; `apriori` one per origin on the triangle and diagonal expected-loss functions; chainladder-python `DevelopmentConstant` parity on the CAS workers' compensation triangle, the crate's fits to the bit off the triangle's ages
