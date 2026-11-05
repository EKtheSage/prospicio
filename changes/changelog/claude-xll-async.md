### Excel

- The bootstraps (`PRO.RES.ODP`, `PRO.RES.MACK.BOOT` and their `.SIMS` and
  `.CDR`), the Bayesian fits (`PRO.RES.BAYES.ODP`, `PRO.RES.BAYES.BF`) and
  `PRO.CAP.COPULA.SIM` run in the background as Excel asynchronous
  functions: their cell shows `#GETTING_DATA` until the result arrives,
  and Excel stays usable. Cells with the same inputs share one run, and
  the last 32 results are kept, so a recalculation that changes none of a
  function's inputs answers at once with the same numbers
  (`docs/design/xll.md` decision 48; #748).
