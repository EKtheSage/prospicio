### Excel

- `PRO.RES.BAYES.ODP` and `PRO.RES.BAYES.BF`: cells that ask for different
  views of one fit in the same recalculation now wait for one sampler run
  instead of each starting their own, and no longer push duplicate fits out of
  the cache. Results are unchanged (#954, #1014).
