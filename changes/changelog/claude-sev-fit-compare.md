### Breaking changes

- Excel: `PRO.SEV.FIT(..., "compare")` with `t` blank now has seven
  body-family rows (the loglogistic and exponential after the Weibull), so
  a legacy array formula over a fixed range of six rows misses the Burr
  and Lomax; a spilled formula needs two free rows more (#802, #863).

### Excel

- `PRO.SEV.FIT` fits the loglogistic and exponential, gives the Pareto
  families' log-likelihood, AIC and BIC, and with a `t` its `"compare"`
  adds the Pareto and Riegel's generalized Pareto and ranks all nine
  families by AIC; a range of names in `family` restricts the comparison.
  The severity functions take `"exponential"` (`p1` the mean) (#802, #863).
