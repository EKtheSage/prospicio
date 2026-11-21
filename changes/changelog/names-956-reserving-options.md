### Reserving

- The rule past the pattern's end takes the same names in Python, R and Excel, from `PastPatternEnd::NAMES`: `reject`, `no_development` (or `none`) and `tail`. Python and R now accept Excel's `none`. No spelling is removed (#956, #1030).
- Clark's and the benchmark curve's growth `curve` takes the same names in Python, R and Excel, from `GrowthCurve::NAMES`: `loglogistic` (or `log-logistic`, `log_logistic`) and `weibull`. Python and R now accept Excel's two extra spellings. No spelling is removed (#956, #1030).
- Meyers' `model` (`crc`, `csr`, `cay`) is read in any case in Python and R, from `MeyersModel::from_name`. R was exact before; Python already took any case (#956, #1030).
