### Examples

- `one_year_view` (notebook and R twin) models reserve risk as the one-year
  claims development result (`OdpBootstrap.one_year`, R `odp_one_year()`), not
  the run-off reserve; places the per-risk XL at the premium it prices (TVaR
  99% plus a 10% cost of capital, deposit plus expected reinstatements); and
  charges the net view what the reinsurance costs (ceded premiums less the
  surplus's ceding commission, plus reinstatement premiums), reporting the TVaR
  saving net of cost and the cost against it (#266, #267, #268).
