### Aggregate and reinsurance

- An aggregate (attritional) part on a timeline: `Timeline` keeps it by
  accident period (each accident year of `Timeline::from_years`, the whole
  term of `Timeline::new`, which no longer refuses it) and
  `Timeline::with_aggregate_pattern` spreads it over the period's months by
  twelve monthly weights, uniform by default; each month's amount falls to
  the contract years as an event in that month would, so each year's quota
  share or stop loss sees its part. Python `aggregate_pattern=` on
  `Timeline` and `Timeline.from_years`, R `aggregate_pattern =` on
  `timeline()` and `timeline_from_years()` (#540).
