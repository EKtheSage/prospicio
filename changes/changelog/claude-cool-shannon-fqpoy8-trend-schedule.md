### Pricing

- Trend factors to a treaty period from a calendar-year rate schedule:
  `ScheduleTrend` over a `RateSchedule` in `prospicio_pricing::ratemaking`,
  losses or premium, risks attaching or losses occurring, an incomplete
  latest year, a calendar year not in the schedule at the latest year's
  rate; Python `schedule_trend`, R `schedule_trend()` (#801).
- The running premium-weighted loss ratio, `cumulative_loss_ratio` in
  Rust, Python and R (#806).

### Excel

- `PRO.PRC.TREND.FACTORS` (trend factors by a calendar-year rate
  schedule), `PRO.PRC.LR.CUMULATIVE` (the running premium-weighted loss
  ratio) and `PRO.REVERSE` (a row reversed into a column, padded or cut
  to a length) (#803, #806).
