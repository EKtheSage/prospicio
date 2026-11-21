### Pricing

- `ScheduleTrend::factors` (and `schedule_trend` in Python and R) rejects an
  inception or accident year more than 500 years from the schedule's years,
  such as `1e17` or a date serial like `20260101`, with an error that names
  the value; before, it hung or returned `inf`. In R, a year beyond the
  integer range is an error, not `NA` (#950, #1015).
