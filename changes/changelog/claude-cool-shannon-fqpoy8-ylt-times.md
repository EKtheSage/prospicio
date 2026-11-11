### Aggregate and reinsurance

- `read_ylt(day = , hour = )` takes its rule from one Rust function,
  `ylt::day_hour_times` (Python `prospicio.aggregate.day_hour_times`, R
  `day_hour_times()`), so Python and R check the same way (#838, #866): a day or
  hour that is missing or not a number is an error naming the row in both
  (R's `which()` used to drop it and Python's `float()` failed without the
  row; the user's decision of 2026-11-11).
