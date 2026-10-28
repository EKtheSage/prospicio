### Aggregate and reinsurance

- A year loss table's time of loss: `YearLossTable::with_times` in Rust,
  `read_ylt(..., time=)` or `day=` and `hour=` columns and
  `EventSet.from_ylt(times=)` in Python, `read_ylt(time, day, hour)` and
  `event_set_from_ylt(times)` in R. A year's rows are ordered by time, then
  slot, and the events carry the times, so the hours clause and
  reinstatements pro rata as to time run on a file read with it. The hours
  clause keeps labels on request: `EventSet::grouped_with` and
  `grouped_by` with `GroupLabels` in Rust, `EventSet.grouped(labels=,
  by=)` in Python and `hours_clause(labels, by)` in R, where `by="peril"`
  keeps each peril's windows apart; `grouped` alone still drops the labels
  (#594, #651).
