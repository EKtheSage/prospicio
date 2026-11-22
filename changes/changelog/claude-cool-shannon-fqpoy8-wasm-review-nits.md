### JavaScript (WebAssembly)

- `Triangle.fromLong` with neither `value` nor `values` says to give one;
  it said "give value or values, not both", the message for giving both.
- `EventSet.events(year)` past the last year says how many years the set
  has; its message no longer subtracts from the number of years.
- `Predictive.summary()` throws when a dimension is named `mean` or
  `std_dev` (a key column of that name): the summary's own column of that
  name replaced the dimension's labels (#1042).
