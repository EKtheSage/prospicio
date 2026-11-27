### Reserving

- `MackProcess::NAMES`, `from_name` and `name`: the Mack bootstrap's
  process names defined once in Rust, which the Python, R and Excel front
  ends now share. Accepted names and defaults are unchanged; Excel's error
  message now lists the names in the same order as Python's and R's (#990).
