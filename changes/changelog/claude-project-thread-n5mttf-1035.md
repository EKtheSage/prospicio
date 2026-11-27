### Distributions and probability

- `QuantileMethod::NAMES` and `TvarMethod::NAMES`, with `from_name` and
  `name`: the quantile and TVaR method names defined once in Rust and
  shared by the Python, R and Excel parsers. The accepted names are
  unchanged; an unknown name's error now ends `got "..."` as the other
  option errors do (#1035).
