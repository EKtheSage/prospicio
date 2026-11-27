### Aggregate and reinsurance

- `CorrelationKind::NAMES` and `ApproximationKind::NAMES`, with
  `from_name` and `name`: the correlation kind (`"pearson"`,
  `"spearman"`) and compound-moment approximation names defined once in
  Rust and shared by the Python and R parsers. The accepted names are
  unchanged; errors end `got "..."`, and R's unknown approximation now
  lists the names (#1035).
