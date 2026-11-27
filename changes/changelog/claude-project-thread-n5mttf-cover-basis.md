### Aggregate and reinsurance

- `CoverBasis::NAMES`, `from_name` and `name`: the cover basis names
  (`"losses_occurring"` or `"lod"`, `"risks_attaching"` or `"rad"`) defined
  once in Rust and shared by the Python and R parsers. The accepted names
  are unchanged; an unknown basis's error now lists the aliases too, as
  `cover basis must be ...` in Python and `basis must be ...` in R (#1035).
