### Guides

- (#718) `submission-schema` searches the investor's IRR with `cashflows(extend=True)` (#675) in
  place of `upper=5.0`, with every printed number unchanged, and names the per-simulation
  patterns of #673 that it does not use yet (#717).

### Examples

- (#718) `ils_deal_analysis` and its R twin use `extend=True` (#675) in place of `upper=5.0`
  (output byte-identical) and say why the notebook keeps one payment pattern (#673).
  `one_year_view` and its R twin load GenIns with `Triangle.read_csv` and `read_triangle()`
  (#552).
