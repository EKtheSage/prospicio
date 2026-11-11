### Reserving

- `Triangle::stack`: one-segment triangles stacked as the segments of one
  triangle under a key, the stacking `HierarchicalBf` takes from a dict of
  triangles in Python and a named list in R, now one function in Rust that
  both bindings call (#838). A value column named as the key is an error in
  both languages (Python accepted it before); the messages name the program
  as `program "a" has no column "missing"` in both.
