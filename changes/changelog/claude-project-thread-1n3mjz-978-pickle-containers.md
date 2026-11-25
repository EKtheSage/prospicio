### Bindings

- `Triangle`, `EventSet` and `Design` can be pickled in Python, so they
  pass to `multiprocessing` and `joblib` workers. `Triangle` and
  `EventSet` also gain `to_json()` and `from_json()`, an exact document of
  the object, also in Rust (`Triangle::to_json`, `EventSet::to_json`), and
  `EventSet` compares with `==` (#978).
