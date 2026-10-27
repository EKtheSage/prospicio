### Reserving

- Python's `apriori` reads a pandas Series by its index, as a dict keyed by origin label or integer year, so `df.set_index("origin")["elr"]` lines up with the origins in any row order; a Series with a default `RangeIndex` is an error asking for origin keys, since its order cannot be checked. Polars Series and numpy arrays stay values in origin order (#644).
