### Models

- `Design` takes its columns as a 2-D float64 NumPy array (rows are the columns) or as a list of 1-D arrays or lists, and its offset and weights as arrays, each read in one copy; `Design.to_numpy()` returns the design matrix as an `(n_rows, n_columns)` array, which the boosting wrappers now use instead of stacking the columns one list at a time (#978).
