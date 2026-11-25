### Distributions and probability

- The Python package now depends on NumPy. `PredictiveDistribution` takes its draws as a 2-D float64 NumPy array, read in one copy, as well as a list of rows, and `PredictiveDistribution.to_numpy()` returns the draws as a read-only array that views the distribution's own buffer without copying (#978).
