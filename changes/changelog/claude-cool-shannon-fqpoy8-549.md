### Models

- Time-ordered splits with a rolling window, a forecast horizon, a gap and a step: `prospicio_models::resample::TimeSplits`, the defaults equal to `time_ordered` and scikit-learn's `TimeSeriesSplit` reproduced (`models.md`, "Time splits with windows, horizons and gaps"). Python `time_splits(periods, n_test, window=, horizon=, gap=, step=)`, R `time_splits()` (#549).
