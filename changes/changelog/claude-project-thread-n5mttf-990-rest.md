### Models

- `Statistic::NAMES` and `from_name` (`prospicio-bayes`) and
  `Smoothing::NAMES`, `from_name` and `name` (`prospicio-glm`): the
  posterior predictive check's statistics and the GAM's smoothing criteria
  named once in Rust and shared by the Python and R parsers. Python's
  `statistics` now also takes a quantile written `"q0.9"`, as R's does
  (#990).

### Reserving

- `AnnualReading::NAMES`, `from_name` and `name`: the annual reading names
  (`"weibull"`, `"loglogistic"`, `"ln-linear"`) defined once in Rust, used
  by the four parsers in Python and R that had copies. Names and messages
  are unchanged (#990).
