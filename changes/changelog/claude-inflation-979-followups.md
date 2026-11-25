### Reserving

- Separation and indexation fixes (#979): `SeparationFit::log_variance`
  (`log_variance` in Python and R) is `None` (`NULL` in R) rather than NaN
  when the geometric method has no degrees of freedom left, so a fit
  compares equal to itself; the geometric method's error for a
  non-positive increment says it is the increment over the normaliser; R
  `future_cells()` refuses an age that is not a whole number of months
  instead of truncating it, and `future_cells()` and `reinflated()` called
  without their arguments say what they need.
