### Distributions and probability

- A stable law inverted with `x_min` given refuses `Tilt::Fixed` (the grid is
  inverted untilted, so the tilt was ignored; `Tilt::Auto` falls back to none,
  now documented). A tilted one-sided stable law on a given `step` with a
  window narrower than its span is placed on the body (the mean), not at its
  light edge. `Stable::tail_mass_above` returns an error where its series is
  not valid (at or below the location, near the body) instead of a value above
  1; `chf::invert` then takes the mass beyond the window from what the window
  leaves of 1 (#945, #1010).
