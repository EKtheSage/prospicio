### Distributions and probability

- A stable law inverted with `x_min` given refuses `Tilt::Fixed` (the grid is
  inverted untilted, so the tilt was ignored; `Tilt::Auto` falls back to none,
  now documented). A tilted one-sided stable law on a given `step` with a
  window narrower than its span is placed on the body (the mean), not at its
  light edge; the automatic tilt there counts the light side's mass below the
  window, which the tilt magnifies as it wraps (it falls to no tilt where that
  mass is large), and `Tilt::Fixed` is refused where it would magnify that
  mass past round-off. `Stable::tail_mass_above` returns an error where its
  series is not valid (at or below the location, near the body) instead of a
  value above 1; `chf::invert` then reports the mass beyond the window as
  what the masses leave of 1 under a tilt that keeps the tail from wrapping,
  and otherwise as an upper bound (`P(X > location)`, or 1) (#945, #1010).
