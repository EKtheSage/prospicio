### Distributions and probability

- A stable law inverted with `x_min` given refuses `Tilt::Fixed` (the grid is
  inverted untilted, so the tilt was ignored; `Tilt::Auto` falls back to none,
  now documented). A tilted one-sided stable law on a given `step` with a
  window that, from its light edge, would not reach the mean is placed on the
  body (the mean); one that reaches it stays at the light edge, as before; the automatic tilt there counts the light side's mass below the
  window, which the tilt magnifies as it wraps (it falls to no tilt where that
  mass is large), and `Tilt::Fixed` is refused where it would magnify that
  mass past round-off. `Stable::tail_mass_above` returns an error where its
  series is not valid (at or below the location, near the body) instead of a
  value above 1 (above `alpha = 1` it answers while its smallest term is
  within 1e-3 of the sum); `chf::invert` then reports the mass beyond the
  window as what the masses leave of 1 under a tilt that keeps the tail from
  wrapping, under a smaller tilt `tau` as that divided by `1 - e^-tau`, and
  untilted as an upper bound (`P(X > location)`, or 1) (#945, #1010).
