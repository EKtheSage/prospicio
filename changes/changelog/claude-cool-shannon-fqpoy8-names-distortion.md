### Bindings

- A distortion's names come from one table, `DistortionKind::NAMES` with
  `from_name` and `name` in `prospicio_prob::distortion` (and
  `Family::kind`), which Python, R and Excel read: R's `distortion()` now
  takes `aggregate`'s short names (`"ph"`, `"dual"`, `"exp"`, `"clin"`,
  `"cll"`, `"ly"`, `"wtdtvar"`) as Python's calibration and Excel do, and
  an unknown family in Excel reads `family must be "ccoc", "ph",
  "proportional_hazard", ..., got "nope"` (#834).
