### Bindings

- The GLM family and link names come from one table each,
  `FamilyKind::NAMES` and `LinkKind::NAMES` with `from_name` and `name` in
  `prospicio-models` (and `Family::kind`, `Link::kind`), which Python, R and
  Excel read: Python and R now take Excel's `"normal"` and `"negbin"`, and
  an unknown family or link reads the same in all three, `family must be
  "gaussian", "normal", ..., got "weibull"` (Excel lists its
  `"quasipoisson"` first). The GLMM family and the two-part models' zero
  link read the same tables and keep their narrower lists; Excel's
  `PRO.PRC.GLM` names the power link as one it has no exponent for (#834).
