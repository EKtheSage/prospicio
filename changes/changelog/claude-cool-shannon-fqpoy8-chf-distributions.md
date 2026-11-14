### Distributions and probability

- The alpha-stable law's algebra: sums of laws with a common `alpha`,
  shifts, scaling by any real number, `t` independent copies, `Pr(X >
  loc)` and the stable limit of a Poisson compound with a Pareto tail
  (`Stable::add`, `shift`, `scale_by`, `copies`, `prob_above_loc`,
  `from_compound_pareto`; Python `+`, `-`, `*`, `/` and `@` on `Stable`,
  R `stable_add()`, `stable_scale()`, `stable_copies()`,
  `stable_from_compound_pareto()`), at parity with `aggregate` 2.1.0
  (#780, #920).
