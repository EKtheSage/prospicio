### Distributions and probability

- Exponential tilting in the characteristic-function engine, with the
  alpha-stable law (`prospicio_prob::Stable` in S1 or S0;
  `ChfOptions::tilt`, automatic, off or a given number of e-folds): a
  stable law with `beta` 1 or -1 is inverted along a tilted line that
  damps its power tail, the bucket capped and the mass beyond the window
  reported, at parity with `aggregate` 2.1.0 (Python `Stable` and
  `invert_chf(tilt=)`, R `stable_distribution()` and `invert_chf(tilt =)`;
  #779, #916).
