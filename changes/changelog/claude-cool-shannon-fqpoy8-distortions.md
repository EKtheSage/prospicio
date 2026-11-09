### Breaking changes

- Rust: `prospicio_prob::Distortion` has the variant `Tsallis` and
  `prospicio_prob::Family` the variant `Tsallis`, so an exhaustive `match`
  on either needs an arm for it (#769, #812).

### Distributions and probability

- The Tsallis distortion of order `alpha`, the Beta(alpha, 1) mixture of
  TVaRs, priced at the mean plus the cumulative residual Tsallis entropy
  (`Distortion::tsallis`; Python `Distortion.tsallis`, R
  `distortion("tsallis", alpha)`, Excel `"tsallis"`), at parity with
  `aggregate` 2.1.0 (#769, #812).
- The Gini index of every distortion, `2 * integral(g) - 1`, and the member
  of a one-parameter family with a given index (`Distortion::gini`,
  `Distortion::from_gini`; Python `.gini` and `Distortion.from_gini`, R
  `@gini` and `distortion_from_gini()`), at parity with `aggregate`
  2.1.0 (#770, #812).
- A set of distortion families calibrated to one premium, with
  `aggregate` 2.1.0's extended set that adds Tsallis
  (`distortion::calibrate_all`, `Family::EXTENDED`; Python and R
  `calibrate_distortions(..., extended=)`, Excel
  `PRO.CAP.DISTORTION.CALIBRATE` with `families` `"extended"`) (#771).
