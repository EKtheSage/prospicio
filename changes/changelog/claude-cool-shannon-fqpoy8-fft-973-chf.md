### Distributions and probability

- The characteristic-function engine (`chf`) takes its transforms through
  `prospicio_math::fft`: a two-sided stable law's convolution is about
  twice as fast, and its result moves by round-off; the inversion's masses
  are unchanged. `chf::Complex64` is the same type as before, re-exported
  from `prospicio_math::fft` (#973).
