---
type: Finding
title: The Gamma sampler, Marsaglia-Tsang instead of inverse transform
description: Gamma::sample draws by Marsaglia and Tsang (2000) with the U^(1/a) boost below shape 1, on the same ChaCha20 streams; 0.13 to 0.22 microseconds a draw at shapes 1e-3 to 1e6 against 3 to 375 by bisecting the cdf, and Mack's ABC lifetime bootstrap with the Gamma in 0.02 s against 39 s. Every Gamma draw changed; draws are no longer monotone in one uniform; below f64::MIN_POSITIVE draws are subnormal, and below about 5e-324 they are 0.
tags: [probability, gamma, sampling, rng, performance, bootstrap, reproducibility]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-08T16:00:00-07:00 }
sources:
  - id: code
    resource: ../crates/prospicio-prob/src/gamma.rs
    title: Gamma::sample, gamma::standard_gamma and their tests
  - id: rng
    resource: ../docs/design/rng.md
    title: Design note, RNG streams, stability log and open question 4
  - id: script
    resource: ../validation/scripts/gamma_sampler.py
    title: Independent reproduction of the pinned draws (cryptography ChaCha20, SciPy ndtri)
  - id: mack
    resource: ../validation/tests/reserving_mack_bootstrap.rs
    title: Mack's lifetime bootstrap against R MackChainLadder, now with the Gamma on ABC
  - id: paper
    resource: https://doi.org/10.1145/358407.358414
    title: Marsaglia and Tsang (2000), A simple method for generating gamma variables, ACM Transactions on Mathematical Software 26(3)
---

# Finding

`Gamma::sample` drew by inverse transform: `quantile(U)`, a bisection on
the regularized incomplete gamma function, whose series or continued
fraction takes on the order of `sqrt(shape)` terms near the mode. In a
release build a draw cost 3.1 µs at shape 1, 8.7 µs at 10, 29 µs at
`1e3`, 375 µs at `1e6` and 42 µs at `1e-3`. Mack's bootstrap of ABC,
whose late cells have shapes in the thousands, took 39 s for its lifetime
view at 20,000 simulations (16 threads) against 0.03 s with the
lognormal.[^mack]

Since 2026-10-08 it draws by Marsaglia and Tsang (2000): for shape
`a >= 1`, `d = a - 1/3`, `c = 1 / sqrt(9d)`, a normal `x` (by inverse
transform) and a uniform `u` per attempt, accept `d (1 + c x)^3` on the
squeeze `u < 1 - 0.0331 x^4` or on `ln u < x^2 / 2 + d (1 - v + ln v)`;
below 1, a draw at `a + 1` times `U^(1/a)`.[^paper] The same function
(`gamma::standard_gamma`) already drew the Student t and Clayton copulas'
Gamma variables; it moved from `copula.rs` to `gamma.rs`, and the
copulas' tests are unchanged. A draw now costs 0.13 to 0.22 µs at every
shape from `1e-3` to `1e6`, and ABC's lifetime view with the Gamma 0.02
s.[^code]

# Validation

At 100,000 draws for each of twelve shapes (`1e-3`, 0.01, 0.1, 0.5,
0.999, 1, 1.5, 3, 10, 100, `1e4`, `1e6`), the mean and the second and
third central moments about the true mean, whose standard errors are
exact from the central moments up to the sixth (cumulants `a (n - 1)!`),
are within 1.92 standard errors of the Gamma's; the chi-square over 100
bins cut at the Gamma's own quantiles is 73 to 109 on 99 degrees of
freedom, below the `1e-6` upper point (181), and 40 on 50 at shape `1e-3`
(upper point 113), whose lower bins merge (below). A squeeze
constant changed to 0.01 sends shape 1's chi-square to 1,440, so the test
has power.[^code] The pinned draws at seed 42, stream 3 are reproduced by
an independent implementation (ChaCha20 from Python's `cryptography`,
SciPy's `ndtri`) to within 2e-15.[^script]

# Consequences

* **Every Gamma draw changed**: `Gamma::sample`, `Dist::Gamma`, the Gamma
  process of `OdpBootstrap` (and its one-year view) and of
  `MackBootstrap`, and the bindings' `sample`. A draw takes at least two
  uniforms (three below shape 1), so later draws on a stream move too.
  The aggregate simulations, the copulas' marginals and the GLM families'
  simulated responses call `quantile` themselves and did not change.
* **Not monotone**: draws are still a pure function of `(seed, stream)`
  and independent of the thread count, but no longer a monotone function
  of one uniform, so common random numbers across scenarios hold only for
  code that calls `quantile`.
* **Underflow at tiny shapes**: below `f64::MIN_POSITIVE` (`2.2e-308`)
  the draw is subnormal and loses precision, and below the smallest
  subnormal (about `5e-324`) it rounds to 0. At shape `1e-3`, where
  `P(X < x)` is about `x^a`, 49% of the mass is below `MIN_POSITIVE` and
  about 47.5% rounds to 0, so the test's bins below `MIN_POSITIVE` merge
  into one there.
* **Re-pinned values**: two seed-pinned regressions moved. The one-year
  Mack validation's mean CDR over SD went from -0.0108, -0.0035, +0.0067
  to -0.0022, +0.0031, -0.0095 (RAA, GenIns, ABC), within 1.4 Monte Carlo
  standard errors of the exact zero. The ODP one-year view's ratios to
  Merz-Wuthrich (`GAP`, 31 of them) moved by up to 2.0 of their Monte
  Carlo standard errors; the totals went from 0.6087, 1.3620 and 1.1329
  to 0.6150, 1.3733 and 1.1356, re-pinned in Rust and in the R (RAA) and
  Python (GenIns) tests. Seed-specific figures in the reserving findings
  moved with them (EVW's Table 4 one-year total at seed 20,261,006 from
  1,778,254 to 1,761,459, the lifetime total at seed 20,261,007 from
  18,703,619 and 2,458,884 to 18,680,964 and 2,451,202), all within Monte
  Carlo error of the references. Every test against an external
  reference (R `BootChainLadder`, `MackChainLadder`, `CDR`, EVW Table 4,
  the exact moments at 200,000 simulations) passed unchanged.
* **The scheme name** stays `chacha20/sim-index/v1`, though `rng.md`'s
  policy asks for a bump on a new sampling method: the stream mapping is
  unchanged, and `Portfolio` reads the scheme to refuse parts that share
  random numbers. The change is in the stability log, and the choice is
  open question 4.[^rng]
* This edits the Probability lane's crate (`prospicio-prob`), with the
  user's approval, from the Reserving lane's branch
  claude/fast-gamma-sampler.

[^code]: crates/prospicio-prob/src/gamma.rs, `standard_gamma` and its tests
[^rng]: docs/design/rng.md, the stability log and open question 4
[^script]: validation/scripts/gamma_sampler.py
[^mack]: validation/tests/reserving_mack_bootstrap.rs
[^paper]: Marsaglia and Tsang (2000), A simple method for generating gamma variables, ACM Transactions on Mathematical Software 26(3)
