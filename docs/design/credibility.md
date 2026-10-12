# Design note: classical credibility

Status: **Decided; implemented** (limited fluctuation, Bühlmann–Straub) · v0.6 · Depends on: nothing · Lane: Aggregate

## Goal

Weigh a group's own experience against the collective's: the classical
standards for how much experience is enough (limited fluctuation), and the
greatest-accuracy weights of Bühlmann and Bühlmann–Straub. Hierarchical
and Bayesian credibility come with v0.7 (`roadmap.md`, order of work).

## What exists

- `prospicio_pricing::credibility::LimitedFluctuation::new(p, k)`: the full
  standards in expected claims, `n_F = (Φ⁻¹((1 + p)/2) / k)²` for the
  frequency, `n_F cv²` for the severity and `n_F (1 + cv²)` for the pure
  premium, and the square root rule `Z = min(1, √(n / standard))`
  (Werner and Modlin, *Basic Ratemaking*, ch. 12). At `p = 0.90`,
  `k = 0.05` the frequency standard is 1,082.
- `prospicio_pricing::credibility::BuhlmannStraub::new(estimator).fit(ratios,
  weights)`: groups by periods, with weights or (Bühlmann) without, and
  missing cells as NaN in both. The fit holds the collective mean, the
  between variance `a`, the within variance `s²`, `k = s²/a`, and per
  group the weighted mean, total weight, credibility factor and premium.
- Python `prospicio.credibility` (`LimitedFluctuation`,
  `buhlmann_straub`, `BuhlmannStraubFit`); R `credibility_standard()`,
  `limited_fluctuation_credibility()`, `buhlmann_straub()`.

## Decisions

1. **Follow `actuar::cm`** for the one-level model, so its results carry
   over: `s² = Σ w_ij (X_ij - X̄_i)² / (n - I)`; the between variance by
   Bühlmann and Straub's unbiased estimator
   `w (Σ w_i (X̄_i - X̄)² - (I - 1) s²) / (w² - Σ w_i²)` or Bichsel and
   Straub's iterative one (`a = Σ Z_i (X̄_i - X̄_Z)² / (I - 1)` to a
   relative change below `√ε`, at most 100 iterations, started from the
   unbiased value); with one level `actuar`'s `"Buhlmann-Gisler"` and
   `"Ohlsson"` are both the unbiased estimator, so prospicio names the
   estimator, not the authors. With equal weights the iterative estimator
   is not run (it equals the unbiased one).
2. **The collective mean is credibility-weighted**, `Σ Z_i X̄_i / Σ Z_i`,
   which makes the premiums balance in the collective. When `a <= 0` every
   factor is 0 and the collective mean is the weight-averaged mean; the fit
   still reports the negative estimate.
3. **A group with no weight** keeps a mean of 0, a factor of 0 and the
   collective premium, and does not count among the groups `I`.
4. **One namespace in the bindings**, `credibility`, as `architecture.md`
   plans; the Rust code lives in `prospicio-pricing` until a Bayesian side
   justifies `prospicio-credibility`.

## Validation

`validation/tests/pricing.rs` checks 52 values against R `actuar` 3.3.7
(`validation/scripts/r_credibility.R`) on Hachemeister's data
(`validation/reference/hachemeister.csv`): the weighted fit with each
estimator, Bühlmann's model, and the weighted fit with four cells missing,
each with the collective mean, both variances, and every state's factor
and premium, at `1e-12` (`1e-10` on the between variance, `1e-11` on the
factors). Nine limited fluctuation standards match R's `qnorm` at `1e-13`.
The R tests also compare the premiums with `actuar::cm` live when it is
installed.
