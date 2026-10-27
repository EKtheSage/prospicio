# Design note: credibility

Status: **Decided; implemented** (limited fluctuation, Bühlmann–Straub, v0.6; Bayesian hierarchical credibility, v0.7; Hachemeister's regression model, #561) · Depends on: `prospicio-bayes` for the Bayesian side · Lanes: Aggregate (classical), Models (Bayesian)

## Goal

Weigh a group's own experience against the collective's: the classical
standards for how much experience is enough (limited fluctuation), and the
greatest-accuracy weights of Bühlmann and Bühlmann–Straub; and their
Bayesian, hierarchical form, with the structure parameters' uncertainty in
the premium (v0.7, below).

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
- `prospicio_pricing::credibility::Hachemeister::new(estimator,
  adjust_intercept).fit(ratios, weights, design)` (#561): Hachemeister's
  regression model, each risk's weighted least squares line (or any
  design shared by the risks) shrunk towards the collective by a
  credibility matrix `Z_i = A (A + s² W_i)⁻¹`; the fit holds the
  collective, each risk's credibility and own coefficients in the design's
  basis, the credibility matrices and between covariance in the basis
  fitted, `s²`, the transition `R` when adjusted, and `predict(x)`.
- Python `prospicio.credibility` (`LimitedFluctuation`,
  `buhlmann_straub`, `BuhlmannStraubFit`, `hachemeister`,
  `HachemeisterFit`); R `credibility_standard()`,
  `limited_fluctuation_credibility()`, `buhlmann_straub()`,
  `hachemeister()` with `predict()`.

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

Regression credibility (#561), numbered after the Bayesian section's:

9. **Hachemeister follows `actuar`'s `hache`** (#561), both of its forms.
   At the origin (`adjust_intercept = false`, `hache.origin`) the
   credibility matrices are full and the between covariance is
   Hachemeister's iterative pseudo-estimator
   `A = Σ Z_i (b_i - β̂)(b_i - β̂)ᵀ / (I - 1)`, symmetrised, started from
   `Z_i = I` and run until the collective's largest relative change is
   below `tol`, with one more update of `A` and the `Z_i` at the end. There
   is no unbiased estimator of a full `A`, so the origin iterates whatever
   the `StructureEstimator` (taking its `tol` and `max_iter` when
   iterative, `√ε` and 100 otherwise), as `actuar::cm` does under its
   default `"Buhlmann-Gisler"`. With `adjust_intercept = true`
   (`hache.barycenter`) the design is orthogonalised under the collective
   weights `w_t = Σ_i w_it / Σ w` by Gram–Schmidt, which puts the intercept
   at the periods' barycentre and makes `A` and the `Z_i` diagonal; each
   diagonal is Bühlmann–Straub's `a` by the chosen estimator on the
   coefficient's weights `Σ_t w_it x_tk²`, with decision 2's rule when
   `a <= 0`.
10. **`s²` is the mean of the risks' residual variances**
   `Σ_t w_it r_it² / (n_i - p)`, as `actuar` (0 for a risk with exactly
   `p` cells), not a pooled one; with an intercept alone and every cell
   observed it equals Bühlmann–Straub's `s²`, so the adjusted fit on `~1`
   is Bühlmann–Straub's to `1e-10`.
11. **Coefficients in the design's basis, matrices in the basis fitted.**
   The collective and each risk's coefficients are mapped back by `R⁻¹`
   (`actuar`'s `transition`), so `predict` works on design rows as given;
   `Z_i` and `A` stay in the orthogonal basis where they are diagonal. The
   Gram–Schmidt `R` has a positive diagonal; a sign flip of a column of
   `Q` would leave the coefficients and the diagonals unchanged.
12. **Refused, not guessed:** a risk whose observed cells of positive
   weight do not identify the `p` coefficients (`actuar`'s `lm.wfit`
   would drop a column), linearly dependent design columns, a design whose
   first column is not constant or a period with no weight when the
   intercept is adjusted, fewer than two risks with weight, and a
   singular `A + s² W_i` or `Σ Z_i` (two risks cannot identify a `2 × 2`
   `A` at the origin; `actuar` fails there too). A risk with no weight
   takes the collective.

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

Hachemeister's model is checked on the same data with the design
`[1, t]`, `t = 1, ..., 12`, against `actuar::cm(~state, regformula = ~time)`
(`validation/scripts/r_credibility_regression.R`,
`validation/reference/credibility_regression_actuar.csv`, 144 values): the
adjusted fit with each estimator (collectives, coefficients, the
credibility and between diagonals and `s²` at `1e-9`, `s²` at `1e-12`,
each state's premium at `t = 13` at `1e-10`), and the origin fit with all
cells and with four missing. On these data the origin's between
covariance has a correlation of 0.9998 and the iteration has no stable
fixed point: it stops at `√ε` after 47 updates, and run on it wanders by
about `1e-5` (`knowledge/findings/hachemeister-origin-iteration.md`), so
round-off decides its last steps and the origin rows are checked at
`1e-6`. Python and R check the premiums against the same values, and R
against `actuar` live when it is installed.

## Bayesian hierarchical credibility (decided 2026-10-16)

Status: implemented in `prospicio_bayes::credibility` (Models lane), in
Rust, Python (`prospicio.credibility.hierarchical_credibility`,
`credibility_premiums`) and R (`hierarchical_credibility()`,
`credibility_premiums()`, `credibility_predictive()`).

### Model

Bühlmann–Straub's model, and Jewell's hierarchy of it, as a normal
random-effects model: `X_ij | θ_i ~ N(θ_i, σ²/w_ij)`; each group's risk
premium is normal about its node at the level above, `θ_i ~ N(ν_p(i),
τ_L²)`, each node about its parent, `ν_k ~ N(ν_p(k), τ_l²)`, and the top
level about the collective mean `μ`, flat or `N(m, s²)`. The hierarchy is
given as `levels[k][i]`, group `i`'s node at level `k`, outermost first;
nesting is checked.

### Decisions

5. **Integrate the means out.** Given `σ` and the `τ_l` every mean is
   Gaussian, so one pass up the tree (each node's children's messages
   `N(M_c; value, V_c)` combined after integrating the child,
   `V_c + τ²`) gives the likelihood of the data in closed form, and NUTS
   samples only `log σ` and the `log τ_l`, `L + 1` parameters with no
   funnel between a group's mean and its scale. The gradient is forward
   mode, one coordinate a pass (a small dual type in the module), checked
   against finite differences. The means are drawn afterwards from their
   Gaussian conditionals, top down.
6. **The premium is the credibility formula averaged over the posterior.**
   Given the structure parameters, each node's posterior mean is
   `Z M + (1 − Z) E[parent]`, with `M` the credibility-weighted mean of its
   children and `Z = P τ² / (P τ² + 1)` (`P` the precision of its
   children's message; `w_i τ² / (w_i τ² + σ²)` for a group): exactly
   Bühlmann–Straub's premiums with one level and Bühlmann–Gisler's
   hierarchical ones with more, with a flat prior on `μ`.
   `credibility_premiums` exposes that formula at given variances; the
   fit's `premiums` are its mean over the posterior draws
   (Rao–Blackwellized), and `predictive` draws the risk premiums or a
   next period's ratios, with parameter and process uncertainty.
7. **Weakly informative defaults, from the data.** Half-normal priors at
   five times the spread the data show (`σ`: the pooled within standard
   deviation; each `τ_l`: the standard deviation of that level's weighted
   means, or the groups' where a level has one node), and a flat `μ`.
   Every prior can be set: half-normal or half-Cauchy scales, a normal on
   `μ`.
8. **Where it lives.** The Bayesian side sits in `prospicio-bayes`, next
   to its sampler; the bindings join the classical functions in the one
   `credibility` namespace (decision 4). A `prospicio-credibility` crate is
   still not needed.

### Validation

- `validation/tests/models.rs`, `credibility_premiums_match_actuar`: the
  formula at `actuar::cm`'s structure parameters reproduces its premiums,
  factors and collective means on Hachemeister's data at `1e-10`, for
  Bühlmann–Straub and for the hierarchy of actuar's own example (states
  1 and 3 in one cohort, 2, 4 and 5 in the other) with the
  Bühlmann–Gisler, Ohlsson and iterative estimators
  (`validation/scripts/r_credibility_hierarchical.R`).
- `bayesian_credibility_matches_grid_integration`: the NUTS fit on
  Hachemeister's data (half-normal priors of 60,000 on `σ` and 1,500 on
  `τ`) against exact posterior moments by grid integration, with `μ` and
  the group means integrated by quadrature written independently
  (`validation/scripts/credibility_bayes_grid.py`): the premiums, factors
  and collective mean within a tenth of a posterior standard deviation,
  the variances' means likewise, their standard deviations within 8%
  (20% for `τ²`, whose long right tail makes its sample standard
  deviation noisy: 14% low at 10,000 draws on one seed, 1% at 100,000).
- Unit tests: the integrated likelihood against quadrature over `μ`, the
  gradient against finite differences with and without a hierarchy and a
  prior on `μ`, a group with no weight taking its parent's premium, the
  refused hierarchies, and a simulated structure recovered by NUTS with
  the predictive's means equal to the premiums.

### Not done

Classical estimators of the hierarchy's structure parameters (actuar's
Bühlmann–Gisler and Ohlsson for more than one level) are not built: the
Bayesian fit estimates them, and `credibility_premiums` takes any.
