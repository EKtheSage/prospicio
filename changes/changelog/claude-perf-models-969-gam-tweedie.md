### Models
- Faster GAMs with REML smoothing: for a canonical link the score reuses
  IRLS's information, and the gamma family's dispersion search no longer
  sums over the rows at every step, so a gamma GAM on 50,000 rows fits in
  about 3 s instead of about 60 s. `tweedie_profile` fits its grid of
  powers in parallel (#969, #1060).

### Breaking changes
- GAMs with REML smoothing and a canonical link, or the gamma family, give
  smoothing parameters that move in the last bits, since the information
  and the saturated log-likelihood are summed in another order (#969, #1060).
