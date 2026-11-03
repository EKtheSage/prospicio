### Reserving

- Merz and Wüthrich's paid-incurred chain: `PaidIncurredChain` in Rust
  and Python and `paid_incurred_chain()` in R fit the paid and the
  incurred triangle in one log-normal Bayesian model with a closed-form
  posterior, with each origin's ultimate, reserve, incurred weight and
  standard error. `convention = "r_chainladder"` reproduces R
  ChainLadder's `PaidIncurredChain`, which leaves out part of the
  posterior (`docs/design/reserving-v02.md` decision 32; #560, #734).
