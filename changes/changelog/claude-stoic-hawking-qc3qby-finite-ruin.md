### Aggregate and reinsurance

- Ruin within a finite horizon: `finite_horizon` gives `psi(u, t)`, the
  chance that the surplus is negative at the end of some period up to
  `t`, for any claim count a period, by a convolution recursion on the
  grid; `finite_horizon_poisson` checks Poisson claims in sub-periods,
  which approaches continuous time from below. Python
  `prospicio.aggregate.finite_horizon_ruin`, R `finite_horizon_ruin()`
  (#562).
