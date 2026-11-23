### Pricing

- `ActualState::from_trace` refuses a trace from a valuation, told by its
  nonzero opening balance. A valuation from a month above 0 whose trust and
  operating balances are both exactly 0 is still read as months `0..`; a
  `Trace::first_month` field would tell it, but adding a field to a `Trace`
  that can be built with a struct literal breaks semver, so it is left for
  0.3.0 (#960, #1018, #1043).
- The bonus-malus integrals over a very dispersed gamma heterogeneity (a cv
  above about 4.6) count the mass below `θ = 1e-300`, which they lost before
  (`1e-9` on a stationary probability at cv 6), and `stationary`,
  `optimal_relativities` and the other portfolio integrals return an error
  when they have not settled to `1e-8` at 4,096 panels, where they returned
  the last value (#960, #1018).
- `DealBook::summary` gives NaN for a ratio to a mean premium of 0, and
  `DealBook::adequacy` NaN for a return on a capital of 0. Before, they
  were `inf` (or NaN for 0 over 0) (#960, #1018).

### Aggregate and reinsurance

- `ruin::wiener_hopf` returns an error when every step `X - cW` is a
  multiple of two or more grid points (claims on a sub-lattice and a wait
  that rounds to one of its points). Before, every `psi` was NaN (#960, #1018).
