### Aggregate and reinsurance

- `EventSet.from_years`, `from_long` and `from_ylt` in Python take NumPy arrays for their years, losses, slots, times and sums insured, read in one copy each, and `EventSet.to_numpy()` returns the losses and each year's offsets as read-only arrays that view the event set's own buffers. In Rust, `EventSet::losses` and `EventSet::offsets` give the same flat buffers (#978).
