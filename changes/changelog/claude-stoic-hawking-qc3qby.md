### Distributions and probability

- Bootstrap standard errors of a statistics grid
  (`StatisticsGrid::bootstrap_se`): each replicate resamples the
  simulations once for every column, so the items' errors share their
  resamples, and is read off counts on each column sorted once;
  `bootstrap_se_holding` keeps a fixed part of each simulation and
  resamples only the rest. Python `bootstrap_se` (with `hold_fixed`), R
  `bootstrap_se()` (with `fixed`) (#488).
