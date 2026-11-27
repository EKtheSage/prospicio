### Models

- R `cross_validate()` and `grid_search()` take `map`, an `lapply`-shaped
  function that runs the fits, for example on a PSOCK cluster, and
  `seed`, which gives each fit its own seed so that the scores do not
  depend on the backend. `grid_search()` makes one task per candidate and
  split. The defaults keep the serial behaviour and scores (#761).
