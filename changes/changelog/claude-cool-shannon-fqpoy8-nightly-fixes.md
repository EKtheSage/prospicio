### Packaging

- The R package builds on Windows with Rust 1.100 and later: `Makevars.win`
  links `-lsynchronization`, without which the install failed with
  undefined references to `WakeByAddressSingle` (#927).
