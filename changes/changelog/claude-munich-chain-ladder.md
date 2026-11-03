### Breaking changes

- Rust: `prospicio_reserving::Error` has a new variant `Munich`, so an
  exhaustive `match` on it needs an arm (#559, #731).

### Reserving

- The Munich chain ladder (Quarg and Mack 2004): paid and incurred
  developed together, each step's factor corrected by the origin's
  paid-to-incurred residual, at parity with R ChainLadder's
  `MunichChainLadder` and chainladder-python's `MunichAdjustment`, with
  tails and over segments (`MunichChainLadder`, `MunichFit`; Python
  `MunichChainLadder`, R `munich_chain_ladder()`;
  `docs/design/reserving-v02.md` decision 29; #559, #731).
