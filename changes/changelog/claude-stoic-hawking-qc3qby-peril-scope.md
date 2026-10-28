### Breaking changes

- Rust: `Layer` gains the public field `scope`; code that builds one in
  full adds it (`None` covers every loss as before) (#584, #660).

### Aggregate and reinsurance

- Layers scoped to a peril or class on a multi-peril event set
  (`Layer::scope(label, values)`): the layer's occurrence and annual terms
  see only the losses with those label values, so another peril's events
  neither use its annual limit nor count toward its deductible, and the
  losses it does not cover reach later inuring stages whole.
  `EventSet::filter` gives the set such a layer sees. A tower with a
  scoped layer needs labelled events; the grids, annual losses and the
  P&L ledger refuse it. Tower JSON format version 7. Python
  `Layer.with_scope`, `EventSet.filter`; R `with_scope()`,
  `filter_events()` (#584, #660).
