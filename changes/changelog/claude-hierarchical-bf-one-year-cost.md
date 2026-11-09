### Breaking changes

- Rust: `prospicio_reserving::bayes::BayesOneYearFit` has the public field
  `warnings: Vec<String>`, so a struct literal of it needs the field (add
  `warnings: Vec::new()`) and an exhaustive pattern needs `..` (#762, #821).

### Reserving

- `HierarchicalBf`'s one-year view refits with a warm-started light refit
  by default (no pilot run, the sampler scaled by the opening posterior, 4
  chains of 150 warm-up and 250 kept draws), within 0.004 sd of full
  refits in the total on the reference panel (and 0.042 sd of each
  program's own) at about a tenth of their cost; `BayesOneYear::light_refit` sets another, and a refit mode chosen
  keeps the fit's own sampler as for `Compartmental`. Python
  `HierarchicalBf.one_year(light_refit=)` (its `refit` now defaults to
  `None`, still moment matching then refitting) and R
  `bayes_hierarchical_bf_one_year(light_refit =)`. A panel whose programs
  stop at different ages now warns, naming the programs whose next diagonal
  falls at ages they never observed (`BayesOneYearFit::warnings`; Python
  `UserWarning` and `warnings`, R `warning()` and `warnings`) (#762, #821).
