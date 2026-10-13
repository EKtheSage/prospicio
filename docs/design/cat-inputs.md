# Design note: catastrophe XL inputs

Status: **In progress** (PR 1 of 2: event loss tables with secondary
uncertainty, the OEP curve, simulated years, year loss tables) · Depends on:
`EventSet`, towers · Lane: Aggregate · Issue: #196

## Goal

A cat XL is priced from a catastrophe model's output: an event loss table
(ELT), one row per event with its annual rate and its loss, often with
secondary uncertainty (a distribution of the loss given the event, not a
point), or a year loss table (YLT) of simulated years. Before this note a
user parsed a YLT into `EventSet::from_years` by hand and had no way to use
an ELT. Issue #196 also lists merging a cat event set with a non-cat one,
an hours clause that groups per-risk losses into occurrences, and clash
across lines.

## What exists (PR 1)

`prospicio_aggregate::cat`:

- `EventLossTable::new(rates, means, sd_independent, sd_correlated,
  exposures)`: one row per event.
- `rate()` (`Λ = Σ rate`), `expected_annual_loss()`,
  `exceedance_frequency(x)` (`Σ rate_i P(L_i > x)`), `oep(x)` and
  `occurrence_pml(p)`.
- `simulate(n_years, seed) -> (EventSet, Vec<Vec<usize>>)`: simulated
  years, each loss with the row of its event, ready for any tower.

`EventSet::from_long(years, losses, n_years, seed)` reads a YLT in long
form (one row per loss with its year), filling years with no loss.

Bindings: Python `prospicio.aggregate.EventLossTable` and
`EventSet.from_long`; R `event_loss_table()`, `elt_oep()`,
`elt_occurrence_pml()`, `simulate_elt()` and `event_set_from_long()`.

## Decisions

1. **Events are independent Poisson processes**, so the ELT is a compound
   Poisson model: a Poisson count with mean `Λ`, each event the row `i`
   with probability `rate_i / Λ`. The OEP is then exact,
   `1 − exp(−Σ rate_i P(L_i > x))`, and is the reference the simulation is
   checked against.
2. **Secondary uncertainty is a beta on `[0, exposure]`** with the row's
   mean and standard deviation, the usual convention of catastrophe models'
   ELTs. A row with no standard deviation loses its mean exactly.
3. **The standard deviation is `σ_independent + σ_correlated`**, summed, not
   in quadrature, as the models report them for one portfolio. The split
   matters only when ELTs are combined by event across portfolios, which is
   not done here.
4. **An infeasible row is an error that names it.** A beta with mean `m`
   on `[0, E]` needs `σ² < m (E − m)`; real tables sometimes break it.
   Clamping would change the table silently; the error says which row and
   the largest standard deviation it allows.
5. **Simulation draws one Poisson count, then picks each event by its
   rate** (a binary search on the cumulative rates), so a year costs its
   number of events, not the table's length, and year `i` draws from its
   own stream: any year replays alone.
6. **Event ids are returned beside the `EventSet`**, not stored in it, so
   the set's shape is unchanged; PR 2's merge and any later clash can use
   them.
7. **No file reading in the crate** (the WASM scope): an ELT or YLT enters
   as columns, from a data frame in Python or R.

## Validation

- Each row's beta has the row's mean and standard deviation (unit test).
- Simulated years against the analytic OEP, within four binomial standard
  errors, at the steps of a table without secondary uncertainty and at
  return periods 10, 50, 100 and 250 years of one with it; the mean count
  and the mean annual loss within four standard errors of `Λ` and the
  expected annual loss (`validation/tests/aggregate.rs`, module
  `event_loss_table`, 200,000 years).
- The PML inverts the OEP to 1e-9.

## Next

- **PR 2:** merging a cat event set with a non-cat one year by year, and an
  hours clause that groups per-risk losses with times into occurrences.
- Clash across lines stays parked with the bivariate aggregates
  (`docs/design/aggregate.md`); #196 closes when the user decides that.
