# Design note: catastrophe XL inputs

Status: **Implemented** (PR 1: event loss tables with secondary
uncertainty, the OEP curve, simulated years, year loss tables; PR 2:
merging event sets, the hours clause; clash stays parked; PR 3: a year
loss table checked and expanded, read from CSV or Parquet by the bindings)
· Depends on: `EventSet`, towers · Lane: Aggregate · Issues: #196, #484

## Goal

A cat XL is priced from a catastrophe model's output: an event loss table
(ELT), one row per event with its annual rate and its loss, often with
secondary uncertainty (a distribution of the loss given the event, not a
point), or a year loss table (YLT) of simulated years. Before this note a
user parsed a YLT into `EventSet::from_years` by hand and had no way to use
an ELT. Issue #196 also lists merging a cat event set with a non-cat one,
an hours clause that groups per-risk losses into occurrences, and clash
across lines.

## What exists

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

PR 2: `EventSet::merge(&other)` puts two sets' losses in each year
together (a cat event set with a non-cat one), and `EventSet::grouped(
window_hours)` groups each year's losses into occurrences by an hours
clause.

PR 3 (#484): `prospicio_aggregate::ylt::YearLossTable` holds a vendor's
year loss table as columns (each row's year from 1, loss, optional event
slot and labels), checks it (`check_unique(dims)`: no repeated year and
slot within the labels `dims`; `check_values(dim, expected)`; losses finite
and non-negative and years from 1 on construction) and expands it to a
run's simulations (`events(n_sims, Expansion::Tile | Resample { seed })`).
`ylt::hash_bytes` hashes the source file, which the events carry as
`EventSet::input_hash` into every result's provenance. Python
`prospicio.aggregate.read_ylt(path, ...)` and R `read_ylt()` read CSV or
Parquet (pyarrow or arrow, optional) into it; `EventSet.from_ylt` and
`event_set_from_ylt()` take columns.

Bindings: Python `prospicio.aggregate.EventLossTable`,
`EventSet.from_long`, `EventSet.merge` and `EventSet.grouped`; R
`event_loss_table()`, `elt_oep()`, `elt_occurrence_pml()`, `simulate_elt()`,
`event_set_from_long()`, `merge_event_sets()` and `hours_clause()`.

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
8. **Merged years interleave by time**, when both sets carry times, so
   annual terms are used up in the order the losses happened; without
   times, the first set's losses come first. Times or sums insured on only
   one set are an error rather than dropped.
9. **The hours clause opens a window at each first loss in turn.** A
   treaty lets the cedant place its windows where they recover most; that
   choice depends on the layer, so the greedy windows are the plain
   reading, and a user can group losses another way before building the
   set. A year is 8,760 hours; a window includes a loss exactly
   `window_hours` after its start. An occurrence's time is its window's
   start, and sums insured are not carried, since it spans risks.
10. **A YLT's rows stay separate losses** (#484). Rows are ordered by year,
    then slot, the file's order breaking ties. Rows that share a slot (one
    event hitting several classes) are not summed into one occurrence, as
    the reference model the issue reproduces does not sum them; a user who
    wants one occurrence per event groups the rows first. The duplicate
    check is by year, slot and the labels the user names (`unique_by`,
    peril in the issue), every label by default, since which labels split
    an event is the vendor's convention.
11. **Tiling needs a whole multiple; resampling draws per simulation.**
    `Tile` gives simulation `i` year `i mod n_years` and refuses a count
    that is not a multiple of the table's years, which would weight the
    early years more; `Resample` draws simulation `i`'s year from
    `StreamRng::new(seed, i)`, so any simulation replays alone. `n_years`
    is the largest year unless given, for a table whose last years have
    no loss.
12. **The file's hash is the results' input hash.** The bindings hash the
    bytes they read (blake3 through `InputHasher`, so it is not a plain
    blake3 of the file), and the events pass it to provenance's
    `input_hash`; merging two sets with hashes hashes the pair in order.
    The samplers stay unrecorded, as for any years from elsewhere.

## Validation

- Each row's beta has the row's mean and standard deviation (unit test).
- Merging keeps every loss, orders by time (the first set first on a
  tie), and refuses sets of different lengths or with times or sums
  insured on one side only; grouping keeps each year's total and puts a
  loss exactly at the window's end inside it (unit tests).
- Simulated years against the analytic OEP, within four binomial standard
  errors, at the steps of a table without secondary uncertainty and at
  return periods 10, 50, 100 and 250 years of one with it; the mean count
  and the mean annual loss within four standard errors of `Λ` and the
  expected annual loss (`validation/tests/aggregate.rs`, module
  `event_loss_table`, 200,000 years).
- The PML inverts the OEP to 1e-9.
- A YLT orders its rows by year and slot, tiles (refusing a count that is
  not a multiple), resamples each simulation from its own stream with
  about equal shares of the years, names the rows a check fails on, and
  carries the source hash into a result's provenance and through a merge
  (unit tests in `ylt.rs`; Python and R tests read a synthetic CSV, and
  Parquet when pyarrow is installed).

## Next

- Clash across lines is done in `prospicio_aggregate::bivariate`
  (`docs/design/aggregate.md`, "Two lines and clash"); #196 closed without
  it.
