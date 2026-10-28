# Design note: catastrophe XL inputs

Status: **Implemented** (PR 1: event loss tables with secondary
uncertainty, the OEP curve, simulated years, year loss tables; PR 2:
merging event sets, the hours clause; clash stays parked; PR 3: a year
loss table checked and expanded, read from CSV or Parquet by the bindings;
PR 4: a year loss table's times of loss, and labels kept through the hours
clause)
· Depends on: `EventSet`, towers · Lane: Aggregate · Issues: #196, #484, #594

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

PR 4 (#594): `YearLossTable::with_times` takes each row's time of loss, a
fraction of its year, so a table read from a file carries times
(`EventSet::has_times`) and the hours clause and reinstatements pro rata
as to time run on it; the bindings read a `time` column, or `day` and
`hour` columns. `EventSet::grouped_with(window_hours, GroupLabels)` keeps
the labels through the hours clause and `EventSet::grouped_by(dim,
window_hours, GroupLabels)` applies it within each value of a label
dimension, a peril say. Python `EventSet.grouped(window_hours, labels=,
by=)` and `read_ylt(..., time=, day=, hour=)`; R `hours_clause(x,
window_hours, labels, by)` and `read_ylt(..., time, day, hour)`.

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
13. **A YLT's times order its rows** (#594). With times, rows are ordered
    by year, then time, then slot, the file's order breaking the last
    ties, since an `EventSet`'s losses are chronological within a year.
    A time is a fraction of the year in `[0, 1]`, the range
    `EventSet::with_times` takes (the issue asked for `[0, 1)`; 1 is kept
    so a time built from a set's own times always reads back). From day
    and hour columns the time is `((day - 1) * 24 + hour) / 8760`, day 1
    to 365 and hour 0 to 24, on the hours clause's 8,760-hour year; a
    366th day is refused rather than folded into a leap year the clause
    does not have.
14. **The hours clause keeps labels only when asked** (#594). `grouped`
    still drops the labels (`GroupLabels::Drop`), so every existing call,
    the cat guides' among them, gives the same result; a set that kept
    them would split `Tower::apply`'s result by label where it did not
    before. `GroupLabels::Agree` keeps the labels of an occurrence whose
    losses agree and refuses one whose losses disagree, naming the year,
    the time and the two labels; `GroupLabels::Largest` takes the largest
    loss's labels (the earliest of equal ones), the issue's other rule.
    `grouped_by(dim, ...)` keeps windows of different values of `dim`
    apart (a windstorm and a flood a day apart stay two occurrences),
    lists the occurrences by the time each window opens, and keeps the
    labels (`Drop` read as `Agree` in the bindings); it is one pass over
    the year, equal to grouping each value's set and merging them by
    time. An aggregate part keeps its labelled columns when the labels
    are kept and is summed into one column when they are dropped, as
    before. Whether `grouped` should keep labels by default is left for
    the user: switching it would change the result of every call on a
    labelled set.

15. **Losses stored as negative numbers are read with a flag, and a
    negative loss in a layer is an error** (#593, proposed by the
    Aggregate lane for the user's acceptance). Some vendor and cedent files
    write losses as money going out. `read_ylt(..., losses_negative=True)`
    (R `losses_negative = TRUE`, Rust `YearLossTable::from_negative_losses`)
    reads them with the sign flipped and refuses a positive value, which
    such a file does not hold; without the flag a negative loss stays an
    error, naming its row. A negative loss given to a layer directly lay
    below its attachment and ceded nothing without a word: `Layer::try_ceded`,
    `try_ceded_by_event` and `Tower::try_ceded` refuse a negative or
    non-finite loss, and the Python and R `ceded`, `ceded_by_event` and the
    tower's `ceded` use them. The Rust `ceded` functions keep their `f64`
    results, since every path that builds an event set already checks its
    losses, with docs that point to the checked ones. There is no event
    loss table reader to give the flag: an `EventLossTable` is built from
    columns whose means and rates it already checks.

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
- A YLT with a time column read and grouped equals the hand-built set
  (times out of order and slots that would order them otherwise); equal
  times fall back on the slots; times outside `[0, 1]` are refused. An
  hours clause on a set with one label value keeps it under either rule
  and drops it by default; mixed labels are refused under `Agree` and
  take the largest loss's under `Largest`; grouping by peril keeps two
  perils' windows apart and equals the plain clause on one peril (unit
  tests in `ylt.rs` and `monte_carlo.rs`; the same file read and grouped
  in Python and R, and day and hour columns).
- A file of negative losses read with the flag equals its sign-flipped
  copy; a positive value in it, or a negative loss without the flag, is
  refused naming the row; a negative or non-finite loss given to a layer,
  its events or a tower is an error in the checked calls and the bindings
  (unit tests in `ylt.rs` and `reinsurance.rs`; Python and R tests).

## Next

- Clash across lines is done in `prospicio_aggregate::bivariate`
  (`docs/design/aggregate.md`, "Two lines and clash"); #196 closed without
  it.
