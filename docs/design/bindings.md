# Binding conventions

Conventions that every front end (Python, R, Excel, JavaScript) follows
when it wraps a core crate. Each section names the issue that set it.

## Option names (#990)

A string option that several front ends parse (a process distribution, an
interpolation rule, a family name) has its vocabulary defined once, on the
Rust type that owns it:

- `pub const NAMES: &'static [(&'static str, Self)]`: every name a front
  end accepts, with its variant, the canonical name of each variant first
  and any alias after it.
- `pub fn from_name(name: &str) -> Option<Self>`: the lookup in `NAMES`.
- `pub fn name(self) -> &'static str`: the canonical name, for a front end
  that reports the option back (a Python property, an R print method).

A front end does not repeat the table. It normalizes its own input first,
then looks the name up, and on `None` builds its error with its crate's
`unknown_name(arg, got, NAMES)`, which lists the accepted names in the
form `process must be "gamma" or "none", got "poisson"`:

- **Python** takes the string as given.
- **R** narrows the argument with `match.arg` in the R wrapper, so
  abbreviations keep working, and passes the full name to Rust.
- **Excel** reads the cell with `text_option`, which trims and lower-cases
  it, and maps a blank cell to the type's `Default`.

The default of an omitted option stays in each front end's signature
(Python's keyword default, R's first `match.arg` choice, Excel's blank
cell), so the published signatures do not change.

`Average`, `SigmaInterpolation` and `Interpolation` followed this pattern
first; `ProcessDistribution` is the pilot that #990 named. The next
families, each in a PR of its own, are the ones whose parsers are copied
in two or three front ends, starting with those whose accepted names
already differ: `MackProcess` (Excel lists its names in another order than Python and R), then
`PairFamily`, `Archimedean`, `AnnualReading`, `Statistic`, `Smoothing`
and `CorrelationKind`.
