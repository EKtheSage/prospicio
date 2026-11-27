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
first; `ProcessDistribution` is the pilot that #990 named, and
`MackProcess` followed it (Excel's error had listed its names in another
order than Python's and R's; all three now list them in `NAMES`' order:
gamma, lognormal, residuals, normal, none). The next families, each in a
PR of its own, are the ones whose parsers are copied in two or three
front ends: `PairFamily`, `Archimedean`, `AnnualReading`, `Statistic`,
`Smoothing` and `CorrelationKind`.

## Method documentation (#991)

A method's definition, formulas, assumptions and references are written
once, in the Rust doc comment of the type or function that implements it
(`docs/architecture.md`, "Documentation"). A binding's documentation
describes the language API: what the object is for in a sentence or two,
its parameters, defaults, return values, errors and examples. In place of
the method's exposition it links to the Rust reference on docs.rs, at the
item's page under `latest`:

- **Python** (numpydoc in the PyO3 doc comment): a sentence that ends with
  the bare URL, as in "The method, its formulas and its reference are
  documented once, in the Rust reference: https://docs.rs/...". A bare
  URL is clickable on the great-docs site and readable in `help()`.
- **R** (roxygen2 with markdown): the same sentence with a markdown link,
  `[Rust reference](https://docs.rs/...)`, which roxygen writes as
  `\href{}{}`: a link on the pkgdown site and in the HTML help, and the
  URL in text help.

The link names the crate that defines the item (`prospicio-pricing`, not
the umbrella `prospicio`). A formula that
the user needs to read a return value (a column's definition, say) may
stay in the binding's `Returns` section.

Limited fluctuation credibility (`LimitedFluctuation`,
`credibility_standard()`) is the pilot. The next families, a PR each, are
the ones whose binding docs repeat formulas: the Mack bootstrap, then the
other credibility models (Buhlmann-Straub, Hachemeister).
