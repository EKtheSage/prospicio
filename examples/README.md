# Examples

Worked analyses that use the library end to end, as Jupyter notebooks. The
repository keeps their code only: the notebooks are committed without
outputs, so GitHub shows the code and not the results. The results (the
printed numbers, tables and charts) are on the
[docs site](https://ekthesage.github.io/prospicio/), which runs every
notebook in full each time it is built and publishes it in the Python
site's User Guide, under "Worked examples". CI also runs each notebook's
code with fewer simulations (`python/tests/test_examples.py`), so a change
that breaks one fails its pull request. The full run and the site build
happen only after a merge to `main` (`.github/workflows/pages.yml`): a pull
request that touches only `examples/` or `docs/guides/` does not build the
site, so a notebook that fails only with the full simulation counts, or a
page that does not render, shows up first in that Pages run.

| Example | Results | What it shows |
|---|---|---|
| [`one_year_view.ipynb`](one_year_view.ipynb) | [On the site](https://ekthesage.github.io/prospicio/python/user-guide/one_year_view.html) | A small P&C insurer's year ahead: reserve risk (the one-year claims development result of GenIns, re-reserved on the ODP bootstrap), fleet motor premium risk (Poisson and gamma GLMs into a collective model), property per-risk (risk profile, surplus inuring to a per-risk XL with a reinstatement pro rata as to time), the XL placed at its risk-loaded price, then the three risks joined with a rank correlation and TVaR 99% allocated by Euler, gross and net of reinsurance and its cost |
| [`reinsurance_pricing.ipynb`](reinsurance_pricing.ipynb) | [On the site](https://ekthesage.github.io/prospicio/python/user-guide/reinsurance_pricing.html) | Six reinsurance contracts priced on the FFT grid and compared with Mildenhall's `aggregate` (321 independent comparisons against its reference file, `aggregate` not needed): an occurrence tower, a two-layer program and a share, an aggregate limit with reinstatements, stop loss and aggregate deductible, retro and swing premium with profit and sliding-scale commission, and risk-adjusted premiums from distortions at a capital standard |
| [`reading_path.ipynb`](reading_path.ipynb) | [On the site](https://ekthesage.github.io/prospicio/python/user-guide/reading_path.html) | A map to the guides: one small synthetic book (a property line's claim count and severity, a casualty paid triangle) carried through seven stations, each running a few lines of prospicio and then naming the guides to read next and the question each answers: severity and claim count, the annual aggregate by FFT, a per-risk layer by exposure and experience rating blended by credibility and priced on a pentagon, a loss ratio indication, chain ladder, Mack and the ODP bootstrap, five distortions calibrated to the same return and their natural allocation, and pointers to the Bayesian and GLM guides. Runs in a few seconds |
| [`submission_evaluation.ipynb`](submission_evaluation.ipynb) | [On the site](https://ekthesage.github.io/prospicio/python/user-guide/submission_evaluation.html) | A reinsurance submission evaluated end to end, from raw files to a quote decision: a synthetic commercial property per-risk program (premium and rate changes, a large-loss listing, a triangle, a risk profile, treaty terms in `data/submission/`, written by `data/make_submission.py` with the answer key in `data/answer_key.json`). On-level premium and trend, chain ladder and Cape Cod, development and trend of large losses, a Pareto fit with truncation and censoring, experience and exposure rating blended by credibility, the program on the grid, risk metrics, technical premium against the broker's rates on line, sensitivity and parameter uncertainty, and the estimates against the generator's true parameters |
| [`ils_deal_analysis.ipynb`](ils_deal_analysis.ipynb) | [On the site](https://ekthesage.github.io/prospicio/python/user-guide/ils_deal_analysis.html) | A collateralised quota share analysed for an ILS investor, from the submission's triangle to the investor's IRR and MOIC: a diagonal backtest, an industry prior given as a mean and a cv (peer programs synthesised), the Bayesian Bornhuetter-Ferguson, a 2026 loss ratio, the structure with a flat and a sliding-scale commission, collateral and monthly cash flows, downside and risk ratios, the two structures compared, and monitoring against the one-year view. The steps the library does not cover yet are marked as workarounds, with their issues (#598 to #605). Runs in well under a minute |

For explanations in prose, with runnable code and checked numbers, see the
[guides](../docs/guides/README.md) (`reading_path.ipynb` is a route through them); the first, [What a reinsurance price is](../docs/guides/reinsurance-price.md),
explains the pricing that the notebook `reinsurance_pricing.ipynb` checks.
The guides are on the docs site too, in the same User Guide.

## R versions

`one_year_view.qmd`, `reinsurance_pricing.qmd`, `reading_path.qmd` and `ils_deal_analysis.qmd` are Quarto twins of the notebooks of the same names for the R package, with the same data, seeds and model. Rendered against the library they give the notebooks' numbers: `one_year_view.qmd` and `reading_path.qmd` every printed digit, `reinsurance_pricing.qmd` the same 343 comparisons (321 independent). Install the R package (`remotes::install_github`, see `R/prospicio/bootstrap.R`) and `ggplot2`, then from anywhere in the repository:

```sh
quarto render examples/one_year_view.qmd
quarto render examples/reinsurance_pricing.qmd
quarto render examples/reading_path.qmd
quarto render examples/ils_deal_analysis.qmd
```

They are also pages of the docs site, under "Articles", "Worked examples" on the R site: `cargo xtask r` stages them as pkgdown articles and runs them in full, and fails if `reinsurance_pricing.qmd` reports a comparison outside its tolerance. That build runs in CI's `docs` job when a pull request touches the bindings, `R/` or `xtask/`, and in the Pages run after a merge to `main`; a pull request that touches only `examples/` does not render them. Rendered `.html` is not committed.

## Data for the guides

Some guides read small synthetic data sets kept in `data/`, each written by a
seeded script next to it, which a test in `python/tests/test_guides.py`
checks the committed files against:

- `chalk_portfolio.csv` and `chalk_toys.csv`, written by
  `make_chalk_portfolio.py`, for the CAS Monograph 16 guides.
- `cat_book_yelt.csv`, `cat_book_elt_alder.csv`, `cat_book_cedents.csv` and
  `cat_book_contracts.csv`, written by `make_cat_book.py` (numpy only, seed
  20261023), for the property-catastrophe track (`docs/guides/cat-*.md`): a
  fictional reinsurer's contracts for three cedents, an event loss table and
  2,000 simulated years of dated events, in millions of a synthetic currency.
  The guide [`cat-0-overview.md`](../docs/guides/cat-0-overview.md) describes
  them.

## Running a notebook

To run one, clone the repository (the notebooks read their data from
`validation/` and `examples/data/`) and install the released package with Jupyter and
`matplotlib` for the charts:

```sh
pip install prospicio matplotlib jupyter
```

With [uv](https://docs.astral.sh/uv/) nothing needs installing: from
`examples/`, this starts Jupyter in a temporary environment with the
released package:

```sh
uv run --no-project --with prospicio --with matplotlib --with jupyter jupyter lab
```

PyPI has prebuilt wheels for Python 3.9
and later on Windows, macOS and Linux, so no Rust toolchain is needed. Then
open the notebook in Jupyter.

To run a notebook against unreleased changes in this repository, use the
uv project in `python/` (needs Rust). `submission_evaluation.ipynb` needs
this route until the release after 0.0.2, since it uses functions that
0.0.2 does not have (`RateHistory`, `Period`, `occurrence_layer_statistics`
and others). From the repository root:

```sh
uv run --project python --group examples jupyter lab
```

This keeps one environment in `python/.venv` with prospicio built from
source, Jupyter, `matplotlib` and `nbstripout`. uv rebuilds the extension
whenever a Rust source changes (`[tool.uv] cache-keys` in
`python/pyproject.toml`), so the notebook never runs a stale build.

## Keeping outputs out of git

Outputs (and execution counts) are never committed. A git filter,
[nbstripout](https://github.com/kynan/nbstripout), removes them from a
notebook whenever it is staged, so running a notebook and saving it leaves
nothing to commit and no pull request to open. `.gitattributes` already
names the filter for `*.ipynb`; each clone registers the filter's command
once, from the repository root:

```sh
uv run --project python --group examples nbstripout --install --attributes .gitattributes
git config filter.nbstripout.extrakeys "metadata.language_info.version"
```

The first line writes the filter into `.git/config`, pointing at
`python/.venv`'s Python (it leaves `.gitattributes` as it is, since the
lines are there). The second line also strips the Python version, which
Jupyter records in the notebook on every run. Without uv, `pip install
nbstripout` and run `nbstripout --install --attributes .gitattributes`
instead of the first line. If `python/.venv` is deleted, run the first line
again: the filter is marked required, so `git add` of a notebook fails
until its command works.

After a run, `git status` may still list the notebook as modified, because
git compares file sizes before it applies the filter. `git diff` shows
nothing, and `git add` stages nothing and clears the listing. Without the
filter a run does show as a change; do not commit it
(`python/tests/test_examples.py` fails on a notebook staged with outputs).
