# Examples

Worked analyses that use the library end to end, as Jupyter notebooks. The
repository keeps their code only: the notebooks are committed without
outputs, so GitHub shows the code and not the results. The results (the
printed numbers, tables and charts) are on the
[docs site](https://ekthesage.github.io/prospicio/), which runs every
notebook in full each time it is built and publishes it in the Python
site's User Guide, under "Worked examples". CI also runs each notebook's
code with fewer simulations (`python/tests/test_examples.py`), so a change
that breaks one fails its pull request.

| Example | Results | What it shows |
|---|---|---|
| [`one_year_view.ipynb`](one_year_view.ipynb) | [On the site](https://ekthesage.github.io/prospicio/python/user-guide/one_year_view.html) | A small P&C insurer's year ahead: reserve risk (ODP bootstrap of GenIns), fleet motor premium risk (Poisson and gamma GLMs into a collective model), property per-risk (risk profile, surplus inuring to a per-risk XL with a reinstatement pro rata as to time), a risk-loaded XL price, then the three risks joined with a rank correlation and TVaR 99% allocated by Euler, gross and net of reinsurance |
| [`reinsurance_pricing.ipynb`](reinsurance_pricing.ipynb) | [On the site](https://ekthesage.github.io/prospicio/python/user-guide/reinsurance_pricing.html) | Six reinsurance contracts priced on the FFT grid and compared with Mildenhall's `aggregate` (321 independent comparisons against its reference file, `aggregate` not needed): an occurrence tower, a two-layer program and a share, an aggregate limit with reinstatements, stop loss and aggregate deductible, retro and swing premium with profit and sliding-scale commission, and risk-adjusted premiums from distortions at a capital standard |

For explanations in prose, with runnable code and checked numbers, see the
[guides](../docs/guides/README.md); the first, [What a reinsurance price is](../docs/guides/reinsurance-price.md),
explains the pricing that the notebook `reinsurance_pricing.ipynb` checks.
The guides are on the docs site too, in the same User Guide.

## Running a notebook

To run one, clone the repository (the notebooks read their data from
`validation/`) and install the released package with Jupyter and
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
uv project in `python/` (needs Rust). From the repository root:

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
