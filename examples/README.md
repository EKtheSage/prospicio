# Examples

Worked analyses that use the library end to end, as Jupyter notebooks
saved with their results, so GitHub shows the numbers and charts without
running anything. CI runs each notebook's code with fewer simulations
(`python/tests/test_examples.py`), so they stay current.

| Example | What it shows |
|---|---|
| [`one_year_view.ipynb`](one_year_view.ipynb) | A small P&C insurer's year ahead: reserve risk (ODP bootstrap of GenIns), fleet motor premium risk (Poisson and gamma GLMs into a collective model), property per-risk (risk profile, surplus inuring to a per-risk XL with a reinstatement pro rata as to time), a risk-loaded XL price, then the three risks joined with a rank correlation and TVaR 99% allocated by Euler, gross and net of reinsurance |
| [`reinsurance_pricing.ipynb`](reinsurance_pricing.ipynb) | Six reinsurance contracts priced on the FFT grid and compared with Mildenhall's `aggregate` (321 independent comparisons against its reference file, `aggregate` not needed): an occurrence tower, a two-layer program and a share, an aggregate limit with reinstatements, stop loss and aggregate deductible, retro and swing premium with profit and sliding-scale commission, and risk-adjusted premiums from distortions at a capital standard |

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
open the notebook in Jupyter. To run a notebook against unreleased changes
in this repository, build the package from source instead
(`cd python && maturin develop`, which needs Rust). To save it again with
fresh outputs:

```sh
jupyter nbconvert --to notebook --execute --inplace examples/one_year_view.ipynb
```
