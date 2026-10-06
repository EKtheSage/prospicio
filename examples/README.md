# Examples

Worked analyses that use the library end to end, as Jupyter notebooks
saved with their results, so GitHub shows the numbers and charts without
running anything. CI runs each notebook's code with fewer simulations
(`python/tests/test_examples.py`), so they stay current.

| Example | What it shows |
|---|---|
| [`one_year_view.ipynb`](one_year_view.ipynb) | A small P&C insurer's year ahead: reserve risk (ODP bootstrap of GenIns), fleet motor premium risk (Poisson and gamma GLMs into a collective model), property per-risk (risk profile, surplus inuring to a per-risk XL with a reinstatement pro rata as to time), a risk-loaded XL price, then the three risks joined with a rank correlation and TVaR 99% allocated by Euler, gross and net of reinsurance |

To run one, build the package (`cd python && maturin develop`) and open
the notebook in Jupyter (the charts need `matplotlib`). To save it again
with fresh outputs:

```sh
jupyter nbconvert --to notebook --execute --inplace examples/one_year_view.ipynb
```
