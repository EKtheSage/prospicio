# Examples

Worked analyses that use the library end to end. CI runs each one with
fewer simulations (`python/tests/test_examples.py`), so they stay current.

| Example | What it shows |
|---|---|
| [`one_year_view.py`](one_year_view.py) | A small P&C insurer's year ahead: reserve risk (ODP bootstrap of GenIns), fleet motor premium risk (Poisson and gamma GLMs into a collective model), property per-risk (risk profile, surplus inuring to a per-risk XL with a reinstatement pro rata as to time), a risk-loaded XL price, then the three risks joined with a rank correlation and TVaR 99% allocated by Euler, gross and net of reinsurance |

Run from the repository root after `cd python && maturin develop`:

```sh
python examples/one_year_view.py
```
