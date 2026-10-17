# Guides

Explanations for actuaries, in prose, written around what the library does.
Each guide starts from a question an actuary asks (what is a reinsurance price,
how is a submission evaluated) and answers it with a running example, so a
reader who knows primary pricing can follow it without the reinsurance
vocabulary. The notebooks in [`examples/`](../../examples/README.md) are
worked analyses; the guides are the reading that goes before them.

The guides are also published on the [docs site](https://ekthesage.github.io/prospicio/),
in the Python site's User Guide beside the notebooks, which the site runs so
their results show there (the repository keeps the notebooks without outputs).
`cargo xtask python` builds that User Guide from this folder: every `.md` file
here but this README becomes a page, its first `#` heading the page's title,
and a link to a notebook in `examples/` points to the notebook's page.

Every guide's code is tested. `python/tests/test_guides.py` runs each guide's
```` ```python ```` blocks in order in one namespace, and when a block is
followed by a ```` ```text ```` block, requires the block to print exactly that
text, so a number in a guide cannot go stale when the library changes. A block
marked ```` ```python no-run ```` is shown but not run. To run the tests:

```sh
cd python
maturin develop
pytest tests/test_guides.py
```

Start here: [`choosing-a-method.md`](choosing-a-method.md). It maps each
situation (a rate indication, a per-risk or casualty layer, a catastrophe cover,
a quota share, a stop loss, a reserve, capital) to the method and the calls,
and sends you to the guide that goes deeper.

| Guide | What it answers |
|---|---|
| [`choosing-a-method.md`](choosing-a-method.md) | Which prospicio tool for which job: a decision table over the situations an actuary meets, the calls, the traps and where the library stops |

**Pricing**

| Guide | What it answers |
|---|---|
| [`reinsurance-price.md`](reinsurance-price.md) | What a reinsurance price is: what is priced and what comes out, the technical premium of an excess of loss layer and a quota share, how price follows a risk profile without a rating plan, a contract alone and in a book, hard and soft markets, what the reinsurer decides, and what the model leaves out |
| [`experience-and-exposure-rating.md`](experience-and-exposure-rating.md) | Rating a layer from the cedant's own losses (preparing them as-if for currency, development, trend and an index clause, then the burning cost, as-if history under past programs, a fitted Pareto), from its exposure (curves, increased limit factors, risk profiles), and how to blend the two |
| [`risk-loads-and-distortions.md`](risk-loads-and-distortions.md) | Which risk load or distortion to use (constant cost of capital, proportional hazard, Wang, TVaR and others), how to calibrate it to quotes, and how to price a contract in a portfolio |
| [`capital-modeling-monograph-15.md`](capital-modeling-monograph-15.md) | CAS Monograph 15 (Major and Mildenhall) reproduced: the InsCo example priced by five distortions and allocated to units, the 35 xs 65 reinsurance decision, and the correlated three-unit book, with the monograph's printed numbers beside prospicio's |

**Distributions**

| Guide | What it answers |
|---|---|
| [`severity-and-frequency.md`](severity-and-frequency.md) | Choosing and fitting a severity and a claim count: lognormal, Pareto family, mixtures, censoring and truncation, and count families by dispersion |
| [`aggregate-distributions.md`](aggregate-distributions.md) | The distribution of a year's total loss: FFT, Panjer, simulation or moments, the grid, and when each is exact, fast or unsafe |

**Reserving**

| Guide | What it answers |
|---|---|
| [`reserving-methods.md`](reserving-methods.md) | Which reserving method when: chain ladder, Mack, Bornhuetter-Ferguson, bootstraps, Bayesian models, tails and the one-year view, with the reserve's uncertainty as a distribution |

Money in the guides is in millions of a synthetic currency, and every example is
synthetic. The guides write about the library and do not change it; a gap found
while writing one becomes an issue for the lane that owns the code
(`AGENTS.md`).
