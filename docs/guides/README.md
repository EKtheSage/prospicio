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

| Guide | On the site | What it answers |
|---|---|---|
| [`reinsurance-price.md`](reinsurance-price.md) | [What a reinsurance price is](https://ekthesage.github.io/prospicio/python/user-guide/reinsurance-price.html) | What a reinsurance price is: what is priced and what comes out, the technical premium of an excess of loss layer and a quota share, how price follows a risk profile without a rating plan, a contract alone and in a book, hard and soft markets, what the reinsurer decides, and what the model leaves out |

Money in the guides is in millions of a synthetic currency, and every example is
synthetic. The guides write about the library and do not change it; a gap found
while writing one becomes an issue for the lane that owns the code
(`AGENTS.md`).
