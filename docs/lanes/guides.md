# Guides lane

**Session:** Local (the guides session). **Owns:** `examples/`, `docs/guides/`, `python/tests/test_guides.py`, `R/prospicio/tests/test-guides.R`, `test_reinsurance_pricing` and future notebook tests in `python/tests/test_examples.py`, the `*.ipynb` lines in `.gitattributes`

Knowledge, documentation, guides and how-tos for actuaries.

This lane writes about the library and does not change crates; a gap it finds becomes an issue for the lane that owns the code

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/guides.md](done/guides.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next

- [guides-open] open, waiting on the user
- [guides-current-next] keep the guides and notebooks current as the library changes
