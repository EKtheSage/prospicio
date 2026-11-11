# Probability lane

**Session:** Cloud. **Owns:** `crates/prospicio-prob/`, `crates/prospicio-math/`, `validation/tests/distributions.rs`

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/probability.md](done/probability.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next

- [reference-generators] #844 items 2 and 3: pinned environments for the Python generators (PEP 723 inline metadata or one requirements file per reference package family, so `uv run validation/scripts/<name>.py` rebuilds a reference) and a weekly job, outside the required checks, that reruns the cheap pure-Python generators and compares their output with the committed files within each row's tolerance
