---
type: Environment Fact
title: Notebooks without outputs, executed on the docs site
description: The notebooks in examples/ are committed without outputs and kept so by an nbstripout git filter; cargo xtask python stages the guides and the executed notebooks as the great-docs User Guide, so their results live on the docs site; git status quirks of the filter, and great-docs 0.17's user guide limits.
tags: [environment, docs, notebooks, examples, guides, git, python]
status: stable
generated: { by: claude-code/local-session, at: 2026-10-13T00:00:00Z }
stale_after: 2027-01-13T00:00:00Z
sources:
  - id: readme
    resource: ../examples/README.md
    title: examples/README.md, Keeping outputs out of git
  - id: xtask
    resource: ../xtask/src/main.rs
    title: cargo xtask python, stage_user_guide
  - id: config
    resource: ../python/great-docs.yml
    title: python/great-docs.yml, user_guide
  - id: test
    resource: ../python/tests/test_examples.py
    title: test_notebooks_are_committed_without_outputs
---

# The policy

* Notebooks in `examples/` are committed without outputs, execution counts
  or the Python version in `metadata.language_info`, so re-running one never
  needs a commit. `.gitattributes` sets `*.ipynb filter=nbstripout` and
  `diff=ipynb`; each clone registers the command once (`examples/README.md`),
  with `filter.nbstripout.extrakeys` set to `metadata.language_info.version`,
  which nbstripout 0.9.1 does not strip by default.[^readme]
* nbstripout reads its settings only from git config and its command line:
  0.9.1 has no `pyproject.toml` section, so the extra key cannot be committed
  and is part of the per-clone setup.
* nbstripout renumbers cell ids `0`, `1`, ... unless `--keep-id`, and drops
  `cell.metadata.execution`; cell tags (`plot`) are kept. The committed
  notebooks were stripped with nbstripout itself, so the filter is a no-op on
  them.
* `test_notebooks_are_committed_without_outputs` reads the index copy (`git
  show :examples/<name>.ipynb`), not the working tree, so a local run with
  the filter does not fail it while a notebook staged with outputs does.[^test]

# git status with the filter

After a notebook is run in place, `git status` still lists it as modified:
git flags a file whose size differs from the size recorded in the index
without running the clean filter. `git diff` is empty, and `git add` stages
the same blob (checked: the index hashes before and after were equal) and
clears the listing. Checked on 2026-10-13 with nbstripout 0.9.1 and git for
Windows under `core.autocrlf=true`, the filter configured through
`GIT_CONFIG_COUNT` environment variables so that `.git/config` (shared by
all worktrees) was not touched.

# How the docs site gets them

* `cargo xtask python` syncs `python/` with the `docs` and `examples` groups,
  then, after the tests, stages the Python site's User Guide in
  `target/user-guide` (`stage_user_guide`): each `docs/guides/*.md` but
  `README.md` as `1-<name>.md` under the sidebar section "Guides", and each
  `examples/*.ipynb` executed by `jupyter nbconvert --to markdown --execute`
  (with `--ExecutePreprocessor.timeout=600`, since nbconvert's default stops
  a cell after 30 s) as `2-<name>.md`, with its charts in `<name>_files/`, under "Worked
  examples". The first `#` heading becomes the page's front matter `title`,
  and a guide's link `../../examples/<name>.ipynb` becomes `<name>.md`.[^xtask]
* `python/great-docs.yml` sets `user_guide: ../target/user-guide`. great-docs
  strips the numeric prefix from each page's file name, so the pages are
  `python/user-guide/<name>.html` on the site, and copies `<name>_files/` as
  an asset directory.[^config]
* `.github/workflows/pages.yml` runs `cargo xtask docs` on merges to `main`
  that touch `examples/` or `docs/guides/` too. The ci.yml `docs` job does not
  run for a PR that touches only those folders; the Python tests run the
  notebooks' and guides' code there.
* Executing both notebooks in full takes about 35 s on the local machine.

# great-docs 0.17.0 user guide limits

* The user guide reads `.qmd` and `.md` only; `.ipynb` files are not
  discovered, which is why the notebooks are exported to Markdown.
* With a custom directory (`user_guide: <dir>`) every `.md` file is a page,
  `README.md` included, and Quarto does not render `README.md` (the build
  left it as a raw file the sidebar linked to). An explicit `user_guide:` list
  cannot name a custom directory, so the guides are staged without their
  README.
* Custom `sections:` exclude `README.md` but derive the output directory
  from `dir`, so a directory outside `python/` (`../docs/guides`) would be
  written outside the site.
* On Windows great-docs wrote `_quarto.yml` in the locale's encoding and
  then failed to read it ("stream did not contain valid UTF-8") because the
  reference titles hold non-ASCII letters; `PYTHONUTF8=1` fixes it, and
  `cargo xtask python` now sets it.

[^readme]: examples/README.md, Keeping outputs out of git
[^xtask]: xtask/src/main.rs, stage_user_guide
[^config]: python/great-docs.yml
[^test]: python/tests/test_examples.py
