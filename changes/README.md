# Change fragments

`knowledge/log.md` and `CHANGELOG.md` are shared by every PR, so two PRs
that each added a line at the top conflicted on every merge. Instead, a PR
adds one small file of its own here, and `validation/scripts/fold_changes.py`
folds the files into the two shared files later. Nobody edits
`knowledge/log.md` or `CHANGELOG.md` by hand for a new entry.

Name each file after the branch with `/` turned into `-`, so names are
unique per PR (`claude/bayes-cay` becomes `claude-bayes-cay`). Write plain
ASCII punctuation and no em dashes.

## `changes/knowledge/<YYYY-MM-DD>-<slug>.md`

One or more bullets in the log's format. The date in the name is the
`## YYYY-MM-DD` heading the bullets go under; it must be a real date and
not later than tomorrow. Links are root-absolute within `knowledge/` and
must resolve to a file (an http(s) URL is also allowed; a relative link is
rejected). The kind is one of Creation, Update,
Correction, Deprecation or Note. A bullet is one line; a continuation line
starts with two spaces.

```markdown
* **Creation**: [Burn 0.21's activations in f64](/references/burn.md), from an audit of `prospicio-nn` (branch claude/sharp-wilbur-a14ce0): the default `sigmoid` casts to f32.
```

File name: `2026-10-13-claude-sharp-wilbur-a14ce0.md`.

## `changes/changelog/<slug>.md`

One or more `### <Section>` headings, each followed by bullets for
`CHANGELOG.md`'s "Unreleased" section (`- ` bullets, continuation lines
indented two spaces). Use a section name `CHANGELOG.md` already has
(Distributions and probability, Aggregate and reinsurance, Examples, Guides,
Reserving, Pricing, ...). A new name is allowed: the fold creates the
section and reports it. Mark breaking changes and give the PR number as
`CHANGELOG.md` does: the number exists only once the PR is open, so open
the PR first, then push the number into the fragment. Bullets for the same
section, from one fragment or several, are merged in file-name order.

```markdown
### Pricing

- The overall rate indication in `prospicio_pricing::ratemaking`, matching
  Werner and Modlin's Appendices A to C (#180).
```

A PR with no user-facing change adds no changelog fragment.

## Checking and folding

```sh
python3 validation/scripts/fold_changes.py --check      # CI runs this on every PR
python3 validation/scripts/fold_changes.py --dry-run    # show the diff of a fold
python3 validation/scripts/fold_changes.py              # fold, then delete the fragments
```

The fold puts each knowledge fragment at the top of its date's section in
`knowledge/log.md` (creating the heading in newest-first position when it
is missing; fragments of one date in file-name order), appends each
changelog fragment's bullets to the end of the matching section of
"Unreleased" (creating a missing section at the end), and deletes the
folded files. Running it twice changes nothing the second time.

Only one PR folds at a time. Between releases a session folds only when the
user asks, in a small PR of its own that contains nothing else; until then,
read this directory together with `knowledge/log.md` for the latest entries.
The release PR always folds before it moves "Unreleased" into a version
section (`docs/release.md`).
