# Agent instructions

Shared guidance for every coding agent working in this repo: local Claude Code, Claude Code cloud sessions, and Codex. `CLAUDE.md` imports this file.

## Several agent sessions at once

A local session and one or more cloud sessions may be working on this repo at the same time. They do not share a working tree. The GitHub remote is the only thing they have in common.

- **One branch per session.** Never push to a branch that another session owns. Cloud sessions use `claude/<name>` branches.
- **One git worktree per session or task.** Start every piece of work in its own worktree, never in the main checkout, so that uncommitted edits, builds and branches of different sessions cannot mix: `git fetch origin && git worktree add -b <branch> ../risk-rs-<name> origin/main`, then work, run the checks and push from there, and open the PR from there. The main checkout (`risk-rs/`) holds whatever the user is editing by hand: do not write into it, stash it or reset it. The editing tools resolve relative paths against the main checkout and refuse paths outside the workspace, so edit worktree files through the shell with absolute paths, and check `git status` in the main checkout afterwards. When the PR merges, `git worktree remove ../risk-rs-<name>` and delete the branch. `git worktree list` shows what exists.
- **Nothing is visible until it is pushed.** Commit and push often enough that other sessions can fetch your work. Before you start work that depends on another session, run `git fetch` and look at its branch.
- **Stay inside your assigned scope.** The user gives each session its own work (crates, files, or a phase). Do not edit files outside that scope without asking. If you must, say so clearly in your PR.
- **Merge through PRs into `main`.** Only the user merges. After a merge, other sessions rebase onto or merge `main` before they continue.
- **Record shared decisions in the repo.** Put them in this file or in `docs/`, not only in chat, so every session picks them up on its next pull.
- **Record collected knowledge in `knowledge/`.** Facts found while working (datasets, reference implementations and their quirks, numerical findings, environment facts) go into the bundle in `knowledge/`, in the [Open Knowledge Format v0.2](https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md): one markdown file per concept with YAML frontmatter (`type`, `title`, `description`, `tags`, `sources`, `generated`, and `verified` when a test that CI runs confirms it), listed in its directory's `index.md`, with a dated entry written as a fragment in `changes/knowledge/` (see "Change fragments" below), not in `knowledge/log.md`. Design decisions stay in `docs/design/`. `validation/scripts/check_okf.py` checks the bundle in CI.
- **Leave the docs current in the item's own PR.** The PR that finishes an item also updates everything that describes it: its lane's status file in `docs/lanes/` (move the item's bullet to Done, set Next; see "Lane status files" below), its design note's status and decisions, `docs/roadmap.md` (its lane section and milestone rows) and `docs/architecture.md` if a roadmap row changes, a fragment in `changes/changelog/` for a user-facing change (breaking ones marked), which becomes a line under "Unreleased" in `CHANGELOG.md`, `CLAUDE.md` or this file if a working rule changed, and `knowledge/` (next bullet). Never leave these for a follow-up PR; a reviewer treats a stale status bullet or status as a defect of the PR.
- **Update the knowledge bundle when an item is done.** Before you report a finished work item, add what it taught you to `knowledge/`: new or corrected concepts (a dataset, a reference implementation's quirk, a numerical finding, an environment fact), `verified` on facts that a CI test now confirms, `status: deprecated` on facts it made wrong, and a dated fragment in `changes/knowledge/`. Do it in the item's own PR. If the item taught nothing new, say so in the report.
- **Change fragments, not edits to the shared logs.** `knowledge/log.md` and `CHANGELOG.md` are folded from small files so that PRs running at once do not conflict. A PR never edits those two files for a new entry; it adds `changes/knowledge/<YYYY-MM-DD>-<slug>.md` and, for a user-facing change, `changes/changelog/<slug>.md`, where the slug is the branch name with `/` turned into `-`. `changes/README.md` has the formats. CI runs `validation/scripts/fold_changes.py --check` on every PR. Only one PR folds at a time, so folds do not conflict either: the release PR always folds (`docs/release.md`), and between releases a session folds (`python3 validation/scripts/fold_changes.py`) only when the user asks, in a small PR of its own that contains nothing else. Until a fold, read `changes/knowledge/` together with `knowledge/log.md` for the latest entries.
- **Report when an item is done.** When you finish a work item (a PR merged, an issue closed, or a task the user asked for), end your reply to the user with a summary in three parts:
  1. **Work by session**: what each session worked on since the last report: this session's items with their PRs, and the other sessions' merged PRs, from `git log origin/main` and the open PRs.
  2. **Repo status**: `main` (the latest merges and whether CI is green), open PRs and issues, and anything blocked and on whom.
  3. **Next steps**: what comes next in your lanes, in order, and anything waiting on the user.
- **Messages go one way.** A local session can send a cloud session a message, but the cloud session cannot reply. Leave status and questions in your PR description or in commit messages, where the user and other sessions can read them.

## Work lanes

Work is split by crate so that sessions rarely touch the same file. A lane owns the files listed for it; edit another lane's files only with the user's go-ahead.

| Lane | Session | Owns | Status |
|---|---|---|---|
| Reserving | Local | `crates/prospicio-reserving/`, `validation/tests/reserving*.rs`, `docs/design/triangle.md`, `docs/design/reserving-v02.md`, `docs/design/reserving-bayes.md`, `docs/design/reserving-compartmental.md` | [docs/lanes/reserving.md](docs/lanes/reserving.md) |
| Probability | Cloud | `crates/prospicio-prob/`, `crates/prospicio-math/`, `validation/tests/distributions.rs` | [docs/lanes/probability.md](docs/lanes/probability.md) |
| Aggregate | Cloud (the Probability session) | `crates/prospicio-aggregate/`, `crates/prospicio-pricing/`, `validation/tests/aggregate.rs`, `validation/tests/pricing.rs`, `docs/design/aggregate.md`, `docs/design/pareto.md`, `docs/design/credibility.md`, `docs/design/ratemaking.md`, `docs/design/experience-rating.md`, `docs/design/time-value.md`, `docs/design/cat-inputs.md`, `docs/design/pnl.md`, `docs/design/casualty-exposure.md` | [docs/lanes/aggregate.md](docs/lanes/aggregate.md) |
| Models | Cloud (the Probability session) | `crates/prospicio-models/`, `crates/prospicio-glm/`, `crates/prospicio-nn/`, `crates/prospicio-bayes/`, `validation/tests/models.rs`, `validation/tests/fremtpl2.rs`, `docs/design/models.md` | [docs/lanes/models.md](docs/lanes/models.md) |
| Excel | Cloud (the Excel session) | `crates/prospicio-xll/`, `docs/design/xll.md`, the `xll` task in `xtask/src/main.rs`, `.github/workflows/xll.yml` | [docs/lanes/excel.md](docs/lanes/excel.md) |
| Guides | Local (the guides session) | `examples/`, `docs/guides/`, `python/tests/test_guides.py`, `R/prospicio/tests/test-guides.R`, `test_reinsurance_pricing` and future notebook tests in `python/tests/test_examples.py`, the `*.ipynb` lines in `.gitattributes` | [docs/lanes/guides.md](docs/lanes/guides.md) |

The ODP bootstrap needs both a `Triangle` and a `PredictiveDistribution`. It starts in the Reserving lane after both lanes have merged.

### Lane status files

The table above holds only static facts. What a lane has done, is doing and will do next lives in its status file, `docs/lanes/<lane>.md` (the lane name in lowercase), with the sections `## Now` (open PRs), `## Next` and `## Done`.

- **One bullet per line.** Each item is a single physical line, never wrapped, so that two PRs editing different items never touch the same line.
- **A PR edits only the bullets about its own item.** It moves its item to Done and sets Next; it does not rewrite, reorder or reflow other bullets.
- **Union merge.** `docs/lanes/*.md` use git's union merge (`.gitattributes`: `docs/lanes/*.md merge=union`), which keeps both sides' added lines, so concurrent PRs do not conflict. After a merge, check that no bullet was kept twice.
- **Every agent reads every status file before planning.** Claude Code loads them with this file through the imports below (`CLAUDE.md` imports `AGENTS.md`, which imports each status file), so they are in context exactly as the old table was. Codex and any agent that does not expand `@` imports reads `docs/lanes/*.md` at the start of a task. A new lane adds its import line here.

@docs/lanes/reserving.md
@docs/lanes/probability.md
@docs/lanes/aggregate.md
@docs/lanes/models.md
@docs/lanes/excel.md
@docs/lanes/guides.md

### Shared files

- **`crates/prospicio-core/`** is used by every lane. Change it only in a small PR of its own that holds nothing else, so it can merge first. Other lanes merge `main` before they build on the change.
- **Bindings**: each lane puts its wrappers in its own module file (for example `crates/prospicio-python/src/reserving.rs`, `crates/prospicio-r/src/distributions.rs`, `R/prospicio/R/reserving.R`, `python/prospicio/reserving.py`). The crate's `lib.rs` only registers modules, one line per lane.
- **Root `Cargo.toml` and `Cargo.lock`**: add dependencies only when a lane needs them. When `Cargo.lock` conflicts, take either side and let `cargo` regenerate it. Do not merge it by hand.
- **`AGENTS.md` and `docs/architecture.md`**: a lane updates its own status file (`docs/lanes/<lane>.md`), and the roadmap row its item finishes, in the item's own PR; any other change, including the lane table's Session and Owns columns, goes in a small PR of its own. Each lane edits only its own design notes under `docs/design/`.
- **Releases**: `crates/prospicio` (the umbrella crate), `.github/workflows/release.yml` and `docs/release.md`. A new `prospicio-*` crate gets a `version` next to its path in `[workspace.dependencies]` and a re-export in `crates/prospicio/src/lib.rs`; check it with `cargo publish --workspace --dry-run`.
- **Keep PRs small** (one type or method each) and merge `main` often. The longer a branch lives, the harder its conflicts get.

## Checks

`.github/workflows/ci.yml` runs on every PR and on pushes to `main`. A PR merges only when CI is green on its latest commit.

To keep billed minutes down (the repo is private; Windows minutes count double), PRs run Ubuntu only, a push to `main` runs only the Rust job, the docs job builds only when a PR touches the bindings, `python/`, `R/`, `xtask/` or `ci.yml`, the slow suites run in release builds of their own jobs (`bayes`, `fremtpl2`) only when a PR touches their inputs, and the full matrix with Windows runs weekly and on demand (`workflow_dispatch`). Cargo builds are cached, and a new push cancels the branch's older run. Run the Actions tab's "Run workflow" before a release, or after a change that could break only on Windows.

Run the matching `ci.yml` checks locally before you push, so CI does not go red on something a local run would have caught:

- Always: `cargo fmt --all --check`, `cargo clippy --all-targets --features prospicio-validation/slow-bayes -- -D warnings`, `cargo test`, `python3 validation/scripts/fold_changes.py --check`
- Bayesian parity tests: a plain `cargo test` skips them (about 220 s in debug). When a PR touches `prospicio-reserving`, `prospicio-bayes`, `prospicio-prob`, `prospicio-core`, `prospicio-math`, `prospicio-glm`, `prospicio-models` or their validation files, run them in release (minutes now, with the one-year views' refits): `cargo test --release -p prospicio-validation --features slow-bayes --test 'reserving_bayes*' --test 'reserving_meyers*' --test 'reserving_compartmental*'`, or one file with `--test <name>`. They are the `[[test]]` entries of `validation/Cargo.toml` that require the `slow-bayes` feature; a new Bayesian parity file gets such an entry there and nothing else (CI's `bayes` job reads the list). The slowest of them, the Meyers and compartmental one-year validation, run a smoke version (Meyers' CRC alone; compartmental model 1 alone, without its refit comparison) unless `RISK_RS_FULL_BAYES` is set (`prospicio_validation::full_bayes`, the user's decision of 2026-10-16): pull requests run the smoke version, the weekly and manual (`workflow_dispatch`) runs set the variable. A new slow arm of that kind runs only under the gate, with one cheap model in the smoke run; read the gate at run time (`if full_bayes()`), not with `cfg`, so clippy keeps linting every arm. When a PR changes such a model, run the full version locally too: `RISK_RS_FULL_BAYES=1 cargo test --release ...`
- Bayesian unit tests: `prospicio-reserving`'s unit tests that sample with NUTS are ignored unless its `slow-tests` feature is on (`#[cfg_attr(not(feature = "slow-tests"), ignore = "...")]`), since in debug they took about 230 s single-threaded and about 5 minutes of CI's rust job. A plain `cargo test` still compiles them, and clippy lints them. Under the same conditions as the parity tests, run them in release (about 4 s): `cargo test --release -p prospicio-reserving --features slow-tests --lib`, as CI's `bayes` job does. A new unit test that fits with NUTS and takes more than about a second in debug gets the same attribute; a cheap check (errors, layouts, gradients, a short run) stays in the debug run. The seed-pinned `bayes::odp::tests::the_one_year_view_is_pinned_by_seed` stays plainly `#[ignore]`d (platform-specific): run it by name with `-- --ignored the_one_year_view_is_pinned_by_seed`
- Crates in the WASM scope touched (`prospicio-core`, `prospicio-math`, `prospicio-prob`, `prospicio-aggregate`, `prospicio-pricing`, `prospicio-reserving`, and the umbrella `prospicio`): `rustup target add wasm32-unknown-unknown` once, then `cargo check --target wasm32-unknown-unknown -p <crate>`. No threads, files or C dependencies on their default path
- Python bindings touched: `cargo clippy -p prospicio-python -- -D warnings`, then `cd python && maturin develop && pytest tests`
- R bindings touched: `cargo clippy -p prospicio-r -- -D warnings`, `install.packages("S7")` if missing, `R CMD INSTALL R/prospicio`, every `R/prospicio/tests/*.R` with `Rscript`
- Bindings, their doc comments or docs config touched: `cargo xtask docs --check` (needs Quarto, Python 3.11+ for great-docs, and the R packages `roxygen2`, `pkgload`, `pkgdown`), and commit the regenerated stub, `man/` and `NAMESPACE`
- A new Bayesian reserving model implements `YearModel` (`crates/prospicio-reserving/src/bayes/one_year.rs`, `reserving-bayes.md` decision 17), with its draws stored on an unconstrained scale and carrying with each posterior draw anything the year's cells inform, and passes the one-year conformance suite: `one_year::conformance::check` in its unit tests and an entry in `every_model` of `validation/tests/reserving_bayes_one_year_conformance.rs`. A change to a model that implements `YearModel` (now `BayesOdp`, `BayesBf`, `Meyers` and `Compartmental`) re-runs both halves of the suite

If a check cannot run in your environment (for example, R is not installed), rely on CI for it and say so in the PR.
