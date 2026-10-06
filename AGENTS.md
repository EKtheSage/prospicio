# Agent instructions

Shared guidance for every coding agent working in this repo: local Claude Code, Claude Code cloud sessions, and Codex. `CLAUDE.md` imports this file.

## Several agent sessions at once

A local session and one or more cloud sessions may be working on this repo at the same time. They do not share a working tree. The GitHub remote is the only thing they have in common.

- **One branch per session.** Never push to a branch that another session owns. Cloud sessions use `claude/<name>` branches.
- **Nothing is visible until it is pushed.** Commit and push often enough that other sessions can fetch your work. Before you start work that depends on another session, run `git fetch` and look at its branch.
- **Stay inside your assigned scope.** The user gives each session its own work (crates, files, or a phase). Do not edit files outside that scope without asking. If you must, say so clearly in your PR.
- **Merge through PRs into `main`.** Only the user merges. After a merge, other sessions rebase onto or merge `main` before they continue.
- **Record shared decisions in the repo.** Put them in this file or in `docs/`, not only in chat, so every session picks them up on its next pull.
- **Record collected knowledge in `knowledge/`.** Facts found while working (datasets, reference implementations and their quirks, numerical findings, environment facts) go into the bundle in `knowledge/`, in the [Open Knowledge Format v0.2](https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md): one markdown file per concept with YAML frontmatter (`type`, `title`, `description`, `tags`, `sources`, `generated`, and `verified` when a test that CI runs confirms it), listed in its directory's `index.md`, with a dated entry in `knowledge/log.md`. Design decisions stay in `docs/design/`. `validation/scripts/check_okf.py` checks the bundle in CI.
- **Update the knowledge bundle when an item is done.** Before you report a finished work item, add what it taught you to `knowledge/`: new or corrected concepts (a dataset, a reference implementation's quirk, a numerical finding, an environment fact), `verified` on facts that a CI test now confirms, `status: deprecated` on facts it made wrong, and a dated entry in `knowledge/log.md`. Do it in the item's own PR where you can, or in a follow-up PR right after. If the item taught nothing new, say so in the report.
- **Report when an item is done.** When you finish a work item (a PR merged, an issue closed, or a task the user asked for), end your reply to the user with a summary in three parts:
  1. **Work by session**: what each session worked on since the last report: this session's items with their PRs, and the other sessions' merged PRs, from `git log origin/main` and the open PRs.
  2. **Repo status**: `main` (the latest merges and whether CI is green), open PRs and issues, and anything blocked and on whom.
  3. **Next steps**: what comes next in your lanes, in order, and anything waiting on the user.
- **Messages go one way.** A local session can send a cloud session a message, but the cloud session cannot reply. Leave status and questions in your PR description or in commit messages, where the user and other sessions can read them.

## Work lanes

Work is split by crate so that sessions rarely touch the same file. A lane owns the files listed for it; edit another lane's files only with the user's go-ahead.

| Lane | Session | Owns | Current work |
|---|---|---|---|
| Reserving | Local | `crates/act-reserving/`, `validation/tests/reserving.rs`, root `src/` | `Triangle` per `docs/design/triangle.md`, Chain Ladder, Mack, parity on RAA / GenIns / ABC; delete the `src/` sandbox once `act-reserving` covers it |
| Probability | Cloud | `crates/act-prob/`, `crates/act-math/`, `validation/tests/distributions.rs` | Done: distributions, `Grid`, `Counting`, Pareto family, `PredictiveDistribution` (join, reorder, blend), distortions, allocation, copulas, EVT, the `Dist` enum. Next: Python and R bindings on `Dist`, then `Custom` (`docs/design/distributions.md`) |
| Aggregate | Cloud (the Probability session) | `crates/act-aggregate/`, `crates/act-pricing/`, `validation/tests/aggregate.rs`, `validation/tests/pricing.rs`, `docs/design/aggregate.md`, `docs/design/pareto.md` | Done: compound distributions, towers (in the bindings' `reinsurance` namespace), collective model, layer rating, tower matching, risk-loaded pricing, MBBEFD exposure curves. Next: surplus treaties (needs sums insured per risk; `docs/design/aggregate.md`) |
| Models | Cloud (the Probability session) | `crates/act-models/`, `crates/act-glm/`, `crates/act-nn/`, `crates/act-bayes/`, `validation/tests/models.rs`, `validation/tests/fremtpl2.rs`, `docs/design/models.md` | Done: `act-models`, `act-glm` (GLM, GAM, elastic net, Tweedie; statsmodels parity incl. freMTPL2), `act-nn`, `act-bayes` (NUTS, Bayesian GLM, stacking), boosting adapters over LightGBM and XGBoost (Python and R). Next: quantile and distributional boosting objectives (`docs/design/models.md`). The triangle-to-design bridge and the diagonal backtest stay in the Reserving lane |

The ODP bootstrap needs both a `Triangle` and a `PredictiveDistribution`. It starts in the Reserving lane after both lanes have merged.

### Shared files

- **`crates/act-core/`** is used by every lane. Change it only in a small PR of its own that holds nothing else, so it can merge first. Other lanes merge `main` before they build on the change.
- **Bindings**: each lane puts its wrappers in its own module file (for example `crates/act-python/src/reserving.rs`, `crates/act-r/src/distributions.rs`, `R/actuarialrs/R/reserving.R`, `python/actuarialrs/reserving.py`). The crate's `lib.rs` only registers modules, one line per lane.
- **Root `Cargo.toml` and `Cargo.lock`**: add dependencies only when a lane needs them. When `Cargo.lock` conflicts, take either side and let `cargo` regenerate it. Do not merge it by hand.
- **`AGENTS.md` and `docs/architecture.md`**: change them in small PRs. Each lane edits only its own design notes under `docs/design/`.
- **Keep PRs small** (one type or method each) and merge `main` often. The longer a branch lives, the harder its conflicts get.

## Checks

`.github/workflows/ci.yml` runs on every PR and on pushes to `main`. A PR merges only when CI is green on its latest commit.

To keep billed minutes down (the repo is private; Windows minutes count double), PRs run Ubuntu only, a push to `main` runs only the Rust job, the docs job builds only when a PR touches the bindings, `python/`, `R/`, `xtask/` or `ci.yml`, and the full matrix with Windows runs weekly and on demand (`workflow_dispatch`). Cargo builds are cached, and a new push cancels the branch's older run. Run the Actions tab's "Run workflow" before a release, or after a change that could break only on Windows.

Run the matching `ci.yml` checks locally before you push, so CI does not go red on something a local run would have caught:

- Always: `cargo fmt --all --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`
- Python bindings touched: `cargo clippy -p act-python -- -D warnings`, then `cd python && maturin develop && pytest tests`
- R bindings touched: `cargo clippy -p act-r -- -D warnings`, `install.packages("S7")` if missing, `R CMD INSTALL R/actuarialrs`, every `R/actuarialrs/tests/*.R` with `Rscript`
- Bindings, their doc comments or docs config touched: `cargo xtask docs --check` (needs Quarto, Python 3.11+ for great-docs, and the R packages `roxygen2`, `pkgload`, `pkgdown`), and commit the regenerated stub, `man/` and `NAMESPACE`

If a check cannot run in your environment (for example, R is not installed), rely on CI for it and say so in the PR.
