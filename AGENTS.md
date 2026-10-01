# Agent instructions

Shared guidance for every coding agent working in this repo: local Claude Code, Claude Code cloud sessions, and Codex. `CLAUDE.md` imports this file.

## Several agent sessions at once

A local session and one or more cloud sessions may be working on this repo at the same time. They do not share a working tree. The GitHub remote is the only thing they have in common.

- **One branch per session.** Never push to a branch that another session owns. Cloud sessions use `claude/<name>` branches.
- **Nothing is visible until it is pushed.** Commit and push often enough that other sessions can fetch your work. Before you start work that depends on another session, run `git fetch` and look at its branch.
- **Stay inside your assigned scope.** The user gives each session its own work (crates, files, or a phase). Do not edit files outside that scope without asking. If you must, say so clearly in your PR.
- **Merge through PRs into `main`.** Only the user merges. After a merge, other sessions rebase onto or merge `main` before they continue.
- **Record shared decisions in the repo.** Put them in this file or in `docs/`, not only in chat, so every session picks them up on its next pull.
- **Messages go one way.** A local session can send a cloud session a message, but the cloud session cannot reply. Leave status and questions in your PR description or in commit messages, where the user and other sessions can read them.

## Work lanes

Work is split by crate so that sessions rarely touch the same file. A lane owns the files listed for it; edit another lane's files only with the user's go-ahead.

| Lane | Session | Owns | Current work (v0.1) |
|---|---|---|---|
| Reserving | Local | `crates/act-reserving/`, `validation/tests/reserving.rs`, root `src/` | `Triangle` per `docs/design/triangle.md`, Chain Ladder, Mack, parity on RAA / GenIns / ABC; delete the `src/` sandbox once `act-reserving` covers it |
| Probability | Cloud | `crates/act-prob/`, `crates/act-math/`, `validation/tests/distributions.rs` | Distributions per `docs/design/distributions.md`, `PredictiveDistribution` per `docs/design/predictive-distribution.md` |
| Aggregate | Unassigned | `crates/act-aggregate/` | Not started (v0.3) |

The ODP bootstrap needs both a `Triangle` and a `PredictiveDistribution`. It starts in the Reserving lane after both lanes have merged.

### Shared files

- **`crates/act-core/`** is used by every lane. Change it only in a small PR of its own that holds nothing else, so it can merge first. Other lanes merge `main` before they build on the change.
- **Bindings**: each lane puts its wrappers in its own module file (for example `crates/act-python/src/reserving.rs`, `crates/act-r/src/distributions.rs`, `R/actuarialrs/R/reserving.R`, `python/actuarialrs/reserving.py`). The crate's `lib.rs` only registers modules, one line per lane.
- **Root `Cargo.toml` and `Cargo.lock`**: add dependencies only when a lane needs them. When `Cargo.lock` conflicts, take either side and let `cargo` regenerate it. Do not merge it by hand.
- **`AGENTS.md` and `docs/architecture.md`**: change them in small PRs. Each lane edits only its own design notes under `docs/design/`.
- **Keep PRs small** (one type or method each) and merge `main` often. The longer a branch lives, the harder its conflicts get.

## Checks

`.github/workflows/ci.yml` runs on every PR and on pushes to `main`. A PR merges only when CI is green on its latest commit.

Run the matching `ci.yml` checks locally before you push, so CI does not go red on something a local run would have caught:

- Always: `cargo fmt --all --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`
- Python bindings touched: `cargo clippy -p act-python -- -D warnings`, then `cd python && maturin develop && pytest tests`
- R bindings touched: `cargo clippy -p act-r -- -D warnings`, `install.packages("S7")` if missing, `R CMD INSTALL R/actuarialrs`, `Rscript R/actuarialrs/tests/test-distributions.R`
- Bindings, their doc comments or docs config touched: `cargo xtask docs --check` (needs Quarto, Python 3.11+ for great-docs, and the R packages `roxygen2`, `pkgload`, `pkgdown`), and commit the regenerated stub, `man/` and `NAMESPACE`

If a check cannot run in your environment (for example, R is not installed), rely on CI for it and say so in the PR.
