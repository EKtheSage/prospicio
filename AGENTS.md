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

## CI status

**GitHub Actions is out of monthly credits until 2026-10-01.** Until then, `.github/workflows/ci.yml` will not run on PRs.

- Do not wait on CI checks, poll for them, or treat a missing CI result as a failure.
- Run the `ci.yml` checks locally before you open or update a PR, and list what you ran, with results, in the PR description:
  - Always: `cargo fmt --all --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`
  - Python bindings touched: `cargo clippy -p act-python -- -D warnings`, then `cd python && maturin develop && pytest tests`
  - R bindings touched: `cargo clippy -p act-r -- -D warnings`, `install.packages("S7")` if missing, `R CMD INSTALL R/actuarialrs`, `Rscript R/actuarialrs/tests/test-distributions.R`
- If a check cannot run in your environment (for example, R is not installed), say that in the PR. Do not claim it passed.
- On or after 2026-10-01, remove this section once CI is running again.
