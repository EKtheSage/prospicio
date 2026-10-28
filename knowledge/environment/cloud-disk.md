---
type: Environment Fact
title: Cloud session disk space
description: The cloud container's writable disk is a fixed allowance; a session that builds and tests the whole workspace fills it (target/debug reached 25 GB, 8.7 GB of it incremental), and deleting target/debug/incremental and stale scratch targets frees enough to go on; building without debug info keeps a full check near 3 GB.
tags: [environment, cloud, disk, cargo]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-16T12:00:00Z }
stale_after: 2027-01-16T00:00:00Z
sources:
  - id: session
    resource: process:cloud session claude/stoic-hawking-qc3qby, 2026-10-16
    title: A uv install failing with "No space left on device" (os error 28)
  - id: lean
    resource: process:cloud session claude/stoic-hawking-qc3qby, 2026-10-27
    title: A full workspace check under the lean cargo environment (#594)
---

# Facts (2026-10-16)

* `df` reported 100% used with 39 GB on the volume: the session's
  allowance, not the disk. After a full `cargo test`, the Bayesian
  parity tests in release and the Python and R builds, `target/debug`
  held 25 GB (16 GB `deps`, 8.7 GB `incremental`) and `target/release`
  1 GB.[^session]
* `rm -rf target/debug/incremental` freed about 9 GB at no cost beyond a
  slower next incremental build; `uv cache prune` and removing scratch
  crates' `target/` directories freed the rest needed.[^session]
* Deletes succeed while writes fail, so cleaning up works from inside the
  full container; the next step that writes (a `uv pip install`, a link
  step) is the one that reports the error.[^session]

# Facts (2026-10-27)

* Debug info is most of a debug target: without it, a full workspace
  `cargo clippy --all-targets`, `cargo test`, a WASM check and a release
  build for the R package left `target/` at 3.3 GB, where a target built
  with the default debug info had reached 21 GB in the same session and
  filled the allowance (truncating a file being written). The lean
  environment is `CARGO_INCREMENTAL=0` and `CARGO_PROFILE_DEV_DEBUG=0`,
  `CARGO_PROFILE_TEST_DEBUG=0`, `CARGO_PROFILE_RELEASE_DEBUG=0`, exported
  before every cargo command; one `CARGO_TARGET_DIR` shared by the
  session's worktrees keeps the dependencies built once.[^lean]

[^session]: A uv install failing with "No space left on device" (os error 28)
[^lean]: A full workspace check under the lean cargo environment (#594)
