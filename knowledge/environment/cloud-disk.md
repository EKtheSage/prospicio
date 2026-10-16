---
type: Environment Fact
title: Cloud session disk space
description: The cloud container's writable disk is a fixed allowance; a session that builds and tests the whole workspace fills it (target/debug reached 25 GB, 8.7 GB of it incremental), and deleting target/debug/incremental and stale scratch targets frees enough to go on.
tags: [environment, cloud, disk, cargo]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-16T12:00:00Z }
stale_after: 2027-01-16T00:00:00Z
sources:
  - id: session
    resource: process:cloud session claude/stoic-hawking-qc3qby, 2026-10-16
    title: A uv install failing with "No space left on device" (os error 28)
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

[^session]: A uv install failing with "No space left on device" (os error 28)
