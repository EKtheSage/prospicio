---
type: Environment Fact
title: Publishing to PyPI and crates.io
description: How the crates and the Python package package for the registries, what each registry's trusted publishing needs, the name check, and that crates.io now publishes by trusted publishing (the token secret is gone).
tags: [environment, release, pypi, crates-io, maturin]
status: stable
stale_after: 2027-04-07T00:00:00Z
generated: { by: claude-code/cloud-session, at: 2026-10-07T12:00:00Z }
sources:
  - id: crates-rl
    resource: https://crates.io/docs/rate-limits
    title: crates.io, rate limits
  - id: v001
    resource: https://github.com/EKtheSage/prospicio/actions/runs/37720822127
    title: Release workflow run for v0.0.1
  - id: release
    resource: ../docs/release.md
    title: docs/release.md, the release checklist
  - id: workflow
    resource: ../.github/workflows/release.yml
    title: Release workflow
  - id: crates-tp
    resource: https://crates.io/docs/trusted-publishing
    title: crates.io, Trusted Publishing
  - id: pypi-tp
    resource: https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/
    title: PyPI, creating a project through OIDC
---

# Facts

* On 2026-10-07 `prospicio` and `prospicio-core` returned 404 from the
  PyPI JSON API and the crates.io API, so the names were free. Check
  again before the first release: `curl -s -o /dev/null -w '%{http_code}'
  https://pypi.org/pypi/prospicio/json`, and
  `https://crates.io/api/v1/crates/prospicio` (crates.io wants a
  `User-Agent`).
* `cargo publish --workspace --dry-run` (cargo 1.90+) packages every
  publishable crate in dependency order and builds each from its package
  against a temporary local registry, so it checks the workspace before
  any crate is on crates.io. All 11 crates passed on 2026-10-07; the
  packages are 11 kB to 117 kB, far under the 10 MB limit.[^release]
* A path dependency must also carry a `version` for `cargo publish`; the
  `prospicio-*` entries in `[workspace.dependencies]` repeat the workspace
  version for that reason.
* crates.io trusted publishing is set per crate and only for a crate that
  already exists, so the first upload needs an API token with the
  `publish-new` scope.[^crates-tp] PyPI instead takes a *pending publisher* for a
  project that does not exist yet, but that does not reserve the name
  until the first upload.[^pypi-tp]
* `maturin sdist` with `include = [{ path = "prospicio/**/*", format =
  "sdist" }]` copies whatever is in `python/prospicio`, including a local
  `prospicio_native.abi3.so` and `__pycache__`. Build the sdist from a
  clean checkout, as the release workflow does.[^workflow]
* The sdist holds the workspace `Cargo.toml`, `Cargo.lock` and the crates
  the Python binding depends on (not `prospicio-nn`, `prospicio-r`,
  `validation` or `xtask`), and builds a wheel on its own.
* The Python tests pass against an installed wheel when run as `pytest
  tests` from `python/`: pytest puts `python/tests` on `sys.path`, not
  `python/`, so the installed package is the one imported. The docstring
  test of `prospicio.boosting` needs LightGBM, so the wheel test installs
  the `dev` dependency group (`uv pip install --group dev`).
* The first release, v0.0.1 (2026-10-08), published to PyPI from the
  workflow at the first try. crates.io refused it twice. First with `400
  Bad Request: A verified email address is required to publish crates`:
  the account needs a verified email before any upload. Then, after five
  new crates, with `429 Too Many Requests: You have published too many new
  crates in a short period of time`, with a retry time about 10 minutes
  out: crates.io limits how fast *new* crate names are created (a burst,
  then about one every 10 minutes; versions of existing crates are not
  limited this way).[^crates-rl] An 11-crate first release therefore takes about an
  hour.[^v001]
* Rerunning a failed `cargo publish --workspace` job does not resume: it
  starts again from `prospicio-core`, which now exists, and fails on it.
  The remaining crates were published one at a time with `cargo publish
  -p <crate>` from a checkout of the tag, in dependency order
  (`-glm`, `-pricing`, `-reserving`, `-bayes`, `-nn`, then `prospicio`).[^v001]
* On 2026-10-08 the user set up crates.io trusted publishing for all 11
  crates, as `docs/release.md` ("After the first release") lists, and
  deleted the `CARGO_REGISTRY_TOKEN` Actions secret.[^release] Without the
  secret the release workflow's `crates-io` job asks crates.io for a
  short-lived token (`rust-lang/crates-io-auth-action`, which runs only
  when the secret is empty) and publishes with it.[^workflow] The API token
  the secret held still exists until the user revokes it on crates.io
  (*Account Settings → API Tokens*); only the user can, so it is theirs to
  do, not a session's.

[^crates-rl]: crates.io, rate limits
[^v001]: Release workflow run for v0.0.1
