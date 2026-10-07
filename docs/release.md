# Releasing

A release publishes the Python package `prospicio` to PyPI and the Rust
crates to crates.io from one version tag. `.github/workflows/release.yml`
does the work; this note is the checklist around it.

## What is published

| Registry | Name | Built from |
|---|---|---|
| PyPI | `prospicio` | `python/` and `crates/prospicio-python`: an sdist, and one abi3 wheel (Python 3.9+) each for Linux x86_64 and aarch64 (manylinux), macOS (universal2) and Windows x64 |
| crates.io | `prospicio` | `crates/prospicio`: the umbrella crate. It re-exports the others as modules (`prospicio::reserving`, `prospicio::prob`, ...); `bayes` and `nn` are features |
| crates.io | `prospicio-core`, `-math`, `-prob`, `-aggregate`, `-pricing`, `-reserving`, `-models`, `-glm`, `-bayes`, `-nn` | `crates/prospicio-*` |

`prospicio-python`, `prospicio-r`, `validation` and `xtask` set
`publish = false`. CRAN is separate and not set up yet.

Every crate and the Python package take their version from
`workspace.package.version` in the root `Cargo.toml`. The `prospicio-*`
entries in `[workspace.dependencies]` repeat it (`cargo publish` puts that
version in place of the path), so a version bump changes both.

## One-time setup (the repository owner)

Do these once, before the first release.

1. **PyPI.** Sign in at pypi.org (create the account and turn on 2FA if
   needed). Under *Your account → Publishing*, add a **pending publisher**
   on the *GitHub* tab, one value per field:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `prospicio` |
   | Owner | `EKtheSage` |
   | Repository name | `prospicio` (the name only: no owner, no slash) |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   No token is stored anywhere. A pending publisher does not reserve the
   name; the first upload does.
2. **crates.io, first release.** Sign in at crates.io with GitHub and
   verify an email address. Trusted publishing works only for crates that
   already exist, so the first release uses a token: under *Account
   Settings → API Tokens*, create one with the scopes `publish-new` and
   `publish-update`, and an expiry a few days out. Add it to this
   repository as the Actions secret `CARGO_REGISTRY_TOKEN`
   (*Settings → Secrets and variables → Actions*).
3. **GitHub environments** (optional but recommended). The publish jobs run
   in the environments `pypi` and `crates-io`; GitHub creates them on first
   use. Adding yourself as a required reviewer on each (*Settings →
   Environments*) makes every publish wait for your approval.

## Each release

1. On `main`, set `workspace.package.version` and the `prospicio-*`
   versions in `[workspace.dependencies]` to the new version, in a PR.
2. After it merges, run the **Release** workflow by hand (*Actions →
   Release → Run workflow* on `main`). Without a tag it builds and tests
   every wheel, the sdist and the crates, and publishes nothing.
3. Tag the merge commit and push the tag:

   ```sh
   git tag v0.0.1
   git push origin v0.0.1
   ```

   The workflow checks that the tag matches the workspace version, builds
   and tests again, then publishes to PyPI and crates.io.

Neither registry lets a version be replaced. A bad release is yanked
(`cargo yank`, or *Manage project* on PyPI) and fixed in the next version.
`cargo publish --workspace` uploads the crates one at a time. If it stops
partway, the crates already uploaded stay published, and rerunning it fails
on them; publish the remaining crates with `cargo publish -p <crate>`, in
dependency order.

## After the first release

1. On crates.io, open each of the 11 crates' *Settings → Trusted
   Publishing* and add GitHub with the same fields as PyPI's: owner
   `EKtheSage`, repository `prospicio` (no owner or slash), workflow
   `release.yml`, environment `crates-io`.
2. Delete the `CARGO_REGISTRY_TOKEN` secret and revoke the token. The
   workflow then asks crates.io for a short-lived token itself
   (`rust-lang/crates-io-auth-action`).

## Publishing from a laptop instead

If the workflow is not wanted for the first release, the same result comes
from a clean checkout of the tagged commit:

```sh
cargo login                        # paste the crates.io token
cargo publish --workspace          # every publishable crate, in order
```

PyPI uploads from a laptop need wheels for every platform, so they are
best left to the workflow.
