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
`publish = false`. CRAN is separate and not set up yet. The R package
installs from GitHub with `remotes::install_github("EKtheSage/prospicio",
subdir = "R/prospicio")`: `R/prospicio/bootstrap.R`, which pkgbuild runs
before `R CMD build`, copies `prospicio-r` and the workspace crates it
depends on into the tarball's `src/rust/`. CRAN will also need the
third-party crates vendored for an offline build.

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
2. **crates.io.** Sign in at crates.io with GitHub and verify an email
   address. On each of the 11 crates' *Settings → Trusted Publishing*, add
   GitHub with the same fields as PyPI's: owner `EKtheSage`, repository
   `prospicio` (no owner or slash), workflow `release.yml`, environment
   `crates-io`. The workflow then asks crates.io for a short-lived token
   itself (`rust-lang/crates-io-auth-action`); no token is stored.
   Trusted publishing works only for crates that already exist, so a new
   `prospicio-*` crate is published once by hand (see "Publishing from a
   laptop instead", with a token scoped to `publish-new` and a short
   expiry, revoked afterwards) and then added here. The first release,
   v0.0.1, used such a token for all 11 crates.
3. **GitHub environments** (optional but recommended). The publish jobs run
   in the environments `pypi` and `crates-io`; GitHub creates them on first
   use. Adding yourself as a required reviewer on each (*Settings →
   Environments*) makes every publish wait for your approval.

## Each release

1. On `main`, set `workspace.package.version` and the `prospicio-*`
   versions in `[workspace.dependencies]` to the new version, fold the
   pending change fragments (`python3 validation/scripts/fold_changes.py`,
   which empties `changes/` into `knowledge/log.md` and `CHANGELOG.md`),
   and move `CHANGELOG.md`'s "Unreleased" items into a section for the new
   version (breaking changes first, each item with its PR number), in a PR.
2. After it merges, run the **Release** workflow by hand (*Actions →
   Release → Run workflow* on `main`). Without a tag it builds and tests
   every wheel, the sdist and the crates, and publishes nothing.
3. Tag the merge commit and push the tag:

   ```sh
   git tag v0.0.2
   git push origin v0.0.2
   ```

   The workflow checks that the tag matches the workspace version, builds
   and tests again, then publishes to PyPI and crates.io.

Neither registry lets a version be replaced. A bad release is yanked
(`cargo yank`, or *Manage project* on PyPI) and fixed in the next version.
`cargo publish --workspace` uploads the crates one at a time. If it stops
partway, the crates already uploaded stay published, and rerunning it fails
on them; publish the remaining crates with `cargo publish -p <crate>`, in
dependency order.

## Publishing from a laptop instead

For a new crate, or if the workflow cannot be used, the same result comes
from a clean checkout of the tagged commit:

```sh
cargo login                        # paste the crates.io token
cargo publish --workspace          # every publishable crate, in order
```

PyPI uploads from a laptop need wheels for every platform, so they are
best left to the workflow.
