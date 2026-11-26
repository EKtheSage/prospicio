# Releasing

A release publishes the Python package `prospicio` to PyPI, the Rust
crates to crates.io and the JavaScript package `prospicio` to npm from one
version tag. `.github/workflows/release.yml`
does the work; this note is the checklist around it.

## What is published

| Registry | Name | Built from |
|---|---|---|
| PyPI | `prospicio` | `python/` and `crates/prospicio-python`: an sdist, and one abi3 wheel (Python 3.9+) each for Linux x86_64 and aarch64 (manylinux), macOS (universal2) and Windows x64; each wheel is tested on Python 3.12, and the Linux x86_64 wheel on 3.9 as well (#962) |
| crates.io | `prospicio` | `crates/prospicio`: the umbrella crate. It re-exports the others as modules (`prospicio::reserving`, `prospicio::prob`, ...); `bayes` and `nn` are features |
| crates.io | `prospicio-core`, `-math`, `-prob`, `-aggregate`, `-pricing`, `-reserving`, `-models`, `-glm`, `-bayes`, `-nn` | `crates/prospicio-*` |
| npm | `prospicio` | `crates/prospicio-wasm`, built by `cargo xtask wasm`: one WebAssembly module with glue for bundlers, Node and a plain browser page (`docs/design/wasm.md`). Published once the owner has done step 4 of the one-time setup |

The Excel add-in is not on a registry: `.github/workflows/xll.yml` attaches
`prospicio-xll-<tag>.zip` (`prospicio.xll`, the test workbook, its README
and a SHA-256) to a **draft** GitHub release for the tag, which the owner
reviews and publishes (`docs/design/xll.md`, decision 15).

`prospicio-python`, `prospicio-r`, `prospicio-xll`, `prospicio-wasm`,
`validation` and `xtask` set `publish = false` (the JavaScript package
goes to npm, not crates.io). CRAN is separate and not set up yet. The R package
installs from GitHub with `remotes::install_github("EKtheSage/prospicio",
subdir = "R/prospicio")`: `R/prospicio/bootstrap.R`, which pkgbuild runs
before `R CMD build`, copies `prospicio-r` and the workspace crates it
depends on into the tarball's `src/rust/`. CRAN will also need the
third-party crates vendored for an offline build.

Every crate, the Python package and the npm package take their version
from `workspace.package.version` in the root `Cargo.toml` (`cargo xtask
wasm` writes it into the package's `package.json`). The `prospicio-*`
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
   in the environments `pypi`, `crates-io` and `npm`; GitHub creates them on
   first use. Adding yourself as a required reviewer on each (*Settings →
   Environments*) makes every publish wait for your approval.
4. **npm.** Until this step is done the workflow's `npm` job is skipped and
   a tag publishes to PyPI and crates.io only. npm lets a trusted publisher
   be added only on a package that exists, so the first version goes up by
   hand:

   1. Create an account at npmjs.com with 2FA. The name `prospicio` was
      free on 2026-11-20 (`npm view prospicio` returned 404).
   2. After a release's tag run has finished, download its `npm-package`
      artifact (*Actions → Release → the run → Artifacts*), unpack it, and
      in that folder run `npm login` and `npm publish`. This is the same
      package the run built and tested; it reserves the name with that
      release's version.
   3. Before the **next** release, and no more than two days before its
      tag (a trusted publisher that has not published within two days of
      being created expires): on npmjs.com open the package's *Settings →
      Trusted Publisher*, choose GitHub Actions and give

      | Field | Value |
      |---|---|
      | Organization or user | `EKtheSage` |
      | Repository | `prospicio` |
      | Workflow filename | `release.yml` |
      | Environment name | `npm` |
      | Allowed actions | `npm publish` |

      npm does not check these when they are saved; a wrong value shows
      only as an authentication error at the publish.
   4. In the repository's *Settings → Secrets and variables → Actions →
      Variables*, add `NPM_PUBLISH` with the value `true`. From then on a
      tag publishes the package from the `npm` job, with no stored token
      and with a provenance attestation.

## Version plan

After 0.0.9 comes 0.1.0, then 0.2.0, and so on up to 0.9.0. The 0.9 line is
the release candidates: 0.9.0, then 0.9.1 and later patches as needed. Then
1.0. Before 1.0 a minor bump may break the API; a patch bump stays
compatible. A patch may add to the API (a new function, an option alias, a
new package) as well as fix bugs (the user's decision of 2026-11-20), but
it does not change the numbers a seeded run gives, since results must be
reproducible from a seed (`docs/architecture.md`). The scope of each
version is the user's decision (see below).

## Scope of the next release

What goes into a release is the user's decision. The chosen issues and PRs
carry the GitHub milestone of that version; an issue without one waits for a
later release. The label `P4` marks an issue the user does not plan for any
upcoming release.

**0.0.6** was released on 2026-10-28 with the scope the user set on
2026-10-23 (the work in flight and small fixes).

**0.0.7** (the user's decision of 2026-10-28): the four P1s 0.0.6 deferred,
the rest of #600's breaking change, and small fixes. Already on `main` since
v0.0.6: #673 (#600, breaking), #675 (#670, breaking), #583.

| Lane | New work | Small items |
|---|---|---|
| Reserving | #552 (`read_csv` takes a reader or sits behind an optional feature, and `csv` and `calamine` are optional dependencies, so the WASM check still passes), #678 and #679 (pattern draws from Meyers and Compartmental, breaking as #673 was) | |
| Models | #545, then #546 | #549, #575 |
| Excel | #680 (pattern ages on the triangle functions) | #635 (both its parts; one argument order with #680) |
| Guides | `[guides-current]`: #673's pattern draws, #552's loaders, #675's `extend` in place of the notebook's `upper=5.0` | #642 (the Friedland guide and its R twin too) |
| Shared | #572, a PR of its own (the local session) | |

Probability and Aggregate have no 0.0.7 item: the cloud session that runs
them carries the Models items, and picks up at most one or two of the P3s
#593 and #596 after them. Everything else waits for 0.0.8 or later (see
below).

**0.0.8** (the user's decision of 2026-10-29): the gaps the ILS deal notebook
found, the breaking change #580 left from 0.0.7, two new Models fits, and
Excel parity with the Python and R bindings. On 2026-11-03 the user added
#547 (local explanations), built with the two fits in #727.

| Lane | New work | Small items |
|---|---|---|
| Reserving | Done: #598 (a new underwriting year's predictive loss ratio, #728), then #604 (an industry prior from Schedule P triangles, #732) | Done: #717 (faster monthly Weibull and loglogistic patterns, #726) |
| Reserving and bindings | Done in #725: #580 (one `NotSupported` error, breaking) | |
| Reserving | Done: #599's design note (#733); the model waits for the user's choice of pooling structure | |
| Models | Done in #727: #548 (zero-inflated and hurdle counts), #550 (quantile and expectile regression), #547 (ICE curves, accumulated local effects and SHAP values) | |
| Excel | Not started: parity with the Python and R bindings (the issue was not filed); moves to 0.0.9 | Done: the duplicates of Excel's own functions removed and the README rewritten (#723) |
| Guides | Moves to 0.0.9: `examples/ils_deal_analysis.ipynb` uses #598 and #604 in place of its hard-coded prior (#730). Done: the submission-schema guide pays each simulation on its own pattern (#736) | `[guides-current]` for 0.0.8's changes; the ILS notebook's workaround table (#724) |

Probability and Aggregate have no 0.0.8 item. Merged beyond the scope, ahead
of 0.0.9: #554 and #555 (#739), #556 (#729), #557 (#735), #558 (#740), #559
(#731) and #560 (#734).

Status on 2026-11-04: the release PR sets 0.0.8; the Excel parity batches and
#730 move to 0.0.9.

**0.0.9** (the user's decision of 2026-11-04): close as many of the open
issues as possible. Only IFRS 17 (#553, `P4`), claim-level reserving and the
survival models stay parked. #599's pooling structure is the recommended one
(`reserving-bayes.md` open decision 26, decided 2026-11-04).

| Lane | New work | Small items |
|---|---|---|
| Reserving | #599: `HierarchicalBf` with the recommended pooling (a non-centred lognormal program effect, one partially pooled pattern, half-normal spread priors, the ODP scale pooled per unit of premium), then its one-year view; Compartmental at a sub-annual grain (`[compartmental-subannual]`, built on 2026-11-06) | |
| Aggregate | | #722 (an accident-year payment pattern on the underwriting basis in `monthly_patterns`) |
| Models | #577 (tabular networks with categorical embeddings and multi-task heads) | |
| Excel | parity with the Python and R bindings in batches (#742) | |
| Guides | Done: #730 (an industry prior from a user-fetched Schedule P line, replacing the ILS notebook's hard-coded prior) | `[guides-current-next]` for the methods 0.0.8 added |
| Shared | #579 (one large file split per PR) | |

Status on 2026-11-08: the release PR sets 0.0.9, and every 0.0.9 item is
done. #599 (`HierarchicalBf`, #757, and its one-year view, #758), Compartmental
at a sub-annual grain (#759, #763), #722 (#745), #577 (#749), #742 (closed
with its PRs #784 to #796), #730 (#756) and #579 (closed with #750 to #755,
#765 to #767, #793 and #794). #576 is parked (#787).

Probability has no open item. Parked: #553 (IFRS 17, `P4`; the user took
it up on 2026-11-14, `reserving-ifrs17.md`), claim-level
reserving and the survival models. #576 (the WASM JavaScript package) was
parked by the user's decision of 2026-11-07 (the XLL already reaches Excel
users, and a browser package needs more enterprise work to deploy, such as
security review and user groups on Azure) and taken up again on 2026-11-20
for 0.2.1: the package is built and tested (`docs/design/wasm.md`), and
it is published to npm as `prospicio` (the user's decision of
2026-11-20) by the release workflow's `npm` job, once the owner has done
step 4 of the one-time setup above. The crates stay WASM-compatible: the
`wasm32-unknown-unknown` check in `AGENTS.md` is unchanged.

**0.1.0** (the user's decision of 2026-11-07, planned; the release after
0.0.9): parity with Python `aggregate` 2.1.0 (released 2026-11-07), except
the characteristic-function engine, and the 0.0.9 leftovers the user moves
here. Milestone 0.1.0. #768 goes first: the other items check their numbers
against its references.
On 2026-11-08 the user moved two 0.0.9 leftovers here: #782 (Models: a NaN
Pareto k from `gpd_fit` in the one-year view) and #762 (Reserving: the cost
and usability decisions on `HierarchicalBf`'s one-year view). Both now carry
milestone 0.1.0.

| Lane | New work | Small items |
|---|---|---|
| Aggregate (Probability for its files) | #768 (re-pin the parity references to aggregate 2.1.0, and settle three open checks) | |
| Reserving | #762 (the cost and usability decisions on `HierarchicalBf`'s one-year view; moved from 0.0.9 on 2026-11-08) | |
| Models | #782 (a NaN Pareto k from `gpd_fit` in the one-year view; moved from 0.0.9 on 2026-11-08) | |
| Probability | #769 (Tsallis distortion), #770 (Gini index of a distortion, `from_gini`) | #774 (refuse invalid discrete inputs, breaking), #775 (Tweedie parameter conversions) |
| Aggregate | #772 (disappearing deductible), #773 (OEP of a program, net or ceded), #777 (event loss tables across units), #778 (a validation verdict), #776 (lattice bucket rule and warning) | #771 (the extended distortion calibration set) |

Status on 2026-11-13: the release PR sets 0.1.0, and every 0.1.0 item is
done. The `aggregate` 2.1.0 parity issues closed with #811 to #820, #762
with #821 and #782 with #814. The codebase audit of 2026-11-09 (#822) was
added to the release and finished with it: its 30 sub-issues closed, the
last with #909 to #912 (#834 and #843). The severity fits (#800, #802, #804,
#805), the trend schedule (#801, #803, #806; the AP demo's cumulative loss
ratios wait for its inputs) and the Excel functions merged since 0.0.9 are
in it too.

**0.2.0** was released on 2026-11-20: the characteristic-function engine, #779, the laws that
need it, #780, and an event loss table as a compound severity, #781.
Milestone 0.2.0.

Status on 2026-11-20: the release PR sets 0.2.0, and every 0.2.0 item is
done. #779 closed with #915 and #916 (and the inversion's TVaR, #964), #780
with #917 to #920 and #781 with #921 (its Excel function, #922); the
engine's Excel functions came with #930. Also in
the release: discounted reserves and the IFRS 17 risk adjustment (#553,
#924 to #926), the AP pricing demo's cumulative loss ratios as a test
(#806, #923, #929) and the fixes of the nightly run (#927). The codebase
review of 2026-11-19 (#931, its sub-issues #932 to #983) is not part of
0.2.0, except three of its findings, which the user held the tag for on
2026-11-20: #937 (a two-sided stable law on a given step above the split's
limit, #988), #938 (the Tsallis complement at a large order, #987) and
#934 (a RustSec advisory that failed `cargo deny`, #986). The others wait
for the user's decision on their release.

**0.2.1** (the user's decision of 2026-11-20): the fixes the codebase review
of 2026-11-19 found, the additive items that are ready, and the CI work.
The milestone tracks all of it, including work that does not change a
package.

| Kind | Items |
|---|---|
| Bugs (P1) | #939, #940, #941, #942, #943, #975 (GIL releases and FFI panics in the Python bindings and the Excel add-in) |
| Bugs (P2) | #935, #944 to #954 |
| Bugs (P3) | #955, #958, #959, #960 |
| Additive | #956 (option names made the same in Python, R and Excel, with the old names kept as deprecated aliases), #576 (the WASM JavaScript package, #1039 and #1040) |
| Numerics with unchanged results | #971 (one LU factorization in `linalg::inverse`, the rearrangement algorithm's sweeps) |
| CI and tooling | #936, #962, #965, #966, #967, #968 |
| Docs | #961 |

Status on 2026-11-23: the release PR sets 0.2.1, with no breaking change.
The fixes of the review, the option-name aliases (#956), the JavaScript package
(#576) and the CI work are in the changelog section; #961 and the CI items
that change no package are tracked by the milestone only.

**0.3.0** (the user's decision of 2026-11-20, planned): the performance
work that changes how simulations draw or sum (so a seeded run's numbers
change, noted under "Breaking changes"), the refactors of the bindings, and
two actuarial tracks.

| Kind | Items |
|---|---|
| Performance (each starts with a benchmark) | #969, #970, #972 |
| Bindings and refactors | #957, #963 (with small `macro_rules!` helpers for the repeated glue), #976, #977, #978, #990, #991 |
| Actuarial | #979 (calendar-year and superimposed inflation), #980 (reinsurer default risk on towers) |
| Breaking, deferred from 0.2.1 | #1043 (`Trace::first_month` restored, `Trace` `#[non_exhaustive]`) |

#973, #974, #761 and #981 to #983 have no milestone yet.

Status on 2026-11-26: the release PR sets 0.3.0 (the user's decision of
2026-11-26 to cut it with what is done). In it: the performance items #969,
#970 and #972, the binding refactors #957, #963, #976 and #977, the
inflation track #979 and reinsurer default #980, #1043, and most of #978
(JSON artifacts and pickling for the fitted models and containers, `RMatrix`
returns in R, NumPy arrays in and out of `PredictiveDistribution`, `Design`
and `EventSet`, which makes NumPy a runtime dependency of the Python
package). Merged beyond the scope: the FFT consolidation (#973) and the
retrospective rating and earned exposure work (#981). Moved to the next
release: the rest of #978 (pickling the reserving fits), #990 and #991.

## Each release

1. On `main`, set `workspace.package.version`, the `prospicio-*`
   versions in `[workspace.dependencies]` and `Version:` in
   `R/prospicio/DESCRIPTION` to the new version (the release workflow
   fails when `DESCRIPTION` differs), fold the
   pending change fragments (`python3 validation/scripts/fold_changes.py`,
   which empties `changes/` into `knowledge/log.md` and `CHANGELOG.md`
   and deletes the lane files' finished Next bullets),
   and move `CHANGELOG.md`'s "Unreleased" items into a section for the new
   version (breaking changes first, each item with its PR number), in a PR.
2. After it merges, run the **Release** workflow by hand (*Actions →
   Release → Run workflow* on `main`). Without a tag it builds and tests
   every wheel, the sdist and the crates, and publishes nothing; it also
   runs so every week on its own (#828), so the last scheduled run shows
   whether the wheels still build on every platform. Its
   `semver` job runs `cargo semver-checks check-release --workspace
   --release-type minor` against each crate's last version on crates.io
   (locally: `cargo install cargo-semver-checks --locked`, then the same
   command). Until 1.0 it reports and does not block: every break it
   lists must match an item under "Breaking changes" in the new
   `CHANGELOG.md` section, from a change fragment marked breaking
   (`changes/README.md`); a break with no such item gets one before the
   tag. From 1.0 a break fails the release.

   Licences, RustSec advisories and crate sources are checked on every
   pull request that changes the dependencies and on the nightly run
   (`cargo deny check`, `deny.toml`). An advisory found by the nightly run
   is fixed, or ignored in `deny.toml` with its reason, before a
   release.
3. Tag the merge commit and push the tag:

   ```sh
   git tag v0.0.2
   git push origin v0.0.2
   ```

   The workflow checks that the tag and the R package's `DESCRIPTION`
   match the workspace version, builds
   and tests again, then publishes to PyPI and crates.io, and to npm once
   `NPM_PUBLISH` is set (one-time setup, step 4).

   The **XLL** workflow builds the Excel add-in for the same tag and
   attaches it to a draft GitHub release; publish the draft (*Releases*)
   when the add-in is to go out with the version.

   A cloud session can do steps 1 and 2 but cannot push the tag: its git
   access stops at its own branch, and the push comes back `403`. The
   owner pushes the tag.

No registry lets a version be replaced. A bad release is yanked
(`cargo yank`, *Manage project* on PyPI, `npm deprecate` on npm) and fixed
in the next version.
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

The npm package builds on any machine with the `wasm32-unknown-unknown`
target, Node and the pinned `wasm-bindgen` tool (`cargo xtask wasm` prints
the install command):

```sh
cargo xtask wasm                   # builds and tests target/wasm/pkg
cd target/wasm/pkg
npm login
npm publish
```
