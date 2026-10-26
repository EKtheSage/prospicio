---
type: Environment Fact
title: The dependency tree under cargo deny
description: What cargo deny 0.20.2 found on main on 2026-10-23 (two unmaintained crates, two crates under MPL-2.0 and CC0-1.0, no vulnerabilities, every source crates.io), why each is an exception in deny.toml, and the two breaks cargo semver-checks reports against 0.0.5.
tags: [environment, dependencies, licences, advisories, release]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-23T20:00:00Z }
stale_after: 2027-04-23T00:00:00Z
sources:
  - id: deny
    resource: process:cargo deny check (cargo-deny 0.20.2) on origin/main, 2026-10-23
    title: cargo deny check on main
  - id: issue
    resource: https://github.com/EKtheSage/prospicio/issues/573
    title: Add cargo-semver-checks to release preparation and cargo-deny to CI
---

# The dependency tree under cargo deny

`cargo deny check` (`deny.toml`, all features) on `main` on 2026-10-23,
before any exception, found:

- **Advisories:** no vulnerabilities. Two crates are flagged as
  unmaintained: `bincode` 2.0.1 (RUSTSEC-2025-0141), pulled in only by
  `burn-core` under `prospicio-nn` (prospicio's network artifacts are JSON
  and do not use it), and `paste` 1.0.15 (RUSTSEC-2024-0436), a
  compile-time proc macro of `burn-ndarray` and of `gemm` (under `faer`
  and `nuts-rs`). Neither has a safe upgrade; both are ignored with the
  reason, and go when Burn and faer move off them.
- **Licences:** everything is MIT, Apache-2.0 (one with the LLVM
  exception), BSD-2-Clause, ISC, Zlib or Unicode-3.0, except `colored`
  3.1.1 (MPL-2.0, via `burn-tensor` under `prospicio-nn`) and
  `tiny-keccak` 2.0.2 (CC0-1.0, via `const-random` and `ahash` under
  `arrow`, which `prospicio-prob` uses). Each is allowed for that crate
  alone, so that any new crate under another licence is reviewed. Both
  licences are on CRAN's list of free licences, which matters once the
  third-party crates are vendored into the R tarball (`docs/release.md`).
- **Bans:** 17 crates appear in two or three versions (`syn`,
  `getrandom`, `windows-sys`, `thiserror`, `hashbrown`, ...); they are
  warnings, not failures.
- **Sources:** every crate comes from crates.io; no git dependencies.

Installing the tool takes about four minutes in a cloud session
(`cargo install cargo-deny --locked`); CI uses
`EmbarkStudios/cargo-deny-action@v2` instead.

## cargo semver-checks against 0.0.5

`cargo semver-checks check-release --workspace --release-type minor`
(0.51.0, about 3 minutes in a cloud session after a 6-minute install) on
`main` the same day reported two breaks against the published 0.0.5,
both already in change fragments marked breaking: new public fields on
`TowerGrids` and `AnnualGrids` (`expected_deal_ceding_commission`,
`expected_deal_brokerage`; `constructible_struct_adds_field`) and the new
variant `prospicio_reserving::Error::LatestDiagonalOnly`
(`enum_variant_added`). Removing a public method (`Month::months_since`)
on a scratch copy fails with `inherent_method_missing` and exit code 100.
`--release-type minor` is needed before 1.0: under Cargo's rules every
0.0.x bump may break anything, so the default reports nothing. Both kinds
of break are avoidable for later additions with `#[non_exhaustive]` on
public structs and enums that will grow.
