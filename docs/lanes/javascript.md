# JavaScript lane

**Session:** Cloud (the JavaScript session). **Owns:** `crates/prospicio-wasm/`, `docs/design/wasm.md`, the `wasm` task in `xtask/src/main.rs` and its filter in `xtask/src/check.rs`, the `wasm` job and the `changes` job's `wasm` output in `.github/workflows/ci.yml`

This lane wraps the crates in the WASM scope and changes none of them. JavaScript is only a front end to Rust: a calculation the crates lack goes to the owning lane first

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/javascript.md](done/javascript.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next

- [wasm-npm-setup] the one-time npm setup is the user's: the first version published by hand, the trusted publisher added before the next release, then the repository variable `NPM_PUBLISH` set to `true` (`docs/release.md`, one-time setup, step 4)
- [wasm-more-functions] further functions as the user asks, each in its lane's module file (`wasm.md` O5)
