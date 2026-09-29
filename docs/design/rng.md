# Design note: RNG streams

Status: **Draft for review; core implemented** · Phase 0 · Depends on: `distributions.md` (sampling by inverse transform)

## Goal

Every result is bit-identical regardless of thread count, and any single
simulation can be replayed from `(seed, stream id)`.

## What exists (Phase 0)

`act_core::StreamRng::new(seed, stream)`:

- **Generator:** ChaCha20 (original 64-bit block counter, 64-bit nonce
  layout) from `rand_chacha`, pinned to an exact version in the workspace.
- **Key:** the 64-bit `seed` expanded to 256 bits with SplitMix64. We own
  this step, so a `rand_core` update cannot change it.
- **Stream:** the stream id is the ChaCha nonce. Each stream holds 2^64
  blocks (2^70 bytes).
- **Uniforms:** `next_open01()` returns `(k + 0.5) / 2^53` for the top 53
  bits `k`, so values are strictly inside `(0, 1)` and safe for any quantile
  function.
- **Tests:**
  - golden values, reproduced independently with Python's `cryptography`
    ChaCha20;
  - the same results from Rayon pools of 1, 4 and 16 threads;
  - the same lognormal draws pinned in Rust, Python and R tests.

Because the construction is standard ChaCha20, any language with a ChaCha20
implementation can regenerate a stream for audit.

## Stream allocation scheme (`chacha20/sim-index/v1`)

1. **One stream per outer simulation index.** Simulation `i` uses stream
   `i` and nothing else touches it. This is the unit the Rayon outer loop
   parallelizes over.
2. **Within a simulation, draws are consumed in a documented order**, e.g.
   for the ODP bootstrap: residual resampling for every cell in
   origin-major order, then process variance per future cell. The order is
   part of the model's spec and changes only with a scheme version bump.
3. **Sub-streams** for models that need independent pieces within one
   simulation (for example, per-event severities when the event count is
   random). Proposed: `StreamRng::substream(k)` sets the block counter to
   `k · 2^32`. That gives 2^32 sub-streams of 2^32 blocks (256 GiB) each,
   so adding draws to one piece never shifts another.
4. **Independent model components** (two lines simulated separately) use
   different seeds derived from the user seed with a documented label hash,
   never offsets of stream ids, so their stream ranges cannot collide.

## Distribution sampling

Inverse transform by default (see `distributions.md`): monotone in the
uniform, so common random numbers work across scenarios. Faster methods
(ziggurat for normals) may be added per family as an opt-in with its own
scheme version.

## Stability policy

- Output for a given `(seed, stream)` is a public contract. Golden tests in
  `act-core` and `act-prob` guard it.
- Changing it (generator, key expansion, uniform conversion, sampling
  method or draw order) requires bumping the scheme name, e.g.
  `…/v2`, recording it in `Provenance`, and a changelog entry.
- `rand_chacha` / `rand_core` are pinned with `=`; bumping them requires
  the golden tests to pass unchanged.

## Front ends

- Python passes `seed` and `stream` as Python ints, converted to `u64`.
- R has no 64-bit integers, so seeds arrive as doubles and must be whole
  numbers below 2^53 (enforced in `act-r`).
- WASM: ChaCha20 has no OS dependencies. Parallel paths fall back to a
  sequential loop over the same streams and give identical results.

## Open questions

1. **ChaCha20 vs ChaCha8/12.** ChaCha20 is the conservative choice and what
   exists today. ChaCha8 is about 2× faster and statistically sound for
   simulation, but a change later would be a scheme bump. Decide before
   v0.1 ships.
2. **Philox** (counter-based, as used by NumPy and JAX) would make streams
   reproducible from NumPy directly. That is worth it only if users
   regenerate our streams in NumPy; ChaCha20 is already reproducible via
   standard crypto libraries.
3. **Sub-stream width** (`2^32 × 2^32` blocks) is an assumption about
   model sizes; confirm against the claim-level model (v0.8), the most
   draw-hungry use.
