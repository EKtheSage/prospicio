# Design note: RNG streams

Status: **Draft for review; core implemented** · Phase 0 · Depends on: `distributions.md` (sampling by inverse transform)

## Goal

Every result is bit-identical regardless of thread count, and any single
simulation can be replayed from `(seed, stream id)`.

## What exists (Phase 0)

`prospicio_core::StreamRng::new(seed, stream)`:

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
scheme version. The Gamma's Marsaglia–Tsang sampler replaced its inverse
transform outright (2026-10-08, see the stability log): the quantile was
a bisection costing up to hundreds of microseconds a draw at large
shapes, which made the reserving bootstraps' Gamma process slow.

## Stability policy

- Output for a given `(seed, stream)` is a public contract. Golden tests in
  `prospicio-core` and `prospicio-prob` guard it.
- Changing it (generator, key expansion, uniform conversion, sampling
  method or draw order) requires bumping the scheme name, e.g.
  `…/v2`, recording it in `Provenance`, and a changelog entry.
  **Exception pending a decision (open question 4):** the Gamma sampler
  changed on 2026-10-08 under `v1` (stability log). Until that is
  decided, a result with Gamma draws saved before that date and one saved
  after carry the same provenance, `chacha20/sim-index/v1`, but do not
  replay the same; only the date tells them apart.
- `rand_chacha` / `rand_core` are pinned with `=`; bumping them requires
  the golden tests to pass unchanged.

## Stability log

- **2026-10-08, `Gamma::sample`**: inverse transform replaced by
  Marsaglia and Tsang (2000), with the `U^(1/shape)` boost below shape 1,
  on the same streams (`prospicio_prob::gamma::standard_gamma`, the
  sampler the Student t and Clayton copulas already used). Every Gamma
  draw changes: `Gamma::sample`, `Dist::Gamma`, and the Gamma process of
  `OdpBootstrap` and `MackBootstrap`. A draw now takes a variable number
  of uniforms (at least two; one more below shape 1), so later draws on
  the same stream move too, and draws are no longer monotone in one
  uniform. Nothing else changes: streams, uniforms, every other family,
  and code that calls `Gamma::quantile` itself. Golden test:
  `gamma::tests::sample_is_pinned`, reproduced independently by
  `validation/scripts/gamma_sampler.py`. The scheme name stays
  `chacha20/sim-index/v1`, against the policy above: the mapping of
  simulations to streams is unchanged, and `Portfolio` reads the scheme
  to detect two parts sharing random numbers, which a bump would hide
  between a part from before and one from after. Whether to bump it
  anyway is open (question 4).

## Front ends

- Python passes `seed` and `stream` as Python ints, converted to `u64`.
- R has no 64-bit integers, so seeds arrive as doubles and must be whole
  numbers below 2^53 (enforced in `prospicio-r`).
- WASM: ChaCha20 has no OS dependencies. Parallel paths fall back to a
  sequential loop over the same streams and give identical results.

## Open questions

1. ~~**ChaCha20 vs ChaCha8/12**~~: decided 2026-10-04, ChaCha20, unless an
   alternative gives a substantial improvement. Measured on 2026-10-04
   (release build, one core): raw `u64` output is 2.1× faster with ChaCha8
   (50M draws: 0.30 s ChaCha20, 0.19 s ChaCha12, 0.14 s ChaCha8), but a
   lognormal draw by inverse transform, the typical use, is only 2.7%
   faster (10M draws: 2.16 s vs 2.10 s), because the quantile function
   costs about 30 times the generator. Not substantial; a scheme bump is
   not worth it.
2. **Philox** (counter-based, as used by NumPy and JAX) would make streams
   reproducible from NumPy directly. That is worth it only if users
   regenerate our streams in NumPy; ChaCha20 is already reproducible via
   standard crypto libraries.
3. **Sub-stream width** (`2^32 × 2^32` blocks) is an assumption about
   model sizes; confirm against the claim-level model (v0.8), the most
   draw-hungry use.
4. **A sampler change and the scheme name** (raised 2026-10-08 by the
   Gamma sampler): the policy bumps the scheme for a new sampling method,
   but `stream_scheme` names the mapping of simulations to streams, which
   `Portfolio` compares to refuse two independent parts on the same
   seed. Options: keep `v1` and version samplers in the stability log
   (done for the Gamma), or bump to `v2` and teach `Portfolio` that `v1`
   and `v2` share streams.
