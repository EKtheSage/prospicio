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
   part of the model's spec and changes only with a stability-log entry
   and a new id in the samplers record (see the stability policy), not
   with a scheme bump.
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
(ziggurat for normals) may be added per family, each with its own sampler
id (stability policy). The Gamma's Marsaglia–Tsang sampler replaced its
inverse transform outright (2026-10-08, see the stability log): the
quantile was a bisection costing up to hundreds of microseconds a draw at
large shapes, which made the reserving bootstraps' Gamma process slow.

## Stability policy

- Output for a given `(seed, stream)` is a public contract. Golden tests in
  `prospicio-core` and `prospicio-prob` guard it.
- Two records in `Provenance` version it, and each change bumps exactly
  one of them (decided 2026-10-08, open question 4):
  - **The stream scheme** (`stream_scheme`, now `chacha20/sim-index/v1`)
    names how simulations map to streams and which uniforms a stream
    yields: generator, key expansion, uniform conversion, and the
    allocation of streams and sub-streams. A change to any of these bumps
    it, e.g. `…/v2`. `PredictiveDistribution::join` compares seed and
    scheme, and nothing else, to refuse two independent parts that share
    random numbers (`Provenance::shares_streams`).
  - **The samplers** (`samplers`) name how those uniforms become draws:
    `(family, sampler id)` pairs sorted by family, now
    `gamma = marsaglia-tsang/2026-10`, from the build's table
    `prospicio_prob::provenance::SAMPLERS`, recorded by every seeded
    result (`Provenance::seed`). A family missing from the table uses its
    first sampler: inverse transform for every distribution and counting
    family, the documented method for a copula's frailty. A change to a
    family's sampling method or to the uniforms it consumes gives the
    family a new id (`<method>/<year>-<month>`, adding the day for a
    second change in a month; ids are never reused), never a scheme bump.
    A change to a model's documented draw order does the same under the
    model's name (no model has changed its order yet).
  - Replaying a result (`Provenance::replays_same_draws`) needs the same
    seed, scheme and samplers. A sampler change therefore stops replay
    matching without weakening `join`'s seed check.
- Every change gets a stability-log entry below and a changelog entry.
- Results saved before 2026-10-08 carry no samplers (`None`, "not
  recorded"): they all say `chacha20/sim-index/v1`, whether their Gamma
  draws came by inverse transform (before the Gamma change) or by
  Marsaglia–Tsang (on 2026-10-08, before the split), so
  `replays_same_draws` never matches them, while `join` still refuses
  them next to a part on the same seed.
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
  `chacha20/sim-index/v1`: the mapping of simulations to streams is
  unchanged, and `join` reads the scheme to detect two parts sharing
  random numbers, which a bump would hide between a part from before and
  one from after. The sampler is versioned instead as
  `gamma = marsaglia-tsang/2026-10` (next entry).
- **2026-10-08, `Provenance::samplers`**: the sampler versions split from
  the stream scheme (open question 4). No draw changes. Every seeded
  result records the build's sampler table, today
  `[("gamma", "marsaglia-tsang/2026-10")]`; `Provenance::shares_streams`
  (used by `join`) compares seed and scheme, and
  `Provenance::replays_same_draws` also the samplers. Arrow IPC files
  carry them as an optional `samplers` key in the provenance JSON; the
  format version stays `1`, because readers ignore keys they do not know
  and read a missing key as not recorded, so files written before the
  split (the golden `validation/reference/predictive_distribution_v1.arrow`
  among them) still read, with `samplers` `None`. Python and R show the
  field in `provenance()`. Tests: `provenance::tests`,
  `portfolio::tests::join_refuses_a_shared_seed_across_a_sampler_change`,
  `ipc::tests::provenance_without_samplers_reads_as_not_recorded` and
  `validation/tests/predictive.rs`.

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
4. ~~**A sampler change and the scheme name**~~ (raised 2026-10-08 by the
   Gamma sampler): decided 2026-10-08, split the label in two.
   `stream_scheme` keeps meaning only how simulations map to streams,
   which `join` compares to refuse two independent parts on the same
   seed; a second record, `Provenance::samplers`, names the sampler
   versions (`gamma = marsaglia-tsang/2026-10`), which replay compares.
   A sampler change bumps only its sampler id, so the seed-reuse check
   never weakens. The alternatives were to keep `v1` and version samplers
   only in this note's log (no record in a result, so a saved result
   could not tell which sampler drew it), or to bump to `v2` and teach
   `join` which schemes share streams (a table that every sampler change
   would grow).
