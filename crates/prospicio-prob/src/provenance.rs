//! Where a result came from: model, parameters, seed, versions and a hash
//! of the input.
//!
//! Two fields say how a simulated result's random numbers were made, and
//! they answer different questions (`docs/design/rng.md`):
//!
//! - [`Provenance::stream_scheme`] names the rule mapping simulations to
//!   streams. Two results with the same seed and scheme draw the same
//!   uniforms in every simulation, so [`Provenance::shares_streams`], which
//!   [`crate::PredictiveDistribution::join`] uses to refuse "independent"
//!   parts that share random numbers, compares only these.
//! - [`Provenance::samplers`] names the samplers turning those uniforms
//!   into draws. [`Provenance::replays_same_draws`] compares it as well: a
//!   result replays only under the same samplers.
//!
//! A sampler change gives the sampler a new id in [`SAMPLERS`] and leaves
//! the stream scheme alone, so the seed-reuse check holds across it.

/// Name of the rule mapping simulations to RNG streams used by
/// [`crate::PredictiveDistribution::simulate`]: simulation `i` draws only
/// from `StreamRng::new(seed, i)`. See `docs/design/rng.md`. It covers the
/// mapping only (generator, key expansion, uniforms, stream allocation);
/// the samplers that turn uniforms into draws are versioned in
/// [`SAMPLERS`].
pub const SIM_INDEX_SCHEME: &str = "chacha20/sim-index/v1";

/// Id of the Gamma distribution's sampler ([`crate::Gamma`]'s `sample`,
/// and so `Dist::Gamma` and the reserving bootstraps' Gamma process):
/// Marsaglia and Tsang (2000), with the `U^(1/shape)` boost below shape 1,
/// since 2026-10-08 (`docs/design/rng.md`, stability log); inverse
/// transform before.
///
/// The Student t copula's chi-square and the Clayton copula's frailty run
/// the same code, but they belong to the copulas' documented method
/// ([`crate::copula`]), Marsaglia–Tsang since release, and this entry does
/// not describe them: in a record without it they are still drawn by
/// Marsaglia–Tsang.
pub const GAMMA_SAMPLER: &str = "marsaglia-tsang/2026-10";

/// The samplers this build draws with, as `(family, sampler id)` pairs
/// sorted by family, recorded by [`Provenance::seed`] (and
/// [`current_samplers`]).
///
/// A family is listed once its sampler differs from the one it was first
/// released with. A family missing from a recorded table uses that first
/// sampler: inverse transform for every [`crate::Distribution`] and
/// [`crate::Counting`] family (except the inverse gamma, which draws as
/// one over a Gamma variate and so follows the `gamma` entry), the method documented in [`crate::copula`]
/// for a copula's frailty (its Gamma variates included). Today only the
/// Gamma distribution is listed (inverse transform until 2026-10-08).
///
/// A change to a sampler's draws on a given stream gives it a new id,
/// `<method>/<year>-<month>` of the change (a second change in one month
/// adds the day), with an entry in the stability log of
/// `docs/design/rng.md`. Ids are compared as text and never reused. A
/// change to a model's documented draw order is listed the same way under
/// the model's name (none so far).
pub const SAMPLERS: &[(&str, &str)] = &[("gamma", GAMMA_SAMPLER)];

/// Audit record carried by every [`crate::PredictiveDistribution`], so a
/// result can be traced to its model and replayed from its seed.
///
/// ```
/// use prospicio_prob::Provenance;
/// use prospicio_prob::provenance::{GAMMA_SAMPLER, SIM_INDEX_SCHEME};
///
/// let p = Provenance::new("odp_bootstrap").param("n_sims", 10_000);
/// assert_eq!(p.model, "odp_bootstrap");
/// assert_eq!(p.parameters, [("n_sims".to_string(), "10000".to_string())]);
/// assert_eq!(p.versions[0].0, "prospicio-prob");
/// assert_eq!(p.samplers, None);
///
/// let p = p.seed(42, SIM_INDEX_SCHEME);
/// assert_eq!(
///     p.samplers,
///     Some(vec![("gamma".to_string(), GAMMA_SAMPLER.to_string())])
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// Model that produced the result, e.g. `"odp_bootstrap"`.
    pub model: String,
    /// Model parameters in the order the model reports them.
    pub parameters: Vec<(String, String)>,
    /// Seed of the simulation, if the result is simulated.
    pub seed: Option<u64>,
    /// Rule mapping simulations to streams, e.g. [`SIM_INDEX_SCHEME`]: which
    /// uniforms a simulation draws, not how they become draws (that is
    /// [`samplers`](Self::samplers)).
    pub stream_scheme: Option<String>,
    /// Samplers the draws were made with, as `(family, sampler id)` pairs
    /// sorted by family: the [`SAMPLERS`] of the build that drew them, so
    /// every sampler the draws may have used, not only those they did use.
    /// A family missing from it draws by its first sampler (inverse
    /// transform for a distribution, the documented method for a copula's
    /// frailty). `None` when not recorded: a result without draws, one
    /// computed from draws this build did not make or cannot vouch for
    /// (years of losses from elsewhere, a blend of models with different
    /// records), or one made by a build from before this field was split
    /// from the stream scheme on 2026-10-08, whose Gamma draws (if any) may
    /// come from either Gamma sampler.
    pub samplers: Option<Vec<(String, String)>>,
    /// Crate versions involved, starting with `prospicio-prob`.
    pub versions: Vec<(String, String)>,
    /// Hash of the model's canonical input, from [`InputHasher`].
    pub input_hash: Option<String>,
}

impl Provenance {
    /// Provenance for `model`, recording this crate's version.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            parameters: Vec::new(),
            seed: None,
            stream_scheme: None,
            samplers: None,
            versions: vec![("prospicio-prob".into(), env!("CARGO_PKG_VERSION").into())],
            input_hash: None,
        }
    }

    /// Adds a model parameter.
    pub fn param(mut self, name: impl Into<String>, value: impl ToString) -> Self {
        self.parameters.push((name.into(), value.to_string()));
        self
    }

    /// Records the version of another crate involved, e.g. the model's.
    pub fn version(mut self, krate: impl Into<String>, version: impl Into<String>) -> Self {
        self.versions.push((krate.into(), version.into()));
        self
    }

    /// Records the seed, the stream scheme it was used with, and this
    /// build's [`SAMPLERS`] as the samplers: call it when this build makes
    /// the draws. A result computed from another result's draws takes that
    /// result's record with [`draws_from`](Self::draws_from) instead.
    pub fn seed(mut self, seed: u64, stream_scheme: impl Into<String>) -> Self {
        self.seed = Some(seed);
        self.stream_scheme = Some(stream_scheme.into());
        self.samplers = Some(current_samplers());
        self
    }

    /// Copies the seed, stream scheme and samplers of `source`, for a result
    /// computed from `source`'s draws without drawing again (a reinsurance
    /// tower applied to a reserve bootstrap).
    pub fn draws_from(mut self, source: &Provenance) -> Self {
        self.seed = source.seed;
        self.stream_scheme = source.stream_scheme.clone();
        self.samplers = source.samplers.clone();
        self
    }

    /// Records the hash of the model's canonical input, normally
    /// [`InputHasher::finish`].
    pub fn input_hash(mut self, hash: impl Into<String>) -> Self {
        self.input_hash = Some(hash.into());
        self
    }

    /// Whether simulation `i` of `self` and of `other` draw the same
    /// uniforms: both have a seed and a stream scheme, and they are equal.
    /// The samplers are not compared, since two results drawing the same
    /// uniforms are dependent whatever turns them into draws; this is the
    /// check [`crate::PredictiveDistribution::join`] makes for independent
    /// parts.
    ///
    /// ```
    /// use prospicio_prob::Provenance;
    /// use prospicio_prob::provenance::SIM_INDEX_SCHEME;
    ///
    /// let a = Provenance::new("odp_bootstrap").seed(1, SIM_INDEX_SCHEME);
    /// let mut b = Provenance::new("collective").seed(1, SIM_INDEX_SCHEME);
    /// b.samplers = None; // as read from a file saved before samplers were recorded
    /// assert!(a.shares_streams(&b));
    /// assert!(!a.shares_streams(&Provenance::new("collective").seed(2, SIM_INDEX_SCHEME)));
    /// assert!(!Provenance::new("fit").shares_streams(&Provenance::new("fit")));
    /// ```
    pub fn shares_streams(&self, other: &Provenance) -> bool {
        self.seed.is_some()
            && self.stream_scheme.is_some()
            && self.seed == other.seed
            && self.stream_scheme == other.stream_scheme
    }

    /// Whether `self` and `other` make the same draws simulation by
    /// simulation: they [share streams](Self::shares_streams) and record
    /// the same samplers. A record without samplers matches nothing, since
    /// its Gamma draws may come from either sampler. With the same model,
    /// parameters and input hash, the results are then the same.
    ///
    /// To ask whether this build replays a saved result, compare it with a
    /// record this build makes on the same seed:
    ///
    /// ```
    /// use prospicio_prob::Provenance;
    /// use prospicio_prob::provenance::SIM_INDEX_SCHEME;
    ///
    /// let saved = Provenance::new("odp_bootstrap").seed(7, SIM_INDEX_SCHEME);
    /// let now = Provenance::new("odp_bootstrap").seed(7, SIM_INDEX_SCHEME);
    /// assert!(saved.replays_same_draws(&now));
    ///
    /// // Drawn when `Gamma::sample` was by inverse transform: same
    /// // streams, other draws.
    /// let mut old = saved.clone();
    /// old.samplers = Some(vec![]);
    /// assert!(old.shares_streams(&now));
    /// assert!(!old.replays_same_draws(&now));
    /// ```
    pub fn replays_same_draws(&self, other: &Provenance) -> bool {
        self.shares_streams(other) && self.samplers.is_some() && self.samplers == other.samplers
    }
}

/// [`SAMPLERS`] as owned pairs: the samplers of draws this build makes,
/// for a record of draws made earlier in this build (years of losses
/// simulated and kept to apply terms to later).
pub fn current_samplers() -> Vec<(String, String)> {
    SAMPLERS
        .iter()
        .map(|&(family, id)| (family.into(), id.into()))
        .collect()
}

/// BLAKE3 key-derivation context for [`InputHasher`]. It separates input
/// hashes from any other use of BLAKE3, and changes (`v2`, …) whenever the
/// encoding below changes, so hashes from different encodings never match.
pub const INPUT_HASH_CONTEXT: &str = "risk-rs 2026-09-30 input-hash v1";

/// Hashes a model's input for [`Provenance::input_hash`], so an audit can
/// confirm that two results came from the same data.
///
/// Each field is written as a one-byte type tag, a little-endian `u64`
/// length, then the payload, so neither field boundaries nor types can be
/// confused: `str("ab").str("c")` differs from `str("a").str("bc")`, and
/// `u64(1)` from `i64(1)`. Floats are hashed by their bits, with `-0.0`
/// written as `0.0` and every NaN as one canonical NaN, so numerically
/// equal inputs hash equal.
///
/// ```
/// use prospicio_prob::provenance::InputHasher;
///
/// let a = InputHasher::new().str("origin").f64s(&[100.0, 150.0]).finish();
/// let b = InputHasher::new().str("origin").f64s(&[100.0, 150.0]).finish();
/// assert_eq!(a, b);
/// assert!(a.starts_with("blake3:"));
/// assert_ne!(a, InputHasher::new().str("origin").f64s(&[100.0, 151.0]).finish());
/// ```
#[derive(Debug, Clone)]
pub struct InputHasher {
    inner: blake3::Hasher,
}

impl Default for InputHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl InputHasher {
    const BYTES: u8 = 1;
    const STR: u8 = 2;
    const U64: u8 = 3;
    const I64: u8 = 4;
    const F64S: u8 = 5;

    /// An empty hasher keyed by [`INPUT_HASH_CONTEXT`].
    pub fn new() -> Self {
        Self {
            inner: blake3::Hasher::new_derive_key(INPUT_HASH_CONTEXT),
        }
    }

    fn header(&mut self, tag: u8, len: usize) {
        self.inner.update(&[tag]);
        self.inner.update(&(len as u64).to_le_bytes());
    }

    /// Adds raw bytes, e.g. an Arrow IPC buffer.
    pub fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        self.header(Self::BYTES, bytes.len());
        self.inner.update(bytes);
        self
    }

    /// Adds UTF-8 text, e.g. a column or dimension name.
    pub fn str(&mut self, text: &str) -> &mut Self {
        self.header(Self::STR, text.len());
        self.inner.update(text.as_bytes());
        self
    }

    /// Adds an unsigned integer.
    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.header(Self::U64, 8);
        self.inner.update(&value.to_le_bytes());
        self
    }

    /// Adds a signed integer.
    pub fn i64(&mut self, value: i64) -> &mut Self {
        self.header(Self::I64, 8);
        self.inner.update(&value.to_le_bytes());
        self
    }

    /// Adds a sequence of floats; the length is the number of values.
    pub fn f64s(&mut self, values: &[f64]) -> &mut Self {
        self.header(Self::F64S, values.len());
        for &x in values {
            self.inner.update(&canonical_bits(x).to_le_bytes());
        }
        self
    }

    /// The hash of everything added so far, as `"blake3:"` followed by 64
    /// hex digits. The hasher can keep taking fields afterwards.
    pub fn finish(&self) -> String {
        format!("blake3:{}", self.inner.finalize().to_hex())
    }
}

/// Bits of `x` with `-0.0` mapped to `0.0` and every NaN to one NaN.
fn canonical_bits(x: f64) -> u64 {
    if x == 0.0 {
        0
    } else if x.is_nan() {
        f64::NAN.to_bits()
    } else {
        x.to_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_table_is_sorted_with_one_id_per_family() {
        assert!(SAMPLERS.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(
            SAMPLERS
                .iter()
                .all(|(f, id)| !f.is_empty() && !id.is_empty())
        );
        assert_eq!(SAMPLERS, [("gamma", GAMMA_SAMPLER)]);
    }

    #[test]
    fn seed_records_the_samplers_and_draws_from_copies_them() {
        let p = Provenance::new("m").seed(3, SIM_INDEX_SCHEME);
        assert_eq!(p.samplers, Some(current_samplers()));

        let mut old = Provenance::new("odp_bootstrap").seed(3, SIM_INDEX_SCHEME);
        old.samplers = None;
        let derived = Provenance::new("tower").draws_from(&old);
        assert_eq!(derived.seed, Some(3));
        assert_eq!(derived.stream_scheme.as_deref(), Some(SIM_INDEX_SCHEME));
        assert_eq!(derived.samplers, None);
        assert_eq!(Provenance::new("t").draws_from(&p).samplers, p.samplers);
        assert_eq!(
            Provenance::new("t").draws_from(&Provenance::new("fit")),
            Provenance::new("t")
        );
    }

    #[test]
    fn replay_needs_streams_and_samplers_but_stream_sharing_needs_streams_only() {
        let now = Provenance::new("m").seed(9, SIM_INDEX_SCHEME);
        let mut older_gamma = now.clone();
        older_gamma.samplers = Some(vec![("gamma".into(), "marsaglia-tsang/2026-09".into())]);
        let mut unrecorded = now.clone();
        unrecorded.samplers = None;
        let other_seed = Provenance::new("m").seed(10, SIM_INDEX_SCHEME);
        let other_scheme = Provenance::new("m").seed(9, "chacha20/sim-index/v2");

        assert!(now.replays_same_draws(&now.clone()));
        for p in [&older_gamma, &unrecorded] {
            assert!(now.shares_streams(p) && p.shares_streams(&now));
            assert!(!now.replays_same_draws(p) && !p.replays_same_draws(&now));
        }
        assert!(!unrecorded.replays_same_draws(&unrecorded.clone()));
        for p in [&other_seed, &other_scheme] {
            assert!(!now.shares_streams(p));
            assert!(!now.replays_same_draws(p));
        }
        let unseeded = Provenance::new("fit");
        assert!(!unseeded.shares_streams(&unseeded.clone()));
        assert!(!unseeded.replays_same_draws(&unseeded.clone()));
    }

    #[test]
    fn matches_an_independent_blake3() {
        // Reproduced with the Python `blake3` package from bytes built by
        // hand from the documented encoding:
        // validation/scripts/input_hash_golden.py.
        let h = InputHasher::new()
            .str("origin")
            .i64(1981)
            .u64(10)
            .f64s(&[5012.0, -0.0])
            .bytes(b"arrow")
            .finish();
        assert_eq!(h, GOLDEN);
        assert_eq!(InputHasher::new().finish(), GOLDEN_EMPTY);
    }

    const GOLDEN: &str = "blake3:ed7cae0b254756dbb56c21f9e069020938edd8bd1ddab977a75662dac9e53380";
    const GOLDEN_EMPTY: &str =
        "blake3:78ea3e208e78c7ab65dbea55824fdc5df3d386ef598ad5419bbf61d456013244";

    fn hash(f: impl FnOnce(&mut InputHasher)) -> String {
        let mut h = InputHasher::new();
        f(&mut h);
        h.finish()
    }

    #[test]
    fn field_boundaries_and_types_are_distinct() {
        assert_ne!(
            hash(|h| {
                h.str("ab").str("c");
            }),
            hash(|h| {
                h.str("a").str("bc");
            })
        );
        assert_ne!(
            hash(|h| {
                h.u64(1);
            }),
            hash(|h| {
                h.i64(1);
            })
        );
        assert_ne!(
            hash(|h| {
                h.str("a");
            }),
            hash(|h| {
                h.bytes(b"a");
            })
        );
        assert_ne!(
            hash(|h| {
                h.f64s(&[1.0, 2.0]);
            }),
            hash(|h| {
                h.f64s(&[1.0]).f64s(&[2.0]);
            })
        );
    }

    #[test]
    fn equal_numbers_hash_equal() {
        assert_eq!(
            hash(|h| {
                h.f64s(&[-0.0]);
            }),
            hash(|h| {
                h.f64s(&[0.0]);
            })
        );
        let other_nan = f64::from_bits(f64::NAN.to_bits() | 1);
        assert_eq!(
            hash(|h| {
                h.f64s(&[other_nan]);
            }),
            hash(|h| {
                h.f64s(&[f64::NAN]);
            })
        );
        assert_ne!(
            hash(|h| {
                h.f64s(&[1.0]);
            }),
            hash(|h| {
                h.f64s(&[1.0 + f64::EPSILON]);
            })
        );
    }

    #[test]
    fn is_not_plain_blake3() {
        assert_ne!(
            InputHasher::new().finish(),
            format!("blake3:{}", blake3::hash(b"").to_hex())
        );
    }
}
