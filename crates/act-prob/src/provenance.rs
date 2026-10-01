//! Where a result came from: model, parameters, seed, versions and a hash
//! of the input.

/// Name of the rule mapping simulations to RNG streams used by
/// [`crate::PredictiveDistribution::simulate`]: simulation `i` draws only
/// from `StreamRng::new(seed, i)`. See `docs/design/rng.md`.
pub const SIM_INDEX_SCHEME: &str = "chacha20/sim-index/v1";

/// Audit record carried by every [`crate::PredictiveDistribution`], so a
/// result can be traced to its model and replayed from its seed.
///
/// ```
/// use act_prob::Provenance;
///
/// let p = Provenance::new("odp_bootstrap").param("n_sims", 10_000);
/// assert_eq!(p.model, "odp_bootstrap");
/// assert_eq!(p.parameters, [("n_sims".to_string(), "10000".to_string())]);
/// assert_eq!(p.versions[0].0, "act-prob");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// Model that produced the result, e.g. `"odp_bootstrap"`.
    pub model: String,
    /// Model parameters in the order the model reports them.
    pub parameters: Vec<(String, String)>,
    /// Seed of the simulation, if the result is simulated.
    pub seed: Option<u64>,
    /// Rule mapping simulations to streams, e.g. [`SIM_INDEX_SCHEME`].
    pub stream_scheme: Option<String>,
    /// Crate versions involved, starting with `act-prob`.
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
            versions: vec![("act-prob".into(), env!("CARGO_PKG_VERSION").into())],
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

    /// Records the seed and the stream scheme it was used with.
    pub fn seed(mut self, seed: u64, stream_scheme: impl Into<String>) -> Self {
        self.seed = Some(seed);
        self.stream_scheme = Some(stream_scheme.into());
        self
    }

    /// Records the hash of the model's canonical input, normally
    /// [`InputHasher::finish`].
    pub fn input_hash(mut self, hash: impl Into<String>) -> Self {
        self.input_hash = Some(hash.into());
        self
    }
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
/// use act_prob::provenance::InputHasher;
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
