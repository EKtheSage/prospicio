"""Reproduce the pinned Gamma draws in prospicio-prob's `gamma` tests
(`sample_is_pinned`) independently: ChaCha20 from the `cryptography`
package, the normal quantile from SciPy's `ndtri`, and Marsaglia and Tsang
(2000) written out here.

    uv run --no-project --with cryptography --with scipy python validation/scripts/gamma_sampler.py

The stream is `StreamRng::new(seed, stream)` (docs/design/rng.md): the
original ChaCha20 layout with a 64-bit block counter from 0 and the stream
as the 64-bit nonce, keyed by the seed expanded with SplitMix64; a uniform
is `(top 53 bits + 0.5) / 2^53`. SciPy's `ndtri` and prospicio's
`norm_quantile` can differ in the last bits, so the draws agree to about
1e-15, not bit for bit.
"""

import math
import struct

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from scipy.special import ndtri

MASK = (1 << 64) - 1


def key(seed):
    """SplitMix64 expansion of a 64-bit seed to a 256-bit key."""
    state, out = seed, b""
    for _ in range(4):
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        z ^= z >> 31
        out += struct.pack("<Q", z)
    return out


class Stream:
    def __init__(self, seed, stream):
        # cryptography's 16-byte nonce: the 64-bit counter, then the nonce.
        nonce = struct.pack("<Q", 0) + struct.pack("<Q", stream)
        cipher = Cipher(algorithms.ChaCha20(key(seed), nonce), mode=None)
        self.keystream = cipher.encryptor()

    def open01(self):
        k = struct.unpack("<Q", self.keystream.update(b"\0" * 8))[0] >> 11
        return (k + 0.5) / 2.0**53


def standard_gamma(rng, shape):
    """Marsaglia and Tsang (2000), with the U^(1/shape) boost below 1."""
    if shape < 1:
        g = standard_gamma(rng, shape + 1)
        return g * rng.open01() ** (1 / shape)
    d = shape - 1 / 3
    c = 1 / math.sqrt(9 * d)
    while True:
        x = float(ndtri(rng.open01()))
        v = 1 + c * x
        if v <= 0:
            continue
        v = v**3
        u = rng.open01()
        if u < 1 - 0.0331 * x**4 or math.log(u) < 0.5 * x * x + d - d * v + d * math.log(v):
            return d * v


# Gamma(0.3, scale 2) twice, then Gamma(2.5, scale 400) twice, from one stream.
rng = Stream(42, 3)
draws = [2.0 * standard_gamma(rng, 0.3) for _ in range(2)]
draws += [400.0 * standard_gamma(rng, 2.5) for _ in range(2)]
PINNED = [0.23214754851650782, 0.7755693217108093, 110.9058975015963, 560.609408990688]
for ours, pinned in zip(draws, PINNED):
    print(f"{ours!r:>22} {pinned!r:>22}  relative difference {abs(ours / pinned - 1):.1e}")
    assert abs(ours / pinned - 1) < 1e-14
