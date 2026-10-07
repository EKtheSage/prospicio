"""Reproduce the golden values in prospicio-prob's provenance tests with an
independent BLAKE3 (the `blake3` Python package, not the Rust crate).

    pip install blake3
    python validation/scripts/input_hash_golden.py

The bytes are built from the encoding described on `InputHasher`: per
field, a one-byte tag, a little-endian u64 length, then the payload.
"""

import struct

import blake3

CTX = "risk-rs 2026-09-30 input-hash v1"  # INPUT_HASH_CONTEXT
BYTES, STR, U64, I64, F64S = 1, 2, 3, 4, 5


def field(tag, length, payload):
    return bytes([tag]) + struct.pack("<Q", length) + payload


def f64(x):
    # -0.0 is written as 0.0 (the tests use no NaN).
    return struct.pack("<Q", 0) if x == 0.0 else struct.pack("<d", x)


def digest(msg):
    return "blake3:" + blake3.blake3(msg, derive_key_context=CTX).hexdigest()


msg = (
    field(STR, 6, b"origin")
    + field(I64, 8, struct.pack("<q", 1981))
    + field(U64, 8, struct.pack("<Q", 10))
    + field(F64S, 2, f64(5012.0) + f64(-0.0))
    + field(BYTES, 5, b"arrow")
)
print("GOLDEN      ", digest(msg))
print("GOLDEN_EMPTY", digest(b""))
