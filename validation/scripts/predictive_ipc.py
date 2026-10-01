"""Check act-prob's PredictiveDistribution Arrow IPC format (version 1)
with an independent implementation (pyarrow, not the Rust arrow crates).

    pip install pyarrow
    python validation/scripts/predictive_ipc.py           # write the fixture
    python validation/scripts/predictive_ipc.py FILE.arrow  # check FILE

Without arguments, writes validation/reference/predictive_distribution_v1.arrow
from the format documented in crates/act-prob/src/ipc.rs. The Rust test
validation/tests/predictive.rs reads it back, so a v1 file written by
another tool keeps loading. With a path, reads a file written by Rust and
checks that it follows the same format.
"""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather

FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "reference"
    / "predictive_distribution_v1.arrow"
)

DIMS = ["lob", "origin"]
KEYS = [
    [{"text": "Auto"}, {"period": {"start": "2019-01", "grain": "Y"}}],
    [{"text": "Auto"}, {"period": {"start": "2020-07", "grain": "S"}}],
    [{"text": "Home"}, {"int": 7}],
]
NAMES = ["Auto/2019", "Auto/2020H2", "Home/7"]
COLUMNS = [[1.5, -2.0, 1e300], [0.0, 4.25, 5.0], [7.0, 8.0, 9.0]]
PROVENANCE = {
    "model": "fixture",
    "parameters": [["n_sims", "3"]],
    "seed": "18446744073709551615",
    "stream_scheme": "chacha20/sim-index/v1",
    "versions": [["act-prob", "0.0.1"]],
    "input_hash": None,
}


def write_fixture():
    fields = [
        pa.field(name, pa.float64(), nullable=False,
                 metadata={"risk_rs.key": json.dumps(key)})
        for name, key in zip(NAMES, KEYS)
    ]
    schema = pa.schema(fields, metadata={
        "risk_rs.format": "predictive_distribution",
        "risk_rs.format_version": "1",
        "risk_rs.dims": json.dumps(DIMS),
        "risk_rs.provenance": json.dumps(PROVENANCE),
    })
    table = pa.table([pa.array(c, pa.float64()) for c in COLUMNS], schema=schema)
    feather.write_feather(table, FIXTURE, compression="uncompressed")
    print(f"wrote {FIXTURE}")


def check(path):
    table = feather.read_table(path)
    meta = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
    assert meta["risk_rs.format"] == "predictive_distribution", meta
    assert meta["risk_rs.format_version"] == "1", meta
    dims = json.loads(meta["risk_rs.dims"])
    prov = json.loads(meta["risk_rs.provenance"])
    assert isinstance(prov["model"], str)
    assert prov["seed"] is None or int(prov["seed"]) >= 0
    for field in table.schema:
        assert field.type == pa.float64(), field
        key = json.loads(field.metadata[b"risk_rs.key"])
        assert len(key) == len(dims), (field.name, key)
    for column in table.columns:
        assert column.null_count == 0
    print(f"ok: {table.num_rows} sims x {table.num_columns} components, dims {dims}")
    print(table.slice(0, 5))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        check(sys.argv[1])
    else:
        write_fixture()
