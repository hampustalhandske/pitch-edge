"""The research ledger's hash-chain format is shared with other projects' ledgers, which recompute it
independently: these fixed vectors fail if the columns, the canonical JSON or the chain hash change.

Format: row_json = json.dumps({column: canonical value}, sort_keys=True, separators=(",", ":")), with
datetimes as naive-UTC isoformat, NaN as the string "NaN" and None as null; row_hash =
sha256_hex(f"{prev_hash}|{table}|{row_json}"); the first entry's prev_hash is "0" * 64."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from pitch_edge.research.ledger import GENESIS, _chain_hash, _row_json
from pitch_edge.research.ledger_format import (
    FORMAT_VERSION,
    format_spec,
    render_checksum,
    render_spec,
    spec_filename,
    vector_rows,
)

REGISTRATION = {
    "id": "demo-v1",
    "family": "demo",
    "sha256": "ab" * 32,
    "path": "research/registry/demo-v1.yaml",
    "registered_at": datetime(2026, 10, 28, 1, 2, 3),
    "recorded_at": datetime(2026, 10, 28, 1, 2, 4, 500000),
}
TRIAL = {
    "trial_id": "t1",
    "hypothesis_id": "demo-v1",
    "family": "demo",
    "config_json": '{"theta":0.02}',
    "config_hash": "cd" * 32,
    "git_sha": "0" * 40,
    "data_snapshot": "lake@2026-10-28",
    "run_at": datetime(2026, 10, 28, 2, 0, 0),
    "n": 300,
    "metric_name": "clv_log",
    "metric_value": 0.0123,
    "ci_lo": float("nan"),
    "ci_hi": None,
    "p_value": 0.01,
    "direction": "positive",
    "extra_json": "{}",
}
# A review row (here an `amendment`). `detail_json` is the review detail as `json.dumps(detail, sort_keys=True)`
# with the default separators (", " and ": "), and it enters the row JSON as a string.
REVIEW = {
    "hypothesis_id": "demo-v1",
    "kind": "amendment",
    "verdict": "RECORDED",
    "detail_json": '{"amends_sha256": "' + "ab" * 32 + '", "rules": ["a rule"]}',
    "recorded_at": datetime(2026, 10, 28, 1, 30, 0, 250000),
}
REGISTRATION_HASH = "13c98f3cdd15afc3fd9573cd933721ee65a81746ae35992fe81d064f361d3894"
TRIAL_HASH = "1887978452d3a9f6fde6560cf0fafbd63acf36bf81a3b20ae06dd1eb2ad1d623"
REVIEW_HASH = "297a6a2f82ae5ea5c8ced6c545e51085a274b03030101d26add9d144e9048bf4"  # chained after TRIAL


def test_genesis_is_64_zeros():
    assert GENESIS == "0" * 64


def test_registration_row_json_is_sorted_compact_naive_iso():
    assert _row_json("registrations", REGISTRATION) == (
        '{"family":"demo","id":"demo-v1","path":"research/registry/demo-v1.yaml",'
        '"recorded_at":"2026-10-28T01:02:04.500000","registered_at":"2026-10-28T01:02:03",'
        f'"sha256":"{"ab" * 32}"}}'
    )


def test_chain_hash_vectors():
    first = _chain_hash(GENESIS, "registrations", REGISTRATION)
    assert first == REGISTRATION_HASH
    assert _chain_hash(first, "trials", TRIAL) == TRIAL_HASH
    assert _chain_hash(TRIAL_HASH, "reviews", REVIEW) == REVIEW_HASH


def test_review_row_json_embeds_detail_as_a_string():
    assert _row_json("reviews", REVIEW) == (
        '{"detail_json":"{\\"amends_sha256\\": \\"' + "ab" * 32 + '\\", \\"rules\\": [\\"a rule\\"]}",'
        '"hypothesis_id":"demo-v1","kind":"amendment","recorded_at":"2026-10-28T01:30:00.250000",'
        '"verdict":"RECORDED"}'
    )


def test_trial_row_json_encodes_nan_as_string_and_none_as_null():
    row_json = _row_json("trials", TRIAL)
    assert '"ci_lo":"NaN"' in row_json
    assert '"ci_hi":null' in row_json
    assert '"n":300,' in row_json


# --------------------------------------------------------------------------- published, versioned format file

SPEC_DIR = Path(__file__).resolve().parents[2] / "docs" / "contracts"


def test_the_published_format_file_matches_the_code():
    path = SPEC_DIR / spec_filename()
    assert path.exists(), f"missing {path}: run `pitch_edge.research.ledger_format.write_spec(Path('docs/contracts'))`"
    assert path.read_text() == render_spec(), "the format changed: bump FORMAT_VERSION and publish a new file"
    checksum = path.with_name(path.name + ".sha256").read_text()
    assert checksum == render_checksum()
    assert checksum.split()[0] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_published_vectors_are_the_fixed_vectors_above():
    spec = format_spec()
    assert spec["version"] == FORMAT_VERSION
    assert [(t, r) for t, r in vector_rows()][0][1] == REGISTRATION
    assert [v["row_hash"] for v in spec["vectors"]] == [REGISTRATION_HASH, TRIAL_HASH, REVIEW_HASH]
    for v in spec["vectors"]:
        payload = f"{v['prev_hash']}|{v['table']}|{v['row_canonical']}".encode()
        assert hashlib.sha256(payload).hexdigest() == v["row_hash"]
    assert spec["chain"]["genesis_prev_hash"] == GENESIS


def test_a_changed_canonicalisation_no_longer_matches_the_published_file(monkeypatch):
    import pitch_edge.research.ledger_format as lf

    monkeypatch.setattr(lf, "_row_json", lambda table, row: json.dumps({k: str(v) for k, v in row.items()}))
    assert lf.render_spec() != (SPEC_DIR / spec_filename()).read_text()
