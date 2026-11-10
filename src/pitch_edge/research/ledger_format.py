"""The research ledger's hash-chain format as a versioned contract other projects can vendor.

agentic-trading writes ledgers in this format and edgelab recomputes chains exported by `Ledger.export_chain`, so
the format is published as one JSON document, `docs/contracts/research-ledger-format.v<FORMAT_VERSION>.json`, built
by `format_spec()` from the constants the ledger itself hashes with. A consumer copies the file (recording the
pitch-edge sha it came from) and checks its own implementation against the `vectors` in it, without this repository.
A `.sha256` file beside it (`sha256sum` format) lets a consumer check its copy.
`tests/unit/test_research_ledger_format.py` fails if the committed files and the code disagree, so a format change
cannot ship without a new version number and a new file.

    from pitch_edge.research.ledger_format import format_spec, write_spec
    write_spec(Path("docs/contracts"))  # writes research-ledger-format.v1.json
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pitch_edge.research.ledger import _COLUMNS, _TIME_COLUMN, DIRECTIONS, GENESIS, REVIEW_KINDS, _chain_hash, _row_json

FORMAT_NAME = "pitch-edge-research-ledger"
FORMAT_VERSION = 1

_SQL_TYPES: dict[str, dict[str, str]] = {
    "registrations": {
        "id": "string",
        "family": "string",
        "sha256": "string",
        "path": "string",
        "registered_at": "timestamp",
        "recorded_at": "timestamp",
    },
    "trials": {
        "trial_id": "string",
        "hypothesis_id": "string",
        "family": "string",
        "config_json": "string",
        "config_hash": "string",
        "git_sha": "string",
        "data_snapshot": "string",
        "run_at": "timestamp",
        "n": "integer",
        "metric_name": "string",
        "metric_value": "number|null",
        "ci_lo": "number|null",
        "ci_hi": "number|null",
        "p_value": "number|null",
        "direction": "string",
        "extra_json": "string",
    },
    "reviews": {
        "hypothesis_id": "string",
        "kind": "string",
        "verdict": "string",
        "detail_json": "string",
        "recorded_at": "timestamp",
    },
}

_VECTOR_ROWS: list[tuple[str, dict[str, Any]]] = [
    (
        "registrations",
        {
            "id": "demo-v1",
            "family": "demo",
            "sha256": "ab" * 32,
            "path": "research/registry/demo-v1.yaml",
            "registered_at": datetime(2026, 10, 28, 1, 2, 3),
            "recorded_at": datetime(2026, 10, 28, 1, 2, 4, 500000),
        },
    ),
    (
        "trials",
        {
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
        },
    ),
    (
        "reviews",
        {
            "hypothesis_id": "demo-v1",
            "kind": "amendment",
            "verdict": "RECORDED",
            "detail_json": '{"amends_sha256": "' + "ab" * 32 + '", "rules": ["a rule"]}',
            "recorded_at": datetime(2026, 10, 28, 1, 30, 0, 250000),
        },
    ),
]


def vector_rows() -> list[tuple[str, dict[str, Any]]]:
    """The (table, row) pairs the published vectors are computed from, chained in this order from genesis."""
    return [(table, dict(row)) for table, row in _VECTOR_ROWS]


def format_spec() -> dict[str, Any]:
    """The format document: tables and their hashed columns, the canonicalisation and chain rules, the export line
    kinds, and test vectors (each row's exact hashed string and its hash, chained from genesis in list order)."""
    vectors = []
    prev = GENESIS
    for table, row in _VECTOR_ROWS:
        row_hash = _chain_hash(prev, table, row)
        vectors.append(
            {"table": table, "prev_hash": prev, "row_canonical": _row_json(table, row), "row_hash": row_hash}
        )
        prev = row_hash
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "tables": {
            table: {
                "columns": {c: _SQL_TYPES[table][c] for c in cols},
                "time_column": _TIME_COLUMN[table],
                "row_key": {
                    "registrations": "id",
                    "trials": "trial_id",
                    "reviews": "sha256 hex of row_canonical (reviews have no natural key)",
                }[table],
            }
            for table, cols in _COLUMNS.items()
        },
        "enums": {"direction": list(DIRECTIONS), "review_kind": list(REVIEW_KINDS)},
        "canonicalisation": {
            "row_canonical": "json.dumps({column: canonical(value) for column in the table's columns}, "
            "sort_keys=True, separators=(',', ':')); only the listed columns are hashed",
            "timestamp": "naive UTC, Python datetime.isoformat() (microseconds only when non-zero)",
            "nan": 'the string "NaN"',
            "null": "JSON null",
            "json_text_columns": "config_json, extra_json and detail_json enter the row as strings, unchanged; "
            "detail_json is json.dumps(detail, sort_keys=True) with the default separators",
            "encoding": "UTF-8; json.dumps default ensure_ascii=True",
        },
        "chain": {
            "genesis_prev_hash": GENESIS,
            "row_hash": "sha256_hex(f'{prev_hash}|{table}|{row_canonical}')",
            "order": "write order; each entry's prev_hash is the previous entry's row_hash",
        },
        "export_lines": {
            "chain_entry": ["table", "row_canonical", "prev_hash", "row_hash"],
            "unchained_row": ["table", "row_canonical", "chained (false)"],
            "registration_yaml": [
                "table ('registration_yaml')",
                "hypothesis_id",
                "sha256",
                "yaml",
                "chained?",
                "mismatch?",
            ],
            "summary": [
                "family",
                "upto",
                "n_tests",
                "n_distinct_hypothesis_config",
                "n_return_kind",
                "return_test_keys",
                "test_keys",
            ],
        },
        "vectors": vectors,
    }


def spec_filename() -> str:
    return f"research-ledger-format.v{FORMAT_VERSION}.json"


def render_spec() -> str:
    """The format document as the exact text of the published file (2-space indent, trailing newline)."""
    return json.dumps(format_spec(), indent=2) + "\n"


def render_checksum() -> str:
    """The text of the published checksum file: `sha256sum` format, `<hex>  <spec file name>` and a newline."""
    digest = hashlib.sha256(render_spec().encode("utf-8")).hexdigest()
    return f"{digest}  {spec_filename()}\n"


def write_spec(directory: Path) -> Path:
    """Write the format document and its `.sha256` file into `directory` (created if missing); returns the
    document's path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / spec_filename()
    path.write_text(render_spec())
    path.with_name(path.name + ".sha256").write_text(render_checksum())
    return path
