"""Append-only research ledger: every registration, trial and review ever recorded.

The ledger is its own DuckDB file (default `<data_dir>/research/ledger.duckdb`), separate from the
warehouse. Several study processes may append to it at once: every call opens a short-lived connection
and, when another process holds the file lock, retries with jittered exponential backoff for up to
`lock_timeout_s` seconds before raising `LedgerLockTimeout`. Read methods open read-only connections and
never create the file or its schema; the schema is created (or additively migrated) only by a write.

Tables:

- `registrations` — one row per hypothesis id: family, sha256 of the registration file, its path
  (repository-relative when inside the repository), the declared `registered_at`, and when the ledger
  first saw it (`recorded_at`).
- `trials` — one row per scored trial (`research.trial.run_trial`) or imported study-local trial.
- `reviews` — independent reviews: `audit` (skeptic-auditor), `edgelab`, `paper` (forward test).
- `ledger_chain` — one entry per row written through this class, in write order:
  row_hash = sha256(prev_hash | table | canonical JSON of the row). `verify_chain` recomputes it, so an
  edited or deleted row, or a row inserted around this class, is detectable. `export_chain` writes the
  chain out with the exact hashed strings, so another project can recompute it from the file alone.

`record_trial` validates every row before writing it (registered grid point, registered metric and
direction, unchanged registration file, 0 <= p <= 1, ci_lo <= estimate <= ci_hi, p consistent with the
stored one-sided p-values); rows marked `imported` in their extra JSON — study-local trials brought in
by `research.importer` — skip the grid/metric checks, since their configs and metric names are the
study's own, and are handled separately by the gates.

The class exposes no update or delete path: dropped ideas keep counting towards the multiple-testing
corrections in `research.gates`, and a registration can never be edited after the fact — an edited file
is refused, and the remedy is a new id with a bumped version suffix.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, TypeVar

import duckdb
import pandas as pd

from pitch_edge.config import get_settings
from pitch_edge.research.errors import LedgerLockTimeout, ProtocolError
from pitch_edge.research.registry import (
    Registration,
    config_hash,
    load,
    relative_registry_path,
    resolve_registry_path,
    sha256,
)
from pitch_edge.research.stats import directional_p_value

__all__ = ["DIRECTIONS", "REVIEW_KINDS", "ChainReport", "Ledger", "ProtocolError", "TrialRecord", "default_ledger_path"]

DIRECTIONS = ("positive", "negative", "two-sided")
REVIEW_KINDS = ("audit", "edgelab", "paper")
GENESIS = "0" * 64
_TOL = 1e-12

_COLUMNS: dict[str, tuple[str, ...]] = {
    "registrations": ("id", "family", "sha256", "path", "registered_at", "recorded_at"),
    "trials": (
        "trial_id",
        "hypothesis_id",
        "family",
        "config_json",
        "config_hash",
        "git_sha",
        "data_snapshot",
        "run_at",
        "n",
        "metric_name",
        "metric_value",
        "ci_lo",
        "ci_hi",
        "p_value",
        "direction",
        "extra_json",
    ),
    "reviews": ("hypothesis_id", "kind", "verdict", "detail_json", "recorded_at"),
}
_TIME_COLUMN = {"registrations": "recorded_at", "trials": "run_at", "reviews": "recorded_at"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS registrations (
    id VARCHAR PRIMARY KEY,
    family VARCHAR NOT NULL,
    sha256 VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    registered_at TIMESTAMP NOT NULL,
    recorded_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS trials (
    trial_id VARCHAR PRIMARY KEY,
    hypothesis_id VARCHAR NOT NULL,
    family VARCHAR NOT NULL,
    config_json VARCHAR NOT NULL,
    config_hash VARCHAR NOT NULL,
    git_sha VARCHAR NOT NULL,
    data_snapshot VARCHAR NOT NULL,
    run_at TIMESTAMP NOT NULL,
    n BIGINT NOT NULL,
    metric_name VARCHAR NOT NULL,
    metric_value DOUBLE,
    ci_lo DOUBLE,
    ci_hi DOUBLE,
    p_value DOUBLE,
    direction VARCHAR NOT NULL,
    extra_json VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    hypothesis_id VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    verdict VARCHAR NOT NULL,
    detail_json VARCHAR NOT NULL,
    recorded_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger_chain (
    seq BIGINT PRIMARY KEY,
    table_name VARCHAR NOT NULL,
    row_key VARCHAR NOT NULL,
    prev_hash VARCHAR NOT NULL,
    row_hash VARCHAR NOT NULL,
    chained_at TIMESTAMP NOT NULL
);
"""

T = TypeVar("T")


def default_ledger_path() -> Path:
    """`<PITCH_EDGE_DATA_DIR>/research/ledger.duckdb` (gitignored with the rest of `data/`)."""
    return get_settings().data_dir / "research" / "ledger.duckdb"


def utc_now() -> datetime:
    """Current UTC time as a naive datetime (the ledger stores every timestamp as naive UTC)."""
    return datetime.now(UTC).replace(tzinfo=None)


def _canonical(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value


def _row_json(table: str, row: dict[str, Any]) -> str:
    return json.dumps({c: _canonical(row[c]) for c in _COLUMNS[table]}, sort_keys=True, separators=(",", ":"))


def _row_key(table: str, row: dict[str, Any]) -> str:
    if table == "registrations":
        return str(row["id"])
    if table == "trials":
        return str(row["trial_id"])
    return hashlib.sha256(_row_json(table, row).encode()).hexdigest()  # reviews have no natural key


def _chain_hash(prev_hash: str, table: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{prev_hash}|{table}|{_row_json(table, row)}".encode()).hexdigest()


def _is_lock_error(exc: BaseException) -> bool:
    return isinstance(exc, duckdb.IOException) and "lock" in str(exc).lower()


@dataclass(frozen=True)
class TrialRecord:
    """One row of the `trials` table. `trial_id` and `run_at` default to a fresh uuid4 hex and now (UTC).
    `config_json`/`extra_json` are JSON strings; `p_value` is one-sided in the registered direction
    (two-sided for `two-sided`)."""

    hypothesis_id: str
    family: str
    config_json: str
    config_hash: str
    git_sha: str
    data_snapshot: str
    n: int
    metric_name: str
    metric_value: float | None
    ci_lo: float | None
    ci_hi: float | None
    p_value: float | None
    direction: str
    extra_json: str
    trial_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    run_at: datetime = field(default_factory=utc_now)

    def as_row(self) -> dict[str, Any]:
        """Column -> value with plain Python types (so the stored row hashes the same when read back)."""

        def num(x: float | None) -> float | None:
            return None if x is None else float(x)

        return {
            "trial_id": str(self.trial_id),
            "hypothesis_id": self.hypothesis_id,
            "family": self.family,
            "config_json": self.config_json,
            "config_hash": self.config_hash,
            "git_sha": self.git_sha,
            "data_snapshot": self.data_snapshot,
            "run_at": pd.Timestamp(self.run_at).to_pydatetime().replace(tzinfo=None),
            "n": int(self.n),
            "metric_name": self.metric_name,
            "metric_value": num(self.metric_value),
            "ci_lo": num(self.ci_lo),
            "ci_hi": num(self.ci_hi),
            "p_value": num(self.p_value),
            "direction": self.direction,
            "extra_json": self.extra_json,
        }


@dataclass(frozen=True)
class ChainReport:
    """Result of `Ledger.verify_chain`.

    `ok` is False when any chain entry no longer matches its row (`broken`: "table:key: reason") or any
    row without a chain entry is newer than the chain's first entry (`unchained_after_start`: table ->
    keys). Rows older than the chain (written before hash-chaining existed) are counted in
    `legacy_unchained` and do not make the report fail.
    """

    ok: bool
    n_chained: int
    broken: list[str] = field(default_factory=list)
    unchained_after_start: dict[str, list[str]] = field(default_factory=dict)
    legacy_unchained: dict[str, int] = field(default_factory=dict)


class Ledger:
    """Append-only access to the research ledger at `path`. Read methods return pandas frames ordered by
    insertion time and return empty results when the file does not exist yet."""

    def __init__(self, path: str | Path | None = None, lock_timeout_s: float = 60.0) -> None:
        self._path = Path(path) if path is not None else default_ledger_path()
        self._lock_timeout_s = lock_timeout_s

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------ connections

    def _retrying(self, fn: Callable[[], T]) -> T:
        deadline = time.monotonic() + self._lock_timeout_s
        delay = 0.01
        while True:
            try:
                return fn()
            except duckdb.Error as exc:
                if not _is_lock_error(exc):
                    raise
                if time.monotonic() >= deadline:
                    raise LedgerLockTimeout(
                        f"{self._path} stayed locked by another process for {self._lock_timeout_s:.0f}s"
                    ) from exc
                time.sleep(min(delay * (1 + random.random()), max(0.0, deadline - time.monotonic())))
                delay = min(delay * 2, 1.0)

    @contextmanager
    def _reader(self) -> Iterator[duckdb.DuckDBPyConnection | None]:
        """A read-only connection, or None when the ledger file does not exist yet."""
        if not self._path.exists():
            yield None
            return
        con = self._retrying(lambda: duckdb.connect(str(self._path), read_only=True))
        try:
            yield con
        finally:
            con.close()

    def _write(self, fn: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        """Run `fn` in one transaction on a read-write connection (schema ensured first), retrying the
        whole attempt while another process holds the lock."""

        def attempt() -> T:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            con = duckdb.connect(str(self._path))
            try:
                con.execute(_SCHEMA)
                con.execute("BEGIN TRANSACTION")
                try:
                    out = fn(con)
                    con.execute("COMMIT")
                except BaseException:
                    con.execute("ROLLBACK")
                    raise
                return out
            finally:
                con.close()

        return self._retrying(attempt)

    @staticmethod
    def _has_table(con: duckdb.DuckDBPyConnection, table: str) -> bool:
        row = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]).fetchone()
        return row is not None and bool(row[0])

    def _select(self, table: str, where: str = "", params: list[Any] | None = None, order: str = "") -> pd.DataFrame:
        with self._reader() as con:
            if con is None or not self._has_table(con, table):
                return pd.DataFrame(columns=list(_COLUMNS[table]))
            return con.execute(f"SELECT * FROM {table}{where}{order}", params or []).df()

    @staticmethod
    def _append(con: duckdb.DuckDBPyConnection, table: str, row: dict[str, Any]) -> None:
        cols = _COLUMNS[table]
        con.execute(
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", [row[c] for c in cols]
        )
        last = con.execute("SELECT seq, row_hash FROM ledger_chain ORDER BY seq DESC LIMIT 1").fetchone()
        seq, prev = (0, GENESIS) if last is None else (int(last[0]) + 1, str(last[1]))
        con.execute(
            "INSERT INTO ledger_chain VALUES (?, ?, ?, ?, ?, ?)",
            [seq, table, _row_key(table, row), prev, _chain_hash(prev, table, row), utc_now()],
        )

    @staticmethod
    def _registration_row(con: duckdb.DuckDBPyConnection, hypothesis_id: str) -> dict[str, Any] | None:
        cur = con.execute("SELECT * FROM registrations WHERE id = ?", [hypothesis_id])
        row = cur.fetchone()
        return None if row is None else dict(zip([d[0] for d in cur.description], row, strict=True))

    # ------------------------------------------------------------------ registrations

    def register(self, registration: Registration) -> bool:
        """Record a file-backed registration. Returns True when newly recorded, False when the same id is
        already registered with the same file hash. Raises ProtocolError when the id is registered with a
        different hash (the file was edited after registration) or the registration was not loaded from
        a file."""
        if registration.source_path is None or registration.source_sha256 is None:
            raise ProtocolError("only a registration loaded from its file (registry.load) can be registered")
        sha = registration.source_sha256

        def op(con: duckdb.DuckDBPyConnection) -> bool:
            existing = self._registration_row(con, registration.id)
            if existing is not None:
                if existing["sha256"] != sha:
                    raise ProtocolError(
                        f"{registration.id} is already registered with sha256 {existing['sha256'][:12]}… but the "
                        f"file now hashes to {sha[:12]}…; registrations are immutable — bump the version (a new id "
                        f"such as …-v2) instead of editing it"
                    )
                return False
            row = {
                "id": registration.id,
                "family": registration.family,
                "sha256": sha,
                "path": relative_registry_path(registration.source_path),  # type: ignore[arg-type]
                "registered_at": registration.registered_at.astimezone(UTC).replace(tzinfo=None),
                "recorded_at": utc_now(),
            }
            self._append(con, "registrations", row)
            return True

        return self._write(op)

    def registration(self, hypothesis_id: str) -> dict[str, Any] | None:
        """The `registrations` row for `hypothesis_id` as a dict, or None if it was never registered."""
        with self._reader() as con:
            if con is None or not self._has_table(con, "registrations"):
                return None
            return self._registration_row(con, hypothesis_id)

    def registrations(self, family: str | None = None) -> pd.DataFrame:
        """All registrations (optionally one family's), oldest first."""
        if family is None:
            return self._select("registrations", order=" ORDER BY recorded_at, id")
        return self._select("registrations", " WHERE family = ?", [family], " ORDER BY recorded_at, id")

    # ------------------------------------------------------------------ trials

    @staticmethod
    def _validate_trial(reg_row: dict[str, Any], row: dict[str, Any]) -> None:
        hid = row["hypothesis_id"]
        if row["family"] != reg_row["family"]:
            raise ProtocolError(f"trial family {row['family']!r} differs from registered family {reg_row['family']!r}")
        if row["direction"] not in DIRECTIONS:
            raise ProtocolError(f"direction must be one of {DIRECTIONS}, got {row['direction']!r}")
        config = json.loads(row["config_json"])
        if config_hash(config) != row["config_hash"]:
            raise ProtocolError("config_hash is not the hash of config_json")
        extra = json.loads(row["extra_json"])
        if not isinstance(extra, dict):
            raise ProtocolError("extra_json must be a JSON object")
        p, est, lo, hi = row["p_value"], row["metric_value"], row["ci_lo"], row["ci_hi"]
        if p is not None and not (0.0 <= p <= 1.0):
            raise ProtocolError(f"p_value must lie in [0, 1], got {p}")
        if lo is not None and hi is not None and lo > hi:
            raise ProtocolError(f"ci_lo {lo} > ci_hi {hi}")
        if extra.get("imported") is True:
            if "window" not in extra:
                raise ProtocolError("an imported trial must record its window")
            return

        path = resolve_registry_path(reg_row["path"])
        try:
            current = sha256(path)
        except OSError as exc:
            raise ProtocolError(f"registration file for {hid} is missing at {path}") from exc
        if current != reg_row["sha256"]:
            raise ProtocolError(f"{path} was edited after {hid} was registered; bump the version")
        reg = load(path)
        if row["config_hash"] not in {config_hash(c) for c in reg.expand_grid()}:
            raise ProtocolError(f"config {config} is not a point of the registered grid {reg.grid}")
        metrics = [reg.primary_metric, *reg.secondary_metrics]
        if row["metric_name"] not in metrics:
            raise ProtocolError(f"metric {row['metric_name']!r} is not registered for {hid}: {metrics}")
        expected_dir = reg.direction_for(config)
        if row["direction"] != expected_dir:
            raise ProtocolError(f"direction {row['direction']!r} differs from the registered {expected_dir!r}")
        if None in (p, est, lo, hi) or any(math.isnan(x) for x in (p, est, lo, hi)):
            raise ProtocolError("a native trial needs p_value, metric_value, ci_lo and ci_hi")
        if not (lo - _TOL <= est <= hi + _TOL):
            raise ProtocolError(f"need ci_lo <= metric_value <= ci_hi, got {lo} <= {est} <= {hi}")
        for key in ("p_gt0", "p_lt0", "n_blocks", "registration_sha256"):
            if extra.get(key) is None:
                raise ProtocolError(f"extra_json must record {key}")
        if extra["registration_sha256"] != reg_row["sha256"]:
            raise ProtocolError("extra_json registration_sha256 differs from the registered sha256")
        expected_p = directional_p_value(float(extra["p_gt0"]), float(extra["p_lt0"]), row["direction"])
        if abs(expected_p - p) > _TOL:
            raise ProtocolError(f"p_value {p} is inconsistent with p_gt0/p_lt0 for direction {row['direction']}")

    def record_trial(self, record: TrialRecord) -> None:
        """Append one trial row after validating it (see the module docstring). Raises ProtocolError for
        an invalid row and duckdb.ConstraintException when `trial_id` already exists."""
        row = record.as_row()

        def op(con: duckdb.DuckDBPyConnection) -> None:
            reg_row = self._registration_row(con, record.hypothesis_id)
            if reg_row is None:
                raise ProtocolError(f"{record.hypothesis_id} is not registered; register it before recording trials")
            self._validate_trial(reg_row, row)
            self._append(con, "trials", row)

        self._write(op)

    def trials(self, hypothesis_id: str | None = None, family: str | None = None) -> pd.DataFrame:
        """Trial rows, optionally filtered by hypothesis id and/or family, oldest first."""
        clauses, params = [], []
        if hypothesis_id is not None:
            clauses.append("hypothesis_id = ?")
            params.append(hypothesis_id)
        if family is not None:
            clauses.append("family = ?")
            params.append(family)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._select("trials", where, params, " ORDER BY run_at, trial_id")

    # ------------------------------------------------------------------ reviews

    def record_review(self, hypothesis_id: str, kind: str, verdict: str, detail: dict[str, Any] | None = None) -> None:
        """Append a review of a registered hypothesis. `kind` is one of `REVIEW_KINDS`; `detail` is stored
        as JSON (for `paper`: `n_settled`, `weeks`, `clv_ci_lo`, ... as read by `research.gates`)."""
        if kind not in REVIEW_KINDS:
            raise ProtocolError(f"review kind must be one of {REVIEW_KINDS}, got {kind!r}")
        row = {
            "hypothesis_id": hypothesis_id,
            "kind": kind,
            "verdict": verdict,
            "detail_json": json.dumps(detail or {}, sort_keys=True),
            "recorded_at": utc_now(),
        }

        def op(con: duckdb.DuckDBPyConnection) -> None:
            if self._registration_row(con, hypothesis_id) is None:
                raise ProtocolError(f"{hypothesis_id} is not registered")
            self._append(con, "reviews", row)

        self._write(op)

    def reviews(self, hypothesis_id: str | None = None) -> pd.DataFrame:
        """Review rows, optionally for one hypothesis, oldest first."""
        if hypothesis_id is None:
            return self._select("reviews", order=" ORDER BY recorded_at")
        return self._select("reviews", " WHERE hypothesis_id = ?", [hypothesis_id], " ORDER BY recorded_at")

    # ------------------------------------------------------------------ integrity

    def _snapshot(self) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], list[tuple[Any, ...]]] | None:
        """Every row by table and row key, and the chain entries in order; None when there is no ledger file."""
        with self._reader() as con:
            if con is None:
                return None
            tables = {t: self._has_table(con, t) for t in (*_COLUMNS, "ledger_chain")}
            rows: dict[str, list[dict[str, Any]]] = {}
            for t, cols in _COLUMNS.items():
                if not tables[t]:
                    rows[t] = []
                    continue
                cur = con.execute(f"SELECT {', '.join(cols)} FROM {t}")
                rows[t] = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
            chain = (
                con.execute(
                    "SELECT seq, table_name, row_key, prev_hash, row_hash, chained_at FROM ledger_chain ORDER BY seq"
                ).fetchall()
                if tables["ledger_chain"]
                else []
            )
        by_key: dict[str, dict[str, list[dict[str, Any]]]] = {t: {} for t in _COLUMNS}
        for t, rs in rows.items():
            for r in rs:
                by_key[t].setdefault(_row_key(t, r), []).append(r)
        return by_key, chain

    def verify_chain(self) -> ChainReport:
        """Recompute the hash chain over every chained row and find rows written around this class."""
        snapshot = self._snapshot()
        if snapshot is None:
            return ChainReport(ok=True, n_chained=0)
        by_key, chain = snapshot

        broken: list[str] = []
        chained: dict[str, set[str]] = {t: set() for t in _COLUMNS}
        prev = GENESIS
        for seq, table, key, prev_hash, row_hash, _ in chain:
            if prev_hash != prev:
                broken.append(f"chain:{seq}: prev_hash does not link to the previous entry")
            candidates = by_key.get(table, {}).get(key, [])
            if not candidates:
                broken.append(f"{table}:{key}: row missing or altered")
            elif not any(_chain_hash(prev_hash, table, r) == row_hash for r in candidates):
                broken.append(f"{table}:{key}: row content changed since it was written")
            chained.setdefault(table, set()).add(key)
            prev = row_hash

        start = min((c[5] for c in chain), default=None)
        after: dict[str, list[str]] = {}
        legacy: dict[str, int] = {}
        for t, keyed in by_key.items():
            for key, rs in keyed.items():
                if key in chained[t]:
                    continue
                for r in rs:
                    ts = r[_TIME_COLUMN[t]]
                    if start is not None and ts is not None and ts >= start:
                        after.setdefault(t, []).append(key)
                    else:
                        legacy[t] = legacy.get(t, 0) + 1
        return ChainReport(
            ok=not broken and not after,
            n_chained=len(chain),
            broken=broken,
            unchained_after_start=after,
            legacy_unchained=legacy,
        )

    def export_chain(
        self, family: str, upto: str, metric_kinds: Mapping[str, str | None] | None = None
    ) -> list[dict[str, Any]]:
        """The ledger as lines another project can verify on its own, for one family's trial count.

        Lines, in order:
        - every chain entry from genesis to the entry whose `row_hash` is `upto`, of every table and family:
          `{"table", "row_canonical", "prev_hash", "row_hash"}`, where `row_canonical` is the exact string that
          was hashed, so `sha256(f"{prev_hash}|{table}|{row_canonical}") == row_hash` (genesis `prev_hash` is
          64 zeros);
        - the family's rows that were never chained (written before the chain existed):
          `{"table", "row_canonical", "chained": false}`;
        - `{"summary": {family, upto, n_tests, n_distinct_hypothesis_config, n_return_kind, test_keys}}` over the
          family's trial rows in the lines above; `test_keys` are the sorted distinct `"<hypothesis_id>|<config_hash>"`
          and `n_return_kind` counts those whose hypothesis has metric kind `return` in `metric_kinds` (None when
          `metric_kinds` is not given).

        Trial rows carry `family`, `hypothesis_id` and `config_hash`; registrations carry `id` and `family`.
        Raises `ProtocolError` when the chain does not verify up to `upto` or `upto` is not in it.
        """
        snapshot = self._snapshot()
        if snapshot is None:
            raise ProtocolError(f"no ledger at {self.path}")
        by_key, chain = snapshot
        lines: list[dict[str, Any]] = []
        family_rows: list[tuple[str, dict[str, Any]]] = []
        hypotheses = {str(r["id"]) for rs in by_key["registrations"].values() for r in rs if r["family"] == family}

        def in_family(table: str, row: dict[str, Any]) -> bool:
            if table == "reviews":
                return str(row["hypothesis_id"]) in hypotheses
            return row["family"] == family

        prev, reached = GENESIS, False
        for seq, table, key, prev_hash, row_hash, _ in chain:
            if prev_hash != prev:
                raise ProtocolError(f"chain:{seq}: prev_hash does not link to the previous entry")
            row = next((r for r in by_key[table].get(key, []) if _chain_hash(prev_hash, table, r) == row_hash), None)
            if row is None:
                raise ProtocolError(f"{table}:{key}: no stored row matches chain entry {seq}")
            lines.append(
                {"table": table, "row_canonical": _row_json(table, row), "prev_hash": prev_hash, "row_hash": row_hash}
            )
            if in_family(table, row):
                family_rows.append((table, row))
            prev = row_hash
            if row_hash == upto:
                reached = True
                break
        if not reached:
            raise ProtocolError(f"{upto} is not a row_hash in the chain")

        chained_keys = {(table, key) for _, table, key, _, _, _ in chain}
        for table, keyed in by_key.items():
            for key, rs in sorted(keyed.items()):
                if (table, key) in chained_keys:
                    continue
                for r in rs:
                    if in_family(table, r):
                        lines.append({"table": table, "row_canonical": _row_json(table, r), "chained": False})
                        family_rows.append((table, r))

        trials = [r for table, r in family_rows if table == "trials"]
        keys = sorted({f"{r['hypothesis_id']}|{r['config_hash']}" for r in trials})
        n_return = (
            None if metric_kinds is None else sum(1 for k in keys if metric_kinds.get(k.split("|", 1)[0]) == "return")
        )
        lines.append(
            {
                "summary": {
                    "family": family,
                    "upto": upto,
                    "n_tests": len(trials),
                    "n_distinct_hypothesis_config": len(keys),
                    "n_return_kind": n_return,
                    "test_keys": keys,
                }
            }
        )
        return lines
