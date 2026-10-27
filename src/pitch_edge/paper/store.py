"""The paper ledger's own DuckDB file (default `data/paper/paper.duckdb`, never the warehouse).

Tables:

- `quotes`: every market price we fetched, one row per (source, event, selection, capture time).
  Bookmaker rows carry `decimal_odds`; exchange / prediction-market rows carry `bid`, `ask`, `last`,
  `size` (dollars per $1 share, `size` = shares available at the best ask). `captured_at` is when *we*
  fetched it (UTC); `raw_json` is the source payload for that selection.
- `fixtures`: one row per canonical fixture (see `paper/fixtures.py`) with each source's event id.
- `decisions`: append-only paper bets. `append_decision` is the only writer and it refuses a decision
  whose fixture has already kicked off, whose inputs were captured after the decision time, or whose
  id already exists. There is no method that changes or removes a decision.
- `settlements`: at most one *final* row per decision, written only from a lake-sourced close (closing
  price and its source, result, CLV, paper P&L).
- `provisional_settlements`: at most one row per decision without a lake close after the grace period
  (our own post-decision snapshot, or `none`); superseded in reporting by a later final settlement.
- `decision_flags`: append-only annotations (`void` for postponed matches or false fixture merges).
- `decide_log`: one row per strategy run (opportunity frequency: fixtures evaluated / qualifying). All
  strategies of one loop cycle share the cycle's `decided_at`, so its distinct values are the cycle record
  (`cycle_starts`).
- `collect_log`: one row per collector call, including the-odds-api credits remaining (budget guard).

All timestamps are stored as naive UTC `TIMESTAMP` and returned as tz-aware UTC pandas columns.
Every public writer requires tz-aware datetimes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from pitch_edge.paper.fixtures import (
    ambiguous_tokens,
    canonical_league,
    canonical_team,
    fixture_key,
    match_candidate,
    source_family,
)

logger = logging.getLogger(__name__)

SELECTIONS = ("home", "draw", "away")
DEFAULT_QUOTE_LOOKBACK = timedelta(days=7)
_EVENT_ID_COLUMNS = {"odds_api": "odds_api_event_id", "polymarket": "polymarket_event_id", "kalshi": "kalshi_event_id"}
_TIME_COLUMNS = (
    "kickoff_at",
    "captured_at",
    "decided_at",
    "settled_at",
    "called_at",
    "first_seen_at",
    "last_seen_at",
    "flagged_at",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS quotes (
    quote_id VARCHAR PRIMARY KEY, source VARCHAR NOT NULL, source_event_id VARCHAR NOT NULL,
    fixture_key VARCHAR NOT NULL, home VARCHAR, away VARCHAR, league VARCHAR, kickoff_at TIMESTAMP NOT NULL,
    market VARCHAR NOT NULL, selection VARCHAR NOT NULL, decimal_odds DOUBLE, bid DOUBLE, ask DOUBLE,
    last DOUBLE, size DOUBLE, instrument_id VARCHAR, captured_at TIMESTAMP NOT NULL, raw_json VARCHAR
);
CREATE TABLE IF NOT EXISTS fixtures (
    fixture_key VARCHAR PRIMARY KEY, league VARCHAR, kickoff_at TIMESTAMP, home VARCHAR, away VARCHAR,
    home_key VARCHAR, away_key VARCHAR, odds_api_event_id VARCHAR, polymarket_event_id VARCHAR,
    kalshi_event_id VARCHAR, first_seen_at TIMESTAMP, last_seen_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS decisions (
    decision_id VARCHAR PRIMARY KEY, strategy_id VARCHAR NOT NULL, hypothesis_id VARCHAR NOT NULL,
    decided_at TIMESTAMP NOT NULL, fixture_key VARCHAR NOT NULL, kickoff_at TIMESTAMP NOT NULL,
    venue VARCHAR NOT NULL, selection VARCHAR NOT NULL, price_taken DOUBLE NOT NULL,
    paper_stake_units DOUBLE NOT NULL, model_prob DOUBLE, inputs_quote_ids VARCHAR[] NOT NULL,
    code_version VARCHAR NOT NULL, params_json VARCHAR
);
CREATE TABLE IF NOT EXISTS settlements (
    decision_id VARCHAR PRIMARY KEY, closing_price DOUBLE, fair_close_odds DOUBLE, closing_source VARCHAR,
    result VARCHAR, clv_log DOUBLE, pnl DOUBLE, sharp_fair_close_odds DOUBLE, clv_vs_sharp_log DOUBLE,
    settled_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS provisional_settlements (
    decision_id VARCHAR PRIMARY KEY, closing_price DOUBLE, fair_close_odds DOUBLE, closing_source VARCHAR,
    result VARCHAR, clv_log DOUBLE, pnl DOUBLE, sharp_fair_close_odds DOUBLE, clv_vs_sharp_log DOUBLE,
    settled_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS decision_flags (
    decision_id VARCHAR NOT NULL, flag VARCHAR NOT NULL, reason VARCHAR, flagged_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS decide_log (
    strategy_id VARCHAR NOT NULL, decided_at TIMESTAMP NOT NULL, n_fixtures_evaluated INTEGER,
    n_fixtures_qualifying INTEGER, n_new_decisions INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS collect_log (
    source VARCHAR NOT NULL, scope VARCHAR, called_at TIMESTAMP NOT NULL, status VARCHAR NOT NULL,
    remaining INTEGER, used INTEGER, n_quotes INTEGER, note VARCHAR
);
"""


class DecisionError(RuntimeError):
    """A decision or settlement violates the ledger's rules (duplicate, unknown or future inputs)."""


class KickoffGuardError(DecisionError):
    """A decision was attempted for a fixture whose kickoff is at or before the decision time."""


def _naive_utc(dt: datetime) -> datetime:
    """Tz-aware datetime -> naive UTC (storage form). Raises ValueError for naive input."""
    if dt.tzinfo is None:
        raise ValueError(f"timestamp must be timezone-aware, got naive {dt!r}")
    return dt.astimezone(UTC).replace(tzinfo=None)


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Quote:
    """One price for one selection of one event at one capture time."""

    source: str
    source_event_id: str
    home: str
    away: str
    league: str  # canonical league code (see paper/fixtures.py)
    kickoff_at: datetime
    selection: str  # home | draw | away
    captured_at: datetime
    market: str = "1x2"
    decimal_odds: float | None = None
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    size: float | None = None
    instrument_id: str | None = None
    raw_json: str = "{}"

    @property
    def quote_id(self) -> str:
        key = f"{self.source}|{self.source_event_id}|{self.market}|{self.selection}|{_naive_utc(self.captured_at)}"
        return hashlib.sha1(key.encode()).hexdigest()[:20]


@dataclass(frozen=True)
class Decision:
    """A paper bet. `price_taken` is decimal odds net of fees; `inputs_quote_ids` are the quotes used."""

    strategy_id: str
    hypothesis_id: str
    decided_at: datetime
    fixture_key: str
    kickoff_at: datetime
    venue: str
    selection: str
    price_taken: float
    paper_stake_units: float
    model_prob: float | None
    inputs_quote_ids: list[str]
    code_version: str
    params: dict[str, Any] = field(default_factory=dict)
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    code_hash: str | None = None  # sha256 over src/pitch_edge/paper/**/*.py, see paper/provenance.py


@dataclass(frozen=True)
class Settlement:
    """Closing price, result and paper outcome for one decision."""

    decision_id: str
    closing_price: float | None
    fair_close_odds: float | None
    closing_source: str
    result: str | None
    clv_log: float | None
    pnl: float | None
    settled_at: datetime
    sharp_fair_close_odds: float | None = None
    clv_vs_sharp_log: float | None = None


def _with_utc(df: pd.DataFrame) -> pd.DataFrame:
    for col in _TIME_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.tz_localize("UTC")
    return df


def _chain_hash(prev_hash: str, record: dict[str, Any]) -> str:
    return hashlib.sha256((prev_hash + json.dumps(record, sort_keys=True)).encode()).hexdigest()


def verify_decision_chain(path: str | Path) -> tuple[bool, int]:
    """Check the decisions JSONL hash chain: every line's `hash` = sha256(prev hash + record JSON) and its
    `prev_hash` is the previous line's hash. Returns (intact, number of lines)."""
    prev = ""
    lines = Path(path).read_text().splitlines() if Path(path).exists() else []
    for line in lines:
        entry = json.loads(line)
        if entry.get("prev_hash") != prev or entry.get("hash") != _chain_hash(prev, entry["decision"]):
            return False, len(lines)
        prev = entry["hash"]
    return True, len(lines)


class PaperStore:
    """Read/append access to the paper ledger file. Use as a context manager or call `close()`.

    Every decision is also appended to an append-only JSONL hash chain (`chain_path`, default
    `decisions.jsonl` next to the database); on a read-write open, decisions missing from it are exported.
    """

    def __init__(self, path: str | Path, read_only: bool = False, chain_path: str | Path | None = None):
        self.path = Path(path)
        self.read_only = read_only
        self.chain_path = Path(chain_path) if chain_path else self.path.parent / "decisions.jsonl"
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path), read_only=read_only)
        if not read_only:
            self.con.execute(_SCHEMA)
            self.con.execute("ALTER TABLE collect_log ADD COLUMN IF NOT EXISTS cost INTEGER")
            self.con.execute("ALTER TABLE decisions ADD COLUMN IF NOT EXISTS code_hash VARCHAR")
            self.con.execute("ALTER TABLE decide_log ADD COLUMN IF NOT EXISTS error VARCHAR")
            self._ensure_unique_decisions()
            self._export_chain()

    @classmethod
    def open_read_only(cls, path: str | Path, attempts: int = 20, wait_s: float = 3.0) -> PaperStore | None:
        """A read-only store, or None when the file does not exist. Retries while a writer holds the lock."""
        if not Path(path).exists():
            return None
        for i in range(attempts):
            try:
                return cls(path, read_only=True)
            except duckdb.IOException:
                if i == attempts - 1:
                    raise
                time.sleep(wait_s)
        return None

    def _ensure_unique_decisions(self) -> None:
        dupes = self.con.execute(
            "SELECT count(*) FROM (SELECT strategy_id, fixture_key FROM decisions GROUP BY 1, 2 HAVING count(*) > 1)"
        ).fetchone()
        if dupes and dupes[0]:
            logger.warning(
                "%d (strategy, fixture) pairs have several decisions; uniqueness enforced in code only", dupes[0]
            )
            return
        self.con.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS decisions_strategy_fixture ON decisions (strategy_id, fixture_key)"
        )

    def _chain_state(self) -> tuple[set[str], str]:
        ids: set[str] = set()
        prev = ""
        if self.chain_path.exists():
            for line in self.chain_path.read_text().splitlines():
                entry = json.loads(line)
                ids.add(entry["decision"]["decision_id"])
                prev = entry["hash"]
        return ids, prev

    def _export_chain(self) -> int:
        """Append every decision not yet in the JSONL chain, oldest first. Returns lines written."""
        ids, prev = self._chain_state()
        cur = self.con.execute("SELECT * FROM decisions ORDER BY decided_at, decision_id")
        cols = [c[0] for c in cur.description]
        written = 0
        with self.chain_path.open("a") as fh:
            for values in cur.fetchall():
                row = dict(zip(cols, values, strict=True))
                if row["decision_id"] in ids:
                    continue
                record = {
                    "decision_id": row["decision_id"],
                    "strategy_id": row["strategy_id"],
                    "hypothesis_id": row["hypothesis_id"],
                    "decided_at": row["decided_at"].isoformat() + "+00:00",
                    "fixture_key": row["fixture_key"],
                    "kickoff_at": row["kickoff_at"].isoformat() + "+00:00",
                    "venue": row["venue"],
                    "selection": row["selection"],
                    "price_taken": row["price_taken"],
                    "paper_stake_units": row["paper_stake_units"],
                    "model_prob": row["model_prob"],
                    "inputs_quote_ids": list(row["inputs_quote_ids"]),
                    "code_version": row["code_version"],
                    "code_hash": row.get("code_hash"),
                    "params": json.loads(row["params_json"] or "{}"),
                }
                h = _chain_hash(prev, record)
                fh.write(json.dumps({"decision": record, "prev_hash": prev, "hash": h}) + "\n")
                prev = h
                written += 1
        return written

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> PaperStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def tables(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT table_name FROM information_schema.tables").fetchall()]

    def _df(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        return _with_utc(self.con.execute(sql, params or []).df())

    # ------------------------------------------------------------------ fixtures + quotes
    def _link_fixture(self, q: Quote) -> str:
        """Fixture key for a quote's event: by known source event id, exact key, or fuzzy match; else new."""
        family = source_family(q.source)
        id_col = _EVENT_ID_COLUMNS.get(family)
        now = _naive_utc(q.captured_at)
        if id_col:
            row = self.con.execute(
                f"SELECT fixture_key FROM fixtures WHERE {id_col} = ?", [q.source_event_id]
            ).fetchone()
            if row:
                self.con.execute(
                    "UPDATE fixtures SET last_seen_at = greatest(last_seen_at, ?) WHERE fixture_key = ?", [now, row[0]]
                )
                return str(row[0])
        key = fixture_key(q.league, q.kickoff_at, q.home, q.away)
        home_key, away_key = canonical_team(q.home), canonical_team(q.away)
        ko = _naive_utc(q.kickoff_at)
        found = self.con.execute("SELECT fixture_key FROM fixtures WHERE fixture_key = ?", [key]).fetchone()
        if not found:
            window = timedelta(hours=3)
            candidates = self.con.execute(
                "SELECT fixture_key, league, kickoff_at, home_key, away_key FROM fixtures "
                "WHERE league = ? AND kickoff_at BETWEEN ? AND ?",
                [q.league, ko - window, ko + window],
            ).fetchall()
            league_clubs = self.con.execute(
                "SELECT home_key FROM fixtures WHERE league = ? UNION SELECT away_key FROM fixtures WHERE league = ?",
                [q.league, q.league],
            ).fetchall()
            ambiguous = ambiguous_tokens({r[0] for r in league_clubs} | {home_key, away_key})
            matched = match_candidate(q.league, ko, home_key, away_key, candidates, ambiguous=ambiguous)
            if matched is None:
                n_pass = sum(
                    match_candidate(q.league, ko, home_key, away_key, [c], ambiguous=ambiguous) is not None
                    for c in candidates
                )
                if n_pass > 1:
                    logger.warning("%s event %s matches %d fixtures; not linked", q.source, q.source_event_id, n_pass)
            else:
                found = (matched,)
        if not found:
            self.con.execute(
                "INSERT INTO fixtures (fixture_key, league, kickoff_at, home, away, home_key, away_key, "
                "first_seen_at, last_seen_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [key, q.league, ko, q.home, q.away, home_key, away_key, now, now],
            )
            found = (key,)
        fk = str(found[0])
        if id_col:
            self.con.execute(
                f"UPDATE fixtures SET {id_col} = coalesce({id_col}, ?), last_seen_at = greatest(last_seen_at, ?) "
                "WHERE fixture_key = ?",
                [q.source_event_id, now, fk],
            )
        return fk

    def add_quotes(self, quotes: list[Quote]) -> int:
        """Store quotes (idempotent on quote id), linking each event to a canonical fixture. Returns rows added."""
        if not quotes:
            return 0
        added = 0
        event_keys: dict[tuple[str, str], str] = {}
        for q in quotes:
            if q.selection not in SELECTIONS:
                raise ValueError(f"selection must be one of {SELECTIONS}, got {q.selection!r}")
            ev = (q.source, q.source_event_id)
            if ev not in event_keys:
                event_keys[ev] = self._link_fixture(q)
            before = self.con.execute("SELECT count(*) FROM quotes WHERE quote_id = ?", [q.quote_id]).fetchone()
            if before and before[0]:
                continue
            self.con.execute(
                "INSERT INTO quotes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    q.quote_id,
                    q.source,
                    q.source_event_id,
                    event_keys[ev],
                    q.home,
                    q.away,
                    q.league,
                    _naive_utc(q.kickoff_at),
                    q.market,
                    q.selection,
                    q.decimal_odds,
                    q.bid,
                    q.ask,
                    q.last,
                    q.size,
                    q.instrument_id,
                    _naive_utc(q.captured_at),
                    q.raw_json,
                ],
            )
            added += 1
        return added

    def quotes(self) -> pd.DataFrame:
        return self._df("SELECT * FROM quotes ORDER BY captured_at, source, source_event_id, selection")

    def fixtures(self) -> pd.DataFrame:
        return self._df("SELECT * FROM fixtures ORDER BY kickoff_at, fixture_key")

    def latest_quotes(self, as_of: datetime, lookback: timedelta = DEFAULT_QUOTE_LOOKBACK) -> pd.DataFrame:
        """Latest quote per (source, fixture, selection) captured in (`as_of` - `lookback`, `as_of`], for
        fixtures that have not kicked off by `as_of`."""
        t = _naive_utc(as_of)
        return self._df(
            "SELECT * FROM quotes WHERE captured_at <= ? AND captured_at > ? AND kickoff_at > ? "
            "QUALIFY row_number() OVER (PARTITION BY source, fixture_key, market, selection "
            "ORDER BY captured_at DESC) = 1 ORDER BY fixture_key, source, selection",
            [t, t - lookback, t],
        )

    def known_kickoffs(self, source: str) -> dict[str, datetime]:
        """Most recently recorded kickoff (tz-aware UTC) per source event id for `source`."""
        rows = self.con.execute(
            "SELECT source_event_id, arg_max(kickoff_at, captured_at) FROM quotes WHERE source = ? "
            "GROUP BY source_event_id",
            [source],
        ).fetchall()
        return {str(eid): ko.replace(tzinfo=UTC) for eid, ko in rows}

    def odds_api_history(self, since: datetime, until: datetime) -> pd.DataFrame:
        """Every the-odds-api quote captured in (`since`, `until`] for fixtures not kicked off by `until`."""
        return self._df(
            "SELECT * FROM quotes WHERE source LIKE 'odds_api:%' AND captured_at > ? AND captured_at <= ? "
            "AND kickoff_at > ? ORDER BY fixture_key, captured_at, source, selection",
            [_naive_utc(since), _naive_utc(until), _naive_utc(until)],
        )

    def quotes_for_fixture(self, fixture: str, before: datetime) -> pd.DataFrame:
        """Every quote for `fixture` captured strictly before `before`."""
        return self._df(
            "SELECT * FROM quotes WHERE fixture_key = ? AND captured_at < ? ORDER BY captured_at",
            [fixture, _naive_utc(before)],
        )

    # ------------------------------------------------------------------ decisions
    def append_decision(self, d: Decision) -> None:
        """Append one immutable decision after checking the kickoff guard and point-in-time inputs.

        Raises KickoffGuardError if the fixture kicked off at or before `decided_at`; DecisionError if the
        id exists, an input quote is unknown, belongs to another fixture, or was captured after
        `decided_at`; ValueError for naive timestamps or an unknown selection.
        """
        decided, kickoff = _naive_utc(d.decided_at), _naive_utc(d.kickoff_at)
        if d.selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}, got {d.selection!r}")
        if kickoff <= decided:
            raise KickoffGuardError(f"fixture {d.fixture_key} kicked off at {d.kickoff_at}, not after {d.decided_at}")
        if not d.inputs_quote_ids:
            raise DecisionError("a decision must reference the quotes it was based on")
        rows = self.con.execute(
            "SELECT quote_id, fixture_key, captured_at, kickoff_at FROM quotes WHERE list_contains(?, quote_id)",
            [list(d.inputs_quote_ids)],
        ).fetchall()
        known = {r[0] for r in rows}
        missing = set(d.inputs_quote_ids) - known
        if missing:
            raise DecisionError(f"unknown input quotes: {sorted(missing)}")
        for qid, fk, captured, q_kickoff in rows:
            if fk != d.fixture_key:
                raise DecisionError(f"input quote {qid} belongs to fixture {fk}, not {d.fixture_key}")
            if captured > decided:
                raise DecisionError(f"input quote {qid} was captured at {captured}, after the decision {decided}")
            if q_kickoff <= decided:
                raise KickoffGuardError(f"input quote {qid} lists kickoff {q_kickoff}, not after {decided}")
        exists = self.con.execute("SELECT count(*) FROM decisions WHERE decision_id = ?", [d.decision_id]).fetchone()
        if exists and exists[0]:
            raise DecisionError(f"decision {d.decision_id} already exists; decisions are immutable")
        same = self.con.execute(
            "SELECT count(*) FROM decisions WHERE strategy_id = ? AND fixture_key = ?", [d.strategy_id, d.fixture_key]
        ).fetchone()
        if same and same[0]:
            raise DecisionError(f"{d.strategy_id} already has a decision on {d.fixture_key}; one per fixture")
        self.con.execute(
            "INSERT INTO decisions (decision_id, strategy_id, hypothesis_id, decided_at, fixture_key, kickoff_at, "
            "venue, selection, price_taken, paper_stake_units, model_prob, inputs_quote_ids, code_version, "
            "params_json, code_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                d.decision_id,
                d.strategy_id,
                d.hypothesis_id,
                decided,
                d.fixture_key,
                kickoff,
                d.venue,
                d.selection,
                float(d.price_taken),
                float(d.paper_stake_units),
                d.model_prob,
                list(d.inputs_quote_ids),
                d.code_version,
                json.dumps(d.params, sort_keys=True),
                d.code_hash,
            ],
        )
        self._export_chain()

    def decisions(self) -> pd.DataFrame:
        return self._df("SELECT * FROM decisions ORDER BY decided_at, decision_id")

    def decided_fixtures(self, strategy_id: str) -> set[str]:
        rows = self.con.execute("SELECT DISTINCT fixture_key FROM decisions WHERE strategy_id = ?", [strategy_id])
        return {r[0] for r in rows.fetchall()}

    # ------------------------------------------------------------------ settlements
    def append_settlement(self, s: Settlement) -> None:
        """Write the one final settlement of a decision (lake-sourced close). Raises DecisionError if the
        decision is unknown or already finally settled."""
        self._append_settlement_row("settlements", s)

    def append_provisional_settlement(self, s: Settlement) -> None:
        """Write the one provisional settlement of a decision (no lake close yet: our own snapshot or none).
        A later final settlement supersedes it in reporting; the provisional row itself is never changed."""
        self._append_settlement_row("provisional_settlements", s)

    def _append_settlement_row(self, table: str, s: Settlement) -> None:
        known = self.con.execute("SELECT count(*) FROM decisions WHERE decision_id = ?", [s.decision_id]).fetchone()
        if not known or not known[0]:
            raise DecisionError(f"cannot settle unknown decision {s.decision_id}")
        done = self.con.execute(f"SELECT count(*) FROM {table} WHERE decision_id = ?", [s.decision_id]).fetchone()
        if done and done[0]:
            raise DecisionError(f"decision {s.decision_id} already has a row in {table}")
        self.con.execute(
            f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                s.decision_id,
                s.closing_price,
                s.fair_close_odds,
                s.closing_source,
                s.result,
                s.clv_log,
                s.pnl,
                s.sharp_fair_close_odds,
                s.clv_vs_sharp_log,
                _naive_utc(s.settled_at),
            ],
        )

    def settlements(self) -> pd.DataFrame:
        return self._df("SELECT * FROM settlements ORDER BY settled_at, decision_id")

    def provisional_settlements(self) -> pd.DataFrame:
        return self._df("SELECT * FROM provisional_settlements ORDER BY settled_at, decision_id")

    # ------------------------------------------------------------------ flags
    def flag_decision(self, decision_id: str, flag: str, reason: str, *, flagged_at: datetime) -> None:
        """Append an annotation to a decision (e.g. `void` for a postponed match or a false fixture merge).
        The decision itself is never changed. Raises DecisionError for an unknown decision."""
        known = self.con.execute("SELECT count(*) FROM decisions WHERE decision_id = ?", [decision_id]).fetchone()
        if not known or not known[0]:
            raise DecisionError(f"cannot flag unknown decision {decision_id}")
        self.con.execute(
            "INSERT INTO decision_flags VALUES (?, ?, ?, ?)", [decision_id, flag, reason, _naive_utc(flagged_at)]
        )

    def decision_flags(self) -> pd.DataFrame:
        return self._df("SELECT * FROM decision_flags ORDER BY flagged_at, decision_id")

    def voided_decision_ids(self) -> set[str]:
        rows = self.con.execute("SELECT DISTINCT decision_id FROM decision_flags WHERE flag = 'void'").fetchall()
        return {r[0] for r in rows}

    def unsettled_decisions(self) -> pd.DataFrame:
        return self._df(
            "SELECT d.* FROM decisions d ANTI JOIN settlements s USING (decision_id) "
            "WHERE d.decision_id NOT IN (SELECT decision_id FROM decision_flags WHERE flag = 'void') "
            "ORDER BY d.kickoff_at"
        )

    # ------------------------------------------------------------------ collector bookkeeping
    def record_call(
        self,
        source: str,
        scope: str | None,
        *,
        status: str,
        called_at: datetime,
        remaining: int | None = None,
        used: int | None = None,
        n_quotes: int | None = None,
        note: str | None = None,
        cost: int | None = None,
    ) -> None:
        """Log one collector call (status `ok`, `error`, `refused_budget`, `skipped_cadence`, ...).
        `cost` is the credits that call consumed, when the source reports it."""
        self.con.execute(
            "INSERT INTO collect_log (source, scope, called_at, status, remaining, used, n_quotes, note, cost) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [source, scope, _naive_utc(called_at), status, remaining, used, n_quotes, note, cost],
        )

    def last_used(self, source: str) -> int | None:
        """Most recently reported cumulative credits used for `source`, or None."""
        row = self.con.execute(
            "SELECT used FROM collect_log WHERE source = ? AND used IS NOT NULL ORDER BY called_at DESC LIMIT 1",
            [source],
        ).fetchone()
        return None if row is None else int(row[0])

    def cost_per_call(self, source: str, default: int = 2) -> int:
        """Credits the most recent measured successful call of `source` cost, else `default`."""
        row = self.con.execute(
            "SELECT cost FROM collect_log WHERE source = ? AND status = 'ok' AND cost IS NOT NULL "
            "ORDER BY called_at DESC LIMIT 1",
            [source],
        ).fetchone()
        return default if row is None else int(row[0])

    def budget(self, source: str) -> int | None:
        """Most recently reported remaining credits for `source`, or None if never reported."""
        row = self.con.execute(
            "SELECT remaining FROM collect_log WHERE source = ? AND remaining IS NOT NULL "
            "ORDER BY called_at DESC LIMIT 1",
            [source],
        ).fetchone()
        return None if row is None else int(row[0])

    def last_ok_call(self, source: str, scope: str) -> datetime | None:
        """When `source` last fetched `scope` successfully (`ok`, or `parse_error`: the call was made and
        paid for) as tz-aware UTC, or None."""
        row = self.con.execute(
            "SELECT max(called_at) FROM collect_log WHERE source = ? AND scope = ? AND status IN ('ok', 'parse_error')",
            [source, scope],
        ).fetchone()
        return None if row is None or row[0] is None else row[0].replace(tzinfo=UTC)

    def record_decide_run(
        self,
        strategy_id: str,
        decided_at: datetime,
        *,
        n_fixtures_evaluated: int | None,
        n_fixtures_qualifying: int | None,
        n_new_decisions: int,
        error: str | None = None,
    ) -> None:
        """Log one strategy run: fixtures it could evaluate, fixtures with a qualifying bet, decisions added."""
        self.con.execute(
            "INSERT INTO decide_log (strategy_id, decided_at, n_fixtures_evaluated, n_fixtures_qualifying, "
            "n_new_decisions, error) VALUES (?, ?, ?, ?, ?, ?)",
            [strategy_id, _naive_utc(decided_at), n_fixtures_evaluated, n_fixtures_qualifying, n_new_decisions, error],
        )

    def decide_log(self) -> pd.DataFrame:
        return self._df("SELECT * FROM decide_log ORDER BY decided_at, strategy_id")

    def cycle_starts(self) -> list[datetime]:
        """Time of every loop cycle that ran the strategies, oldest first (tz-aware UTC): the distinct
        `decided_at` values of `decide_log`."""
        rows = self.con.execute("SELECT DISTINCT decided_at FROM decide_log ORDER BY decided_at").fetchall()
        return [r[0].replace(tzinfo=UTC) for r in rows]

    def consecutive_errors(self, source: str, scope: str) -> tuple[int, datetime | None]:
        """Number of `error` calls for (`source`, `scope`) since its last fetched call, and when the latest
        was (tz-aware UTC). Skipped calls do not interrupt the run of errors."""
        rows = self.con.execute(
            "SELECT status, called_at FROM collect_log WHERE source = ? AND scope = ? "
            "AND status IN ('ok', 'parse_error', 'error') ORDER BY called_at DESC",
            [source, scope],
        ).fetchall()
        n = 0
        for status, _ in rows:
            if status != "error":
                break
            n += 1
        return n, (rows[0][1].replace(tzinfo=UTC) if n else None)

    def collect_log(self) -> pd.DataFrame:
        return self._df("SELECT * FROM collect_log ORDER BY called_at")


__all__ = [
    "Decision",
    "DecisionError",
    "KickoffGuardError",
    "PaperStore",
    "Quote",
    "SELECTIONS",
    "Settlement",
    "canonical_league",
    "utc_now",
]
