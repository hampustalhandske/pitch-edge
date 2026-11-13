"""Lake freeze state: an append-only event log that tells every lake reader in this repo to stay away.

While the lake is frozen (its owner is rebuilding or migrating it), nothing here may read it: `Lake()` refuses
with `LakeFrozenError`, `pitch-edge lake-sync` prints a skip and exits 0, and `pitch-edge control gate` exits
`FROZEN_EXIT` so scheduled scripts stop before starting a sync.

The log is `<data_dir>/control/lake_freeze.jsonl` (`data_dir` from `pitch_edge.config.get_settings()`), one JSON
object per line:

    {"event": "freeze" | "thaw", "at": "<ISO 8601 UTC>", "reason": "<text>", "recorded_at": "<ISO 8601 UTC>"}

Lines are only ever appended. A freeze is active from its `at` until the first thaw strictly after it; a freeze
recorded while another is active adds nothing, and a thaw with no active freeze is refused (the same rule as the
paper ledger's pauses and resumes, `pitch_edge.paper.store.pause_intervals`). A missing log means no freeze was
ever recorded. A log that cannot be read or parsed raises `FreezeLogError`, and every caller treats that as
frozen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pitch_edge.config import get_settings

FROZEN_EXIT = 10  # `pitch-edge control gate` exit code while frozen (or while the log is unreadable)
EVENTS = ("freeze", "thaw")


class FreezeLogError(RuntimeError):
    """The freeze log exists but cannot be read or parsed; callers must treat the lake as frozen."""


class FreezeStateError(ValueError):
    """A freeze or thaw that the current state does not allow (a thaw with no active freeze, a second freeze)."""


@dataclass(frozen=True)
class FreezeEvent:
    """One line of the freeze log. `at` is when the freeze or thaw takes effect, `recorded_at` when it was
    written; both tz-aware UTC."""

    event: str
    at: datetime
    reason: str
    recorded_at: datetime

    def to_json(self) -> str:
        return json.dumps(
            {
                "event": self.event,
                "at": self.at.isoformat(),
                "reason": self.reason,
                "recorded_at": self.recorded_at.isoformat(),
            }
        )


def freeze_log_path() -> Path:
    """`<data_dir>/control/lake_freeze.jsonl`."""
    return get_settings().data_dir / "control" / "lake_freeze.jsonl"


def read_events(path: Path | None = None) -> list[FreezeEvent]:
    """Every event in the log, in file order; empty when the log does not exist. Raises `FreezeLogError` when the
    file cannot be read or a non-blank line is not a valid event."""
    path = path or freeze_log_path()
    if not path.exists():
        return []
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise FreezeLogError(f"cannot read {path}: {exc}") from exc
    events = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(_parse_event(line))
        except (ValueError, TypeError, KeyError) as exc:
            raise FreezeLogError(f"{path} line {lineno} is not a valid freeze event: {exc}") from exc
    return events


def _parse_event(line: str) -> FreezeEvent:
    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise ValueError("not a JSON object")
    event = obj["event"]
    if event not in EVENTS:
        raise ValueError(f"unknown event {event!r}")
    reason = obj["reason"]
    if not isinstance(reason, str):
        raise ValueError("reason is not a string")
    return FreezeEvent(event, _parse_utc(obj["at"]), reason, _parse_utc(obj["recorded_at"]))


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"timestamp {value!r} is not a string")
    ts = datetime.fromisoformat(value)
    if ts.tzinfo is None:
        raise ValueError(f"timestamp {value!r} has no timezone")
    return ts.astimezone(UTC)


def freeze_intervals(events: list[FreezeEvent]) -> list[tuple[FreezeEvent, datetime | None]]:
    """The frozen intervals, oldest first, as (the freeze event that started it, end time or None while still
    active). A freeze is active from its `at` until the first thaw strictly after it; a thaw sorts before a freeze
    at the same time; thaws with no active freeze and freezes inside an active one add nothing."""
    ordered = sorted(events, key=lambda e: (e.at, e.event == "freeze"))
    out: list[tuple[FreezeEvent, datetime | None]] = []
    start: FreezeEvent | None = None
    for e in ordered:
        if e.event == "freeze" and start is None:
            start = e
        elif e.event == "thaw" and start is not None and e.at > start.at:
            out.append((start, e.at))
            start = None
    if start is not None:
        out.append((start, None))
    return out


def active_freeze_event(at: datetime | None = None, path: Path | None = None) -> FreezeEvent | None:
    """The freeze event whose interval contains `at` (tz-aware UTC, default now), or None when the lake may be
    read then. Raises `FreezeLogError` when the log is unreadable."""
    at = _require_utc(at or datetime.now(UTC), "at")
    for start, end in freeze_intervals(read_events(path)):
        if start.at <= at and (end is None or at < end):
            return start
    return None


def active_freeze(at: datetime | None = None, path: Path | None = None) -> datetime | None:
    """Start of the freeze active at `at` (tz-aware UTC, default now), or None. Raises `FreezeLogError` when the
    log is unreadable."""
    event = active_freeze_event(at, path)
    return event.at if event is not None else None


def record_freeze(at: datetime, reason: str, path: Path | None = None) -> FreezeEvent:
    """Append a freeze taking effect at `at` (tz-aware UTC). Raises `FreezeStateError` when a freeze is already
    active at `at`, `FreezeLogError` when the log is unreadable."""
    at = _require_utc(at, "at")
    active = active_freeze_event(at, path)
    if active is not None:
        raise FreezeStateError(f"a lake freeze is already active since {active.at:%Y-%m-%d %H:%M:%S} UTC")
    return _append(FreezeEvent("freeze", at, reason, datetime.now(UTC)), path)


def record_thaw(at: datetime, reason: str, path: Path | None = None) -> FreezeEvent:
    """Append a thaw ending the freeze active at `at` (tz-aware UTC). Raises `FreezeStateError` when no freeze is
    active at `at`, `FreezeLogError` when the log is unreadable."""
    at = _require_utc(at, "at")
    if active_freeze_event(at, path) is None:
        raise FreezeStateError(f"no lake freeze is active at {at:%Y-%m-%d %H:%M:%S} UTC")
    return _append(FreezeEvent("thaw", at, reason, datetime.now(UTC)), path)


def _append(event: FreezeEvent, path: Path | None) -> FreezeEvent:
    path = path or freeze_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(event.to_json() + "\n")
    return event


def _require_utc(ts: datetime, name: str) -> datetime:
    if ts.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware (UTC)")
    return ts.astimezone(UTC)
