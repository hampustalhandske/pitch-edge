"""`pitch-edge control ...`: record and read the lake freeze (see `pitch_edge.control`).

- `control freeze --reason R [--at T]` records a freeze from T (default now); refused (exit 1) while one is active.
- `control thaw --reason R [--at T]` ends the freeze active at T (default now); refused (exit 1) without one.
- `control status` prints whether the lake is frozen now, since when and why.
- `control gate` is for scheduled scripts: exit 0 when the lake may be read, `FROZEN_EXIT` (10) while frozen or
  while the freeze log is unreadable.

Output is plain, unwrapped text, one line per message, so scheduled scripts' logs keep each message whole.
"""

from __future__ import annotations

from datetime import UTC, datetime

import typer

from pitch_edge.control import (
    FROZEN_EXIT,
    FreezeLogError,
    FreezeStateError,
    active_freeze_event,
    freeze_log_path,
    record_freeze,
    record_thaw,
)

control_app = typer.Typer(help="Lake freeze: while it is active nothing in pitch-edge reads the lake.")


def freeze_notice(at: datetime | None = None) -> str | None:
    """One line describing why the lake must not be read at `at` (default now), or None when it may be:
    "lake freeze since <start> UTC (<reason>)", or a note that the freeze log is unreadable and so treated as
    frozen."""
    try:
        freeze = active_freeze_event(at)
    except FreezeLogError as exc:
        return f"the lake freeze log is unreadable, so the lake is treated as frozen ({exc})"
    if freeze is None:
        return None
    return f"lake freeze since {freeze.at:%Y-%m-%d %H:%M:%S} UTC ({freeze.reason})"


def _parse_at(value: str) -> datetime:
    """ISO 8601 time; a value without a zone is read as UTC; empty means now."""
    if not value:
        return datetime.now(UTC)
    try:
        ts = datetime.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"not an ISO 8601 time: {value!r}") from exc
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


@control_app.command()
def freeze(
    reason: str = typer.Option(..., "--reason", help="Why the lake must not be read (e.g. 'lake rebuild LK-111')"),
    at: str = typer.Option("", "--at", help="When the freeze starts, ISO 8601 UTC (default: now)"),
) -> None:
    """Record a lake freeze from --at. Until `control thaw`, `Lake()` refuses, `lake-sync` skips and `control gate`
    exits 10."""
    when = _parse_at(at)
    try:
        record_freeze(when, reason)
    except (FreezeStateError, FreezeLogError) as exc:
        typer.echo(f"freeze not recorded: {exc}")
        raise typer.Exit(1) from exc
    typer.echo(f"lake freeze from {when:%Y-%m-%d %H:%M:%S} UTC recorded ({reason}) in {freeze_log_path()}")


@control_app.command()
def thaw(
    reason: str = typer.Option(..., "--reason", help="Who ended the freeze and why"),
    at: str = typer.Option("", "--at", help="When the freeze ends, ISO 8601 UTC (default: now)"),
) -> None:
    """End the lake freeze active at --at. Refuses (exit 1) when no freeze is active then."""
    when = _parse_at(at)
    try:
        record_thaw(when, reason)
    except (FreezeStateError, FreezeLogError) as exc:
        typer.echo(f"thaw not recorded: {exc}")
        raise typer.Exit(1) from exc
    typer.echo(f"lake thaw at {when:%Y-%m-%d %H:%M:%S} UTC recorded ({reason})")


@control_app.command()
def status() -> None:
    """Print whether the lake is frozen now. Read-only."""
    notice = freeze_notice()
    typer.echo(notice if notice is not None else "lake not frozen")
    typer.echo(f"freeze log: {freeze_log_path()}")


@control_app.command()
def gate() -> None:
    """For scheduled scripts: exit 0 when the lake may be read now, `FROZEN_EXIT` (10) while a freeze is active
    or the freeze log is unreadable. Read-only."""
    notice = freeze_notice()
    if notice is not None:
        typer.echo(notice)
        raise typer.Exit(FROZEN_EXIT)
    typer.echo("lake not frozen")
