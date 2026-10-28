"""`pitch-edge paper ...`: collect quotes, log paper decisions, settle them, report CLV.

Paper only: every command reads public market data or the ledger file; none places an order.
The ledger lives at `$PITCH_EDGE_PAPER_DB` (default `<data_dir>/paper/paper.duckdb`). Collectors fetch
before opening the ledger for writing, and `status`/`report` open it read-only, so a long network sweep
never blocks a reader.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

from pitch_edge.config import get_settings
from pitch_edge.data.http import CachedHttpClient
from pitch_edge.paper.store import PaperStore, utc_now

paper_app = typer.Typer(
    help="Forward paper-trading ledger (read-only market data; never places an order).",
    pretty_exceptions_show_locals=False,  # a traceback must never print local variables such as an API key
)

ALL_SOURCES = ("odds_api", "polymarket", "kalshi")


def paper_db_path() -> Path:
    """Ledger file: `$PITCH_EDGE_PAPER_DB`, else `<data_dir>/paper/paper.duckdb`."""
    env = os.environ.get("PITCH_EDGE_PAPER_DB")
    return Path(env) if env else get_settings().data_dir / "paper" / "paper.duckdb"


def _http(min_interval_s: float) -> CachedHttpClient:
    return CachedHttpClient(get_settings().data_dir / "raw" / "paper_http", min_interval_s=min_interval_s)


def _split(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


@paper_app.command()
def collect(
    sources: str = typer.Option(",".join(ALL_SOURCES), help="Comma-separated: odds_api, polymarket, kalshi"),
    leagues: str = typer.Option("", help="the-odds-api sport keys (default: the 7 configured leagues)"),
    budget_floor: int = typer.Option(60, help="Never call the-odds-api while credits remaining < this"),
    min_hours: float = typer.Option(23.5, help="Call each the-odds-api league at most once per this many hours"),
    max_monthly_credits: int = typer.Option(
        450, help="Trim the league list so projected monthly the-odds-api spend fits"
    ),
    bookmakers: str = typer.Option("", help="the-odds-api bookmaker filter (default: every book in regions uk,eu)"),
    force: bool = typer.Option(False, help="Ignore the the-odds-api cadence window (the budget floor still applies)"),
) -> None:
    """Fetch current quotes from the chosen sources into the ledger."""
    from pitch_edge.paper.collectors import KalshiCollector, OddsApiCollector, PolymarketCollector

    chosen = _split(sources)
    unknown = set(chosen) - set(ALL_SOURCES)
    if unknown:
        raise typer.BadParameter(f"unknown source(s): {sorted(unknown)}")
    failed = False
    db = paper_db_path()
    for src in chosen:
        try:
            if src == "odds_api":
                key = os.environ.get("ODDS_API_KEY", "")
                if not key:
                    rprint("[red]odds_api: ODDS_API_KEY is not set; skipped[/red]")
                    failed = True
                    continue
                collector = OddsApiCollector(
                    _http(1.0),
                    key,
                    db,  # opened only around each write
                    leagues=_split(leagues) or None,
                    budget_floor=budget_floor,
                    min_hours_between_calls=min_hours,
                    max_monthly_credits=max_monthly_credits,
                    bookmakers=_split(bookmakers) or None,
                )
                quotes = collector.collect(force=force)  # stores each league's quotes itself
                with PaperStore(db) as store:
                    store.record_call(src, "collect", status="ok", called_at=utc_now(), n_quotes=len(quotes))
                    note = f", credits remaining {store.budget('odds_api')}"
                rprint(f"{src}: {len(quotes)} quotes{note}")
                continue
            if src == "polymarket":
                quotes = PolymarketCollector(_http(0.2)).collect()
            else:
                quotes = KalshiCollector(_http(0.2), db).collect()
            with PaperStore(db) as store:
                added = store.add_quotes(quotes)
                store.record_call(src, "collect", status="ok", called_at=utc_now(), n_quotes=added)
            rprint(f"{src}: {len(quotes)} quotes ({added} new)")
        except Exception as exc:  # one failing source must not stop the others
            failed = True
            with PaperStore(db) as store:
                store.record_call(src, "collect", status="error", called_at=utc_now(), note=str(exc)[:500])
            rprint(f"[red]{src}: failed: {exc}[/red]")
    if failed:
        raise typer.Exit(1)


@paper_app.command()
def decide(
    strategy: list[str] = typer.Option(None, "--strategy", help="Strategy id (repeatable; default: all)"),  # noqa: B008
) -> None:
    """Run strategies on the latest quotes and append their paper decisions (immutable)."""
    from pitch_edge.paper.strategies import available_strategies, run_decide

    registry = available_strategies()
    ids = list(strategy or []) or list(registry)
    unknown = set(ids) - set(registry)
    if unknown:
        raise typer.BadParameter(f"unknown strategy: {sorted(unknown)}; available: {sorted(registry)}")
    from pitch_edge.paper.provenance import PAPER_ROOT, git_version, snapshot_code

    db = paper_db_path()
    code_hash = snapshot_code(PAPER_ROOT, db.parent / "code")  # archive this exact code once per hash
    with PaperStore(db) as store:
        added = run_decide(
            store, [registry[i]() for i in ids], now=utc_now(), code_version=git_version(), code_hash=code_hash
        )
    for sid, n in added.items():
        rprint(f"{sid}: {n} new decisions")


@paper_app.command()
def settle(
    grace_days: float = typer.Option(4.0, help="Days after kickoff before falling back to our own snapshot"),
) -> None:
    """Settle finished decisions against the lake's closing prices (Betfair Exchange / Polymarket bars)."""
    from datetime import timedelta

    from pitch_edge.lake.client import Lake
    from pitch_edge.paper.settle import LakeCloseSource
    from pitch_edge.paper.settle import settle as run_settle

    with PaperStore(paper_db_path()) as store, Lake() as lake:
        counts = run_settle(store, LakeCloseSource(lake), now=utc_now(), grace=timedelta(days=grace_days))
        n_prov = len(store.provisional_settlements())
        n_void = len(store.voided_decision_ids())
    rprint(
        f"final settlements written {counts['settled']}, waiting for a lake close/result {counts['waiting']}; "
        f"provisional rows in total {n_prov}, void decisions {n_void}"
    )


@paper_app.command()
def report(n_boot: int = 2000) -> None:
    """Per strategy and segment (primary / exploratory / unregistered): decisions, settlements, mean CLV on
    lake closes with a match-day block-bootstrap CI, ROI, evaluation-window coverage (share of hours with a
    loop cycle), and opportunity frequency. Read-only."""
    from pitch_edge.paper.report import report_notes, summarise, v2_summary

    store = PaperStore.open_read_only(paper_db_path())
    if store is None:
        rprint("No decisions in the paper ledger yet (no ledger file).")
        return
    with store:
        rep = summarise(store, n_boot=n_boot)
        v2 = v2_summary(store, n_boot=n_boot)
        opp = store.con.execute(
            "SELECT strategy_id, count(*), avg(n_fixtures_evaluated), avg(n_fixtures_qualifying), "
            "max(n_fixtures_qualifying) FROM decide_log WHERE n_fixtures_evaluated IS NOT NULL GROUP BY 1 ORDER BY 1"
        ).fetchall()
    for sid, runs, ev, qual, qmax in opp:
        rprint(f"{sid}: {runs} snapshots, mean fixtures evaluated {ev:.1f}, mean qualifying {qual:.2f} (max {qmax})")
    if rep.empty:
        rprint("No decisions in the paper ledger yet.")
        return
    t = Table(title="Paper ledger")
    for c in rep.columns:
        if c != "registration_cut":
            t.add_column(c)
    for row in rep.drop(columns=["registration_cut"]).itertuples(index=False):
        t.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])
    rprint(t)
    for note in report_notes(rep):
        rprint(f"- {note}")
    if not v2["arms"].empty:
        t2 = Table(title="soft-vs-sharp-live-v2 (primary segment, final lake closes only)")
        for c in v2["arms"].columns:
            t2.add_column(c)
        for row in v2["arms"].itertuples(index=False):
            t2.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])
        rprint(t2)
        p = v2["paired"]
        rprint(
            f"paired model - pinnacle CLV per fixture (no bet = 0): n={p['n_fixtures']} "
            f"(pending {p['n_pending']}), mean {p['mean_diff']:.4f}, 95% match-day block-bootstrap CI "
            f"[{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]; {v2['cut']}"
        )
    elif v2["cut"]:
        rprint(f"soft-vs-sharp-live-v2: no primary-segment decisions yet ({v2['cut']})")


@paper_app.command()
def pause(
    since: str = typer.Option(
        ..., "--since", help="When the loop was stopped, ISO 8601 UTC (e.g. 2026-10-28T07:48:17Z)"
    ),
    reason: str = typer.Option(..., "--reason", help="Why the loop was stopped"),
) -> None:
    """Record that the loop was stopped on purpose from --since until its next cycle, so `paper status` counts
    those cycles as paused rather than missed. Recording the same start twice adds nothing."""
    import pandas as pd

    start = pd.Timestamp(since)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    with PaperStore(paper_db_path()) as store:
        added = store.record_pause(start.to_pydatetime(), reason)
    rprint(f"pause from {start:%Y-%m-%d %H:%M:%S} UTC {'recorded' if added else 'already recorded'}")


@paper_app.command()
def status() -> None:
    """Row counts, quotes per source, the-odds-api credits remaining, and loop cycles observed vs expected
    with the longest gaps. Read-only."""
    from pitch_edge.paper.report import CYCLE_INTERVAL, cycle_gaps, open_pause

    store = PaperStore.open_read_only(paper_db_path())
    if store is None:
        rprint(f"No paper ledger yet (no quotes collected) at {paper_db_path()}")
        return
    with store:
        con = store.con
        t = Table(title=f"Paper ledger {paper_db_path()}")
        t.add_column("item")
        t.add_column("value")
        for table in ("quotes", "fixtures", "decisions", "settlements", "provisional_settlements", "decision_flags"):
            n = con.execute(f"SELECT count(*) FROM {table}").fetchone()
            t.add_row(table, str(n[0] if n else 0))
        for src, n, fx, last in con.execute(
            "SELECT source, count(*), count(DISTINCT fixture_key), max(captured_at) FROM quotes "
            "GROUP BY source ORDER BY source"
        ).fetchall():
            t.add_row(f"quotes[{src}]", f"{n} rows, {fx} fixtures, last {last} UTC")
        t.add_row("odds_api credits remaining", str(store.budget("odds_api")))
        starts = store.cycle_starts()
        pauses = store.pause_starts()
        gaps = cycle_gaps(starts, CYCLE_INTERVAL, pauses)
        missed = [g for g in gaps if not g.paused]
        if starts:
            expected = round((starts[-1] - starts[0]) / CYCLE_INTERVAL) + 1
            t.add_row(
                "cycles observed vs expected",
                f"{len(starts)} of {expected} since {starts[0]:%Y-%m-%d %H:%M} UTC (last {starts[-1]:%Y-%m-%d %H:%M})",
            )
            t.add_row("cycle gaps", f"{len(missed)} gap(s), {sum(g.missed for g in missed)} missed cycles")
            paused = [g for g in gaps if g.paused]
            if paused:
                t.add_row("paused", f"{len(paused)} pause(s), {sum(g.missed for g in paused)} paused cycles")
        current = open_pause(starts, pauses)
        if current is not None:
            t.add_row("loop", f"paused since {current:%Y-%m-%d %H:%M} UTC (no cycle since)")
        rprint(t)
        for g in sorted(missed, key=lambda g: g.missed, reverse=True)[:5]:
            rprint(f"gap {g.start:%Y-%m-%d %H:%M} -> {g.end:%Y-%m-%d %H:%M} UTC: {g.missed} missed cycles")
        for g in (g for g in gaps if g.paused):
            rprint(f"pause {g.start:%Y-%m-%d %H:%M} -> {g.end:%Y-%m-%d %H:%M} UTC: {g.missed} paused cycles")
