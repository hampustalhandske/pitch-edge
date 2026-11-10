"""`pitch-edge` command line: setup, ingest, features, backtest, ablation, ask, export, serve."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import pandas as pd
import typer
from rich import print as rprint
from rich.table import Table

from pitch_edge.config import get_settings
from pitch_edge.data.storage import Warehouse

app = typer.Typer(help="PITCH-EDGE — football intelligence & market-edge research. Answers questions, never bets.")


def _wh() -> Warehouse:
    settings = get_settings()
    settings.ensure_dirs()
    return Warehouse(settings.db_path)


@app.callback()
def _setup(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )


@app.command()
def ingest(
    no_statsbomb: bool = typer.Option(False, help="Skip StatsBomb event data"),
    no_weather: bool = typer.Option(False, help="Skip Wikidata venues + Open-Meteo weather"),
    statsbomb_max: int = typer.Option(40, help="Max matches with events per StatsBomb competition"),
) -> None:
    """Pull every free source into the local warehouse (idempotent, cached); football-data comes from the
    `lake_fd_matches` mirror, kept current by `pitch-edge lake-sync --only fd_matches`."""
    from pitch_edge.data.ingest import ingest_everything

    with _wh() as wh:
        report = ingest_everything(
            wh,
            include_statsbomb=not no_statsbomb,
            include_weather=not no_weather,
            statsbomb_max_matches=statsbomb_max,
        )
        rprint({k: v for k, v in report.items() if v})
        rprint(f"[bold]matches:[/bold] {wh.count('matches'):,}  [bold]odds rows:[/bold] {wh.count('odds'):,}")


@app.command("ingest-pmxt")
def ingest_pmxt(
    start: str = typer.Argument(..., help="First hour to backfill, YYYY-MM-DD or YYYY-MM-DDTHH"),
    end: str = typer.Argument(..., help="Last hour to backfill (inclusive), same format as START"),
    discover_only: bool = typer.Option(False, help="Only refresh dim_soccer_markets, skip the hourly backfill"),
) -> None:
    """Backfill the PMXT Polymarket orderbook archive for soccer markets over [START, END].

    Restartable: run under `nohup caffeinate -i` for a multi-year range (thousands of hourly
    files) — killing and re-running this command never re-downloads an hour already ingested
    (see `ingest_pmxt_orderbook_range`). Always run `--discover-only` first, or let this command
    do it (it always refreshes `dim_soccer_markets` before backfilling), since the hourly filter
    depends on it.
    """
    from pitch_edge.data.ingest import ingest_pmxt_orderbook_range, ingest_pmxt_soccer_markets

    with _wh() as wh:
        n_markets = ingest_pmxt_soccer_markets(wh)
        rprint(f"dim_soccer_markets: +{n_markets} new (total {wh.count('dim_soccer_markets'):,})")
        if discover_only:
            return
        report = ingest_pmxt_orderbook_range(wh, pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime())
        fetched = {h: n for h, n in report.items() if n}
        rprint(
            f"backfilled {len(fetched)}/{len(report)} hours with new rows; pmxt_orderbook total: {wh.count('pmxt_orderbook'):,}"
        )


@app.command("map-pmxt")
def map_pmxt() -> None:
    """Recompute pmxt_match_map (Polymarket condition_id -> our match_id/outcome_side) from
    whatever dim_soccer_markets/matches currently hold. Cheap, idempotent, no network calls —
    safe to re-run any time either table grows."""
    from pitch_edge.data.ingest import build_pmxt_match_map

    with _wh() as wh:
        build_pmxt_match_map(wh)
        rprint(f"pmxt_match_map: {wh.count('pmxt_match_map'):,} rows")


@app.command()
def features(min_date: str = "2015-07-01", leagues: str = "") -> None:
    """Build and persist the pre-match feature store."""
    from pitch_edge.pipeline import load_feature_frame, persist_features

    lg = [s.strip() for s in leagues.split(",") if s.strip()] or None
    with _wh() as wh:
        f = load_feature_frame(wh, leagues=lg, min_date=min_date)
        n = persist_features(wh, f)
        rprint(f"features: {n:,} rows, {f.shape[1]} columns")


@app.command()
def backtest(
    min_date: str = "2015-07-01",
    leagues: str = "",
    models: str = typer.Option(
        "dixon_coles,gbdt,stochastic_strength,transformer_sequence,sentiment_only",
        help="Comma-separated model names",
    ),
    edge: float = 0.03,
    retrain_days: int = 30,
    label: str = "main",
) -> None:
    """Walk-forward backtest with CLV / Kelly / calibration; writes reports/<label>/CASE_STUDY.md."""
    from pitch_edge.backtest.engine import WalkForwardConfig
    from pitch_edge.backtest.report import results_table
    from pitch_edge.models import available_models
    from pitch_edge.pipeline import load_feature_frame, run_backtests

    wanted = {s.strip() for s in models.split(",")}
    lg = [s.strip() for s in leagues.split(",") if s.strip()] or None
    with _wh() as wh:
        f = load_feature_frame(wh, leagues=lg, min_date=min_date)
        if f.empty:
            rprint("[red]no features — run `pitch-edge ingest` first[/red]")
            raise typer.Exit(1)
        chosen = [m for m in available_models() if m.name in wanted]
        results = run_backtests(
            f, chosen, WalkForwardConfig(edge_threshold=edge, retrain_every_days=retrain_days), wh=wh, label=label
        )
        _print_df(results_table(results).round(4), "Backtest summary")


@app.command("backtest-timing")
def backtest_timing(
    edge_threshold: float = 0.03,
    retrain_days: int = 30,
    label: str = "pmxt_timing",
) -> None:
    """Tick-timing backtest over real Polymarket order-book prices: does entering when the model
    and the live market disagree beat betting the first or last available price? Requires
    `pitch-edge map-pmxt` to have run first. Writes data/backtest/<label>/tick_timing_{bets,summary}.csv."""
    from pitch_edge.backtest.tick_timing import TickTimingConfig, run_tick_timing_backtest

    with _wh() as wh:
        result = run_tick_timing_backtest(
            wh, TickTimingConfig(edge_threshold=edge_threshold, retrain_every_days=retrain_days)
        )
        if result["summary"].empty:
            rprint("[red]no bets placed — check pmxt_match_map/pmxt_orderbook coverage[/red]")
            raise typer.Exit(1)
        out_dir = get_settings().backtest_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)
        result["bets"].to_csv(out_dir / "tick_timing_bets.csv", index=False)
        result["summary"].to_csv(out_dir / "tick_timing_summary.csv", index=False)
        _print_df(result["summary"].round(4), "Tick-timing backtest (ranked by Sharpe)")


def _ask_models() -> list:
    """Fast models only — the `ask` agent fits fresh per question, so a slow sequence model would
    make an interactive question take minutes instead of seconds. No market-odds model: there's no
    live-odds feed, so a historical market column would just be bias on the live `ask` path."""
    from pitch_edge.models import default_models

    return default_models()


@app.command()
def ask(
    question: str = typer.Argument(..., help="'top N bets', '<team> vs <team>', or 'what's on <day>'"),
    as_of: str | None = typer.Option(
        None, "--as-of", help="Answer as of this date (YYYY-MM-DD); default: most recent real closing-odds date"
    ),
    min_date: str = "2018-07-01",
) -> None:
    """The only Q&A entrypoint: standardized questions only, grounded in real backtest evidence
    and real early-quote market odds, answered as of an explicit point in time (there is no live
    odds feed today, so there is no genuine 'now' to answer as of instead)."""
    from pitch_edge.agents.qa_graph import ask_question, build_qa_graph
    from pitch_edge.backtest.as_of import default_as_of
    from pitch_edge.pipeline import build_rag_index, load_feature_frame

    with _wh() as wh:
        features = load_feature_frame(wh, min_date=min_date)
        if features.empty:
            rprint("[red]no features — run `pitch-edge ingest` and `pitch-edge features` first[/red]")
            raise typer.Exit(1)
        cutoff = pd.Timestamp(as_of) if as_of else default_as_of(wh)
        slice_path = get_settings().backtest_dir / "main" / "slice_evidence.csv"
        slice_evidence = pd.read_csv(slice_path) if slice_path.exists() else pd.DataFrame()
        index = build_rag_index(wh)
        graph = build_qa_graph(features, slice_evidence, index, _ask_models(), wh=wh)
        state = ask_question(graph, question, cutoff)

    rprint(f"[dim]as of {cutoff.date()}[/dim]\n")
    if state.message:
        rprint(state.message)
        for line in state.data_notes:
            rprint(f"[yellow]{line}[/yellow]")
        return
    if state.answer is None:
        rprint("[yellow]no answer produced[/yellow]")
        return
    rprint(f"[bold]{state.answer.overview}[/bold] [dim](backend={state.answer.backend})[/dim]\n")
    for note in state.answer.notes:
        rprint(f"  [dim]{note.match_id} {note.home_team} vs {note.away_team}:[/dim] {note.take}")
    for line in state.data_notes:
        rprint(f"[yellow]{line}[/yellow]")
    if not state.answer.citations_grounded:
        rprint(f"\n[yellow]ungrounded numbers flagged: {state.answer.ungrounded_numbers}[/yellow]")


@app.command("rag-eval")
def rag_eval(n: int = 80, k: int = 5) -> None:
    """Retrieval quality (hit@1, hit@k, MRR) on synthetic QA drawn from the indexed corpus."""
    from pitch_edge.rag.documents import match_documents, news_documents, prediction_documents
    from pitch_edge.rag.eval import evaluate_retrieval, summarize_eval, synthetic_questions
    from pitch_edge.rag.index import VectorIndex

    with _wh() as wh:
        docs = match_documents(wh.matches_with_closing_odds(bookmakers=("PS", "Mkt")), limit=3000)
        preds = wh.read("model_predictions")
        if not preds.empty:
            latest = preds.sort_values("run_id")["run_id"].iloc[-1]
            for m, g in preds[preds["run_id"] == latest].groupby("model_name"):
                docs += prediction_documents(g.tail(500), str(m), None)
        news = wh.read("news_items")
        if not news.empty:
            docs += news_documents(news)
    idx = VectorIndex()
    items = synthetic_questions(docs, n=n)
    results = evaluate_retrieval(idx, items, k=k)
    out = get_settings().artifacts_dir / "rag_eval.csv"
    results.to_csv(out, index=False)
    _print_df(summarize_eval(results, k=k).round(3), f"Retrieval eval — index={idx.backend}, {idx.count():,} docs")


@app.command()
def ablation(
    min_date: str = "2015-07-01",
    leagues: str = "",
    groups: str = typer.Option(
        "referee,weather,travel_fatigue,elo,rolling_form,wiki_attention,rotation_load,squad_value"
    ),
    edge: float = 0.03,
    retrain_days: int = 30,
) -> None:
    """Feature-group ablation for the GBDT (the CASE_STUDY experiment)."""
    from pitch_edge.ablation import run_ablation
    from pitch_edge.backtest.engine import WalkForwardConfig
    from pitch_edge.pipeline import load_feature_frame

    lg = [s.strip() for s in leagues.split(",") if s.strip()] or None
    with _wh() as wh:
        f = load_feature_frame(wh, leagues=lg, min_date=min_date)
        table, _ = run_ablation(
            f,
            [g.strip() for g in groups.split(",")],
            WalkForwardConfig(edge_threshold=edge, retrain_every_days=retrain_days),
        )
        out = get_settings().artifacts_dir / "ablation.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        _print_df(table.round(4), "Feature-group ablation (GBDT, ¼-Kelly)")


@app.command()
def setup(
    env_file: str = typer.Option(".env", help="git-ignored file the keys are written to"),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="report status only, never prompt"),
) -> None:
    """Guided onboarding for the OPTIONAL keys: shows what each unlocks, asks for the ones you have,
    writes them to a git-ignored .env (never echoed, never logged) and says what to run next.

    Everything works with zero keys; this only unlocks extras. None of the keys is a bookmaker or
    exchange credential — nothing accepted here can place an order or move money, by construction.
    """
    from pitch_edge.keys import key_status, status_summary, write_env_value

    rows = key_status(dotenv_path=env_file)
    t = Table(title="Optional keys — nothing below is required")
    for c in ("status", "env var", "unlocks", "sign-up"):
        t.add_column(c)
    for r in rows:
        t.add_row(
            "✅ set (" + str(r["source"]) + ")" if r["set"] else "—  missing", r["env"], r["unlocks"], r["signup"]
        )
    rprint(t)
    written: list[str] = []
    if not non_interactive:
        for r in rows:
            if r["set"]:
                continue
            if not typer.confirm(f"Do you have a {r['name']} value for {r['env']}?", default=False):
                continue
            value = typer.prompt(f"{r['env']} (input hidden)", hide_input=True, default="", show_default=False)
            if value.strip():
                write_env_value(env_file, r["env"], value.strip())
                written.append(r["env"])
    summary = status_summary(dotenv_path=env_file)
    rprint(
        f"\n[bold]{summary['n_api_set']} of {summary['n_api_total']} optional API keys unlocked[/bold]"
        + (
            " · Anthropic key active (Fable 5.1 explanations)"
            if summary["anthropic_active"]
            else " · explainer runs on the offline template"
        )
    )
    if written:
        rprint(f"wrote {len(written)} value(s) to {env_file} (owner-only permissions; values are never printed)")
    else:
        rprint("nothing written — the system keeps running fully keyless")
    nxt = summary["next_unlock"]
    if nxt:
        rprint(f"[dim]next unlock: {nxt['env']} → {nxt['unlocks'][:90]}… ({nxt['signup']})[/dim]")
    rprint(
        "[bold]next:[/bold] "
        + ("uv run pitch-edge ingest  (then `refresh`)" if written else "uv run pitch-edge refresh")
    )
    rprint(
        "[dim]No key here can place a bet or move money; there is no bookmaker account anywhere in the system.[/dim]"
    )


@app.command()
def export(out_dir: str = "") -> None:
    """Export every warehouse table to Parquet (the GCS/BigQuery hand-off format)."""
    with _wh() as wh:
        paths = wh.export_parquet(out_dir or None)
        rprint(f"exported {len(paths)} tables")


@app.command()
def health() -> None:
    """Data pipeline health per source, football-data lag behind the lake, and fixtures stored twice under
    different spellings."""
    from pitch_edge.data.quality import find_duplicate_fixtures

    with _wh() as wh:
        _print_df(wh.health(), "Pipeline health")
        _print_football_data_lag(wh)
        dupes = find_duplicate_fixtures(wh)
        if dupes.empty:
            rprint("[bold]duplicate fixtures:[/bold] none")
        else:
            _print_df(dupes.head(50), f"Duplicate fixtures ({len(dupes)} pairs, first 50)")


@app.command()
def refresh(fast: bool = typer.Option(False, help="skip the weather ingest for a quicker run")) -> None:
    """Ingest -> features -> backtests -> RAG index in one go."""
    from pitch_edge.pipeline import full_refresh

    summary = full_refresh(fast=fast)
    rprint(json.dumps({k: v for k, v in summary.items() if k != "ingest"}, default=str, indent=2)[:4000])


@app.command()
def serve(port: int = 8501) -> None:  # pragma: no cover
    """Launch the dashboard (NiceGUI) — reads warehouse tables/artifacts only, never computes new numbers."""
    import os

    os.environ["PITCH_EDGE_DASHBOARD_PORT"] = str(port)
    from pitch_edge.dashboard.web import main as run_dashboard

    run_dashboard()


@app.command()
def schedule() -> None:  # pragma: no cover
    """Run the local APScheduler loop (hourly news/markets, nightly refresh)."""
    from pitch_edge.scheduler import run_forever

    run_forever()


@app.command("lake-sync")
def lake_sync(
    only: list[str] = typer.Option(None, "--only", help="Mirror name(s) to sync (repeatable; default: all)"),  # noqa: B008
    since: str = typer.Option("", help="Re-pull windows from this date (YYYY-MM-DD); default: each mirror's start"),
    full_refresh: bool = typer.Option(
        False, "--full-refresh", help="Empty each selected mirror first, then re-pull it"
    ),
    spine: bool = typer.Option(
        True, "--spine/--no-spine", help="After syncing fd_matches, rebuild the football-data matches/odds rows"
    ),
    skip_if_busy: bool = typer.Option(
        False,
        "--skip-if-busy",
        help="Exit 0 with 'skipped: warehouse busy' when another process holds the warehouse (for schedulers)",
    ),
) -> None:
    """Mirror lakehouse tables into the warehouse as lake_* tables (idempotent; needs LAKEHOUSE_API_* in .env).

    Syncing `fd_matches` also rebuilds the warehouse spine's football-data rows from the mirror. While a lake
    freeze is recorded (`pitch-edge control freeze`) it prints "skipped: lake freeze since ..." and exits 0 without
    contacting the lake."""
    import os

    from pitch_edge.control_cli import freeze_notice

    notice = freeze_notice()
    if notice is not None:
        typer.echo(f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} skipped: {notice}")  # one unwrapped log line
        raise typer.Exit(0)
    if not os.environ.get("LAKEHOUSE_API_TOKEN"):
        rprint("[red]LAKEHOUSE_API_TOKEN is not set (put LAKEHOUSE_API_URL / LAKEHOUSE_API_TOKEN in .env)[/red]")
        raise typer.Exit(2)

    from pitch_edge.data.ingest import build_football_data_spine
    from pitch_edge.lake.client import Lake
    from pitch_edge.lake.mirror import MIRRORS, run_lake_sync

    wh = _open_warehouse(skip_if_busy)
    try:
        with Lake() as lake:
            result = run_lake_sync(
                wh, lake, only=list(only or []) or None, since=since or None, full_refresh=full_refresh
            )
        for name, rows in result.items():
            table = MIRRORS[name].table
            rprint(f"{table}: {'FAILED' if rows is None else f'{rows:,} rows written'} ({wh.count(table):,} stored)")
        if spine and result.get("fd_matches") is not None:
            rprint(f"football-data spine: {build_football_data_spine(wh):,} matches written")
    finally:
        wh.close()
    if any(v is None for v in result.values()):
        raise typer.Exit(1)


def _open_warehouse(skip_if_busy: bool) -> Warehouse:
    """Open the warehouse; with `skip_if_busy`, a lock held by another process prints a timestamped
    "skipped: warehouse busy" line and exits 0 (the next schedule retries). Any other error propagates."""
    import duckdb

    try:
        return _wh()
    except duckdb.IOException as exc:
        if skip_if_busy and "lock" in str(exc).lower():
            stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            rprint(f"{stamp} skipped: warehouse busy ({get_settings().db_path} is locked by another process)")
            raise typer.Exit(0) from exc
        raise


def _print_football_data_lag(wh: Warehouse) -> None:
    """Print the football-data lag table and warnings (see `pitch_edge.lake.freshness`). The lake column is
    left empty, with a note, when the lake cannot be queried."""
    import os

    from pitch_edge.lake.freshness import MAX_LAG_DAYS, MAX_SYNC_AGE_HOURS, football_data_lag, lake_newest_dates

    leagues = _backtest_leagues()
    lake_newest = None
    note = ""
    if os.environ.get("LAKEHOUSE_API_TOKEN"):
        try:
            from pitch_edge.lake.client import Lake

            with Lake() as lake:
                lake_newest = lake_newest_dates(lake, leagues)
        except Exception as exc:  # noqa: BLE001 - health must still print without the lake
            note = f" (lake not queried: {type(exc).__name__})"
    else:
        note = " (lake not queried: LAKEHOUSE_API_TOKEN is not set)"
    lag = football_data_lag(wh, lake_newest, leagues)
    _print_df(lag.table, "Football-data lag (newest completed match per backtest league)")
    rprint(
        f"[bold]football-data mirror last sync:[/bold] {lag.last_sync or 'never'}  "
        f"[bold]spine last build:[/bold] {lag.last_spine_build or 'never'}{note}"
    )
    if lag.warnings:
        for w in lag.warnings:
            rprint(f"[yellow]WARN football-data lag:[/yellow] {w}")
    else:
        rprint(
            f"[bold]football-data lag:[/bold] ok (limits: lake gap {MAX_LAG_DAYS} d, sync age {MAX_SYNC_AGE_HOURS} h)"
        )


def _backtest_leagues() -> list[str]:
    from pitch_edge.pipeline import DEFAULT_BACKTEST_LEAGUES

    return list(DEFAULT_BACKTEST_LEAGUES)


def _print_df(df: pd.DataFrame, title: str) -> None:
    t = Table(title=title)
    for c in df.columns:
        t.add_column(str(c))
    for row in df.itertuples(index=False):
        t.add_row(*[str(v) for v in row])
    rprint(t)


# `pitch-edge research {register,status,grade,ledger}` — see research/cli.py.
from pitch_edge.research.cli import research_app  # noqa: E402

app.add_typer(research_app, name="research")

# `pitch-edge paper {collect,decide,settle,report,status}` — see paper/cli.py.
from pitch_edge.paper.cli import paper_app  # noqa: E402

app.add_typer(paper_app, name="paper")

# `pitch-edge control {freeze,thaw,status,gate}` — see control_cli.py.
from pitch_edge.control_cli import control_app  # noqa: E402

app.add_typer(control_app, name="control")


if __name__ == "__main__":  # pragma: no cover
    app()
