"""`pitch-edge` command line: setup, ingest, features, backtest, intel, rag, signals, export, serve."""

from __future__ import annotations

import json
import logging

import pandas as pd
import typer
from rich import print as rprint
from rich.table import Table

from pitch_edge.agents.graph import SignalState
from pitch_edge.config import get_settings
from pitch_edge.data.storage import Warehouse

app = typer.Typer(help="PITCH-EDGE — football intelligence & market-edge research. Surfaces signals, never bets.")


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
    leagues: str = typer.Option("", help="Comma-separated football-data.co.uk codes (default: all)"),
    seasons: str = typer.Option("", help="Comma-separated season start years (default: 2005-2025)"),
    no_statsbomb: bool = typer.Option(False, help="Skip StatsBomb event data"),
    no_weather: bool = typer.Option(False, help="Skip Wikidata venues + Open-Meteo weather"),
    statsbomb_max: int = typer.Option(40, help="Max matches with events per StatsBomb competition"),
) -> None:
    """Pull every free source into the local warehouse (idempotent, cached)."""
    from pitch_edge.data.ingest import ingest_everything

    lg = [s.strip() for s in leagues.split(",") if s.strip()] or None
    yrs = [int(s) for s in seasons.split(",") if s.strip()] or None
    with _wh() as wh:
        report = ingest_everything(
            wh,
            lg,
            yrs,
            include_statsbomb=not no_statsbomb,
            include_weather=not no_weather,
            statsbomb_max_matches=statsbomb_max,
        )
        rprint({k: v for k, v in report.items() if v})
        rprint(f"[bold]matches:[/bold] {wh.count('matches'):,}  [bold]odds rows:[/bold] {wh.count('odds'):,}")


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
        "dixon_coles,gbdt,gbdt_mkt,gru_sequence",
        help="Comma-separated model names (also: transformer_sequence)",
    ),
    edge: float = 0.03,
    retrain_days: int = 30,
    label: str = "main",
) -> None:
    """Walk-forward backtest with CLV / Kelly / calibration; writes reports/<label>/REPORT.md."""
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


@app.command()
def rag(question: str = typer.Argument(..., help="Ask the system a grounded question")) -> None:
    """Query the RAG layer (local-LLM query parsing -> retrieval -> grounded explanation)."""
    from pitch_edge.rag.generate import GroundedGenerator
    from pitch_edge.rag.index import VectorIndex
    from pitch_edge.rag.query_parser import parse_query, retrieval_query_text

    idx = VectorIndex()
    parsed = parse_query(question)
    query_text = retrieval_query_text(parsed)
    docs = [d for d, _ in idx.query(query_text, k=6)]
    ans = GroundedGenerator().answer(question, docs)
    parsed_note = f", parsed_teams={parsed.home_team!r} vs {parsed.away_team!r}" if parsed.home_team else ""
    rprint(f"[dim]backend={ans.backend}, index={idx.backend}, docs={idx.count()}{parsed_note}[/dim]\n")
    rprint(ans.text)


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
def signals(model: str = "gbdt", min_date: str = "2018-07-01") -> None:
    """Run the LangGraph pipeline to the human approval gate and print pending proposals."""
    from pitch_edge.models import default_models
    from pitch_edge.pipeline import build_signal_pipeline, load_feature_frame

    with _wh() as wh:
        f = load_feature_frame(wh, min_date=min_date)
        m = next(x for x in default_models() if x.name == model)
        m.fit(f)
        pipe = build_signal_pipeline(wh, f, m)
        thread_id, state = pipe.run_to_gate()
        rprint(f"[bold]thread:[/bold] {thread_id}  (paused at HUMAN APPROVAL GATE)")
        props = state.get("proposals", [])
        if not props:
            rprint("no proposals passed the risk manager")
        else:
            _print_df(
                pd.DataFrame(props)[
                    [
                        "match_id",
                        "home_team",
                        "away_team",
                        "outcome",
                        "model_probability",
                        "market_probability",
                        "edge",
                        "decimal_odds",
                        "stake",
                    ]
                ].round(3),
                "Pending proposals",
            )
        pending = wh.read("paper_trades") if wh.table_exists("paper_trades") else pd.DataFrame()
        (get_settings().artifacts_dir / "pending_signals.json").write_text(
            json.dumps({"thread_id": thread_id, "proposals": props}, default=str, indent=2)
        )
        rprint(f"[dim]{len(pending)} paper trades logged historically[/dim]")


def _run_agentic_pipeline(wh: Warehouse, model: str, min_date: str) -> tuple[str, SignalState]:
    """Shared by `agentic-signals` and `predict`: build deps, fit models, run the fully
    deterministic agentic graph to the human-approval gate. No LLM involved anywhere in here —
    routing, edge detection and risk sizing all run on real resources already in the warehouse."""
    from pitch_edge.agents.graph import GraphDependencies
    from pitch_edge.agents.orchestrator import AgenticSignalPipeline
    from pitch_edge.agents.risk import RiskLimits, RiskManager
    from pitch_edge.models import available_models, default_models
    from pitch_edge.pipeline import (
        build_rag_index,
        live_quotes_from_odds_api,
        load_feature_frame,
        synthetic_quotes_from_elo,
        upcoming_fixture_frame,
    )

    f = load_feature_frame(wh, min_date=min_date)
    if f.empty:
        rprint("[red]no features — run `pitch-edge ingest` and `pitch-edge features` first[/red]")
        raise typer.Exit(1)
    fallback_model = next(x for x in default_models() if x.name == model)
    fitted = available_models()
    for m in fitted:
        m.fit(f)
    fallback_model.fit(f)
    fixtures_df = upcoming_fixture_frame(wh, f)

    def scout() -> list[dict]:
        return (
            fixtures_df.assign(date=fixtures_df["date"].astype(str)).to_dict(orient="records")
            if not fixtures_df.empty
            else []
        )

    def fetch_odds(rows: list[dict]) -> list[dict]:
        if not rows:
            return []
        rows_df = pd.DataFrame(rows)
        live = live_quotes_from_odds_api(wh, rows_df)
        synthetic = synthetic_quotes_from_elo(rows_df)
        return [live.get(q["match_id"], q) for q in synthetic]

    def sink(alerts: list[dict]) -> None:
        if alerts:
            wh.upsert("paper_trades", pd.DataFrame(alerts))

    deps = GraphDependencies(
        scout=scout,
        featurize=lambda rows: rows,
        infer=lambda rows: [],  # replaced by the per-league router inside build_agentic_graph
        fetch_odds=fetch_odds,
        risk=RiskManager(RiskLimits()),
        sink=sink,
        model_name=fallback_model.name,
    )
    reports_dir = get_settings().backtest_dir / "main"
    index = build_rag_index(wh)
    pipe = AgenticSignalPipeline(deps, reports_dir=reports_dir, index=index, wh=wh, available_models=fitted)
    thread_id, state = pipe.run_to_gate()
    return thread_id, state


def _print_proposals(state: SignalState, title: str) -> None:
    reviews_by_key = {(r["match_id"], r["outcome"]): r for r in state.get("reviews", [])}
    props = state.get("proposals", [])
    if not props:
        rprint("no proposals passed the risk manager")
        return
    table = pd.DataFrame(props)
    table["verdict"] = [reviews_by_key.get((p["match_id"], p["outcome"]), {}).get("verdict", "") for p in props]
    table["reasons"] = [
        "; ".join(reviews_by_key.get((p["match_id"], p["outcome"]), {}).get("reasons", [])) for p in props
    ]
    table["real_odds"] = table["bookmaker"] != "synthetic_elo_book"
    _print_df(
        table[
            [
                "match_id",
                "home_team",
                "away_team",
                "outcome",
                "model_probability",
                "market_probability",
                "edge",
                "decimal_odds",
                "stake",
                "real_odds",
                "verdict",
                "reasons",
            ]
        ].round(3),
        title,
    )


@app.command("agentic-signals")
def agentic_signals(
    model: str = typer.Option("gbdt", help="Default/fallback model — the per-league router picks the real one"),
    min_date: str = "2018-07-01",
) -> None:
    """Genuinely agentic signals pipeline: data-quality screening, per-league model routing,
    deterministic edge review — same human-approval gate as `signals`. Fully deterministic, no LLM
    involved — for a plain-language explanation of the resulting proposals, run `pitch-edge predict`
    instead, which calls the LLM exactly once over the whole batch."""
    with _wh() as wh:
        thread_id, state = _run_agentic_pipeline(wh, model, min_date)
        rprint(f"[bold]thread:[/bold] {thread_id}  (paused at HUMAN APPROVAL GATE)")
        rprint(f"[dim]screened: {len(state.get('selected_fixtures', []))} fixtures scouted[/dim]")
        _print_proposals(state, "Pending proposals (agentic)")
        (get_settings().artifacts_dir / "pending_agentic_signals.json").write_text(
            json.dumps(
                {
                    "thread_id": thread_id,
                    "proposals": state.get("proposals", []),
                    "selected_fixtures": state.get("selected_fixtures", []),
                    "reviews": state.get("reviews", []),
                },
                default=str,
                indent=2,
            )
        )


@app.command()
def predict(
    model: str = typer.Option("gbdt", help="Default/fallback model — the per-league router picks the real one"),
    min_date: str = "2018-07-01",
    llm_model: str | None = typer.Option(
        None, help="Override the local Ollama model (default: settings.local_llm_model)"
    ),
) -> None:
    """Run the deterministic agentic pipeline, then ask the LLM once — a single call over the
    whole batch of proposals, not one per fixture — to explain the results in plain language. This
    is the ONLY command that touches an LLM; everything it explains was already computed
    deterministically before this call happens. Falls back to a plain-text template if Ollama isn't
    running, so you can always see your proposals."""
    from pitch_edge.agents.explainer import explain_signals

    with _wh() as wh:
        thread_id, state = _run_agentic_pipeline(wh, model, min_date)
        rprint(f"[bold]thread:[/bold] {thread_id}  (paused at HUMAN APPROVAL GATE)")
        _print_proposals(state, "Pending proposals")
        explanation = explain_signals(state.get("proposals", []), state.get("reviews", []), llm_model=llm_model)
        rprint(f"\n[bold]explanation[/bold] [dim](backend={explanation.backend})[/dim]\n{explanation.overview}\n")
        for note in explanation.notes:
            rprint(f"  [dim]{note.match_id} {note.outcome}:[/dim] {note.take}")


@app.command("replay-eval")
def replay_eval(
    as_of: str | None = typer.Option(
        None, help="Override T0 (YYYY-MM-DD); default: auto-computed from real closing odds"
    ),
    window_days: int = 30,
) -> None:
    """T0 replay-simulation: train on data strictly before T0, run the agentic pipeline on the next
    `window_days` as if they were upcoming fixtures, then reveal real closing odds/results and check
    whether the reviewer's deterministic trust/distrust verdicts actually correlate with realized
    edge. No LLM involved — the reviewer's verdict is a real-edge_bits/real-odds rule, so this is
    validating that rule, not an LLM's judgment."""
    from pitch_edge.backtest.replay import run_replay_eval

    with _wh() as wh:
        result = run_replay_eval(wh, get_settings().backtest_dir / "replay", as_of=as_of, window_days=window_days)
        rprint(
            f"[bold]T0:[/bold] {result['t0'].date()}  [bold]window:[/bold] {result['t0'].date()} -> {result['window_end'].date()}"
        )
        rprint(
            f"[dim]{result['n_fixtures']} fixtures in window, {result['n_proposals']} proposals, {result['n_scored']} scored[/dim]"
        )
        if result["summary"].empty:
            rprint(
                "[yellow]no proposals could be scored (no real closing odds in this window, or nothing proposed)[/yellow]"
            )
        else:
            _print_df(result["summary"].round(4), "Realized edge by reviewer verdict")


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
        + (" · GCP mirror configured" if summary["gcp_configured"] else "")
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
def artifacts() -> None:
    """Build dashboard artifacts: team strengths, feature importance, data universe."""
    from pitch_edge.artifacts import build_all_artifacts

    with _wh() as wh:
        rprint(build_all_artifacts(wh))


@app.command()
def export(out_dir: str = "") -> None:
    """Export every warehouse table to Parquet (the GCS/BigQuery hand-off format)."""
    with _wh() as wh:
        paths = wh.export_parquet(out_dir or None)
        rprint(f"exported {len(paths)} tables")


@app.command()
def health() -> None:
    """Data pipeline health per source."""
    with _wh() as wh:
        _print_df(wh.health(), "Pipeline health")


@app.command()
def refresh(fast: bool = typer.Option(False, help="skip weather + GRU for a quicker run")) -> None:
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


def _print_df(df: pd.DataFrame, title: str) -> None:
    t = Table(title=title)
    for c in df.columns:
        t.add_column(str(c))
    for row in df.itertuples(index=False):
        t.add_row(*[str(v) for v in row])
    rprint(t)


if __name__ == "__main__":  # pragma: no cover
    app()
