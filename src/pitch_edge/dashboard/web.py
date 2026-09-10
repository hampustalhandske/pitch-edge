"""PITCH-EDGE — the dashboard.

Five views, nothing else: Overview (does any model beat the closing price, by how much) ·
Data universe (what's loaded, how fresh) · Backtest & calibration (walk-forward results per
model) · Suggestions & approval (today's proposals, gated on a human) · Ask the system
(grounded, cited Q&A).

Built with NiceGUI — a Python-native UI framework on top of FastAPI + Vue/Quasar. Chosen over a
separate JS frontend because the "Python-only" platform value in CLAUDE.md is real: every view
here is a straight read of a warehouse table or a `reports/`/`artifacts/` file (see
`pitch_edge.dashboard.data`, the *only* module that touches those paths), so there is no API
contract worth its own service, no client-side build step, and no second language for a
single maintainer to keep in sync. NiceGUI still gets the "shocked and impressed" bar: real
CSS layout (not Streamlit's fixed block-container), Quasar components (cards, tables, dialogs,
notifications), instant reactive updates without a full-page rerun, and the same Plotly charts
the old dashboard used, restyled.

This module never trains a model, never fetches odds, and never places anything. The one
write path (`data.record_decisions`) appends `approved_paper` / `rejected` rows to the
`paper_trades` warehouse table after a named human explicitly picks proposals — the same shape
the retired Streamlit app used. Nothing here can move real money.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nicegui import run, ui

from pitch_edge.dashboard import data as D
from pitch_edge.dashboard.theme import DIVERGING, INK, OUTCOME, SERIES, STATUS, install_template

install_template()

PAGE_BG = "#0d0d0d"
SURFACE = "#161615"
BORDER = "rgba(255,255,255,0.08)"


# ============================================================================= small helpers
def card(*, title: str | None = None) -> ui.card:
    c = (
        ui.card()
        .classes("w-full")
        .style(f"background:{SURFACE}; border:1px solid {BORDER}; border-radius:14px; box-shadow:none;")
    )
    with c:
        if title:
            ui.label(title).classes("text-sm font-medium").style(f"color:{INK['secondary']}; letter-spacing:.01em;")
    return c


def kpi(label: str, value: str, sub: str = "", tone: str = "neutral") -> None:
    color = {"good": STATUS["good"], "bad": STATUS["critical"], "neutral": INK["primary"]}[tone]
    with (
        ui.column()
        .classes("gap-0")
        .style(
            f"background:{SURFACE}; border:1px solid {BORDER}; border-radius:14px; padding:14px 18px; min-width:170px;"
        )
    ):
        ui.label(label).style(f"color:{INK['muted']}; font-size:.78rem; letter-spacing:.02em;")
        ui.label(value).style(f"color:{color}; font-size:1.6rem; font-weight:600; line-height:1.25;")
        if sub:
            ui.label(sub).style(f"color:{INK['muted']}; font-size:.75rem;")


def banner(text: str, kind: str = "note") -> None:
    border = STATUS["warning"] if kind == "warn" else SERIES[0]
    bg = "#241f13" if kind == "warn" else SURFACE
    ui.html(
        f'<div style="border-left:3px solid {border}; background:{bg}; color:{INK["secondary"]}; '
        f'padding:10px 14px; border-radius:8px; font-size:.88rem; line-height:1.5;">{text}</div>'
    ).classes("w-full")


def plot(fig: go.Figure, height: int = 320) -> None:
    fig.update_layout(height=height)
    ui.plotly(fig).classes("w-full").style(f"height:{height}px;")


def frame_table(df: pd.DataFrame, *, height: str = "360px", row_key: str | None = None) -> ui.table | None:
    if df.empty:
        ui.label("No rows yet.").style(f"color:{INK['muted']};")
        return None
    columns = [
        {"name": c, "label": c.replace("_", " "), "field": c, "sortable": True, "align": "left"} for c in df.columns
    ]
    rows = json.loads(df.to_json(orient="records"))
    t = ui.table(columns=columns, rows=rows, row_key=row_key or df.columns[0], pagination=10).classes("w-full")
    t.style(f"background:{SURFACE};")
    return t


def section_title(text: str, caption: str = "") -> None:
    ui.label(text).style("font-size:1.15rem; font-weight:600; letter-spacing:-.01em;")
    if caption:
        ui.label(caption).style(f"color:{INK['muted']}; font-size:.85rem; margin-top:-6px;")


# ============================================================================= 1. overview
def build_overview() -> None:
    d = D.overview_data()
    universe, summ_run, run_id, pending = d["universe"], d["summ_run"], d["run"], d["pending"]

    with ui.row().classes("w-full gap-3 flex-wrap"):
        if universe:
            kpi("Matches", D.fmt(universe.get("matches")))
            kpi("Divisions", D.fmt(universe.get("leagues")))
            kpi("Odds rows", D.fmt(universe.get("odds")))
            kpi("Backtest run", (run_id or "—")[:12])
        else:
            banner("No data yet — run <code>uv run pitch-edge refresh</code>.", "warn")

    ui.space().classes("h-1")
    section_title("What the latest run says")
    if summ_run.empty:
        banner("Run <code>uv run pitch-edge backtest</code> to populate this view.", "warn")
        return

    main = summ_run[summ_run.get("label", "main") == "main"] if "label" in summ_run else summ_run
    cal = main.drop_duplicates("model")[
        ["model", "multiclass_log_loss", "market_multiclass_log_loss", "n_predictions"]
    ].copy()
    cal["bits_vs_market"] = D.bits(cal["multiclass_log_loss"], cal["market_multiclass_log_loss"])
    best = cal.sort_values("bits_vs_market", ascending=False).iloc[0]
    q = main[main["strategy"] == "kelly_quarter"].set_index("model") if "strategy" in main else pd.DataFrame()
    clv_src = main["closing_price_source"].iloc[0] if "closing_price_source" in main else ""
    price_src = main["bet_price_source"].iloc[0] if "bet_price_source" in main else "?"

    with ui.row().classes("w-full gap-3 flex-wrap"):
        kpi(
            "Best model vs closing price",
            f"{best['bits_vs_market']:+.3f} bits",
            "positive = model beats the no-vig price",
            "good" if best["bits_vs_market"] > 0 else "bad",
        )
        kpi("Out-of-sample predictions", D.fmt(cal["n_predictions"].max()))
        kpi(
            "Mean CLV (¼-Kelly)",
            "n/a" if clv_src == "bet_price_no_closing_available" else f"{q['mean_clv_pct'].max():+.2%}",
            "no closing line in this spine yet" if clv_src == "bet_price_no_closing_available" else "",
        )
        kpi("Proposals at the gate", str(len(pending.get("proposals", []))))

    verdict = (
        "no model beats the market on this run"
        if best["bits_vs_market"] < 0
        else f"<b>{best['model']}</b> beats the market by {best['bits_vs_market']:.3f} bits"
    )
    price_note = (
        "market-average closing price — CLV is zero by construction until football-data.co.uk's early Pinnacle "
        "line is back"
        if clv_src == "bet_price_no_closing_available"
        else "early Pinnacle"
    )
    banner(
        f"<b>Verdict:</b> {verdict}. Bets are priced at <b>{price_src}</b> ({price_note}). "
        "Read <i>bits vs market</i> first, ROI last.",
        "warn" if best["bits_vs_market"] < 0 else "note",
    )

    with ui.row().classes("w-full gap-3 items-stretch"):
        with card().classes("flex-1"):
            race = cal.sort_values("bits_vs_market")
            fig = go.Figure(
                go.Bar(
                    x=race["bits_vs_market"],
                    y=race["model"],
                    orientation="h",
                    marker_color=[STATUS["good"] if v > 0 else SERIES[0] for v in race["bits_vs_market"]],
                    text=[f"{v:+.3f}" for v in race["bits_vs_market"]],
                    textposition="outside",
                    cliponaxis=False,
                )
            )
            fig.add_vline(x=0, line_color=INK["axis"])
            fig.update_layout(
                title="Model race — information vs the closing price (bits/match)",
                xaxis_title="bits vs market",
                yaxis_title="",
                showlegend=False,
            )
            plot(fig, 320)
        by_league = pd.DataFrame(universe.get("by_league", []))
        if not by_league.empty:
            with card().classes("flex-1"):
                by_league = by_league.copy()
                by_league["tier"] = np.select(
                    [
                        by_league["league_code"].str.startswith("SWE"),
                        by_league["league_code"].isin(D.BIG5),
                        by_league["league_code"].str.len() == 3,
                    ],
                    ["Sweden", "Big-5", "Developing"],
                    default="Other European",
                )
                tier = by_league.groupby("tier", as_index=False)["n"].sum().sort_values("n", ascending=False)
                fig2 = px.bar(
                    tier,
                    x="tier",
                    y="n",
                    color="tier",
                    color_discrete_sequence=SERIES[:4],
                    title="Matches by market tier",
                    text_auto=".3s",
                )
                fig2.update_layout(showlegend=False, xaxis_title="", yaxis_title="matches")
                plot(fig2, 320)


# ============================================================================= 2. data universe
def build_data_universe() -> None:
    universe = D.artifact("data_universe.json")
    section_title(
        "What is loaded, from where",
        "football-data.co.uk (spine when reachable), the open Club-Football-Match-Data 2000-2025 compilation, "
        "openfootball, StatsBomb Open Data, Club Elo, Wikidata venues, Open-Meteo weather, news RSS, "
        "Polymarket + Kalshi, the open Transfermarkt extract, TheSportsDB.",
    )
    by_league = pd.DataFrame(universe.get("by_league", []))
    if not by_league.empty:
        by_league = by_league.copy()
        by_league["tier"] = np.select(
            [
                by_league["league_code"].str.startswith("SWE"),
                by_league["league_code"].isin(D.BIG5),
                by_league["league_code"].str.len() == 3,
            ],
            ["Sweden (focus)", "Big-5 top flight", "Developing / under-covered"],
            default="Other European",
        )
        with card():
            fig = px.bar(
                by_league.sort_values("n", ascending=False),
                x="league_code",
                y="n",
                color="tier",
                color_discrete_map={
                    "Big-5 top flight": SERIES[0],
                    "Other European": SERIES[2],
                    "Developing / under-covered": SERIES[1],
                    "Sweden (focus)": SERIES[3],
                },
                hover_data=["country", "first_date", "last_date"],
                labels={"n": "matches", "league_code": "division"},
                title="Matches per division",
            )
            fig.update_layout(legend_title="")
            plot(fig, 380)

        with ui.row().classes("w-full gap-3 items-stretch"):
            by_season = pd.DataFrame(universe.get("by_season", []))
            if not by_season.empty:
                with card().classes("flex-1"):
                    f = px.area(
                        by_season, x="season", y="n", title="Matches per season", color_discrete_sequence=[SERIES[0]]
                    )
                    f.update_layout(xaxis_tickangle=-60, yaxis_title="matches", xaxis_title="")
                    plot(f, 300)
            ob = pd.DataFrame(universe.get("odds_by_book", []))
            if not ob.empty:
                with card().classes("flex-1"):
                    ob = ob.copy()
                    ob["label"] = ob["bookmaker"] + np.where(ob["is_closing"], " (closing)", "") + " · " + ob["market"]
                    f = px.bar(
                        ob,
                        x="n",
                        y="label",
                        orientation="h",
                        title="Odds rows by bookmaker · market",
                        color_discrete_sequence=[SERIES[0]],
                    )
                    f.update_layout(yaxis_title="", xaxis_title="rows")
                    plot(f, 300)

    section_title("Row counts, per table")
    counts = pd.DataFrame(
        [
            {"table": k, "rows": universe.get(k, 0)}
            for k in (
                "matches",
                "odds",
                "weather",
                "venues",
                "news_items",
                "market_snapshots",
                "statsbomb_matches",
                "statsbomb_events",
                "openfootball_matches",
                "features",
                "model_predictions",
                "backtest_bets",
                "player_embeddings",
                "inplay_paths",
            )
        ]
    )
    tm = D.query(
        "SELECT 'tm_appearances' AS t, count(*) n FROM tm_appearances UNION ALL "
        "SELECT 'tm_game_events', count(*) FROM tm_game_events UNION ALL "
        "SELECT 'tm_players', count(*) FROM tm_players UNION ALL "
        "SELECT 'tm_player_valuations', count(*) FROM tm_player_valuations"
    )
    if not tm.empty:
        counts = pd.concat([counts, tm.rename(columns={"t": "table", "n": "rows"})], ignore_index=True)
    with card():
        frame_table(counts, row_key="table")

    section_title("Pipeline health", "last successful run per data source (`pipeline_runs`)")
    health = pd.DataFrame(universe.get("sources", []))
    if health.empty:
        health = D.query("SELECT * FROM pipeline_runs ORDER BY finished_at DESC LIMIT 200")
    with card():
        if not health.empty and "last_success" in health:
            h = health.copy()
            h["age_h"] = (
                (pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.to_datetime(h["last_success"])).dt.total_seconds()
                / 3600
            ).round(1)
            h["ok"] = np.where(h["last_status"] == "ok", "ok", "stale/failing")
            frame_table(
                h.sort_values("source")[["ok", "source", "last_success", "age_h", "runs", "errors", "rows_last"]],
                row_key="source",
            )
        else:
            frame_table(health)


# ============================================================================= 3. backtest & calibration
def build_backtest() -> None:
    summ_all = D.read("backtest_summaries")
    section_title("Walk-forward backtest", "no shuffled CV across time; bets priced at real, vig-inclusive odds")
    if summ_all.empty:
        banner("No backtest yet — run <code>uv run pitch-edge backtest</code>.", "warn")
        return

    run_id = D.latest_run(summ_all)
    summ_run = summ_all[summ_all["run_id"] == run_id] if run_id else summ_all
    bets_all = D.read("backtest_bets")
    bets_run = (
        bets_all[bets_all["backtest_id"].str.contains(f":{run_id}", regex=False)]
        if run_id and not bets_all.empty
        else bets_all
    )
    if not bets_run.empty:
        bets_run = bets_run.assign(
            label=bets_run["backtest_id"].str.split(":").str[0],
            model=bets_run["backtest_id"].str.split(":").str[1],
            strategy=bets_run["outcome"].str.split("|").str[1],
            side=bets_run["outcome"].str.split("|").str[0],
        )
    preds_all = D.read("model_predictions")
    preds_run = preds_all[preds_all["run_id"] == run_id] if run_id and not preds_all.empty else preds_all

    labels = sorted(summ_run["label"].unique()) if "label" in summ_run else ["main"]
    label_sel = ui.select(labels, value=("main" if "main" in labels else labels[0]), label="Slice").classes("w-56")
    strat_sel = ui.select([], label="Staking").classes("w-56")
    body = ui.column().classes("w-full gap-3")
    _state = {"rendering": False}

    def render() -> None:
        if _state["rendering"]:
            return  # a programmatic strat_sel.value assignment below re-enters this; ignore that echo
        _state["rendering"] = True
        try:
            body.clear()
            label = label_sel.value
            summ = summ_run[summ_run["label"] == label] if "label" in summ_run else summ_run
            bets = bets_run[bets_run["label"] == label] if not bets_run.empty and "label" in bets_run else bets_run
            strategies = sorted(summ["strategy"].unique())
            strat_sel.set_options(strategies)
            if strat_sel.value not in strategies:
                strat_sel.value = strategies[0] if strategies else None
            strat = strat_sel.value

            with body:
                src = (
                    summ[["bet_price_source", "closing_price_source"]].drop_duplicates().iloc[0]
                    if "bet_price_source" in summ
                    else None
                )
                if src is not None and src["closing_price_source"] == "bet_price_no_closing_available":
                    banner(
                        f"Bets priced at <b>{src['bet_price_source']}</b> (market-average price). No separate closing "
                        "line in the current spine, so <b>CLV is 0 by construction</b> — compare log-loss with the "
                        "market and treat ROI as noise.",
                        "warn",
                    )
                cal = summ.drop_duplicates("model")[
                    [
                        "model",
                        "multiclass_log_loss",
                        "market_multiclass_log_loss",
                        "multiclass_brier",
                        "market_multiclass_brier",
                        "n_predictions",
                    ]
                ].copy()
                cal["bits_vs_market"] = D.bits(cal["multiclass_log_loss"], cal["market_multiclass_log_loss"])
                view = (
                    summ[summ["strategy"] == strat].merge(cal[["model", "bits_vs_market"]], on="model")
                    if strat
                    else pd.DataFrame()
                )
                cols = [
                    c
                    for c in (
                        "model",
                        "bits_vs_market",
                        "n_bets",
                        "roi",
                        "mean_clv_pct",
                        "clv_t_stat",
                        "sharpe",
                        "max_drawdown",
                        "final_bankroll",
                        "multiclass_log_loss",
                        "market_multiclass_log_loss",
                    )
                    if c in view.columns
                ]
                with card(title="Results"):
                    if not view.empty:
                        frame_table(view[cols].sort_values("bits_vs_market", ascending=False).round(4), row_key="model")

                if not bets.empty and strat:
                    sub = bets[bets["strategy"] == strat].sort_values("date")
                    with card():
                        fig = go.Figure()
                        for i, (m, g) in enumerate(sub.groupby("model")):
                            fig.add_trace(
                                go.Scatter(
                                    x=g["date"],
                                    y=g["bankroll_after"],
                                    name=m,
                                    mode="lines",
                                    line={"color": SERIES[i % len(SERIES)]},
                                )
                            )
                        fig.add_hline(y=1000, line_dash="dot", line_color=INK["axis"])
                        fig.update_layout(title=f"Paper bankroll — {strat}", yaxis_title="bankroll", xaxis_title="")
                        plot(fig, 360)

                with card(title="Calibration"):
                    preds = preds_run.copy()
                    if not preds.empty and "label" in bets_run and not bets.empty:
                        preds = preds[
                            preds["match_id"].isin(set(bets["match_id"])) | preds["model_name"].isin(cal["model"])
                        ]
                    if preds.empty:
                        ui.label("No predictions stored for this slice.").style(f"color:{INK['muted']};")
                    else:
                        long = cal.melt(
                            id_vars="model",
                            value_vars=["multiclass_log_loss", "market_multiclass_log_loss"],
                            var_name="who",
                            value_name="log-loss",
                        )
                        long["who"] = long["who"].map(
                            {"multiclass_log_loss": "model", "market_multiclass_log_loss": "no-vig market"}
                        )
                        with ui.row().classes("w-full gap-3 items-stretch"):
                            with ui.column().classes("flex-1"):
                                f = px.bar(
                                    long,
                                    x="model",
                                    y="log-loss",
                                    color="who",
                                    barmode="group",
                                    color_discrete_sequence=[SERIES[0], INK["muted"]],
                                    title="Log-loss: model vs market (lower is better)",
                                )
                                f.update_layout(legend_title="", xaxis_title="")
                                plot(f, 340)
                            with ui.column().classes("flex-1"):
                                models = sorted(preds["model_name"].unique())
                                reliability_box = ui.column().classes("w-full")

                                def render_reliability(m: str) -> None:
                                    reliability_box.clear()
                                    from pitch_edge.models.calibration import reliability_curve

                                    p = preds[preds["model_name"] == m]
                                    f2 = go.Figure()
                                    f2.add_trace(
                                        go.Scatter(
                                            x=[0, 1],
                                            y=[0, 1],
                                            mode="lines",
                                            name="perfect",
                                            line={"dash": "dash", "color": INK["axis"]},
                                        )
                                    )
                                    for k_, o in enumerate(D.OUTCOMES):
                                        mp, ef = reliability_curve(
                                            (p["result"] == k_).astype(int).to_numpy(),
                                            p[f"p_{o}"].to_numpy(),
                                            n_bins=12,
                                        )
                                        f2.add_trace(
                                            go.Scatter(
                                                x=mp,
                                                y=ef,
                                                mode="markers+lines",
                                                name=o,
                                                line={"color": OUTCOME[o]},
                                                marker={"size": 8},
                                            )
                                        )
                                    f2.update_layout(
                                        xaxis_title="predicted probability", yaxis_title="observed frequency"
                                    )
                                    with reliability_box:
                                        plot(f2, 340)

                                m_sel = ui.select(
                                    models, value=models[0] if models else None, label="Reliability diagram"
                                ).classes("w-full")
                                m_sel.on_value_change(lambda e: render_reliability(e.value))
                                if models:
                                    render_reliability(models[0])

                by_lg_path = D.SETTINGS.backtest_dir / label / "by_league.csv"
                if by_lg_path.exists():
                    bl = pd.read_csv(by_lg_path)
                    piv = bl.pivot_table(index="league_code", columns="model", values="edge_bits")
                    with card():
                        f = px.imshow(
                            piv,
                            color_continuous_scale=DIVERGING,
                            zmin=-0.08,
                            zmax=0.08,
                            aspect="auto",
                            title="Bits vs market by division × model (blue = model ahead)",
                            text_auto=".3f",
                        )
                        f.update_layout(xaxis_title="", yaxis_title="", coloraxis_colorbar_title="bits")
                        plot(f, max(300, 26 * len(piv)))

                fi = D.read("feature_importance")
                if not fi.empty:
                    with card():
                        top = fi.sort_values("importance", ascending=False).head(22)
                        f = go.Figure(
                            go.Bar(x=top["importance"], y=top["feature"], orientation="h", marker_color=SERIES[0])
                        )
                        f.update_layout(
                            title=f"GBDT feature importance ({top['backend'].iloc[0]})",
                            yaxis={"autorange": "reversed"},
                            yaxis_title="",
                            xaxis_title="importance",
                        )
                        plot(f, 560)

                abl_path = D.SETTINGS.artifacts_dir / "ablation.csv"
                if abl_path.exists():
                    a = pd.read_csv(abl_path)
                    with card():
                        f = go.Figure(
                            go.Bar(
                                x=a["delta_log_loss"],
                                y=a["group_removed"],
                                orientation="h",
                                marker_color=[
                                    STATUS["critical"] if v > 0.002 else STATUS["good"] if v < -0.0005 else INK["muted"]
                                    for v in a["delta_log_loss"]
                                ],
                                text=[f"{v:+.4f}" for v in a["delta_log_loss"]],
                                textposition="outside",
                                cliponaxis=False,
                            )
                        )
                        f.add_vline(x=0, line_color=INK["axis"])
                        f.update_layout(
                            title="Feature-group ablation — Δ log-loss when the group is removed (right = helps)",
                            xaxis_title="Δ log-loss",
                            yaxis_title="",
                        )
                        plot(f, 300)
        finally:
            _state["rendering"] = False

    label_sel.on_value_change(lambda e: render())
    strat_sel.on_value_change(lambda e: render())
    render()


# ============================================================================= 4. suggestions & approval
def build_suggestions() -> None:
    section_title("Proposals waiting at the human approval gate")
    banner(
        "LangGraph: scout → features → inference → odds → edge detector → risk manager → "
        "<b>interrupt</b> → human → paper-trade log. Future-fixture quotes are <b>synthetic</b> "
        "(Elo-derived) until a live odds provider is configured. This step cannot be skipped or "
        "automated — nothing here can place a real bet.",
        "warn",
    )
    pending = D.pending_signals()
    props = pending.get("proposals", [])
    if not props:
        banner("No pending proposals. Run <code>uv run pitch-edge signals</code> and reload.", "warn")
    else:
        dfp = pd.DataFrame(props)
        show = dfp[
            [
                "date",
                "home_team",
                "away_team",
                "outcome",
                "model_probability",
                "market_probability",
                "edge",
                "decimal_odds",
                "stake",
                "bookmaker",
            ]
        ].copy()
        for col in ("model_probability", "market_probability", "edge"):
            show[col] = (show[col] * 100).round(1)
        with card(title="Today's proposals"):
            frame_table(show.reset_index().rename(columns={"index": "row"}), row_key="row")

        with card():
            fig = go.Figure()
            labels = dfp["home_team"] + " v " + dfp["away_team"]
            fig.add_trace(
                go.Bar(x=labels, y=dfp["market_probability"], name="market (no-vig)", marker_color=INK["muted"])
            )
            fig.add_trace(go.Bar(x=labels, y=dfp["model_probability"], name="model", marker_color=SERIES[0]))
            fig.update_layout(
                barmode="group",
                title="Model vs market probability",
                yaxis_tickformat=".0%",
                xaxis_title="",
                yaxis_title="",
            )
            plot(fig, 320)

        with card(title="Human approval — required"):
            ui.label(
                "Pick which proposals to approve as paper trades. Nothing is ever placed with a bookmaker or "
                "exchange; this only logs a row for later CLV review."
            ).style(f"color:{INK['secondary']}; font-size:.85rem;")
            approver = ui.input("Your name (required)").classes("w-64")
            checks: dict[str, ui.checkbox] = {}
            with ui.column().classes("w-full gap-1"):
                for r in dfp.itertuples():
                    key = f"{r.match_id}|{r.outcome}"
                    checks[key] = ui.checkbox(
                        f"{r.home_team} vs {r.away_team} · {r.outcome} @ {r.decimal_odds} "
                        f"(model {r.model_probability:.1%} vs market {r.market_probability:.1%}, edge {r.edge:+.1%})"
                    )
            record_btn = ui.button("Record decisions (paper only)").props("color=primary")

            def on_record() -> None:
                if not approver.value:
                    ui.notify("Enter your name — this is the human gate.", type="warning")
                    return
                approved = {k for k, cb in checks.items() if cb.value}
                n = D.record_decisions(props, approved, approver.value, pending.get("thread_id"))
                D.clear_pending_signals(pending.get("thread_id"))
                ui.notify(
                    f"Recorded {n} approved / {len(props) - n} rejected paper trades. Nothing was placed anywhere.",
                    type="positive",
                )
                record_btn.disable()

            record_btn.on_click(on_record)

    section_title("Paper-trade log")
    pt = D.read("paper_trades")
    with card():
        if not pt.empty:
            frame_table(pt.sort_values("approved_at", ascending=False).head(50), row_key="trade_id")
        else:
            ui.label("No paper trades recorded yet.").style(f"color:{INK['muted']};")


# ============================================================================= 5. ask the system
def build_ask() -> None:
    section_title("Ask the system — grounded, cited answers")
    from pitch_edge.config import get_settings

    settings = get_settings()
    banner(
        "Hybrid retrieval (dense ∪ BM25, reciprocal-rank fusion, cross-encoder rerank) over match reports, "
        f"model explanations, event-data summaries and news. Generation: <b>{settings.anthropic_model}</b> when a "
        "key is set, otherwise a deterministic template — the layout below is identical either way. The LLM only "
        "explains; every number is checked against the sources and it never produces a probability.",
    )
    examples = [
        "Why does the model favour the away side in Liverpool vs Arsenal?",
        "Scouting report on Bayern Munich",
        "What happened in Real Madrid vs Barcelona last season?",
        "Which teams have injury news this week?",
    ]
    with card():
        q = ui.input("Question", placeholder=examples[0]).classes("w-full")
        ui.label("Try: " + " · ".join(examples)).style(f"color:{INK['muted']}; font-size:.8rem;")
        ask_btn = ui.button("Ask", icon="search").props("color=primary")
        result = ui.column().classes("w-full gap-2")

    async def on_ask() -> None:
        question = q.value.strip()
        if not question:
            return
        result.clear()
        with result:
            ui.spinner(size="lg")
            ui.label("retrieving…").style(f"color:{INK['muted']};")
        ask_btn.disable()
        try:
            answer, hits, backend_count = await run.io_bound(_ask_sync, question)  # type: ignore[misc]
        finally:
            ask_btn.enable()
        result.clear()
        with result:
            # doc ids / feature names are full of underscores (e.g. `match:cfmd_D1_...`); escape them so
            # Markdown doesn't read pairs of underscores as emphasis and mangle a citation mid-word.
            ui.markdown(answer.text.replace("_", "\\_"))
            badge = (
                "all figures matched a source"
                if answer.verified
                else f"unmatched figures: {', '.join(answer.unverified_numbers)}"
            )
            ui.label(
                f"generator = {answer.backend}{' · ' + answer.model if answer.model else ''} · index has "
                f"{backend_count:,} docs · {badge} · citations: {', '.join(answer.citations) or '—'}"
            ).style(f"color:{INK['muted']}; font-size:.78rem;")
            with ui.expansion("Retrieved documents (with retrieval provenance)").classes("w-full"):
                for h in hits:
                    prov = " + ".join(h.sources) + (
                        f" · rerank {h.rerank_score:.2f}" if h.rerank_score is not None else ""
                    )
                    with ui.column().classes("gap-0"):
                        ui.label(f"{h.document.doc_id} · {prov}").style(
                            f"color:{INK['secondary']}; font-size:.8rem; font-weight:600;"
                        )
                        ui.label(h.document.text).style(f"color:{INK['muted']}; font-size:.82rem;")

    ask_btn.on_click(on_ask)
    q.on("keydown.enter", on_ask)


def _ask_sync(question: str):
    from pitch_edge.rag.generate import GroundedGenerator
    from pitch_edge.rag.index import VectorIndex

    idx = VectorIndex()
    hits = idx.query_hits(question, k=6)
    answer = GroundedGenerator().answer(question, [h.document for h in hits])
    return answer, hits, idx.count()


# ============================================================================= app shell
VIEWS = [
    ("Overview", "insights", build_overview),
    ("Data universe", "database", build_data_universe),
    ("Backtest & calibration", "query_stats", build_backtest),
    ("Suggestions & approval", "fact_check", build_suggestions),
    ("Ask the system", "chat", build_ask),
]


@ui.page("/")
def index() -> None:
    ui.add_head_html(
        f"<style>"
        f"body {{ background:{PAGE_BG} !important; }}"
        f".q-tab {{ text-transform:none; font-size:.92rem; }}"
        f".q-page {{ background:{PAGE_BG}; }}"
        f"::-webkit-scrollbar {{ width:10px; height:10px; }}"
        f"::-webkit-scrollbar-thumb {{ background:#333; border-radius:6px; }}"
        f"</style>"
    )
    ui.dark_mode().enable()
    ui.colors(primary=SERIES[0])

    with (
        ui.header()
        .classes("items-center justify-between")
        .style(f"background:{PAGE_BG}; border-bottom:1px solid {BORDER}; padding:10px 20px;")
    ):
        with ui.row().classes("items-center gap-3"):
            ui.label("PITCH-EDGE").style("font-size:1.25rem; font-weight:700; letter-spacing:-.01em;")
            for text in ("no auto-betting", "paper bankroll", "ToS-clean data"):
                ui.label(text).style(
                    "background:#17304f; color:#86b6ef; border-radius:999px; padding:2px 10px; font-size:.72rem;"
                )
        ui.label("Refresh: uv run pitch-edge refresh").style(f"color:{INK['muted']}; font-size:.78rem;")

    with ui.tabs().classes("w-full").style(f"background:{PAGE_BG}; border-bottom:1px solid {BORDER};") as tabs:
        tab_objs = [ui.tab(name, icon=icon) for name, icon, _ in VIEWS]

    with ui.tab_panels(tabs, value=tab_objs[0]).classes("w-full").style(f"background:{PAGE_BG};"):
        for (_name, _icon, builder), tab in zip(VIEWS, tab_objs, strict=True):
            with ui.tab_panel(tab).classes("w-full gap-3").style("max-width:1400px; margin:0 auto;"):
                with ui.column().classes("w-full gap-3"):
                    try:
                        builder()
                    except Exception as exc:  # noqa: BLE001 - never blank-page the whole app on one view's error
                        banner(f"This view failed to render: <code>{exc}</code>", "warn")


def main() -> None:
    port = int(os.environ.get("PITCH_EDGE_DASHBOARD_PORT", "8501"))
    ui.run(title="PITCH-EDGE", port=port, reload=False, show=False, favicon="⚽", dark=True)


if __name__ in {"__main__", "__mp_main__"}:
    # NiceGUI's browser-less test harness (`nicegui.testing.User`) loads this file with
    # `runpy.run_path(..., run_name="__main__")` and relies on this guard calling `ui.run()` to
    # register routes/config; `ui.run()` itself detects the simulation and returns before binding
    # a real socket (see `helpers.is_user_simulation()`), so this is safe under both `pitch-edge
    # serve` and the test harness.
    main()
