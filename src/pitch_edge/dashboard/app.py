"""PITCH-EDGE — single-page Streamlit dashboard.

Every figure is read from warehouse tables or artifacts written by the pipeline; the app never
trains a model, never fetches odds, and never places anything. Signals stop at a human approval
gate and are logged as paper trades only.

Sections (one page, `st.tabs`): Overview · Match Intel · Data universe · Backtest & calibration · Models & players ·
In-play · Signals & approval · Ask the system · Alternative data & Sweden · Pipeline health.

Match Intel renders dossiers written by `pitch-edge intel` (a warehouse table); it never builds one here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pitch_edge.config import get_settings
from pitch_edge.dashboard.theme import CSS, DIVERGING, INK, OUTCOME, SERIES, STATUS, install_template
from pitch_edge.data.storage import Warehouse
from pitch_edge.models.calibration import reliability_curve

st.set_page_config(page_title="PITCH-EDGE", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")
install_template()
st.markdown(CSS, unsafe_allow_html=True)
SETTINGS = get_settings()
SETTINGS.ensure_dirs()
OUTCOMES = ("home", "draw", "away")
BIG5 = ["E0", "D1", "SP1", "I1", "F1"]


# ============================================================================= data access
DB = str(SETTINGS.db_path)


@st.cache_data(ttl=300, show_spinner=False)
def _read_cached(db: str, table: str, where: str | None) -> pd.DataFrame:
    """Short-lived read-only connection per query so the app never blocks a pipeline writer.
    `db` is an explicit argument so it is always part of the cache key (no stale frames across warehouses)."""
    try:
        with Warehouse(db, read_only=True) as wh:
            return wh.read(table, where)
    except Exception:  # noqa: BLE001 - table may not exist yet / writer holds the lock
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def _query_cached(db: str, sql: str) -> pd.DataFrame:
    try:
        with Warehouse(db, read_only=True) as wh:
            return wh.query(sql)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _read(table: str, where: str | None = None) -> pd.DataFrame:
    return _read_cached(DB, table, where)


def _query(sql: str) -> pd.DataFrame:
    return _query_cached(DB, sql)


def _artifact(name: str) -> dict:
    p = SETTINGS.artifacts_dir / name
    return json.loads(p.read_text()) if p.exists() else {}


def _latest_run(summ: pd.DataFrame) -> str | None:
    rid_path = SETTINGS.artifacts_dir / "run_id.txt"
    if rid_path.exists():
        rid = rid_path.read_text().strip()
        if not summ.empty and (summ["run_id"] == rid).any():
            return rid
    return summ.sort_values("run_id")["run_id"].iloc[-1] if not summ.empty else None


def _fmt(n, digits: int = 0) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—"
    return f"{n:,.{digits}f}"


def note(text: str, kind: str = "note") -> None:
    st.markdown(f'<div class="pe-{kind}">{text}</div>', unsafe_allow_html=True)


def _bits(model_ll: float, market_ll: float) -> float:
    return (market_ll - model_ll) / np.log(2)


# ============================================================================= load
universe = _artifact("data_universe.json")
summ_all = _read("backtest_summaries")
RUN = _latest_run(summ_all)
summ_run = summ_all[summ_all["run_id"] == RUN] if RUN else summ_all
bets_all = _read("backtest_bets")
bets_run = (
    bets_all[bets_all["backtest_id"].str.contains(f":{RUN}", regex=False)] if RUN and not bets_all.empty else bets_all
)
preds_all = _read("model_predictions")
preds_run = preds_all[preds_all["run_id"] == RUN] if RUN and not preds_all.empty else preds_all
if not bets_run.empty:
    bets_run = bets_run.assign(
        label=bets_run["backtest_id"].str.split(":").str[0],
        model=bets_run["backtest_id"].str.split(":").str[1],
        strategy=bets_run["outcome"].str.split("|").str[1],
        side=bets_run["outcome"].str.split("|").str[0],
    )
pending_path = SETTINGS.artifacts_dir / "pending_signals.json"
pending = json.loads(pending_path.read_text()) if pending_path.exists() else {"thread_id": None, "proposals": []}
health = pd.DataFrame(universe.get("sources", [])) if universe else pd.DataFrame()

# ============================================================================= sidebar
with st.sidebar:
    st.markdown("### ⚽ PITCH-EDGE")
    st.caption("Football intelligence & market-edge research")
    st.markdown(
        '<span class="pe-pill">no auto-betting</span><span class="pe-pill">paper bankroll</span><span class="pe-pill">ToS-clean data</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    if universe:
        st.markdown(
            f"**Warehouse** · {_fmt(universe.get('matches'))} matches · {_fmt(universe.get('leagues'))} divisions"
        )
        st.markdown(f"**Data as of** · {str(universe.get('date_range', ['', ''])[1])}")
        st.markdown(f"**Universe built** · {str(universe.get('generated_at', ''))[:16].replace('T', ' ')}")
    st.markdown(f"**Backtest run** · `{RUN or '—'}`")
    if not health.empty:
        bad = int((health["last_status"] != "ok").sum()) if "last_status" in health else 0
        st.markdown(f"**Sources** · {len(health)} tracked, {bad} failing")
    st.markdown("---")
    st.markdown(
        "**How to read this app**\n\n"
        "1. *Backtest* — is any model more informative than the closing price? (bits vs market, CLV)\n"
        "2. *Calibration* — can the probabilities be staked on at all?\n"
        "3. *Signals* — what the pipeline proposes today, waiting for a human.\n\n"
        "Every number traces to a walk-forward backtest at real, vig-inclusive prices."
    )
    st.markdown("---")
    from pitch_edge.keys import status_summary

    KEYS = status_summary()
    if not st.session_state.get("hide_key_nudge"):
        nxt = KEYS["next_unlock"]
        st.markdown(
            f"**Optional keys** · {KEYS['n_api_set']} of {KEYS['n_api_total']} unlocked · "
            + (
                "Anthropic key active (Fable 5.1 explanations)"
                if KEYS["anthropic_active"]
                else "explainer on the offline template"
            )
            + (
                f"\n\nAdd `{nxt['env']}` for {nxt['unlocks'].split(' — ')[0].split(':')[0]} → `uv run pitch-edge setup`"
                if nxt
                else ""
            )
        )
        st.caption("Keys are entered in the terminal and stored in a git-ignored .env — never through this page.")
        if st.button("dismiss", key="dismiss_keys"):
            st.session_state["hide_key_nudge"] = True
    st.markdown("---")
    st.caption("Refresh: `uv run pitch-edge refresh` · Docs: README, CASE_STUDY, MODEL_CARDS, API_KEYS")

# ============================================================================= hero
st.title("PITCH-EDGE")
st.markdown(
    "Free data → Dixon-Coles / gradient boosting / GRU & Transformer / GNN → walk-forward backtests scored against the "
    "closing line → a LangGraph pipeline that **stops at a human approval gate**."
)
if universe:
    c = st.columns(6)
    c[0].metric("Matches", _fmt(universe.get("matches")))
    c[1].metric("Divisions", _fmt(universe.get("leagues")))
    c[2].metric("Countries", _fmt(universe.get("countries")))
    c[3].metric("Teams", _fmt(universe.get("teams")))
    c[4].metric("Odds rows", _fmt(universe.get("odds")))
    c[5].metric(
        "Player-data rows",
        _fmt(
            _query("SELECT count(*) AS n FROM tm_appearances")["n"].iloc[0]
            if not _query("SELECT count(*) AS n FROM tm_appearances").empty
            else 0
        ),
    )
else:
    st.info("No data yet — run `uv run pitch-edge refresh` (or `ingest` → `backtest` → `artifacts`).")

tabs = st.tabs(
    [
        "Overview",
        "Match Intel",
        "Data universe",
        "Backtest & calibration",
        "Models & players",
        "In-play",
        "Signals & approval",
        "Ask the system",
        "Alt data & Sweden",
        "Pipeline health",
    ]
)

# ============================================================================= 0 overview
with tabs[0]:
    st.subheader("What the latest run says")
    if summ_run.empty:
        st.info("Run `uv run pitch-edge backtest` to populate this view.")
    else:
        main = summ_run[summ_run.get("label", "main") == "main"] if "label" in summ_run else summ_run
        cal = main.drop_duplicates("model")[
            ["model", "multiclass_log_loss", "market_multiclass_log_loss", "n_predictions"]
        ].copy()
        cal["bits_vs_market"] = _bits(cal["multiclass_log_loss"], cal["market_multiclass_log_loss"])
        best = cal.sort_values("bits_vs_market", ascending=False).iloc[0]
        q = main[main["strategy"] == "kelly_quarter"].set_index("model")
        k = st.columns(4)
        k[0].metric(
            "Best model vs closing price",
            f"{best['bits_vs_market']:+.3f} bits",
            help="Positive = model more informative than the no-vig closing price. Negative = the price knows more.",
        )
        k[1].metric("Out-of-sample predictions", _fmt(cal["n_predictions"].max()))
        clv_src = main["closing_price_source"].iloc[0] if "closing_price_source" in main else ""
        k[2].metric(
            "Mean CLV (¼-Kelly)",
            "n/a — no closing line"
            if clv_src == "bet_price_no_closing_available"
            else f"{q['mean_clv_pct'].max():+.2%}",
        )
        k[3].metric("Proposals at the gate", len(pending.get("proposals", [])))
        verdict = (
            "no model beats the market on this run"
            if best["bits_vs_market"] < 0
            else f"{best['model']} beats the market by {best['bits_vs_market']:.3f} bits"
        )
        price_src = main["bet_price_source"].iloc[0] if "bet_price_source" in main else "?"
        price_note = (
            "market-average closing price — CLV is zero by construction until football-data.co.uk's early Pinnacle line is back"
            if clv_src == "bet_price_no_closing_available"
            else "early Pinnacle"
        )
        note(
            f"<b>Verdict:</b> {verdict}. Bets are priced at <b>{price_src}</b> ({price_note}). "
            "Read <i>bits vs market</i> first, ROI last.",
            "warn" if best["bits_vs_market"] < 0 else "note",
        )
        c1, c2 = st.columns([3, 2])
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
            title="Model race — information vs the closing price (bits per match)",
            height=300,
            xaxis_title="bits vs market",
            yaxis_title="",
            showlegend=False,
            hovermode="closest",
        )
        c1.plotly_chart(fig, width="stretch")
        by_league = pd.DataFrame(universe.get("by_league", []))
        if not by_league.empty:
            by_league["tier"] = np.select(
                [
                    by_league["league_code"].str.startswith("SWE"),
                    by_league["league_code"].isin(BIG5),
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
            fig2.update_layout(height=300, showlegend=False, xaxis_title="", yaxis_title="matches", hovermode="closest")
            c2.plotly_chart(fig2, width="stretch")
        st.markdown(
            "**Where next** — the case study (`CASE_STUDY.md`) documents the ablation, the developing-market slice and the deep-learning entry; `API_KEYS.md` lists the free keys that add lineups, injuries and live prices."
        )

# ============================================================================= 1 match intel
with tabs[1]:
    st.subheader("Match Intel — the pre-match dossier")
    note(
        "Built by <code>uv run pitch-edge intel &lt;home&gt; &lt;away&gt; [--date …]</code> and stored as a versioned warehouse row. "
        "Every line is a number from a named table/report or a cited RAG document, checked by <code>verify_citations</code>; "
        "section 9 lists what is <b>missing for this fixture</b>. Nothing here is a betting instruction."
    )
    doss = _read("dossiers")
    if doss.empty:
        st.info("No dossiers yet — run `uv run pitch-edge intel Brighton Forest` (or any two clubs) and reload.")
    else:
        latest = doss.sort_values("version").groupby("fixture_key").tail(1).sort_values("match_date", ascending=False)
        intel_labels = {
            r.fixture_key: f"{r.home_team} vs {r.away_team} · {str(r.match_date)[:10]} · {r.fixture_source} · v{r.version}"
            for r in latest.itertuples()
        }
        pick = st.selectbox(
            "Fixture", options=list(intel_labels), format_func=lambda k: intel_labels[k], key="intel_pick"
        )
        row = latest[latest["fixture_key"] == pick].iloc[0]
        payload = json.loads(row["json"])
        c = st.columns(5)
        c[0].metric("Citation check", "✅ verified" if bool(row["verified"]) else "❌ unverified")
        c[1].metric("Claims", int(row["n_claims"]))
        c[2].metric("Gaps", int(row["n_gaps"]))
        c[3].metric("Lineup", "confirmed" if bool(row["confirmed_lineup"]) else "provisional")
        c[4].metric("Versions", int((doss["fixture_key"] == pick).sum()))
        if not bool(row["verified"]):
            note(
                f"Unmatched figures: {row['unverified_numbers']} — this dossier failed the citation check and should not be relied on.",
                "warn",
            )
        gaps = next((sct for sct in payload["sections"] if sct["key"] == "gaps"), None)
        if gaps:
            items = [cl["text"] for cl in gaps["claims"] if cl["status"] == "missing"]
            if items:
                note("<b>What would change the answer</b><br/>" + "<br/>".join(f"• {t}" for t in items), "warn")
        for i, sct in enumerate(payload["sections"], 1):
            flag = "" if sct["status"] == "ok" else f" · {sct['status'].upper()}"
            with st.expander(f"{i}. {sct['title']}{flag}", expanded=i in (1, 3, 9)):
                for cl in sct["claims"]:
                    st.markdown(f"- {cl['text']} " + " ".join(f"`{s_}`" for s_ in cl["sources"]))
        series = pd.DataFrame(payload.get("market_series", []))
        if not series.empty:
            series["ts"] = pd.to_datetime(series["ts"])
            f = go.Figure()
            for (venue, outcome), g in series.groupby(["venue", "outcome"]):
                f.add_trace(
                    go.Scatter(
                        x=g["ts"],
                        y=g["prob"],
                        mode="lines+markers",
                        name=f"{venue} · {outcome}",
                        line={
                            "color": OUTCOME[outcome],
                            "dash": "solid" if venue == series["venue"].iloc[0] else "dot",
                        },
                    )
                )
            f.update_layout(
                height=320,
                title="Prediction-market implied probability over time (stored snapshots)",
                yaxis_tickformat=".0%",
                xaxis_title="",
                yaxis_title="",
            )
            st.plotly_chart(f, width="stretch")
        versions = doss[doss["fixture_key"] == pick].sort_values("version", ascending=False)
        if len(versions) > 1:
            from pitch_edge.intel.dossier import diff_dossiers, render_diff

            d = diff_dossiers(versions.iloc[1]["json"], versions.iloc[0]["json"])
            with st.expander(f"What changed since v{versions.iloc[1]['version']}", expanded=d["changed"]):
                st.code(render_diff(d))
        if payload.get("narrative"):
            st.markdown(f"**Analyst narrative** ({payload.get('narrative_backend')}, citation-verified)")
            st.markdown(payload["narrative"])
        with st.expander("Plain-text dossier"):
            st.code(row["markdown"])

# ============================================================================= 1 data universe
with tabs[2]:
    st.subheader("What is loaded, from where")
    st.caption(
        "football-data.co.uk (spine when reachable), the open Club-Football-Match-Data 2000-2025 compilation (38 divisions, Elo, form, market/max odds), "
        "openfootball, StatsBomb Open Data, Club Elo, Wikidata venues, Open-Meteo weather, BBC/Guardian/Sky/ESPN + SVT/Sportbladet/Expressen/DN/GP RSS, "
        "Polymarket + Kalshi, the open Transfermarkt extract, TheSportsDB."
    )
    by_league = pd.DataFrame(universe.get("by_league", []))
    if not by_league.empty:
        by_league["tier"] = np.select(
            [
                by_league["league_code"].str.startswith("SWE"),
                by_league["league_code"].isin(BIG5),
                by_league["league_code"].str.len() == 3,
            ],
            ["Sweden (focus)", "Big-5 top flight", "Developing / under-covered"],
            default="Other European",
        )
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
            hover_data=["country", "first_date", "last_date", "with_stats", "with_elo"],
            labels={"n": "matches", "league_code": "division"},
            title="Matches per division",
        )
        fig.update_layout(height=400, legend_title="", hovermode="closest")
        st.plotly_chart(fig, width="stretch")
        c1, c2 = st.columns([3, 2])
        by_season = pd.DataFrame(universe.get("by_season", []))
        if not by_season.empty:
            f = px.area(by_season, x="season", y="n", title="Matches per season", color_discrete_sequence=[SERIES[0]])
            f.update_layout(height=300, xaxis_tickangle=-60, yaxis_title="matches", xaxis_title="")
            c1.plotly_chart(f, width="stretch")
        ob = pd.DataFrame(universe.get("odds_by_book", []))
        if not ob.empty:
            ob["label"] = ob["bookmaker"] + np.where(ob["is_closing"], " (closing)", "") + " · " + ob["market"]
            f = px.bar(
                ob,
                x="n",
                y="label",
                orientation="h",
                title="Odds rows by bookmaker · market",
                color_discrete_sequence=[SERIES[0]],
            )
            f.update_layout(height=300, yaxis_title="", xaxis_title="rows", hovermode="closest")
            c2.plotly_chart(f, width="stretch")
    venues = _read("venues")
    if not venues.empty:
        f = px.scatter_geo(
            venues,
            lat="lat",
            lon="lon",
            hover_name="team",
            color="geo_source",
            scope="europe",
            color_discrete_sequence=[SERIES[0], SERIES[1]],
            title=f"{len(venues)} geocoded home grounds — drive the weather & travel features",
        )
        f.update_layout(height=480, legend_title="")
        st.plotly_chart(f, width="stretch")
    counts = pd.DataFrame(
        [
            {"table": k, "rows": universe.get(k, 0)}
            for k in (
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
    tm = _query(
        "SELECT 'tm_appearances' AS t, count(*) n FROM tm_appearances UNION ALL SELECT 'tm_game_events', count(*) FROM tm_game_events UNION ALL SELECT 'tm_players', count(*) FROM tm_players UNION ALL SELECT 'tm_player_valuations', count(*) FROM tm_player_valuations"
    )
    if not tm.empty:
        counts = pd.concat([counts, tm.rename(columns={"t": "table", "n": "rows"})], ignore_index=True)
    st.dataframe(counts, hide_index=True, width="stretch")

# ============================================================================= 2 backtest & calibration
with tabs[3]:
    st.subheader("Walk-forward backtest")
    if summ_run.empty:
        st.info("No backtest yet — run `uv run pitch-edge backtest`.")
    else:
        labels = sorted(summ_run["label"].unique()) if "label" in summ_run else ["main"]
        cA, cB = st.columns([1, 1])
        label = cA.selectbox(
            "Slice",
            labels,
            index=labels.index("main") if "main" in labels else 0,
            help="main = 11 European divisions since 2015 · developing = 16 under-covered markets since 2013",
        )
        summ = summ_run[summ_run["label"] == label] if "label" in summ_run else summ_run
        bets = bets_run[bets_run["label"] == label] if not bets_run.empty and "label" in bets_run else bets_run
        strat = cB.selectbox("Staking", sorted(summ["strategy"].unique()), index=0)
        src = (
            summ[["bet_price_source", "closing_price_source"]].drop_duplicates().iloc[0]
            if "bet_price_source" in summ
            else None
        )
        if src is not None and src["closing_price_source"] == "bet_price_no_closing_available":
            note(
                f"Bets priced at <b>{src['bet_price_source']}</b> (market-average price). No separate closing line in the current spine, so <b>CLV is 0 by construction</b> — "
                "compare log-loss with the market and treat ROI as noise.",
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
        cal["bits_vs_market"] = _bits(cal["multiclass_log_loss"], cal["market_multiclass_log_loss"])
        view = summ[summ["strategy"] == strat].merge(cal[["model", "bits_vs_market"]], on="model")
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
        st.dataframe(
            view[cols].sort_values("bits_vs_market", ascending=False).round(4), hide_index=True, width="stretch"
        )

        if not bets.empty:
            sub = bets[bets["strategy"] == strat].sort_values("date")
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
            fig.update_layout(height=380, title=f"Paper bankroll — {strat}", yaxis_title="bankroll", xaxis_title="")
            st.plotly_chart(fig, width="stretch")
            m_sel = st.selectbox("Drill into", sorted(sub["model"].unique()))
            g = sub[sub["model"] == m_sel].copy()
            g["month"] = pd.to_datetime(g["date"]).dt.to_period("M").astype(str)
            c1, c2, c3 = st.columns(3)
            monthly = g.groupby("month", as_index=False)["profit"].sum()
            f = go.Figure(
                go.Bar(
                    x=monthly["month"],
                    y=monthly["profit"],
                    marker_color=[STATUS["good"] if v >= 0 else STATUS["critical"] for v in monthly["profit"]],
                )
            )
            f.update_layout(height=280, title="Monthly P&L", xaxis_title="", yaxis_title="", hovermode="closest")
            c1.plotly_chart(f, width="stretch")
            by_lg = (
                g.groupby("league_code", as_index=False)
                .agg(profit=("profit", "sum"), n=("won", "size"))
                .sort_values("profit")
            )
            f = go.Figure(
                go.Bar(
                    x=by_lg["profit"],
                    y=by_lg["league_code"],
                    orientation="h",
                    marker_color=[STATUS["good"] if v >= 0 else STATUS["critical"] for v in by_lg["profit"]],
                    customdata=by_lg["n"],
                    hovertemplate="%{y}: %{x:.0f} over %{customdata} bets<extra></extra>",
                )
            )
            f.update_layout(height=280, title="P&L by division", xaxis_title="", yaxis_title="", hovermode="closest")
            c2.plotly_chart(f, width="stretch")
            ob = (
                g.groupby(pd.cut(g["bet_odds"], [1, 1.5, 2, 2.5, 3, 4, 6, 12]), observed=True)
                .agg(profit=("profit", "sum"), n=("won", "size"))
                .reset_index()
            )
            ob["band"] = ob["bet_odds"].astype(str)
            f = go.Figure(
                go.Bar(
                    x=ob["band"],
                    y=ob["profit"],
                    marker_color=[STATUS["good"] if v >= 0 else STATUS["critical"] for v in ob["profit"]],
                    customdata=ob["n"],
                    hovertemplate="%{x}: %{y:.0f} over %{customdata} bets<extra></extra>",
                )
            )
            f.update_layout(
                height=280, title="P&L by odds band", xaxis_title="decimal odds", yaxis_title="", hovermode="closest"
            )
            c3.plotly_chart(f, width="stretch")

        st.subheader("Calibration")
        preds = preds_run.copy()
        if not preds.empty and "label" in bets_run and not bets.empty:
            preds = preds[preds["match_id"].isin(set(bets["match_id"])) | preds["model_name"].isin(cal["model"])]
        if preds.empty:
            st.info("No predictions stored for this slice.")
        else:
            c1, c2 = st.columns([2, 3])
            long = cal.melt(
                id_vars="model",
                value_vars=["multiclass_log_loss", "market_multiclass_log_loss"],
                var_name="who",
                value_name="log-loss",
            )
            long["who"] = long["who"].map(
                {"multiclass_log_loss": "model", "market_multiclass_log_loss": "no-vig market"}
            )
            f = px.bar(
                long,
                x="model",
                y="log-loss",
                color="who",
                barmode="group",
                color_discrete_sequence=[SERIES[0], INK["muted"]],
                title="Log-loss: model vs market (lower is better)",
            )
            f.update_layout(height=340, legend_title="", xaxis_title="", hovermode="closest")
            c1.plotly_chart(f, width="stretch")
            m = c2.selectbox("Reliability diagram", sorted(preds["model_name"].unique()), key="calib_model")
            p = preds[preds["model_name"] == m]
            f = go.Figure()
            f.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines", name="perfect", line={"dash": "dash", "color": INK["axis"]}
                )
            )
            for k_, o in enumerate(OUTCOMES):
                mp, ef = reliability_curve(
                    (p["result"] == k_).astype(int).to_numpy(), p[f"p_{o}"].to_numpy(), n_bins=12
                )
                f.add_trace(
                    go.Scatter(x=mp, y=ef, mode="markers+lines", name=o, line={"color": OUTCOME[o]}, marker={"size": 8})
                )
            f.update_layout(
                height=340, xaxis_title="predicted probability", yaxis_title="observed frequency", hovermode="closest"
            )
            c2.plotly_chart(f, width="stretch")
            pm = p.sort_values("date").copy()
            pm["month"] = pd.to_datetime(pm["date"]).dt.to_period("M").astype(str)
            onehot = np.eye(3)[pm["result"].astype(int).to_numpy()]
            pm["model"] = ((pm[["p_home", "p_draw", "p_away"]].to_numpy() - onehot) ** 2).sum(axis=1)
            okm = pm[["mkt_home", "mkt_draw", "mkt_away"]].notna().all(axis=1)
            pm.loc[okm, "market"] = (
                (pm.loc[okm, ["mkt_home", "mkt_draw", "mkt_away"]].to_numpy() - onehot[okm.to_numpy()]) ** 2
            ).sum(axis=1)
            trend = pm.groupby("month")[["model", "market"]].mean().reset_index()
            f = go.Figure()
            f.add_trace(go.Scatter(x=trend["month"], y=trend["model"], name=f"{m}", line={"color": SERIES[0]}))
            f.add_trace(
                go.Scatter(
                    x=trend["month"], y=trend["market"], name="market", line={"color": INK["muted"], "dash": "dot"}
                )
            )
            f.update_layout(
                height=300, title="Monthly multiclass Brier — model vs market", xaxis_title="", yaxis_title="Brier"
            )
            st.plotly_chart(f, width="stretch")
        by_lg = SETTINGS.reports_dir / label / "by_league.csv"
        if by_lg.exists():
            bl = pd.read_csv(by_lg)
            piv = bl.pivot_table(index="league_code", columns="model", values="edge_bits")
            f = px.imshow(
                piv,
                color_continuous_scale=DIVERGING,
                zmin=-0.08,
                zmax=0.08,
                aspect="auto",
                title="Bits vs market by division × model (blue = model ahead, red = market ahead)",
                text_auto=".3f",
            )
            f.update_layout(
                height=max(300, 26 * len(piv)), xaxis_title="", yaxis_title="", coloraxis_colorbar_title="bits"
            )
            st.plotly_chart(f, width="stretch")
        abl = SETTINGS.reports_dir / "ablation.csv"
        if abl.exists():
            a = pd.read_csv(abl)
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
                height=300,
                title="Feature-group ablation — Δ log-loss when the group is removed (right = the group helps)",
                xaxis_title="Δ log-loss",
                yaxis_title="",
                hovermode="closest",
            )
            st.plotly_chart(f, width="stretch")

# ============================================================================= 3 models & players
with tabs[4]:
    st.subheader("Models, features, team strengths")
    fi = _read("feature_importance")
    ts = _read("team_strength")
    c1, c2 = st.columns([2, 3])
    if not fi.empty:
        top = fi.sort_values("importance", ascending=False).head(22)
        f = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h", marker_color=SERIES[0]))
        f.update_layout(
            height=560,
            title=f"GBDT feature importance ({top['backend'].iloc[0]})",
            yaxis={"autorange": "reversed"},
            yaxis_title="",
            xaxis_title="importance",
            hovermode="closest",
        )
        c1.plotly_chart(f, width="stretch")
    if not ts.empty:
        lg = c2.selectbox("League", sorted(ts["league_code"].unique()), key="ts_league")
        t = ts[ts["league_code"] == lg]
        f = px.scatter(
            t,
            x="attack",
            y="defence",
            text="team",
            color_discrete_sequence=[SERIES[0]],
            title=f"Dixon-Coles attack vs defence — {lg} · home adv {t['home_advantage'].iloc[0]:.2f}, ρ {t['rho'].iloc[0]:.3f}",
        )
        f.update_traces(textposition="top center", marker={"size": 9})
        f.update_layout(height=560, hovermode="closest")
        c2.plotly_chart(f, width="stretch")
    st.markdown("**Model cards** — training window, features, leakage checks, calibration, last-fit diagnostics.")
    card_dir = SETTINGS.reports_dir / "main"
    cards = sorted(card_dir.glob("model_card_*.json")) if card_dir.exists() else []
    for extra in ("gnn_card.json", "inplay_card.json"):
        if (SETTINGS.artifacts_dir / extra).exists():
            cards.append(SETTINGS.artifacts_dir / extra)
    cc = st.columns(min(4, max(1, len(cards))))
    for i, path in enumerate(cards):
        with cc[i % len(cc)].expander(path.stem.replace("model_card_", "").replace("_card", "")):
            st.json(json.loads(path.read_text()))

    st.subheader("Player embeddings — GNN over StatsBomb passing networks")
    emb = _read("player_embeddings")
    sim = _read("player_similarity")
    gnn_card = _artifact("gnn_card.json")
    if emb.empty:
        st.info("No embeddings yet — `uv run pitch-edge artifacts` after StatsBomb ingestion.")
    else:
        from sklearn.decomposition import PCA

        ecols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
        X = emb[ecols].to_numpy()
        Z = PCA(n_components=2, random_state=0).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-9))
        plot = emb.assign(pc1=Z[:, 0], pc2=Z[:, 1])
        top_teams = plot["team"].value_counts().head(7).index
        plot["group"] = np.where(plot["team"].isin(top_teams), plot["team"], "Other")
        c1, c2 = st.columns([3, 2])
        f = px.scatter(
            plot,
            x="pc1",
            y="pc2",
            color="group",
            hover_name="player",
            size="n_matches",
            size_max=13,
            color_discrete_sequence=SERIES,
            title=f"{len(plot):,} players · 2-D projection of {len(ecols)}-d embeddings"
            + (
                f" · link-prediction AUROC {gnn_card.get('link_prediction_auroc'):.2f}"
                if gnn_card.get("link_prediction_auroc")
                else ""
            ),
        )
        f.update_layout(height=520, legend_title="", hovermode="closest")
        c1.plotly_chart(f, width="stretch")
        options = sorted(sim["player"].unique()) if not sim.empty else sorted(emb["player"])
        player = c2.selectbox("Players similar to", options, key="sim_player")
        if not sim.empty:
            c2.dataframe(
                sim[sim["player"] == player]
                .sort_values("similarity", ascending=False)[["similar_player", "similar_team", "similarity"]]
                .round(3),
                hide_index=True,
                width="stretch",
            )
        c2.caption(
            "Cosine similarity in embedding space — 'plays a similar role in the passing structure', not 'equally good'."
        )

    st.subheader("Player history — open Transfermarkt extract")
    tm_players = _query(
        "SELECT player_id, name, position, sub_position, date_of_birth, country_of_citizenship FROM tm_players"
    )
    if tm_players.empty:
        st.info("Player tables not loaded — `uv run pitch-edge ingest` (open Transfermarkt dataset, ~210 MB).")
    else:
        q = st.text_input("Player name", placeholder="e.g. Saka, Isak, Gyökeres", key="tm_player")
        if q:
            hits = tm_players[tm_players["name"].str.contains(q, case=False, na=False)].head(10)
            if hits.empty:
                st.warning("No player matched.")
            else:
                tm_hits = hits.set_index("player_id")
                pid_sel = st.selectbox(
                    "Match",
                    list(tm_hits.index),
                    format_func=lambda i: f"{tm_hits.loc[i, 'name']} ({tm_hits.loc[i, 'position']})",
                )
                pid = int(pid_sel if pid_sel is not None else tm_hits.index[0])
                apps = _query(
                    f"SELECT a.date, a.competition_id, a.minutes_played, a.goals, a.assists, a.yellow_cards, a.red_cards, g.home_club_name, g.away_club_name, g.home_club_goals, g.away_club_goals FROM tm_appearances a LEFT JOIN tm_games g USING (game_id) WHERE a.player_id = {pid} ORDER BY a.date DESC"
                )
                vals = _query(
                    f"SELECT date, market_value_in_eur, current_club_name FROM tm_player_valuations WHERE player_id = {pid} ORDER BY date"
                )
                c1, c2 = st.columns(2)
                if not vals.empty:
                    f = px.line(
                        vals,
                        x="date",
                        y="market_value_in_eur",
                        markers=True,
                        color_discrete_sequence=[SERIES[0]],
                        title="Market value (EUR)",
                    )
                    f.update_layout(height=300, xaxis_title="", yaxis_title="")
                    c1.plotly_chart(f, width="stretch")
                if not apps.empty:
                    apps["season"] = pd.to_datetime(apps["date"]).dt.year - (
                        pd.to_datetime(apps["date"]).dt.month < 7
                    ).astype(int)
                    per = apps.groupby("season", as_index=False).agg(
                        goals=("goals", "sum"), assists=("assists", "sum"), minutes=("minutes_played", "sum")
                    )
                    f = px.bar(
                        per,
                        x="season",
                        y=["goals", "assists"],
                        barmode="group",
                        color_discrete_sequence=[SERIES[0], SERIES[1]],
                        title="Goals & assists per season",
                    )
                    f.update_layout(height=300, legend_title="", xaxis_title="", yaxis_title="", hovermode="closest")
                    c2.plotly_chart(f, width="stretch")
                    st.dataframe(apps.head(40), hide_index=True, width="stretch")
        subs = _query(
            "WITH s AS (SELECT game_id, club_name, min(minute) AS first_sub, count(*) AS n FROM tm_game_events WHERE type = 'Substitutions' GROUP BY 1,2) SELECT club_name, count(*) AS games, round(avg(first_sub),1) AS avg_first_sub_minute, round(avg(n),2) AS avg_subs FROM s GROUP BY 1 HAVING count(*) >= 60 ORDER BY 3"
        )
        if not subs.empty:
            sel = pd.concat([subs.head(10), subs.tail(10)])
            f = go.Figure(
                go.Bar(
                    x=sel["avg_first_sub_minute"],
                    y=sel["club_name"],
                    orientation="h",
                    marker_color=[SERIES[1]] * 10 + [SERIES[0]] * 10,
                    customdata=sel["games"],
                    hovertemplate="%{y}: %{x:.1f}' over %{customdata} games<extra></extra>",
                )
            )
            f.update_layout(
                height=460,
                title="Earliest (orange) vs latest (blue) first-substitution clubs — a rotation / fatigue proxy",
                xaxis_title="average minute of first substitution",
                yaxis_title="",
                hovermode="closest",
            )
            st.plotly_chart(f, width="stretch")

# ============================================================================= 4 in-play
with tabs[5]:
    st.subheader("In-play win probability — GRU over 5-minute state vectors")
    paths = _read("inplay_paths")
    if paths.empty:
        st.info("No in-play paths yet — `uv run pitch-edge artifacts`.")
    else:
        if "home_team" in paths:
            paths["label"] = (
                paths["home_team"].astype(str)
                + " "
                + paths["home_goals"].fillna(0).astype(int).astype(str)
                + "–"
                + paths["away_goals"].fillna(0).astype(int).astype(str)
                + " "
                + paths["away_team"].astype(str)
                + " · "
                + paths["league"].astype(str)
            )
        else:
            paths["label"] = paths["statsbomb_match_id"].astype(str)
        choice = st.selectbox("Match", sorted(paths["label"].unique()))
        g = paths[paths["label"] == choice].sort_values("minute")
        f = go.Figure()
        for o in OUTCOMES:
            f.add_trace(go.Scatter(x=g["minute"], y=g[o], name=f"{o} · GRU", line={"color": OUTCOME[o], "width": 2.5}))
            f.add_trace(
                go.Scatter(
                    x=g["minute"],
                    y=g[f"base_{o}"],
                    name=f"{o} · logistic",
                    line={"color": OUTCOME[o], "dash": "dot", "width": 1.5},
                    opacity=0.7,
                )
            )
        f.update_layout(height=440, yaxis={"range": [0, 1], "title": "probability"}, xaxis_title="minute", title=choice)
        st.plotly_chart(f, width="stretch")
        note(
            "State = score difference, xG difference, shots in the last 10', minute, Elo prior. No Betfair in-play line is available for a market comparison — that gap is stated in the model card."
        )

# ============================================================================= 5 signals
with tabs[6]:
    st.subheader("Proposals waiting at the human approval gate")
    note(
        "LangGraph: scout → features → inference → odds → edge detector → risk manager → <b>interrupt</b> → human → paper-trade log. "
        "Future-fixture quotes are <b>synthetic</b> (Elo-derived, labelled) until a live odds provider is configured (see API_KEYS.md); edge evidence lives in the backtest tab.",
        "warn",
    )
    props = pd.DataFrame(pending.get("proposals", []))
    if props.empty:
        st.info("No pending proposals. Run `uv run pitch-edge signals` and reload.")
    else:
        show = props[
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
        st.dataframe(show, width="stretch", hide_index=True)
        f = go.Figure()
        f.add_trace(
            go.Bar(
                x=props["home_team"] + " v " + props["away_team"],
                y=props["market_probability"],
                name="market (no-vig)",
                marker_color=INK["muted"],
            )
        )
        f.add_trace(
            go.Bar(
                x=props["home_team"] + " v " + props["away_team"],
                y=props["model_probability"],
                name="model",
                marker_color=SERIES[0],
            )
        )
        f.update_layout(
            barmode="group",
            height=300,
            title="Model vs market probability for each proposal",
            yaxis_tickformat=".0%",
            xaxis_title="",
            yaxis_title="",
            hovermode="closest",
        )
        st.plotly_chart(f, width="stretch")
        approver = st.text_input("Your name — required; this is the human gate", key="approver")
        chosen = st.multiselect(
            "Approve which proposals (paper only)?", options=[f"{r.match_id}|{r.outcome}" for r in props.itertuples()]
        )
        if st.button("Record decisions", type="primary", disabled=not approver):
            wh_rw = Warehouse(SETTINGS.db_path)
            now = datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")
            rows = [
                {
                    **r._asdict(),
                    "trade_id": f"{pending['thread_id']}:{r.match_id}|{r.outcome}",
                    "status": "approved_paper" if f"{r.match_id}|{r.outcome}" in chosen else "rejected",
                    "approved_by": approver,
                    "approved_at": now,
                    "run_id": pending["thread_id"],
                }
                for r in props.itertuples()
            ]
            wh_rw.upsert("paper_trades", pd.DataFrame(rows).drop(columns=["Index"], errors="ignore"))
            wh_rw.close()
            pending_path.write_text(json.dumps({"thread_id": pending["thread_id"], "proposals": []}))
            st.success(
                f"Recorded {len(chosen)} approved / {len(rows) - len(chosen)} rejected paper trades. Nothing was placed anywhere."
            )
            st.cache_data.clear()
    pt = _read("paper_trades")
    st.markdown("**Paper-trade log**")
    st.dataframe(
        pt.sort_values("approved_at", ascending=False).head(50) if not pt.empty else pt,
        width="stretch",
        hide_index=True,
    )

# ============================================================================= 6 ask
with tabs[7]:
    st.subheader("Ask the system — grounded, cited answers")
    note(
        "Hybrid retrieval (Chroma dense ∪ BM25, reciprocal-rank fusion, cross-encoder rerank) over match reports, model explanations, StatsBomb summaries and news. "
        f"Generation: <b>{SETTINGS.anthropic_model}</b> when a key is set, otherwise a deterministic template. The LLM only explains — every number is checked against the sources and it never produces a probability."
    )
    examples = [
        "Why does the model favour the away side in Liverpool vs Arsenal?",
        "Scouting report on Bayern Munich",
        "What happened in Real Madrid vs Barcelona last season?",
        "Which teams have injury news this week?",
        "Vad hände i Malmö FF mot AIK?",
    ]
    q = st.text_input("Question", placeholder=examples[0], key="rag_q")
    st.caption("Try: " + " · ".join(f"_{e}_" for e in examples))
    eval_path = SETTINGS.reports_dir / "rag_eval.csv"
    if eval_path.exists():
        from pitch_edge.rag.eval import summarize_eval

        ev = summarize_eval(pd.read_csv(eval_path))
        allrow = ev[ev["kind"] == "ALL"].iloc[0]
        st.caption(
            f"Retrieval eval on synthetic QA: hit@1 {allrow['hit_at_1']:.0%} · hit@5 {allrow['hit_at_k']:.0%} · MRR {allrow['mrr']:.2f} (n={int(allrow['n'])})"
        )
    if q:
        from pitch_edge.rag.generate import GroundedGenerator
        from pitch_edge.rag.index import VectorIndex

        with st.spinner("retrieving…"):
            idx = VectorIndex()
            hits = idx.query_hits(q, k=6)
            ans = GroundedGenerator().answer(q, [h.document for h in hits])
        st.markdown(ans.text)
        badge = (
            "✅ every figure matched a source"
            if ans.verified
            else f"⚠️ unmatched figures: {', '.join(ans.unverified_numbers)}"
        )
        st.caption(
            f"generator = {ans.backend}{' · ' + ans.model if ans.model else ''} · index = {idx.backend} ({idx.count():,} docs) · {badge} · citations: {', '.join(ans.citations) or '—'}"
        )
        with st.expander("Retrieved documents (with retrieval provenance)"):
            for h in hits:
                prov = " + ".join(h.sources) + (f" · rerank {h.rerank_score:.2f}" if h.rerank_score is not None else "")
                st.markdown(f"**{h.document.doc_id}** · {prov}\n\n{h.document.text}")

# ============================================================================= 7 alt data & sweden
with tabs[8]:
    st.subheader("Alternative data — what is real, what is stubbed")
    st.markdown(
        "| Signal | Data | Status |\n|---|---|---|\n"
        "| News sentiment velocity | BBC / Guardian / Sky / ESPN + SVT / Sportbladet / Expressen / DN / GP RSS + VADER | **live, real** |\n"
        "| Referee × home-bias | fouls & cards per side from the spine | **real, in models** (inactive on the fallback spine) |\n"
        "| Kickoff weather | Open-Meteo archive at geocoded grounds | **real, in models** |\n"
        "| Travel / fatigue | Wikidata venue distance + fixture congestion | **real, in models** |\n"
        "| Prediction-market prices | Polymarket + Kalshi | real; divergence flagged, not backtested |\n"
        "| Player history / rotation | open Transfermarkt extract | real; in the players view |\n"
        "| Crowd-audio momentum | librosa extractor | **stub** — no licensed audio feed |\n"
        "| Charter-flight tracking | OpenSky | **stub** — no team attribution possible |"
    )
    feats = _read("features")
    news = _read("news_items")
    c1, c2 = st.columns(2)
    if not news.empty and "team" in news:
        from pitch_edge.data.alt.news import sentiment_velocity

        vel = sentiment_velocity(news)
        if not vel.empty:
            vel = vel.sort_values("sent_velocity")
            f = go.Figure(
                go.Bar(
                    x=vel["team"],
                    y=vel["sent_velocity"],
                    marker_color=[STATUS["good"] if v > 0 else STATUS["critical"] for v in vel["sent_velocity"]],
                    customdata=np.stack([vel["n_items"], vel["injury_items"]], axis=1),
                    hovertemplate="%{x}: Δ %{y:+.2f} · %{customdata[0]} items, %{customdata[1]} injury<extra></extra>",
                )
            )
            f.update_layout(
                height=340,
                title="Sentiment velocity by team — Δ mean sentiment, last 48h vs prior 48h (live)",
                xaxis_title="",
                yaxis_title="",
                hovermode="closest",
            )
            c1.plotly_chart(f, width="stretch")
        c2.dataframe(
            news.sort_values("published_at", ascending=False)[
                ["published_at", "feed", "team", "sentiment", "is_injury_news", "title"]
            ].head(20),
            hide_index=True,
            width="stretch",
            height=340,
        )
    if not feats.empty:
        c3, c4, c5 = st.columns(3)
        if "wx_precipitation" in feats and feats["wx_precipitation"].notna().any():
            w = feats.dropna(subset=["wx_precipitation"]).copy()
            w["rain"] = pd.cut(
                w["wx_precipitation"], [-0.01, 0.0, 0.5, 2, 100], labels=["dry", "drizzle", "rain", "heavy"]
            )
            wg = (
                w.groupby("rain", observed=True)
                .agg(goals=("total_goals", "mean"), n=("total_goals", "size"))
                .reset_index()
            )
            f = px.bar(
                wg,
                x="rain",
                y="goals",
                hover_data=["n"],
                color_discrete_sequence=[SERIES[0]],
                title="Goals per match by kickoff precipitation",
                text_auto=".2f",
            )
            f.update_layout(height=300, xaxis_title="", yaxis_title="goals", hovermode="closest")
            c3.plotly_chart(f, width="stretch")
        if "away_travel_km" in feats and feats["away_travel_km"].notna().any():
            t = feats.dropna(subset=["away_travel_km"]).copy()
            t["trip"] = pd.cut(
                t["away_travel_km"],
                [0, 100, 250, 500, 1000, 5000],
                labels=["<100 km", "100–250", "250–500", "500–1000", ">1000"],
            )
            tg = (
                t.groupby("trip", observed=True)
                .agg(away_win=("result", lambda s: (s == 2).mean()), n=("result", "size"))
                .reset_index()
            )
            f = px.bar(
                tg,
                x="trip",
                y="away_win",
                hover_data=["n"],
                color_discrete_sequence=[SERIES[1]],
                title="Away win rate by travel distance",
                text_auto=".0%",
            )
            f.update_layout(height=300, xaxis_title="", yaxis_title="", yaxis_tickformat=".0%", hovermode="closest")
            c4.plotly_chart(f, width="stretch")
        if "ref_home_bias" in feats and feats["ref_home_bias"].notna().any():
            rb = (
                feats.dropna(subset=["ref_home_bias"])
                .groupby("referee")
                .agg(bias=("ref_home_bias", "last"), n=("ref_home_bias", "size"))
                .reset_index()
            )
            rb = rb[rb["n"] >= 30].sort_values("bias")
            sel = pd.concat([rb.head(8), rb.tail(8)])
            f = go.Figure(
                go.Bar(
                    x=sel["bias"],
                    y=sel["referee"],
                    orientation="h",
                    marker_color=[SERIES[1] if v > 0 else SERIES[0] for v in sel["bias"]],
                )
            )
            f.update_layout(
                height=300,
                title="Referee home-bias (share of fouls on the away side − 0.5)",
                xaxis_title="",
                yaxis_title="",
                hovermode="closest",
            )
            c5.plotly_chart(f, width="stretch")
        else:
            c5.info(
                "Referee feature inactive: the fallback spine has no referee names. Returns with football-data.co.uk."
            )
    ms = _read("market_snapshots")
    if not ms.empty:
        venue_col = ms["venue"].fillna("polymarket") if "venue" in ms else pd.Series("polymarket", index=ms.index)
        counts = venue_col.value_counts()
        st.markdown(
            f"**Prediction markets (latest snapshots)** — Polymarket {counts.get('polymarket', 0):,} rows · Kalshi {counts.get('kalshi', 0):,} rows"
        )
        st.dataframe(
            ms.assign(venue=venue_col)
            .sort_values(["snapshot_ts", "volume"], ascending=False)
            .head(25)[["venue", "question", "outcome", "probability", "decimal_odds", "volume", "snapshot_ts"]],
            hide_index=True,
            width="stretch",
        )

    st.subheader("Sweden — the focus under-covered market")
    swe = _query("SELECT * FROM matches WHERE league_code IN ('SWE','SWE1','SWE2','SWE3') ORDER BY date DESC")
    if swe.empty:
        st.info("No Swedish matches yet — `uv run pitch-edge ingest`.")
    else:
        c = st.columns(4)
        c[0].metric("Matches", f"{len(swe):,}")
        c[1].metric("Divisions", swe["league_code"].nunique())
        c[2].metric("Teams", pd.concat([swe["home_team"], swe["away_team"]]).nunique())
        c[3].metric("Latest", str(pd.Timestamp(swe["date"].max()).date()))
        swe["season_year"] = pd.to_datetime(swe["date"]).dt.year
        rows_tbl = []
        for (yr, lg), g in swe.groupby(["season_year", "league_code"]):
            pts: dict[str, int] = {}
            for _, r in g.iterrows():
                hg, ag = r["home_goals"], r["away_goals"]
                pts[r["home_team"]] = pts.get(r["home_team"], 0) + (3 if hg > ag else 1 if hg == ag else 0)
                pts[r["away_team"]] = pts.get(r["away_team"], 0) + (3 if ag > hg else 1 if hg == ag else 0)
            rows_tbl += [{"season": yr, "league_code": lg, "team": t, "points": p} for t, p in pts.items()]
        tbl = pd.DataFrame(rows_tbl)
        cA, cB = st.columns(2)
        yr = cA.selectbox("Season", sorted(tbl["season"].unique(), reverse=True))
        lg = cB.selectbox("Division", sorted(tbl[tbl["season"] == yr]["league_code"].unique()))
        f = px.bar(
            tbl[(tbl["season"] == yr) & (tbl["league_code"] == lg)].sort_values("points", ascending=False),
            x="team",
            y="points",
            color_discrete_sequence=[SERIES[3]],
            title=f"{lg} {yr} — points from results loaded",
            text_auto=True,
        )
        f.update_layout(height=340, xaxis_tickangle=-45, xaxis_title="", yaxis_title="", hovermode="closest")
        st.plotly_chart(f, width="stretch")
        news_sv = _query(
            "SELECT published_at, feed, team, sentiment, is_injury_news, is_lineup_news, title FROM news_items WHERE language = 'sv' ORDER BY published_at DESC LIMIT 40"
        )
        if not news_sv.empty:
            st.markdown(
                "**Swedish-language news** (SVT, Sportbladet, Expressen, DN, GP) — injury/lineup flags use Swedish keywords."
            )
            st.dataframe(news_sv, hide_index=True, width="stretch")
        note(
            "Odds for Swedish matches (Pinnacle closing, market max/avg, Betfair Exchange) arrive with football-data.co.uk's `SWE.csv` once that site is reachable; Superettan/Ettan depth grows with an Everysport key."
        )

# ============================================================================= 8 health
with tabs[9]:
    st.subheader("Data pipeline health")
    h = health if not health.empty else _query("SELECT * FROM pipeline_runs ORDER BY finished_at DESC LIMIT 200")
    if not h.empty and "last_success" in h:
        h = h.copy()
        h["age_h"] = (
            (pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.to_datetime(h["last_success"])).dt.total_seconds() / 3600
        ).round(1)
        h["ok"] = np.where(h["last_status"] == "ok", "✅", "⚠️")
        st.dataframe(
            h.sort_values("source")[["ok", "source", "last_success", "age_h", "runs", "errors", "rows_last"]],
            hide_index=True,
            width="stretch",
            height=520,
        )
    else:
        st.dataframe(h, hide_index=True, width="stretch")
    st.markdown(
        f"**Optional keys** · {KEYS['n_api_set']} of {KEYS['n_api_total']} unlocked · active: {', '.join(KEYS['active']) or 'none'} · "
        f"missing: {', '.join(KEYS['missing']) or 'none'} → `uv run pitch-edge setup` (terminal only; nothing is collected through the browser)."
    )
    note(
        "football-data.co.uk and Club Elo were returning 503/502 during the recorded runs; the spine was built from the open 2000-2025 compilation instead. "
        "Re-run `pitch-edge ingest` when they are back — rows upgrade in place (Pinnacle early/closing pairs → real CLV, referee names, Swedish odds)."
    )
