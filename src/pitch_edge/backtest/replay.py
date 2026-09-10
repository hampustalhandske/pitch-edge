"""T0 replay-simulation: does the reviewer's deterministic verdict correlate with anything real?

Most agentic demos stop at "it runs." This is the evaluation that checks whether it is worth
anything: pick a cutoff `T0` strictly before the most recent real (non-synthetic) closing odds in
the warehouse, train everything — features, models, `by_league.csv` — on data before `T0` only,
then feed the `[T0, T0+30d)` window through the exact `agentic-signals` pipeline as if those
fixtures were upcoming (synthetic odds, no access to their real results or closing prices). Only
*after* the reviewer has produced its trust/distrust/needs_info verdicts do we reveal the real
closing odds and results already sitting in the warehouse, and check whether "trust" fixtures
actually had better realized edge/hit-rate than "distrust" ones.

`agents/reviewer.py`'s verdict is now a deterministic rule (real edge_bits sign + real-vs-synthetic
odds), not an LLM judgment — so this replay is validating that *rule*, the same way a backtest
validates a model. That's arguably more useful than validating an LLM's opinion: the rule is fixed
and auditable, so a replay result here is reproducible evidence about the rule itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pitch_edge.agents.graph import GraphDependencies
from pitch_edge.agents.orchestrator import AgenticSignalPipeline
from pitch_edge.agents.risk import RiskLimits, RiskManager
from pitch_edge.backtest.engine import WalkForwardConfig
from pitch_edge.data.storage import Warehouse
from pitch_edge.models import available_models
from pitch_edge.models.base import MatchModel
from pitch_edge.odds.utils import no_vig_probabilities
from pitch_edge.pipeline import load_feature_frame, run_backtests, synthetic_quotes_from_elo
from pitch_edge.rag.index import VectorIndex

logger = logging.getLogger(__name__)

RESULT_TO_OUTCOME = {0: "home", 1: "draw", 2: "away"}


def compute_t0(
    wh: Warehouse, bet_prefix: str = "PS", closing_prefix: str = "PSC", window_days: int = 30
) -> pd.Timestamp:
    """max(date with a real Pinnacle closing price) - window_days. Only football-data.co.uk rows
    carry a genuine early/closing pair (`bet_prefix`/`closing_prefix`); the xgabora fallback spine
    never does, so this never picks a window where CLV would be zero by construction."""
    matches = wh.matches_with_closing_odds(bookmakers=(bet_prefix,))
    close_col = f"{closing_prefix}H"
    if matches.empty or close_col not in matches.columns:
        raise ValueError(f"no matches with a real {closing_prefix}* closing price found — cannot compute T0")
    real = matches[matches[close_col].notna()]
    if real.empty:
        raise ValueError(f"no matches with a real {closing_prefix}* closing price found — cannot compute T0")
    return pd.Timestamp(pd.to_datetime(real["date"]).max()) - pd.Timedelta(days=window_days)


def _hidden_fixture_rows(window: pd.DataFrame) -> list[dict]:
    """Strips real odds/results, keeping only what a live `upcoming_fixture_frame` would carry."""
    hide = {"home_goals", "away_goals", "result", "total_goals"}
    hide |= {c for c in window.columns if c.startswith(("PS", "B365", "Mkt", "Avg", "Max", "BFE", "WH", "VC"))}
    rows = window.drop(columns=[c for c in hide if c in window.columns]).copy()
    rows["date"] = rows["date"].astype(str)
    rows["lineup_source"] = "provisional"
    return rows.to_dict(orient="records")


def _score_proposals(proposals: list[dict], reviews: list[dict], known: pd.DataFrame) -> pd.DataFrame:
    """Reveal real closing odds/results for the window and score each proposal: did the picked
    outcome occur, and how did the model's probability compare to the *real* no-vig closing
    probability for that outcome (a genuine, after-the-fact accuracy check)."""
    known_by_id = known.set_index("match_id")
    verdict_by_key = {(r["match_id"], r["outcome"]): r["verdict"] for r in reviews}
    rows = []
    for p in proposals:
        if p["match_id"] not in known_by_id.index:
            continue
        row = known_by_id.loc[p["match_id"]]
        close = (row.get("PSCH"), row.get("PSCD"), row.get("PSCA"))
        if any(pd.isna(c) for c in close):
            continue  # no real closing price for this fixture — cannot score it honestly
        close_probs = dict(zip(("home", "draw", "away"), no_vig_probabilities(*close), strict=True))
        actual_outcome = RESULT_TO_OUTCOME.get(int(row["result"])) if pd.notna(row.get("result")) else None
        if actual_outcome is None:
            continue
        rows.append(
            {
                "match_id": p["match_id"],
                "outcome": p["outcome"],
                "verdict": verdict_by_key.get((p["match_id"], p["outcome"]), "unknown"),
                "model_probability": p["model_probability"],
                "market_probability_at_bet": p["market_probability"],
                "real_closing_probability": close_probs[p["outcome"]],
                "edge_vs_real_close": p["model_probability"] - close_probs[p["outcome"]],
                "hit": p["outcome"] == actual_outcome,
            }
        )
    return pd.DataFrame(rows)


def _summarize_by_verdict(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(columns=["verdict", "n", "hit_rate", "mean_edge_vs_real_close", "log_loss"])
    rows = []
    for verdict, g in scored.groupby("verdict"):
        p_hit = np.clip(np.where(g["hit"], g["model_probability"], 1 - g["model_probability"]), 1e-12, 1)
        rows.append(
            {
                "verdict": verdict,
                "n": len(g),
                "hit_rate": float(g["hit"].mean()),
                "mean_edge_vs_real_close": float(g["edge_vs_real_close"].mean()),
                "log_loss": float(-np.log(p_hit).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("verdict")


def run_replay_eval(
    wh: Warehouse,
    reports_dir: str | Path,
    as_of: str | pd.Timestamp | None = None,
    window_days: int = 30,
    index: VectorIndex | None = None,
    backtest_config: WalkForwardConfig | None = None,
    models: list[MatchModel] | None = None,
    leagues: list[str] | None = None,
) -> dict:
    """Runs the full T0 replay described in this module's docstring. Returns a dict with `t0`,
    `window_end`, the per-proposal scored frame, and the trust-vs-distrust summary — and writes
    `replay_eval.csv` / `replay_eval_summary.csv` to `reports_dir` (pass `Settings.backtest_dir /
    "replay"`, the local data dir — this is raw per-fixture scoring data, not the public case
    study; write that separately once a replay run is worth publishing)."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    t0 = pd.Timestamp(as_of) if as_of else compute_t0(wh, window_days=window_days)
    window_end = t0 + pd.Timedelta(days=window_days)
    logger.info("replay-eval: T0=%s window_end=%s", t0.date(), window_end.date())

    features = load_feature_frame(wh, leagues=leagues, min_date="2015-07-01")
    if features.empty:
        raise ValueError("no features available — run `pitch-edge ingest` and `pitch-edge features` first")
    features["date"] = pd.to_datetime(features["date"])
    train = features[features["date"] < t0]
    window = features[(features["date"] >= t0) & (features["date"] < window_end)]
    if train.empty or window.empty:
        raise ValueError(
            f"not enough data around T0={t0.date()} to run a replay (train={len(train)}, window={len(window)})"
        )

    # Train-only evidence: by_league.csv restricted to strictly-before-T0 data. Both the data dir
    # and the report dir point under `reports_dir` (already the local backtest data dir here, not
    # the public `reports/` tree — see the caller in `cli.py::replay_eval`) so this internal-only
    # training pass never drops a case-study file in the public tree.
    train_reports_dir = reports_dir / "replay_train"
    fitted = models or available_models()
    run_backtests(
        train,
        fitted,
        backtest_config or WalkForwardConfig(),
        wh=None,
        data_dir=train_reports_dir,
        report_dir=train_reports_dir,
        label="replay_train",
    )
    for m in fitted:
        m.fit(train)

    hidden_rows = _hidden_fixture_rows(window)

    def scout() -> list[dict]:
        return hidden_rows

    def fetch_odds(rows: list[dict]) -> list[dict]:
        return synthetic_quotes_from_elo(pd.DataFrame(rows)) if rows else []

    deps = GraphDependencies(
        scout=scout,
        featurize=lambda rows: rows,
        infer=lambda rows: [],  # replaced by the per-league router inside build_agentic_graph
        fetch_odds=fetch_odds,
        risk=RiskManager(RiskLimits()),
        sink=lambda alerts: None,
        model_name="replay",
    )
    pipe = AgenticSignalPipeline(
        deps,
        reports_dir=train_reports_dir,
        index=index or VectorIndex(),
        wh=wh,
        available_models=fitted,
        # A production index can already contain real match/prediction docs for this window (it was
        # ingested from the same warehouse we're replaying); this is the only thing standing between
        # the reviewer and reading the real score before it's supposed to be revealed.
        context_as_of=str(t0.date()),
    )
    _, state = pipe.run_to_gate()
    proposals, reviews = state.get("proposals", []), state.get("reviews", [])

    known = (
        window[["match_id", "result", "PSCH", "PSCD", "PSCA"]].copy()
        if "PSCH" in window.columns
        else window[["match_id", "result"]].assign(PSCH=np.nan, PSCD=np.nan, PSCA=np.nan)
    )
    scored = _score_proposals(proposals, reviews, known)
    summary = _summarize_by_verdict(scored)

    scored.to_csv(reports_dir / "replay_eval.csv", index=False)
    summary.to_csv(reports_dir / "replay_eval_summary.csv", index=False)
    return {
        "t0": t0,
        "window_end": window_end,
        "n_fixtures": len(window),
        "n_proposals": len(proposals),
        "n_scored": len(scored),
        "scored": scored,
        "summary": summary,
    }
