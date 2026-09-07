"""Stage E — Phase 5 real-data run (restartable; run with `nohup caffeinate -i uv run python scripts/stage_e.py &`).

1. Wikipedia pageviews for the backtest divisions (skipped if already loaded today)
2. feature store rebuilt WITH Transfermarkt rotation/referee context + attention anomalies
3. ablation on the pre-registered groups (referee, travel_fatigue, wiki_attention, rotation_load) -> reports/ablation_phase5.csv
4. referee tendency table + referee-announcement event study -> reports/referee_lag.csv
5. cross-venue lead-lag on stored snapshots -> reports/lead_lag.csv
6. two dossiers (one played, one scheduled) -> `dossiers` table
Each step logs to logs/stage_e.log and is idempotent.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
(ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(ROOT / "logs" / "stage_e.log"), logging.StreamHandler()],
)
log = logging.getLogger("stage_e")

from pitch_edge.ablation import run_ablation  # noqa: E402
from pitch_edge.backtest.engine import WalkForwardConfig  # noqa: E402
from pitch_edge.backtest.event_study import referee_announcement_study, referee_tendency_table  # noqa: E402
from pitch_edge.config import get_settings  # noqa: E402
from pitch_edge.data.ingest import ingest_wikipedia_attention  # noqa: E402
from pitch_edge.data.storage import Warehouse  # noqa: E402
from pitch_edge.odds.leadlag import cross_venue_lead_lag  # noqa: E402
from pitch_edge.pipeline import load_feature_frame, persist_features  # noqa: E402

LEAGUES = ["E0", "E1", "D1", "SP1", "I1", "F1", "N1", "P1", "B1", "SC0", "T1"]
GROUPS = ["referee", "travel_fatigue", "wiki_attention", "rotation_load"]


def step(name):
    def deco(fn):
        def wrapped(*a, **k):
            t0 = time.time()
            log.info("=== %s: start", name)
            out = fn(*a, **k)
            log.info("=== %s: done in %.1f min", name, (time.time() - t0) / 60)
            return out

        return wrapped

    return deco


@step("1 wikipedia pageviews")
def s1(wh):
    if wh.table_exists("wiki_pageviews"):
        mx = wh.query("SELECT max(date) d FROM wiki_pageviews")["d"].iloc[0]
        if pd.notna(mx) and pd.Timestamp(mx) >= pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(
            days=2
        ):
            log.info("pageviews already current (max %s) — skipping", mx)
            return
    ingest_wikipedia_attention(wh)


@step("2 features with context")
def s2(wh):
    f = load_feature_frame(wh, leagues=LEAGUES, min_date="2015-07-01", with_context=True)
    persist_features(wh, f)
    cov = {
        c: float(f[c].notna().mean()) for c in ("ref_cards_per_game", "rot_home_minutes_7d", "pv_home_anom") if c in f
    }
    log.info("features: %d rows, %d cols, coverage %s", len(f), f.shape[1], cov)
    return f


@step("3 ablation")
def s3(f):
    table, results = run_ablation(f, GROUPS, WalkForwardConfig(retrain_every_days=90))
    out = get_settings().reports_dir / "ablation_phase5.csv"
    table.to_csv(out, index=False)
    log.info("\n%s", table.round(4).to_string())
    full = results["gbdt_full"]
    log.info(
        "full model log-loss %.4f vs market %.4f",
        full.calibration["multiclass_log_loss"],
        full.calibration.get("market_multiclass_log_loss", float("nan")),
    )
    return table


@step("4 referee study")
def s4(wh, f):
    tend = referee_tendency_table(f) if "referee" in f else pd.DataFrame()
    if not tend.empty:
        wh.replace("referee_tendency", tend)
    moves = wh.read("referee_announcement_moves")
    table = referee_announcement_study(moves, tend)
    table.to_csv(get_settings().reports_dir / "referee_lag.csv", index=False)
    log.info("referees with tendency data: %d · paired announcements: %d\n%s", len(tend), len(moves), table.to_string())


@step("5 lead-lag")
def s5(wh):
    table = cross_venue_lead_lag(wh.read("market_snapshots"))
    table.to_csv(get_settings().reports_dir / "lead_lag.csv", index=False)
    log.info("\n%s", table.tail(5).to_string())


@step("6 dossiers")
def s6(wh):
    from pitch_edge.intel import DossierBuilder, find_fixture, save_dossier
    from pitch_edge.intel.predict import predict_fixture

    for home, away, date, fit in (
        ("Aston Villa", "Arsenal", "2026-08-31", False),  # played: walk-forward fold prediction
        ("Aston Villa", "Nott'm Forest", None, True),  # next weekend: raw fitted prediction + forecast weather
        ("Brighton", "Nott'm Forest", None, True),  # months ahead: the "cannot be known yet" rendering
    ):
        try:
            fx = find_fixture(wh, home, away, date, fetch_upcoming=True)
            pred = predict_fixture(wh, fx) if (fit and not fx.played) else None
            d = DossierBuilder(wh, fetch=True).build(fx.home_team, fx.away_team, fixture=fx, prediction=pred)
            row = save_dossier(wh, d)
            log.info(
                "dossier %s v%s verified=%s claims=%d gaps=%d",
                d.fixture_key,
                d.version,
                row["verified"],
                row["n_claims"],
                row["n_gaps"],
            )
            print(d.markdown())
        except Exception as exc:  # noqa: BLE001
            log.exception("dossier %s vs %s failed: %s", home, away, exc)


if __name__ == "__main__":
    only = set(sys.argv[1:])
    with Warehouse(get_settings().db_path) as wh:
        if not only or "1" in only:
            s1(wh)
        f = s2(wh) if (not only or "2" in only or "3" in only or "4" in only) else wh.read("features")
        if not only or "3" in only:
            s3(f)
        if not only or "4" in only:
            s4(wh, f)
        if not only or "5" in only:
            s5(wh)
        if not only or "6" in only:
            s6(wh)
    log.info("stage E complete")
