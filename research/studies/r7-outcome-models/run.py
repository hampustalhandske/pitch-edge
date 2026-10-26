"""Score the six registered R7 hypotheses (family `outcome-models`) and record every grid point and metric through the
research harness.

  run.py <model> edge   -> outcome-models-<slug>-edge-bits-v1   (primary edge_bits + secondaries, evaluation window,
                                                                  plus the primary on the exploratory window)
  run.py <model> bet    -> outcome-models-<slug>-bet-rule-v1    (primary clv_log + secondaries, same)

`<model>` is dixon_coles | gbdt | stochastic_strength. Reads data/research/r7/pred_<model>.parquet (generate.py) and
writes per-decision frames, stability tables and a summary JSON to data/research/r7/. Pit assertions (pit.py) must have
passed (data/research/r7/pit.json). Nothing touches the warehouse.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pitch_edge.research.devig import power, shin
from pitch_edge.research.ledger import Ledger, default_ledger_path
from pitch_edge.research.registry import load
from pitch_edge.research.stats import block_bootstrap
from pitch_edge.research.trial import run_trial

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "data" / "research" / "r7"
SLUG = {"dixon_coles": "dixon-coles", "gbdt": "gbdt", "stochastic_strength": "stochastic-strength"}
SRC = REPO / "src" / "pitch_edge"
CODE_FILES = [HERE / "run.py", HERE / "generate.py", HERE / "pit.py", HERE / "snapshot.py",
              SRC / "backtest" / "engine.py", SRC / "models" / "poisson.py", SRC / "models" / "gbdt.py",
              SRC / "models" / "stochastic.py", SRC / "models" / "calibration.py"]
EXPLORATORY = (pd.Timestamp("2015-07-01"), pd.Timestamp("2019-06-30"))
EVALUATION = (pd.Timestamp("2019-07-01"), pd.Timestamp("2025-06-30"))
SEED, N_BOOT = 20261021, 2000
EDGE_THRESHOLD, MIN_ODDS, MAX_ODDS, TOP_N, WINDOW_DAYS = 0.03, 1.2, 12.0, 4, 14
EDGE_CEILING_BITS, CLV_CEILING = 0.02, 0.02
OUTCOMES = ("home", "draw", "away")
# model probabilities are clipped to [EPS, 1] before the log, the project's log-loss convention
# (backtest/slices.py, backtest/engine.py); the engine's isotonic-calibrated p_* contain exact zeros
EPS = 1e-12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def prepare(model: str) -> pd.DataFrame:
    """Per-match frame with everything the metrics need (complete Pinnacle open+close, result, raw/cal probs)."""
    p = pd.read_parquet(OUT / f"pred_{model}.parquet")
    p["date"] = pd.to_datetime(p["date"])
    open_c, close_c = ["PSH", "PSD", "PSA"], ["PSCH", "PSCD", "PSCA"]
    p["has_pair"] = p[open_c + close_c].notna().all(axis=1)
    return p


def add_markets(d: pd.DataFrame) -> pd.DataFrame:
    """Adds de-vigged Pinnacle close (power -> pc_*, shin -> pcs_*) and the multiplicative de-vigged open (mo_*)."""
    d = d.copy()
    close = d[["PSCH", "PSCD", "PSCA"]].to_numpy(dtype=float)
    opn = d[["PSH", "PSD", "PSA"]].to_numpy(dtype=float)
    pc, ps = power(close).probs, shin(close).probs
    inv = 1.0 / opn
    mo = inv / inv.sum(axis=1, keepdims=True)
    for j, o in enumerate(OUTCOMES):
        d[f"pc_{o}"], d[f"pcs_{o}"], d[f"mo_{o}"] = pc[:, j], ps[:, j], mo[:, j]
    return d


def take(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return d[cols].to_numpy(dtype=float)


def edge_frame(d: pd.DataFrame) -> pd.DataFrame:
    y = d["result"].to_numpy(dtype=int)
    ar = np.arange(len(d))
    raw = np.clip(take(d, [f"raw_{o}" for o in OUTCOMES]), EPS, 1.0)
    cal = np.clip(take(d, [f"p_{o}" for o in OUTCOMES]), EPS, 1.0)
    pc, pcs, mo = take(d, [f"pc_{o}" for o in OUTCOMES]), take(d, [f"pcs_{o}" for o in OUTCOMES]), take(d, [f"mo_{o}" for o in OUTCOMES])
    out = d[["match_id", "date", "league_code"]].copy()
    out["edge_bits"] = np.log2(raw[ar, y]) - np.log2(pc[ar, y])
    out["edge_bits_shin"] = np.log2(raw[ar, y]) - np.log2(pcs[ar, y])
    out["edge_bits_vs_open_mult"] = np.log2(raw[ar, y]) - np.log2(mo[ar, y])
    out["edge_bits_calibrated"] = np.log2(cal[ar, y]) - np.log2(pc[ar, y])
    out["calibrated_prob_clipped"] = take(d, [f"p_{o}" for o in OUTCOMES])[ar, y] < EPS
    out["open_vs_close_bits"] = np.log2(mo[ar, y]) - np.log2(pc[ar, y])  # descriptive: the open's own shortfall
    return out


def bet_frame(d: pd.DataFrame) -> pd.DataFrame:
    """The engine's rule on RAW probabilities: best-edge outcome per match when edge > threshold and odds in range."""
    raw, mo = take(d, [f"raw_{o}" for o in OUTCOMES]), take(d, [f"mo_{o}" for o in OUTCOMES])
    odds = take(d, ["PSH", "PSD", "PSA"])
    edge = raw - mo
    ok = (edge > EDGE_THRESHOLD) & (odds >= MIN_ODDS) & (odds <= MAX_ODDS)
    masked = np.where(ok, edge, -np.inf)
    k = masked.argmax(axis=1)
    sel = np.isfinite(masked[np.arange(len(d)), k])
    return _bets(d, sel, k, edge)


def _bets(d: pd.DataFrame, sel: np.ndarray, k: np.ndarray, edge: np.ndarray) -> pd.DataFrame:
    odds = take(d, ["PSH", "PSD", "PSA"])
    pc, pcs = take(d, [f"pc_{o}" for o in OUTCOMES]), take(d, [f"pcs_{o}" for o in OUTCOMES])
    ar = np.arange(len(d))
    taken, p_c, p_s = odds[ar, k], pc[ar, k], pcs[ar, k]
    won = (d["result"].to_numpy(dtype=int) == k)
    out = d[["match_id", "date", "league_code"]].copy()
    out["outcome"] = np.array(OUTCOMES)[k]
    out["edge"] = edge[ar, k]
    out["odds"] = taken
    out["clv_log"] = np.log(taken * p_c)
    out["clv_log_shin"] = np.log(taken * p_s)
    out["roi_flat"] = np.where(won, taken - 1.0, -1.0)
    return out[sel].reset_index(drop=True)


def top4_frame(d: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """`ask` literal rule: consecutive 14-day windows from `start`; the 4 matches with the largest best-outcome edge."""
    raw, mo = take(d, [f"raw_{o}" for o in OUTCOMES]), take(d, [f"mo_{o}" for o in OUTCOMES])
    edge = raw - mo
    k = edge.argmax(axis=1)
    b = _bets(d, np.ones(len(d), dtype=bool), k, edge)
    b["window"] = ((b["date"] - start).dt.days // WINDOW_DAYS).to_numpy()
    b = b.sort_values(["window", "edge", "match_id"], ascending=[True, False, True])
    return b.groupby("window", sort=True).head(TOP_N).reset_index(drop=True)


def per_decision(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    return pd.DataFrame({
        "decision_id": frame["match_id"].to_numpy(),
        "block": frame["date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "value": frame[col].to_numpy(dtype=float),
        "league": frame["league_code"].to_numpy(),
    })


def record(reg, config, frame, col, metric, extra, ledger, window="evaluation"):
    pdf = per_decision(frame, col)
    for attempt in range(30):
        try:
            res = run_trial(reg, config, pdf, metric, ledger, extra["data_snapshot"], extra, n_boot=N_BOOT, seed=SEED,
                            window=window, code_files=CODE_FILES)
            break
        except duckdb.IOException as exc:
            if "lock" not in str(exc).lower():
                raise
            time.sleep(min(2 + 2 * attempt, 30))
    else:
        raise RuntimeError("ledger stayed locked")
    print(f"  {metric:28s} {window:11s} {res.metric_value:+.5g} [{res.ci_lo:+.5g}, {res.ci_hi:+.5g}] p={res.p_value:.4f} "
          f"n={res.n} blocks={res.n_blocks}")
    return res


def stability(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    frame = frame.sort_values(["date", "match_id"]).reset_index(drop=True)
    day = frame["date"].dt.strftime("%Y-%m-%d").to_numpy()
    rows = []

    def add(label, mask):
        sub = frame[mask]
        if len(sub) < 2:
            return
        r = block_bootstrap(sub[col].to_numpy(dtype=float), day[mask], n_boot=N_BOOT, seed=SEED)
        rows.append({"slice": label, "n": len(sub), "mean": r.estimate, "lo": r.lo, "hi": r.hi, "blocks": r.n_blocks})

    add("all", np.ones(len(frame), dtype=bool))
    season = np.where(frame["date"].dt.month >= 7, frame["date"].dt.year, frame["date"].dt.year - 1)
    for s in sorted(set(season)):
        add(f"season {s}/{str(s + 1)[2:]}", season == s)
    uniq = sorted(set(day))
    thirds = np.array_split(np.array(uniq), 3)
    for i, t in enumerate(thirds, 1):
        add(f"third {i} ({t[0]}..{t[-1]})", np.isin(day, t))
    for lg in sorted(frame["league_code"].unique()):
        add(f"league {lg}", (frame["league_code"] == lg).to_numpy())
    return pd.DataFrame(rows)


def main() -> None:
    model, kind = sys.argv[1], sys.argv[2]
    pit = json.loads((OUT / "pit.json").read_text())
    ledger = Ledger(default_ledger_path())
    p = prepare(model)
    cut = pd.read_parquet(OUT / f"pred_{model}_cut.parquet")
    full = p.set_index("match_id")
    cut = cut.set_index("match_id")
    common = cut.index.intersection(full.index)
    rep_diff = float(np.abs(full.loc[common, [f"raw_{o}" for o in OUTCOMES]].to_numpy(dtype=float)
                           - cut.loc[common, [f"raw_{o}" for o in OUTCOMES]].to_numpy(dtype=float)).max())
    clean = bool(rep_diff < 1e-9 and not pit["feature_truncation"]["columns_differing"]
                 and not pit["feature_truncation"]["differing_rows_not_a_same_day_referee_pair"]
                 and pit["prediction_truncation"][model]["max_abs_diff"] < 1e-9
                 and all(pit["fold_discipline"][model][k] for k in ("train_before_fold", "fold_before_match", "within_14d")))
    snapshot = (f"warehouse load_feature_frame(min_date=2015-07-01) 2026-10-21 features.parquet sha256:{sha(OUT / 'features.parquet')}; "
                f"pred_{model}.parquet sha256:{sha(OUT / f'pred_{model}.parquet')}")
    slug = SLUG[model]
    reg = load(REPO / "research" / "registry" / f"outcome-models-{slug}-{'edge-bits' if kind == 'edge' else 'bet-rule'}-v1.yaml")
    print(reg.id, "clean:", clean, "| rerun max diff", rep_diff)
    d_all = p[p["has_pair"]].copy()
    dropped_all = int((~p["has_pair"]).sum())
    d_all = add_markets(d_all)
    summary: dict = {"model": model, "hypothesis": reg.id, "clean": clean, "snapshot": snapshot,
                     "n_predicted_total": int(len(p)), "n_without_pinnacle_pair": dropped_all}
    config = reg.expand_grid()[0]
    season = np.where(p["date"].dt.month >= 7, p["date"].dt.year, p["date"].dt.year - 1)
    cov = p.assign(season=season).groupby(["league_code", "season"])["has_pair"].agg(["size", "sum"])
    cov.rename(columns={"size": "predicted", "sum": "with_pinnacle_pair"}).reset_index().to_csv(
        OUT / f"{model}_coverage.csv", index=False)
    for window, (lo, hi) in (("evaluation", EVALUATION), ("exploratory", EXPLORATORY)):
        d = d_all[(d_all["date"] >= lo) & (d_all["date"] <= hi)].reset_index(drop=True)
        pw = p[(p["date"] >= lo) & (p["date"] <= hi)]
        miss = pw.groupby("league_code")["has_pair"].agg(["size", "sum"]).rename(columns={"size": "predicted", "sum": "with_pair"})
        extra_base = {"clean": clean, "data_snapshot": snapshot, "pit_assertion": json.dumps(pit)[:1500],
                      "n_matches_in_window": int(len(pw)), "n_dropped_no_pinnacle_pair": int(len(pw) - len(d)),
                      "retrain_every_days": 14, "rerun_max_abs_prob_diff_vs_cut_run": rep_diff}
        print(f"-- {window}: {len(d)} matches with complete Pinnacle pair ({len(pw) - len(d)} dropped of {len(pw)})")
        if kind == "edge":
            f = edge_frame(d)
            f.to_parquet(OUT / f"{model}_edge_{window}.parquet")
            extra = {**extra_base, "within_ceiling": bool(f["edge_bits"].mean() < EDGE_CEILING_BITS),
                     "open_vs_close_bits_mean": float(f["open_vs_close_bits"].mean()),
                     "n_calibrated_prob_clipped": int(f["calibrated_prob_clipped"].sum()),
                     "coverage_by_league": miss.reset_index().to_dict("records")}
            res = record(reg, config, f, "edge_bits", "edge_bits", extra, ledger, window)
            summary[f"{window}_primary"] = [res.metric_value, res.ci_lo, res.ci_hi, res.n, res.n_blocks, res.p_value]
            if window == "evaluation":
                for m in ("edge_bits_shin", "edge_bits_vs_open_mult", "edge_bits_calibrated"):
                    r2 = record(reg, config, f, m, m, {"clean": clean, "data_snapshot": snapshot}, ledger, window)
                    summary[m] = [r2.metric_value, r2.ci_lo, r2.ci_hi, r2.n, r2.p_value]
                st = stability(f, "edge_bits")
                st.to_csv(OUT / f"{model}_edge_stability.csv", index=False)
                print(st.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
                ob = block_bootstrap(f["open_vs_close_bits"].to_numpy(), f["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                                     n_boot=N_BOOT, seed=SEED)
                summary["open_vs_close_bits"] = [ob.estimate, ob.lo, ob.hi]
        else:
            b = bet_frame(d)
            b.to_parquet(OUT / f"{model}_bets_{window}.parquet")
            extra = {**extra_base, "within_ceiling": bool(b["clv_log"].mean() < CLV_CEILING),
                     "n_bets": int(len(b)), "bets_per_match": float(len(b) / max(len(d), 1)),
                     "outcome_mix": b["outcome"].value_counts().to_dict(), "mean_edge": float(b["edge"].mean()),
                     "mean_odds": float(b["odds"].mean())}
            res = record(reg, config, b, "clv_log", "clv_log", extra, ledger, window)
            summary[f"{window}_primary"] = [res.metric_value, res.ci_lo, res.ci_hi, res.n, res.n_blocks, res.p_value]
            if window == "evaluation":
                for m in ("roi_flat", "clv_log_shin"):
                    r2 = record(reg, config, b, m, m, {"clean": clean, "data_snapshot": snapshot}, ledger, window)
                    summary[m] = [r2.metric_value, r2.ci_lo, r2.ci_hi, r2.n, r2.p_value]
                t4 = top4_frame(d, EVALUATION[0])
                t4.to_parquet(OUT / f"{model}_top4.parquet")
                r4 = record(reg, config, t4, "clv_log", "clv_log_top4_per_window", {"clean": clean, "data_snapshot": snapshot}, ledger, window)
                summary["clv_log_top4_per_window"] = [r4.metric_value, r4.ci_lo, r4.ci_hi, r4.n, r4.p_value]
                st = stability(b, "clv_log")
                st.to_csv(OUT / f"{model}_bet_stability.csv", index=False)
                print(st.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    (OUT / f"{model}_{kind}_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
