"""Score closing-line-prediction-soft-book-vs-model-close-v1: back a named soft book's pre-closing price when
it beats the xg closing-line model's predicted fair close, and measure CLV against the de-vigged Pinnacle
close (Betfair Exchange close in the post segment).

Inputs: data/research/r3/panel.parquet (Pinnacle prices, de-vigged closes, results), the books' pre-closing
prices from data/research/r3/fd_odds.parquet, and the registered xg-finishing
walk-forward predictions data/research/r3/features/closing-line-prediction-xg-finishing-v1__power.parquet.
For every (book, theta) grid point: primary `clv_log` on the evaluation window, then the secondaries
`clv_log_shin_close`, `roi`, `clv_log_post_betfair`, each through run_trial. Per-bet frames go to
data/research/r3/strategy/<book>_<theta>.parquet.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r3-common"))

import r3lib as R  # noqa: E402

from pitch_edge.research.ledger import Ledger  # noqa: E402
from pitch_edge.research.registry import load  # noqa: E402

REG = R.REPO / "research" / "registry" / "closing-line-prediction-soft-book-vs-model-close-v1.yaml"
PRED = R.D / "features" / "closing-line-prediction-xg-finishing-v1__power.parquet"
CODE_FILES = [
    HERE / "run.py",
    HERE.parent / "r3-common" / "r3lib.py",
    HERE.parent / "r3-common" / "features.py",
    HERE.parent / "r3-features" / "run.py",
]
OUT = R.D / "strategy"
CAP = 1.5
OUT_CODES = ("H", "D", "A")
# R3_REPRODUCE=1: rerun grid points that already have a trial (a reproduction check; they must match the
# result of record exactly). R3_CHECK_ONLY=1 additionally records nothing and only compares.
REPRODUCE = bool(os.environ.get("R3_REPRODUCE") or os.environ.get("R3_CHECK_ONLY"))
CHECK_ONLY = bool(os.environ.get("R3_CHECK_ONLY"))


def bets(frame: pd.DataFrame, book: str, theta: float) -> pd.DataFrame:
    """One row per bet: the largest-edge outcome per match when edge = log(price * p_hat_close) > theta."""
    price_cols = [f"oa_{book}{c}" for c in OUT_CODES]
    assert all(not c.endswith("C" + c[-1]) for c in price_cols)  # pre-closing columns only (no B365CH etc.)
    price = frame[price_cols].to_numpy(dtype=float)
    pin_open = frame[[f"pinnacle_{o}_open" for o in R.OUTCOMES]].to_numpy(dtype=float)
    phat = frame[[f"phat_model_{o}" for o in R.OUTCOMES]].to_numpy(dtype=float)
    valid = np.isfinite(price) & (price > 1.0) & (price <= CAP * pin_open)
    edge = np.where(valid, np.log(np.where(valid, price, 1.0) * phat), -np.inf)
    k = np.argmax(edge, axis=1)
    e = edge[np.arange(len(k)), k]
    take = np.isfinite(e) & (e > theta)
    f = frame.loc[take].copy()
    k = k[take]
    idx = np.arange(len(k))
    f["outcome"] = np.array(R.OUTCOMES)[k]
    f["price"] = price[take][idx, k]
    f["edge_at_open"] = e[take]
    f["p_close_power"] = f[[f"p_close_power_{o}" for o in R.OUTCOMES]].to_numpy()[idx, k]
    f["p_close_shin"] = f[[f"p_close_shin_{o}" for o in R.OUTCOMES]].to_numpy()[idx, k]
    f["clv"] = np.log(f["price"] * f["p_close_power"])
    f["clv_shin"] = np.log(f["price"] * f["p_close_shin"])
    won = f["ft_result"].to_numpy() == np.array(OUT_CODES)[k]
    f["roi"] = np.where(won, f["price"] - 1.0, -1.0)
    f["has_result"] = f["ft_result"].isin(OUT_CODES)
    return f


def book_prices() -> pd.DataFrame:
    """Pre-closing 1X2 prices of the registered books from the fd_odds snapshot, keyed like the panel."""
    d = pd.read_parquet(R.D / "fd_odds.parquet")
    d["match_date"] = pd.to_datetime(d["match_date"])
    d["match_id"] = (
        d["division"] + "|" + d["match_date"].dt.strftime("%Y-%m-%d") + "|" + d["home_team"] + "|" + d["away_team"]
    )
    cols = [f"oa_{b}{c}" for b in ("B365", "BW", "WH") for c in OUT_CODES]
    return d[["match_id", *cols]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reg = load(REG)
    ledger = Ledger(R.REPO / "data" / "research" / "ledger.duckdb")
    panel = R.load_panel()
    pred = pd.read_parquet(PRED)[["match_id"] + [f"phat_model_{o}" for o in R.OUTCOMES]]
    frame = panel.drop(columns=[c for c in panel.columns if c.startswith("oa_")])
    frame = frame.merge(pred, on="match_id", how="inner", validate="one_to_one")
    frame = frame.merge(book_prices(), on="match_id", how="left", validate="one_to_one")
    ev0, ev1 = pd.Timestamp(reg.window.evaluation[0]), pd.Timestamp(reg.window.evaluation[1])
    code = R.code_sha(CODE_FILES)
    from pitch_edge.research.registry import config_hash

    done = ledger.trials(hypothesis_id=reg.id)
    done_keys = set(zip(done["config_hash"], done["metric_name"], strict=True)) if not done.empty else set()

    def rec(config: dict, f: pd.DataFrame, col: str, metric: str, extra: dict):  # type: ignore[no-untyped-def]
        """Record once: a (config, metric) already in the ledger is never rerun (first run is of record)."""
        if (config_hash(config), metric) in done_keys and not REPRODUCE:
            print(f"   {metric}: already recorded, not rerun")
            return None
        if f["match_date"].nunique() < 2:
            print(f"   {metric}: n={len(f)} on {f['match_date'].nunique()} match-day(s), not scoreable")
            return None
        return R.record(reg, config, f, col, metric, extra, ledger, CODE_FILES, CHECK_ONLY)

    for config in reg.expand_grid():
        book, theta = config["book"], float(config["theta"])
        b1, b2 = bets(frame, book, theta), bets(frame, book, theta)
        deterministic = bool(np.array_equal(b1["clv"].to_numpy(), b2["clv"].to_numpy()))
        b1["decision_id"] = b1["match_id"]
        b1.drop(columns=[c for c in b1.columns if c.startswith("oa_")]).to_parquet(OUT / f"{book}_{theta}.parquet")
        ev = b1[(b1["segment"] == "pinnacle") & (b1["match_date"] >= ev0) & (b1["match_date"] <= ev1)]
        ev = ev.sort_values(["match_date", "match_id"]).reset_index(drop=True)
        post = b1[b1["segment"] == "post_betfair"].sort_values(["match_date", "match_id"]).reset_index(drop=True)
        if ev.empty:
            print(book, theta, "no evaluation bets")
            continue
        clv_mean = float(ev["clv"].mean())
        extra = {
            "clean": deterministic,
            "pit_assertion": "xg_known_at < snapshot_date asserted in the feature run; prices are pre-closing columns",
            "within_ceiling": bool(clv_mean < 0.02),
            "leave_one_league_out": R.loo_league(ev, "clv"),
            "code_files": [str(f.relative_to(R.REPO)) for f in CODE_FILES],
            "code_sha256_16": code,
            "mean_price": float(ev["price"].mean()),
            "mean_edge_at_open": float(ev["edge_at_open"].mean()),
            "outcome_mix": ev["outcome"].value_counts().to_dict(),
            "eligible_matches": int(
                frame[(frame["segment"] == "pinnacle") & (frame["match_date"] >= ev0) & (frame["match_date"] <= ev1)][
                    f"oa_{book}H"
                ]
                .notna()
                .sum()
            ),
        }
        r = rec(config, ev, "clv", "clv_log", extra)
        if CHECK_ONLY:
            rec(config, ev, "clv_shin", "clv_log_shin_close", {})
            rec(config, ev[ev["has_result"]], "roi", "roi", {})
            if len(post):
                rec(config, post, "clv", "clv_log_post_betfair", {})
            continue
        if r is not None:
            print(
                f"{book} theta={theta}: CLV {100 * r.metric_value:+.2f}% [{100 * r.ci_lo:+.2f}, {100 * r.ci_hi:+.2f}] "
                f"n={r.n} p={r.p_value:.4f} mean edge@open {100 * extra['mean_edge_at_open']:.2f}% "
                f"eligible {extra['eligible_matches']} mix {extra['outcome_mix']} clean={deterministic}"
            )
        rs = rec(config, ev, "clv_shin", "clv_log_shin_close", {"code_sha256_16": code})
        evr = ev[ev["has_result"]]
        ro = rec(config, evr, "roi", "roi", {"code_sha256_16": code})
        line = ""
        if rs is not None and ro is not None:
            line = (
                f"   Shin-close CLV {100 * rs.metric_value:+.2f}% [{100 * rs.ci_lo:+.2f}, {100 * rs.ci_hi:+.2f}]"
                f"  ROI {100 * ro.metric_value:+.1f}% [{100 * ro.ci_lo:+.1f}, {100 * ro.ci_hi:+.1f}]"
            )
        if len(post):
            rp = rec(config, post, "clv", "clv_log_post_betfair", {"code_sha256_16": code})
            if rp is not None:
                line += f"  post-Betfair CLV {100 * rp.metric_value:+.2f}% [{100 * rp.ci_lo:+.2f}, {100 * rp.ci_hi:+.2f}] n={rp.n}"
        print(line)
        by_season = ev.groupby("season_start")["clv"].agg(["size", "mean"])
        by_div = ev.groupby("division")["clv"].agg(["size", "mean"])
        print("   by season", {int(k): (int(v["size"]), round(100 * v["mean"], 2)) for k, v in by_season.iterrows()})
        print("   by div", {k: (int(v["size"]), round(100 * v["mean"], 2)) for k, v in by_div.iterrows()})


if __name__ == "__main__":
    main()
