"""Score closing-line-prediction-bw-fair-ablation-v1 (post-hoc): the bwin theta=0 rule of
closing-line-prediction-soft-book-vs-model-close-v1 with a simpler fair price — the R2-Q2 baseline's predicted
close (`baseline_close`) or the de-vigged Pinnacle open (`pinnacle_open`) — on the same universe.

Records `clv_log` (primary) and `overlap_with_xg_rule` (1 when the same match and outcome is also bet by the
registered xg-model rule) for each grid point. Per-bet frames: data/research/r3/strategy/ablation_<fair>.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r3-common"))
sys.path.insert(0, str(HERE))

import r3lib as R  # noqa: E402
import run as S  # noqa: E402

from pitch_edge.research.ledger import Ledger  # noqa: E402
from pitch_edge.research.registry import load  # noqa: E402

REG = R.REPO / "research" / "registry" / "closing-line-prediction-bw-fair-ablation-v1.yaml"


def main() -> None:
    reg = load(REG)
    ledger = Ledger(R.REPO / "data" / "research" / "ledger.duckdb")
    panel = R.load_panel()
    pred = pd.read_parquet(S.PRED)
    keep = ["match_id"] + [f"phat_{k}_{o}" for k in ("model", "base") for o in R.OUTCOMES]
    frame = panel.drop(columns=[c for c in panel.columns if c.startswith("oa_")])
    frame = frame.merge(pred[keep], on="match_id", how="inner", validate="one_to_one")
    frame = frame.merge(S.book_prices(), on="match_id", how="left", validate="one_to_one")
    ev0, ev1 = pd.Timestamp(reg.window.evaluation[0]), pd.Timestamp(reg.window.evaluation[1])
    xg_rule = S.bets(frame, "BW", 0.0)
    xg_keys = set(zip(xg_rule["match_id"], xg_rule["outcome"], strict=True))
    code = R.code_sha([HERE / "ablation.py", HERE / "run.py", HERE.parent / "r3-common" / "r3lib.py"])
    for config in reg.expand_grid():
        fair = config["fair"]
        f = frame.copy()
        src = {
            "baseline_close": [f"phat_base_{o}" for o in R.OUTCOMES],
            "pinnacle_open": [f"p_open_power_{o}" for o in R.OUTCOMES],
        }[fair]
        f[[f"phat_model_{o}" for o in R.OUTCOMES]] = f[src].to_numpy()
        b = S.bets(f, "BW", 0.0)
        b2 = S.bets(f, "BW", 0.0)
        b["decision_id"] = b["match_id"]
        b["overlap"] = [float((m, o) in xg_keys) for m, o in zip(b["match_id"], b["outcome"], strict=True)]
        b.drop(columns=[c for c in b.columns if c.startswith("oa_")]).to_parquet(
            R.D / "strategy" / f"ablation_{fair}.parquet"
        )
        ev = b[(b["segment"] == "pinnacle") & (b["match_date"] >= ev0) & (b["match_date"] <= ev1)]
        ev = ev.sort_values(["match_date", "match_id"]).reset_index(drop=True)
        extra = {
            "clean": bool(np.array_equal(b["clv"].to_numpy(), b2["clv"].to_numpy())),
            "within_ceiling": bool(ev["clv"].mean() < 0.02),
            "leave_one_league_out": R.loo_league(ev, "clv"),
            "code_sha256_16": code,
            "mean_edge_at_open": float(ev["edge_at_open"].mean()),
            "outcome_mix": ev["outcome"].value_counts().to_dict(),
            "post_hoc": True,
        }
        r = R.record(reg, config, ev, "clv", "clv_log", extra, ledger)
        o = R.record(reg, config, ev, "overlap", "overlap_with_xg_rule", {"code_sha256_16": code}, ledger)
        by_div = ev.groupby("division")["clv"].agg(["size", "mean"])
        print(
            f"BW fair={fair}: CLV {100 * r.metric_value:+.2f}% [{100 * r.ci_lo:+.2f}, {100 * r.ci_hi:+.2f}] n={r.n} "
            f"edge@open {100 * extra['mean_edge_at_open']:.2f}% overlap with xg rule {o.metric_value:.2f} "
            f"mix {extra['outcome_mix']}"
        )
        print("   by div", {k: (int(v["size"]), round(100 * v["mean"], 2)) for k, v in by_div.iterrows()})


if __name__ == "__main__":
    main()
