"""Score the registered R3 feature-group hypotheses (family `closing-line-prediction`) on their evaluation
windows and record every grid point through the research harness.

For each registration and each `devig` grid point: walk-forward predictions for every season in the
evaluation window (plus the post-2025-07 Betfair-close segment), a determinism check (the walk-forward is
run twice and must agree exactly), then `run_trial` for the primary metric `delta_kl_bits` and the
secondaries `delta_r2_move`, `delta_ll_result_bits`, `delta_kl_bits_post_betfair`. Per-match frames go
to data/research/r3/features/<id>__<devig>.parquet and a stability table to <id>__<devig>__stability.csv.

Usage: run.py <slug> [<slug> ...]   slugs: pi-rating clubelo xg-finishing manager-change schedule
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

GROUP_OF = {
    "pi-rating": "pi",
    "clubelo": "clubelo",
    "xg-finishing": "xg",
    "manager-change": "manager",
    "schedule": "schedule",
}
CODE_FILES = [
    HERE / "run.py",
    HERE.parent / "r3-common" / "r3lib.py",
    HERE.parent / "r3-common" / "features.py",
    HERE.parent / "r3-common" / "pull.py",
    HERE.parent / "r3-common" / "lake.py",
]
OUT = R.D / "features"
R2_CEILING, KL_CEILING = 0.02, 0.002
# R3_CHECK_ONLY=1: compute every grid point and compare it with the ledger's earliest trial; record nothing.
CHECK_ONLY = bool(os.environ.get("R3_CHECK_ONLY"))


def stability(s: pd.DataFrame) -> pd.DataFrame:
    s = R.add_r2_value(s)
    rows = []
    for key, g in (
        [("all", s)]
        + [(f"season {k}", g) for k, g in s.groupby("season_start")]
        + [(f"div {k}", g) for k, g in s.groupby("division")]
    ):
        rows.append(
            {
                "slice": key,
                "n": len(g),
                "delta_kl_bits": g["delta_kl_bits"].mean(),
                "delta_r2_move_pp": 100 * (g["sse_base"].sum() - g["sse_model"].sum()) / g["sse0"].sum(),
            }
        )
    return pd.DataFrame(rows)


def main(slugs: list[str]) -> None:
    out = OUT if not os.environ.get("R3_DRY_LEDGER") else R.D / "dry"
    out.mkdir(parents=True, exist_ok=True)
    panel = R.load_panel()
    # R3_DRY_LEDGER / R3_DRY_REG_DIR point a code check at a scratch ledger and scratch registrations
    # (exploratory windows only); those rows are not counted trials.
    ledger = (
        Ledger(os.environ["R3_DRY_LEDGER"])
        if os.environ.get("R3_DRY_LEDGER")
        else Ledger(R.REPO / "data" / "research" / "ledger.duckdb")
    )
    reg_dir = Path(os.environ.get("R3_DRY_REG_DIR", str(REPO_REG)))
    code = R.code_sha(CODE_FILES)
    for slug in slugs:
        reg = load(reg_dir / f"closing-line-prediction-{slug}-v1.yaml")
        group = R.GROUPS[GROUP_OF[slug]]
        ev0, ev1 = pd.Timestamp(reg.window.evaluation[0]), pd.Timestamp(reg.window.evaluation[1])
        seasons = list(range(ev0.year if ev0.month >= 7 else ev0.year - 1, ev1.year))
        for config in reg.expand_grid():
            m = config["devig"]
            s1 = R.walk_forward(panel, group, m, seasons, post=True)
            s2 = R.walk_forward(panel, group, m, seasons, post=True)
            deterministic = bool(
                np.array_equal(s1["delta_kl_bits"].to_numpy(), s2["delta_kl_bits"].to_numpy())
                and np.array_equal(s1["sse_model"].to_numpy(), s2["sse_model"].to_numpy())
            )
            ev = s1[(s1["segment"] == "pinnacle") & (s1["match_date"] >= ev0) & (s1["match_date"] <= ev1)]
            ev = R.add_r2_value(ev.sort_values(["match_date", "match_id"]).reset_index(drop=True))
            post = s1[s1["segment"] == "post_betfair"].sort_values(["match_date", "match_id"]).reset_index(drop=True)
            s1.to_parquet(out / f"{reg.id}__{m}.parquet")
            st = stability(ev)
            st.to_csv(out / f"{reg.id}__{m}__stability.csv", index=False)
            d_r2 = float(ev["delta_r2_move"].mean())
            d_kl = float(ev["delta_kl_bits"].mean())
            extra = {
                "clean": deterministic,
                "pit_assertion": f"{group.known_at} < snapshot_date asserted on {len(ev)} eval rows",
                "within_ceiling": bool(d_r2 < R2_CEILING and d_kl < KL_CEILING),
                "leave_one_league_out": R.loo_league(ev, "delta_kl_bits"),
                "code_files": [str(f.relative_to(R.REPO)) for f in CODE_FILES],
                "code_sha256_16": code,
                "r2_base": float(1 - ev["sse_base"].sum() / ev["sse0"].sum()),
                "r2_model": float(1 - ev["sse_model"].sum() / ev["sse0"].sum()),
                "kl_open": float(ev["kl_open"].mean()),
                "kl_base": float(ev["kl_base"].mean()),
                "n_by_season": {str(k): int(v) for k, v in ev.groupby("season_start").size().items()},
                "row_counts": {"eval": len(ev), "post": len(post), "panel": len(panel)},
            }
            res = R.record(reg, config, ev, "delta_kl_bits", "delta_kl_bits", extra, ledger, CODE_FILES, CHECK_ONLY)
            if CHECK_ONLY:
                for col, metric, fr in (
                    ("delta_r2_move", "delta_r2_move", ev),
                    ("delta_ll_result_bits", "delta_ll_result_bits", ev.dropna(subset=["delta_ll_result_bits"])),
                    ("delta_kl_bits", "delta_kl_bits_post_betfair", post),
                ):
                    if len(fr):
                        R.record(reg, config, fr, col, metric, {}, ledger, CODE_FILES, True)
                continue
            print(
                f"{reg.id} {m}: delta_kl_bits {res.metric_value:+.3g} [{res.ci_lo:+.3g}, {res.ci_hi:+.3g}] "
                f"p={res.p_value:.4f} n={res.n} clean={deterministic}"
            )
            r2 = R.record(
                reg, config, ev, "delta_r2_move", "delta_r2_move", {"code_sha256_16": code}, ledger, CODE_FILES
            )
            print(
                f"   delta_r2_move {100 * r2.metric_value:+.3f} pp [{100 * r2.ci_lo:+.3f}, {100 * r2.ci_hi:+.3f}]"
                f"  R2 base {extra['r2_base']:.4f} model {extra['r2_model']:.4f}"
            )
            evr = ev.dropna(subset=["delta_ll_result_bits"])
            ll = R.record(
                reg,
                config,
                evr,
                "delta_ll_result_bits",
                "delta_ll_result_bits",
                {"code_sha256_16": code},
                ledger,
                CODE_FILES,
            )
            print(f"   delta_ll_result_bits {ll.metric_value:+.3g} [{ll.ci_lo:+.3g}, {ll.ci_hi:+.3g}] n={ll.n}")
            if len(post):
                pr = R.record(
                    reg,
                    config,
                    post,
                    "delta_kl_bits",
                    "delta_kl_bits_post_betfair",
                    {"code_sha256_16": code},
                    ledger,
                    CODE_FILES,
                )
                print(
                    f"   post Betfair delta_kl_bits {pr.metric_value:+.3g} [{pr.ci_lo:+.3g}, {pr.ci_hi:+.3g}] n={pr.n}"
                )
            print(st.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


REPO_REG = R.REPO / "research" / "registry"

if __name__ == "__main__":
    main(sys.argv[1:])
