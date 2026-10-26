"""Score the registered R6 hypotheses (family `closing-line-prediction`) and record every grid point through the
research harness.

  run.py h1   closing-line-prediction-cross-book-linear-v1   (estimator: ols | ridge)
  run.py h2   closing-line-prediction-cross-book-gbm-v1      (hgb: shallow | deeper)
  run.py h3   closing-line-prediction-book-informativeness-v1 (slope_type: marginal | partial)

H1/H2: walk-forward by season over the evaluation window 2019/20-2024/25 (fit on every earlier Pinnacle-segment
season from 2012/13), run twice to check determinism, then run_trial for the primary `delta_kl_bits` (league =
division for the leave-one-league-out CIs) and the registered secondaries. The post segment (2025-07+, Betfair
Exchange close) is recorded under the registered `post_segment` window. Per-match frames and a stability table
go to data/research/r6/<id>__<config>.parquet / __stability.csv.

H3: see h3.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r6lib as L  # noqa: E402

from pitch_edge.research.ledger import Ledger  # noqa: E402
from pitch_edge.research.registry import load  # noqa: E402
from pitch_edge.research.trial import TrialResult, run_trial  # noqa: E402

REG_DIR = L.registry_dir()
LEDGER = L.ledger_path()
IDS = {
    "h1": "closing-line-prediction-cross-book-linear-v1",
    "h2": "closing-line-prediction-cross-book-gbm-v1",
    "h3": "closing-line-prediction-book-informativeness-v1",
}
CODE_FILES = [HERE / "run.py", HERE / "r6lib.py", HERE / "h3.py"]
KL_CEILING, R2_CEILING = 0.001, 0.05
EVAL_SEASONS = [2017, 2018] if L.DRY else [2019, 2020, 2021, 2022, 2023, 2024]


def record(
    reg,
    config: dict,
    frame: pd.DataFrame,
    value_col: str,
    metric: str,
    extra: dict,
    ledger: Ledger,
    window: str = "evaluation",
    id_col: str = "match_id",
) -> TrialResult:
    """run_trial (league = division) with retry on a DuckDB lock, then verify the row is in the ledger."""
    pdf = pd.DataFrame(
        {
            "decision_id": frame[id_col].to_numpy(),
            "block": pd.to_datetime(frame["match_date"]).dt.strftime("%Y-%m-%d").to_numpy(),
            "value": frame[value_col].to_numpy(dtype=float),
            "league": frame["division"].to_numpy(),
        }
    )
    before = len(ledger.trials(hypothesis_id=reg.id))
    for attempt in range(30):
        try:
            res = run_trial(
                reg,
                config,
                pdf,
                metric,
                ledger,
                L.data_snapshot_label(),
                extra,
                n_boot=L.N_BOOT,
                seed=L.SEED,
                window=window,
                code_files=CODE_FILES,
            )
            break
        except duckdb.IOException as exc:
            if "lock" not in str(exc).lower():
                raise
            time.sleep(min(2 + 2 * attempt, 30))
    else:
        raise RuntimeError("ledger stayed locked")
    after = ledger.trials(hypothesis_id=reg.id)
    if len(after) != before + 1 or res.trial_id not in set(after["trial_id"]):
        raise RuntimeError(f"trial {res.trial_id} not found in the ledger after recording")
    print(
        f"  {metric:32s} {res.metric_value:+.4g} [{res.ci_lo:+.4g}, {res.ci_hi:+.4g}] p={res.p_value:.4f} n={res.n} "
        f"blocks={res.n_blocks} ({window})"
    )
    return res


def stability(s: pd.DataFrame) -> pd.DataFrame:
    s = L.add_r2_value(s)
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
                "kl_open": g["kl_open"].mean(),
                "kl_base": g["kl_base"].mean(),
                "kl_model": g["kl_model"].mean(),
                "delta_r2_move_pp": 100 * (g["sse_base"].sum() - g["sse_model"].sum()) / g["sse0"].sum(),
                "r2_base_pp": 100 * (1 - g["sse_base"].sum() / g["sse0"].sum()),
                "r2_model_pp": 100 * (1 - g["sse_model"].sum() / g["sse0"].sum()),
            }
        )
    return pd.DataFrame(rows)


def model_of(key: str, config: dict) -> str:
    if key == "h1":
        return config["estimator"]
    return {"tiny": "hgb_tiny", "res_tiny": "hgbres_tiny"}[config["hgb"]]


def run_prediction(key: str) -> None:
    reg = load(REG_DIR / f"{IDS[key]}.yaml")
    ledger = Ledger(LEDGER)
    panel = L.load_panel()
    out = L.OUT
    out.mkdir(parents=True, exist_ok=True)
    ridge_ref = None
    if key == "h2":  # paired reference: the H1 ridge model on the same rows (secondary delta_kl_bits_vs_ridge)
        ridge_ref = L.walk_forward(panel, "ridge", "power", EVAL_SEASONS, post=False)
    for config in reg.expand_grid():
        model = model_of(key, config)
        tag = f"{reg.id}__{model}"
        print(f"{reg.id} {config} -> {model}")
        s1 = L.walk_forward(panel, model, "power", EVAL_SEASONS, post=True)
        s2 = L.walk_forward(panel, model, "power", EVAL_SEASONS, post=True)
        deterministic = bool(
            np.array_equal(s1["kl_model"].to_numpy(), s2["kl_model"].to_numpy())
            and np.array_equal(s1["kl_base"].to_numpy(), s2["kl_base"].to_numpy())
        )
        ev = s1[(s1["segment"] == "pinnacle") & (s1["match_date"] >= L.EVAL_START) & (s1["match_date"] <= L.EVAL_END)]
        ev = L.add_r2_value(ev.sort_values(["match_date", "match_id"]).reset_index(drop=True))
        post = s1[s1["segment"] == "post_betfair"].sort_values(["match_date", "match_id"]).reset_index(drop=True)
        s1.to_parquet(out / f"{tag}.parquet")
        st = stability(ev)
        st.to_csv(out / f"{tag}__stability.csv", index=False)
        d_kl = float(ev["delta_kl_bits"].mean())
        d_r2 = float(ev["delta_r2_move"].mean())
        extra = {
            "clean": deterministic,
            "pit_assertion": (
                "assert_open_only on every test fold: features read only OPEN_COLUMNS and are identical "
                f"with closes/results blanked; training seasons precede test seasons; {len(ev)} eval rows"
            ),
            "within_ceiling": bool(d_kl < KL_CEILING and d_r2 < R2_CEILING),
            "r2_base": float(1 - ev["sse_base"].sum() / ev["sse0"].sum()),
            "r2_model": float(1 - ev["sse_model"].sum() / ev["sse0"].sum()),
            "kl_open": float(ev["kl_open"].mean()),
            "kl_base": float(ev["kl_base"].mean()),
            "kl_model": float(ev["kl_model"].mean()),
            "ridge_lambdas": sorted(set(ev["ridge_lambda"])) if model == "ridge" else None,
            "n_by_season": {str(k): int(v) for k, v in ev.groupby("season_start").size().items()},
            "row_counts": {"eval": len(ev), "post": len(post), "panel": len(panel)},
        }
        record(reg, config, ev, "delta_kl_bits", "delta_kl_bits", extra, ledger)
        small = {"clean": deterministic}
        record(reg, config, ev, "delta_r2_move", "delta_r2_move", small, ledger)
        record(reg, config, ev, "kl_gain_vs_open_bits", "kl_gain_vs_open_bits", small, ledger)
        evr = ev.dropna(subset=["delta_ll_result_bits"])
        record(reg, config, evr, "delta_ll_result_bits", "delta_ll_result_bits", small, ledger)
        sh = L.walk_forward(panel, model, "shin", EVAL_SEASONS, post=False)
        sh = sh[(sh["match_date"] >= L.EVAL_START) & (sh["match_date"] <= L.EVAL_END)].sort_values(
            ["match_date", "match_id"]
        )
        sh.to_parquet(out / f"{tag}__shin.parquet")
        record(reg, config, sh, "delta_kl_bits", "delta_kl_bits_shin", small, ledger)
        if ridge_ref is not None:
            m = ev.merge(
                ridge_ref[["match_id", "kl_model"]].rename(columns={"kl_model": "kl_ridge"}),
                on="match_id",
                validate="one_to_one",
            )
            assert len(m) == len(ev)
            m["vs_ridge"] = m["kl_ridge"] - m["kl_model"]
            record(reg, config, m, "vs_ridge", "delta_kl_bits_vs_ridge", small, ledger)
        if len(post):
            record(
                reg, config, post, "delta_kl_bits", "delta_kl_bits_post_betfair", small, ledger, window="post_segment"
            )
            record(
                reg,
                config,
                post,
                "kl_gain_vs_open_bits",
                "kl_gain_vs_open_bits_post_betfair",
                small,
                ledger,
                window="post_segment",
            )
        print(st.head(8).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        (out / f"{tag}__extra.json").write_text(json.dumps(extra, indent=2, default=str))


if __name__ == "__main__":
    which = sys.argv[1]
    if which in ("h1", "h2"):
        run_prediction(which)
    elif which == "h3":
        import h3

        h3.main(IDS["h3"], record)
    else:
        raise SystemExit(__doc__)
