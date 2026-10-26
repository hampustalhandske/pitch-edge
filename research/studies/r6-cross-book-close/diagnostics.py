"""POST-HOC diagnostics for R6 (not registered, not counted trials; written after the counted runs).

1. decomposition: KL gain over the Pinnacle open of the baseline and of each model, exploratory vs evaluation
   (why the evaluation delta exceeds the exploratory one).
2. placebo: the cross-book block of the TEST rows permuted across matches within match-day (models trained on
   real features); the delta over the baseline should vanish or turn negative.
3. drop-one-book: ridge and res_tiny re-fitted with one core book blanked everywhere (it becomes 'missing');
   the loss in delta_kl_bits is that book's contribution.
4. asynchrony screens on the counted res_tiny frame: rows where some core book quotes above the same-row Max,
   and same-day snapshot rows (match on the Friday/Tuesday of the snapshot).
Outputs data/research/r6/diagnostics.json. Usage: diagnostics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r6lib as L  # noqa: E402

from pitch_edge.research.stats import block_bootstrap  # noqa: E402

SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]


def ci(v: np.ndarray, days: np.ndarray) -> list[float]:
    r = block_bootstrap(v, days, n_boot=L.N_BOOT, seed=L.SEED)
    return [r.estimate, r.lo, r.hi]


def placebo(panel: pd.DataFrame, model: str) -> pd.DataFrame:
    """Walk-forward with the test rows' cross-book block permuted within match-day."""
    pin = panel[panel["segment"] == "pinnacle"]
    rng = np.random.default_rng(L.SEED)
    orig = L.build_features
    out = []
    for s in SEASONS:
        train, test = pin[pin["season_start"] < s], pin[pin["season_start"] == s]
        days = test["match_date"].to_numpy()
        perm = np.arange(len(test))
        for d in np.unique(days):
            idx = np.flatnonzero(days == d)
            perm[idx] = rng.permutation(idx)

        def patched(df: pd.DataFrame, method: str, _test_ids=set(test["match_id"])) -> L.Features:  # noqa: B006
            f = orig(df, method)
            if len(df) == len(test) and set(df["match_id"]) == _test_ids:  # noqa: B023 (patch is undone in this iteration)
                f.book = [b[perm] for b in f.book]  # noqa: B023
            return f

        L.build_features = patched
        try:
            mh_b, mh_m, _ = L.fit_predict(model, train, test, "power")
        finally:
            L.build_features = orig
        out.append(L.score(test, "power", mh_b, mh_m))
    return pd.concat(out, ignore_index=True)


def drop_book(panel: pd.DataFrame, model: str, book: str) -> pd.DataFrame:
    p = panel.copy()
    for s in "HDA":
        p[f"{book}{s}"] = np.nan
    return L.walk_forward(p, model, "power", SEASONS)


def main() -> None:
    panel = L.load_panel()
    res: dict = {}
    # 1. decomposition
    var = pd.read_csv(L.OUT / "explore" / "variants.csv").set_index("model")
    dec = {
        "exploratory_2014_2018": {
            m: {
                "kl_open": var.loc[m, "kl_open"],
                "gain_base_vs_open": var.loc[m, "kl_open"] - var.loc[m, "kl_base"],
                "gain_model_vs_open": var.loc[m, "kl_open"] - var.loc[m, "kl_model"],
            }
            for m in ("ridge", "hgbres_tiny", "ols", "hgb_tiny")
        }
    }
    for tag in (
        "closing-line-prediction-cross-book-linear-v1__ridge",
        "closing-line-prediction-cross-book-gbm-v1__hgbres_tiny",
    ):
        s = pd.read_parquet(L.OUT / f"{tag}.parquet")
        ev = s[s["segment"] == "pinnacle"]
        dec.setdefault("evaluation_by_season", {})[tag] = {
            int(k): {
                "kl_open": g["kl_open"].mean(),
                "gain_base_vs_open": (g["kl_open"] - g["kl_base"]).mean(),
                "gain_model_vs_open": (g["kl_open"] - g["kl_model"]).mean(),
            }
            for k, g in ev.groupby("season_start")
        }
        dec.setdefault("evaluation_all", {})[tag] = {
            "kl_open": ev["kl_open"].mean(),
            "gain_base_vs_open": (ev["kl_open"] - ev["kl_base"]).mean(),
            "gain_model_vs_open": (ev["kl_open"] - ev["kl_model"]).mean(),
        }
    res["decomposition"] = dec
    print(json.dumps(dec["evaluation_all"], indent=1))
    # 2. placebo
    for model in ("ridge", "hgbres_tiny"):
        s = placebo(panel, model)
        res[f"placebo_{model}"] = ci(s["delta_kl_bits"].to_numpy(), s["match_date"].dt.strftime("%Y-%m-%d").to_numpy())
        print("placebo", model, res[f"placebo_{model}"])
    # 3. drop-one-book
    for model in ("ridge", "hgbres_tiny"):
        full = pd.read_parquet(
            L.OUT / f"closing-line-prediction-cross-book-{'linear' if model == 'ridge' else 'gbm'}-v1__{model}.parquet"
        )
        full = full[full["segment"] == "pinnacle"].set_index("match_id")
        for b in L.CORE_BOOKS:
            s = drop_book(panel, model, b).set_index("match_id")
            loss = (s["kl_model"] - full.loc[s.index, "kl_model"]).to_numpy()  # > 0: dropping b hurts
            days = pd.to_datetime(s["match_date"]).dt.strftime("%Y-%m-%d").to_numpy()
            res.setdefault(f"drop_book_{model}", {})[b] = {
                "loss_bits": ci(loss, days),
                "delta_kl_without": float(s["delta_kl_bits"].mean()),
            }
            print("drop", model, b, res[f"drop_book_{model}"][b])
    # 4. asynchrony screens (counted res_tiny frame)
    s = pd.read_parquet(L.OUT / "closing-line-prediction-cross-book-gbm-v1__hgbres_tiny.parquet")
    s = s[s["segment"] == "pinnacle"].merge(
        panel[
            [
                "match_id",
                "snapshot_date",
                *[f"{b}{o}" for b in L.CORE_BOOKS for o in "HDA"],
                *[f"max{o}_open" for o in "HDA"],
            ]
        ],
        on="match_id",
    )
    above = np.zeros(len(s), dtype=bool)
    for b in L.CORE_BOOKS:
        for o in "HDA":
            above |= (s[f"{b}{o}"] > s[f"max{o}_open"] + 1e-9).to_numpy()
    same_day = (pd.to_datetime(s["match_date"]) == pd.to_datetime(s["snapshot_date"])).to_numpy()
    days = pd.to_datetime(s["match_date"]).dt.strftime("%Y-%m-%d").to_numpy()
    for name, mask in (
        ("book_above_max", above),
        ("no_book_above_max", ~above),
        ("same_day", same_day),
        ("not_same_day", ~same_day),
    ):
        res.setdefault("screens_res_tiny", {})[name] = {
            "n": int(mask.sum()),
            "delta_kl_bits": ci(s["delta_kl_bits"].to_numpy()[mask], days[mask]),
        }
    print(json.dumps(res["screens_res_tiny"], indent=1))
    (L.OUT / "diagnostics.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
