"""R4 estimation: decision frames, walk-forward OLS (baseline vs baseline + crowd feature), per-decision Δ OOS R².

Frames (built from `panel.parquet`, one row per decision hour H; `h_before` = hours from H to kickoff, 24..2):

- `h1_frame(panel, L)` — attention vs move SIZE, one row per (match, side, H) for side in {home, away} whose club
  has Bluesky features. Target y = |p_end - p_0| of that side's "club wins" market. Baseline: hour-to-kickoff
  fixed effects, p_H (1 - p_H), |p_H - p_H6| (own recent move size), log(1 + trades24) (Polymarket's own
  activity). Crowd: log(1 + n_L) and abn_L of that club. Decision = (match, side).
- `h2_frame(panel, L, kind)` — skew vs move DIRECTION, one row per (match, H) with Bluesky features for both clubs.
  Target y = (p_end - p_0)_home - (p_end - p_0)_away. Baseline: hour fixed effects, own recent move
  (p_H - p_H6)_home - (p_H - p_H6)_away, level p_H_home - p_H_away. Crowd (one regressor): mention skew
  abn_L_home - abn_L_away (`kind="mention"`) or lexicon skew sent_L_home - sent_L_away (`kind="sentiment"`).
  Decision = match.

Rows need finite p_H, p_H6, p_0 and p_end on the sides used. Walk-forward: for each fold start t_k (the evaluation
start, then the first of every following month) both models are fitted by OLS on every row whose kickoff is
before t_k (exploratory rows included) and predict the rows with kickoff in [t_k, t_k+1). A decision's value is
the mean over its hours of (e_base^2 - e_full^2) divided by the evaluation rows' mean of (y - mean_train y)^2 —
the pooled-variance scale of an OOS R² — so the mean of values is the Δ OOS R² (positive = the crowd feature
helps) up to the per-decision weighting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIDES2 = ("home", "away")


def _hour_fe(h: pd.Series) -> pd.DataFrame:
    return pd.get_dummies(h.astype(int), prefix="hb", drop_first=True, dtype=float)


def h1_frame(panel: pd.DataFrame, L: int) -> pd.DataFrame:
    parts = []
    for s in SIDES2:
        need = [f"p_H_{s}", f"p_H6_{s}", f"p_0_{s}", f"p_end_{s}", f"bs_n_{L}_{s}", f"bs_abn_{L}_{s}"]
        d = panel.dropna(subset=need)
        f = pd.DataFrame(
            {
                "decision_id": d.group_id + "|" + s,
                "group_id": d.group_id,
                "side": s,
                "league": d.league_code,
                "ko": d.ko,
                "block": d.ko.dt.strftime("%Y-%m-%d"),
                "h_before": d.h_before,
                "y": (d[f"p_end_{s}"] - d[f"p_0_{s}"]).abs(),
                "pq": d[f"p_H_{s}"] * (1 - d[f"p_H_{s}"]),
                "absmove6": (d[f"p_H_{s}"] - d[f"p_H6_{s}"]).abs(),
                "logtrades": np.log1p(d[f"trades24_{s}"]),
                "logn": np.log1p(d[f"bs_n_{L}_{s}"]),
                "abn": d[f"bs_abn_{L}_{s}"],
            },
            index=d.index,
        )
        parts.append(f)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["ko", "decision_id", "h_before"], ascending=[True, True, False]).reset_index(drop=True)


H1_BASE = ["pq", "absmove6", "logtrades"]
H1_CROWD = ["logn", "abn"]


def h2_frame(panel: pd.DataFrame, L: int, kind: str) -> pd.DataFrame:
    col = {"mention": "abn", "sentiment": "sent"}[kind]
    need = [f"{c}_{s}" for s in SIDES2 for c in ("p_H", "p_H6", "p_0", "p_end")]
    need += [f"bs_{col}_{L}_{s}" for s in SIDES2]
    d = panel.dropna(subset=need)
    mv = lambda s: d[f"p_end_{s}"] - d[f"p_0_{s}"]  # noqa: E731
    rm = lambda s: d[f"p_H_{s}"] - d[f"p_H6_{s}"]  # noqa: E731
    f = pd.DataFrame(
        {
            "decision_id": d.group_id,
            "group_id": d.group_id,
            "league": d.league_code,
            "ko": d.ko,
            "block": d.ko.dt.strftime("%Y-%m-%d"),
            "h_before": d.h_before,
            "y": mv("home") - mv("away"),
            "move6": rm("home") - rm("away"),
            "level": d["p_H_home"] - d["p_H_away"],
            "skew": d[f"bs_{col}_{L}_home"] - d[f"bs_{col}_{L}_away"],
        },
        index=d.index,
    )
    return f.sort_values(["ko", "h_before"], ascending=[True, False]).reset_index(drop=True)


H2_BASE = ["move6", "level"]
H2_CROWD = ["skew"]


def design(df: pd.DataFrame, cols: list[str], fe_levels: list[int]) -> np.ndarray:
    fe = pd.get_dummies(pd.Categorical(df.h_before.astype(int), categories=fe_levels), dtype=float)
    fe = fe.iloc[:, 1:].to_numpy()
    return np.column_stack([np.ones(len(df)), fe, df[cols].to_numpy(float)])


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def fold_starts(eval_from: str, eval_to: str) -> list[pd.Timestamp]:
    t0 = pd.Timestamp(eval_from, tz="UTC")
    end = pd.Timestamp(eval_to, tz="UTC") + pd.Timedelta(days=1)
    starts = [t0]
    m = (t0 + pd.offsets.MonthBegin(1)).normalize()
    while m < end:
        starts.append(m)
        m = m + pd.offsets.MonthBegin(1)
    return starts + [end]


def walk_forward(
    df: pd.DataFrame, base: list[str], crowd: list[str], eval_from: str, eval_to: str
) -> tuple[pd.DataFrame, list[dict]]:
    """OOS predictions of the baseline and full models for rows with kickoff in [eval_from, eval_to]."""
    fe_levels = sorted(df.h_before.astype(int).unique(), reverse=True)
    starts = fold_starts(eval_from, eval_to)
    out, folds = [], []
    for a, b in zip(starts[:-1], starts[1:], strict=True):
        tr = df[df.ko < a]
        te = df[(df.ko >= a) & (df.ko < b)]
        if len(te) == 0:
            continue
        assert tr.ko.max() < a <= te.ko.min()
        yb = tr.y.to_numpy()
        bb = ols(design(tr, base, fe_levels), yb)
        bf = ols(design(tr, base + crowd, fe_levels), yb)
        te = te.copy()
        te["pred_base"] = design(te, base, fe_levels) @ bb
        te["pred_full"] = design(te, base + crowd, fe_levels) @ bf
        te["ybar_train"] = yb.mean()
        out.append(te)
        folds.append(
            {
                "fold_start": str(a.date()),
                "n_train_rows": len(tr),
                "n_test_rows": len(te),
                "crowd_coef": dict(zip(crowd, bf[-len(crowd) :].round(6).tolist(), strict=True)),
            }
        )
    return pd.concat(out, ignore_index=True), folds


def per_decision(pred: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per decision: value = mean over its hours of (e_base^2 - e_full^2) / pooled OOS variance."""
    e_b = (pred.y - pred.pred_base) ** 2
    e_f = (pred.y - pred.pred_full) ** 2
    sst = float(np.mean((pred.y - pred.ybar_train) ** 2))
    p = pred.assign(d=(e_b - e_f) / sst)
    dec = (
        p.groupby("decision_id")
        .agg(block=("block", "first"), league=("league", "first"), value=("d", "mean"), n_hours=("d", "size"))
        .reset_index()
        .sort_values(["block", "decision_id"])
        .reset_index(drop=True)
    )
    summ = {
        "rows": len(pred),
        "decisions": len(dec),
        "sst_per_row": sst,
        "mse_base": float(e_b.mean()),
        "mse_full": float(e_f.mean()),
        "oos_r2_base": float(1 - e_b.mean() / sst),
        "oos_r2_full": float(1 - e_f.mean() / sst),
        "delta_oos_r2_pooled": float((e_b.mean() - e_f.mean()) / sst),
    }
    return dec, summ
