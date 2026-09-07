"""Event study: does a newly announced strong-tendency referee move the no-vig price?

Pre-registered in CASE_STUDY.md before the first run:
  H1: for fixtures where the announced referee's home-bias or cards-per-game z-score exceeds +1
      (or is below −1), the absolute pre→post-announcement move in the no-vig home probability is
      larger than for neutral referees (|z| ≤ 0.5).
  H0: no difference. Test: Welch t on |move| between the two groups, one-sided; α = 0.05.
The result is reported whichever way it goes; with too few paired announcements the table says
"insufficient" and the count needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

REF_MIN_MATCHES = 20


def referee_tendency_table(matches: pd.DataFrame, min_matches: int = REF_MIN_MATCHES) -> pd.DataFrame:
    """Per referee: matches, cards per game, home share of cards, and z-scores vs all referees.
    Uses the whole history (a descriptive table for the dossier), not a pre-match expanding value."""
    need = {"referee", "home_yellows", "away_yellows"}
    if matches.empty or not need.issubset(matches.columns):
        return pd.DataFrame(
            columns=["referee", "matches", "cards_per_game", "home_card_share", "z_cards", "z_home_bias", "as_of"]
        )
    df = matches.dropna(subset=["referee"]).copy()
    df = df[df["referee"].astype(str).str.strip() != ""]
    if df.empty:
        return pd.DataFrame(
            columns=["referee", "matches", "cards_per_game", "home_card_share", "z_cards", "z_home_bias", "as_of"]
        )
    hr = df.get("home_reds", pd.Series(0, index=df.index)).fillna(0)
    ar = df.get("away_reds", pd.Series(0, index=df.index)).fillna(0)
    df["home_cards"] = df["home_yellows"].fillna(0) + hr
    df["away_cards"] = df["away_yellows"].fillna(0) + ar
    df["cards"] = df["home_cards"] + df["away_cards"]
    g = df.groupby("referee")
    out = pd.DataFrame(
        {
            "matches": g.size(),
            "cards_per_game": g["cards"].mean(),
            "home_card_share": g["home_cards"].sum() / g["cards"].sum().replace(0, np.nan),
            "first_match": g["date"].min() if "date" in df else pd.NaT,
            "last_match": g["date"].max() if "date" in df else pd.NaT,
        }
    ).reset_index()
    out = out[out["matches"] >= min_matches].copy()
    for src, dst in (("cards_per_game", "z_cards"), ("home_card_share", "z_home_bias")):
        sd = out[src].std(ddof=1)
        out[dst] = (out[src] - out[src].mean()) / sd if sd and sd > 0 else 0.0
    out["as_of"] = pd.Timestamp(df["date"].max()) if "date" in df else pd.Timestamp.utcnow().tz_localize(None)
    return out.sort_values("matches", ascending=False).reset_index(drop=True)


def referee_announcement_study(
    paired: pd.DataFrame,
    tendencies: pd.DataFrame,
    outcome: str = "home",
    strong_z: float = 1.0,
    neutral_z: float = 0.5,
    min_per_group: int = 15,
) -> pd.DataFrame:
    """One-row-per-group table + a verdict row. `paired` comes from `pair_with_snapshots`."""
    cols = ["group", "n", "mean_abs_move", "median_abs_move", "mean_signed_move", "t_stat", "p_one_sided", "verdict"]
    if paired.empty or tendencies.empty:
        return pd.DataFrame(
            [
                {
                    "group": "ALL",
                    "n": 0,
                    "mean_abs_move": np.nan,
                    "median_abs_move": np.nan,
                    "mean_signed_move": np.nan,
                    "t_stat": np.nan,
                    "p_one_sided": np.nan,
                    "verdict": f"insufficient: 0 paired announcements (need ≥ {min_per_group} per group)",
                }
            ],
            columns=cols,
        )
    p = paired[paired["outcome"] == outcome].merge(
        tendencies[["referee", "z_cards", "z_home_bias"]], on="referee", how="left"
    )
    p["z"] = p[["z_cards", "z_home_bias"]].abs().max(axis=1)
    strong = p[p["z"] >= strong_z]
    neutral = p[p["z"] <= neutral_z]
    rows = []
    for name, grp in (("strong_tendency", strong), ("neutral", neutral)):
        rows.append(
            {
                "group": name,
                "n": int(len(grp)),
                "mean_abs_move": float(grp["move"].abs().mean()) if len(grp) else np.nan,
                "median_abs_move": float(grp["move"].abs().median()) if len(grp) else np.nan,
                "mean_signed_move": float(grp["move"].mean()) if len(grp) else np.nan,
                "t_stat": np.nan,
                "p_one_sided": np.nan,
                "verdict": "",
            }
        )
    if len(strong) >= min_per_group and len(neutral) >= min_per_group:
        t, p2 = stats.ttest_ind(strong["move"].abs(), neutral["move"].abs(), equal_var=False)
        p1 = p2 / 2 if t > 0 else 1 - p2 / 2
        verdict = (
            "H1 supported (strong-tendency referees move the price more)"
            if p1 < 0.05
            else "H0 not rejected (no detectable extra move)"
        )
        rows.append(
            {
                "group": "ALL",
                "n": int(len(strong) + len(neutral)),
                "mean_abs_move": float(p["move"].abs().mean()),
                "median_abs_move": float(p["move"].abs().median()),
                "mean_signed_move": float(p["move"].mean()),
                "t_stat": float(t),
                "p_one_sided": float(p1),
                "verdict": verdict,
            }
        )
    else:
        rows.append(
            {
                "group": "ALL",
                "n": int(len(p)),
                "mean_abs_move": float(p["move"].abs().mean()) if len(p) else np.nan,
                "median_abs_move": float(p["move"].abs().median()) if len(p) else np.nan,
                "mean_signed_move": float(p["move"].mean()) if len(p) else np.nan,
                "t_stat": np.nan,
                "p_one_sided": np.nan,
                "verdict": f"insufficient: {len(strong)} strong / {len(neutral)} neutral paired announcements (need ≥ {min_per_group} each)",
            }
        )
    return pd.DataFrame(rows, columns=cols)
