"""Referee tendency features from the spine's own match stats.

football-data.co.uk publishes the referee plus fouls/cards per side for the
English divisions, so referee × home-bias is a *real* feature here: for each
match we compute the referee's expanding (strictly prior) averages of cards
per game and the share of fouls called against the away side — the
"referees call ~15% fewer fouls against the home team" effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def referee_features(matches: pd.DataFrame, min_prior_matches: int = 5) -> pd.DataFrame:
    needed = {"referee", "home_fouls", "away_fouls", "home_yellows", "away_yellows"}
    if not needed.issubset(matches.columns):
        return pd.DataFrame({"match_id": matches["match_id"]})
    df = matches.dropna(subset=["referee"]).sort_values("date").copy()
    df["cards"] = df[["home_yellows", "away_yellows"]].sum(axis=1) + df.get("home_reds", 0).fillna(0) + df.get(
        "away_reds", 0
    ).fillna(0)
    fouls_total = df["home_fouls"] + df["away_fouls"]
    df["away_foul_share"] = np.where(fouls_total > 0, df["away_fouls"] / fouls_total, np.nan)
    df["home_card_share"] = np.where(
        df["cards"] > 0, (df["home_yellows"] + df.get("home_reds", 0).fillna(0)) / df["cards"], np.nan
    )

    g = df.groupby("referee")
    out = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "ref_prior_matches": g.cumcount(),
            "ref_cards_per_game": g["cards"].transform(lambda s: s.shift(1).expanding().mean()),
            "ref_away_foul_share": g["away_foul_share"].transform(lambda s: s.shift(1).expanding().mean()),
            "ref_home_card_share": g["home_card_share"].transform(lambda s: s.shift(1).expanding().mean()),
        }
    )
    mask = out["ref_prior_matches"] < min_prior_matches
    out.loc[mask, ["ref_cards_per_game", "ref_away_foul_share", "ref_home_card_share"]] = np.nan
    # home-bias score: >0 means this referee historically calls more fouls on the away side than neutral
    out["ref_home_bias"] = out["ref_away_foul_share"] - 0.5
    return out.reset_index(drop=True)
