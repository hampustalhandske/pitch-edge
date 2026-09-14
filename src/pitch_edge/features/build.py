"""Feature store builder.

Everything here is computed from information available *before kickoff* of
the match in question — rolling stats are shifted by one match, Elo is
updated sequentially, referee/weather/travel are pre-match facts. The single
leakage-prone input is bookmaker odds: only *early* (non-closing) prices are
ever exposed as features, and only in the explicitly-named `*_mkt` models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pitch_edge.data.alt.referee import referee_features
from pitch_edge.data.alt.travel import fatigue_index, rest_and_congestion, travel_distance

TEAM_STATS = ["goals", "shots", "shots_on_target", "corners"]
ROLL_WINDOWS = (5, 10)

BASE_FEATURES = [
    "elo_home",
    "elo_away",
    "elo_diff",
    "elo_exp_home",
    "home_rest_days",
    "away_rest_days",
    "home_matches_last_14d",
    "away_matches_last_14d",
    "away_travel_km",
    "away_fatigue_index",
    "ref_cards_per_game",
    "ref_home_bias",
    "wx_temperature_2m",
    "wx_precipitation",
    "wx_wind_speed_10m",
    # Phase 5 context (features/context.py, data/alt/wikipedia_attention.py) — all strictly pre-match
    "rot_home_minutes_7d",
    "rot_away_minutes_7d",
    "rot_home_days_since_any",
    "rot_away_days_since_any",
    "rot_home_midweek_cup",
    "rot_away_midweek_cup",
    "rot_load_diff",
    "pv_home_anom",
    "pv_away_anom",
    "pv_home_z",
    "pv_away_z",
    "pv_diff",
    # Phase 6 context (features/squad_value.py) — confirmed-lineup value + injury-absence proxy
    "sv_home_xi_value",
    "sv_away_xi_value",
    "sv_xi_value_diff",
    "sv_home_missing_pct",
    "sv_away_missing_pct",
    "sv_missing_pct_diff",
    # News sentiment (data/alt/news.py, features/context.py::news_context) — point-in-time RSS
    # sentiment/injury-item counts per side, strictly pre-match
    "news_sent_home",
    "news_sent_away",
    "news_sent_diff",
    "news_injury_items_home",
    "news_injury_items_away",
]
MARKET_FEATURES = ["mkt_home_p", "mkt_draw_p", "mkt_away_p", "mkt_overround"]


@dataclass
class EloConfig:
    k: float = 20.0
    home_advantage: float = 60.0
    initial: float = 1500.0
    goal_diff_multiplier: bool = True


@dataclass
class FeatureBuilder:
    elo: EloConfig = field(default_factory=EloConfig)
    windows: tuple[int, ...] = ROLL_WINDOWS
    early_odds_prefix: str = "PS"
    market_price_source: str = "none"

    def build(
        self,
        matches: pd.DataFrame,
        weather: pd.DataFrame | None = None,
        venues: pd.DataFrame | None = None,
        context: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """`context`: optional per-match frame (match_id + rot_*/pv_*/tm_referee columns, see
        `features/context.py`). A `tm_referee` column fills the spine's `referee` where it is missing,
        which is what activates `referee_features` on the fallback spine."""
        df = matches.sort_values(["date", "match_id"]).reset_index(drop=True).copy()
        df["date"] = pd.to_datetime(df["date"])
        if context is not None and not context.empty:
            ctx = context.drop_duplicates("match_id")
            ctx = ctx[[c for c in ctx.columns if c == "match_id" or c not in df.columns or c == "tm_referee"]]
            df = df.merge(ctx, on="match_id", how="left")
            if "tm_referee" in df:
                if "referee" not in df:
                    df["referee"] = df["tm_referee"]
                else:
                    df["referee"] = df["referee"].where(df["referee"].notna() & (df["referee"] != ""), df["tm_referee"])
        df["result"] = np.select(
            [df["home_goals"] > df["away_goals"], df["home_goals"] == df["away_goals"]], [0, 1], default=2
        )
        df["total_goals"] = df["home_goals"] + df["away_goals"]

        df = self._add_elo(df)
        df = self._add_rolling_form(df)
        df = df.merge(rest_and_congestion(df), on="match_id", how="left")
        if venues is not None and not venues.empty:
            df = df.merge(travel_distance(df, venues), on="match_id", how="left")
        else:
            df["away_travel_km"] = np.nan
        df["away_fatigue_index"] = fatigue_index(df)
        df = df.merge(referee_features(df), on="match_id", how="left")
        if weather is not None and not weather.empty:
            df = df.merge(weather.drop_duplicates("match_id"), on="match_id", how="left")
        for col in ("wx_temperature_2m", "wx_precipitation", "wx_wind_speed_10m", "wx_relative_humidity_2m"):
            if col not in df:
                df[col] = np.nan
        df = self._add_market_features(df)
        df["league_id"] = df["league_code"].astype("category").cat.codes if "league_code" in df else 0
        return df

    # ------------------------------------------------------------------- elo
    def _add_elo(self, df: pd.DataFrame) -> pd.DataFrame:
        ratings: dict[str, float] = {}
        cfg = self.elo
        home_pre, away_pre, exp_home = [], [], []
        for h, a, hg, ag in zip(df["home_team"], df["away_team"], df["home_goals"], df["away_goals"], strict=True):
            rh = ratings.get(h, cfg.initial)
            ra = ratings.get(a, cfg.initial)
            home_pre.append(rh)
            away_pre.append(ra)
            e_home = 1.0 / (1.0 + 10 ** ((ra - rh - cfg.home_advantage) / 400.0))
            exp_home.append(e_home)
            s_home = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            mult = 1.0
            if cfg.goal_diff_multiplier:
                gd = abs(hg - ag)
                mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
            delta = cfg.k * mult * (s_home - e_home)
            ratings[h] = rh + delta
            ratings[a] = ra - delta
        df["elo_home"] = home_pre
        df["elo_away"] = away_pre
        df["elo_diff"] = df["elo_home"] - df["elo_away"]
        df["elo_exp_home"] = exp_home
        # prefer externally supplied Elo (Club Elo / CFMD) when present, keep ours as fallback
        if "home_elo" in df and df["home_elo"].notna().any():
            df["elo_home"] = df["home_elo"].fillna(df["elo_home"])
            df["elo_away"] = df["away_elo"].fillna(df["elo_away"])
            df["elo_diff"] = df["elo_home"] - df["elo_away"]
        return df

    # ------------------------------------------------------------- rolling
    def _add_rolling_form(self, df: pd.DataFrame) -> pd.DataFrame:
        long = _team_long(df)
        long = long.sort_values(["team", "date", "match_id"])
        g = long.groupby("team", sort=False)
        for w in self.windows:
            for stat in ("gf", "ga", "sf", "sa", "stf", "sta", "cf", "ca", "pts"):
                long[f"{stat}_r{w}"] = g[stat].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
        long["n_prior"] = g.cumcount()
        feat_cols = [c for c in long.columns if "_r" in c] + ["n_prior"]
        home = long[long["is_home"] == 1].set_index("match_id")[feat_cols].add_prefix("h_")
        away = long[long["is_home"] == 0].set_index("match_id")[feat_cols].add_prefix("a_")
        df = df.merge(home, left_on="match_id", right_index=True, how="left")
        df = df.merge(away, left_on="match_id", right_index=True, how="left")
        for w in self.windows:
            df[f"form_diff_r{w}"] = df[f"h_pts_r{w}"] - df[f"a_pts_r{w}"]
            df[f"attack_diff_r{w}"] = df[f"h_gf_r{w}"] - df[f"a_gf_r{w}"]
            df[f"defence_diff_r{w}"] = df[f"a_ga_r{w}"] - df[f"h_ga_r{w}"]
        return df

    # -------------------------------------------------------------- market
    def _add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # early (never closing) prices: Pinnacle if present, else the market-average / Bet365 price
        cols: list[str] = []
        for p in (self.early_odds_prefix, "Mkt", "Avg", "B365"):
            cand = [f"{p}H", f"{p}D", f"{p}A"]
            if all(c in df.columns for c in cand) and df[cand].notna().any().any():
                cols = cand
                self.market_price_source = p
                break
        if cols:
            inv = 1.0 / df[cols]
            total = inv.sum(axis=1)
            df["mkt_overround"] = total - 1.0
            df["mkt_home_p"] = inv[cols[0]] / total
            df["mkt_draw_p"] = inv[cols[1]] / total
            df["mkt_away_p"] = inv[cols[2]] / total
        else:
            for c in MARKET_FEATURES:
                df[c] = np.nan
        return df


def _team_long(df: pd.DataFrame) -> pd.DataFrame:
    def side(prefix: str, other: str, is_home: int) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "match_id": df["match_id"],
                "date": df["date"],
                "team": df[f"{prefix}_team"],
                "is_home": is_home,
                "gf": df[f"{prefix}_goals"],
                "ga": df[f"{other}_goals"],
                "sf": df.get(f"{prefix}_shots", np.nan),
                "sa": df.get(f"{other}_shots", np.nan),
                "stf": df.get(f"{prefix}_shots_on_target", np.nan),
                "sta": df.get(f"{other}_shots_on_target", np.nan),
                "cf": df.get(f"{prefix}_corners", np.nan),
                "ca": df.get(f"{other}_corners", np.nan),
            }
        )
        out["pts"] = np.where(out["gf"] > out["ga"], 3, np.where(out["gf"] == out["ga"], 1, 0))
        return out

    return pd.concat([side("home", "away", 1), side("away", "home", 0)], ignore_index=True)


def feature_columns(df: pd.DataFrame, include_market: bool = False) -> list[str]:
    cols = [c for c in BASE_FEATURES if c in df.columns]
    cols += [c for c in df.columns if any(c.endswith(f"_r{w}") for w in ROLL_WINDOWS)]
    cols += [c for c in ("h_n_prior", "a_n_prior", "league_id") if c in df.columns]
    if include_market:
        cols += [c for c in MARKET_FEATURES if c in df.columns]
    return list(dict.fromkeys(cols))
