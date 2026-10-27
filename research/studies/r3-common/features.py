"""Build the R3 `closing-line-prediction` panel: one row per match with the Pinnacle open/close prices, the
R2-Q2 consensus-gap inputs and every candidate feature group, each with the date it became known.

Decision time. football-data's "open" columns are a snapshot taken on Friday afternoon for games played
Friday-Monday and on Tuesday afternoon for games played Tuesday-Thursday (football-data notes.txt).
`snapshot_date` is that Friday/Tuesday. A feature may only use information dated strictly before
`snapshot_date` (i.e. known by the end of the previous day), which is conservative by at least half a
day relative to the real afternoon snapshot. Every feature group carries a `*_known_at` date and
`assert_point_in_time` checks `known_at < snapshot_date` for every row before the panel is written.

Feature groups (column prefixes):
- `pi_`    pi-ratings (Constantinou & Fenton 2013) from all fd_matches results (every division) dated
           before the snapshot: `pi_egd` = expected goal difference home minus away.
- `elo_`   Club Elo (lake clubelo_ratings, rows exist from 2022-09-15): the latest rating row dated before
           the snapshot (a row's date is the match date and its rating is post-match), `elo_diff` = home
           minus away. Rows older than 150 days are treated as missing.
- `xg_`    Understat finishing luck: over each team's last 20 Understat matches before the snapshot,
           sum((GF - xGF) - (GA - xGA)) / (n + 10); `xg_luck_diff` = home minus away.
- `mgr_`   days since the current manager's appointment (latest tm_manager_tenures.appointed before the
           snapshot), for clubs mapped to Transfermarkt ids.
- `sch_`   rest days and 14-day congestion from every prior fixture known before the snapshot: fd_matches
           league fixtures (any division) plus Transfermarkt all-competition games for mapped clubs.

Team identity. fd teams are keyed (country, name). Understat and Club Elo ids are mapped through the
lake's club_source_map (football_data <-> understat / clubelo via the shared club_id). Transfermarkt
club ids are mapped by fixture co-occurrence: an fd top-flight match is paired with the tm_games
domestic-league game on the same date with the same score, and each fd team takes the Transfermarkt
club it is paired with at least 3x as often as any other club (weighted votes >= 15; same-day pairs
weigh 3, +-1 day pairs 1). Join rates are written to join_rates.csv.

Writes data/research/r3/panel.parquet and data/research/r3/join_rates.csv.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from lake import REPO

from pitch_edge.research import devig as dv

D = REPO / "data" / "research" / "r3"
OUTCOMES = ("home", "draw", "away")
MAIN_TOP = {
    "E0": "GB1", "SP1": "ES1", "I1": "IT1", "F1": "FR1", "T1": "TR1", "D1": "L1", "N1": "NL1",
    "P1": "PO1", "B1": "BE1", "G1": "GR1", "SC0": "SC1",
}  # fmt: skip
UNDERSTAT_LEAGUES = {"EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"}
PI_LAMBDA, PI_GAMMA, PI_B, PI_C = 0.035, 0.7, 10.0, 3.0
XG_WINDOW, XG_SHRINK = 20, 10.0
ELO_MAX_AGE_DAYS = 150
REST_CAP = 21


# ------------------------------------------------------------------------------------------- helpers


def snapshot_date(match_date: pd.Series) -> pd.Series:
    """Friday on/before the match for Fri-Mon games, Tuesday on/before it for Tue-Thu games."""
    wd = match_date.dt.weekday  # Mon=0
    back = np.select([wd == 0, wd == 1, wd == 2, wd == 3, wd == 4, wd == 5, wd == 6], [3, 0, 1, 2, 0, 1, 2])
    return match_date - pd.to_timedelta(back, unit="D")


def tm_date(ts: pd.Series) -> pd.Series:
    """Calendar date of a Transfermarkt timestamp stored as local (CET/CEST) midnight in UTC."""
    t = pd.to_datetime(ts, utc=True)
    return (t + pd.Timedelta(hours=3)).dt.tz_localize(None).dt.normalize()


def assert_point_in_time(panel: pd.DataFrame, known_col: str) -> int:
    """Every non-missing `known_col` must be strictly before `snapshot_date`; returns rows checked."""
    k = panel[known_col]
    ok = k.isna() | (k < panel["snapshot_date"])
    if not ok.all():
        bad = panel.loc[~ok, ["match_date", "snapshot_date", known_col]].head()
        raise AssertionError(f"{known_col} not before snapshot_date:\n{bad}")
    return int(k.notna().sum())


def fd_team_keys() -> tuple[dict[str, int], dict[int, str]]:
    """football_data source_id 'Country:Name' -> club_id, and back. A club with several football-data
    spellings keeps one of them in the back map; pass panel keys through `canonical_key` before merging."""
    csm = pd.read_parquet(D / "club_source_map.parquet")
    f = csm[csm.source == "football_data"]
    fwd = dict(zip(f.source_id, f.club_id, strict=True))
    return fwd, {v: k for k, v in fwd.items()}


def canonical_key(keys: pd.Series, fwd: dict[str, int], back: dict[int, str]) -> pd.Series:
    """Map each fd team key to its club's one spelling in `back`, so every spelling of a club merges with
    the same ClubElo / Understat rows; keys without a club_id stay as they are."""
    return keys.map(fwd).map(back).fillna(keys)


# --------------------------------------------------------------------------------------- universe


def load_universe() -> pd.DataFrame:
    """Main-league fd rows from 2012-07 with Pinnacle open, avg/max open and either a Pinnacle close
    (through 2025-06) or a Betfair Exchange close (post segment from 2025-07)."""
    d = pd.read_parquet(D / "fd_odds.parquet")
    d["match_date"] = pd.to_datetime(d["match_date"])
    need_open = [f"{b}_{o}_open" for b in ("pinnacle", "avg", "max") for o in OUTCOMES]
    ok = d[need_open].notna().all(axis=1) & (d[need_open] > 1.0).all(axis=1)
    pin_close = d[[f"pinnacle_{o}_close" for o in OUTCOMES]].gt(1.0).all(axis=1)
    bf_close = d[[f"betfair_ex_{o}_close" for o in OUTCOMES]].gt(1.0).all(axis=1)
    pre = d["match_date"] <= "2025-06-30"
    d = d[ok & ((pre & pin_close) | (~pre & bf_close))].copy()
    d["segment"] = np.where(pre, "pinnacle", "post_betfair")[ok & ((pre & pin_close) | (~pre & bf_close))]
    d["snapshot_date"] = snapshot_date(d["match_date"])
    d["home_key"] = d["country"] + ":" + d["home_team"]
    d["away_key"] = d["country"] + ":" + d["away_team"]
    d["match_id"] = (
        d["division"] + "|" + d["match_date"].dt.strftime("%Y-%m-%d") + "|" + d["home_team"] + "|" + d["away_team"]
    )
    assert not d["match_id"].duplicated().any()
    for m in ("power", "shin"):
        p = dv.devig_frame(d, [f"pinnacle_{o}_open" for o in OUTCOMES], method=m, prefix="tmp_")
        d[[f"p_open_{m}_{o}" for o in OUTCOMES]] = p[[f"tmp_pinnacle_{o}_open" for o in OUTCOMES]].to_numpy()
        pa = dv.devig_frame(d, [f"avg_{o}_open" for o in OUTCOMES], method=m, prefix="tmp_")
        d[[f"p_avg_{m}_{o}" for o in OUTCOMES]] = pa[[f"tmp_avg_{o}_open" for o in OUTCOMES]].to_numpy()
        close_cols = np.where(
            d["segment"].to_numpy()[:, None] == "pinnacle",
            d[[f"pinnacle_{o}_close" for o in OUTCOMES]].to_numpy(),
            d[[f"betfair_ex_{o}_close" for o in OUTCOMES]].to_numpy(),
        )
        pc = dv.devig(close_cols, method=m).probs
        d[[f"p_close_{m}_{o}" for o in OUTCOMES]] = pc
    return d.reset_index(drop=True)


# ------------------------------------------------------------------------------------- pi-ratings


def _psi(r: float) -> float:
    return math.copysign(PI_B ** (abs(r) / PI_C) - 1.0, r)


def pi_ratings(panel: pd.DataFrame) -> pd.DataFrame:
    """Pi-rating expected goal difference for each panel row, from results dated before its snapshot."""
    fd = pd.read_parquet(D / "fd_all.parquet")
    fd = fd.dropna(subset=["ft_home_goals", "ft_away_goals"])
    fd["match_date"] = pd.to_datetime(fd["match_date"])
    fd["hk"] = fd["country"] + ":" + fd["home_team"]
    fd["ak"] = fd["country"] + ":" + fd["away_team"]
    fd = fd.sort_values("match_date", kind="stable")
    res_dates = fd["match_date"].to_numpy()
    res = list(zip(fd["hk"], fd["ak"], fd["ft_home_goals"].astype(int), fd["ft_away_goals"].astype(int), strict=True))
    tgt = panel[["snapshot_date", "home_key", "away_key"]].copy()
    tgt["i"] = np.arange(len(tgt))
    tgt = tgt.sort_values("snapshot_date", kind="stable")
    rh: dict[str, float] = defaultdict(float)  # home ratings
    ra: dict[str, float] = defaultdict(float)  # away ratings
    out = np.full(len(panel), np.nan)
    known = np.full(len(panel), np.datetime64("NaT"), dtype="datetime64[ns]")
    last_applied = np.datetime64("NaT")
    j = 0
    for snap, hk, ak, i in zip(
        tgt["snapshot_date"].to_numpy(), tgt["home_key"], tgt["away_key"], tgt["i"], strict=True
    ):
        while j < len(res) and res_dates[j] < snap:
            h, a, hg, ag = res[j]
            exp_gd = _psi(rh[h]) - _psi(ra[a])
            gd = hg - ag
            err = abs(gd - exp_gd)
            w = PI_C * math.log10(1.0 + err)
            sh = w if gd > exp_gd else -w
            d_h = sh * PI_LAMBDA
            rh[h] += d_h
            ra[h] += d_h * PI_GAMMA
            d_a = -sh * PI_LAMBDA
            ra[a] += d_a
            rh[a] += d_a * PI_GAMMA
            last_applied = res_dates[j]
            j += 1
        if hk in rh and ak in ra:
            out[i] = _psi(rh[hk]) - _psi(ra[ak])
            known[i] = last_applied
    return pd.DataFrame({"pi_egd": out, "pi_known_at": pd.to_datetime(known)})


# --------------------------------------------------------------------------------------- Club Elo


def _clubelo_overlap_map(ce: pd.DataFrame) -> dict[str, str]:
    """Club Elo slug -> fd team key for slugs club_source_map does not map: the slug's rating-row dates
    (one per match played) must cover >= 90% of the fd team's match dates since 2022-09-15, and among
    such candidates the slug with the most similar name (difflib ratio >= 0.5) is taken."""
    import difflib

    fd = pd.read_parquet(D / "fd_all.parquet")
    fd["match_date"] = pd.to_datetime(fd["match_date"])
    fd = fd[fd["match_date"] >= "2022-09-16"]
    fx = pd.concat(
        [
            pd.DataFrame({"key": fd["country"] + ":" + fd["home_team"], "day": fd["match_date"]}),
            pd.DataFrame({"key": fd["country"] + ":" + fd["away_team"], "day": fd["match_date"]}),
        ]
    )
    taken = set(ce["fd_key"].dropna())
    free_slugs = ce.loc[ce["fd_key"].isna()]
    slug_days = {k: set(g["date"]) for k, g in free_slugs.groupby("club_slug")}
    out: dict[str, str] = {}
    for key, g in fx.groupby("key"):
        if key in taken or len(g) < 10:
            continue
        days = set(g["day"])
        cands = [sl for sl, sd in slug_days.items() if len(days & sd) >= 0.9 * len(days)]
        if not cands:
            continue
        name = key.split(":", 1)[1].lower().replace(" ", "")
        scored = sorted(((difflib.SequenceMatcher(None, name, sl.lower()).ratio(), sl) for sl in cands), reverse=True)
        if scored[0][0] >= 0.5 and scored[0][1] not in out:
            out[scored[0][1]] = key
    return out


def club_elo(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    csm = pd.read_parquet(D / "club_source_map.parquet")
    fwd, back = fd_team_keys()
    slug_to_club = dict(zip(csm[csm.source == "clubelo"].source_id, csm[csm.source == "clubelo"].club_id, strict=True))
    ce = pd.read_parquet(D / "clubelo.parquet")
    ce["date"] = pd.to_datetime(ce["date"], utc=True).dt.tz_localize(None).dt.normalize()
    ce["fd_key"] = ce["club_slug"].map(slug_to_club).map(back)
    extra = _clubelo_overlap_map(ce)
    n_csm = ce.loc[ce["fd_key"].notna(), "club_slug"].nunique()
    ce["fd_key"] = ce["fd_key"].fillna(ce["club_slug"].map(extra))
    ce = ce.dropna(subset=["fd_key"])
    ce["fd_key"] = canonical_key(ce["fd_key"], fwd, back)
    ce = ce.drop_duplicates(["fd_key", "date"]).sort_values("date")
    out = {}
    for side in ("home", "away"):
        left = panel[["snapshot_date", f"{side}_key"]].copy()
        left[f"{side}_key"] = canonical_key(left[f"{side}_key"], fwd, back)
        left["i"] = np.arange(len(left))
        left["by"] = left["snapshot_date"] - pd.Timedelta(days=1)  # rows dated <= snapshot - 1 day
        left = left.sort_values("by")
        m = pd.merge_asof(
            left,
            ce[["date", "fd_key", "elo"]].rename(columns={"fd_key": f"{side}_key"}),
            left_on="by",
            right_on="date",
            by=f"{side}_key",
            direction="backward",
        ).sort_values("i")
        stale = (m["by"] - m["date"]).dt.days > ELO_MAX_AGE_DAYS
        m.loc[stale, ["elo", "date"]] = np.nan
        out[f"elo_{side}"] = m["elo"].to_numpy()
        out[f"elo_{side}_date"] = m["date"].to_numpy()
    f = pd.DataFrame(out)
    f["elo_diff"] = f["elo_home"] - f["elo_away"]
    f["elo_known_at"] = f[["elo_home_date", "elo_away_date"]].max(axis=1)
    f.loc[f["elo_diff"].isna(), "elo_known_at"] = pd.NaT
    mapped_keys = set(ce["fd_key"])
    stats = {"clubelo_fd_teams_mapped": len(mapped_keys), "via_club_source_map": n_csm, "via_overlap": len(extra)}
    return f[["elo_home", "elo_away", "elo_diff", "elo_known_at"]], stats


# ----------------------------------------------------------------------------------- Understat xG


def xg_luck(panel: pd.DataFrame) -> pd.DataFrame:
    csm = pd.read_parquet(D / "club_source_map.parquet")
    fwd, back = fd_team_keys()
    us_map = csm[csm.source == "understat"]
    us_to_fd = {int(s): back.get(c) for s, c in zip(us_map.source_id, us_map.club_id, strict=True)}
    us = pd.read_parquet(D / "us_team_matches.parquet")
    us = us[us["league"].isin(UNDERSTAT_LEAGUES)].dropna(subset=["home_xg", "away_xg", "home_goals", "away_goals"])
    us["day"] = pd.to_datetime(us["date"], utc=True).dt.tz_localize(None).dt.normalize()
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "tid": us.home_team_id,
                    "day": us.day,
                    "luck": (us.home_goals - us.home_xg) - (us.away_goals - us.away_xg),
                }
            ),
            pd.DataFrame(
                {
                    "tid": us.away_team_id,
                    "day": us.day,
                    "luck": (us.away_goals - us.away_xg) - (us.home_goals - us.home_xg),
                }
            ),
        ]
    )
    long["fd_key"] = long["tid"].astype(int).map(us_to_fd)
    long = long.dropna(subset=["fd_key"]).sort_values(["fd_key", "day"])
    g = long.groupby("fd_key")
    long["n"] = g.cumcount() + 1
    long["roll_sum"] = g["luck"].transform(lambda s: s.rolling(XG_WINDOW, min_periods=1).sum())
    long["roll_n"] = long["n"].clip(upper=XG_WINDOW)
    long["luck_shrunk"] = long["roll_sum"] / (long["roll_n"] + XG_SHRINK)
    long = long.sort_values("day", kind="stable")
    out = {}
    for side in ("home", "away"):
        left = panel[["snapshot_date", f"{side}_key"]].copy()
        left[f"{side}_key"] = canonical_key(left[f"{side}_key"], fwd, back)
        left["i"] = np.arange(len(left))
        left["by"] = left["snapshot_date"] - pd.Timedelta(days=1)
        left = left.sort_values("by")
        m = pd.merge_asof(
            left,
            long[["day", "fd_key", "luck_shrunk"]].rename(columns={"fd_key": f"{side}_key"}),
            left_on="by",
            right_on="day",
            by=f"{side}_key",
            direction="backward",
        ).sort_values("i")
        stale = (m["by"] - m["day"]).dt.days > 150
        m.loc[stale, ["luck_shrunk", "day"]] = np.nan
        out[f"xg_luck_{side}"] = m["luck_shrunk"].to_numpy()
        out[f"xg_{side}_day"] = m["day"].to_numpy()
    f = pd.DataFrame(out)
    f["xg_luck_diff"] = f["xg_luck_home"] - f["xg_luck_away"]
    f["xg_known_at"] = f[["xg_home_day", "xg_away_day"]].max(axis=1)
    f.loc[f["xg_luck_diff"].isna(), "xg_known_at"] = pd.NaT
    return f[["xg_luck_home", "xg_luck_away", "xg_luck_diff", "xg_known_at"]]


# ------------------------------------------------------------------------- Transfermarkt mapping


def tm_mapping() -> tuple[dict[str, int], pd.DataFrame]:
    """fd team key -> Transfermarkt club id by fixture co-occurrence (same date +-1, same score)."""
    fd = pd.read_parquet(D / "fd_all.parquet")
    fd = fd[fd["division"].isin(MAIN_TOP) & fd["source_file"].str.startswith("mmz4281/")]
    fd = fd.dropna(subset=["ft_home_goals"])
    fd["match_date"] = pd.to_datetime(fd["match_date"])
    fd = fd[fd["match_date"] >= "2012-07-01"]
    fd["comp"] = fd["division"].map(MAIN_TOP)
    tg = pd.read_parquet(D / "tm_games.parquet")
    tg["day"] = pd.to_datetime(tg["date"]).dt.normalize()
    tg = tg[tg["competition_id"].isin(set(MAIN_TOP.values()))]
    votes: dict[tuple[str, int], int] = defaultdict(int)
    for shift in (0, -1, 1):
        t = tg.assign(day=tg["day"] + pd.Timedelta(days=shift))
        m = fd.merge(
            t,
            left_on=["comp", "match_date", "ft_home_goals", "ft_away_goals"],
            right_on=["competition_id", "day", "home_club_goals", "away_club_goals"],
        )
        w = 3 if shift == 0 else 1
        for side_fd, side_tm in (("home_team", "home_club_id"), ("away_team", "away_club_id")):
            for k, c in zip(m["country"] + ":" + m[side_fd], m[side_tm], strict=True):
                votes[(k, int(c))] += w
    v = pd.DataFrame([(k, c, n) for (k, c), n in votes.items()], columns=["fd_key", "tm_id", "votes"])
    tot = v.groupby("fd_key")["votes"].transform("sum")
    v["share"] = v["votes"] / tot
    ranked = v.sort_values(["fd_key", "votes"], ascending=[True, False])
    ranked["second"] = ranked.groupby("fd_key")["votes"].shift(-1)
    best = ranked.drop_duplicates("fd_key").copy()
    best["second"] = best["second"].fillna(0)
    best = best[(best["votes"] >= 3 * best["second"]) & (best["votes"] >= 15)]
    best = best[~best["tm_id"].duplicated(keep=False)]  # one tm club per fd team
    return dict(zip(best["fd_key"], best["tm_id"].astype(int), strict=True)), best


def manager(panel: pd.DataFrame, fd_to_tm: dict[str, int]) -> pd.DataFrame:
    ten = pd.read_parquet(D / "tm_manager_tenures.parquet")
    ten["appointed_day"] = tm_date(ten["appointed"])
    ten = ten.dropna(subset=["appointed_day"]).sort_values("appointed_day")
    out = {}
    for side in ("home", "away"):
        left = panel[["snapshot_date", "match_date", f"{side}_key"]].copy()
        left["i"] = np.arange(len(left))
        left["tm_id"] = left[f"{side}_key"].map(fd_to_tm)
        left["by"] = left["snapshot_date"] - pd.Timedelta(days=1)
        has = left["tm_id"].notna()
        l2 = left[has].copy()
        l2["tm_id"] = l2["tm_id"].astype(int)
        l2 = l2.sort_values("by")
        m = pd.merge_asof(
            l2,
            ten[["club_id", "appointed_day"]].rename(columns={"club_id": "tm_id"}).astype({"tm_id": int}),
            left_on="by",
            right_on="appointed_day",
            by="tm_id",
            direction="backward",
        ).set_index("i")
        days = pd.Series(np.nan, index=left["i"])
        appt = pd.Series(pd.NaT, index=left["i"], dtype="datetime64[ns]")
        days.loc[m.index] = (m["match_date"] - m["appointed_day"]).dt.days
        appt.loc[m.index] = m["appointed_day"]
        out[f"mgr_days_{side}"] = days.to_numpy()
        out[f"mgr_{side}_appt"] = appt.to_numpy()
    f = pd.DataFrame(out)
    for side in ("home", "away"):
        f[f"mgr_new_{side}"] = np.exp(-f[f"mgr_days_{side}"] / 30.0)
    f["mgr_known_at"] = f[["mgr_home_appt", "mgr_away_appt"]].max(axis=1)
    f.loc[f[["mgr_days_home", "mgr_days_away"]].isna().any(axis=1), "mgr_known_at"] = pd.NaT
    return f[["mgr_days_home", "mgr_days_away", "mgr_new_home", "mgr_new_away", "mgr_known_at"]]


# ----------------------------------------------------------------------------------------- schedule


def schedule(panel: pd.DataFrame, fd_to_tm: dict[str, int]) -> pd.DataFrame:
    fd = pd.read_parquet(D / "fd_all.parquet")
    fd["match_date"] = pd.to_datetime(fd["match_date"])
    fx = pd.concat(
        [
            pd.DataFrame({"key": fd["country"] + ":" + fd["home_team"], "day": fd["match_date"]}),
            pd.DataFrame({"key": fd["country"] + ":" + fd["away_team"], "day": fd["match_date"]}),
        ]
    )
    tg = pd.read_parquet(D / "tm_games.parquet")
    tg["day"] = pd.to_datetime(tg["date"]).dt.normalize()
    tm_to_fd = {v: k for k, v in fd_to_tm.items()}
    tmx = pd.concat(
        [
            pd.DataFrame({"key": tg["home_club_id"].map(tm_to_fd), "day": tg["day"]}),
            pd.DataFrame({"key": tg["away_club_id"].map(tm_to_fd), "day": tg["day"]}),
        ]
    ).dropna()
    fx = pd.concat([fx, tmx]).drop_duplicates().sort_values(["key", "day"])
    days_by_key = {k: g["day"].to_numpy() for k, g in fx.groupby("key")}
    out = {}
    for side in ("home", "away"):
        rest = np.full(len(panel), np.nan)
        cong = np.full(len(panel), np.nan)
        last = np.full(len(panel), np.datetime64("NaT"), dtype="datetime64[ns]")
        for i, (k, snap, md) in enumerate(
            zip(panel[f"{side}_key"], panel["snapshot_date"].to_numpy(), panel["match_date"].to_numpy(), strict=True)
        ):
            arr = days_by_key.get(k)
            if arr is None:
                continue
            pos = np.searchsorted(arr, snap, side="left")  # fixtures dated < snapshot_date
            if pos == 0:
                continue
            prev = arr[pos - 1]
            last[i] = prev
            rest[i] = min((md - prev) / np.timedelta64(1, "D"), REST_CAP)
            lo = np.searchsorted(arr, md - np.timedelta64(14, "D"), side="left")
            cong[i] = pos - lo
        out[f"sch_rest_{side}"] = rest
        out[f"sch_cong14_{side}"] = cong
        out[f"sch_{side}_last"] = last
    f = pd.DataFrame(out)
    f["sch_known_at"] = f[["sch_home_last", "sch_away_last"]].max(axis=1)
    f.loc[f[["sch_rest_home", "sch_rest_away"]].isna().any(axis=1), "sch_known_at"] = pd.NaT
    f["sch_tm_both"] = panel["home_key"].isin(fd_to_tm).to_numpy() & panel["away_key"].isin(fd_to_tm).to_numpy()
    return f[["sch_rest_home", "sch_rest_away", "sch_cong14_home", "sch_cong14_away", "sch_tm_both", "sch_known_at"]]


# ------------------------------------------------------------------------------------------------ main


def main() -> None:
    panel = load_universe()
    print("universe", panel.shape, panel["segment"].value_counts().to_dict())
    pi = pi_ratings(panel)
    elo, elo_stats = club_elo(panel)
    xg = xg_luck(panel)
    fd_to_tm, tm_best = tm_mapping()
    tm_best.to_csv(D / "tm_mapping.csv", index=False)
    mgr = manager(panel, fd_to_tm)
    sch = schedule(panel, fd_to_tm)
    panel = pd.concat([panel, pi, elo, xg, mgr, sch], axis=1)
    panel["season_start"] = np.where(
        panel["match_date"].dt.month >= 7, panel["match_date"].dt.year, panel["match_date"].dt.year - 1
    )
    checked = {
        k: assert_point_in_time(panel, k)
        for k in ("pi_known_at", "elo_known_at", "xg_known_at", "mgr_known_at", "sch_known_at")
    }
    print("point-in-time assertions passed:", checked)
    # Join rates per division and segment (pre-2025-07 Pinnacle segment).
    pre = panel[panel["segment"] == "pinnacle"]
    jr = pre.groupby("division").agg(
        n=("match_id", "size"),
        pi=("pi_egd", lambda s: s.notna().mean()),
        elo_2022p=("elo_diff", lambda s: s.notna().sum()),
        xg=("xg_luck_diff", lambda s: s.notna().mean()),
        mgr=("mgr_days_home", lambda s: s.notna().mean()),
        mgr_both=("mgr_known_at", lambda s: s.notna().mean()),
        sch_tm_both=("sch_tm_both", "mean"),
    )
    elo_window = pre[pre["snapshot_date"] >= "2022-09-16"]
    jr["elo_rate_2022p"] = elo_window.groupby("division")["elo_diff"].apply(lambda s: s.notna().mean())
    jr.to_csv(D / "join_rates.csv")
    print(jr.round(3).to_string())
    print("elo", elo_stats, "tm mapped fd teams", len(fd_to_tm))
    panel.drop(
        columns=[
            c
            for c in panel.columns
            if c.startswith("oa_") and c[3:].rstrip("HDA") not in ("B365", "BW", "WH", "IW", "VC")
        ]
    ).to_parquet(D / "panel.parquet")
    print("wrote panel", panel.shape)


if __name__ == "__main__":
    main()
