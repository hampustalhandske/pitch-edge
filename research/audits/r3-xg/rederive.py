"""Independent re-derivation of `closing-line-prediction-xg-finishing-v1` (skeptic audit).

Uses only the raw lake pulls in data/research/r3/ (fd_odds.parquet, fd_all.parquet, us_team_matches.parquet,
club_source_map.parquet) and none of the study's code (r3lib.py / features.py / run.py). Everything below
-- de-vig, snapshot date, Understat <-> football-data team mapping, the finishing-luck feature, the
walk-forward OLS and the KL scoring -- is re-implemented here.

What it computes:
- the headline: mean over the evaluation window (2019-07-01..2025-06-30) of
  KL(p_close || p_hat_base) - KL(p_close || p_hat_xg) in bits, power and Shin de-vig,
  on (a) the study's universe (any division with both features) and (b) the registered top-5 universe;
- look-ahead diagnostics: each eval match's own Understat record vs its snapshot date, Understat fixtures
  falling between snapshot and kickoff, and the mapping checked against an independent fixture-co-occurrence map;
- placebos: feature lagged by 1 extra match, lagged by 38 matches, teams permuted within league-season
  (100 permutations), feature pairs shuffled within match-day (100 permutations);
- robustness: a richer baseline (adds logit p_open level and its square, per outcome), a goals-only form
  feature and an xG-only feature in place of goals-minus-xG;
- dependence: match-day and ISO-week block bootstraps, and per league-season cluster bootstrap.

Usage: uv run python research/audits/r3-xg/rederive.py   (writes research/audits/r3-xg/out/*.csv|json)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
D = REPO / "data" / "research" / "r3"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
O3 = ("home", "draw", "away")
TOP5 = {"E0": "EPL", "SP1": "La_liga", "I1": "Serie_A", "D1": "Bundesliga", "F1": "Ligue_1"}
EV0, EV1 = pd.Timestamp("2019-07-01"), pd.Timestamp("2025-06-30")
RNG = np.random.default_rng(7)


# ------------------------------------------------------------------ de-vig (own implementations)


def devig_power(odds: np.ndarray) -> np.ndarray:
    q = 1.0 / odds
    lo, hi = np.full(len(q), 0.5), np.full(len(q), 3.0)
    for _ in range(100):
        k = (lo + hi) / 2
        s = (q ** k[:, None]).sum(1)
        hi = np.where(s < 1, k, hi)
        lo = np.where(s >= 1, k, lo)
    k = (lo + hi) / 2
    p = q ** k[:, None]
    return p / p.sum(1, keepdims=True)


def devig_shin(odds: np.ndarray) -> np.ndarray:
    q = 1.0 / odds
    s = q.sum(1, keepdims=True)
    lo, hi = np.zeros(len(q)), np.full(len(q), 0.5)

    def probs(z: np.ndarray) -> np.ndarray:
        z = z[:, None]
        return (np.sqrt(z**2 + 4 * (1 - z) * q**2 / s) - z) / (2 * (1 - z))

    for _ in range(100):
        z = (lo + hi) / 2
        t = probs(z).sum(1)
        lo = np.where(t > 1, z, lo)  # sum decreases in z
        hi = np.where(t <= 1, z, hi)
    p = probs((lo + hi) / 2)
    return p / p.sum(1, keepdims=True)


DEVIG = {"power": devig_power, "shin": devig_shin}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p) - np.log1p(-p)


def kl_bits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (p * np.log2(np.clip(p, 1e-12, 1) / np.clip(q, 1e-12, 1))).sum(1)


# ------------------------------------------------------------------ universe


def snapshot(md: pd.Series) -> pd.Series:
    """Friday on/before for Fri..Mon kickoffs, Tuesday on/before for Tue..Thu kickoffs."""
    wd = md.dt.weekday
    back = wd.map({4: 0, 5: 1, 6: 2, 0: 3, 1: 0, 2: 1, 3: 2})
    return md - pd.to_timedelta(back, unit="D")


def universe() -> pd.DataFrame:
    d = pd.read_parquet(D / "fd_odds.parquet")
    d["match_date"] = pd.to_datetime(d["match_date"])
    d = d[(d.match_date >= "2012-07-01") & (d.match_date <= EV1)]
    cols = [f"{b}_{o}_{s}" for b in ("pinnacle",) for o in O3 for s in ("open", "close")]
    cols += [f"{b}_{o}_open" for b in ("avg", "max") for o in O3]
    d = d[(d[cols] > 1.0).all(axis=1)].copy()
    d["snap"] = snapshot(d.match_date)
    d["season"] = np.where(d.match_date.dt.month >= 7, d.match_date.dt.year, d.match_date.dt.year - 1)
    d["hk"] = d.country + ":" + d.home_team
    d["ak"] = d.country + ":" + d.away_team
    return d.reset_index(drop=True)


# ------------------------------------------------------------------ mapping


def map_csm() -> dict[int, str]:
    csm = pd.read_parquet(D / "club_source_map.parquet")
    fd = csm[csm.source == "football_data"].set_index("club_id")["source_id"]
    us = csm[csm.source == "understat"]
    return {int(s): fd.get(c) for s, c in zip(us.source_id, us.club_id, strict=False) if fd.get(c) is not None}


def map_cooccur(us: pd.DataFrame) -> dict[int, str]:
    """Understat team id -> fd key by fixtures: same league, date +-1, same score; majority >= 80% of votes."""
    fd = pd.read_parquet(D / "fd_all.parquet")
    fd = fd[fd.division.isin(TOP5)].dropna(subset=["ft_home_goals"])
    fd["day"] = pd.to_datetime(fd.match_date)
    fd["lg"] = fd.division.map(TOP5)
    votes = []
    for sh in (-1, 0, 1):
        u = us.drop(columns=["home_team", "away_team", "season"]).assign(day=us.day + pd.Timedelta(days=sh))
        m = u.merge(
            fd,
            left_on=["league", "day", "home_goals", "away_goals"],
            right_on=["lg", "day", "ft_home_goals", "ft_away_goals"],
        )
        votes.append(pd.DataFrame({"tid": m.home_team_id, "key": m.country + ":" + m.home_team}))
        votes.append(pd.DataFrame({"tid": m.away_team_id, "key": m.country + ":" + m.away_team}))
    v = pd.concat(votes).value_counts().rename("n").reset_index()
    v["share"] = v.n / v.groupby("tid").n.transform("sum")
    best = v.sort_values("n", ascending=False).drop_duplicates("tid")
    best = best[best.share >= 0.5]
    return dict(zip(best.tid.astype(int), best.key, strict=False))


# ------------------------------------------------------------------ feature


def load_us() -> pd.DataFrame:
    us = pd.read_parquet(D / "us_team_matches.parquet")
    us = us[us.league != "RFPL"].dropna(subset=["home_xg", "away_xg"])
    us["day"] = us["date"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    return us


def team_series(us: pd.DataFrame, tmap: dict[int, str]) -> dict[str, pd.DataFrame]:
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        rows.append(
            pd.DataFrame(
                {
                    "key": us[f"{side}_team_id"].map(tmap),
                    "day": us.day,
                    "gf": us[f"{side}_goals"],
                    "ga": us[f"{opp}_goals"],
                    "xgf": us[f"{side}_xg"],
                    "xga": us[f"{opp}_xg"],
                    "league": us.league,
                    "season": us.season,
                }
            )
        )
    t = pd.concat(rows).dropna(subset=["key"]).sort_values(["key", "day"])
    t["luck"] = (t.gf - t.xgf) - (t.ga - t.xga)
    t["gd"] = t.gf - t.ga
    t["xgd"] = t.xgf - t.xga
    return {k: g.reset_index(drop=True) for k, g in t.groupby("key")}


def feature(
    df: pd.DataFrame,
    series: dict[str, pd.DataFrame],
    col: str = "luck",
    lag: int = 0,
    window: int = 20,
    shrink: float = 10.0,
    max_age: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    """Per row and side: sum of `col` over the team's last `window` Understat matches with UTC day <
    snapshot date (skipping the `lag` most recent ones) / (n + shrink); NaN if none or the latest is older
    than `max_age` days before snapshot-1. Returns (values (n,2), latest day used (n,2))."""
    vals = np.full((len(df), 2), np.nan)
    last = np.full((len(df), 2), np.datetime64("NaT"), dtype="datetime64[ns]")
    snaps = df.snap.to_numpy()
    for j, kcol in enumerate(("hk", "ak")):
        for i, (k, s) in enumerate(zip(df[kcol].to_numpy(), snaps, strict=False)):
            g = series.get(k)
            if g is None:
                continue
            days = g["day"].to_numpy()
            pos = np.searchsorted(days, s, side="left") - lag
            if pos <= 0:
                continue
            if (s - np.timedelta64(1, "D") - days[pos - 1 + lag]) / np.timedelta64(1, "D") > max_age:
                continue
            x = g[col].to_numpy()[max(0, pos - window) : pos]
            vals[i, j] = x.sum() / (len(x) + shrink)
            last[i, j] = days[pos - 1]
    return vals, last


# ------------------------------------------------------------------ model


def prep(df: pd.DataFrame, method: str) -> dict[str, np.ndarray]:
    f = DEVIG[method]
    po = f(df[[f"pinnacle_{o}_open" for o in O3]].to_numpy(float))
    pc = f(df[[f"pinnacle_{o}_close" for o in O3]].to_numpy(float))
    pa = f(df[[f"avg_{o}_open" for o in O3]].to_numpy(float))
    ratio = df[[f"max_{o}_open" for o in O3]].to_numpy(float) / df[[f"pinnacle_{o}_open" for o in O3]].to_numpy(float)
    return {
        "po": po,
        "pc": pc,
        "move": logit(pc) - logit(po),
        "gavg": logit(pa) - logit(po),
        "gmax": np.minimum(np.log(ratio), np.log(1.5)),
    }


def design(b: dict, idx: np.ndarray, k: int, extra: np.ndarray | None, rich: bool) -> np.ndarray:
    cols = [np.ones(len(idx)), b["gavg"][idx, k], b["gmax"][idx, k]]
    if rich:
        lp = logit(b["po"][idx, k])
        cols += [lp, lp**2]
    x = np.column_stack(cols)
    return x if extra is None else np.column_stack([x, extra[idx]])


def walk_forward(
    df: pd.DataFrame,
    b: dict,
    feat: np.ndarray,
    rich: bool = False,
    base_rich: bool | None = None,
    seasons=range(2019, 2025),
) -> pd.DataFrame:
    """Per-season walk-forward; rows restricted to those where `feat` is finite. Returns eval frame."""
    base_rich = rich if base_rich is None else base_rich
    ok = np.isfinite(feat).all(1)
    out = []
    for s in seasons:
        tr = np.where(ok & (df.season.to_numpy() < s))[0]
        te = np.where(ok & (df.season.to_numpy() == s))[0]
        if len(te) == 0:
            continue
        mb = np.zeros((len(te), 3))
        mm = np.zeros((len(te), 3))
        for k in range(3):
            y = b["move"][tr, k]
            cb = np.linalg.lstsq(design(b, tr, k, None, base_rich), y, rcond=None)[0]
            cm = np.linalg.lstsq(design(b, tr, k, feat, rich), y, rcond=None)[0]
            mb[:, k] = design(b, te, k, None, base_rich) @ cb
            mm[:, k] = design(b, te, k, feat, rich) @ cm
        lpo = logit(b["po"][te])
        qb = 1 / (1 + np.exp(-(lpo + mb)))
        qm = 1 / (1 + np.exp(-(lpo + mm)))
        qb /= qb.sum(1, keepdims=True)
        qm /= qm.sum(1, keepdims=True)
        pc = b["pc"][te]
        sse0 = (b["move"][te] ** 2).sum(1)
        out.append(
            pd.DataFrame(
                {
                    "i": te,
                    "kl_open": kl_bits(pc, b["po"][te]),
                    "kl_base": kl_bits(pc, qb),
                    "kl_model": kl_bits(pc, qm),
                    "sse0": sse0,
                    "sse_b": ((b["move"][te] - mb) ** 2).sum(1),
                    "sse_m": ((b["move"][te] - mm) ** 2).sum(1),
                }
            )
        )
    r = pd.concat(out, ignore_index=True)
    r["d"] = r.kl_base - r.kl_model
    return r


def boot(values: np.ndarray, blocks: np.ndarray, n: int = 2000, seed: int = 1) -> tuple[float, float, float, float]:
    """Block bootstrap of the mean: (estimate, lo, hi, one-sided p of mean <= 0)."""
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    cnts = np.bincount(inv)
    draws = rng.integers(0, len(codes), size=(n, len(codes)))
    means = sums[draws].sum(1) / cnts[draws].sum(1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
        float((means <= 0).mean()),
    )


def headline(df: pd.DataFrame, r: pd.DataFrame) -> dict:
    sub = df.iloc[r.i.to_numpy()]
    d = r.d.to_numpy()
    out = {"n": len(r), "mean_d_bits": d.mean()}
    out["ci_matchday"] = boot(d, sub.match_date.dt.strftime("%Y%m%d").to_numpy())[1:]
    out["ci_isoweek"] = boot(
        d, (sub.snap.dt.isocalendar().year.astype(str) + sub.snap.dt.isocalendar().week.astype(str)).to_numpy()
    )[1:]
    out["ci_league_season"] = boot(d, (sub.division + sub.season.astype(str)).to_numpy())[1:]
    out["d_r2_move_pp"] = 100 * (r.sse_b.sum() - r.sse_m.sum()) / r.sse0.sum()
    out["kl_open_mean"] = r.kl_open.mean()
    out["kl_base_mean"] = r.kl_base.mean()
    out["frac_of_kl_open"] = d.mean() / r.kl_open.mean()
    out["frac_of_kl_base"] = d.mean() / r.kl_base.mean()
    return out


def main() -> None:
    res: dict = {}
    df = universe()
    us = load_us()
    m_csm = map_csm()
    m_co = map_cooccur(us)
    # ---- mapping audit
    us_ids = set(us.home_team_id) | set(us.away_team_id)
    diff = {t: (m_csm.get(t), m_co.get(t)) for t in us_ids if m_csm.get(t) != m_co.get(t)}
    res["mapping"] = {
        "understat_teams": len(us_ids),
        "csm_mapped": sum(t in m_csm for t in us_ids),
        "cooccur_mapped": len(m_co),
        "disagreements": {str(k): v for k, v in diff.items()},
    }
    ser = team_series(us, m_csm)
    # ---- feature (study spec) and check vs study panel
    feat, last = feature(df, ser)
    df["f_h"], df["f_a"] = feat[:, 0], feat[:, 1]
    panel = pd.read_parquet(D / "panel.parquet", columns=["match_id", "xg_luck_home", "xg_luck_away", "xg_known_at"])
    df["match_id"] = df.division + "|" + df.match_date.dt.strftime("%Y-%m-%d") + "|" + df.home_team + "|" + df.away_team
    cmp = df[["match_id", "f_h", "f_a"]].merge(panel, on="match_id", how="outer", indicator=True)
    both = cmp.dropna(subset=["f_h", "xg_luck_home", "f_a", "xg_luck_away"])
    res["feature_vs_study"] = {
        "rows_mine_with_feat": int(np.isfinite(feat).all(1).sum()),
        "rows_study_with_feat": int(panel[["xg_luck_home", "xg_luck_away"]].notna().all(1).sum()),
        "max_abs_diff": float(np.nanmax(np.abs(np.r_[both.f_h - both.xg_luck_home, both.f_a - both.xg_luck_away]))),
        "mine_only": int((cmp[["f_h", "f_a"]].notna().all(1) & cmp.xg_luck_home.isna()).sum()),
        "study_only": int((cmp[["xg_luck_home", "xg_luck_away"]].notna().all(1) & cmp.f_h.isna()).sum()),
    }
    # ---- look-ahead diagnostics on eval rows with the feature
    ev_mask = (df.match_date >= EV0) & np.isfinite(feat).all(1)
    ev = df[ev_mask]
    res["pit"] = {"latest_used_ge_snapshot": int((last[ev_mask.to_numpy()] >= ev.snap.to_numpy()[:, None]).sum())}
    # own Understat record of each eval match: same mapped teams, |day diff| <= 7
    um = us.assign(hk=us.home_team_id.map(m_csm), ak=us.away_team_id.map(m_csm)).dropna(subset=["hk", "ak"])
    j = ev[["hk", "ak", "match_date", "snap"]].reset_index().merge(um[["hk", "ak", "day", "date"]], on=["hk", "ak"])
    j = j[(j.day - j.match_date).abs() <= pd.Timedelta(days=7)]
    res["pit"]["eval_rows_matched_to_own_understat_record"] = int(j["index"].nunique())
    res["pit"]["own_record_day_lt_snapshot"] = int((j.day < j.snap).sum())
    res["pit"]["own_record_day_ne_fd_date"] = int((j.day != j.match_date).sum())
    res["pit"]["own_record_day_ne_fd_date_examples"] = j[j.day != j.match_date].head(10).astype(str).to_dict("records")
    # Understat fixtures of either team strictly between snapshot and kickoff day (not used; counted)
    cnt = 0
    for k_col in ("hk", "ak"):
        for k, s, md in zip(ev[k_col], ev.snap.to_numpy(), ev.match_date.to_numpy(), strict=False):
            days = ser[k]["day"].to_numpy()
            cnt += int(((days >= s) & (days < md)).sum())
    res["pit"]["understat_fixtures_between_snapshot_and_kickoff"] = cnt
    # ---- headline
    top5 = df.division.isin(TOP5).to_numpy()
    for method in ("power", "shin"):
        b = prep(df, method)
        r_all = walk_forward(df, b, feat)
        res[f"headline_{method}_study_universe"] = headline(df, r_all)
        f5 = np.where(top5[:, None], feat, np.nan)
        r5 = walk_forward(df, b, f5)
        res[f"headline_{method}_top5_registered"] = headline(df, r5)
        if method == "power":
            r_all.assign(match_id=df.match_id.to_numpy()[r_all.i]).to_parquet(OUT / "rederived_power.parquet")
    b = prep(df, "power")
    # ---- per-season and per-league
    r = walk_forward(df, b, feat)
    sub = df.iloc[r.i.to_numpy()]
    res["by_season"] = r.groupby(sub.season.to_numpy()).d.mean().to_dict()
    res["by_division"] = r.groupby(sub.division.to_numpy()).d.agg(["mean", "size"]).to_dict("index")
    # coefficients on the full Pinnacle sample (sign check)
    ok = np.isfinite(feat).all(1) & (df.season.to_numpy() <= 2024)
    idx = np.where(ok)[0]
    res["coef_full_sample"] = {
        o: np.linalg.lstsq(design(b, idx, k, feat, False), b["move"][idx, k], rcond=None)[0].tolist()
        for k, o in enumerate(O3)
    }
    # ---- placebos
    plc: dict = {}
    for lag in (1, 38):
        f_l, _ = feature(df, ser, lag=lag)
        both_ok = np.where(np.isfinite(feat).all(1)[:, None], f_l, np.nan)
        plc[f"lag{lag}"] = headline(df, walk_forward(df, b, both_ok))
    # permute teams within league-season: each team's feature series replaced by another team's
    key_ls = pd.concat(
        [
            df[["hk", "division", "season"]].rename(columns={"hk": "k"}),
            df[["ak", "division", "season"]].rename(columns={"ak": "k"}),
        ]
    ).drop_duplicates()
    plc["team_perm"] = team_perm(df, b, ser, feat, key_ls)
    # shuffle feature rows within match-day
    vals = []
    days = df.match_date.to_numpy()
    for it in range(100):
        rng = np.random.default_rng(2000 + it)
        f_s = feat.copy()
        okr = np.where(np.isfinite(feat).all(1))[0]
        for _, g in pd.Series(okr).groupby(days[okr]):
            gi = g.to_numpy()
            f_s[gi] = feat[rng.permutation(gi)]
        vals.append(walk_forward(df, b, f_s).d.mean())
    plc["matchday_shuffle"] = {
        "mean": float(np.mean(vals)),
        "q95": float(np.quantile(vals, 0.95)),
        "max": float(np.max(vals)),
        "share_ge_actual": float(np.mean(np.array(vals) >= r.d.mean())),
    }
    res["placebo"] = plc
    # ---- robustness
    rob: dict = {}
    rob["rich_baseline_both"] = headline(df, walk_forward(df, b, feat, rich=True))
    for col in ("gd", "xgd"):
        f_c, _ = feature(df, ser, col=col)
        f_c = np.where(np.isfinite(feat).all(1)[:, None], f_c, np.nan)
        rob[f"{col}_instead"] = headline(df, walk_forward(df, b, f_c))
        rob[f"luck_plus_{col}_vs_{col}"] = headline(df, walk_forward_pair(df, b, f_c, np.column_stack([feat, f_c])))
    res["robustness"] = rob
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


def team_perm(df: pd.DataFrame, b: dict, ser: dict, feat: np.ndarray, key_ls: pd.DataFrame, n: int = 100) -> dict:
    """Null distribution: within each division-season, every team's feature is replaced by a randomly
    permuted other team's feature computed at the same snapshot date."""
    # Precompute each team's feature at every snapshot date of its division-season.
    snaps_by_ls = df.groupby(["division", "season"]).snap.unique()
    look: dict = {}
    for (dv, s), g in key_ls.groupby(["division", "season"]):
        for k in g.k:
            sr = ser.get(k)
            for sn in snaps_by_ls[(dv, s)]:
                if sr is None:
                    look[(k, sn)] = np.nan
                    continue
                days = sr["day"].to_numpy()
                pos = np.searchsorted(days, np.datetime64(sn), side="left")
                if (
                    pos == 0
                    or (np.datetime64(sn) - np.timedelta64(1, "D") - days[pos - 1]) / np.timedelta64(1, "D") > 150
                ):
                    look[(k, sn)] = np.nan
                    continue
                x = sr["luck"].to_numpy()[max(0, pos - 20) : pos]
                look[(k, sn)] = x.sum() / (len(x) + 10)
    ok = np.isfinite(feat).all(1)
    vals = []
    base_mean = walk_forward(df, b, feat).d.mean()
    for it in range(n):
        rng = np.random.default_rng(1000 + it)
        mp = {}
        for (dv, s), g in key_ls.groupby(["division", "season"]):
            ks = g.k.to_numpy()
            mp.update({(dv, s, a): bb for a, bb in zip(ks, rng.permutation(ks), strict=False)})
        f_p = np.full_like(feat, np.nan)
        for i in np.where(ok)[0]:
            dv, s, sn = df.division.iat[i], df.season.iat[i], df.snap.iat[i]
            f_p[i, 0] = look[(mp[(dv, s, df.hk.iat[i])], sn)]
            f_p[i, 1] = look[(mp[(dv, s, df.ak.iat[i])], sn)]
        f_p[~ok] = np.nan
        vals.append(walk_forward(df, b, f_p).d.mean())
    v = np.array(vals)
    return {
        "actual": float(base_mean),
        "mean": float(np.nanmean(v)),
        "q95": float(np.nanquantile(v, 0.95)),
        "max": float(np.nanmax(v)),
        "share_ge_actual": float(np.nanmean(v >= base_mean)),
        "n_perm": n,
    }


def walk_forward_pair(df: pd.DataFrame, b: dict, f_base: np.ndarray, f_model: np.ndarray) -> pd.DataFrame:
    """Like walk_forward, but the baseline also carries `f_base` (does luck add over the alternative?)."""
    ok = np.isfinite(f_model).all(1)
    out = []
    for s in range(2019, 2025):
        tr = np.where(ok & (df.season.to_numpy() < s))[0]
        te = np.where(ok & (df.season.to_numpy() == s))[0]
        mb = np.zeros((len(te), 3))
        mm = np.zeros((len(te), 3))
        for k in range(3):
            y = b["move"][tr, k]
            cb = np.linalg.lstsq(design(b, tr, k, f_base, False), y, rcond=None)[0]
            cm = np.linalg.lstsq(design(b, tr, k, f_model, False), y, rcond=None)[0]
            mb[:, k] = design(b, te, k, f_base, False) @ cb
            mm[:, k] = design(b, te, k, f_model, False) @ cm
        lpo = logit(b["po"][te])
        qb = 1 / (1 + np.exp(-(lpo + mb)))
        qm = 1 / (1 + np.exp(-(lpo + mm)))
        qb /= qb.sum(1, keepdims=True)
        qm /= qm.sum(1, keepdims=True)
        pc = b["pc"][te]
        out.append(
            pd.DataFrame(
                {
                    "i": te,
                    "kl_open": kl_bits(pc, b["po"][te]),
                    "kl_base": kl_bits(pc, qb),
                    "kl_model": kl_bits(pc, qm),
                    "sse0": (b["move"][te] ** 2).sum(1),
                    "sse_b": ((b["move"][te] - mb) ** 2).sum(1),
                    "sse_m": ((b["move"][te] - mm) ** 2).sum(1),
                }
            )
        )
    r = pd.concat(out, ignore_index=True)
    r["d"] = r.kl_base - r.kl_model
    return r


if __name__ == "__main__":
    main()
