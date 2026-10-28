"""H2 `soft-open-vs-betfair-late-season-stakes-v1`: in the last 5 rounds, back the side with something at stake.

Stakes (point in time). For a match in a division-season, standings use only results dated strictly before its
decision day (the latest Tuesday or Friday on or before the match date: the opening snapshot). With T teams in the division-season the season length is L = 2*(T-1); a team has `left = L - played_before(d)`
matches left including this one. Last 5 rounds: both teams have 1 <= left <= 5. With points P and maximum attainable
points M = P + 3*left, a team's best final rank is 1 + #{rivals with P_j > M} and its worst is
1 + #{rivals with M_j >= P}. Zones (identical for every division): champion (rank 1), top 4, top 6, middle, relegation
(rank >= T-2; takes precedence over the others when the zones would overlap). Something at stake means the best and
worst ranks fall in different zones.

Bet: only when exactly one side has something at stake; take that side to win at max_side_open when
edge = log(max_side_open * p_ref_open[side]) > theta, p_ref_open the power de-vig of the reference book's open
(Pinnacle on train, Betfair Exchange on evaluation). A quote with max_side_open / ref_side_open > 1.5 is a data error
and the match is not eligible. Benchmark: power de-vig of the reference book's close; CLV = log(odds * p_close[side]).

Everything that decides a bet (`stakes_context`, `decide`) sees only earlier results and open prices.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "soft-open-vs-betfair-common"))
import sobf  # noqa: E402

REG_ID = "soft-open-vs-betfair-late-season-stakes-v1"
STUDY = "soft-open-vs-betfair-late-season-stakes"
OUT = sobf.REPO / "data" / "research" / STUDY
HERE = Path(__file__).resolve().parent

EXPLORATORY = ("2012-07-01", "2024-06-30")
THETAS = (0.0, 0.02, 0.05)
MIN_TRAIN_BETS = 300
FALLBACK_THETA = 0.0
CAP = 1.5
LAST_ROUNDS = 5

ID_COLS = ["division", "season", "match_date", "home_team", "away_team"]
RESULT_COLS = ["ft_home_goals", "ft_away_goals"]  # read only through earlier-dated matches
SIDE_J = {"home": 0, "away": 2}
CODE = {"home": "H", "draw": "D", "away": "A"}


def open_cols(ref: str) -> list[str]:
    """Open-price columns the decision may read for reference book `ref`: its own and the max_* opens."""
    return [f"{b}_{o}_open" for b in (ref, "max") for o in sobf.OUTCOMES]


def close_cols(ref: str) -> list[str]:
    return [f"{ref}_{o}_close" for o in sobf.OUTCOMES]


def count_columns(ref: str = "pinnacle") -> list[str]:
    """Columns the outcome-blind bet count reads: ids, past-result goals and open prices. No close, no ft_result."""
    return ID_COLS + RESULT_COLS + open_cols(ref)


def pull_columns(ref: str) -> list[str]:
    return ID_COLS + RESULT_COLS + ["ft_result"] + open_cols(ref) + close_cols(ref)


def pull_queries() -> dict[str, str]:
    """SQL for `train` (exploratory window, Pinnacle) and `eval` (evaluation window, Betfair Exchange)."""

    def q(ref: str, lo: str, hi: str) -> str:
        cols = ", ".join(pull_columns(ref))
        return (
            f"SELECT {cols} FROM fd_matches WHERE {sobf.MAIN_LEAGUE_SQL} "
            f"AND match_date BETWEEN DATE '{lo}' AND DATE '{hi}'"
        )

    return {
        "train": q("pinnacle", *EXPLORATORY),
        "eval": q("betfair_ex", *sobf.EVAL_WINDOW),
    }


# ------------------------------------------------------------------------------------------- stakes


def zone_of(rank: np.ndarray, n_teams: int) -> np.ndarray:
    """Zone code of a final rank: 0 champion, 1 top 4, 2 top 6, 3 middle, 4 relegation (rank >= n_teams - 2)."""
    r = np.asarray(rank)
    z = np.where(r <= 4, 1, np.where(r <= 6, 2, 3))
    z = np.where(r >= n_teams - 2, 4, z)
    return np.where(r == 1, 0, z)


def at_stake(points: np.ndarray, left: np.ndarray) -> np.ndarray:
    """Per team: do its best and worst possible final ranks fall in different zones?

    `points` and `left` are length-T arrays (current points, matches left). Best rank = 1 + rivals whose current
    points exceed this team's maximum attainable points; worst rank = 1 + rivals whose maximum attainable points are
    at least this team's current points.
    """
    p = np.asarray(points, dtype=float)
    m = p + 3.0 * np.maximum(np.asarray(left, dtype=float), 0.0)
    best = 1 + (p[None, :] > m[:, None]).sum(axis=1)
    worst = 1 + (m[None, :] >= p[:, None]).sum(axis=1) - 1  # minus the team itself
    t = len(p)
    return zone_of(best, t) != zone_of(worst, t)


def decision_day(match_day: np.ndarray) -> np.ndarray:
    """Date of the opening snapshot a match is priced from: the latest Tuesday or Friday on or before the match date.

    Results dated on or after this day are not known at the snapshot (Tuesday and Friday matches are played after it).
    """
    d = np.asarray(match_day, dtype="datetime64[D]")
    wd = (d.astype("int64") + 3) % 7  # Monday = 0
    back = np.minimum((wd - 1) % 7, (wd - 4) % 7)
    return d - back.astype("timedelta64[D]")


def _points(hg: np.ndarray, ag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.where(np.isnan(hg) | np.isnan(ag), 0, np.where(hg > ag, 3, np.where(hg == ag, 1, 0)))
    a = np.where(np.isnan(hg) | np.isnan(ag), 0, np.where(ag > hg, 3, np.where(hg == ag, 1, 0)))
    return h, a


def stakes_context(res: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time stakes of both sides of every match in `res` (index kept).

    `res` holds ID_COLS and RESULT_COLS for whole division-seasons. A match's context uses only matches of the same
    division-season dated strictly before its decision day (its own result, same-day and later results are never read).
    Columns: n_teams, complete (every team has exactly L matches with both goals present), home_left, away_left,
    left_ok (both sides have 1..5 matches left), last5 (left_ok and complete), home_stake, away_stake.
    """
    out = pd.DataFrame(
        {
            "n_teams": 0,
            "complete": False,
            "home_left": np.nan,
            "away_left": np.nan,
            "left_ok": False,
            "last5": False,
            "home_stake": False,
            "away_stake": False,
        },
        index=res.index,
    )
    day = pd.to_datetime(res["match_date"]).to_numpy(dtype="datetime64[D]")
    for _, g in res.groupby(["division", "season"], sort=False):
        pos = out.index.get_indexer(g.index)
        teams = np.array(sorted(set(g["home_team"]) | set(g["away_team"])))
        t = len(teams)
        season_len = 2 * (t - 1)
        h = np.searchsorted(teams, g["home_team"].to_numpy())
        a = np.searchsorted(teams, g["away_team"].to_numpy())
        hg = g["ft_home_goals"].to_numpy(dtype=float)
        ag = g["ft_away_goals"].to_numpy(dtype=float)
        played_total = np.bincount(np.r_[h, a], minlength=t)
        complete = bool((played_total == season_len).all() and not (np.isnan(hg) | np.isnan(ag)).any())
        d = day[pos]
        ud, k_match = np.unique(d, return_inverse=True)
        ph, pa = _points(hg, ag)
        dp = np.zeros((len(ud), t))
        dn = np.zeros((len(ud), t))
        np.add.at(dp, (k_match, h), ph)
        np.add.at(dp, (k_match, a), pa)
        np.add.at(dn, (k_match, h), 1)
        np.add.at(dn, (k_match, a), 1)
        zero = np.zeros((1, t))
        cum_pts = np.vstack([zero, np.cumsum(dp, axis=0)])  # row i: results on the first i distinct match dates
        cum_played = np.vstack([zero, np.cumsum(dn, axis=0)])
        cut, k = np.unique(decision_day(d), return_inverse=True)
        n_before = np.searchsorted(ud, cut, side="left")  # match dates strictly before each decision day
        pts_before = cum_pts[n_before]
        played_before = cum_played[n_before]
        left_before = season_len - played_before
        stake = np.vstack([at_stake(pts_before[i], left_before[i]) for i in range(len(cut))])
        lh, la = left_before[k, h], left_before[k, a]
        ok = (lh >= 1) & (lh <= LAST_ROUNDS) & (la >= 1) & (la <= LAST_ROUNDS)
        out.iloc[pos, out.columns.get_loc("n_teams")] = t
        out.iloc[pos, out.columns.get_loc("complete")] = complete
        out.iloc[pos, out.columns.get_loc("home_left")] = lh
        out.iloc[pos, out.columns.get_loc("away_left")] = la
        out.iloc[pos, out.columns.get_loc("left_ok")] = ok
        out.iloc[pos, out.columns.get_loc("last5")] = ok & complete
        out.iloc[pos, out.columns.get_loc("home_stake")] = stake[k, h]
        out.iloc[pos, out.columns.get_loc("away_stake")] = stake[k, a]
    for c in ("complete", "left_ok", "last5", "home_stake", "away_stake"):
        out[c] = out[c].astype(bool)
    return out


# ------------------------------------------------------------------------------------------ decision


def decide(open_frame: pd.DataFrame, ctx: pd.DataFrame, ref: str) -> pd.DataFrame:
    """Per match, from open columns and the stakes context only.

    Raises AssertionError if `open_frame` holds anything but the whitelisted open columns of `ref` and max. Columns:
    valid (six open quotes > 1), last5, asym (last5 and exactly one side at stake), side (0 home / 2 away),
    p_side, price (max_side_open), edge, ratio (price / ref_side_open), eligible (asym & valid & ratio <= 1.5).
    """
    sobf.assert_whitelist(open_frame.columns, open_cols(ref))
    rf = sobf.triple(open_frame, ref, "open")
    mx = sobf.triple(open_frame, "max", "open")
    valid = sobf.odds_ok(rf) & sobf.odds_ok(mx)
    p = np.full(rf.shape, np.nan)
    p[valid] = sobf.devig_power_rows(rf[valid])
    hs, as_ = ctx["home_stake"].to_numpy(), ctx["away_stake"].to_numpy()
    last5 = ctx["last5"].to_numpy()
    asym = last5 & (hs ^ as_)
    side = np.where(as_, 2, 0)
    rows = np.arange(len(rf))
    price, ref_side, p_side = mx[rows, side], rf[rows, side], p[rows, side]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = price / ref_side
        edge = np.log(price * p_side)
    return pd.DataFrame(
        {
            "valid": valid,
            "last5": last5,
            "asym": asym,
            "side": side,
            "p_side": p_side,
            "price": price,
            "edge": edge,
            "ratio": ratio,
            "eligible": asym & valid & ~(ratio > CAP),
        },
        index=open_frame.index,
    )


def count_bets(f: pd.DataFrame, ref: str = "pinnacle", thetas: tuple[float, ...] = THETAS) -> dict[str, Any]:
    """Outcome-blind bet count. `f` must hold exactly `count_columns(ref)`; any close or ft_result column raises.

    Returns the funnel (matches, last-5-round matches, with valid opens, asymmetric, eligible) and the number of bets
    per theta, plus the same funnel under the literal reading that does not require a complete division-season.
    """
    extra = sorted(set(f.columns) - set(count_columns(ref)))
    if extra:
        raise AssertionError(f"count reads only ids, past-result goals and opens; got extra columns {extra}")
    ctx = stakes_context(f[ID_COLS + RESULT_COLS])
    dec = decide(f[open_cols(ref)], ctx, ref)
    lit = ctx.assign(last5=ctx["left_ok"])
    dec_lit = decide(f[open_cols(ref)], lit, ref)

    def funnel(c: pd.DataFrame, d: pd.DataFrame) -> dict[str, Any]:
        return {
            "n_matches": int(len(c)),
            "n_last5": int(c["last5"].sum()),
            "n_last5_valid_opens": int((c["last5"] & d["valid"]).sum()),
            "n_asymmetric": int(d["asym"].sum()),
            "n_asymmetric_valid_opens": int((d["asym"] & d["valid"]).sum()),
            "n_eligible": int(d["eligible"].sum()),
            "bets": {str(t): int((d["eligible"] & (d["edge"] > t)).sum()) for t in thetas},
        }

    out = funnel(ctx, dec)
    out["n_division_seasons"] = int(f.groupby(["division", "season"]).ngroups)
    out["n_division_seasons_incomplete"] = int(
        pd.concat([ctx[["complete"]], f[["division", "season"]]], axis=1)
        .groupby(["division", "season"])["complete"]
        .first()
        .eq(False)
        .sum()
    )
    out["literal_no_completeness_requirement"] = funnel(lit, dec_lit)
    return out


# --------------------------------------------------------------------------------------- scoring side


def universe(f: pd.DataFrame, ref: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Matches with valid opens and a valid reference close, their decisions and the (n, 3) close probabilities.

    `f` holds whole division-seasons (stakes need every earlier result). Decisions use ids, goals and opens only;
    the close columns are read afterwards to build the benchmark.
    """
    ctx = stakes_context(f[ID_COLS + RESULT_COLS])
    dec = decide(f[open_cols(ref)], ctx, ref)
    close = sobf.triple(f, ref, "close")
    keep = dec["valid"].to_numpy() & sobf.odds_ok(close)
    return f[keep], dec[keep], sobf.devig_power_rows(close[keep])


def bets_for(f: pd.DataFrame, dec: pd.DataFrame, p_close: np.ndarray, theta: float) -> pd.DataFrame:
    """Bets of one theta: eligible asymmetric matches with edge > theta; at most one per match."""
    take = dec["eligible"].to_numpy() & (dec["edge"].to_numpy() > theta)
    return sobf.bets_frame(f, dec["side"].to_numpy(), dec["price"].to_numpy(), p_close, take)


def select_key(r: dict[str, Any]) -> tuple[float, float]:
    """Highest mean CLV (rounded to 12 decimals against float noise), then lower theta."""
    return (-round(float(r["mean_clv"]), 12), r["theta"])


def select(train: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose theta on the exploratory window only (Pinnacle open fair, Pinnacle close benchmark).

    Eligible candidates have >= 300 train bets; the best by mean CLV wins, ties to lower theta; with none eligible
    theta = 0.0. Returns (selection dict, candidate rows).
    """
    t = sobf.in_window(train, *EXPLORATORY)
    f, dec, pc = universe(t, "pinnacle")
    rows = []
    for theta in THETAS:
        b = bets_for(f, dec, pc, theta)
        s = sobf.boot(b["clv"].to_numpy(), b["match_date"].dt.strftime("%Y-%m-%d"))
        rows.append(
            {
                "theta": theta,
                "n_bets": int(s["n"]),
                "eligible": bool(s["n"] >= MIN_TRAIN_BETS),
                "mean_clv": s["mean"] if s["n"] else float("nan"),
                "total_clv": float(b["clv"].sum()),
                "ci_lo": s["lo"],
                "ci_hi": s["hi"],
            }
        )
    best = sobf.pick(rows, select_key, MIN_TRAIN_BETS)
    fallback = best is None
    chosen = FALLBACK_THETA if best is None else best["theta"]
    return {"theta": chosen, "fallback": fallback, "candidates": rows}, rows
