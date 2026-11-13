"""H4 `soft-open-vs-betfair-exchange-lag-v1`: back an outcome on the Betfair Exchange when Pinnacle has just shortened
it within the hour before kickoff and the exchange price has not moved.

Data: OddsPapi 1x2 price paths of `pinnacle` and `betfair-ex` (lake `oddspapi_odds`, change-only: a row exists only
when the price or the active flag changes) and the fixtures' `kickoff_utc` (`oddspapi_fixtures`). The point-in-time
column of every path row is `observed_at`; it is held here as int64 nanoseconds UTC (`*_ns`).

Path reading (amendment 1). A stored row holds until the next row of the same fixture, book and outcome, so the quote
in force at t is the latest row with observed_at <= t; with no such row there is no quote. Invalid quotes (amendment
2): a row with price <= MIN_VALID_PRICE is not a quote, so from its observed_at until the next row with a valid price
the book has no valid price in force for that outcome. Both rules reduce to: the latest row at or before t must carry
a price > MIN_VALID_PRICE.

Decision for outcome o of a fixture with kickoff k: the first candidate time t with k - 60 min <= t < k, the candidate
times being the observed_at stamps of the fixture's pinnacle and betfair-ex rows, at which all of these hold:
  - pinnacle and betfair-ex each have a valid quote for o in force at t and at t - 60 min;
  - Pinnacle shortened: 1 - pin(t) / pin(t - 60 min) >= s;
  - the exchange did not move: |bf(t) / bf(t - 60 min) - 1| <= b;
  - Betfair liveness: at least one betfair-ex row of the fixture (any outcome, valid price) has
    t - 60 min <= observed_at <= t;
  - the betfair-ex quote in force at t for o is active.
At the decision, bet 1 unit at bf(t) when the betfair-ex stake_limit in force at t is >= STAKE_S; otherwise there is no
bet on that outcome. One bet per match: the earliest decision time among outcomes with a bet, then the largest Pinnacle
shortening, then home, draw, away order.

Benchmark (amendment 1 and 2): per outcome the latest pinnacle row with observed_at <= k. The close is valid only when
all three outcomes have one, each has a valid price and is active, and the most recent of the three is at most 15 min
before k. The close is power de-vigged. It is computed only from a `FrozenDecisions` object (the decisions are fixed
first) and reads pinnacle rows only, never a result or Betfair data. clv_log = log(eff_odds * p_close[o]) with
eff_odds = 1 + (odds - 1) * (1 - c).

Everything that decides a bet (`decision_points`, `decide`) reads rows with observed_at <= t only; `assert_pit`
checks every path point a decision used.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "soft-open-vs-betfair-common"))
import sobf  # noqa: E402

REG_ID = "soft-open-vs-betfair-exchange-lag-v1"
HERE = Path(__file__).resolve().parent
OUT = sobf.REPO / "data" / "research" / REG_ID
PATHS_DIR = OUT / "paths"

KICKOFF_RANGE = ("2026-01-13", "2026-10-19")  # inclusive kickoff dates (UTC)
EXPLORATORY = ("2026-01-13", "2026-05-31")
EVALUATION = ("2026-06-01", "2026-10-19")
WINDOWS = {"exploratory": EXPLORATORY, "evaluation": EVALUATION}
CHUNK_DAYS = 7

PIN, BF = "pinnacle", "betfair-ex"
BOOKS = (PIN, BF)
MARKET = "1x2"
OUTCOMES = sobf.OUTCOMES  # ("home", "draw", "away"): index j = 0, 1, 2
OUTCOME_LABEL = {"home": "home", "draw": "draw", "away": "away", "101": "home", "102": "draw", "103": "away"}

S_GRID = (0.02, 0.03, 0.05)
B_GRID = (0.0, 0.01)
CANDIDATES = tuple((s, b) for s in S_GRID for b in B_GRID)
FALLBACK = (0.03, 0.0)
MIN_TRAIN_BETS = 100
MIN_N = 300
STAKE_S = 25.0
COMMISSION = 0.02
COMMISSION_ALT = 0.05
CEILING = 0.02
MIN_VALID_PRICE = 1.0  # a row with price <= this is not a quote (amendment 2)
EPS = 1e-12  # float tolerance on the s and b thresholds

MINUTE_NS = 60 * 10**9
LOOKBACK_NS = 60 * MINUTE_NS  # t - 60 min reference and the liveness window
DECISION_SPAN_NS = 60 * MINUTE_NS  # decisions at kickoff - 60 min <= t < kickoff
CLOSE_MAX_AGE_NS = 15 * MINUTE_NS

PATH_COLS = ["fixture_id", "bookmaker", "outcome", "observed_ns", "price", "active", "stake_limit"]
CLOSE_READ_COLS = ["fixture_id", "bookmaker", "outcome", "observed_ns", "price", "active"]
POINT_COLS = [
    "fixture_id", "kickoff_ns", "t_ns", "outcome", "j", "shortening", "bf_change", "pin_t", "pin_ref", "odds",
    "bf_ref", "bf_size", "pin_active_t", "pin_ts_t", "pin_ts_ref", "bf_ts_t", "bf_ts_ref", "live_ts",
]  # fmt: skip


_TS = pa.timestamp("us", tz="UTC")
PATH_SCHEMA = pa.schema(
    [
        ("fixture_id", pa.string()), ("bookmaker", pa.string()), ("outcome", pa.string()), ("observed_at", _TS),
        ("price", pa.float64()), ("stake_limit", pa.float64()), ("active", pa.bool_()), ("kickoff_utc", _TS),
        ("tournament_id", pa.int64()),
    ]
)  # fmt: skip
FIXTURE_SCHEMA = pa.schema(
    [
        ("fixture_id", pa.string()), ("tournament_id", pa.int64()), ("tournament_name", pa.string()),
        ("division", pa.string()), ("season_id", pa.int64()), ("kickoff_utc", _TS), ("home_team", pa.string()),
        ("away_team", pa.string()), ("status_name", pa.string()),
    ]
)  # fmt: skip
FETCH_STATUS_SCHEMA = pa.schema(
    [
        ("fixture_id", pa.string()), ("status", pa.string()), ("pinnacle_rows", pa.int64()),
        ("betfair_ex_rows", pa.int64()), ("first_observed_at", _TS), ("last_observed_at", _TS), ("fetched_at", _TS),
    ]
)  # fmt: skip
FETCH_STATUS_READ = ("ok", "no_history")  # a fixture with any other status, or none, has not been read by the lake


def conform(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """`table` with exactly `schema`'s columns, in order and cast to its types; a zero-row or column-less table comes
    back as an empty table that still carries the schema."""
    if table.num_rows == 0:
        return schema.empty_table()
    return table.select(schema.names).cast(schema)


# ------------------------------------------------------------------------------------------- windows


def day_ns(iso: str) -> int:
    """00:00 UTC of an ISO date as int64 nanoseconds."""
    return int(pd.Timestamp(iso, tz="UTC").value)


def window_of(kickoff_ns: np.ndarray | pd.Series) -> np.ndarray:
    """'exploratory' / 'evaluation' / '' for each kickoff (inclusive UTC kickoff dates of the registered windows)."""
    k = np.asarray(kickoff_ns, dtype=np.int64)
    out = np.full(k.shape, "", dtype=object)
    for name, (lo, hi) in WINDOWS.items():
        out[(k >= day_ns(lo)) & (k < day_ns(hi) + 24 * 60 * MINUTE_NS)] = name
    return out


def window_chunks(lo: str, hi: str, days: int = CHUNK_DAYS) -> list[tuple[str, str]]:
    """Kickoff-date chunks [start, end) covering the inclusive range lo..hi, `days` long (the last one may be shorter)."""
    out = []
    d, end = pd.Timestamp(lo), pd.Timestamp(hi) + pd.Timedelta(days=1)
    while d < end:
        e = min(d + pd.Timedelta(days=days), end)
        out.append((d.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
        d = e
    return out


def chunks(window: str | None = None) -> list[tuple[str, str]]:
    """The pull's kickoff chunks, built per registered window so no chunk straddles the window boundary; with
    `window` given, only that window's chunks."""
    names = list(WINDOWS) if window is None else [window]
    return [c for w in names for c in window_chunks(*WINDOWS[w])]


def paths_sql(lo: str, hi_exclusive: str) -> str:
    """Pinnacle and betfair-ex 1x2 rows for kickoffs in [lo, hi_exclusive) stamped at or before kickoff."""
    return (
        "SELECT fixture_id, bookmaker, outcome, observed_at, price, stake_limit, active, kickoff_utc, tournament_id "
        f"FROM oddspapi_odds WHERE market = '{MARKET}' AND bookmaker IN ('{PIN}', '{BF}') "
        f"AND kickoff_utc >= TIMESTAMPTZ '{lo} 00:00:00+00' AND kickoff_utc < TIMESTAMPTZ '{hi_exclusive} 00:00:00+00' "
        "AND observed_at <= kickoff_utc ORDER BY fixture_id, bookmaker, outcome, observed_at"
    )


def fixtures_sql() -> str:
    lo, hi = KICKOFF_RANGE
    return (
        "SELECT fixture_id, tournament_id, tournament_name, division, season_id, kickoff_utc, home_team, away_team, "
        f"status_name FROM oddspapi_fixtures WHERE kickoff_utc >= TIMESTAMPTZ '{lo} 00:00:00+00' "
        f"AND kickoff_utc < TIMESTAMPTZ '{(pd.Timestamp(hi) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} 00:00:00+00'"
    )


def fetch_status_sql() -> str:
    return (
        "SELECT fixture_id, status, pinnacle_rows, betfair_ex_rows, first_observed_at, last_observed_at, fetched_at "
        "FROM oddspapi_fetch_status"
    )


# --------------------------------------------------------------------------------------- prices, CLV


def valid_price(p: np.ndarray) -> np.ndarray:
    """True where a stored price is a quote: finite and > MIN_VALID_PRICE."""
    a = np.asarray(p, dtype=float)
    return np.isfinite(a) & (a > MIN_VALID_PRICE)


def effective_odds(odds: np.ndarray, c: float) -> np.ndarray:
    """Exchange odds net of commission c on net winnings: 1 + (odds - 1) * (1 - c)."""
    return 1.0 + (np.asarray(odds, dtype=float) - 1.0) * (1.0 - c)


def clv_log(odds: np.ndarray, p_close: np.ndarray, c: float) -> np.ndarray:
    """log(effective_odds(odds, c) * p_close), p_close the de-vigged close probability of the side taken."""
    return np.log(effective_odds(odds, c) * np.asarray(p_close, dtype=float))


# ---------------------------------------------------------------------------------------------- paths


@dataclass(frozen=True)
class QuotePath:
    """One fixture/book/outcome path: ascending unique observed_at (`ts`, int64 ns), price, active, stake_limit."""

    ts: np.ndarray
    price: np.ndarray
    active: np.ndarray
    size: np.ndarray

    def index_at(self, t: np.ndarray) -> np.ndarray:
        """Index of the row in force at each t (latest observed_at <= t); -1 when no row is stamped by t."""
        return np.searchsorted(self.ts, np.asarray(t, dtype=np.int64), side="right") - 1


Paths = dict[tuple[str, str], QuotePath]


def build_paths(rows: pd.DataFrame) -> Paths:
    """Paths of one fixture keyed (bookmaker, outcome label) from rows with PATH_COLS (stake_limit optional).

    Raises ValueError on an unknown outcome code or two rows with the same fixture, book, outcome and observed_at.
    """
    if not len(rows):
        return {}
    label = rows["outcome"].astype(str).map(OUTCOME_LABEL)
    if label.isna().any():
        raise ValueError(f"unknown outcome codes {sorted(set(rows['outcome'].astype(str)[label.isna()]))}")
    r = rows.assign(outcome=label.to_numpy()).sort_values(["bookmaker", "outcome", "observed_ns"], kind="stable")
    out: Paths = {}
    for (bk, o), g in r.groupby(["bookmaker", "outcome"], sort=False):
        ts = g["observed_ns"].to_numpy(dtype=np.int64)
        if len(ts) > 1 and not (np.diff(ts) > 0).all():
            raise ValueError(f"duplicate observed_at on {g['fixture_id'].iloc[0]} {bk} {o}")
        size = g["stake_limit"].to_numpy(dtype=float) if "stake_limit" in g else np.full(len(g), np.nan)
        out[(str(bk), str(o))] = QuotePath(
            ts, g["price"].to_numpy(dtype=float), g["active"].fillna(False).to_numpy(dtype=bool), size
        )
    return out


def candidate_times(paths: Paths, kickoff_ns: int) -> np.ndarray:
    """Sorted unique observed_at stamps of the fixture's pinnacle and betfair-ex rows with k - 60 min <= t < k."""
    stamps = [p.ts for (bk, _), p in paths.items() if bk in BOOKS]
    if not stamps:
        return np.empty(0, dtype=np.int64)
    ts = np.unique(np.concatenate(stamps))
    return ts[(ts >= kickoff_ns - DECISION_SPAN_NS) & (ts < kickoff_ns)]


def _pick(arr: np.ndarray, idx: np.ndarray, fill: float | bool) -> np.ndarray:
    """arr[idx] where idx >= 0, else `fill` (float output for a float fill, the array's dtype otherwise)."""
    a = np.asarray(arr)
    ok = idx >= 0
    out = np.full(idx.shape, fill, dtype=float if isinstance(fill, float) else a.dtype)
    out[ok] = a[idx[ok]]
    return out


_INT_COLS = ("kickoff_ns", "t_ns", "j", "pin_ts_t", "pin_ts_ref", "bf_ts_t", "bf_ts_ref", "live_ts")


def empty_points() -> pd.DataFrame:
    """A zero-row frame with POINT_COLS and their dtypes."""
    dt = {c: "int64" if c in _INT_COLS else "float64" for c in POINT_COLS}
    dt.update({"fixture_id": "object", "outcome": "object", "pin_active_t": "bool"})
    return pd.DataFrame({c: pd.Series(dtype=d) for c, d in dt.items()})


def decision_points(
    fixture_id: str, kickoff_ns: int, paths: Paths, s_min: float = min(S_GRID), b_max: float = max(B_GRID)
) -> pd.DataFrame:
    """Every (t, outcome) of one fixture at which the decision conditions hold for the loosest candidate (s_min, b_max).

    Each candidate's first decision time per outcome is among these rows, so `decide` filters them further. Columns
    POINT_COLS; prices and stake_limit are the quotes in force at t (and at t - 60 min for `*_ref`), `*_ts_*` the
    observed_at of the rows used, `live_ts` the latest valid betfair-ex row within [t - 60 min, t]. Reads no row
    stamped after t (asserted by `assert_pit`).
    """
    T = candidate_times(paths, kickoff_ns)
    empty = empty_points()
    if not len(T):
        return empty
    bf_stamps = [p.ts[valid_price(p.price)] for (bk, _), p in paths.items() if bk == BF]
    live_all = np.sort(np.concatenate(bf_stamps)) if bf_stamps else np.empty(0, dtype=np.int64)
    hi = np.searchsorted(live_all, T, side="right")
    lo = np.searchsorted(live_all, T - LOOKBACK_NS, side="left")
    live = hi > lo
    live_ts = np.where(live, live_all[np.maximum(hi - 1, 0)] if len(live_all) else 0, -1)
    parts = []
    for j, o in enumerate(OUTCOMES):
        P, B = paths.get((PIN, o)), paths.get((BF, o))
        if P is None or B is None:
            continue
        ip, ipr, ib, ibr = P.index_at(T), P.index_at(T - LOOKBACK_NS), B.index_at(T), B.index_at(T - LOOKBACK_NS)
        pin_t, pin_ref = _pick(P.price, ip, np.nan), _pick(P.price, ipr, np.nan)
        bf_t, bf_ref = _pick(B.price, ib, np.nan), _pick(B.price, ibr, np.nan)
        valid = valid_price(pin_t) & valid_price(pin_ref) & valid_price(bf_t) & valid_price(bf_ref)
        with np.errstate(invalid="ignore", divide="ignore"):
            shortening = 1.0 - pin_t / pin_ref
            bf_change = np.abs(bf_t / bf_ref - 1.0)
        bf_active = _pick(B.active, ib, False)
        keep = valid & live & bf_active & (shortening >= s_min - EPS) & (bf_change <= b_max + EPS)
        if not keep.any():
            continue
        k = np.flatnonzero(keep)
        parts.append(
            pd.DataFrame(
                {
                    "fixture_id": fixture_id,
                    "kickoff_ns": np.int64(kickoff_ns),
                    "t_ns": T[k],
                    "outcome": o,
                    "j": j,
                    "shortening": shortening[k],
                    "bf_change": bf_change[k],
                    "pin_t": pin_t[k],
                    "pin_ref": pin_ref[k],
                    "odds": bf_t[k],
                    "bf_ref": bf_ref[k],
                    "bf_size": _pick(B.size, ib, np.nan)[k],
                    "pin_active_t": _pick(P.active, ip, False)[k],
                    "pin_ts_t": P.ts[ip[k]],
                    "pin_ts_ref": P.ts[ipr[k]],
                    "bf_ts_t": B.ts[ib[k]],
                    "bf_ts_ref": B.ts[ibr[k]],
                    "live_ts": live_ts[k],
                }
            )
        )
    if not parts:
        return empty
    pts = pd.concat(parts, ignore_index=True)
    assert_pit(pts)
    return pts


def assert_pit(points: pd.DataFrame) -> None:
    """Raise AssertionError unless every path point a decision used was stamped at or before its decision time t.

    Checks: k - 60 min <= t < k; the quotes in force at t were stamped <= t; the t - 60 min references were stamped
    <= t - 60 min; the liveness row lies in [t - 60 min, t].
    """
    if not len(points):
        return
    t = points["t_ns"].to_numpy(dtype=np.int64)
    k = points["kickoff_ns"].to_numpy(dtype=np.int64)
    checks = {
        "t < kickoff": t < k,
        "t >= kickoff - 60 min": t >= k - DECISION_SPAN_NS,
        "pinnacle quote at t stamped <= t": points["pin_ts_t"].to_numpy(dtype=np.int64) <= t,
        "betfair quote at t stamped <= t": points["bf_ts_t"].to_numpy(dtype=np.int64) <= t,
        "pinnacle reference stamped <= t - 60 min": points["pin_ts_ref"].to_numpy(dtype=np.int64) <= t - LOOKBACK_NS,
        "betfair reference stamped <= t - 60 min": points["bf_ts_ref"].to_numpy(dtype=np.int64) <= t - LOOKBACK_NS,
        "liveness row stamped <= t": points["live_ts"].to_numpy(dtype=np.int64) <= t,
        "liveness row stamped >= t - 60 min": points["live_ts"].to_numpy(dtype=np.int64) >= t - LOOKBACK_NS,
    }
    bad = {name: int((~ok).sum()) for name, ok in checks.items() if not ok.all()}
    if bad:
        raise AssertionError(f"point-in-time violation: {bad}")


# ------------------------------------------------------------------------------------------- decision


def decide(points: pd.DataFrame, s: float, b: float) -> tuple[pd.DataFrame, dict[str, int]]:
    """Bets of candidate (s, b) from `decision_points` rows (any number of fixtures).

    Per fixture and outcome the decision time is the first t with shortening >= s and bf_change <= b; it is a bet when
    the stake_limit in force at t is >= STAKE_S (a missing size is no bet). One bet per fixture: earliest t, then the
    largest shortening, then outcome order. Returns (bets with POINT_COLS, funnel counts).
    """
    m = points[(points["shortening"] >= s - EPS) & (points["bf_change"] <= b + EPS)]
    first = m.sort_values(["fixture_id", "j", "t_ns"], kind="stable").groupby(["fixture_id", "j"], sort=False).head(1)
    size_ok = (first["bf_size"].to_numpy(dtype=float) >= STAKE_S - EPS) & np.isfinite(first["bf_size"].to_numpy(float))
    sized = first[size_ok]
    bets = (
        sized.sort_values(["fixture_id", "t_ns", "shortening", "j"], ascending=[True, True, False, True], kind="stable")
        .groupby("fixture_id", sort=False)
        .head(1)
        .sort_values(["kickoff_ns", "fixture_id"], kind="stable")
        .reset_index(drop=True)
    )
    assert_pit(bets)
    funnel = {
        "fixtures_with_decision": int(first["fixture_id"].nunique()),
        "outcome_decisions": int(len(first)),
        "outcome_decisions_size_fail": int((~size_ok).sum()),
        "bets": int(len(bets)),
    }
    return bets, funnel


# ---------------------------------------------------------------------------------------- the close


DECISION_COLS = ["fixture_id", "kickoff_ns", "t_ns", "j", "odds", "bf_size", "shortening"]


def _digest(frame: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(frame[DECISION_COLS].reset_index(drop=True), index=False).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()


@dataclass(frozen=True)
class FrozenDecisions:
    """Bets fixed before any close is read: a private copy of the frame and its digest over DECISION_COLS."""

    frame: pd.DataFrame
    digest: str


def freeze(bets: pd.DataFrame) -> FrozenDecisions:
    """Fix the decisions (PIT-checked) so a close can be attached; the close functions take only this type."""
    assert_pit(bets)
    f = bets.copy()
    return FrozenDecisions(f, _digest(f))


def close_screen(kickoff_ns: int, paths: Paths) -> dict[str, Any]:
    """Pinnacle close of one fixture: per outcome the latest pinnacle row stamped <= kickoff.

    Valid only when all three outcomes have one, each price is valid (> MIN_VALID_PRICE), each is active and the most
    recent of the three is at most 15 min before kickoff. Returns close_valid, reason ('' / no_rows / missing_outcome
    / invalid_price / inactive / stale), p_home/p_draw/p_away (power de-vig, NaN if invalid), close_ts (most recent
    stamp) and stale_any_outcome (some outcome's latest row is older than 15 min: the stricter reading, a diagnostic).
    Raises AssertionError when given any non-pinnacle path.
    """
    books = {bk for bk, _ in paths}
    if books - {PIN}:
        raise AssertionError(f"the close screen reads pinnacle only; got {sorted(books)}")
    out: dict[str, Any] = {"close_valid": False, "reason": "", "close_ts": -1, "stale_any_outcome": False}
    out.update({f"p_{o}": np.nan for o in OUTCOMES})
    if not paths:
        out["reason"] = "no_rows"
        return out
    price, active, ts = [], [], []
    for o in OUTCOMES:
        p = paths.get((PIN, o))
        i = -1 if p is None else int(p.index_at(np.array([kickoff_ns]))[0])
        if p is None or i < 0:
            out["reason"] = "missing_outcome"
            return out
        price.append(p.price[i])
        active.append(bool(p.active[i]))
        ts.append(int(p.ts[i]))
    out["close_ts"] = max(ts)
    out["stale_any_outcome"] = bool(min(ts) < kickoff_ns - CLOSE_MAX_AGE_NS)
    if not valid_price(np.array(price)).all():
        out["reason"] = "invalid_price"
    elif not all(active):
        out["reason"] = "inactive"
    elif max(ts) < kickoff_ns - CLOSE_MAX_AGE_NS:
        out["reason"] = "stale"
    else:
        pc = sobf.devig_power_rows(np.array([price]))[0]
        out.update({f"p_{o}": float(pc[j]) for j, o in enumerate(OUTCOMES)})
        out["close_valid"] = True
    return out


def close_table(frozen: FrozenDecisions, groups: Iterable[tuple[str, int, pd.DataFrame]]) -> pd.DataFrame:
    """Close screen for each fixture that holds a frozen bet. `groups` yields (fixture_id, kickoff_ns, pinnacle rows
    with CLOSE_READ_COLS); fixtures without a bet are skipped, and fixtures with a bet but no group get 'no_rows'."""
    if not isinstance(frozen, FrozenDecisions):
        raise AssertionError("the close is read only after the decisions are frozen (pass a FrozenDecisions)")
    want = set(frozen.frame["fixture_id"])
    kick = dict(zip(frozen.frame["fixture_id"], frozen.frame["kickoff_ns"], strict=True))
    rows = []
    for fid, ko, g in groups:
        if fid not in want:
            continue
        if int(ko) != int(kick[fid]):
            raise AssertionError(f"kickoff of {fid} differs between decision and close")
        rows.append({"fixture_id": fid, **close_screen(int(ko), build_paths(g))})
        want.discard(fid)
    for fid in sorted(want):
        rows.append({"fixture_id": fid, **close_screen(int(kick[fid]), {})})
    cols = ["fixture_id", "close_valid", "reason", "close_ts", "stale_any_outcome"] + [f"p_{o}" for o in OUTCOMES]
    return pd.DataFrame(rows, columns=cols)


def attach_close(frozen: FrozenDecisions, closes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Score frozen bets against the Pinnacle close. Drops bets whose match fails the screen and counts the drops by
    reason. Adds p_close, clv (net of COMMISSION), clv_c05 (net of COMMISSION_ALT) and clv_gross."""
    if not isinstance(frozen, FrozenDecisions) or _digest(frozen.frame) != frozen.digest:
        raise AssertionError("decisions changed after they were frozen")
    b = frozen.frame.merge(closes, on="fixture_id", how="left", validate="one_to_one")
    if not (b["t_ns"] < b["kickoff_ns"]).all():
        raise AssertionError("a decision is not before kickoff")
    ok = b["close_valid"].fillna(False).astype(bool).to_numpy()
    reasons = b.loc[~ok, "reason"].fillna("no_rows").replace("", "no_rows")
    drops = {str(k): int(v) for k, v in reasons.value_counts().sort_index().items()}
    kept = b[ok].reset_index(drop=True)
    pc = kept[[f"p_{o}" for o in OUTCOMES]].to_numpy(dtype=float)
    kept["p_close"] = pc[np.arange(len(kept)), kept["j"].to_numpy(dtype=int)] if len(kept) else np.empty(0)
    odds, p = kept["odds"].to_numpy(dtype=float), kept["p_close"].to_numpy(dtype=float)
    kept["clv"] = clv_log(odds, p, COMMISSION)
    kept["clv_c05"] = clv_log(odds, p, COMMISSION_ALT)
    kept["clv_gross"] = clv_log(odds, p, 0.0)
    return kept, drops


# ------------------------------------------------------------------------------------------ selection


def select_key(r: dict[str, Any]) -> tuple[float, float, float]:
    """Highest mean clv (net, rounded to 12 decimals against float noise), then higher s, then lower b."""
    return (-round(float(r["mean_clv"]), 12), -float(r["s"]), float(r["b"]))


def select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Selected (s, b) from candidate rows (keys s, b, n_bets, mean_clv): eligible rows have >= MIN_TRAIN_BETS scored
    train bets; the best by `select_key` wins; with none eligible the fallback (0.03, 0.0)."""
    best = sobf.pick(rows, select_key, MIN_TRAIN_BETS)
    s, b = FALLBACK if best is None else (float(best["s"]), float(best["b"]))
    return {"s": s, "b": b, "fallback": best is None}


# -------------------------------------------------------------------------------------- scoring frames


def tournament_table(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Per fixture_id (index): tournament (the OddsPapi tournament_id as text, amendment 4), tournament_name and
    division (reported beside the id; a name such as "Championship" is not unique across countries)."""
    return pd.DataFrame(
        {
            "tournament": fixtures["tournament_id"].astype("Int64").astype(str).to_numpy(),
            "tournament_name": fixtures["tournament_name"].astype("object").to_numpy(),
            "tournament_division": fixtures["division"].astype("object").to_numpy(),
        },
        index=fixtures["fixture_id"].astype(str).to_numpy(),
    )


def scored_frame(bets: pd.DataFrame, tournaments: pd.DataFrame) -> pd.DataFrame:
    """Scored bets plus match_date (kickoff UTC date), division (= tournament id, the column sobf helpers group by),
    tournament_name and tournament_division."""
    out = bets.copy()
    out["match_date"] = pd.to_datetime(out["kickoff_ns"].to_numpy(dtype=np.int64), unit="ns").normalize()
    t = tournaments.reindex(out["fixture_id"].astype(str).to_numpy())
    out["division"] = t["tournament"].fillna("UNKNOWN").to_numpy()
    out["tournament_name"] = t["tournament_name"].to_numpy()
    out["tournament_division"] = t["tournament_division"].to_numpy()
    return out


def per_decision(scored: pd.DataFrame) -> pd.DataFrame:
    """The run_trial frame: decision_id = fixture_id (one bet per match), block = kickoff UTC date, value = net clv,
    league = the raw tournament id (no pooling, amendment 4)."""
    day = pd.to_datetime(scored["match_date"]).dt.strftime("%Y-%m-%d")
    out = pd.DataFrame(
        {
            "decision_id": scored["fixture_id"].astype(str).to_numpy(),
            "block": day.to_numpy(),
            "value": scored["clv"].to_numpy(dtype=float),
            "league": scored["division"].astype(str).to_numpy(),
        }
    )
    return out.sort_values(["block", "decision_id"]).reset_index(drop=True)


def stability(scored: pd.DataFrame, lo: str, hi: str) -> dict[str, Any]:
    """The registered stability checks on raw tournament ids (amendment 4).

    Calendar thirds of [lo, hi] with block-bootstrap CIs; per tournament (n, mean, name, division); leave one
    tournament out with no pooling; the disconfirming flags (negative mean in >= 2 of 3 thirds; tournaments whose
    removal leaves a CI that includes 0 or lies below it). The STANDARDS pooled-below-100 table is added under
    `pooled_diagnostic` (sobf.stability) and is not a registered check.
    """
    day = pd.to_datetime(scored["match_date"]).dt.strftime("%Y-%m-%d").to_numpy()
    tour = scored["division"].astype(str).to_numpy()
    clv = scored["clv"].to_numpy(dtype=float)
    th = sobf.calendar_thirds(scored["match_date"], lo, hi).to_numpy()
    thirds = {str(k): sobf.boot(clv[th == k], day[th == k]) for k in (1, 2, 3)}
    per_t, loto = {}, {}
    for k in sorted(set(tour)):
        m = tour == k
        first = scored[m].iloc[0]
        per_t[k] = {
            "n": int(m.sum()),
            "mean": float(clv[m].mean()),
            "name": None if pd.isna(first["tournament_name"]) else str(first["tournament_name"]),
            "division": None if pd.isna(first["tournament_division"]) else str(first["tournament_division"]),
        }
        loto[k] = sobf.boot(clv[~m], day[~m])
    neg = sum(1 for t in thirds.values() if t["n"] and t["mean"] < 0)
    return {
        "thirds": thirds,
        "per_tournament": per_t,
        "leave_one_tournament_out": loto,
        "negative_thirds": int(neg),
        "disconfirm_two_of_three_thirds_negative": bool(neg >= 2),
        "disconfirm_loto_ci_includes_zero_for": sorted(k for k, t in loto.items() if not (t["lo"] > 0)),
        "pooled_diagnostic": sobf.stability(scored, lo, hi)["per_division"] if len(scored) else {},
    }


# ------------------------------------------------------------------------------------------- outages


def daily_counts(observed_ns: np.ndarray, bookmaker: np.ndarray) -> pd.DataFrame:
    """Rows per UTC observed_at day and bookmaker: columns day (ISO), bookmaker, n."""
    d = pd.to_datetime(np.asarray(observed_ns, dtype=np.int64), unit="ns").strftime("%Y-%m-%d")
    f = pd.DataFrame({"day": d, "bookmaker": np.asarray(bookmaker).astype(str)})
    return f.groupby(["day", "bookmaker"], as_index=False).size().rename(columns={"size": "n"})


def outage_days(counts: pd.DataFrame, kickoff_days: pd.Series, lo: str, hi: str) -> list[dict[str, Any]]:
    """Days in [lo, hi] on which a book has no rows across all fixtures, with the number of kickoffs that day.

    `counts` sums `daily_counts` over the pull; `kickoff_days` is the ISO kickoff day of each universe fixture.
    """
    days = pd.date_range(lo, hi, freq="D").strftime("%Y-%m-%d")
    have = counts.groupby(["day", "bookmaker"])["n"].sum()
    kicks = kickoff_days.value_counts()
    out = []
    for book in BOOKS:
        for d in days:
            if int(have.get((d, book), 0)) == 0:
                out.append({"day": d, "bookmaker": book, "n_kickoffs": int(kicks.get(d, 0))})
    return out


# ------------------------------------------------------------------------------------------- reading


def iter_fixture_groups(
    frame: pd.DataFrame, kickoff: pd.Series, books: Iterable[str] = BOOKS
) -> Iterator[tuple[str, int, pd.DataFrame]]:
    """(fixture_id, kickoff_ns, rows) for each fixture of `frame` present in `kickoff` (fixture_id -> kickoff_ns),
    keeping only `books` rows stamped at or before kickoff."""
    f = frame[frame["bookmaker"].isin(list(books))]
    for fid, g in f.groupby("fixture_id", sort=True):
        if fid not in kickoff.index:
            continue
        ko = int(kickoff[fid])
        yield str(fid), ko, g[g["observed_ns"].to_numpy(dtype=np.int64) <= ko]


def _to_ns(col: Any) -> Any:
    import pyarrow.compute as pc

    return pc.cast(pc.cast(col, pa.timestamp("ns", tz="UTC")), pa.int64())


_PANDAS_DTYPES = {
    "fixture_id": "object", "bookmaker": "object", "outcome": "object", "observed_ns": "int64", "price": "float64",
    "active": "bool", "stake_limit": "float64", "kickoff_ns": "int64", "tournament_id": "int64",
}  # fmt: skip


def read_paths(path: Path, columns: list[str], books: Iterable[str] = BOOKS) -> pd.DataFrame:
    """One pulled chunk as pandas: the requested PATH_COLS-style columns of `books` rows, with observed_at as int64
    nanoseconds (`observed_ns`) and kickoff_utc as `kickoff_ns`. A zero-row chunk (even one written without columns)
    gives a typed empty frame."""
    import pyarrow.parquet as pq

    cols = list(dict.fromkeys(columns + ["kickoff_ns"]))
    if pq.ParquetFile(path).metadata.num_rows == 0:
        return pd.DataFrame({c: pd.Series(dtype=_PANDAS_DTYPES.get(c, "float64")) for c in cols})
    raw = [c for c in columns if c not in ("observed_ns", "kickoff_ns")]
    want = list(dict.fromkeys(raw + ["bookmaker", "observed_at", "kickoff_utc"]))
    t = pq.read_table(path, columns=want, filters=[("bookmaker", "in", list(books))])
    t = t.append_column("observed_ns", _to_ns(t["observed_at"])).append_column("kickoff_ns", _to_ns(t["kickoff_utc"]))
    t = t.drop_columns(["observed_at", "kickoff_utc"])
    df = t.to_pandas()
    for c in ("fixture_id", "bookmaker", "outcome"):
        if c in df:
            df[c] = df[c].astype(str).astype("object")
    return df[[c for c in cols if c in df.columns]]


def read_fixtures(out: Path = OUT) -> pd.DataFrame:
    """fixtures.parquet with kickoff_ns (int64 ns UTC), window and tournament (tournament_id as text) added."""
    import pyarrow.parquet as pq

    t = conform(pq.read_table(out / "fixtures.parquet"), FIXTURE_SCHEMA)
    t = t.append_column("kickoff_ns", _to_ns(t["kickoff_utc"])).drop_columns(["kickoff_utc"])
    f = t.to_pandas()
    f["fixture_id"] = f["fixture_id"].astype(str)
    f["window"] = window_of(f["kickoff_ns"].to_numpy(dtype=np.int64))
    f["tournament"] = tournament_table(f)["tournament"].to_numpy()
    return f


def chunk_stem(lo: str, hi: str) -> str:
    return f"paths_{lo}_{hi}"


def chunk_files(out: Path = OUT, window: str | None = None) -> list[Path]:
    """The pulled chunk files in kickoff order; with `window`, only the chunks of that registered window."""
    return [out / "paths" / f"{chunk_stem(lo, hi)}.parquet" for lo, hi in chunks(window)]


def combined_sha(shas: Iterable[str]) -> str:
    return hashlib.sha256("".join(shas).encode()).hexdigest()[:12]


def load_snapshot(out: Path = OUT) -> dict[str, Any]:
    """snapshot.json of a finished pull, verified against the files on disk.

    Raises SystemExit when the pull is missing or incomplete, mixes lake builds or was served without a build, has
    fixtures the lake has not read (fetch_status other than ok / no_history), or when any chunk, fixtures or
    fetch_status file no longer matches the sha recorded at pull time.
    """
    import json

    p, man = out / "snapshot.json", out / "manifest.json"
    if not p.exists() or not man.exists():
        raise SystemExit(f"no finished pull at {out} (run pull.py)")
    snap = json.loads(p.read_text())
    if not snap.get("complete"):
        raise SystemExit("the pull is incomplete: rerun pull.py")
    if not snap.get("consistent_lake_version"):
        raise SystemExit(f"the pull mixes lake builds ({snap.get('lake_version')}): rerun pull.py --restart")
    if snap.get("lake_version") in (None, "", "unversioned"):
        raise SystemExit("the pull has no lake build (no X-Lakehouse-Version): rerun pull.py --restart")
    if snap.get("fetch_status_unread", 1):
        raise SystemExit(f"{snap.get('fetch_status_unread')} fixtures in range are not read by the lake (fetch_status)")
    manifest = json.loads(man.read_text())["chunks"]
    stems = [chunk_stem(lo, hi) for lo, hi in chunks()]
    if set(manifest) != set(stems):
        raise SystemExit("the manifest does not list exactly the registered chunks: rerun pull.py --restart")
    for stem in stems:
        f = out / "paths" / f"{stem}.parquet"
        if not f.exists() or sobf.file_sha(f) != manifest[stem]["sha"]:
            raise SystemExit(f"{f.name} is missing or changed since the pull: rerun pull.py --restart")
    if combined_sha(manifest[k]["sha"] for k in stems) != snap["paths_sha"]:
        raise SystemExit("the chunk shas do not match snapshot.json: rerun pull.py --restart")
    for stem, meta in snap["files"].items():
        f = out / f"{stem}.parquet"
        if not f.exists() or sobf.file_sha(f) != meta["sha"]:
            raise SystemExit(f"{f.name} is missing or changed since the pull: rerun pull.py --restart")
    return dict(snap)


def snapshot_label(out: Path = OUT) -> str:
    """`data_snapshot` for the ledger: lake build, pull date, number of chunks and the combined chunk sha."""
    s = load_snapshot(out)
    f = s["files"]
    return (
        f"lake {s['lake_version']} pulled {s['pulled_at'][:10]} (oddspapi 1x2 pinnacle+betfair-ex, {s['n_chunks']} "
        f"chunks sha {s['paths_sha']}, fixtures sha {f['fixtures']['sha']})"
    )
