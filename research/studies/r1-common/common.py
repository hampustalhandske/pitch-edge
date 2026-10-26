"""Shared building blocks for the R1 polymarket-efficiency studies.

Everything here reads the parquet snapshot written by `pull.py` (data/research/r1/) and never touches the
lake or the pitch-edge warehouse. Contents:

- `load_matches()`  one row per football-data match that has all three Polymarket 1X2 markets mapped, with a
                    verified UTC kickoff, the de-vigged Betfair Exchange / Pinnacle open and close, the
                    point-in-time `open_known_at`, the realised result, and the condition_id per outcome.
- `PriceBook`       hourly Polymarket bars per condition, merged from the Yes token and the complement of the
                    No token; `at(cond, ts)` returns the last price whose bar had closed by `ts`.
- statistics        power de-vig, match-day block bootstrap, one-sided bootstrap p-values, BH, PSR/DSR, PBO
                    via CSCV.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import optimize, stats

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "research" / "r1"
STUDY = Path(__file__).resolve().parent
REGISTRY = REPO / "research" / "registry"
LONDON = ZoneInfo("Europe/London")
SIDES = ("home", "draw", "away")
RESULT_SIDE = {"H": "home", "D": "draw", "A": "away"}
KICKOFF_TOLERANCE_H = 0.5
COLLECTION_HOUR_LONDON = 18  # football-data collects pre-closing odds "Friday/Tuesday afternoons"
MIN_HALF_SPREAD = 0.005  # half of Polymarket's 1-cent tick: the least a taker can pay over the last trade
FEE_RATE = 0.05
MAX_BF_BOOKSUM = 1.05

EXPLORATORY = ("2023-09-01", "2025-10-31")
EVALUATION = ("2025-11-01", "2026-10-19")


# ---------------------------------------------------------------- de-vig


def power_devig(odds: np.ndarray) -> np.ndarray:
    """De-vig rows of decimal odds (n, 3) with the power method: q_i = (1/o_i)^k with k s.t. sum q_i = 1.

    Rows with a missing or non-positive price return NaN.
    """
    odds = np.asarray(odds, dtype=float)
    out = np.full(odds.shape, np.nan)
    for i, row in enumerate(odds):
        if not np.all(np.isfinite(row)) or np.any(row <= 1.0):
            continue
        pi = 1.0 / row
        f = lambda k: np.sum(pi**k) - 1.0  # noqa: E731, B023 (used within this iteration)
        try:
            k = optimize.brentq(f, 0.2, 5.0)
        except ValueError:
            continue
        out[i] = pi**k
    return out


def power_normalise(p: np.ndarray) -> np.ndarray:
    """Power-normalise rows of raw probabilities (n, 3) so each row sums to 1 (same method as `power_devig`)."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore"):
        odds = np.where(p > 0, 1.0 / p, np.nan)
    return power_devig(odds)


def booksum(odds: pd.DataFrame, cols: list[str]) -> pd.Series:
    return (1.0 / odds[cols]).sum(axis=1, min_count=3)


# ---------------------------------------------------------------- kickoff / known-at


def open_known_at(ko_utc: pd.Timestamp) -> pd.Timestamp:
    """Latest moment football-data's pre-closing ("open") odds for this fixture can have been collected.

    football-data.co.uk notes.txt: "Betting odds for weekend games are collected Friday afternoons, and on
    Tuesday afternoons for midweek games." The conservative bound is therefore the latest Tuesday-or-Friday
    18:00 Europe/London strictly before kickoff.
    """
    ko_local = ko_utc.tz_convert(LONDON)
    day = ko_local.normalize()
    for back in range(0, 8):
        d = day - pd.Timedelta(days=back)
        if d.dayofweek in (1, 4):  # Tuesday, Friday
            cand = pd.Timestamp(datetime(d.year, d.month, d.day, COLLECTION_HOUR_LONDON), tz=LONDON)
            if cand < ko_local:
                return cand.tz_convert(UTC)
    raise AssertionError("unreachable")


def load_matches() -> tuple[pd.DataFrame, dict]:
    """Build the per-match frame and the drop log (each rule with how many matches it removed)."""
    mm = pd.read_parquet(DATA / "map_markets.parquet")
    fd = pd.read_parquet(DATA / "fd.parquet")
    drops: dict[str, int] = {"fd_rows_with_pm_1x2": int(fd.fd_match_key.nunique())}

    mm = mm[mm.market_type.isin(["win_home", "draw", "win_away"])]
    per_side = mm.groupby(["fd_match_key", "outcome_side"]).condition_id.nunique().unstack()
    complete = per_side.reindex(columns=list(SIDES)).eq(1).all(axis=1)
    keep_keys = per_side.index[complete]
    drops["drop_not_exactly_one_market_per_outcome"] = int((~complete).sum())
    cond = (
        mm[mm.fd_match_key.isin(keep_keys)]
        .drop_duplicates(["fd_match_key", "outcome_side"])
        .pivot(index="fd_match_key", columns="outcome_side", values="condition_id")
        .rename(columns=lambda s: f"cond_{s}")
    )
    pm_meta = mm.groupby("fd_match_key").agg(
        close_at=("close_at", "min"),
        close_at_max=("close_at", "max"),
        opens_at=("opens_at", "min"),
        pm_source=("pm_source", "first"),
        league_code=("league_code", "first"),
    )
    pm_res = mm.pivot_table(
        index="fd_match_key", columns="outcome_side", values="winning_outcome_label", aggfunc="first"
    ).rename(columns=lambda s: f"pm_win_{s}")

    df = fd.set_index("fd_match_key").join(cond, how="inner").join(pm_meta).join(pm_res)
    ko_local = pd.to_datetime(df.match_date.astype(str) + " " + df.kickoff_time.astype(str), errors="coerce")
    df["ko_fd"] = ko_local.dt.tz_localize(LONDON, ambiguous="NaT", nonexistent="NaT").dt.tz_convert(UTC)
    gap_h = (df.close_at - df.ko_fd).dt.total_seconds().abs() / 3600
    ok = df.ko_fd.notna() & (gap_h <= KICKOFF_TOLERANCE_H) & (df.close_at_max == df.close_at)
    drops["drop_kickoff_unverified"] = int((~ok).sum())
    df = df[ok].copy()
    df["ko"] = df.ko_fd
    df["open_known_at"] = df.ko.map(open_known_at)

    o = lambda b, w: [f"{b}_{s}_{w}" for s in SIDES]  # noqa: E731
    for book, tag in (("betfair_ex", "bf"), ("pinnacle", "pin")):
        for when in ("open", "close"):
            q = power_devig(df[o(book, when)].to_numpy())
            for j, s in enumerate(SIDES):
                df[f"q_{tag}_{when}_{s}"] = q[:, j]
            df[f"bs_{tag}_{when}"] = booksum(df, o(book, when))
            if tag == "bf":  # an exchange quote with booksum > 1.05 is an illiquid book, not a price
                bad = ~(df[f"bs_{tag}_{when}"] <= MAX_BF_BOOKSUM)
                for s in SIDES:
                    df.loc[bad, f"q_{tag}_{when}_{s}"] = np.nan
    # Sharp open: Pinnacle open where present, else Betfair Exchange open (same football-data collection).
    use_pin = df[[f"q_pin_open_{s}" for s in SIDES]].notna().all(axis=1)
    for s in SIDES:
        df[f"q_open_{s}"] = np.where(use_pin, df[f"q_pin_open_{s}"], df[f"q_bf_open_{s}"])
    df["open_src"] = np.where(use_pin, "pinnacle", np.where(df[f"q_bf_open_{SIDES[0]}"].notna(), "betfair", None))
    df["result_side"] = df.ft_result.map(RESULT_SIDE)
    df["match_day"] = df.match_date.astype(str)
    df["window"] = np.where(
        df.match_day <= EXPLORATORY[1], "exploratory", np.where(df.match_day <= EVALUATION[1], "evaluation", "after")
    )
    drops["matches_kept"] = len(df)
    return df.reset_index(), drops


# ---------------------------------------------------------------- Polymarket prices


class PriceBook:
    """Hourly Polymarket prices per condition, as known at a timestamp.

    Each Yes-token bar contributes (close, high, low); each No-token bar contributes the complement
    (1-close, 1-low, 1-high), since both tokens trade on one complementary book. A bar is known at
    hour_start + 1h. When both tokens printed in the same hour the Yes bar is used.
    """

    def __init__(self, bars: pd.DataFrame):
        lab = bars.outcome_label.str.lower()
        b = bars[lab.isin(["yes", "no"])].copy()  # a handful of bars carry team/draw labels: not a Yes/No token
        yes = b.outcome_label.str.lower().eq("yes")
        b["p"] = np.where(yes, b.close, 1 - b.close)
        b["hi"] = np.where(yes, b.high, 1 - b.low)
        b["lo"] = np.where(yes, b.low, 1 - b.high)
        b["is_yes"] = yes
        b["known_at"] = b.hour_start + pd.Timedelta(hours=1)
        b = b.sort_values(["condition_id", "known_at", "is_yes"]).drop_duplicates(
            ["condition_id", "known_at"], keep="last"
        )
        b["cumvol"] = b.groupby("condition_id").volume_usdc.cumsum()
        self.series: dict[str, tuple[np.ndarray, ...]] = {}
        for cid, g in b.groupby("condition_id", sort=False):
            self.series[cid] = (
                g.known_at.dt.tz_convert("UTC")
                .dt.tz_localize(None)
                .to_numpy()
                .astype("datetime64[ns]")
                .astype(np.int64),
                g.p.to_numpy(),
                g.hi.to_numpy(),
                g.lo.to_numpy(),
                g.cumvol.to_numpy(),
            )

    def at(self, cid: str, ts: pd.Timestamp) -> tuple[float, float, float, float, float]:
        """(price, half_range, staleness_hours, known_at_ns, cum_volume) of the last bar known by `ts`."""
        s = self.series.get(cid)
        nan = (np.nan, np.nan, np.nan, np.nan, np.nan)
        if s is None:
            return nan
        t = pd.Timestamp(ts).value
        i = np.searchsorted(s[0], t, side="right") - 1
        if i < 0:
            return nan
        assert s[0][i] <= t, "bar used before it was known"
        return (s[1][i], (s[2][i] - s[3][i]) / 2, (t - s[0][i]) / 3.6e12, s[0][i], s[4][i])


def pm_snapshot(df: pd.DataFrame, book: PriceBook, offset_h: float | None, col: str = "ko") -> pd.DataFrame:
    """Polymarket prices for every match at `ko - offset_h` hours (or at `col` itself when offset_h is None)."""
    rows = []
    for r in df.itertuples(index=False):
        ts = getattr(r, col) if offset_h is None else r.ko - pd.Timedelta(hours=offset_h)
        rec = {"fd_match_key": r.fd_match_key, "ts": ts}
        for s in SIDES:
            p, hs, st, kn, cv = book.at(getattr(r, f"cond_{s}"), ts)
            if np.isfinite(kn):
                assert kn <= pd.Timestamp(ts).value
            rec.update({f"p_{s}": p, f"hs_{s}": hs, f"stale_{s}": st, f"vol_{s}": cv})
        rows.append(rec)
    return pd.DataFrame(rows)


def pm_probs(
    df: pd.DataFrame, book: PriceBook, h: float, max_stale: float | None, sum_lo: float, sum_hi: float, method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Polymarket 3-way probabilities at kickoff - h and the mask of matches that pass the snapshot rules.

    Rules: all three outcomes have a last trade known by the decision time, the raw prices sum to within
    [sum_lo, sum_hi], and (if `max_stale` is set) the oldest of the three last trades is at most `max_stale`
    hours old. `method` is "power" (power-normalised to sum to 1), "mult" (divided by the sum) or "raw".
    """
    snap = pm_snapshot(df, book, h)
    p = snap[[f"p_{s}" for s in SIDES]].to_numpy()
    st = snap[[f"stale_{s}" for s in SIDES]].max(axis=1).to_numpy()
    ok = np.isfinite(p).all(axis=1)
    raw_sum = p.sum(axis=1)
    ok &= (raw_sum >= sum_lo) & (raw_sum <= sum_hi)
    if max_stale is not None:
        ok &= st <= max_stale
    if method == "power":
        pn = power_normalise(p)
    elif method == "mult":
        pn = p / raw_sum[:, None]
    else:  # "raw": last-trade prices as they are
        pn = p.copy()
    ok &= np.isfinite(pn).all(axis=1)
    return pn, ok


# ---------------------------------------------------------------- statistics


def block_bootstrap(
    values: np.ndarray, blocks: np.ndarray, n_boot: int = 5000, seed: int = 20261020, stat: str = "mean"
) -> np.ndarray:
    """Bootstrap distribution of the mean of `values`, resampling whole blocks (match-days) with replacement."""
    values = np.asarray(values, dtype=float)
    codes, uniq = pd.factorize(pd.Series(blocks))
    sums = np.bincount(codes, weights=values, minlength=len(uniq))
    cnts = np.bincount(codes, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    nb = len(uniq)
    w = np.zeros((n_boot, nb))
    for b in range(n_boot):
        w[b] = np.bincount(rng.integers(0, nb, nb), minlength=nb)
    return (w @ sums) / (w @ cnts)


def block_bootstrap_slope(
    x: np.ndarray, y: np.ndarray, blocks: np.ndarray, n_boot: int = 5000, seed: int = 20261020
) -> tuple[float, np.ndarray]:
    """OLS slope of y on x and its match-day block-bootstrap distribution."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    codes, uniq = pd.factorize(pd.Series(blocks))
    nb = len(uniq)
    agg = np.vstack([np.bincount(codes, weights=v, minlength=nb) for v in (np.ones_like(x), x, y, x * x, x * y)])

    def slope(a: np.ndarray) -> np.ndarray:
        n, sx, sy, sxx, sxy = a
        return (n * sxy - sx * sy) / (n * sxx - sx * sx)

    point = float(slope(agg.sum(axis=1)))
    rng = np.random.default_rng(seed)
    w = np.zeros((n_boot, nb))
    for b in range(n_boot):
        w[b] = np.bincount(rng.integers(0, nb, nb), minlength=nb)
    return point, slope(agg @ w.T)


def one_sided_p(boot: np.ndarray, point: float, direction: str) -> float:
    """Bootstrap p-value for H0: parameter = 0, using the bootstrap distribution shifted to be centred on 0."""
    centred = boot - np.mean(boot)
    n = len(boot)
    if direction == "positive":
        return (1 + np.sum(centred >= point)) / (n + 1)
    if direction == "negative":
        return (1 + np.sum(centred <= point)) / (n + 1)
    return min(1.0, 2 * (1 + np.sum(np.abs(centred) >= abs(point))) / (n + 1))


def bh(pvals: np.ndarray, q: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg: returns (adjusted p-values, reject mask at level q)."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    run = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        run = min(run, p[i] * n / rank)
        adj[i] = run
    return adj, adj <= q


def psr(sr: float, n: int, skew: float, kurt: float, sr0: float) -> float:
    """Probabilistic Sharpe ratio P(true SR > sr0) for a per-observation Sharpe `sr` over n observations."""
    den = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr**2))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(n - 1) / den))


def dsr_threshold(sr_trials: np.ndarray) -> float:
    """Expected maximum Sharpe under the null across the trials (Bailey & Lopez de Prado), per observation."""
    sr_trials = np.asarray([s for s in sr_trials if np.isfinite(s)])
    n = len(sr_trials)
    if n < 2:
        return 0.0
    g = 0.5772156649
    v = np.var(sr_trials, ddof=1)
    return float(np.sqrt(v) * ((1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * np.e))))


def pbo_cscv(perf: np.ndarray, n_blocks: int = 16) -> float:
    """Probability of backtest overfitting via CSCV on a (n_blocks, n_configs) matrix of block performance."""
    perf = np.asarray(perf, float)
    s = perf.shape[0]
    assert s == n_blocks
    logits = []
    for is_idx in itertools.combinations(range(s), s // 2):
        oos_idx = [i for i in range(s) if i not in is_idx]
        is_perf = np.nanmean(perf[list(is_idx)], axis=0)
        oos_perf = np.nanmean(perf[oos_idx], axis=0)
        if np.all(np.isnan(is_perf)):
            continue
        best = int(np.nanargmax(is_perf))
        valid = ~np.isnan(oos_perf)
        if not valid[best]:
            continue
        rank = stats.rankdata(oos_perf[valid])[np.where(np.where(valid)[0] == best)[0][0]]
        omega = rank / (valid.sum() + 1)
        logits.append(np.log(omega / (1 - omega)))
    return float(np.mean(np.array(logits) <= 0)) if logits else float("nan")


# ---------------------------------------------------------------- ledger


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_hash(reg_id: str, cfg: dict) -> str:
    return hashlib.sha256(json.dumps({"id": reg_id, **cfg}, sort_keys=True, default=str).encode()).hexdigest()[:12]


LEDGER_COLS = [
    "id",
    "config_hash",
    "config",
    "registration_sha256",
    "git_sha",
    "data_snapshot",
    "run_at",
    "window",
    "n",
    "n_days",
    "metric",
    "value",
    "ci_lo",
    "ci_hi",
    "p_one_sided",
    "direction",
]


def append_trials(study_dir: Path, rows: list[dict]) -> None:
    """Append trial rows to `<study_dir>/trials.csv` (append-only; header written on first use)."""
    path = study_dir / "trials.csv"
    new = pd.DataFrame(rows)[LEDGER_COLS]
    new.to_csv(path, mode="a", header=not path.exists(), index=False)


def snapshot_tag() -> str:
    snap = json.loads((DATA / "snapshot.json").read_text())
    return f"{snap['pulled_at']} map={snap['map_markets']} fd={snap['fd']} bars={snap['bars']}"
