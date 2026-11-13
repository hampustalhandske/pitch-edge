"""Shared code for the R2 `bookmaker-line-structure` studies.

Loads the fixed fd_matches snapshot, de-vigs odds (multiplicative, power, Shin), runs match-day block
bootstraps from per-day sufficient statistics, and implements the grading statistics used by
research/STANDARDS.md (Benjamini-Hochberg, PSR/DSR, PBO via CSCV). Also appends rows to a study's
interim trials ledger (`trials.csv`) until the research harness exists.

Everything here is a pure function of its inputs; randomness comes only from an explicit seed.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "research" / "r2"
SNAPSHOT = DATA / "fd.parquet"
SNAPSHOT_AT = (DATA / "fd_snapshot_at.txt").read_text().strip() if (DATA / "fd_snapshot_at.txt").exists() else ""
B = 2000
SEED = 20261020
OUTCOMES = ("home", "draw", "away")


# --------------------------------------------------------------------------------------------- data


def load() -> pd.DataFrame:
    """Return the fd_matches snapshot with a datetime `match_date`, `main` (mmz4281 files) and outcome dummies."""
    d = pd.read_parquet(SNAPSHOT)
    d["match_date"] = pd.to_datetime(d["match_date"])
    d["main"] = ~d["source_file"].str.startswith("new/")
    for o, code in zip(OUTCOMES, ("H", "D", "A"), strict=True):
        d[f"y_{o}"] = (d["ft_result"] == code).astype(float)
    d.loc[d["ft_result"].isna(), [f"y_{o}" for o in OUTCOMES]] = np.nan
    goals = d["ft_home_goals"] + d["ft_away_goals"]
    d["y_over25"] = (goals > 2.5).astype(float)
    d.loc[goals.isna(), "y_over25"] = np.nan
    return d


def window(d: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    """Rows with lo <= match_date <= hi (inclusive ISO dates)."""
    return d[(d["match_date"] >= lo) & (d["match_date"] <= hi)].copy()


def assert_open_only(cols: list[str]) -> None:
    """Point-in-time guard: decision features may only come from pre-closing (`*_open`) columns."""
    bad = [c for c in cols if not c.endswith("_open")]
    if bad:
        raise AssertionError(f"decision uses non-open columns: {bad}")


def thirds(dates: pd.Series, lo: str, hi: str) -> pd.Series:
    """Label each date 1/2/3 by equal-length chronological thirds of [lo, hi]."""
    t0, t1 = pd.Timestamp(lo), pd.Timestamp(hi)
    span = (t1 - t0) / 3
    return pd.Series(np.where(dates < t0 + span, 1, np.where(dates < t0 + 2 * span, 2, 3)), index=dates.index)


# ------------------------------------------------------------------------------------------- de-vig


def _implied(odds: np.ndarray) -> np.ndarray:
    return 1.0 / np.asarray(odds, dtype=float)


def devig_multiplicative(odds: np.ndarray) -> np.ndarray:
    """Normalise implied probabilities to sum to 1. `odds` is (n, k) decimal odds; returns (n, k)."""
    q = _implied(odds)
    return q / q.sum(axis=1, keepdims=True)


def devig_power(odds: np.ndarray, iters: int = 80) -> np.ndarray:
    """Power method: p_i = q_i**k with k chosen so the p_i sum to 1 (bisection on k in [0.05, 20])."""
    q = _implied(odds)
    lo = np.full(q.shape[0], 0.05)
    hi = np.full(q.shape[0], 20.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        s = (q ** mid[:, None]).sum(axis=1)
        # sum decreases in k because every q < 1
        too_big = s > 1
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
    k = (lo + hi) / 2
    p = q ** k[:, None]
    return p / p.sum(axis=1, keepdims=True)


def devig_shin(odds: np.ndarray, iters: int = 80) -> np.ndarray:
    """Shin (1993) method: solve the insider share z in [0, 0.5) so the Shin probabilities sum to 1.

    p_i = (sqrt(z^2 + 4 (1 - z) q_i^2 / Q) - z) / (2 (1 - z)), Q = sum q_i. With Q <= 1 (no margin) z = 0
    and the result reduces to the multiplicative method.
    """
    q = _implied(odds)
    big_q = q.sum(axis=1, keepdims=True)

    def probs(z: np.ndarray) -> np.ndarray:
        z = z[:, None]
        return (np.sqrt(z**2 + 4 * (1 - z) * q**2 / big_q) - z) / (2 * (1 - z))

    lo = np.zeros(q.shape[0])
    hi = np.full(q.shape[0], 0.5)
    for _ in range(iters):
        mid = (lo + hi) / 2
        s = probs(mid).sum(axis=1)
        too_big = s > 1  # sum decreases in z
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
    p = probs((lo + hi) / 2)
    return p / p.sum(axis=1, keepdims=True)


DEVIG = {"multiplicative": devig_multiplicative, "power": devig_power, "shin": devig_shin}


def devig_cols(d: pd.DataFrame, cols: list[str], method: str = "power") -> np.ndarray:
    """De-vig the odds in `cols` (one row per match). Rows with any missing/invalid odd return NaN."""
    o = d[cols].to_numpy(dtype=float)
    ok = np.isfinite(o).all(axis=1) & (o > 1.0).all(axis=1)
    out = np.full(o.shape, np.nan)
    if ok.any():
        out[ok] = DEVIG[method](o[ok])
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------------------- bootstrap


def day_weights(n_days: int, b: int = B, seed: int = SEED) -> np.ndarray:
    """(b, n_days) multinomial resampling counts: each draw resamples match-days with replacement."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_days, np.full(n_days, 1.0 / n_days), size=b).astype(float)


def per_day(day: pd.Series, **cols: np.ndarray) -> tuple[pd.Index, dict[str, np.ndarray]]:
    """Sum each array in `cols` within match-day; returns the day index and a dict of per-day sums."""
    frame = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()})
    frame["_day"] = day.to_numpy()
    g = frame.groupby("_day", sort=True).sum()
    return g.index, {k: g[k].to_numpy() for k in cols}


def boot_mean(x: np.ndarray, day: pd.Series, seed: int = SEED) -> dict:
    """Mean of `x` with a match-day block-bootstrap 95% percentile CI, bootstrap SE and z-stat."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "se": np.nan, "z": np.nan, "days": 0}
    _, s = per_day(day, x=x, c=np.ones(n))
    w = day_weights(len(s["x"]), seed=seed)
    boots = (w @ s["x"]) / (w @ s["c"])
    m = x.mean()
    se = boots.std(ddof=1)
    return {
        "n": n,
        "mean": m,
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "se": float(se),
        "z": float(m / se) if se > 0 else np.nan,
        "days": len(s["x"]),
    }


def boot_ols(y: np.ndarray, xs: list[np.ndarray], day: pd.Series, seed: int = SEED) -> dict:
    """OLS of y on [1, *xs] with match-day block-bootstrap CIs for each coefficient (from per-day X'X, X'y)."""
    y = np.asarray(y, dtype=float)
    x = np.column_stack([np.ones(len(y))] + [np.asarray(v, dtype=float) for v in xs])
    k = x.shape[1]
    cols = {}
    for i in range(k):
        cols[f"xy{i}"] = x[:, i] * y
        for j in range(i, k):
            cols[f"xx{i}{j}"] = x[:, i] * x[:, j]
    _, s = per_day(day, **cols)
    w = day_weights(len(s["xy0"]), seed=seed)

    def assemble(sums: dict[str, np.ndarray]) -> np.ndarray:
        nb = sums["xy0"].shape[0]
        xtx = np.zeros((nb, k, k))
        xty = np.zeros((nb, k))
        for i in range(k):
            xty[:, i] = sums[f"xy{i}"]
            for j in range(i, k):
                xtx[:, i, j] = xtx[:, j, i] = sums[f"xx{i}{j}"]
        return np.linalg.solve(xtx, xty[..., None])[..., 0]

    point = assemble({kk: v.sum(keepdims=True) for kk, v in s.items()})[0]
    boots = assemble({kk: w @ v for kk, v in s.items()})
    se = boots.std(axis=0, ddof=1)
    return {
        "coef": point,
        "ci_lo": np.percentile(boots, 2.5, axis=0),
        "ci_hi": np.percentile(boots, 97.5, axis=0),
        "se": se,
        "n": len(y),
    }


def boot_oos_r2(y: np.ndarray, yhat: np.ndarray, day: pd.Series, seed: int = SEED) -> dict:
    """Out-of-sample R^2 = 1 - SSE(model)/SSE(zero forecast) with a match-day block-bootstrap CI.

    The comparison forecast is 0 (no move), which is what an efficient-open hypothesis predicts.
    """
    e1 = (np.asarray(y) - np.asarray(yhat)) ** 2
    e0 = np.asarray(y) ** 2
    _, s = per_day(day, e1=e1, e0=e0)
    w = day_weights(len(s["e1"]), seed=seed)
    boots = 1 - (w @ s["e1"]) / (w @ s["e0"])
    r2 = 1 - e1.sum() / e0.sum()
    se = boots.std(ddof=1)
    return {
        "n": len(e1),
        "mean": float(r2),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "se": float(se),
        "z": float(r2 / se) if se > 0 else np.nan,
    }


def p_value(z: float, direction: str) -> float:
    """p-value from a z-statistic (bootstrap SE); direction positive | negative | two-sided."""
    if not np.isfinite(z):
        return float("nan")
    if direction == "positive":
        return float(1 - stats.norm.cdf(z))
    if direction == "negative":
        return float(stats.norm.cdf(z))
    return float(2 * (1 - stats.norm.cdf(abs(z))))


# ------------------------------------------------------------------------------------------ grading


def bh(pvals: np.ndarray, q: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg: returns (adjusted q-values, reject mask at level q). NaN p-values are ignored."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    adj = np.full(p.shape, np.nan)
    m = ok.sum()
    if m:
        pv = p[ok]
        order = np.argsort(pv)
        ranked = pv[order] * m / np.arange(1, m + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(m)
        out[order] = np.minimum(ranked, 1.0)
        adj[ok] = out
    return adj, np.where(ok, adj < q, False)


def sharpe_moments(x: np.ndarray) -> dict:
    """Per-observation Sharpe ratio (mean/sd), skewness and (non-excess) kurtosis of a return series."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return {"sr": np.nan, "skew": np.nan, "kurt": np.nan}
    return {
        "sr": float(x.mean() / x.std(ddof=1)),
        "skew": float(stats.skew(x)),
        "kurt": float(stats.kurtosis(x, fisher=False)),
    }


def psr(sr: float, sr0: float, n: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe ratio (Bailey & Lopez de Prado 2012): P(true SR > sr0)."""
    den = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if n < 2 or den <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(n - 1) / math.sqrt(den)))


def dsr(sr: float, n: int, skew: float, kurt: float, n_trials: int, var_sr: float) -> float:
    """Deflated Sharpe ratio: PSR against the expected max SR of `n_trials` null trials with variance `var_sr`."""
    if n_trials < 2 or not np.isfinite(var_sr):
        return psr(sr, 0.0, n, skew, kurt)
    g = 0.5772156649
    sr0 = math.sqrt(var_sr) * (
        (1 - g) * stats.norm.ppf(1 - 1 / n_trials) + g * stats.norm.ppf(1 - 1 / (n_trials * math.e))
    )
    return psr(sr, sr0, n, skew, kurt)


def pbo_cscv(perf: np.ndarray, n_blocks: int = 16) -> float:
    """Probability of backtest overfitting via CSCV (Bailey et al. 2017).

    `perf` is (T, N): per-observation returns for N configurations over T time-ordered rows (rows with no bet
    hold 0). Rows are cut into `n_blocks` contiguous blocks; for every half/half split the in-sample best config's
    out-of-sample relative rank is converted to a logit; PBO is the share of logits <= 0.
    """
    t, n = perf.shape
    blocks = np.array_split(np.arange(t), n_blocks)
    block_sum = np.array([perf[b].sum(axis=0) for b in blocks])
    block_cnt = np.array([len(b) for b in blocks], dtype=float)
    logits = []
    for is_idx in itertools.combinations(range(n_blocks), n_blocks // 2):
        oos_idx = [i for i in range(n_blocks) if i not in is_idx]
        is_perf = block_sum[list(is_idx)].sum(axis=0) / block_cnt[list(is_idx)].sum()
        oos_perf = block_sum[oos_idx].sum(axis=0) / block_cnt[oos_idx].sum()
        best = int(np.argmax(is_perf))
        rank = stats.rankdata(oos_perf)[best]
        omega = rank / (n + 1)
        logits.append(math.log(omega / (1 - omega)))
    return float(np.mean(np.array(logits) <= 0))


# ------------------------------------------------------------------------------------------- ledger

LEDGER_FIELDS = [
    "id", "config_hash", "git_sha", "run_at", "n", "metric", "ci_lo", "ci_hi",
    "metric_name", "window", "label", "direction", "p_value", "sr", "skew", "kurt",
    "registration_sha256", "data_snapshot", "config_json",
]  # fmt: skip


def git_sha() -> str:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout
    return sha + ("-dirty" if dirty.strip() else "")


def registration_sha(reg_id: str) -> str:
    return hashlib.sha256((REPO / "research" / "registry" / f"{reg_id}.yaml").read_bytes()).hexdigest()


class Ledger:
    """Appends trial rows to research/studies/<study>/trials.csv (the interim ledger convention)."""

    def __init__(self, study_dir: Path, reg_id: str):
        # R2_DRY_LEDGER_DIR redirects rows to a scratch ledger for code checks (not counted as trials).
        dry = os.environ.get("R2_DRY_LEDGER_DIR")
        self.path = (Path(dry) / f"{reg_id}.csv") if dry else study_dir / "trials.csv"
        self.reg_id = reg_id
        self.reg_sha = registration_sha(reg_id)
        self.sha = git_sha()
        self.run_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def add(
        self,
        config: dict,
        res: dict,
        metric_name: str,
        window_name: str,
        label: str,
        direction: str,
        moments: dict | None = None,
    ) -> dict:
        cfg = json.dumps(config, sort_keys=True)
        row = {
            "id": self.reg_id,
            "config_hash": hashlib.sha256(cfg.encode()).hexdigest()[:16],
            "git_sha": self.sha,
            "run_at": self.run_at,
            "n": int(res.get("n", 0)),
            "metric": _fmt(res.get("mean")),
            "ci_lo": _fmt(res.get("ci_lo")),
            "ci_hi": _fmt(res.get("ci_hi")),
            "metric_name": metric_name,
            "window": window_name,
            "label": label,
            "direction": direction,
            "p_value": _fmt(p_value(res.get("z", np.nan), direction)),
            "sr": _fmt((moments or {}).get("sr")),
            "skew": _fmt((moments or {}).get("skew")),
            "kurt": _fmt((moments or {}).get("kurt")),
            "registration_sha256": self.reg_sha,
            "data_snapshot": f"fd.parquet@{SNAPSHOT_AT}",
            "config_json": cfg,
        }
        new = not self.path.exists()
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        return row


def _fmt(v: object) -> str:
    if v is None:
        return ""
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(v)
    return "" if not np.isfinite(f) else f"{f:.6g}"
