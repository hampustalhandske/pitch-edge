"""Auditor's own helpers for the R6 audit (closing-line-prediction cross-book / book-informativeness).

Independent of the study: nothing from research/studies/r6-cross-book-close or pitch_edge is imported.
Reads only the raw football-data per-book parquet `data/research/r2/audit_q3_raw.parquet`.

Provides: raw loading, an own power de-vig (Newton on the exponent), the H1 universe and feature
matrix built from the registration text, OLS / ridge walk-forward, KL scoring and a match-day block bootstrap.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data" / "research" / "r2" / "audit_q3_raw.parquet"
OUT = REPO / "data" / "research" / "r6_audit"
OUT.mkdir(parents=True, exist_ok=True)
CORE = ("B365", "BW", "IW", "WH", "VC")
LATE = ("1XB", "BF", "BFE")
HDA = ("H", "D", "A")
EVAL_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)


def odds3(d: pd.DataFrame, prefix: str) -> np.ndarray:
    """(n, 3) decimal odds for columns prefix+H/D/A; a row with any missing or <= 1 quote is all-NaN."""
    cols = [prefix + s for s in HDA]
    if not all(c in d.columns for c in cols):
        return np.full((len(d), 3), np.nan)
    x = d[cols].to_numpy(float, copy=True)
    bad = ~(np.isfinite(x).all(1) & (x > 1).all(1))
    x[bad] = np.nan
    return x


def power_devig(o: np.ndarray) -> np.ndarray:
    """Power de-vig: p_i = (1/o_i)^k with k solved by Newton so that sum p = 1. NaN rows stay NaN."""
    q = 1.0 / o
    out = np.full_like(q, np.nan)
    ok = np.isfinite(q).all(1)
    if not ok.any():
        return out
    qq = q[ok]
    lq = np.log(qq)
    k = np.ones(len(qq))
    for _ in range(100):
        pk = qq ** k[:, None]
        f = pk.sum(1) - 1.0
        fp = (pk * lq).sum(1)
        step = f / fp
        k = k - step
        if np.nanmax(np.abs(step)) < 1e-14:
            break
    out[ok] = qq ** k[:, None]
    return out


def lg(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p) - np.log1p(-p)


def raw() -> pd.DataFrame:
    d = pd.read_parquet(RAW)
    d["match_date"] = pd.to_datetime(d["match_date"])
    d["ss"] = np.where(d.match_date.dt.month >= 7, d.match_date.dt.year, d.match_date.dt.year - 1)
    return d


def universe(d: pd.DataFrame) -> pd.DataFrame:
    """H1 Pinnacle-segment universe: 2012-07-01..2025-06-30, Pinnacle open+close triple, Avg and Max open
    (Betbrain BbAv/BbMx before 2019/20, Avg/Max after), a result. Duplicate fixtures keep the first row."""
    d = d[(d.match_date >= "2012-07-01") & (d.match_date <= "2025-06-30")].copy()
    for s in HDA:
        d["av" + s] = d["BbAv" + s].fillna(d["Avg" + s])
        d["mx" + s] = d["BbMx" + s].fillna(d["Max" + s])
    keep = (
        np.isfinite(odds3(d, "PS")).all(1)
        & np.isfinite(odds3(d, "PSC")).all(1)
        & np.isfinite(odds3(d, "av")).all(1)
        & np.isfinite(odds3(d, "mx")).all(1)
        & d.ft_result.isin(["H", "D", "A"]).to_numpy()
    )
    d = d[keep].copy()
    d["mid"] = d.division + "|" + d.match_date.dt.strftime("%Y-%m-%d") + "|" + d.home_team + "|" + d.away_team
    d = d.drop_duplicates("mid").sort_values(["match_date", "mid"]).reset_index(drop=True)
    return d


def core_frame(d: pd.DataFrame) -> dict:
    """Per-match arrays: de-vigged Pinnacle open/close, move, g_avg, g_max, each book's raw logit deviation."""
    ps = odds3(d, "PS")
    po, pc = power_devig(ps), power_devig(odds3(d, "PSC"))
    lpo = lg(po)
    r = {"po": po, "pc": pc, "move": lg(pc) - lpo, "lpo": lpo, "ps": ps}
    r["g_avg"] = lg(power_devig(odds3(d, "av"))) - lpo
    mx = odds3(d, "mx")
    r["g_max"] = np.minimum(np.log(mx / ps), np.log(1.5))
    r["max_sum"] = (1 / mx).sum(1)
    r["pin_over"] = (1 / ps).sum(1) - 1
    for b in CORE + LATE:
        o = odds3(d, b)
        r["dev_" + b] = lg(power_devig(o)) - lpo
        r["lp_" + b] = lg(power_devig(o))
        r["sum_" + b] = (1 / o).sum(1)
        oc = odds3(d, b + "C")
        r["selfmv_" + b] = lg(power_devig(oc)) - lg(power_devig(o))
    return r


def design(r: dict, k: int, books=CORE) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Outcome-k (0 home, 1 draw, 2 away) design: baseline block [g_avg, g_max] and the cross-book block
    (clipped/imputed devs, miss flags, raw overrounds (centred later), Pinnacle overround, max booksum, dispersion)."""
    devs = np.stack([np.clip(r["dev_" + b][:, k], -0.5, 0.5) for b in books])  # (B, n)
    pres = np.isfinite(devs)
    cnt = pres.sum(0)
    mean_p = np.where(cnt > 0, np.where(pres, devs, 0).sum(0) / np.maximum(cnt, 1), np.nan)
    fill = np.where(np.isfinite(mean_p), mean_p, np.clip(r["g_avg"][:, k], -0.5, 0.5))
    cols, names = [], []
    for i, b in enumerate(books):
        cols.append(np.where(pres[i], devs[i], fill))
        names.append("dev_" + b)
    for b in books:
        cols.append((~np.isfinite(r["sum_" + b])).astype(float))
        names.append("miss_" + b)
    for b in books:
        cols.append(r["sum_" + b] - 1)
        names.append("over_" + b)
    lps = np.stack([r["lpo"][:, k]] + [r["lp_" + b][:, k] for b in books])
    m = np.isfinite(lps)
    mu = np.where(m, lps, 0).sum(0) / m.sum(0)
    disp = np.sqrt(np.where(m, (lps - mu) ** 2, 0).sum(0) / m.sum(0))
    cols += [r["pin_over"], r["max_sum"], disp]
    names += ["over_PS", "max_booksum", "dispersion"]
    base = np.column_stack([r["g_avg"][:, k], r["g_max"][:, k]])
    return base, np.column_stack(cols), names


def centre_over(xtr: np.ndarray, xte: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xtr, xte = xtr.copy(), xte.copy()
    for j, nm in enumerate(names):
        if nm.startswith("over_") and nm != "over_PS":
            mu = np.nanmean(xtr[:, j])
            xtr[:, j] = np.where(np.isfinite(xtr[:, j]), xtr[:, j] - mu, 0)
            xte[:, j] = np.where(np.isfinite(xte[:, j]), xte[:, j] - mu, 0)
    return xtr, xte


def ols(xtr, ytr, xte):
    a = np.column_stack([np.ones(len(xtr)), xtr])
    b, *_ = np.linalg.lstsq(a, ytr, rcond=None)
    return np.column_stack([np.ones(len(xte)), xte]) @ b


def ridge(xtr, ytr, xte, seas, lams=(1.0, 10.0, 100.0, 1e3, 1e4, 1e5)):
    """Standardised ridge, unpenalised intercept; lambda by MSE on the last training season (fit on earlier)."""
    mu, sd = xtr.mean(0), xtr.std(0)
    sd[sd == 0] = 1
    z, zt = (xtr - mu) / sd, (xte - mu) / sd

    def fit(zz, yy, lam):
        c = yy.mean()
        return c, np.linalg.solve(zz.T @ zz + lam * np.eye(zz.shape[1]), zz.T @ (yy - c))

    last = seas.max()
    a, v = seas < last, seas == last
    best = (np.inf, lams[0])
    for lam in lams:
        c, b = fit(z[a], ytr[a], lam)
        e = np.mean((ytr[v] - c - z[v] @ b) ** 2)
        if e < best[0] - 1e-15:
            best = (e, lam)
    c, b = fit(z, ytr, best[1])
    return c + zt @ b, best[1]


def phat(po, mv):
    q = 1 / (1 + np.exp(-(lg(po) + mv)))
    return q / q.sum(1, keepdims=True)


def kl(p, q):
    p, q = np.clip(p, 1e-12, 1), np.clip(q, 1e-12, 1)
    return (p * np.log2(p / q)).sum(1)


def boot(v: np.ndarray, blocks: np.ndarray, n: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    """Match-day block bootstrap of the mean (resample days with replacement; ratio of sums)."""
    codes, _ = pd.factorize(pd.Series(blocks))
    s = np.bincount(codes, weights=v)
    c = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    nb = len(s)
    w = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(n)]).astype(float)
    est = (w @ s) / (w @ c)
    return float(v.mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def fmt(t) -> str:
    return f"{t[0]:+.3e} [{t[1]:+.3e}, {t[2]:+.3e}]"
