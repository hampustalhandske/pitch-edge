"""Audit helpers for R2-Q3, written independently of research/studies/r2-common.

Loads the raw odds_all pull (pull.py), de-vigs (power / multiplicative / Shin), applies the registered rule
and scores CLV with a match-day block bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lake import REPO

RAW = REPO / "data" / "research" / "r2" / "audit_q3_raw.parquet"
O = ("H", "D", "A")  # noqa: E741
EXPL = ("2012-07-01", "2019-06-30")
EVAL = ("2019-07-01", "2025-06-30")


def load() -> pd.DataFrame:
    d = pd.read_parquet(RAW)
    d["match_date"] = pd.to_datetime(d["match_date"])
    # one price source per era: Betbrain max before 2019/20, football-data 'Max' after
    for o in O:
        d[f"MX{o}"] = d[f"Max{o}"].where(d[f"Max{o}"].notna(), d[f"BbMx{o}"])
        d[f"Y{o}"] = (d["ft_result"] == o).astype(float)
    return d


def three(d: pd.DataFrame, pre: str, post: str = "") -> np.ndarray:
    """(n,3) array of columns f'{pre}{o}{post}' for o in H, D, A (NaN where missing)."""
    cols = [f"{pre}{o}{post}" for o in O]
    return np.column_stack([d[c].to_numpy(float) if c in d else np.full(len(d), np.nan) for c in cols])


def power(odds: np.ndarray) -> np.ndarray:
    q = 1 / odds
    lo, hi = np.full(len(q), 1e-3), np.full(len(q), 50.0)
    for _ in range(100):
        k = 0.5 * (lo + hi)
        s = np.nansum(q ** k[:, None], axis=1)
        lo = np.where(s > 1, k, lo)
        hi = np.where(s > 1, hi, k)
    p = q ** (0.5 * (lo + hi))[:, None]
    return p / p.sum(axis=1, keepdims=True)


def mult(odds: np.ndarray) -> np.ndarray:
    q = 1 / odds
    return q / q.sum(axis=1, keepdims=True)


def shin(odds: np.ndarray) -> np.ndarray:
    q = 1 / odds
    Q = q.sum(axis=1, keepdims=True)
    lo, hi = np.zeros(len(q)), np.full(len(q), 0.49)

    def pz(z):
        z = z[:, None]
        return (np.sqrt(z * z + 4 * (1 - z) * q * q / Q) - z) / (2 * (1 - z))

    for _ in range(100):
        z = 0.5 * (lo + hi)
        s = pz(z).sum(axis=1)
        lo = np.where(s > 1, z, lo)
        hi = np.where(s > 1, hi, z)
    p = pz(0.5 * (lo + hi))
    return p / p.sum(axis=1, keepdims=True)


DV = {"power": power, "mult": mult, "shin": shin}


def valid(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a).all(axis=1) & (a > 1).all(axis=1)


def rule(price: np.ndarray, ref_open: np.ndarray, theta: float, cap: float | None = 1.5, ref_method: str = "power"):
    """Registered rule. Returns (bet mask, chosen outcome idx, edge at open, price taken)."""
    p = DV[ref_method](ref_open)
    e = np.log(price * p)
    if cap is not None:
        e = np.where(price / ref_open > cap, -np.inf, e)
    j = np.argmax(e, axis=1)
    r = np.arange(len(e))
    return e[r, j] > theta, j, e[r, j], price[r, j]


def boot(x: np.ndarray, day: pd.Series, B: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    """Mean and match-day block-bootstrap 95% percentile CI."""
    x = np.asarray(x, float)
    g = pd.DataFrame({"x": x, "c": 1.0, "d": day.to_numpy()}).groupby("d")[["x", "c"]].sum()
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=B)
    b = (w @ g["x"].to_numpy()) / (w @ g["c"].to_numpy())
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def fmt(m, lo, hi, n=None) -> str:
    s = f"{100 * m:+.2f}% [{100 * lo:+.2f}, {100 * hi:+.2f}]"
    return s + (f" n={n}" if n is not None else "")
