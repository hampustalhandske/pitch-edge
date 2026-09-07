"""Cross-venue lead-lag: does one venue's price move systematically *before* another's?

The steam detector (`odds/steam.py`) flags *that* two venues disagree. This module asks the
quant-desk question: on our own snapshot history, does venue A's no-vig probability for the same
fixture/outcome lead venue B's by a stable number of steps? Two tests per (fixture, outcome):

1. lagged cross-correlation of first differences (the sign and location of the peak), and
2. a Granger-style nested-OLS F-test in both directions (does adding lags of A improve the
   forecast of ΔB beyond B's own lags, and vice versa).

Nothing here is a trading rule. The output is a table with n_obs, the best lag, the peak
correlation and both p-values, plus an aggregate row that says whether the evidence is
"insufficient" (too few aligned snapshots — the honest state of a repo that started snapshotting
hours ago), "no lead", or a lead with its direction. Insufficient data is reported, never filled in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from pitch_edge.data.teams import TeamNameResolver, normalise

_VS_RE = re.compile(r"(.+?)\s+(?:vs\.?|v\.?|versus|@|-)\s+(.+)", re.I)
_DATE_RE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b|\b(\d{4}-\d{2}-\d{2})\b",
    re.I,
)
_TIE_WORDS = {"tie", "draw", "empate"}


@dataclass(frozen=True)
class LeadLagResult:
    fixture_key: str
    outcome: str
    venue_a: str
    venue_b: str
    n_obs: int
    best_lag: int  # >0: A leads B by this many steps; <0: B leads A
    peak_xcorr: float
    p_a_leads_b: float
    p_b_leads_a: float
    verdict: str  # "insufficient" | "no_lead" | "a_leads" | "b_leads" | "bidirectional"


# --------------------------------------------------------------- fixture keys
def fixture_key_from_question(question: str | None, teams: list[str] | None = None) -> tuple[str, str | None]:
    """('teamA|teamB|YYYY-MM-DD', outcome_hint) parsed from a market question. Team order follows the text.
    Kalshi: 'Kalmar wins — If Kalmar wins the Kalmar vs Djurgarden ... scheduled for Sep 7, 2026 ...'.
    Polymarket: 'Arsenal vs Chelsea' / 'Will Arsenal beat Chelsea on 2026-09-07?'."""
    if not question:
        return "", None
    q = str(question)
    date = None
    dm = _DATE_RE.search(q)
    if dm:
        try:
            date = pd.Timestamp(dm.group(4) or f"{dm.group(1)} {dm.group(2)} {dm.group(3)}").strftime("%Y-%m-%d")
        except ValueError:
            date = None
    body = re.split(r"\s+—\s+|\s+-\s+If\s+|\bIf\b", q, maxsplit=1)[-1]
    m = _VS_RE.search(body)
    if not m:
        return "", None
    a = re.sub(r"^(the|if)\s+", "", m.group(1).strip(), flags=re.I)
    a = a.split(" wins the ")[-1].split(" the ")[-1] if " the " in a else a
    b = re.split(
        r"[:?,;]|\s+(?:professional|game|match|soccer|football|on\s|\(|scheduled|originally|winner|to\s+win|moneyline|spread|total|-\s)",
        m.group(2).strip(),
        maxsplit=1,
    )[0]
    a, b = normalise(a), normalise(b).strip(" ?.")
    return f"{a}|{b}|{date or ''}", None


def canonical_team_key(fixture_key: str, resolver: TeamNameResolver) -> str:
    """'kalmar|djurgarden' -> the same key with each side resolved to the spine's canonical name when possible,
    so a Kalshi 'Nottingham Forest' and the spine's "Nott'm Forest" compare equal."""
    if not fixture_key:
        return ""
    parts = fixture_key.split("|")
    sides = [normalise(resolver.resolve(pt, cutoff=0.8) or pt) for pt in parts[:2]]
    return "|".join(sides)


def snapshots_for_fixture(long: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Rows of an aligned snapshot frame (from `align_snapshots`) that belong to home vs away, matched on
    canonical team names (either order is tolerated only if the venue lists the away side first)."""
    if long.empty:
        return long
    resolver = TeamNameResolver([home, away])
    want = f"{normalise(home)}|{normalise(away)}"
    keys = long["fixture_key"].astype(str).map(lambda k: canonical_team_key(k, resolver))
    return long[keys == want]


def outcome_from_row(question: str | None, outcome: str | None, fixture_key: str) -> str | None:
    """Map a venue's outcome label onto home/draw/away using the fixture key's team order."""
    if not fixture_key:
        return None
    a, b, _ = fixture_key.split("|")
    o = normalise(str(outcome or ""))
    if o in _TIE_WORDS or "tie" in o or "draw" in o:
        return "draw"
    if o and (o == a or o in a or a in o):
        return "home"
    if o and (o == b or o in b or b in o):
        return "away"
    q = normalise(str(question or ""))
    if q.startswith(a) and " wins" in q:
        return "home"
    if q.startswith(b) and " wins" in q:
        return "away"
    return None


def align_snapshots(snapshots: pd.DataFrame, freq: str = "60min") -> pd.DataFrame:
    """market_snapshots rows -> long frame (fixture_key, outcome, venue, ts, prob) resampled on a common grid.
    Probabilities are renormalised per (venue, fixture, ts) when all three outcomes are quoted."""
    if snapshots.empty:
        return pd.DataFrame(columns=["fixture_key", "outcome", "venue", "ts", "prob"])
    df = snapshots.copy()
    df["venue"] = df.get("venue", pd.Series(index=df.index, dtype=object)).fillna("polymarket")
    keys = df["question"].map(fixture_key_from_question)
    df["fixture_key"] = [k for k, _ in keys]
    df["side"] = [
        outcome_from_row(q, o, k) for q, o, k in zip(df["question"], df["outcome"], df["fixture_key"], strict=True)
    ]
    df = df[(df["fixture_key"] != "") & df["side"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["fixture_key", "outcome", "venue", "ts", "prob"])
    df["ts"] = pd.to_datetime(df["snapshot_ts"]).dt.floor(freq)
    g = df.groupby(["fixture_key", "venue", "ts", "side"], as_index=False)["probability"].mean()
    tot = g.groupby(["fixture_key", "venue", "ts"])["probability"].transform("sum")
    n = g.groupby(["fixture_key", "venue", "ts"])["probability"].transform("size")
    g["prob"] = np.where(n == 3, g["probability"] / tot, g["probability"])
    return g.rename(columns={"side": "outcome"})[["fixture_key", "outcome", "venue", "ts", "prob"]]


# ------------------------------------------------------------------ the tests
def _lagged_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Peak corr(Δa_{t-k}, Δb_t) over k in [-max_lag, max_lag]; k>0 means a leads b."""
    best_k, best_c = 0, 0.0
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            x, y = a[: len(a) - k] if k else a, b[k:]
        else:
            x, y = a[-k:], b[: len(b) + k]
        if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if abs(c) > abs(best_c):
            best_k, best_c = k, c
    return best_k, best_c


def granger_p_value(y: np.ndarray, x: np.ndarray, lag: int) -> float:
    """p-value of H0 'x does not Granger-cause y' via nested OLS on `lag` own lags vs own + x lags."""
    n = len(y)
    if n <= 3 * lag + 2:
        return float("nan")
    rows_y = np.column_stack([y[lag - i - 1 : n - i - 1] for i in range(lag)])
    rows_x = np.column_stack([x[lag - i - 1 : n - i - 1] for i in range(lag)])
    target = y[lag:]
    ones = np.ones((len(target), 1))
    Xr = np.hstack([ones, rows_y])
    Xu = np.hstack([ones, rows_y, rows_x])
    rss_r = float(np.sum((target - Xr @ np.linalg.lstsq(Xr, target, rcond=None)[0]) ** 2))
    rss_u = float(np.sum((target - Xu @ np.linalg.lstsq(Xu, target, rcond=None)[0]) ** 2))
    df1, df2 = lag, len(target) - Xu.shape[1]
    if df2 <= 0 or rss_u <= 0:
        return float("nan")
    f = ((rss_r - rss_u) / df1) / (rss_u / df2)
    return float(stats.f.sf(f, df1, df2))


def lead_lag_test(a: pd.Series, b: pd.Series, max_lag: int = 3, min_obs: int = 24, alpha: float = 0.05) -> dict:
    """a, b: probability series on the same time index. Returns the LeadLagResult fields except ids."""
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).ffill().dropna()
    da, db = np.diff(joined["a"].to_numpy()), np.diff(joined["b"].to_numpy())
    n = len(da)
    if n < min_obs or np.std(da) == 0 or np.std(db) == 0:
        return {
            "n_obs": int(n),
            "best_lag": 0,
            "peak_xcorr": float("nan"),
            "p_a_leads_b": float("nan"),
            "p_b_leads_a": float("nan"),
            "verdict": "insufficient",
        }
    lag, c = _lagged_xcorr(da, db, max_lag)
    p_ab = granger_p_value(db, da, max_lag)
    p_ba = granger_p_value(da, db, max_lag)
    if np.isnan(p_ab) or np.isnan(p_ba):
        verdict = "insufficient"
    elif p_ab < alpha and p_ba < alpha:
        verdict = "bidirectional"
    elif p_ab < alpha:
        verdict = "a_leads"
    elif p_ba < alpha:
        verdict = "b_leads"
    else:
        verdict = "no_lead"
    return {
        "n_obs": int(n),
        "best_lag": int(lag),
        "peak_xcorr": c,
        "p_a_leads_b": p_ab,
        "p_b_leads_a": p_ba,
        "verdict": verdict,
    }


def cross_venue_lead_lag(
    snapshots: pd.DataFrame,
    venue_a: str = "polymarket",
    venue_b: str = "kalshi",
    freq: str = "60min",
    max_lag: int = 3,
    min_obs: int = 24,
) -> pd.DataFrame:
    """One row per (fixture, outcome) quoted on both venues, plus an 'ALL' aggregate row."""
    cols = [
        "fixture_key",
        "outcome",
        "venue_a",
        "venue_b",
        "n_obs",
        "best_lag",
        "peak_xcorr",
        "p_a_leads_b",
        "p_b_leads_a",
        "verdict",
    ]
    long = align_snapshots(snapshots, freq=freq)
    rows: list[LeadLagResult] = []
    if not long.empty:
        for (fk, oc), g in long.groupby(["fixture_key", "outcome"]):
            wide = g.pivot_table(index="ts", columns="venue", values="prob", aggfunc="mean")
            if venue_a not in wide or venue_b not in wide:
                continue
            r = lead_lag_test(wide[venue_a], wide[venue_b], max_lag=max_lag, min_obs=min_obs)
            rows.append(LeadLagResult(fk, oc, venue_a, venue_b, **r))
    table = pd.DataFrame([r.__dict__ for r in rows], columns=cols)
    tested = table[table["verdict"] != "insufficient"]
    n_pairs = int(len(table))
    if tested.empty:
        max_obs = int(table["n_obs"].max()) if n_pairs else 0
        agg_verdict = f"insufficient ({n_pairs} shared fixture-outcomes, max {max_obs} aligned steps, need {min_obs})"
        agg = {
            "n_obs": max_obs,
            "best_lag": 0,
            "peak_xcorr": float("nan"),
            "p_a_leads_b": float("nan"),
            "p_b_leads_a": float("nan"),
        }
    else:
        share_a = float((tested["verdict"] == "a_leads").mean())
        share_b = float((tested["verdict"] == "b_leads").mean())
        agg = {
            "n_obs": int(tested["n_obs"].sum()),
            "best_lag": int(tested["best_lag"].median()),
            "peak_xcorr": float(tested["peak_xcorr"].median()),
            "p_a_leads_b": float(tested["p_a_leads_b"].median()),
            "p_b_leads_a": float(tested["p_b_leads_a"].median()),
        }
        agg_verdict = (
            f"{venue_a} leads in {share_a:.0%}, {venue_b} leads in {share_b:.0%} of {len(tested)} tested pairs"
        )
    table = pd.concat(
        [
            table,
            pd.DataFrame(
                [
                    {
                        "fixture_key": "ALL",
                        "outcome": "ALL",
                        "venue_a": venue_a,
                        "venue_b": venue_b,
                        **agg,
                        "verdict": agg_verdict,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return table[cols]
