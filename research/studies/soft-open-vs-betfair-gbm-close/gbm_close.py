"""H3 `soft-open-vs-betfair-gbm-close-v1`: a GBM of the sharp close built from soft-book opens only.

Features (28), all from the football-data pre-closing snapshot: avg, max and bet365 open odds (9), the logits of
their power de-vigged probabilities (9), spreads max-avg and bet365-avg in decimal odds (6), the three booksum
overrounds (3) and an era flag (match_date >= 2019-07-01). No Pinnacle or Betfair column and no close column is a
feature: the train file holds Pinnacle closes (target only), the evaluation file holds Betfair closes (benchmark only).

Model: one HistGradientBoostingRegressor per outcome on the logit of the power-de-vigged Pinnacle close, fixed
hyperparameters (the cross-book "tiny" setting). q_hat = sigmoid(predicted logits) renormalised to sum 1. Bet rule:
edge_o = log(max_o_open * q_hat[o]); take the outcome with the largest edge among those with max_o/avg_o <= 1.5; bet
if edge > theta. Theta is chosen on out-of-fold walk-forward predictions of the exploratory window; see `select`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "soft-open-vs-betfair-common"))
import sobf  # noqa: E402

REG_ID = "soft-open-vs-betfair-gbm-close-v1"
STUDY = "soft-open-vs-betfair-gbm-close"
OUT = sobf.REPO / "data" / "research" / STUDY
HERE = Path(__file__).resolve().parent

TRAIN_WINDOW = ("2012-07-01", "2024-06-30")
THETAS = (0.01, 0.02, 0.03)
MIN_OOF_BETS = 1000
FALLBACK_THETA = 0.02
CAP = 1.5
ERA_START = pd.Timestamp("2019-07-01")
SEED = 20261028
HGB_PARAMS: dict[str, Any] = {
    "max_iter": 150,
    "learning_rate": 0.03,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 1000,
    "l2_regularization": 10.0,
}

BOOKS = ("avg", "max", "bet365")
SOURCE_COLS = [f"{b}_{o}_open" for b in BOOKS for o in sobf.OUTCOMES]
ID_COLS = ["division", "match_date", "home_team", "away_team"]
TRAIN_TARGET_COLS = [f"pinnacle_{o}_close" for o in sobf.OUTCOMES]
EVAL_BENCH_COLS = [f"betfair_ex_{o}_close" for o in sobf.OUTCOMES]
FEATURE_NAMES = (
    list(SOURCE_COLS)
    + [f"logit_{b}_{o}" for b in BOOKS for o in sobf.OUTCOMES]
    + [f"spread_{a}_{o}" for a in ("max", "bet365") for o in sobf.OUTCOMES]
    + [f"booksum_{b}" for b in BOOKS]
    + ["era_2019"]
)
FORBIDDEN_SUBSTRINGS = ("pinnacle", "betfair", "close", "ft_")
CODE = {"home": "H", "draw": "D", "away": "A"}


def pull_queries() -> dict[str, str]:
    """SQL for `train` (exploratory window, Pinnacle close as target) and `eval` (evaluation window, Betfair close)."""
    base = f"FROM fd_matches WHERE {sobf.MAIN_LEAGUE_SQL}"
    tr = ", ".join(ID_COLS + SOURCE_COLS + TRAIN_TARGET_COLS)
    ev = ", ".join(ID_COLS + ["ft_result"] + SOURCE_COLS + EVAL_BENCH_COLS)
    return {
        "train": f"SELECT {tr} {base} AND match_date BETWEEN DATE '{TRAIN_WINDOW[0]}' AND DATE '{TRAIN_WINDOW[1]}'",
        "eval": f"SELECT {ev} {base} AND match_date BETWEEN DATE '{sobf.EVAL_WINDOW[0]}' "
        f"AND DATE '{sobf.EVAL_WINDOW[1]}'",
    }


def assert_features(cols: list[str]) -> None:
    """Raise AssertionError unless `cols` is exactly the registered feature list with no sharp/close/result column."""
    bad = [c for c in cols if any(s in c for s in FORBIDDEN_SUBSTRINGS)]
    if bad:
        raise AssertionError(f"forbidden columns in the feature matrix: {bad}")
    if list(cols) != FEATURE_NAMES:
        raise AssertionError(
            f"feature matrix differs from the registered whitelist: {sorted(set(cols) ^ set(FEATURE_NAMES))}"
        )


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def soft_valid(f: pd.DataFrame) -> np.ndarray:
    """Row mask: all nine soft open quotes (avg, max, bet365 x H/D/A) are valid prices."""
    return np.logical_and.reduce([sobf.odds_ok(sobf.triple(f, b, "open")) for b in BOOKS])


def build_features(odds: pd.DataFrame, match_date: pd.Series) -> pd.DataFrame:
    """The 28-column feature matrix from the nine soft open columns and the match date (rows must be valid).

    `odds` must hold exactly `SOURCE_COLS`; anything else raises, so a Pinnacle, Betfair or close column cannot
    enter. The returned columns are asserted equal to the registered list.
    """
    if list(odds.columns) != SOURCE_COLS:
        raise AssertionError(f"odds input must be exactly the soft open columns, got {list(odds.columns)}")
    t = {b: sobf.triple(odds, b, "open") for b in BOOKS}
    feats: dict[str, np.ndarray] = {}
    for c in SOURCE_COLS:
        feats[c] = odds[c].to_numpy(dtype=float)
    for b in BOOKS:
        p = sobf.devig_power_rows(t[b])
        for i, o in enumerate(sobf.OUTCOMES):
            feats[f"logit_{b}_{o}"] = logit(p[:, i])
    for a in ("max", "bet365"):
        for i, o in enumerate(sobf.OUTCOMES):
            feats[f"spread_{a}_{o}"] = t[a][:, i] - t["avg"][:, i]
    for b in BOOKS:
        feats[f"booksum_{b}"] = (1.0 / t[b]).sum(axis=1) - 1.0
    feats["era_2019"] = (pd.to_datetime(match_date) >= ERA_START).to_numpy(dtype=float)
    x = pd.DataFrame(feats, index=odds.index)
    assert_features(list(x.columns))
    return x


def new_model() -> HistGradientBoostingRegressor:
    """The fixed model; nothing is tuned."""
    return HistGradientBoostingRegressor(loss="squared_error", early_stopping=False, random_state=SEED, **HGB_PARAMS)


def fit_predict(x_train: np.ndarray, y_logit: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Fit one regressor per outcome on `y_logit` (n, 3) and return q_hat (n_test, 3): sigmoid then renormalised."""
    pred = np.column_stack([new_model().fit(x_train, y_logit[:, k]).predict(x_test) for k in range(3)])
    q = sigmoid(pred)
    return q / q.sum(axis=1, keepdims=True)


def walk_forward(x: np.ndarray, y_logit: np.ndarray, season: np.ndarray) -> np.ndarray:
    """Out-of-fold q_hat by season: each season is predicted by a model fitted on strictly earlier seasons.

    The first season has no earlier data and stays NaN. Expanding window, no refit inside a season.
    """
    q = np.full((len(x), 3), np.nan)
    seasons = np.unique(season)
    for s in seasons[1:]:
        tr, te = season < s, season == s
        q[te] = fit_predict(x[tr], y_logit[tr], x[te])
    return q


def decide(max_odds: np.ndarray, avg_odds: np.ndarray, q_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per match: chosen outcome, its edge log(max * q_hat) and the max price; outcomes with max/avg > 1.5 are out.

    Matches with no eligible outcome (or NaN q_hat) get edge -inf. Ties go to the first outcome.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        edge = np.log(max_odds * q_hat)
        edge = np.where(max_odds / avg_odds > CAP, -np.inf, edge)
    edge = np.where(np.isnan(edge), -np.inf, edge)
    j = np.argmax(edge, axis=1)
    rows = np.arange(len(edge))
    return j, edge[rows, j], max_odds[rows, j]


def bets_at(f: pd.DataFrame, q_hat: np.ndarray, p_close: np.ndarray, theta: float) -> pd.DataFrame:
    """Bets of one theta on matches `f` with predictions `q_hat` and close probabilities `p_close` (both (n, 3))."""
    j, edge, price = decide(sobf.triple(f, "max", "open"), sobf.triple(f, "avg", "open"), q_hat)
    return sobf.bets_frame(f, j, price, p_close, edge > theta).assign(edge=edge[edge > theta])


def train_universe(train: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Exploratory-window rows with valid soft opens and a valid Pinnacle close.

    Returns (frame, feature matrix, Pinnacle close probabilities (n, 3), season start year per row).
    """
    t = sobf.in_window(train, *TRAIN_WINDOW)
    close = sobf.triple(t, "pinnacle", "close")
    t = t[soft_valid(t) & sobf.odds_ok(close)].sort_values("match_date", kind="stable")
    p_close = sobf.devig_power_rows(sobf.triple(t, "pinnacle", "close"))
    x = build_features(t[SOURCE_COLS], t["match_date"])
    return t, x.to_numpy(), p_close, sobf.season_start(t["match_date"]).to_numpy()


def select_key(r: dict[str, Any]) -> tuple[float, float]:
    """Sort key: highest total out-of-fold CLV (rounded to 9 decimals), then lower theta."""
    return (-round(float(r["total_clv"]), 9), r["theta"])


def select(train: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose theta on out-of-fold predictions of the exploratory window only (benchmark: Pinnacle close).

    The eligible candidate (>= 1000 out-of-fold bets) with the highest total CLV wins, ties to the lower theta;
    with none eligible the fallback 0.02. Returns (selection dict, candidate rows for `trials.csv`).
    """
    f, x, p_close, season = train_universe(train)
    y = logit(p_close)
    q = walk_forward(x, y, season)
    oof = ~np.isnan(q[:, 0])
    f_o, q_o, p_o = f[oof], q[oof], p_close[oof]
    rows = []
    for theta in THETAS:
        b = bets_at(f_o, q_o, p_o, theta)
        s = sobf.boot(b["clv"].to_numpy(), b["match_date"].dt.strftime("%Y-%m-%d"))
        rows.append(
            {
                "theta": theta,
                "n_bets": int(s["n"]),
                "eligible": bool(s["n"] >= MIN_OOF_BETS),
                "mean_clv": s["mean"],
                "total_clv": float(b["clv"].sum()),
                "ci_lo": s["lo"],
                "ci_hi": s["hi"],
            }
        )
    best = sobf.pick(rows, select_key, MIN_OOF_BETS)
    theta, fallback = (FALLBACK_THETA, True) if best is None else (best["theta"], False)
    sel = {"theta": theta, "fallback": fallback, "n_oof_matches": int(oof.sum()), "candidates": rows}
    return sel, rows


def freeze(train: pd.DataFrame) -> FrozenModel:
    """Fit the final per-outcome models once on every exploratory-window row; returns them for frozen prediction."""
    f, x, p_close, _ = train_universe(train)
    y = logit(p_close)
    return FrozenModel([new_model().fit(x, y[:, k]) for k in range(3)], n_train=len(f))


class FrozenModel:
    """Three fitted per-outcome regressors; `predict` returns renormalised q_hat. Never refitted."""

    def __init__(self, models: list[HistGradientBoostingRegressor], n_train: int) -> None:
        self.models = models
        self.n_train = n_train

    def predict(self, x: np.ndarray) -> np.ndarray:
        q = sigmoid(np.column_stack([m.predict(x) for m in self.models]))
        return q / q.sum(axis=1, keepdims=True)


def eval_universe(ev: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Evaluation rows (window and exclusion applied by the caller) with valid soft opens and Betfair close.

    Returns (frame, feature matrix, Betfair close probabilities (n, 3)).
    """
    close = sobf.triple(ev, "betfair_ex", "close")
    e = ev[soft_valid(ev) & sobf.odds_ok(close)]
    p_close = sobf.devig_power_rows(sobf.triple(e, "betfair_ex", "close"))
    return e, build_features(e[SOURCE_COLS], e["match_date"]).to_numpy(), p_close


def kl_bits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-row KL(p || q) in bits."""
    return (p * (np.log2(np.clip(p, 1e-12, None)) - np.log2(np.clip(q, 1e-12, None)))).sum(axis=1)
