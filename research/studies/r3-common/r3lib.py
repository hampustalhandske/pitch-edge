"""Walk-forward closing-line models and scoring for the R3 `closing-line-prediction` studies.

Target. For each match and outcome o in (home, draw, away): move_o = logit(p_close_o) - logit(p_open_o),
with p_open the de-vigged Pinnacle open and p_close the de-vigged Pinnacle close (Betfair Exchange close
in the post segment), both de-vigged by the configured method (`power` or `shin`).

Baseline (the R2-Q2 consensus-gap regression, re-fit inside the walk-forward). Per outcome, OLS of move_o
on [1, g_avg_o, g_max_o], g_avg_o = logit(p_avg_open_o) - logit(p_open_o) (avg de-vigged by the same
method) and g_max_o = log(min(max_open_o / pinnacle_open_o, 1.5)).

Feature model. The baseline regressors plus a feature group's columns (outcome-specific coefficients).
Both models are fitted on exactly the same training rows (rows where the group's features exist).

Walk-forward. For every season s (July-June), both models are fitted on all rows of seasons < s and
predict season s; feature mappings that need fitting (the ordered logit turning a rating difference into
1X2 probabilities) are fitted on the same prior seasons only.

Predicted close: p_hat_o proportional to sigmoid(logit p_open_o + predicted move_o), renormalised.

Per-match scores (positive = the feature model is better):
- delta_kl_bits: KL(p_close || p_hat_base) - KL(p_close || p_hat_model), bits (the primary metric).
- delta_r2_move: (SSE_base - SSE_model) / mean(SSE_zero) per match, where SSE sums the three outcomes'
  squared move errors and SSE_zero is the no-move forecast's; its mean is the incremental OOS R^2.
- delta_ll_result_bits: log2 p_hat_model(result) - log2 p_hat_base(result).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from lake import REPO
from scipy.optimize import minimize

from pitch_edge.research.ledger import Ledger
from pitch_edge.research.registry import Registration, config_hash
from pitch_edge.research.stats import block_bootstrap
from pitch_edge.research.trial import TrialResult, run_trial

D = REPO / "data" / "research" / "r3"
OUTCOMES = ("home", "draw", "away")
EXPLORATORY_END = pd.Timestamp("2019-06-30")
EVAL_START, EVAL_END = pd.Timestamp("2019-07-01"), pd.Timestamp("2025-06-30")
POST_START = pd.Timestamp("2025-07-01")
GMAX_CAP = np.log(1.5)
SEED = 20261020
N_BOOT = 2000


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(D / "panel.parquet")
    for c in ("match_date", "snapshot_date"):
        p[c] = pd.to_datetime(p[c])
    return p


def panel_sha() -> str:
    return hashlib.sha256((D / "panel.parquet").read_bytes()).hexdigest()[:12]


def data_snapshot_label() -> str:
    at = (D / "snapshot_at.txt").read_text().strip()
    return f"r3 panel.parquet sha {panel_sha()} from lake snapshot {at} + warehouse tm_games"


def code_sha(files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------------- base quantities


def base_arrays(df: pd.DataFrame, method: str) -> dict[str, np.ndarray]:
    """(n, 3) arrays: p_open, p_close, move, g_avg, g_max; result index (0 home, 1 draw, 2 away)."""
    po = df[[f"p_open_{method}_{o}" for o in OUTCOMES]].to_numpy()
    pc = df[[f"p_close_{method}_{o}" for o in OUTCOMES]].to_numpy()
    pa = df[[f"p_avg_{method}_{o}" for o in OUTCOMES]].to_numpy()
    ratio = df[[f"max_{o}_open" for o in OUTCOMES]].to_numpy() / df[[f"pinnacle_{o}_open" for o in OUTCOMES]].to_numpy()
    res = df["ft_result"].map({"H": 0, "D": 1, "A": 2}).to_numpy(dtype=float)
    return {
        "p_open": po,
        "p_close": pc,
        "move": logit(pc) - logit(po),
        "g_avg": logit(pa) - logit(po),
        "g_max": np.minimum(np.log(np.clip(ratio, 1e-9, None)), GMAX_CAP),
        "res": res,
    }


# ------------------------------------------------------------------------------ ordered logit (1X2)


def fit_ordered_logit(x: np.ndarray, res: np.ndarray) -> np.ndarray:
    """Ordered logit P(away) = s(c1 - b x), P(away or draw) = s(c2 - b x); returns (b, c1, c2)."""
    ok = np.isfinite(x) & np.isfinite(res)
    x, r = x[ok], res[ok]
    y = np.where(r == 2, 0, np.where(r == 1, 1, 2))  # 0 away, 1 draw, 2 home

    def nll(th: np.ndarray) -> float:
        b, c1, d = th
        c2 = c1 + np.exp(d)
        p_a = sigmoid(c1 - b * x)
        p_ad = sigmoid(c2 - b * x)
        p = np.choose(y, [p_a, p_ad - p_a, 1 - p_ad])
        return -np.sum(np.log(np.clip(p, 1e-12, None)))

    sol = minimize(nll, np.array([0.5, -1.0, 0.0]), method="BFGS")
    b, c1, d = sol.x
    return np.array([b, c1, c1 + np.exp(d)])


def ordered_logit_probs(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """(n, 3) probabilities in (home, draw, away) order."""
    b, c1, c2 = theta
    p_a = sigmoid(c1 - b * x)
    p_ad = sigmoid(c2 - b * x)
    return np.column_stack([1 - p_ad, p_ad - p_a, p_a])


# ------------------------------------------------------------------------------- feature groups


@dataclass(frozen=True)
class Group:
    """A feature group: the panel rows it applies to, and a builder that returns per-outcome feature
    blocks [(n, k) for home, draw, away] for test rows given the training rows (for fitted mappings)."""

    name: str
    available: Callable[[pd.DataFrame], pd.Series]
    build: Callable[[pd.DataFrame, pd.DataFrame, str], list[np.ndarray]]
    known_at: str


def _rating_gap(col: str) -> Callable[[pd.DataFrame, pd.DataFrame, str], list[np.ndarray]]:
    def build(train: pd.DataFrame, test: pd.DataFrame, method: str) -> list[np.ndarray]:
        tr = base_arrays(train, method)
        theta = fit_ordered_logit(train[col].to_numpy(dtype=float), tr["res"])
        p = ordered_logit_probs(theta, test[col].to_numpy(dtype=float))
        po = test[[f"p_open_{method}_{o}" for o in OUTCOMES]].to_numpy()
        gap = logit(p) - logit(po)
        return [gap[:, [k]] for k in range(3)]

    return build


def _same_for_all(cols_fn: Callable[[pd.DataFrame, str], np.ndarray]) -> Callable:
    def build(train: pd.DataFrame, test: pd.DataFrame, method: str) -> list[np.ndarray]:
        x = cols_fn(test, method)
        return [x, x, x]

    return build


def _xg_cols(df: pd.DataFrame, method: str) -> np.ndarray:
    return df[["xg_luck_home", "xg_luck_away"]].to_numpy(dtype=float)


def _mgr_cols(df: pd.DataFrame, method: str) -> np.ndarray:
    nh, na = df["mgr_new_home"].to_numpy(dtype=float), df["mgr_new_away"].to_numpy(dtype=float)
    ph, pa = df[f"p_open_{method}_home"].to_numpy(), df[f"p_open_{method}_away"].to_numpy()
    return np.column_stack([nh, na, nh * ph, na * pa])


def _sch_cols(df: pd.DataFrame, method: str) -> np.ndarray:
    return np.column_stack(
        [
            np.minimum(df["sch_rest_home"].to_numpy(dtype=float), 7.0),
            np.minimum(df["sch_rest_away"].to_numpy(dtype=float), 7.0),
            df["sch_cong14_home"].to_numpy(dtype=float),
            df["sch_cong14_away"].to_numpy(dtype=float),
        ]
    )


GROUPS: dict[str, Group] = {
    "pi": Group("pi", lambda d: d["pi_egd"].notna(), _rating_gap("pi_egd"), "pi_known_at"),
    "clubelo": Group("clubelo", lambda d: d["elo_diff"].notna(), _rating_gap("elo_diff"), "elo_known_at"),
    "xg": Group(
        "xg", lambda d: d["xg_luck_home"].notna() & d["xg_luck_away"].notna(), _same_for_all(_xg_cols), "xg_known_at"
    ),
    "manager": Group(
        "manager",
        lambda d: d["mgr_new_home"].notna() & d["mgr_new_away"].notna(),
        _same_for_all(_mgr_cols),
        "mgr_known_at",
    ),
    "schedule": Group(
        "schedule",
        lambda d: d["sch_tm_both"].astype(bool) & d["sch_rest_home"].notna() & d["sch_rest_away"].notna(),
        _same_for_all(_sch_cols),
        "sch_known_at",
    ),
}


# ------------------------------------------------------------------------------------- walk-forward


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _design(b: dict[str, np.ndarray], k: int, extra: list[np.ndarray] | None) -> np.ndarray:
    cols = [np.ones(len(b["move"])), b["g_avg"][:, k], b["g_max"][:, k]]
    x = np.column_stack(cols)
    return x if extra is None else np.column_stack([x, extra[k]])


def _p_hat(p_open: np.ndarray, move_hat: np.ndarray) -> np.ndarray:
    q = sigmoid(logit(p_open) + move_hat)
    return q / q.sum(axis=1, keepdims=True)


def assert_pit(df: pd.DataFrame, group: Group) -> None:
    """Point-in-time guard: the group's known_at must be strictly before the snapshot date on every row
    used, and only `*_open` odds columns feed the regressors (close columns are targets)."""
    k = pd.to_datetime(df[group.known_at])
    if k.isna().any() or not (k < df["snapshot_date"]).all():
        raise AssertionError(f"{group.name}: known_at not strictly before snapshot_date on some rows")


def walk_forward(
    panel: pd.DataFrame, group: Group, method: str, predict_seasons: list[int], post: bool = False
) -> pd.DataFrame:
    """OOS per-match predictions and scores for each season in `predict_seasons` (and, when `post`, the
    2025-07+ Betfair-close segment predicted by a model fitted on every Pinnacle-segment season)."""
    d = panel[group.available(panel)].copy()
    assert_pit(d, group)
    pin = d[d["segment"] == "pinnacle"]
    out = []
    targets: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for s in predict_seasons:
        targets.append((pin[pin["season_start"] < s], pin[pin["season_start"] == s]))
    if post:
        targets.append((pin, d[d["segment"] == "post_betfair"]))
    for train, test in targets:
        if len(train) < 500 or test.empty:
            continue
        btr, bte = base_arrays(train, method), base_arrays(test, method)
        ftr, fte = group.build(train, train, method), group.build(train, test, method)
        mh_b = np.zeros((len(test), 3))
        mh_m = np.zeros((len(test), 3))
        for k in range(3):
            fin = np.isfinite(btr["move"][:, k])
            mh_b[:, k] = _design(bte, k, None) @ _ols(_design(btr, k, None)[fin], btr["move"][fin, k])
            mh_m[:, k] = _design(bte, k, fte) @ _ols(_design(btr, k, ftr)[fin], btr["move"][fin, k])
        out.append(_score(test, bte, mh_b, mh_m))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _kl_bits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1)
    q = np.clip(q, 1e-12, 1)
    return np.sum(p * np.log2(p / q), axis=1)


def _score(test: pd.DataFrame, b: dict[str, np.ndarray], mh_b: np.ndarray, mh_m: np.ndarray) -> pd.DataFrame:
    ph_b, ph_m = _p_hat(b["p_open"], mh_b), _p_hat(b["p_open"], mh_m)
    sse0 = np.sum(b["move"] ** 2, axis=1)
    sse_b = np.sum((b["move"] - mh_b) ** 2, axis=1)
    sse_m = np.sum((b["move"] - mh_m) ** 2, axis=1)
    res = b["res"]
    has_res = np.isfinite(res)
    ri = np.where(has_res, res, 0).astype(int)
    ll_b = np.log2(ph_b[np.arange(len(ri)), ri])
    ll_m = np.log2(ph_m[np.arange(len(ri)), ri])
    out = pd.DataFrame(
        {
            "match_id": test["match_id"].to_numpy(),
            "match_date": test["match_date"].to_numpy(),
            "division": test["division"].to_numpy(),
            "season_start": test["season_start"].to_numpy(),
            "segment": test["segment"].to_numpy(),
            "kl_open": _kl_bits(b["p_close"], b["p_open"]),
            "kl_base": _kl_bits(b["p_close"], ph_b),
            "kl_model": _kl_bits(b["p_close"], ph_m),
            "sse0": sse0,
            "sse_base": sse_b,
            "sse_model": sse_m,
            "ll_base": np.where(has_res, ll_b, np.nan),
            "ll_model": np.where(has_res, ll_m, np.nan),
        }
    )
    for k, o in enumerate(OUTCOMES):
        out[f"phat_base_{o}"] = ph_b[:, k]
        out[f"phat_model_{o}"] = ph_m[:, k]
        out[f"p_close_{o}"] = b["p_close"][:, k]
        out[f"p_open_{o}"] = b["p_open"][:, k]
    out["delta_kl_bits"] = out["kl_base"] - out["kl_model"]
    out["delta_ll_result_bits"] = out["ll_model"] - out["ll_base"]
    return out


def add_r2_value(s: pd.DataFrame) -> pd.DataFrame:
    """Per-match incremental-R^2 contribution: its mean over the frame equals the incremental OOS R^2."""
    s = s.copy()
    s["delta_r2_move"] = (s["sse_base"] - s["sse_model"]) / s["sse0"].mean()
    return s


def summary(s: pd.DataFrame, label: str) -> dict:
    """Headline numbers of a scored frame with match-day block-bootstrap CIs (descriptive use)."""
    s = add_r2_value(s)
    day = s["match_date"].dt.strftime("%Y-%m-%d").to_numpy()
    out: dict = {"label": label, "n": len(s), "days": int(pd.Series(day).nunique())}
    for col in ("delta_kl_bits", "delta_r2_move"):
        r = block_bootstrap(s[col].to_numpy(), day, n_boot=N_BOOT, seed=SEED)
        out[col] = (r.estimate, r.lo, r.hi)
    rr = s.dropna(subset=["delta_ll_result_bits"])
    r = block_bootstrap(
        rr["delta_ll_result_bits"].to_numpy(), rr["match_date"].dt.strftime("%Y-%m-%d"), n_boot=N_BOOT, seed=SEED
    )
    out["delta_ll_result_bits"] = (r.estimate, r.lo, r.hi)
    out["r2_base"] = 1 - s["sse_base"].sum() / s["sse0"].sum()
    out["r2_model"] = 1 - s["sse_model"].sum() / s["sse0"].sum()
    out["kl_open"] = s["kl_open"].mean()
    out["kl_base"] = s["kl_base"].mean()
    out["kl_model"] = s["kl_model"].mean()
    return out


def fmt(x: tuple[float, float, float], scale: float = 1.0, nd: int = 6) -> str:
    return f"{x[0] * scale:+.{nd}g} [{x[1] * scale:+.{nd}g}, {x[2] * scale:+.{nd}g}]"


# ------------------------------------------------------------------------------------ ledger glue


class ReproductionMismatch(RuntimeError):
    """A rerun's headline differs from the result of record already in the ledger."""


def _league_labels(frame: pd.DataFrame) -> np.ndarray:
    """Division per row, divisions with fewer than 100 rows pooled as 'other' (the stability convention)."""
    div = frame["division"].astype(str)
    return div.where(div.map(div.value_counts()) >= 100, "other").to_numpy()


def reproduction_check(
    reg: Registration, config: dict, pdf: pd.DataFrame, metric_name: str, ledger: Ledger
) -> dict | None:
    """Compare the headline of `pdf` with the earliest ledger trial of the same (hypothesis, config, metric).

    Returns None when the ledger has no such trial; otherwise a dict with both headlines and `ok`, true when
    the estimate and both CI bounds agree to 1e-9 relative and n is equal (the same bootstrap settings as
    `record`: N_BOOT draws, SEED)."""
    t = ledger.trials(hypothesis_id=reg.id)
    t = t[(t["config_hash"] == config_hash(config)) & (t["metric_name"] == metric_name)]
    if t.empty:
        return None
    ref = t.sort_values(["run_at", "trial_id"]).iloc[0]
    b = block_bootstrap(pdf["value"].to_numpy(), pdf["block"].to_numpy(), n_boot=N_BOOT, seed=SEED)
    got = np.array([b.estimate, b.lo, b.hi])
    want = np.array([ref["metric_value"], ref["ci_lo"], ref["ci_hi"]], dtype=float)
    ok = bool(int(ref["n"]) == b.n and np.allclose(got, want, rtol=1e-9, atol=1e-12))
    return {"ok": ok, "n": b.n, "ref_n": int(ref["n"]), "got": got.tolist(), "ref": want.tolist()}


def record(
    reg: Registration,
    config: dict,
    frame: pd.DataFrame,
    value_col: str,
    metric_name: str,
    extra: dict,
    ledger: Ledger,
    code_files: list[Path] | None = None,
    check_only: bool = False,
) -> TrialResult | None:
    """run_trial with retry on a DuckDB lock error, then verify the row is in the ledger.

    When `frame` has a `division` column it is passed as the `league` column (divisions under 100 rows pooled)
    so run_trial computes the leave-one-league-out CIs; `code_files` are hashed into the trial. With
    `check_only` nothing is recorded: the headline is compared with the ledger's earliest trial of the same
    (config, metric) and `ReproductionMismatch` is raised when it differs; returns None."""
    cols = {
        "decision_id": frame["match_id"].to_numpy() if "decision_id" not in frame else frame["decision_id"].to_numpy(),
        "block": frame["match_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "value": frame[value_col].to_numpy(dtype=float),
    }
    if "division" in frame:
        cols["league"] = _league_labels(frame)
    pdf = pd.DataFrame(cols)
    chk = reproduction_check(reg, config, pdf, metric_name, ledger)
    if chk is not None and not chk["ok"]:
        raise ReproductionMismatch(f"{reg.id} {config} {metric_name}: {chk}")
    if check_only:
        print(
            f"   check {metric_name}: {'reproduces the ledger trial' if chk else 'no earlier trial'}"
            + (f" (n={chk['n']})" if chk else "")
        )
        return None
    before = len(ledger.trials(hypothesis_id=reg.id))
    for attempt in range(30):
        try:
            res = run_trial(
                reg,
                config,
                pdf,
                metric_name,
                ledger,
                data_snapshot_label(),
                extra,
                n_boot=N_BOOT,
                seed=SEED,
                code_files=code_files or (),
            )
            break
        except duckdb.IOException as exc:
            if "lock" not in str(exc).lower():
                raise
            time.sleep(min(2 + 2 * attempt, 30))
    else:
        raise RuntimeError("ledger stayed locked")
    after = ledger.trials(hypothesis_id=reg.id)
    if len(after) != before + 1 or res.trial_id not in set(after["trial_id"]):
        raise RuntimeError(f"trial {res.trial_id} not found in the ledger after recording")
    return res


def loo_league(frame: pd.DataFrame, col: str) -> dict[str, float]:
    """Mean of `col` leaving out each division in turn (divisions with < 100 rows pooled as 'other')."""
    div = frame["division"].where(frame["division"].map(frame["division"].value_counts()) >= 100, "other")
    return {str(k): float(frame.loc[div != k, col].mean()) for k in sorted(div.unique())}


def to_json(x: object) -> str:
    return json.dumps(x, default=str, sort_keys=True)
