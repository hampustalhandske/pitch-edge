"""R6 — predicting the de-vigged sharp close from the full cross-book picture at the football-data open snapshot.

Data. `data/research/r2/audit_q3_raw.parquet` only: the raw football-data per-book columns (`odds_all`) of the
22 main divisions, pulled from the lake 2026-10-20T20:43Z. No warehouse, no lake.

Decision time. The football-data pre-closing ("open") snapshot: Friday afternoon for Fri-Mon games, Tuesday
afternoon for Tue-Thu games (football-data notes.txt; R1 takes 18:00 Europe/London as `open_known_at`).
`snapshot_date` is that Friday/Tuesday (R3 convention). Every feature is built only from columns in
`OPEN_COLUMNS` (quotes in that snapshot). `assert_open_only` enforces this in code twice: (1) the feature
builder may only read allow-listed open columns, and (2) features computed from a frame whose closes,
results and every other non-open column are blanked must equal the features computed from the full frame.

Target. Per outcome o in (home, draw, away): move_o = logit(p_close_o) - logit(p_open_o), p_open = the
de-vigged Pinnacle open, p_close = the de-vigged Pinnacle close (2012-07 -> 2025-06) or the de-vigged Betfair
Exchange close (post segment, 2025-07+). Predicted close p_hat_o proportional to sigmoid(logit p_open_o +
predicted move_o), renormalised.

Baseline (R3, re-fitted in the same walk-forward): per-outcome OLS of move_o on [1, g_avg_o, g_max_o],
g_avg_o = logit(p_avg_open_o) - logit(p_open_o), g_max_o = log(min(max_open_o / pinnacle_open_o, 1.5)).

Cross-book features (per outcome o): for each core book b, dev_b_o = logit(p_b_open_o) - logit(p_open_o)
(both de-vigged by the same method), clipped to [-DEV_CLIP, DEV_CLIP]; when b's quote is missing, dev_b_o is
imputed by the mean deviation of the core books that are present (or g_avg_o when none is) and miss_b = 1;
each book's overround (booksum - 1, centred by its training mean when present, 0 when missing) and Pinnacle's
overround; cross-book dispersion = std of logit p over Pinnacle and the present core books; the max booksum;
plus the baseline's g_avg_o and g_max_o. H2 adds the probability level logit(p_open_o).

Walk-forward by season (July-June): fit on every Pinnacle-segment season before s, predict s. The post
segment is predicted by a model fitted on every Pinnacle-segment season.
"""

from __future__ import annotations

import hashlib
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")  # libomp contention makes multi-threaded HGB ~30x slower here
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pitch_edge.research import devig as dv

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data" / "research" / "r2" / "audit_q3_raw.parquet"
OUT = REPO / "data" / "research" / "r6"
OUTCOMES = ("home", "draw", "away")
SFX = {"home": "H", "draw": "D", "away": "A"}
CORE_BOOKS = ("B365", "BW", "IW", "WH", "VC")  # >= 90% open coverage in every season 2012/13-2022/23
LATE_BOOKS = ("1XB", "BF", "BFE")  # football-data columns that exist only from 2024/25
GMAX_CAP = np.log(1.5)
DEV_CLIP = 0.5
EXPLORATORY_END = pd.Timestamp("2019-06-30")
EVAL_START, EVAL_END = pd.Timestamp("2019-07-01"), pd.Timestamp("2025-06-30")
POST_START, POST_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2026-10-19")
SEED = 20261020
N_BOOT = 2000

# Columns of the open snapshot (a feature may read these and nothing else).
OPEN_COLUMNS = frozenset(
    [f"{b}{s}" for b in ("PS", *CORE_BOOKS, *LATE_BOOKS, "BbAv", "BbMx", "Avg", "Max") for s in "HDA"]
)
# Hindsight columns (targets / benchmarks only).
HINDSIGHT_COLUMNS = frozenset(
    [f"{b}C{s}" for b in ("PS", *CORE_BOOKS, *LATE_BOOKS, "Avg", "Max") for s in "HDA"] + ["ft_result"]
)
assert not OPEN_COLUMNS & HINDSIGHT_COLUMNS


DRY = bool(os.environ.get("R6_DRY_LEDGER"))
if DRY:
    # Code check on a scratch ledger with scratch registrations: exploratory rows only, the last two exploratory
    # seasons stand in for the evaluation window. Nothing written in this mode is a counted trial.
    EVAL_START, EVAL_END = pd.Timestamp("2017-07-01"), pd.Timestamp("2019-06-30")
    OUT = OUT / "dry"


def ledger_path() -> Path:
    return Path(os.environ["R6_DRY_LEDGER"]) if DRY else REPO / "data" / "research" / "ledger.duckdb"


def registry_dir() -> Path:
    return Path(os.environ["R6_DRY_REG_DIR"]) if DRY else REPO / "research" / "registry"


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def data_snapshot_label() -> str:
    pulled = pd.read_parquet(RAW, columns=["pulled_at"])["pulled_at"].iloc[0]
    return f"data/research/r2/audit_q3_raw.parquet sha {file_sha(RAW)} (lake fd_matches.odds_all pulled {pulled})"


def snapshot_date(match_date: pd.Series) -> pd.Series:
    """Friday on/before the match for Fri-Mon games, Tuesday on/before it for Tue-Thu games."""
    wd = match_date.dt.weekday  # Mon=0
    back = np.select([wd == 0, wd == 1, wd == 2, wd == 3, wd == 4, wd == 5, wd == 6], [3, 0, 1, 2, 0, 1, 2])
    return match_date - pd.to_timedelta(back, unit="D")


def _triple(d: pd.DataFrame, book: str, close: bool = False) -> np.ndarray:
    cols = [f"{book}{'C' if close else ''}{SFX[o]}" for o in OUTCOMES]
    if not all(c in d.columns for c in cols):
        return np.full((len(d), 3), np.nan)
    x = d[cols].to_numpy(dtype=float, copy=True)
    ok = np.isfinite(x).all(axis=1) & (x > 1.0).all(axis=1)
    x[~ok] = np.nan
    return x


def _coalesce(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    use_a = np.isfinite(a).all(axis=1)
    return np.where(use_a[:, None], a, b)


# ------------------------------------------------------------------------------------------- panel


def load_panel() -> pd.DataFrame:
    """Universe rows with the raw open quotes, the targets' raw closing quotes and identifiers.

    Pinnacle segment (match_date <= 2025-06-30): Pinnacle open + close, avg + max open, a result.
    Post segment (2025-07-01 .. 2026-10-19): Pinnacle open, avg + max open, Betfair Exchange close.
    Avg/max open are Betbrain's BbAv/BbMx before 2019/20 and football-data's Avg/Max after.
    """
    d = pd.read_parquet(RAW)
    d["match_date"] = pd.to_datetime(d["match_date"])
    d = d[d["match_date"] >= "2012-07-01"].copy()
    for o in OUTCOMES:
        s = SFX[o]
        d[f"avg{s}_open"] = d[f"BbAv{s}"].where(d[f"BbAv{s}"].notna(), d[f"Avg{s}"])
        d[f"max{s}_open"] = d[f"BbMx{s}"].where(d[f"BbMx{s}"].notna(), d[f"Max{s}"])
    pin_o = _triple(d, "PS")
    avg = d[[f"avg{SFX[o]}_open" for o in OUTCOMES]].to_numpy(dtype=float)
    mx = d[[f"max{SFX[o]}_open" for o in OUTCOMES]].to_numpy(dtype=float)
    ok_open = (
        np.isfinite(pin_o).all(1)
        & np.isfinite(avg).all(1)
        & (avg > 1).all(1)
        & np.isfinite(mx).all(1)
        & (mx > 1).all(1)
    )
    pin_c = np.isfinite(_triple(d, "PS", close=True)).all(1)
    bfe_c = np.isfinite(_triple(d, "BFE", close=True)).all(1)
    pre = (d["match_date"] <= EVAL_END).to_numpy()
    post = ((d["match_date"] >= POST_START) & (d["match_date"] <= POST_END)).to_numpy()
    has_res = d["ft_result"].isin(["H", "D", "A"]).to_numpy()
    keep = ok_open & ((pre & pin_c & has_res) | (post & bfe_c))
    d = d[keep].copy()
    d["segment"] = np.where(d["match_date"] <= EVAL_END, "pinnacle", "post_betfair")
    d["snapshot_date"] = snapshot_date(d["match_date"])
    d["season_start"] = np.where(d["match_date"].dt.month >= 7, d["match_date"].dt.year, d["match_date"].dt.year - 1)
    d["match_id"] = (
        d["division"] + "|" + d["match_date"].dt.strftime("%Y-%m-%d") + "|" + d["home_team"] + "|" + d["away_team"]
    )
    d = d.drop_duplicates("match_id", keep="first")
    if DRY:
        d = d[d["match_date"] <= EXPLORATORY_END]
    return d.sort_values(["match_date", "match_id"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------- features


def _devig(x: np.ndarray, method: str) -> np.ndarray:
    out = np.full_like(x, np.nan)
    ok = np.isfinite(x).all(axis=1)
    if ok.any():
        out[ok] = dv.devig(x[ok], method=method).probs
    return out


@dataclass
class Features:
    """Per-outcome design blocks for one frame: `base[k]` the baseline regressors (g_avg, g_max), `book[k]`
    the cross-book block, `level[k]` logit p_open; `names` the cross-book column names."""

    base: list[np.ndarray]
    book: list[np.ndarray]
    level: list[np.ndarray]
    names: list[str]
    p_open: np.ndarray
    dev: dict[str, np.ndarray]  # raw (unclipped, un-imputed) deviations per book, (n, 3), NaN when missing


class _OpenOnly:
    """Read-only view of a frame that refuses any column outside OPEN_COLUMNS (and the derived avg/max opens)."""

    ALLOWED_DERIVED = frozenset(f"{a}{SFX[o]}_open" for a in ("avg", "max") for o in OUTCOMES)

    def __init__(self, d: pd.DataFrame) -> None:
        self._d = d
        self.columns = [c for c in d.columns if c in OPEN_COLUMNS or c in self.ALLOWED_DERIVED]

    def __len__(self) -> int:
        return len(self._d)

    def __getitem__(self, cols: list[str]) -> pd.DataFrame:
        bad = [c for c in cols if c not in OPEN_COLUMNS and c not in self.ALLOWED_DERIVED]
        if bad:
            raise AssertionError(f"feature builder tried to read non-open columns {bad}")
        return self._d[cols]


def build_features(df: pd.DataFrame, method: str) -> Features:
    """Open-snapshot features. Reads only OPEN_COLUMNS through `_OpenOnly` (raises otherwise)."""
    d = _OpenOnly(df)

    def tri(book: str) -> np.ndarray:
        cols = [f"{book}{SFX[o]}" for o in OUTCOMES]
        if not all(c in d.columns for c in cols):
            return np.full((len(d), 3), np.nan)
        x = d[cols].to_numpy(dtype=float, copy=True)
        ok = np.isfinite(x).all(axis=1) & (x > 1.0).all(axis=1)
        x[~ok] = np.nan
        return x

    pin = tri("PS")
    po = _devig(pin, method)
    lpo = logit(po)
    avg = d[[f"avg{SFX[o]}_open" for o in OUTCOMES]].to_numpy(dtype=float)
    mx = d[[f"max{SFX[o]}_open" for o in OUTCOMES]].to_numpy(dtype=float)
    g_avg = logit(_devig(avg, method)) - lpo
    g_max = np.minimum(np.log(np.clip(mx / pin, 1e-9, None)), GMAX_CAP)
    dev: dict[str, np.ndarray] = {}
    lp_books: dict[str, np.ndarray] = {}
    booksum: dict[str, np.ndarray] = {}
    for b in CORE_BOOKS + LATE_BOOKS:
        q = tri(b)
        lp = logit(_devig(q, method))
        lp_books[b] = lp
        dev[b] = lp - lpo
        booksum[b] = (1.0 / q).sum(axis=1)
    core = np.stack([np.clip(dev[b], -DEV_CLIP, DEV_CLIP) for b in CORE_BOOKS], axis=0)  # (B, n, 3)
    present = np.isfinite(core)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_present = np.nanmean(core, axis=0)  # NaN where no core book is present
    fill = np.where(np.isfinite(mean_present), mean_present, np.clip(g_avg, -DEV_CLIP, DEV_CLIP))
    lp_all = np.stack([lpo] + [lp_books[b] for b in CORE_BOOKS], axis=0)
    with np.errstate(all="ignore"):
        disp = np.nanstd(lp_all, axis=0)  # (n, 3); Pinnacle always present
    pin_over = (1.0 / pin).sum(axis=1) - 1.0
    max_sum = (1.0 / mx).sum(axis=1)
    names: list[str] = []
    blocks: list[list[np.ndarray]] = [[], [], []]
    for i, b in enumerate(CORE_BOOKS):
        miss = ~present[i, :, 0]
        for k in range(3):
            blocks[k].append(np.where(miss, fill[:, k], core[i, :, k]))
        names.append(f"dev_{b}")
    for b in CORE_BOOKS:
        miss = ~np.isfinite(booksum[b])
        for k in range(3):
            blocks[k].append(miss.astype(float))
        names.append(f"miss_{b}")
    for b in CORE_BOOKS:
        ov = booksum[b] - 1.0
        for k in range(3):
            blocks[k].append(ov)  # NaN when missing; centred/filled at fit time (training means)
        names.append(f"over_{b}")
    for k in range(3):
        blocks[k].append(pin_over)
        blocks[k].append(max_sum)
        blocks[k].append(disp[:, k])
    names += ["over_PS", "max_booksum", "dispersion"]
    return Features(
        base=[np.column_stack([g_avg[:, k], g_max[:, k]]) for k in range(3)],
        book=[np.column_stack(blocks[k]) for k in range(3)],
        level=[lpo[:, [k]] for k in range(3)],
        names=names,
        p_open=po,
        dev=dev,
    )


def assert_open_only(df: pd.DataFrame, method: str) -> int:
    """Open-only guard: features from `df` must equal features from a copy whose every non-open column
    (closes, results, Max/Avg closes, kickoff, ...) is blanked. Returns the number of rows checked."""
    blank = df.copy()
    keep = OPEN_COLUMNS | _OpenOnly.ALLOWED_DERIVED
    for c in blank.columns:
        if c not in keep:
            blank[c] = np.nan if blank[c].dtype.kind in "fc" else None
    for c in HINDSIGHT_COLUMNS & set(blank.columns):
        assert blank[c].isna().all()
    a, b = build_features(df, method), build_features(blank, method)
    for k in range(3):
        for x, y in ((a.base[k], b.base[k]), (a.book[k], b.book[k]), (a.level[k], b.level[k])):
            if not np.array_equal(x, y, equal_nan=True):
                raise AssertionError("features changed when hindsight columns were blanked")
    if not (df["snapshot_date"] <= df["match_date"]).all():
        raise AssertionError("snapshot_date after match_date")
    return len(df)


def targets(df: pd.DataFrame, method: str) -> dict[str, np.ndarray]:
    """Hindsight targets: de-vigged open and close (Pinnacle close, or Betfair Exchange close in the post
    segment), the logit move, and the result index (0 home, 1 draw, 2 away; NaN when absent)."""
    po = _devig(_triple(df, "PS"), method)
    is_pin = (df["segment"] == "pinnacle").to_numpy()[:, None]
    close = np.where(is_pin, _triple(df, "PS", close=True), _triple(df, "BFE", close=True))
    pc = _devig(close, method)
    res = df["ft_result"].map({"H": 0, "D": 1, "A": 2}).to_numpy(dtype=float)
    return {"p_open": po, "p_close": pc, "move": logit(pc) - logit(po), "res": res}


# ------------------------------------------------------------------------------------------- models


def _prep_book(train_x: np.ndarray, test_x: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Overround columns: subtract the training mean of present values; missing -> 0 (the miss_ flag carries it)."""
    tr, te = train_x.copy(), test_x.copy()
    for j, nm in enumerate(names):
        if nm.startswith("over_") and nm != "over_PS":
            mu = np.nanmean(tr[:, j])
            tr[:, j] = np.where(np.isfinite(tr[:, j]), tr[:, j] - mu, 0.0)
            te[:, j] = np.where(np.isfinite(te[:, j]), te[:, j] - mu, 0.0)
    return tr, te


def _ols_fit_predict(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray) -> np.ndarray:
    a = np.column_stack([np.ones(len(xtr)), xtr])
    beta = np.linalg.lstsq(a, ytr, rcond=None)[0]
    return np.column_stack([np.ones(len(xte)), xte]) @ beta


RIDGE_LAMBDAS = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)


def _ridge_beta(xs: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    ym = y.mean()
    g = xs.T @ xs + lam * np.eye(xs.shape[1])
    return ym, np.linalg.solve(g, xs.T @ (y - ym))


def _ridge_fit_predict(
    xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, seasons: np.ndarray
) -> tuple[np.ndarray, float]:
    """Ridge on standardised features (unpenalised intercept); lambda chosen from RIDGE_LAMBDAS by the MSE on
    the last training season of a model fitted on the earlier training seasons; then refit on all training."""
    mu, sd = xtr.mean(axis=0), xtr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    zs_tr, zs_te = (xtr - mu) / sd, (xte - mu) / sd
    last = seasons.max()
    inner_tr, inner_va = seasons < last, seasons == last
    best_lam, best = RIDGE_LAMBDAS[0], np.inf
    if inner_tr.sum() > 500 and inner_va.sum() > 100:
        for lam in RIDGE_LAMBDAS:
            ym, b = _ridge_beta(zs_tr[inner_tr], ytr[inner_tr], lam)
            mse = float(np.mean((ytr[inner_va] - ym - zs_tr[inner_va] @ b) ** 2))
            if mse < best - 1e-15:
                best, best_lam = mse, lam
    ym, b = _ridge_beta(zs_tr, ytr, best_lam)
    return ym + zs_te @ b, best_lam


HGB_CONFIGS = {
    "shallow": {
        "max_iter": 200,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 200,
        "l2_regularization": 1.0,
    },
    "deeper": {
        "max_iter": 400,
        "learning_rate": 0.03,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
    },
    "tiny": {
        "max_iter": 150,
        "learning_rate": 0.03,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 1000,
        "l2_regularization": 10.0,
    },
}


def _hgb_fit_predict(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, cfg: str) -> np.ndarray:
    m = HistGradientBoostingRegressor(loss="squared_error", early_stopping=False, random_state=SEED, **HGB_CONFIGS[cfg])
    m.fit(xtr, ytr)
    return m.predict(xte)


def _p_hat(p_open: np.ndarray, move_hat: np.ndarray) -> np.ndarray:
    q = sigmoid(logit(p_open) + move_hat)
    return q / q.sum(axis=1, keepdims=True)


def _kl_bits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1)
    q = np.clip(q, 1e-12, 1)
    return np.sum(p * np.log2(p / q), axis=1)


def fit_predict(
    model: str, train: pd.DataFrame, test: pd.DataFrame, method: str
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Walk-forward step: returns (baseline move_hat, model move_hat), each (n_test, 3), and fit info.

    model: 'ols' | 'ridge' (H1 cross-book linear), 'hgb_shallow' | 'hgb_deeper' (H2), 'base' (baseline twice).
    """
    ftr, fte = build_features(train, method), build_features(test, method)
    ttr = targets(train, method)
    seasons = train["season_start"].to_numpy()
    mh_b = np.zeros((len(test), 3))
    mh_m = np.zeros((len(test), 3))
    info: dict = {"lambda": [], "n_train": len(train)}
    for k in range(3):
        y = ttr["move"][:, k]
        fin = np.isfinite(y)
        mh_b[:, k] = _ols_fit_predict(ftr.base[k][fin], y[fin], fte.base[k])
        if model == "base":
            mh_m[:, k] = mh_b[:, k]
            continue
        btr, bte = _prep_book(ftr.book[k], fte.book[k], ftr.names)
        xtr = np.column_stack([ftr.base[k], btr])
        xte = np.column_stack([fte.base[k], bte])
        if model == "ols":
            mh_m[:, k] = _ols_fit_predict(xtr[fin], y[fin], xte)
        elif model == "ridge":
            pred, lam = _ridge_fit_predict(xtr[fin], y[fin], xte, seasons[fin])
            mh_m[:, k] = pred
            info["lambda"].append(lam)
        elif model.startswith("hgbres_"):
            # ridge first, then HGB fitted on the ridge's in-sample residual (a boosted correction of the linear fit)
            pred_tr, lam = _ridge_fit_predict(xtr[fin], y[fin], xtr[fin], seasons[fin])
            pred_te, _ = _ridge_fit_predict(xtr[fin], y[fin], xte, seasons[fin])
            xtr2 = np.column_stack([xtr, ftr.level[k]])[fin]
            xte2 = np.column_stack([xte, fte.level[k]])
            mh_m[:, k] = pred_te + _hgb_fit_predict(xtr2, y[fin] - pred_tr, xte2, model.split("_", 1)[1])
            info["lambda"].append(lam)
        elif model.startswith("hgb_"):
            xtr2 = np.column_stack([xtr, ftr.level[k]])
            xte2 = np.column_stack([xte, fte.level[k]])
            mh_m[:, k] = _hgb_fit_predict(xtr2[fin], y[fin], xte2, model.split("_", 1)[1])
        else:
            raise ValueError(model)
    return mh_b, mh_m, info


def score(test: pd.DataFrame, method: str, mh_b: np.ndarray, mh_m: np.ndarray) -> pd.DataFrame:
    """Per-match scores (positive = model better than the R3 baseline)."""
    t = targets(test, method)
    ph_b, ph_m = _p_hat(t["p_open"], mh_b), _p_hat(t["p_open"], mh_m)
    sse0 = np.sum(t["move"] ** 2, axis=1)
    sse_b = np.sum((t["move"] - mh_b) ** 2, axis=1)
    sse_m = np.sum((t["move"] - mh_m) ** 2, axis=1)
    res = t["res"]
    has = np.isfinite(res)
    ri = np.where(has, res, 0).astype(int)
    out = pd.DataFrame(
        {
            "match_id": test["match_id"].to_numpy(),
            "match_date": test["match_date"].to_numpy(),
            "division": test["division"].to_numpy(),
            "season_start": test["season_start"].to_numpy(),
            "segment": test["segment"].to_numpy(),
            "kl_open": _kl_bits(t["p_close"], t["p_open"]),
            "kl_base": _kl_bits(t["p_close"], ph_b),
            "kl_model": _kl_bits(t["p_close"], ph_m),
            "sse0": sse0,
            "sse_base": sse_b,
            "sse_model": sse_m,
            "ll_base": np.where(has, np.log2(ph_b[np.arange(len(ri)), ri]), np.nan),
            "ll_model": np.where(has, np.log2(ph_m[np.arange(len(ri)), ri]), np.nan),
        }
    )
    for k, o in enumerate(OUTCOMES):
        out[f"p_open_{o}"] = t["p_open"][:, k]
        out[f"p_close_{o}"] = t["p_close"][:, k]
        out[f"phat_base_{o}"] = ph_b[:, k]
        out[f"phat_model_{o}"] = ph_m[:, k]
    out["delta_kl_bits"] = out["kl_base"] - out["kl_model"]
    out["kl_gain_vs_open_bits"] = out["kl_open"] - out["kl_model"]
    out["delta_ll_result_bits"] = out["ll_model"] - out["ll_base"]
    return out


def walk_forward(panel: pd.DataFrame, model: str, method: str, seasons: list[int], post: bool = False) -> pd.DataFrame:
    """OOS per-match scores for each season in `seasons` (fit on all prior Pinnacle-segment seasons), plus the
    post segment (fit on every Pinnacle-segment season) when `post`."""
    pin = panel[panel["segment"] == "pinnacle"]
    jobs = [(pin[pin["season_start"] < s], pin[pin["season_start"] == s]) for s in seasons]
    if post:
        jobs.append((pin, panel[panel["segment"] == "post_betfair"]))
    out = []
    for train, test in jobs:
        if len(train) < 5000 or test.empty:
            continue
        assert train["match_date"].max() < test["match_date"].min(), "training rows must precede the test rows"
        assert_open_only(test, method)
        mh_b, mh_m, info = fit_predict(model, train, test, method)
        s = score(test, method, mh_b, mh_m)
        s["ridge_lambda"] = str(info["lambda"]) if info["lambda"] else ""
        out.append(s)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def add_r2_value(s: pd.DataFrame) -> pd.DataFrame:
    """Per-match incremental-R^2 contribution; its mean over the frame is the incremental OOS R^2 of the move."""
    s = s.copy()
    s["delta_r2_move"] = (s["sse_base"] - s["sse_model"]) / s["sse0"].mean()
    return s
