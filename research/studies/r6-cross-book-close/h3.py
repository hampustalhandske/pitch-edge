"""H3 — descriptive book informativeness (closing-line-prediction-book-informativeness-v1).

Rows: (match, outcome) in the Pinnacle segment of the evaluation window 2019-07-01..2025-06-30. For book b:
  x = dev_b = logit p_b_open - logit p_pin_open (power de-vig, the open snapshot; rows with |x| > DEV_CLIP are
      treated as quote errors and dropped, count reported),
  y = Pinnacle move = logit p_pin_close - logit p_pin_open (hindsight target).
marginal slope  = OLS slope of y on [1, x]                -> the share of b's deviation that Pinnacle closes;
partial slope   = coefficient of x in OLS y ~ 1 + x + g_avg -> information beyond the soft consensus;
self share      = OLS slope of -(logit p_b_close - logit p_b_open) on [1, x] (rows with b's close) -> the share
                  of b's deviation that b itself closes. On the same rows marginal + self + persistence = 1.
Primary: slope_contrast = slope(VC) - slope(IW) on rows where both are present (both slope types).
Slopes enter run_trial as per-row influence values (mean = slope exactly); exact match-day block-bootstrap CIs
are stored beside them. BH (q = 0.10, one-sided > 0) across the 8 books x 2 slope types.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import r6lib as L

from pitch_edge.research.ledger import Ledger
from pitch_edge.research.registry import load
from pitch_edge.research.stats import benjamini_hochberg, block_bootstrap

BOOKS = L.CORE_BOOKS + L.LATE_BOOKS
TOP, BOTTOM = "VC", "IW"  # exploratory 2012/13-2018/19 marginal slopes: VC 0.281 (highest), IW 0.055 (lowest)


def coef_influence(a: np.ndarray, y: np.ndarray, j: int) -> tuple[float, np.ndarray]:
    """OLS coefficient j of y on design a (with intercept column) and per-row influence values (mean = coef)."""
    g = a.T @ a
    beta = np.linalg.solve(g, a.T @ y)
    e = y - a @ beta
    ginv_row = np.linalg.solve(g, np.eye(a.shape[1]))[j]
    return float(beta[j]), beta[j] + len(y) * (a @ ginv_row) * e


def exact_coef_ci(a: np.ndarray, y: np.ndarray, j: int, blocks: np.ndarray) -> dict:
    """Coefficient j with its exact match-day block-bootstrap percentile CI (whole blocks resampled)."""
    codes, uniq = pd.factorize(pd.Series(blocks))
    nb, p = len(uniq), a.shape[1]
    aa = np.zeros((nb, p, p))
    ay = np.zeros((nb, p))
    np.add.at(aa, codes, a[:, :, None] * a[:, None, :])
    np.add.at(ay, codes, a * y[:, None])
    rng = np.random.default_rng(L.SEED)
    w = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(L.N_BOOT)]).astype(float)
    g = np.einsum("bk,kij->bij", w, aa)
    r = w @ ay
    boot = np.linalg.solve(g, r[:, :, None])[:, j, 0]
    point = float(np.linalg.solve(aa.sum(0), ay.sum(0))[j])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "coef": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": len(y),
        "n_days": nb,
        "p_gt0": float((1 + np.sum(boot <= 0)) / (L.N_BOOT + 1)),
    }


def long_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, outcome) with the move, g_avg, and each book's open deviation and own close move."""
    f = L.build_features(panel, "power")
    t = L.targets(panel, "power")
    frames = []
    for k, o in enumerate(L.OUTCOMES):
        d = pd.DataFrame(
            {
                "decision_id": panel["match_id"].to_numpy() + "|" + o,
                "match_id": panel["match_id"].to_numpy(),
                "match_date": panel["match_date"].to_numpy(),
                "season_start": panel["season_start"].to_numpy(),
                "division": panel["division"].to_numpy(),
                "outcome": o,
                "y": t["move"][:, k],
                "g_avg": f.base[k][:, 0],
            }
        )
        for b in BOOKS:
            d[f"x_{b}"] = f.dev[b][:, k]
            close = L._devig(L._triple(panel, b, close=True), "power")
            opn = L._devig(L._triple(panel, b), "power")
            d[f"self_{b}"] = -(L.logit(close[:, k]) - L.logit(opn[:, k]))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def book_rows(df: pd.DataFrame, b: str) -> pd.DataFrame:
    ok = (
        np.isfinite(df[f"x_{b}"]) & np.isfinite(df["y"]) & np.isfinite(df["g_avg"]) & (df[f"x_{b}"].abs() <= L.DEV_CLIP)
    )
    return df[ok]


def design(d: pd.DataFrame, b: str, kind: str) -> np.ndarray:
    cols = [np.ones(len(d)), d[f"x_{b}"].to_numpy()]
    if kind == "partial":
        cols.append(d["g_avg"].to_numpy())
    return np.column_stack(cols)


def main(reg_id: str, record) -> None:  # noqa: ANN001  (record: run.record)
    reg = load(L.registry_dir() / f"{reg_id}.yaml")
    ledger = Ledger(L.ledger_path())
    panel = L.load_panel()
    ev_panel = panel[
        (panel["segment"] == "pinnacle") & (panel["match_date"] >= L.EVAL_START) & (panel["match_date"] <= L.EVAL_END)
    ]
    L.assert_open_only(ev_panel, "power")
    df = long_rows(ev_panel)
    df2 = long_rows(ev_panel)
    deterministic = bool(
        df.drop(columns=["decision_id", "match_id", "outcome", "division"]).equals(
            df2.drop(columns=["decision_id", "match_id", "outcome", "division"])
        )
    )
    out_dir = L.OUT / "h3"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"deterministic": deterministic, "books": {}}
    for config in reg.expand_grid():
        kind = config["slope_type"]
        print(f"{reg.id} {config}")
        # primary: contrast on common rows
        both = book_rows(book_rows(df, TOP), BOTTOM)
        c_top, v_top = coef_influence(design(both, TOP, kind), both["y"].to_numpy(), 1)
        c_bot, v_bot = coef_influence(design(both, BOTTOM, kind), both["y"].to_numpy(), 1)
        prim = both.assign(value=v_top - v_bot)
        ex_top = exact_coef_ci(
            design(both, TOP, kind), both["y"].to_numpy(), 1, both["match_date"].astype(str).to_numpy()
        )
        extra = {
            "clean": deterministic,
            "pit_assertion": "assert_open_only on the evaluation panel; x from open columns only, y and self are hindsight targets",
            "within_ceiling": bool(-1.0 <= c_top - c_bot <= 1.0),
            "contrast": {
                "top": TOP,
                "bottom": BOTTOM,
                "slope_top": c_top,
                "slope_bottom": c_bot,
                "exact_top": ex_top,
                "rows_both": len(both),
            },
        }
        record(reg, config, prim, "value", "slope_contrast", extra, ledger, id_col="decision_id")
        for b in BOOKS:
            d = book_rows(df, b)
            n_err = int((np.isfinite(df[f"x_{b}"]) & (df[f"x_{b}"].abs() > L.DEV_CLIP)).sum())
            if len(d) < 300:
                print(f"  {b}: n={len(d)} < 300, skipped")
                continue
            a = design(d, b, kind)
            y = d["y"].to_numpy()
            coef, v = coef_influence(a, y, 1)
            ex = exact_coef_ci(a, y, 1, d["match_date"].astype(str).to_numpy())
            seasons = {
                int(s): float(coef_influence(design(g, b, kind), g["y"].to_numpy(), 1)[0])
                for s, g in d.groupby("season_start")
                if len(g) >= 300
            }
            info = {
                "clean": deterministic,
                "exact": ex,
                "quote_errors_dropped": n_err,
                "by_season": seasons,
                "sd_x": float(d[f"x_{b}"].std()),
                "mean_abs_x": float(d[f"x_{b}"].abs().mean()),
            }
            record(reg, config, d.assign(value=v), "value", f"slope_{b}", info, ledger, id_col="decision_id")
            entry = summary["books"].setdefault(b, {})
            entry[kind] = {**ex, "by_season": seasons, "quote_errors_dropped": n_err}
            if kind == "marginal":
                ds = d[np.isfinite(d[f"self_{b}"])]
                if len(ds) >= 300:
                    a2 = design(ds, b, "marginal")
                    cs, vs = coef_influence(a2, ds[f"self_{b}"].to_numpy(), 1)
                    cb, vb = coef_influence(a2, ds["y"].to_numpy(), 1)
                    exs = exact_coef_ci(a2, ds[f"self_{b}"].to_numpy(), 1, ds["match_date"].astype(str).to_numpy())
                    gap = block_bootstrap(
                        vs - vb, ds["match_date"].astype(str).to_numpy(), n_boot=L.N_BOOT, seed=L.SEED
                    )
                    record(
                        reg,
                        config,
                        ds.assign(value=vs),
                        "value",
                        f"self_share_{b}",
                        {"clean": deterministic, "exact": exs},
                        ledger,
                        id_col="decision_id",
                    )
                    entry["self"] = {
                        **exs,
                        "pinnacle_share_same_rows": cb,
                        "persistence": 1 - cs - cb,
                        "self_minus_pinnacle": [gap.estimate, gap.lo, gap.hi],
                    }
    # BH across books x slope types (one-sided > 0, exact-bootstrap p)
    keys, ps = [], []
    for b, e in summary["books"].items():
        for kind in ("marginal", "partial"):
            if kind in e:
                keys.append((b, kind))
                ps.append(e[kind]["p_gt0"])
    rejected, qvals = benjamini_hochberg(np.array(ps), q=0.10)
    for (b, kind), q, rej in zip(keys, qvals, rejected, strict=True):
        summary["books"][b][kind]["bh_q"] = float(q)
        summary["books"][b][kind]["bh_reject"] = bool(rej)
    for e in summary["books"].values():
        informative = bool(e.get("partial", {}).get("bh_reject", False))
        s = e.get("self", {}).get("self_minus_pinnacle")
        stale = bool(s is not None and s[1] > 0 and not informative)
        e["label"] = "informative" if informative else ("stale" if stale else "mixed")
    (out_dir / "h3_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    rows = []
    for b, e in summary["books"].items():
        rows.append(
            {
                "book": b,
                "label": e["label"],
                "marginal": e.get("marginal", {}).get("coef"),
                "m_lo": e.get("marginal", {}).get("ci_lo"),
                "m_hi": e.get("marginal", {}).get("ci_hi"),
                "m_q": e.get("marginal", {}).get("bh_q"),
                "partial": e.get("partial", {}).get("coef"),
                "p_lo": e.get("partial", {}).get("ci_lo"),
                "p_hi": e.get("partial", {}).get("ci_hi"),
                "p_q": e.get("partial", {}).get("bh_q"),
                "self": e.get("self", {}).get("coef"),
                "self_lo": e.get("self", {}).get("ci_lo"),
                "self_hi": e.get("self", {}).get("ci_hi"),
                "persist": e.get("self", {}).get("persistence"),
                "n": e.get("marginal", {}).get("n"),
            }
        )
    tab = pd.DataFrame(rows).sort_values("partial", ascending=False)
    tab.to_csv(out_dir / "h3_ranking.csv", index=False)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
