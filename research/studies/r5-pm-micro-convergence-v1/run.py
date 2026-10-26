"""r5-pm-micro-convergence-v1: R1's convergence slope re-measured on Polymarket order-book mids.

Per (match, outcome) at ts = kickoff - T, with the football-data open known by ts:
  gap  = mid(ts) - q_open            (Yes mid of the book in force at ts minus the de-vigged sharp open)
  move = mid(last book before kickoff) - mid(ts)
Primary: OLS slope of move on gap, scored via per-row influence values (mean = slope) in run_trial; the exact
block-bootstrap slope CI is stored beside it. Secondaries on the same grid point: R1's last-trade slope on the
same rows, and the sharp open->close move on the mid gap.

Usage: arch -arm64 uv run python research/studies/r5-pm-micro-convergence-v1/run.py [--counts]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r5-common"))

from harness import exact_slope_ci, record, slope_values  # noqa: E402
from parta import CODE_FILES, R5, SIDES, leave_one_league_out, load, snapshot_tag, subperiods  # noqa: E402

REG_ID = "r5-pm-micro-convergence-v1"
MAX_AGE_T, MAX_AGE_LAST = 6.0, 3.0
R1_STALE_T, R1_STALE_LAST = 6.0, 3.0


def frame(df: pd.DataFrame, pbook, books, hours: float) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    recs = []
    for r in df.itertuples(index=False):
        ts = r.ko - pd.Timedelta(hours=hours)
        if not (r.open_known_at <= ts):
            continue
        for s in SIDES:
            qo = getattr(r, f"q_open_{s}")
            if not np.isfinite(qo):
                continue
            cid = getattr(r, f"cond_{s}")
            bt, bl = books.at(cid, ts), books.at(cid, r.ko)
            if not (books.usable(bt, MAX_AGE_T) and books.usable(bl, MAX_AGE_LAST)):
                continue
            assert bt["known_at_ns"] <= ts.value and bl["known_at_ns"] <= r.ko.value
            p, _, st, kn, _ = pbook.at(cid, ts)
            pl, _, stl, knl, _ = pbook.at(cid, r.ko)
            last_ok = np.isfinite(p) and np.isfinite(pl) and st <= R1_STALE_T and stl <= R1_STALE_LAST
            recs.append(
                {
                    "decision_id": f"{r.fd_match_key}|{s}",
                    "block": r.match_day,
                    "division": r.division,
                    "side": s,
                    "q_open": qo,
                    "q_close": getattr(r, f"q_bf_close_{s}"),
                    "mid_t": bt["mid"],
                    "mid_last": bl["mid"],
                    "same_book": bt["known_at_ns"] == bl["known_at_ns"],
                    "p_t": p if last_ok else np.nan,
                    "p_last": pl if last_ok else np.nan,
                }
            )
    out = pd.DataFrame(recs)
    if len(out):
        out["gap"] = out.mid_t - out.q_open
        out["move"] = out.mid_last - out.mid_t
    return out


def slope_on(fr: pd.DataFrame, m: np.ndarray) -> float:
    x, y = fr.gap.to_numpy()[m], fr.move.to_numpy()[m]
    xc = x - x.mean()
    return float(np.sum(xc * (y - y.mean())) / np.sum(xc * xc))


def main(counts_only: bool) -> None:
    df, pbook, books, info = load()
    snap = None if counts_only else snapshot_tag()
    detail: dict = {"info": info, "trials": {}}
    for hours in (24, 6, 2):
        fr = frame(df, pbook, books, float(hours))
        if counts_only:
            print(
                f"T-{hours}h rows {len(fr)}, days {fr.block.nunique() if len(fr) else 0}, with last-trade {int(fr.p_t.notna().sum()) if len(fr) else 0}"
            )
            continue
        fr = fr.reset_index(drop=True)
        days = fr.block.to_numpy()
        exact = exact_slope_ci(fr.gap.to_numpy(), fr.move.to_numpy(), days)
        extra = {
            "clean": True,
            "within_ceiling": bool(-1.2 <= exact["slope"] <= 0.5),
            "exact_block_bootstrap_slope": exact,
            "subperiod_estimates": subperiods(fr.gap.to_numpy(), days, lambda m: slope_on(fr, m)),  # noqa: B023 (called now)
            "leave_one_league_out": leave_one_league_out(fr.division.to_numpy(), lambda m: slope_on(fr, m)),  # noqa: B023 (called now)
            "share_same_book_at_t_and_last": float(fr.same_book.mean()),
            "mean_abs_gap": float(fr.gap.abs().mean()),
            "mean_spread_proxy_note": "mids from books with spread <= 0.05",
            "by_side": {s: slope_on(fr, (fr.side == s).to_numpy()) for s in SIDES if (fr.side == s).sum() > 2},
        }
        cfg = {"hours_before_kickoff": hours}
        v = slope_values(fr.gap.to_numpy(), fr.move.to_numpy())
        r = record(
            REG_ID,
            cfg,
            pd.DataFrame({"decision_id": fr.decision_id, "block": days, "value": v}),
            "ols_slope_move_on_gap_mid",
            snap,
            extra,
            CODE_FILES + [Path(__file__)],
        )
        lt = fr[fr.p_t.notna()].reset_index(drop=True)
        if len(lt) > 3:
            x, y = (lt.p_t - lt.q_open).to_numpy(), (lt.p_last - lt.p_t).to_numpy()
            ex = exact_slope_ci(x, y, lt.block.to_numpy())
            record(
                REG_ID,
                cfg,
                pd.DataFrame({"decision_id": lt.decision_id, "block": lt.block, "value": slope_values(x, y)}),
                "ols_slope_move_on_gap_lasttrade_same_rows",
                snap,
                {"clean": True, "exact_block_bootstrap_slope": ex},
                CODE_FILES + [Path(__file__)],
            )
        sh = fr[np.isfinite(fr.q_close)].reset_index(drop=True)
        if len(sh) > 3:
            x, y = sh.gap.to_numpy(), (sh.q_close - sh.q_open).to_numpy()
            ex = exact_slope_ci(x, y, sh.block.to_numpy())
            record(
                REG_ID,
                cfg,
                pd.DataFrame({"decision_id": sh.decision_id, "block": sh.block, "value": slope_values(x, y)}),
                "sharp_open_to_close_slope_on_gap_mid",
                snap,
                {"clean": True, "exact_block_bootstrap_slope": ex},
                CODE_FILES + [Path(__file__)],
            )
        detail["trials"][f"T-{hours}h"] = {
            "slope": None if r is None else r.metric_value,
            "n": len(fr),
            "n_lasttrade": len(lt),
            **extra,
        }
    if not counts_only:
        out = R5 / REG_ID
        out.mkdir(parents=True, exist_ok=True)
        (out / "detail.json").write_text(json.dumps(detail, indent=2, default=str))


if __name__ == "__main__":
    main("--counts" in sys.argv)
