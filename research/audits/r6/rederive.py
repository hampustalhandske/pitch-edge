"""Independent re-derivation of H1 (closing-line-prediction-cross-book-linear-v1): OLS and ridge Δ KL bits vs
the R3 baseline, season walk-forward, evaluation 2019/20-2024/25. Compares per match with the study's parquet.

Usage: OMP_NUM_THREADS=1 arch -arm64 uv run python research/audits/r6/rederive.py
Writes data/research/r6_audit/rederive_{ols,ridge}.parquet and research/audits/r6/out/rederive.json.
"""

from __future__ import annotations

import json

import alib as A
import numpy as np
import pandas as pd


def walk(d: pd.DataFrame, r: dict, model: str, seasons=A.EVAL_SEASONS, base_fn=None, book_tf=None) -> pd.DataFrame:
    """Season walk-forward. base_fn(train_idx, test_idx, k) may override the baseline move prediction;
    book_tf(xtr, xte, tr, te, k) may transform the cross-book block (placebos)."""
    ss = d.ss.to_numpy()
    out = []
    for s in seasons:
        tr, te = np.flatnonzero(ss < s), np.flatnonzero(ss == s)
        mb, mm = np.zeros((len(te), 3)), np.zeros((len(te), 3))
        lams = []
        for k in range(3):
            base, book, names = A.design(r, k)
            y = r["move"][:, k]
            mb[:, k] = A.ols(base[tr], y[tr], base[te]) if base_fn is None else base_fn(tr, te, k)
            btr, bte = A.centre_over(book[tr], book[te], names)
            if book_tf is not None:
                btr, bte = book_tf(btr, bte, tr, te, k)
            xtr, xte = np.column_stack([base[tr], btr]), np.column_stack([base[te], bte])
            if model == "ols":
                mm[:, k] = A.ols(xtr, y[tr], xte)
            elif model == "ridge":
                mm[:, k], lam = A.ridge(xtr, y[tr], xte, ss[tr])
                lams.append(lam)
            elif model == "base":
                mm[:, k] = mb[:, k]
        po, pc = r["po"][te], r["pc"][te]
        o = pd.DataFrame(
            {
                "mid": d.mid.to_numpy()[te],
                "match_date": d.match_date.to_numpy()[te],
                "division": d.division.to_numpy()[te],
                "ss": s,
                "kl_open": A.kl(pc, po),
                "kl_base": A.kl(pc, A.phat(po, mb)),
                "kl_model": A.kl(pc, A.phat(po, mm)),
            }
        )
        o["delta"] = o.kl_base - o.kl_model
        o["lams"] = str(lams)
        out.append(o)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    d = A.universe(A.raw())
    r = A.core_frame(d)
    res = {"n_universe": len(d)}
    study = {
        "ols": "closing-line-prediction-cross-book-linear-v1__ols",
        "ridge": "closing-line-prediction-cross-book-linear-v1__ridge",
    }
    for m in ("ols", "ridge"):
        w = walk(d, r, m)
        w.to_parquet(A.OUT / f"rederive_{m}.parquet")
        days = w.match_date.dt.strftime("%Y-%m-%d").to_numpy()
        st = pd.read_parquet(A.REPO / "data" / "research" / "r6" / f"{study[m]}.parquet")
        st = st[st.segment == "pinnacle"]
        j = w.merge(
            st[["match_id", "kl_base", "kl_model", "delta_kl_bits"]],
            left_on="mid",
            right_on="match_id",
            how="outer",
            indicator=True,
        )
        both = j[j._merge == "both"]
        res[m] = {
            "n": len(w),
            "delta": A.boot(w.delta.to_numpy(), days),
            "kl_open": w.kl_open.mean(),
            "gain_vs_open": float((w.kl_open - w.kl_model).mean()),
            "by_season": w.groupby("ss").delta.mean().to_dict(),
            "lambdas": sorted(set(w.lams)) if m == "ridge" else None,
            "study_n": len(st),
            "study_delta": float(st.delta_kl_bits.mean()),
            "only_mine": int((j._merge == "left_only").sum()),
            "only_study": int((j._merge == "right_only").sum()),
            "max_abs_diff_kl_model": float((both.kl_model_x - both.kl_model_y).abs().max()),
            "max_abs_diff_kl_base": float((both.kl_base_x - both.kl_base_y).abs().max()),
        }
        print(m, json.dumps(res[m], default=str, indent=1))
    (A.REPO / "research" / "audits" / "r6" / "out").mkdir(exist_ok=True)
    (A.REPO / "research" / "audits" / "r6" / "out" / "rederive.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
