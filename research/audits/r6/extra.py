"""Extra checks: (1) H2 res_tiny re-derived with the auditor's features + sklearn HGB (registered hyper-parameters);
(2) drop-VC / VC-only ablations of H1 ridge vs the era-refit baseline; (3) BW suppressor check with level controls;
(4) Δ by snapshot->kickoff horizon for the auditor's ridge vs the era baseline.

Usage: OMP_NUM_THREADS=1 arch -arm64 uv run python research/audits/r6/extra.py
"""

from __future__ import annotations

import json

import alib as A
import numpy as np
import pandas as pd
from rederive import walk
from sklearn.ensemble import HistGradientBoostingRegressor
from timing import coef_ci, long


def main() -> None:
    raw = A.raw()
    d = A.universe(raw)
    r = A.core_frame(d)
    ss = d.ss.to_numpy()
    res: dict = {}
    days = lambda w: w.match_date.dt.strftime("%Y-%m-%d").to_numpy()  # noqa: E731

    # (1) H2 res_tiny
    out = []
    for s in A.EVAL_SEASONS:
        tr, te = np.flatnonzero(ss < s), np.flatnonzero(ss == s)
        mb, mm = np.zeros((len(te), 3)), np.zeros((len(te), 3))
        for k in range(3):
            base, book, names = A.design(r, k)
            y = r["move"][:, k]
            mb[:, k] = A.ols(base[tr], y[tr], base[te])
            btr, bte = A.centre_over(book[tr], book[te], names)
            xtr, xte = np.column_stack([base[tr], btr]), np.column_stack([base[te], bte])
            ptr, _ = A.ridge(xtr, y[tr], xtr, ss[tr])
            pte, _ = A.ridge(xtr, y[tr], xte, ss[tr])
            h = HistGradientBoostingRegressor(
                loss="squared_error",
                early_stopping=False,
                random_state=20261020,
                max_iter=150,
                learning_rate=0.03,
                max_leaf_nodes=7,
                min_samples_leaf=1000,
                l2_regularization=10.0,
            )
            h.fit(np.column_stack([xtr, r["lpo"][tr, k]]), y[tr] - ptr)
            mm[:, k] = pte + h.predict(np.column_stack([xte, r["lpo"][te, k]]))
        po, pc = r["po"][te], r["pc"][te]
        out.append(
            pd.DataFrame(
                {
                    "mid": d.mid.to_numpy()[te],
                    "match_date": d.match_date.to_numpy()[te],
                    "delta": A.kl(pc, A.phat(po, mb)) - A.kl(pc, A.phat(po, mm)),
                }
            )
        )
    w = pd.concat(out, ignore_index=True)
    res["H2_res_tiny"] = A.boot(w.delta.to_numpy(), days(w))
    st = pd.read_parquet(A.REPO / "data/research/r6/closing-line-prediction-cross-book-gbm-v1__hgbres_tiny.parquet")
    st = st[st.segment == "pinnacle"].set_index("match_id")
    res["H2_res_tiny_max_abs_diff_vs_study"] = float(
        (w.set_index("mid").delta - st.loc[w.mid, "delta_kl_bits"].to_numpy()).abs().max()
    )
    print("H2 res_tiny", A.fmt(res["H2_res_tiny"]), res["H2_res_tiny_max_abs_diff_vs_study"])

    # (2) ablations vs era baseline
    era = (ss >= 2019).astype(int)

    def era_base(tr, te, k):
        base = A.design(r, k)[0]
        y = r["move"][:, k]
        same = tr[era[tr] == era[te][0]]
        use = same if len(same) >= 5000 else tr
        return A.ols(base[use], y[use], base[te])

    for lab, books in (
        ("all", None),
        ("drop_VC", ["VC"]),
        ("VC_only", ["B365", "BW", "IW", "WH"]),
        ("none", list(A.CORE)),
    ):
        dd = raw.copy()
        for b in books or []:
            for s_ in "HDA":
                dd[b + s_] = np.nan
        d2 = A.universe(dd)
        r2 = A.core_frame(d2)
        w2 = walk(d2, r2, "ridge", base_fn=None)
        era2 = (d2.ss.to_numpy() >= 2019).astype(int)

        def eb(tr, te, k, r2=r2, era2=era2):
            base = A.design(r2, k)[0]
            y = r2["move"][:, k]
            same = tr[era2[tr] == era2[te][0]]
            use = same if len(same) >= 5000 else tr
            return A.ols(base[use], y[use], base[te])

        w3 = walk(d2, r2, "ridge", base_fn=eb)
        res[f"ablation_{lab}"] = {
            "vs_R3": A.boot(w2.delta.to_numpy(), days(w2)),
            "vs_era": A.boot(w3.delta.to_numpy(), days(w3)),
            "vs_era_by_season": w3.groupby("ss").delta.mean().round(7).to_dict(),
        }
        print(
            lab,
            "vs R3",
            A.fmt(res[f"ablation_{lab}"]["vs_R3"]),
            "vs era",
            A.fmt(res[f"ablation_{lab}"]["vs_era"]),
            res[f"ablation_{lab}"]["vs_era_by_season"],
        )

    # (3) BW suppressor with level controls (evaluation rows, all five named present)
    f = long(d, r)
    ev = f[f.ss.between(2019, 2024)]
    m = np.all([np.isfinite(ev["x_" + b]) & (ev["x_" + b].abs() <= 0.5) for b in A.CORE], axis=0) & np.isfinite(ev.y)
    a5 = ev[m]
    lpo = np.concatenate([r["lpo"][:, k] for k in range(3)])[f.ss.between(2019, 2024).to_numpy()][m]
    one = np.ones(len(a5))
    od = pd.get_dummies(a5.o).to_numpy(float)[:, 1:]
    others = a5[["x_B365", "x_IW", "x_WH", "x_VC"]].to_numpy()
    X = np.column_stack([one, a5.x_BW, others, a5.g_avg, lpo, lpo**2, od])
    res["BW_partial_with_level_controls"] = coef_ci(X, a5.y.to_numpy(), a5.date.to_numpy(), 1)
    # does BW's deviation load on the common component of the other books' devs?
    res["corr_BW_with_mean_others"] = float(np.corrcoef(a5.x_BW, others.mean(1))[0, 1])
    print("BW partial with level controls", res["BW_partial_with_level_controls"][:3], res["corr_BW_with_mean_others"])

    # (4) horizon split of the ridge gain vs the era baseline
    w4 = walk(d, r, "ridge", base_fn=era_base)
    hrs = pd.read_parquet(A.OUT / "hours.parquet").drop_duplicates("mid").set_index("mid").hrs
    w4["hrs"] = hrs.reindex(w4.mid).to_numpy()
    w4["hb"] = pd.cut(w4.hrs, [-24, 8, 30, 54, 100], labels=["same-day", "next-day", "2-days", "3-days"]).astype(str)
    res["ridge_vs_era_by_horizon"] = {
        h: [len(g)] + list(A.boot(g.delta.to_numpy(), days(g))) for h, g in w4.groupby("hb")
    }
    print(json.dumps(res["ridge_vs_era_by_horizon"], indent=0))
    (A.REPO / "research/audits/r6/out/extra.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
