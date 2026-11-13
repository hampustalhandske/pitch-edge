"""Coefficients for the DRAFT (unregistered) forward-only v2 strategy: a fair-close model restricted to inputs the
paper collector can observe live (the-odds-api): Pinnacle, the soft consensus, and the books that exist both in
football-data history and in the-odds-api (WH = williamhill, VC = betvictor, 1XB = onexbet, BF = betfair_sb_uk).

Per outcome o: move_o = a_o + b_o * g_avg_o + sum_b c_b,o * dev_b,o + sum_b m_b,o * miss_b, OLS on every
Pinnacle-segment row 2012/13-2024/25 (a missing book contributes dev = 0 and miss = 1). This is a descriptive fit
used to freeze numbers for a forward test; it is not a test and has no out-of-sample score here.
Output: data/research/r6/v2_live_coefficients.json. Usage: v2_coefficients.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r6lib as L  # noqa: E402

LIVE_BOOKS = {"WH": "williamhill", "VC": "betvictor", "1XB": "onexbet", "BF": "betfair_sb_uk"}


def main() -> None:
    panel = L.load_panel()
    pin = panel[panel["segment"] == "pinnacle"]
    L.assert_open_only(pin, "power")
    f, t = L.build_features(pin, "power"), L.targets(pin, "power")
    out: dict = {
        "rows": len(pin),
        "seasons": "2012/13-2024/25",
        "books": LIVE_BOOKS,
        "dev_clip": L.DEV_CLIP,
        "outcomes": {},
    }
    for k, o in enumerate(L.OUTCOMES):
        cols, names = [np.ones(len(pin)), f.base[k][:, 0]], ["intercept", "g_avg"]
        for b in LIVE_BOOKS:
            d = f.dev[b][:, k]
            ok = np.isfinite(d) & (np.abs(d) <= L.DEV_CLIP)
            cols += [np.where(ok, d, 0.0), (~ok).astype(float)]
            names += [f"dev_{b}", f"miss_{b}"]
        a = np.column_stack(cols)
        y = t["move"][:, k]
        fin = np.isfinite(y) & np.isfinite(a).all(axis=1)
        beta = np.linalg.lstsq(a[fin], y[fin], rcond=None)[0]
        out["outcomes"][o] = {n: float(v) for n, v in zip(names, beta, strict=True)}
        out["outcomes"][o]["n"] = int(fin.sum())
    (L.OUT / "v2_live_coefficients.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
