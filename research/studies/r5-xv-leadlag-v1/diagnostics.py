"""POST-HOC diagnostic for r5-xv-leadlag-v1 (not a registered trial; nothing is written to the ledger).

The close tests price Polymarket at the order-book mid in force at H*, but the archive holds about one
snapshot per market every four hours, so that mid can be hours older than the Kalshi quote. A stale
Polymarket mid would make Kalshi look more informative about the Betfair close for a reason unrelated to
price discovery. This re-estimates b (slope of q_close - PM_mid on Kalshi_mid - PM_mid) by the age of the
Polymarket book at H*, and with Polymarket priced at a fresh last trade in the hour ending at H* instead.
Writes data/research/r5/r5-xv-leadlag-v1/diagnostics.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "r5-common"))

from harness import exact_slope_ci  # noqa: E402
from partb import R5, panel, universe  # noqa: E402
from run import close_frame  # noqa: E402


def main() -> None:
    u, _ = universe()
    u = u[u.window == "evaluation"]
    pan = panel(u)
    cf = close_frame(pan, u)
    out = {
        "all": exact_slope_ci(cf.x.to_numpy(), cf.y.to_numpy(), cf.block.to_numpy()),
        "pm_book_age_quantiles_h": cf.b_age_h.quantile([0.25, 0.5, 0.75]).round(2).tolist(),
    }
    for lo, hi in ((0, 1), (1, 6), (6, 24)):
        m = ((cf.b_age_h > lo) | (lo == 0)) & (cf.b_age_h <= hi)
        s = cf[m]
        out[f"book_age_{lo}-{hi}h"] = (
            exact_slope_ci(s.x.to_numpy(), s.y.to_numpy(), s.block.to_numpy()) if len(s) > 5 else {"n": len(s)}
        )
    lt = cf[cf.p_fresh.notna()].copy()
    x, y = (lt.mid - lt.p_fresh).to_numpy(), (lt.q_close - lt.p_fresh).to_numpy()
    out["pm_fresh_last_trade_instead_of_book"] = (
        exact_slope_ci(x, y, lt.block.to_numpy()) if len(lt) > 5 else {"n": len(lt)}
    )
    out["note_last_trade"] = "last-trade bounce biases b upward; shown only as a bracket"
    dest = R5 / "r5-xv-leadlag-v1"
    (dest / "diagnostics.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
