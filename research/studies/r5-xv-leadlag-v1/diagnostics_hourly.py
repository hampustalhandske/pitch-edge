"""POST-HOC diagnostic for r5-xv-leadlag-v1 hourly tests (not registered trials; no ledger writes).

Non-synchronous sampling: the Kalshi mid is the quote at the end of hour H, the Polymarket price is the last
trade inside the hour, possibly well before H. That alone makes Kalshi's move in hour H "predict" Polymarket's
move in hour H + 1. If the measured lead is mostly this, it should shrink when Polymarket traded many times in
hour H (its last trade then sits near the end of the hour). Splits both hourly slopes by Polymarket's trade
count in hour H. Writes data/research/r5/r5-xv-leadlag-v1/diagnostics_hourly.json.
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
from run import hourly_frame  # noqa: E402


def main() -> None:
    u, _ = universe()
    u = u[u.window == "evaluation"]
    pan = panel(u, with_book=False)
    hf = hourly_frame(pan)
    nt = pan.set_index(["event_ticker", "side", "H"]).n_trades
    hf["n_trades_H"] = [nt.get((a, b, c)) for a, b, c in zip(hf.event_ticker, hf.side, hf.H, strict=True)]
    out = {}
    for name, m in {
        "n_trades_1": hf.n_trades_H == 1,
        "n_trades_2_4": hf.n_trades_H.between(2, 4),
        "n_trades_5_19": hf.n_trades_H.between(5, 19),
        "n_trades_20plus": hf.n_trades_H >= 20,
    }.items():
        s = hf[m]
        out[name] = {
            "kalshi_leads_pm": exact_slope_ci(s.dK.to_numpy(), s.dP_next.to_numpy(), s.block.to_numpy(), n_boot=2000),
            "pm_leads_kalshi": exact_slope_ci(s.dP.to_numpy(), s.dK_next.to_numpy(), s.block.to_numpy(), n_boot=2000),
            "contemporaneous": exact_slope_ci(s.dK.to_numpy(), s.dP.to_numpy(), s.block.to_numpy(), n_boot=2000),
        }
    (R5 / "r5-xv-leadlag-v1" / "diagnostics_hourly.json").write_text(json.dumps(out, indent=2, default=str))
    for k, v in out.items():
        print(
            k,
            {
                kk: (round(vv["slope"], 3), round(vv["ci_lo"], 3), round(vv["ci_hi"], 3), vv["n"])
                for kk, vv in v.items()
            },
        )


if __name__ == "__main__":
    main()
