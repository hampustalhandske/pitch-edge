"""Follow-up checks for the xg-finishing-v1 audit, built on the audit's own rederive.py (not the study's code).

- chronological thirds of the evaluation rows, with match-day block-bootstrap CIs;
- leave-one-division-out, with CIs (the registration's "a single division carries it" disconfirmer);
- snapshot-cutoff sensitivity: feature computed only from Understat matches dated before snapshot - 1 and
  before snapshot - 3 (protects against an earlier-than-assumed 'open' snapshot);
- a last-match-only feature (window 1, no shrink change), which would be large if the signal came from
  a match the open price had not yet seen.

Usage: uv run python research/audits/r3-xg/extra.py   (writes research/audits/r3-xg/out/extra.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rederive as A  # noqa: E402


def main() -> None:
    df = A.universe()
    us = A.load_us()
    ser = A.team_series(us, A.map_csm())
    feat, _ = A.feature(df, ser)
    b = A.prep(df, "power")
    r = A.walk_forward(df, b, feat)
    sub = df.iloc[r.i.to_numpy()].reset_index(drop=True)
    day = sub.match_date.dt.strftime("%Y%m%d").to_numpy()
    out: dict = {}
    order = np.argsort(sub.match_date.to_numpy(), kind="stable")
    thirds = np.array_split(order, 3)
    out["thirds"] = [
        {
            "from": str(sub.match_date.iloc[t].min().date()),
            "to": str(sub.match_date.iloc[t].max().date()),
            "n": len(t),
            "est_lo_hi_p": A.boot(r.d.to_numpy()[t], day[t]),
        }
        for t in thirds
    ]
    out["leave_one_division_out"] = {}
    top = sub.division.isin(A.TOP5).to_numpy()
    for dv in A.TOP5:
        m = (sub.division != dv).to_numpy() & top
        out["leave_one_division_out"][dv] = {"n": int(m.sum()), "est_lo_hi_p": A.boot(r.d.to_numpy()[m], day[m])}
    out["by_division_ci"] = {
        dv: A.boot(r.d.to_numpy()[(sub.division == dv).to_numpy()], day[(sub.division == dv).to_numpy()])
        for dv in A.TOP5
    }
    ok = np.isfinite(feat).all(1)[:, None]
    for shift in (1, 3):
        d2 = df.assign(snap=df.snap - pd.Timedelta(days=shift))
        f_s, _ = A.feature(d2, ser)
        f_s = np.where(ok, f_s, np.nan)
        out[f"cutoff_snapshot_minus_{shift}d"] = A.headline(df, A.walk_forward(df, b, f_s))
    f1, _ = A.feature(df, ser, window=1, shrink=0.0)
    f1 = np.where(ok, f1, np.nan)
    out["last_match_only"] = A.headline(df, A.walk_forward(df, b, f1))
    (A.OUT / "extra.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
