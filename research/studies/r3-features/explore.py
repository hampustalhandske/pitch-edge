"""EXPLORATORY (pre-registration): walk-forward scores of each R3 feature group on seasons whose
predictions fall entirely before the evaluation window (match dates <= 2019-06-30; training data is
earlier still). Used only to check the code and the feature definitions; not a counted trial.

Writes data/research/r3/explore_summary.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3-common"))

import r3lib as R  # noqa: E402

EXPLORE_SEASONS = {
    "pi": list(range(2013, 2019)),
    "xg": list(range(2015, 2019)),
    "manager": list(range(2013, 2019)),
    "schedule": list(range(2013, 2019)),
}


def main() -> None:
    panel = R.load_panel()
    rows = []
    for g, seasons in EXPLORE_SEASONS.items():
        s = R.walk_forward(panel, R.GROUPS[g], "power", seasons)
        assert s["match_date"].max() <= R.EXPLORATORY_END
        out = R.summary(s, f"{g} power exploratory")
        rows.append(out)
        print(
            f"{g:9s} n={out['n']:6d} dKL(bits) {R.fmt(out['delta_kl_bits'], 1, 3)}  dR2 {R.fmt(out['delta_r2_move'], 100, 3)}%"
            f"  dLLres {R.fmt(out['delta_ll_result_bits'], 1, 3)}  R2 base {out['r2_base']:.4f} model {out['r2_model']:.4f}"
            f"  KL open/base/model {out['kl_open']:.5f}/{out['kl_base']:.5f}/{out['kl_model']:.5f}"
        )
    (R.D / "explore_summary.json").write_text(R.to_json(rows))


if __name__ == "__main__":
    main()
