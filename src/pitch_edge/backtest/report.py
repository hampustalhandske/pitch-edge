"""Markdown/JSON backtest report writer — includes the losing strategies on purpose."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pitch_edge.backtest.engine import BacktestResult


def results_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    return pd.concat([r.summaries for r in results.values()], ignore_index=True)


def calibration_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = [{"model": name, **r.calibration} for name, r in results.items()]
    return pd.DataFrame(rows)


def league_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Per division: model log-loss vs no-vig market log-loss. Positive `edge_bits` means the
    model is more informative than the market price in that division — the information-asymmetry test."""
    import numpy as np

    rows = []
    for name, r in results.items():
        p = r.predictions
        if "league_code" not in p.columns:
            continue
        for code, g in p.groupby("league_code"):
            y = g["result"].to_numpy().astype(int)
            P = g[["p_home", "p_draw", "p_away"]].to_numpy()
            M = g[["mkt_home", "mkt_draw", "mkt_away"]].to_numpy()
            ok = ~np.isnan(M).any(axis=1)
            if ok.sum() < 50:
                continue
            ll = -np.mean(np.log(np.clip(P[ok][np.arange(ok.sum()), y[ok]], 1e-12, 1)))
            mll = -np.mean(np.log(np.clip(M[ok][np.arange(ok.sum()), y[ok]], 1e-12, 1)))
            rows.append(
                {
                    "model": name,
                    "league_code": code,
                    "n": int(ok.sum()),
                    "log_loss": ll,
                    "market_log_loss": mll,
                    "edge_bits": (mll - ll) / np.log(2),
                }
            )
    return (
        pd.DataFrame(rows).sort_values(["model", "edge_bits"], ascending=[True, False])
        if rows
        else pd.DataFrame(columns=["model", "league_code", "n", "log_loss", "market_log_loss", "edge_bits"])
    )


def _merge_existing(out: Path, name: str, new: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Keep rows from earlier passes for models not in this run (e.g. GRU run in a second pass)."""
    path = out / name
    if not path.exists():
        return new
    old = pd.read_csv(path)
    if "model" in old.columns:
        old = old[~old["model"].isin(models)]
    return pd.concat([old, new], ignore_index=True) if not old.empty else new


def write_report(
    results: dict[str, BacktestResult],
    out_dir: str | Path,
    title: str = "Walk-forward backtest",
    merge_existing: bool = True,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = list(results)
    summary = results_table(results)
    calib = calibration_table(results)
    leagues = league_table(results)
    if merge_existing:
        summary = _merge_existing(out, "backtest_summary.csv", summary, models)
        calib = _merge_existing(out, "calibration.csv", calib, models)
        leagues = _merge_existing(out, "by_league.csv", leagues, models)
    summary.to_csv(out / "backtest_summary.csv", index=False)
    calib.to_csv(out / "calibration.csv", index=False)
    leagues.to_csv(out / "by_league.csv", index=False)
    for name, r in results.items():
        r.bets.to_parquet(out / f"bets_{name}.parquet", index=False)
        r.predictions.to_parquet(out / f"predictions_{name}.parquet", index=False)

    any_cfg = next(iter(results.values())).config
    preamble = (
        "Walk-forward, no shuffling. Bets priced at early Pinnacle odds "
        f"(`{any_cfg.bet_prefix}`), CLV measured vs Pinnacle closing (`{any_cfg.closing_prefix}`), "
        f"edge threshold {any_cfg.edge_threshold:.0%}, retrain every {any_cfg.retrain_every_days} days."
    )
    (out / "preamble.txt").write_text(preamble)
    return render_report(out, title, preamble)


def render_report(out_dir: str | Path, title: str = "Walk-forward backtest", preamble: str | None = None) -> Path:
    """Render REPORT.md from the CSVs in `out_dir` (lets multi-pass runs produce one combined report)."""
    out = Path(out_dir)
    summary = pd.read_csv(out / "backtest_summary.csv") if (out / "backtest_summary.csv").exists() else pd.DataFrame()
    calib = pd.read_csv(out / "calibration.csv") if (out / "calibration.csv").exists() else pd.DataFrame()
    leagues = pd.read_csv(out / "by_league.csv") if (out / "by_league.csv").exists() else pd.DataFrame()
    if preamble is None:
        preamble = (out / "preamble.txt").read_text() if (out / "preamble.txt").exists() else ""
    lines = [f"# {title}", ""]
    lines += [
        preamble,
        "",
        "## Calibration (all test predictions)",
        "",
        _md(calib.round(4)),
        "",
        "## Model vs market by division (where is the information gap?)",
        "",
        _md(leagues.round(4)),
        "",
        "## Staking results (including the losers)",
        "",
        _md(summary.round(4)),
        "",
        "## Reading this honestly",
        "",
        "- `mean_clv_pct` and `clv_t_stat` are the evidence of edge; `roi` over a few hundred bets is mostly noise.",
        "- A model whose `multiclass_log_loss` is worse than `market_multiclass_log_loss` is *less* informative than the closing line on its own.",
        "- Fractional Kelly vs flat: Kelly compounds edge *and* error — compare drawdowns, not just final bankroll.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    (out / "summary.json").write_text(
        json.dumps(
            {
                "summaries": summary.to_dict(orient="records"),
                "calibration": calib.to_dict(orient="records"),
            },
            default=str,
            indent=2,
        )
    )
    return out / "REPORT.md"


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no rows_"
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])
