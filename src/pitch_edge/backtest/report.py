"""Markdown/JSON backtest report writer — includes the losing strategies on purpose."""

from __future__ import annotations

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


def model_league_performance(data_dir: str | Path, league_code: str) -> dict[str, dict] | None:
    """Read `data_dir/by_league.csv` (the local backtest data dir, `Settings.backtest_dir/<label>`),
    return {model_name: {edge_bits, log_loss, n}} for this league, or None if the file/league has
    no rows. Real walk-forward evidence for the model router
    (`agents/router.py::select_model_for_league`) — never a hardcoded model choice."""
    path = Path(data_dir) / "by_league.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["league_code"] == league_code]
    if df.empty:
        return None
    return {
        str(row["model"]): {
            "edge_bits": float(row["edge_bits"]),
            "log_loss": float(row["log_loss"]),
            "n": int(row["n"]),
        }
        for _, row in df.iterrows()
    }


def _merge_existing(out: Path, name: str, new: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Keep rows from earlier passes for models not in this run (e.g. GRU run in a second pass)."""
    path = out / name
    if not path.exists():
        return new
    old = pd.read_csv(path)
    if "model" in old.columns:
        old = old[~old["model"].isin(models)]
    return pd.concat([old, new], ignore_index=True) if not old.empty else new


def _overall_edge(calib: pd.DataFrame) -> pd.DataFrame:
    """One row per model: log-loss vs the market's, combined across all test predictions —
    the headline number. Positive `edge_bits` means the model beats the closing/market price."""
    import numpy as np

    if calib.empty:
        return pd.DataFrame(columns=["model", "n", "log_loss", "market_log_loss", "edge_bits"])
    df = calib.rename(columns={"multiclass_log_loss": "log_loss", "market_multiclass_log_loss": "market_log_loss"})
    df["edge_bits"] = (df["market_log_loss"] - df["log_loss"]) / np.log(2)
    return df[["model", "n_predictions", "log_loss", "market_log_loss", "edge_bits"]].rename(
        columns={"n_predictions": "n"}
    )


def _best_strategy(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per model: the staking strategy with the best CLV (the edge evidence), not the
    best ROI (noise over a few hundred bets)."""
    if summary.empty:
        return summary
    return summary.loc[summary.groupby("model")["mean_clv_pct"].idxmax()].reset_index(drop=True)


def write_report(
    results: dict[str, BacktestResult],
    data_dir: str | Path,
    report_dir: str | Path,
    title: str = "Walk-forward backtest",
    merge_existing: bool = True,
) -> Path:
    """Write the local, machine-readable CSVs (`data_dir` — used by the dashboard and the
    agentic model router, never committed) plus one public, one-page `CASE_STUDY.md`
    (`report_dir` — this is the only thing that belongs under `reports/`). No raw
    bets/predictions are written here — those already live in the warehouse (`backtest_bets`,
    read via `Warehouse`)."""
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)
    models = list(results)
    summary = results_table(results)
    calib = calibration_table(results)
    leagues = league_table(results)
    if merge_existing:
        summary = _merge_existing(data, "backtest_summary.csv", summary, models)
        calib = _merge_existing(data, "calibration.csv", calib, models)
        leagues = _merge_existing(data, "by_league.csv", leagues, models)
    summary.to_csv(data / "backtest_summary.csv", index=False)
    calib.to_csv(data / "calibration.csv", index=False)
    leagues.to_csv(data / "by_league.csv", index=False)

    any_cfg = next(iter(results.values())).config
    preamble = (
        "Walk-forward, no shuffling. Bets priced at early Pinnacle odds "
        f"(`{any_cfg.bet_prefix}`), CLV measured vs Pinnacle closing (`{any_cfg.closing_prefix}`), "
        f"edge threshold {any_cfg.edge_threshold:.0%}, retrain every {any_cfg.retrain_every_days} days."
    )
    (data / "preamble.txt").write_text(preamble)
    return render_report(data, report_dir, title, preamble)


def render_report(
    data_dir: str | Path, report_dir: str | Path, title: str = "Walk-forward backtest", preamble: str | None = None
) -> Path:
    """Render a one-page `CASE_STUDY.md` in `report_dir` from the CSVs in `data_dir` — headline
    edge vs market and best staking result per model, nothing per-division or per-strategy (that
    detail lives in the local CSVs, not in the public report). Lets multi-pass runs produce one
    combined report."""
    data = Path(data_dir)
    report = Path(report_dir)
    report.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(data / "backtest_summary.csv") if (data / "backtest_summary.csv").exists() else pd.DataFrame()
    calib = pd.read_csv(data / "calibration.csv") if (data / "calibration.csv").exists() else pd.DataFrame()
    if preamble is None:
        preamble = (data / "preamble.txt").read_text() if (data / "preamble.txt").exists() else ""
    headline = _overall_edge(calib)
    staking = _best_strategy(summary)
    keep_cols = ["model", "strategy", "n_bets", "mean_clv_pct", "roi", "sharpe"]
    staking = staking[[c for c in keep_cols if c in staking.columns]]
    lines = [
        f"# {title}",
        "",
        preamble,
        "",
        "## Model vs market (all divisions combined)",
        "",
        _md(headline.round(4)),
        "",
        "## Best staking result per model (ranked by CLV, not ROI)",
        "",
        _md(staking.round(4)),
        "",
        "_`mean_clv_pct` is the evidence of edge; `roi` over a few hundred bets is mostly noise. "
        "Negative `edge_bits` means the model is less informative than the closing price._",
    ]
    (report / "CASE_STUDY.md").write_text("\n".join(lines))
    return report / "CASE_STUDY.md"


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no rows_"
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])
