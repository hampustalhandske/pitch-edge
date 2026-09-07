"""Walk-forward backtest engine for any `MatchModel`, any staking strategy.

Rules that never bend:
* Train on matches strictly before the fold's test window, roll forward
  by calendar (default: retrain every 30 days). No shuffling across time.
* Bets are priced at the *early* quoted odds (`bet_prefix`, default Pinnacle
  `PS`), never at fair/no-vig odds, so the vig is paid on every bet.
* CLV is measured against the *closing* line (`closing_prefix`, default
  Pinnacle closing `PSC`); a fixture with no closing price is not bet.
* Calibration is fitted on realised past folds only (isotonic per outcome).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pitch_edge.backtest.kelly import fractional_kelly_stake
from pitch_edge.backtest.metrics import summarize_backtest
from pitch_edge.models.base import OUTCOMES, MatchModel
from pitch_edge.models.calibration import IsotonicCalibrator, brier_score, log_loss_score

logger = logging.getLogger(__name__)

SIDE = {"home": "H", "draw": "D", "away": "A"}


@dataclass
class StakingStrategy:
    name: str = "kelly_quarter"
    kind: str = "kelly"  # "kelly" | "flat"
    fraction: float = 0.25
    max_stake_pct: float = 0.03
    flat_stake_pct: float = 0.01

    def stake(self, p: float, odds: float, bankroll: float) -> float:
        if self.kind == "flat":
            return bankroll * self.flat_stake_pct
        return fractional_kelly_stake(p, odds, bankroll, fraction=self.fraction, max_stake_pct=self.max_stake_pct)


DEFAULT_STRATEGIES = [
    StakingStrategy("kelly_quarter", "kelly", 0.25, 0.03),
    StakingStrategy("kelly_half", "kelly", 0.5, 0.05),
    StakingStrategy("flat_1pct", "flat", flat_stake_pct=0.01),
]


@dataclass
class WalkForwardConfig:
    min_train_matches: int = 400
    retrain_every_days: int = 30
    edge_threshold: float = 0.03
    min_odds: float = 1.2
    max_odds: float = 12.0
    starting_bankroll: float = 1000.0
    bet_prefix: str = "PS"
    closing_prefix: str = "PSC"
    calibrate: bool = True
    calibration_min_rows: int = 300
    strategies: list[StakingStrategy] = field(default_factory=lambda: list(DEFAULT_STRATEGIES))
    max_bets_per_match: int = 1  # bet only the best edge per match


@dataclass
class BacktestResult:
    model_name: str
    predictions: pd.DataFrame  # every test match: probs, market probs, result
    bets: pd.DataFrame  # one row per (strategy, bet)
    summaries: pd.DataFrame  # one row per strategy
    calibration: dict[str, float]
    config: WalkForwardConfig


class WalkForwardBacktester:
    def __init__(self, config: WalkForwardConfig | None = None):
        self.config = config or WalkForwardConfig()
        self.bet_price_source = self.config.bet_prefix
        self.closing_price_source = self.config.closing_prefix

    # ------------------------------------------------------------- helpers
    def _odds_cols(self, prefix: str) -> dict[str, str]:
        return {o: f"{prefix}{SIDE[o]}" for o in OUTCOMES}

    def _prepare(self, features: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df = features.sort_values(["date", "match_id"]).reset_index(drop=True).copy()
        df["date"] = pd.to_datetime(df["date"])
        bet = self._odds_cols(cfg.bet_prefix)
        close = self._odds_cols(cfg.closing_prefix)
        for col in list(bet.values()) + list(close.values()):
            if col not in df.columns:
                df[col] = np.nan
        # bet-price fallback: no Pinnacle early prices (e.g. xgabora-only spine) -> market-average price (Mkt*/Avg*)
        if df[list(bet.values())].isna().all().all():
            for alt in ("Mkt", "Avg", "B365"):
                alt_cols = self._odds_cols(alt)
                if all(c in df.columns for c in alt_cols.values()) and df[list(alt_cols.values())].notna().any().any():
                    logger.warning("No %s* bet odds — falling back to %s* prices", cfg.bet_prefix, alt)
                    for o in OUTCOMES:
                        df[bet[o]] = df[alt_cols[o]]
                    self.bet_price_source = alt
                    break
        else:
            self.bet_price_source = cfg.bet_prefix
        # closing fallback: if no separate closing column exists at all, use the bet price (CLV then = 0, flagged)
        if df[list(close.values())].isna().all().all():
            self.closing_price_source = "bet_price_no_closing_available"
            logger.warning("No closing odds (%s*) available — CLV will be zero by construction", cfg.closing_prefix)
            for o in OUTCOMES:
                df[close[o]] = df[bet[o]]
        return df

    # ----------------------------------------------------------------- run
    def run(self, features: pd.DataFrame, model: MatchModel) -> BacktestResult:
        cfg = self.config
        df = self._prepare(features)
        bet_cols, close_cols = self._odds_cols(cfg.bet_prefix), self._odds_cols(cfg.closing_prefix)
        if len(df) <= cfg.min_train_matches:
            raise ValueError(f"Not enough matches ({len(df)}) for min_train_matches={cfg.min_train_matches}")

        start_date = df.loc[cfg.min_train_matches, "date"]
        fold_starts = pd.date_range(start_date, df["date"].max(), freq=f"{cfg.retrain_every_days}D")
        preds: list[pd.DataFrame] = []
        calibrators: dict[str, IsotonicCalibrator] | None = None
        realised_probs: list[pd.DataFrame] = []

        for i, fold_start in enumerate(fold_starts):
            fold_end = fold_starts[i + 1] if i + 1 < len(fold_starts) else df["date"].max() + pd.Timedelta(days=1)
            train = df[df["date"] < fold_start]
            test = df[(df["date"] >= fold_start) & (df["date"] < fold_end)]
            if test.empty or len(train) < cfg.min_train_matches:
                continue
            model.fit(train)
            raw = model.predict_proba(test)
            if raw.isna().any().any() or not np.isfinite(raw.to_numpy()).all():
                logger.warning(
                    "%s produced non-finite probabilities in fold %d; replacing with uniform", model.name, i + 1
                )
                bad = ~np.isfinite(raw.to_numpy()).all(axis=1)
                raw.loc[bad, list(OUTCOMES)] = 1.0 / 3.0
            if i % 10 == 0:
                logger.info(
                    "%s fold %d/%d: train=%d test=%d", model.name, i + 1, len(fold_starts), len(train), len(test)
                )
            out = (
                test[
                    ["match_id", "date", "league_code", "home_team", "away_team", "home_goals", "away_goals", "result"]
                ].copy()
                if "league_code" in test
                else test[["match_id", "date", "home_team", "away_team", "home_goals", "away_goals", "result"]].copy()
            )
            for o in OUTCOMES:
                out[f"raw_{o}"] = raw[o].to_numpy()
                out[f"bet_odds_{o}"] = test[bet_cols[o]].to_numpy()
                out[f"close_odds_{o}"] = test[close_cols[o]].to_numpy()
            # calibration on realised past folds only
            if cfg.calibrate and calibrators is not None:
                cal = np.column_stack([calibrators[o].transform(out[f"raw_{o}"].to_numpy()) for o in OUTCOMES])
                cal = cal / cal.sum(axis=1, keepdims=True)
            else:
                cal = out[[f"raw_{o}" for o in OUTCOMES]].to_numpy()
            for j, o in enumerate(OUTCOMES):
                out[f"p_{o}"] = cal[:, j]
            preds.append(out)
            realised_probs.append(out)
            hist = pd.concat(realised_probs)
            if cfg.calibrate and len(hist) >= cfg.calibration_min_rows:
                calibrators = {
                    o: IsotonicCalibrator().fit(
                        hist[f"raw_{o}"].to_numpy(), (hist["result"] == k).astype(int).to_numpy()
                    )
                    for k, o in enumerate(OUTCOMES)
                }
            model.observe(test)

        if not preds:
            raise ValueError("Walk-forward produced no test folds")
        predictions = pd.concat(preds, ignore_index=True)
        predictions = self._market_probs(predictions)
        bets = self._place_bets(predictions)
        summaries = self._summaries(bets, model.name)
        calibration = self._calibration(predictions)
        return BacktestResult(model.name, predictions, bets, summaries, calibration, cfg)

    # -------------------------------------------------------------- market
    def _market_probs(self, preds: pd.DataFrame) -> pd.DataFrame:
        cols = [f"bet_odds_{o}" for o in OUTCOMES]
        ok = preds[cols].notna().all(axis=1) & (preds[cols] > 1).all(axis=1)
        mp = np.full((len(preds), 3), np.nan)
        inv = 1.0 / preds.loc[ok, cols].to_numpy(dtype=float)
        mp[ok.to_numpy()] = inv / inv.sum(axis=1, keepdims=True)
        for j, o in enumerate(OUTCOMES):
            preds[f"mkt_{o}"] = mp[:, j]
            preds[f"edge_{o}"] = preds[f"p_{o}"] - preds[f"mkt_{o}"]
        return preds

    # ---------------------------------------------------------------- bets
    def _place_bets(self, preds: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        records = []
        for strat in cfg.strategies:
            bankroll = cfg.starting_bankroll
            for _, r in preds.iterrows():
                candidates = []
                for k, o in enumerate(OUTCOMES):
                    edge, odds, close = r[f"edge_{o}"], r[f"bet_odds_{o}"], r[f"close_odds_{o}"]
                    if np.isnan(edge) or np.isnan(odds) or np.isnan(close):
                        continue
                    if edge > cfg.edge_threshold and cfg.min_odds <= odds <= cfg.max_odds:
                        candidates.append((edge, k, o, odds, close))
                candidates.sort(reverse=True)
                for edge, k, o, odds, close in candidates[: cfg.max_bets_per_match]:
                    p = float(np.clip(r[f"p_{o}"], 1e-4, 1 - 1e-4))
                    stake = strat.stake(p, odds, bankroll)
                    if stake <= 0:
                        continue
                    won = int(r["result"] == k)
                    profit = stake * (odds - 1) if won else -stake
                    bankroll += profit
                    records.append(
                        {
                            "strategy": strat.name,
                            "match_id": r["match_id"],
                            "date": r["date"],
                            "league_code": r.get("league_code"),
                            "home_team": r["home_team"],
                            "away_team": r["away_team"],
                            "outcome": o,
                            "model_probability": p,
                            "market_probability": r[f"mkt_{o}"],
                            "edge": edge,
                            "bet_odds": odds,
                            "closing_odds": close,
                            "stake": stake,
                            "won": won,
                            "profit": profit,
                            "bankroll_after": bankroll,
                        }
                    )
        return pd.DataFrame.from_records(records)

    def _summaries(self, bets: pd.DataFrame, model_name: str) -> pd.DataFrame:
        rows = []
        for strat in self.config.strategies:
            sub = bets[bets["strategy"] == strat.name] if not bets.empty else bets
            s = (
                summarize_backtest(sub)
                if not sub.empty
                else summarize_backtest(pd.DataFrame(columns=["stake", "profit", "bet_odds", "closing_odds", "won"]))
            )
            if not sub.empty:
                path = np.concatenate([[self.config.starting_bankroll], sub["bankroll_after"].to_numpy()])
                s["max_drawdown"] = float(((np.maximum.accumulate(path) - path) / np.maximum.accumulate(path)).max())
                s["final_bankroll"] = float(path[-1])
                clv = sub["closing_odds"].to_numpy() ** -1 / sub["bet_odds"].to_numpy() ** -1 - 1
                s["clv_positive_share"] = float((clv > 0).mean())
                s["clv_t_stat"] = (
                    float(clv.mean() / (clv.std(ddof=1) / np.sqrt(len(clv))))
                    if len(clv) > 2 and clv.std(ddof=1) > 0
                    else 0.0
                )
            else:
                s.update(
                    {
                        "max_drawdown": 0.0,
                        "final_bankroll": self.config.starting_bankroll,
                        "clv_positive_share": 0.0,
                        "clv_t_stat": 0.0,
                    }
                )
            rows.append(
                {
                    "model": model_name,
                    "strategy": strat.name,
                    **s,
                    "bet_price_source": self.bet_price_source,
                    "closing_price_source": self.closing_price_source,
                }
            )
        return pd.DataFrame(rows)

    def _calibration(self, preds: pd.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        y = preds["result"].to_numpy()
        P = preds[[f"p_{o}" for o in OUTCOMES]].to_numpy()
        onehot = np.eye(3)[y]
        out["multiclass_brier"] = float(np.mean(np.sum((P - onehot) ** 2, axis=1)))
        out["multiclass_log_loss"] = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))
        for k, o in enumerate(OUTCOMES):
            out[f"brier_{o}"] = brier_score((y == k).astype(int), preds[f"p_{o}"].to_numpy())
            out[f"log_loss_{o}"] = log_loss_score((y == k).astype(int), preds[f"p_{o}"].to_numpy())
        # market benchmark on the same matches
        M = preds[[f"mkt_{o}" for o in OUTCOMES]].to_numpy()
        ok = ~np.isnan(M).any(axis=1)
        if ok.any():
            out["market_multiclass_brier"] = float(np.mean(np.sum((M[ok] - onehot[ok]) ** 2, axis=1)))
            out["market_multiclass_log_loss"] = float(
                -np.mean(np.log(np.clip(M[ok][np.arange(ok.sum()), y[ok]], 1e-12, 1)))
            )
        out["n_predictions"] = int(len(preds))
        return out


def compare_models(
    features: pd.DataFrame, models: list[MatchModel], config: WalkForwardConfig | None = None
) -> dict[str, BacktestResult]:
    bt = WalkForwardBacktester(config)
    results: dict[str, BacktestResult] = {}
    for m in models:
        logger.info("Backtesting %s", m.name)
        results[m.name] = bt.run(features, m)
    return results
