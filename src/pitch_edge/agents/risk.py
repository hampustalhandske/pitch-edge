"""Risk Manager: bankroll rules applied to the *paper* bankroll only.

Fractional Kelly sizing with hard caps: per-bet, per-match, per-day exposure
and a drawdown circuit breaker. Produces sized *proposals*; nothing is placed
anywhere — the proposals go to the human approval gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from pitch_edge.backtest.kelly import fractional_kelly_stake


@dataclass
class RiskLimits:
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.03
    max_match_exposure_pct: float = 0.03
    max_daily_exposure_pct: float = 0.10
    min_edge: float = 0.03
    min_odds: float = 1.2
    max_odds: float = 12.0
    drawdown_halt_pct: float = 0.25  # stop proposing if paper bankroll is down this much from peak
    max_open_proposals: int = 10


@dataclass
class RiskState:
    bankroll: float = 1000.0
    peak_bankroll: float = 1000.0
    open_exposure_today: float = 0.0
    open_by_match: dict[str, float] = field(default_factory=dict)


@dataclass
class Proposal:
    match_id: str
    date: str
    home_team: str
    away_team: str
    outcome: str
    model_probability: float
    market_probability: float
    edge: float
    decimal_odds: float
    stake: float
    bookmaker: str
    model_name: str
    rationale: str


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None, state: RiskState | None = None):
        self.limits = limits or RiskLimits()
        self.state = state or RiskState()

    def halted(self) -> bool:
        dd = (self.state.peak_bankroll - self.state.bankroll) / max(self.state.peak_bankroll, 1e-9)
        return dd >= self.limits.drawdown_halt_pct

    def size(self, edges: pd.DataFrame, model_name: str, bookmaker: str = "PS") -> list[Proposal]:
        """edges columns: match_id, date, home_team, away_team, outcome, model_probability,
        market_probability, edge, decimal_odds."""
        lim, st = self.limits, self.state
        proposals: list[Proposal] = []
        if self.halted() or edges.empty:
            return proposals
        ranked = edges.sort_values("edge", ascending=False)
        for _, r in ranked.iterrows():
            if len(proposals) >= lim.max_open_proposals:
                break
            if r["edge"] < lim.min_edge or not (lim.min_odds <= r["decimal_odds"] <= lim.max_odds):
                continue
            stake = fractional_kelly_stake(
                float(r["model_probability"]),
                float(r["decimal_odds"]),
                st.bankroll,
                fraction=lim.kelly_fraction,
                max_stake_pct=lim.max_stake_pct,
            )
            room_match = lim.max_match_exposure_pct * st.bankroll - st.open_by_match.get(str(r["match_id"]), 0.0)
            room_day = lim.max_daily_exposure_pct * st.bankroll - st.open_exposure_today
            stake = round(min(stake, room_match, room_day), 2)
            if stake <= 0:
                continue
            st.open_by_match[str(r["match_id"])] = st.open_by_match.get(str(r["match_id"]), 0.0) + stake
            st.open_exposure_today += stake
            proposals.append(
                Proposal(
                    str(r["match_id"]),
                    str(pd.Timestamp(r["date"]).date()),
                    str(r["home_team"]),
                    str(r["away_team"]),
                    str(r["outcome"]),
                    float(r["model_probability"]),
                    float(r["market_probability"]),
                    float(r["edge"]),
                    float(r["decimal_odds"]),
                    stake,
                    bookmaker,
                    model_name,
                    f"model {r['model_probability']:.1%} vs no-vig market {r['market_probability']:.1%} "
                    f"(edge {r['edge']:+.1%}); {lim.kelly_fraction:g}-Kelly stake capped at {lim.max_stake_pct:.0%} of paper bankroll",
                )
            )
        return proposals

    def settle(self, profit: float) -> None:
        self.state.bankroll += profit
        self.state.peak_bankroll = max(self.state.peak_bankroll, self.state.bankroll)
