"""Paper strategies: `PaperStrategy.decide(snapshot) -> list[Decision]`.

A `Snapshot` is everything a strategy may look at when deciding at `as_of`: the latest quote per
(source, fixture, selection) captured at or before `as_of` for fixtures that have not kicked off, plus
the fixtures each strategy has already decided (one paper bet per fixture per strategy).

Built-ins:

- `control-fav-pinnacle` (hypothesis `control`): back the Pinnacle favourite at Pinnacle's price.
  Expected CLV is about zero; it checks that collection, settlement and CLV maths are wired correctly.
- `soft-vs-sharp-live` (hypothesis `bookmaker-line-structure-live-soft-vs-sharp-v1`): back the best soft-book
  price that beats the de-vigged sharp price in the same the-odds-api response by more than 2%.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from pitch_edge.paper.pricing import power_devig_odds
from pitch_edge.paper.r6_model import COEFFICIENTS_SHA256, NON_SOFT_BOOKS, R6LiveModel
from pitch_edge.paper.store import SELECTIONS, Decision, DecisionError, PaperStore

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    """Point-in-time view of the ledger's quotes for strategies deciding at `as_of`."""

    as_of: datetime
    quotes: pd.DataFrame
    code_version: str
    decided: dict[str, set[str]] = field(default_factory=dict)
    # Every the-odds-api quote of the last `HISTORY_WINDOW` (all responses, not only the latest per book),
    # for strategies whose decision time is a particular earlier response.
    odds_history: pd.DataFrame | None = None

    HISTORY_WINDOW = timedelta(hours=80)

    @classmethod
    def from_store(cls, store: PaperStore, as_of: datetime, code_version: str) -> Snapshot:
        decisions = store.decisions()
        decided = {s: set(g["fixture_key"]) for s, g in decisions.groupby("strategy_id")} if len(decisions) else {}
        return cls(
            as_of=as_of,
            quotes=store.latest_quotes(as_of),
            code_version=code_version,
            decided=decided,
            odds_history=store.odds_api_history(as_of - cls.HISTORY_WINDOW, as_of),
        )

    def already_decided(self, strategy_id: str) -> set[str]:
        return self.decided.get(strategy_id, set())

    def fresh(self, source: str, max_age: timedelta) -> pd.DataFrame:
        """Latest quotes from `source` captured within `max_age` of `as_of`."""
        q = self.quotes
        if q.empty:
            return q
        return q[(q["source"] == source) & (q["captured_at"] >= pd.Timestamp(self.as_of - max_age))]

    def three_way(self, source: str, max_age: timedelta) -> dict[str, pd.DataFrame]:
        """Fixture key -> its home/draw/away bookmaker quotes from `source` (only complete, priced sets)."""
        q = self.fresh(source, max_age)
        out: dict[str, pd.DataFrame] = {}
        for fk, g in q.groupby("fixture_key"):
            g = g.set_index("selection")
            if set(SELECTIONS) <= set(g.index) and g.loc[list(SELECTIONS), "decimal_odds"].gt(1.0).all():
                out[str(fk)] = g.loc[list(SELECTIONS)]
        return out


def _best_quote(cands: pd.DataFrame) -> pd.Series:
    """The quote with the highest decimal odds; equal odds go to the smallest source, then quote id, so the book a
    decision names does not depend on row order."""
    order = cands.sort_values(["decimal_odds", "source", "quote_id"], ascending=[False, True, True], kind="stable")
    return order.iloc[0]


class PaperStrategy(ABC):
    """A rule that turns a `Snapshot` into paper decisions. Never places a real order."""

    strategy_id: str
    hypothesis_id: str
    # Set by `decide` when the strategy reports opportunity frequency for the snapshot it just saw:
    # {"n_fixtures_evaluated": ..., "n_fixtures_qualifying": ...} (qualifying counts already-decided fixtures too).
    last_stats: dict[str, int] | None = None

    @abstractmethod
    def decide(self, snapshot: Snapshot) -> list[Decision]:
        """Decisions for fixtures this strategy has not decided yet."""


class ControlFavPinnacle(PaperStrategy):
    """Back the Pinnacle favourite at Pinnacle's price (pipeline sanity check, CLV should be about zero)."""

    strategy_id = "control-fav-pinnacle"
    hypothesis_id = "control"

    def __init__(self, max_quote_age_h: float = 26.0, stake: float = 1.0):
        self.max_quote_age_h = max_quote_age_h
        self.stake = stake

    def decide(self, snapshot: Snapshot) -> list[Decision]:
        done = snapshot.already_decided(self.strategy_id)
        out: list[Decision] = []
        for fk, book in snapshot.three_way("odds_api:pinnacle", timedelta(hours=self.max_quote_age_h)).items():
            if fk in done:
                continue
            odds = [float(book.loc[s, "decimal_odds"]) for s in SELECTIONS]
            fair = power_devig_odds(odds)
            i = min(range(3), key=lambda k: odds[k])
            out.append(
                Decision(
                    strategy_id=self.strategy_id,
                    hypothesis_id=self.hypothesis_id,
                    decided_at=snapshot.as_of,
                    fixture_key=fk,
                    kickoff_at=book["kickoff_at"].iloc[0].to_pydatetime(),
                    venue="odds_api:pinnacle",
                    selection=SELECTIONS[i],
                    price_taken=odds[i],
                    paper_stake_units=self.stake,
                    model_prob=fair[i],
                    inputs_quote_ids=list(book["quote_id"]),
                    code_version=snapshot.code_version,
                    params={"max_quote_age_h": self.max_quote_age_h},
                )
            )
        return out


class SoftVsSharpLive(PaperStrategy):
    """Back the best soft-book price when it beats the de-vigged sharp price in the same response by > `margin`.

    For each fixture, only the most recent the-odds-api response (one capture time, at most
    `max_quote_age_h` old) is used. Fair odds come from the power-de-vigged Pinnacle quote in that
    response, else Betfair Exchange (UK, then EU), else Smarkets. For each outcome the best price among
    soft books (any book that is neither a sharp reference nor an exchange, or only `soft_books` when
    given) qualifies when price > (1 + margin) x fair odds, price <= the maximum quote for that outcome in
    the response, and price <= `max_over_sharp` x the sharp book's own quote (a guard against stale or
    erroneous soft prices). One decision per fixture: the outcome with the largest edge.
    """

    strategy_id = "soft-vs-sharp-live"
    hypothesis_id = "bookmaker-line-structure-live-soft-vs-sharp-v1"
    sharp_sources = ("odds_api:pinnacle", "odds_api:betfair_ex_uk", "odds_api:betfair_ex_eu", "odds_api:smarkets")
    non_soft_books = frozenset({"pinnacle", "betfair_ex_uk", "betfair_ex_eu", "betfair_ex_au", "smarkets", "matchbook"})

    def __init__(
        self,
        margin: float = 0.02,
        max_over_sharp: float = 1.10,
        max_quote_age_h: float = 1.0,
        soft_books: tuple[str, ...] | None = None,
        stake: float = 1.0,
    ):
        self.margin = margin
        self.max_over_sharp = max_over_sharp
        self.max_quote_age_h = max_quote_age_h
        self.soft_books = soft_books
        self.stake = stake

    def _is_soft(self, source: str) -> bool:
        book = source.split(":", 1)[1] if ":" in source else source
        if self.soft_books is not None:
            return book in self.soft_books
        return book not in self.non_soft_books

    def decide(self, snapshot: Snapshot) -> list[Decision]:
        done = snapshot.already_decided(self.strategy_id)
        q = snapshot.quotes
        q = q[q["source"].str.startswith("odds_api:") & q["decimal_odds"].gt(1.0)] if not q.empty else q
        q = q[q["captured_at"] >= pd.Timestamp(snapshot.as_of - timedelta(hours=self.max_quote_age_h))]
        evaluated = qualifying = 0
        out: list[Decision] = []
        for fk, g in q.groupby("fixture_key"):
            fk = str(fk)
            resp = g[g["captured_at"] == g["captured_at"].max()]
            sharp_src, sharp = None, None
            for src in self.sharp_sources:
                b = resp[resp["source"] == src].set_index("selection")
                if set(SELECTIONS) <= set(b.index):
                    sharp_src, sharp = src, b.loc[list(SELECTIONS)]
                    break
            soft = resp[resp["source"].map(self._is_soft)]
            if sharp is None or soft.empty:
                continue
            evaluated += 1
            sharp_odds = [float(sharp.loc[s, "decimal_odds"]) for s in SELECTIONS]
            fair_p = dict(zip(SELECTIONS, power_devig_odds(sharp_odds), strict=True))
            best: tuple[float, pd.Series] | None = None
            for sel in SELECTIONS:
                cands = soft[soft["selection"] == sel]
                if cands.empty:
                    continue
                top = _best_quote(cands)
                price = float(top["decimal_odds"])
                fair_odds = 1.0 / fair_p[sel]
                ok = (
                    price > (1.0 + self.margin) * fair_odds
                    and price <= float(resp.loc[resp["selection"] == sel, "decimal_odds"].max())
                    and price <= self.max_over_sharp * float(sharp.loc[sel, "decimal_odds"])
                )
                edge = price / fair_odds - 1.0
                if ok and (best is None or edge > best[0]):
                    best = (edge, top)
            if best is None:
                continue
            qualifying += 1
            if fk in done:
                continue
            edge, top = best
            sel = str(top["selection"])
            book = str(top["source"]).split(":", 1)[1]
            out.append(
                Decision(
                    strategy_id=self.strategy_id,
                    hypothesis_id=self.hypothesis_id,
                    decided_at=snapshot.as_of,
                    fixture_key=fk,
                    kickoff_at=top["kickoff_at"].to_pydatetime(),
                    venue=str(top["source"]),
                    selection=sel,
                    price_taken=float(top["decimal_odds"]),
                    paper_stake_units=self.stake,
                    model_prob=fair_p[sel],
                    inputs_quote_ids=[*sharp["quote_id"], top["quote_id"]],
                    code_version=snapshot.code_version,
                    params={
                        "book": book,
                        "sharp_source": sharp_src,
                        "edge": edge,
                        "margin": self.margin,
                        "max_over_sharp": self.max_over_sharp,
                        "max_quote_age_h": self.max_quote_age_h,
                        "response_captured_at": top["captured_at"].isoformat(),
                        "soft_last_update_age_s": _last_update_age_s(top),
                        "sharp_last_update_age_s": max(
                            (a for a in (_last_update_age_s(r) for _, r in sharp.iterrows()) if a is not None),
                            default=None,
                        ),
                    },
                )
            )
        self.last_stats = {"n_fixtures_evaluated": evaluated, "n_fixtures_qualifying": qualifying}
        return out


class SoftVsSharpLiveV2(PaperStrategy):
    """One arm of `bookmaker-line-structure-live-soft-vs-sharp-v2`: the v1 soft-vs-sharp rule with the fair
    value from either the de-vigged live Pinnacle quote (`pinnacle`) or the frozen R6 live model
    (`r6_live_model`, see paper/r6_model.py), decided at one fixed time.

    Decision time: the first the-odds-api response captured 20-76 h before kickoff; every input comes from
    that response, so there is one decision opportunity per fixture per arm. The response must be at most
    `max_response_age_h` old when `decide` runs (else the opportunity is missed, never taken late).
    Rule (both arms): for each outcome take the best soft-book price (soft = v1's definition) and back it when
    price > (1 + margin) / p_fair, price <= the response's max for that outcome, and price <= 1.10 x Pinnacle's
    quote; one decision per fixture, the largest edge (price x p_fair - 1). Both arms evaluate the same
    response every time, so each decision records whether the other arm bet (the paired comparison).

    Audit caveat (research/audits/closing-line-prediction-r6.md): the onexbet and williamhill coefficients are
    not safe labels (one season of 1XB history; WH's sign is unstable), so a bet the model arm keeps or drops
    because of those books is not evidence about those books; betvictor carries most of R6's gain and has no
    football-data history after 2023/24.

    `coefficients_path` (default: `r6_model.default_coefficients_path()`) and `coefficients_sha256` (default:
    the registered `COEFFICIENTS_SHA256`) say which coefficients file the model arm loads and the sha256 it
    must have; `decide` raises `CoefficientHashError` for any other file.
    """

    hypothesis_id = "bookmaker-line-structure-live-soft-vs-sharp-v2"
    ARMS = ("pinnacle", "r6_live_model")
    INFORMATIVE_BOOKS = frozenset({"betvictor", "betfair_sb_uk"})

    def __init__(
        self,
        arm: str,
        coefficients_path: str | None | object = None,
        margin: float = 0.02,
        max_over_sharp: float = 1.10,
        min_hours_before: float = 20.0,
        max_hours_before: float = 76.0,
        max_response_age_h: float = 1.0,
        stake: float = 1.0,
        coefficients_sha256: str = COEFFICIENTS_SHA256,
    ):
        if arm not in self.ARMS:
            raise ValueError(f"arm must be one of {self.ARMS}, got {arm!r}")
        self.arm = arm
        self.strategy_id = f"soft-vs-sharp-live-v2-{arm}"
        self.coefficients_path = coefficients_path
        self.coefficients_sha256 = coefficients_sha256
        self.margin = margin
        self.max_over_sharp = max_over_sharp
        self.min_hours_before = min_hours_before
        self.max_hours_before = max_hours_before
        self.max_response_age_h = max_response_age_h
        self.stake = stake
        self._model: R6LiveModel | None = None

    def _load_model(self) -> R6LiveModel:
        if self._model is None:
            self._model = R6LiveModel.load(self.coefficients_path, self.coefficients_sha256)  # type: ignore[arg-type]
        return self._model

    def _first_responses(self, snapshot: Snapshot) -> dict[str, pd.DataFrame]:
        """Fixture -> quotes of its first response captured 20-76 h before kickoff, if that response is fresh."""
        hist = snapshot.odds_history
        if hist is None or hist.empty:
            return {}
        hours = (hist["kickoff_at"] - hist["captured_at"]).dt.total_seconds() / 3600.0
        inside = hist[(hours >= self.min_hours_before) & (hours <= self.max_hours_before)]
        out: dict[str, pd.DataFrame] = {}
        oldest_ok = pd.Timestamp(snapshot.as_of - timedelta(hours=self.max_response_age_h))
        for fk, g in inside.groupby("fixture_key"):
            first = g["captured_at"].min()
            if first >= oldest_ok:
                out[str(fk)] = g[g["captured_at"] == first]
        return out

    def _evaluate(self, resp: pd.DataFrame, p_fair: list[float]) -> tuple[float, pd.Series] | None:
        pin = resp[resp["source"] == "odds_api:pinnacle"].set_index("selection")
        soft = resp[~resp["source"].str.split(":", n=1).str[1].isin(NON_SOFT_BOOKS)]
        best: tuple[float, pd.Series] | None = None
        for k, sel in enumerate(SELECTIONS):
            cands = soft[soft["selection"] == sel]
            if cands.empty:
                continue
            top = _best_quote(cands)
            price = float(top["decimal_odds"])
            ok = (
                price > (1.0 + self.margin) / p_fair[k]
                and price <= float(resp.loc[resp["selection"] == sel, "decimal_odds"].max())
                and price <= self.max_over_sharp * float(pin.loc[sel, "decimal_odds"])
            )
            edge = price * p_fair[k] - 1.0
            if ok and (best is None or edge > best[0]):
                best = (edge, top)
        return best

    def decide(self, snapshot: Snapshot) -> list[Decision]:
        model = self._load_model()  # refuses a coefficients file that is not the registered one
        done = snapshot.already_decided(self.strategy_id)
        evaluated = qualifying = 0
        out: list[Decision] = []
        for fk, resp in self._first_responses(snapshot).items():
            resp = resp[resp["decimal_odds"].gt(1.0)]
            pin = resp[resp["source"] == "odds_api:pinnacle"].set_index("selection")
            if not set(SELECTIONS) <= set(pin.index):
                continue
            books: dict[str, tuple[float, float, float]] = {}
            for src, g in resp.groupby("source"):
                g = g.set_index("selection")
                if set(SELECTIONS) <= set(g.index):
                    books[str(src).split(":", 1)[1]] = tuple(float(g.loc[s, "decimal_odds"]) for s in SELECTIONS)  # type: ignore[assignment]
            pin_odds = books.pop("pinnacle")
            try:
                fv = model.fair_value(pin_odds, books)
            except ValueError:
                continue
            evaluated += 1
            fair = {"pinnacle": fv.p_pin, "r6_live_model": fv.p_fair}
            picks = {arm: self._evaluate(resp, fair[arm]) for arm in self.ARMS}
            mine = picks[self.arm]
            other = picks["r6_live_model" if self.arm == "pinnacle" else "pinnacle"]
            if mine is None:
                continue
            qualifying += 1
            if fk in done:
                continue
            edge, top = mine
            sel = str(top["selection"])
            book = str(top["source"]).split(":", 1)[1]
            captured = top["captured_at"]
            out.append(
                Decision(
                    strategy_id=self.strategy_id,
                    hypothesis_id=self.hypothesis_id,
                    decided_at=snapshot.as_of,
                    fixture_key=fk,
                    kickoff_at=top["kickoff_at"].to_pydatetime(),
                    venue=str(top["source"]),
                    selection=sel,
                    price_taken=float(top["decimal_odds"]),
                    paper_stake_units=self.stake,
                    model_prob=fair[self.arm][SELECTIONS.index(sel)],
                    inputs_quote_ids=[*pin.loc[list(SELECTIONS), "quote_id"], top["quote_id"]],
                    code_version=snapshot.code_version,
                    params={
                        "arm": self.arm,
                        "p_fair": fair[self.arm],
                        "p_pin": fv.p_pin,
                        "model": fv.as_params(),
                        "book": book,
                        "informative_book": book in self.INFORMATIVE_BOOKS,
                        "edge": edge,
                        "other_arm_bets": other is not None,
                        "other_arm_selection": None if other is None else str(other[1]["selection"]),
                        "other_arm_book": None if other is None else str(other[1]["source"]).split(":", 1)[1],
                        "response_captured_at": captured.isoformat(),
                        "hours_before_kickoff": (top["kickoff_at"] - captured).total_seconds() / 3600.0,
                        "soft_last_update_age_s": _last_update_age_s(top),
                        "coefficients_sha256": model.sha256,
                        "margin": self.margin,
                        "max_over_sharp": self.max_over_sharp,
                        "window_h": [self.min_hours_before, self.max_hours_before],
                        "max_response_age_h": self.max_response_age_h,
                    },
                )
            )
        self.last_stats = {"n_fixtures_evaluated": evaluated, "n_fixtures_qualifying": qualifying}
        return out


def _last_update_age_s(quote: pd.Series) -> int | None:
    """Seconds between a bookmaker's own `last_update` (from the quote's raw JSON) and our capture time."""
    try:
        stamp = json.loads(quote["raw_json"] or "{}").get("last_update")
    except (TypeError, ValueError):
        return None
    if not stamp:
        return None
    updated = pd.Timestamp(stamp)
    updated = updated.tz_localize("UTC") if updated.tzinfo is None else updated.tz_convert("UTC")
    return int((quote["captured_at"] - updated).total_seconds())


_REGISTRY: dict[str, Callable[[], PaperStrategy]] = {
    ControlFavPinnacle.strategy_id: ControlFavPinnacle,
    SoftVsSharpLive.strategy_id: SoftVsSharpLive,
    "soft-vs-sharp-live-v2-pinnacle": lambda: SoftVsSharpLiveV2("pinnacle"),
    "soft-vs-sharp-live-v2-r6_live_model": lambda: SoftVsSharpLiveV2("r6_live_model"),
}


def available_strategies() -> dict[str, Callable[[], PaperStrategy]]:
    """Strategy id -> factory for every built-in strategy."""
    return dict(_REGISTRY)


def run_decide(
    store: PaperStore,
    strategies: list[PaperStrategy],
    now: datetime,
    code_version: str,
    code_hash: str | None = None,
) -> dict[str, int]:
    """Build a snapshot at `now`, run each strategy and append its decisions. Returns decisions added per strategy.

    A decision the store rejects (kickoff guard, point-in-time inputs) is logged and skipped, never forced.
    Each strategy run is logged in the store's `decide_log`, with its opportunity counts when it reports them.
    A strategy that raises is logged (with the error) and skipped; the others still run. Every decision is
    stamped with `code_hash` (see paper/provenance.py).
    """
    snapshot = Snapshot.from_store(store, as_of=now, code_version=code_version)
    added: dict[str, int] = {}
    for strat in strategies:
        n = 0
        strat.last_stats = None
        try:
            proposed = strat.decide(snapshot)
            error = None
        except Exception as exc:
            logger.exception("%s failed to decide", strat.strategy_id)
            proposed, error = [], f"{type(exc).__name__}: {exc}"[:500]
        for d in proposed:
            if d.code_hash is None:
                d = dataclasses.replace(d, code_hash=code_hash)
            try:
                store.append_decision(d)
                n += 1
            except DecisionError as exc:
                logger.warning("%s: decision on %s rejected: %s", strat.strategy_id, d.fixture_key, exc)
        added[strat.strategy_id] = n
        stats: dict[str, int] = strat.last_stats or {}
        store.record_decide_run(
            strat.strategy_id,
            now,
            n_fixtures_evaluated=stats.get("n_fixtures_evaluated"),
            n_fixtures_qualifying=stats.get("n_fixtures_qualifying"),
            n_new_decisions=n,
            error=error,
        )
    return added
