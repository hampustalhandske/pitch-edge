"""Paper ledger storage: own DuckDB file, cross-source fixture keys, immutable decisions, kickoff guard."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from pitch_edge.paper.fixtures import canonical_league, canonical_team, fixture_key
from pitch_edge.paper.store import (
    Decision,
    DecisionError,
    KickoffGuardError,
    PaperStore,
    Quote,
    Settlement,
)

pytestmark = pytest.mark.unit

T0 = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
KO = datetime(2026, 10, 24, 14, 0, tzinfo=UTC)


def _quote(
    source="odds_api:pinnacle",
    sel="home",
    odds=2.0,
    captured=T0,
    home="Ipswich Town",
    away="Nottingham Forest",
    league="EPL",
    event="ev1",
    kickoff=KO,
    **kw,
) -> Quote:
    return Quote(
        source=source,
        source_event_id=event,
        home=home,
        away=away,
        league=league,
        kickoff_at=kickoff,
        selection=sel,
        captured_at=captured,
        decimal_odds=odds,
        raw_json="{}",
        **kw,
    )


def _decision(quote_ids, decided_at=T0, kickoff=KO, fixture=None, **kw) -> Decision:
    fixture = fixture or fixture_key("EPL", KO, "Ipswich Town", "Nottingham Forest")
    base = {
        "strategy_id": "s",
        "hypothesis_id": "control",
        "decided_at": decided_at,
        "fixture_key": fixture,
        "kickoff_at": kickoff,
        "venue": "odds_api:pinnacle",
        "selection": "home",
        "price_taken": 2.0,
        "paper_stake_units": 1.0,
        "model_prob": 0.5,
        "inputs_quote_ids": list(quote_ids),
        "code_version": "abc",
    }
    base.update(kw)
    return Decision(**base)


@pytest.fixture
def store(tmp_path):
    s = PaperStore(tmp_path / "paper" / "paper.duckdb")
    yield s
    s.close()


def test_store_creates_its_own_file_and_tables(tmp_path):
    path = tmp_path / "x" / "paper.duckdb"
    with PaperStore(path) as s:
        assert path.exists()
        assert {"quotes", "fixtures", "decisions", "settlements"} <= set(s.tables())


def test_canonical_names_join_across_sources():
    assert canonical_league("odds_api", "soccer_epl") == canonical_league("kalshi", "KXEPLGAME") == "EPL"
    assert canonical_league("polymarket", "lal") == canonical_league("odds_api", "soccer_spain_la_liga")
    assert canonical_league("polymarket", "xyz") == "polymarket:xyz"
    assert canonical_team("Ipswich Town") == canonical_team("Ipswich Town FC") == canonical_team("Ipswich")
    assert fixture_key("EPL", KO, "Ipswich Town", "Nottingham Forest") == "EPL|2026-10-24|ipswich|nottm forest"


def test_quotes_from_different_sources_land_on_one_fixture(store):
    store.add_quotes([_quote()])
    store.add_quotes(
        [
            _quote(
                source="polymarket",
                event="pm1",
                home="Ipswich Town FC",
                away="Nottingham Forest FC",
                kickoff=KO + timedelta(minutes=5),
                odds=None,
                bid=0.4,
                ask=0.42,
            )
        ]
    )
    store.add_quotes([_quote(source="kalshi", event="KX1", home="Ipswich", away="Nott'm Forest")])
    fx = store.fixtures()
    assert len(fx) == 1
    row = fx.iloc[0]
    assert row["odds_api_event_id"] == "ev1" and row["polymarket_event_id"] == "pm1"
    assert row["kalshi_event_id"] == "KX1"
    assert store.quotes()["fixture_key"].nunique() == 1


def test_fuzzy_link_requires_same_league_and_close_kickoff(store):
    store.add_quotes([_quote()])
    store.add_quotes([_quote(source="polymarket", event="pm2", kickoff=KO + timedelta(days=1))])
    store.add_quotes([_quote(source="kalshi", event="KX2", league="CHAMPIONSHIP")])
    assert len(store.fixtures()) == 3


def test_adding_the_same_quote_twice_is_idempotent(store):
    q = _quote()
    store.add_quotes([q])
    store.add_quotes([q])
    assert len(store.quotes()) == 1


def test_decision_roundtrip_and_duplicate_id_rejected(store):
    store.add_quotes([_quote()])
    qid = store.quotes()["quote_id"].iloc[0]
    d = _decision([qid])
    store.append_decision(d)
    got = store.decisions()
    assert len(got) == 1 and got.iloc[0]["decision_id"] == d.decision_id
    assert list(got.iloc[0]["inputs_quote_ids"]) == [qid]
    with pytest.raises(DecisionError):
        store.append_decision(d)  # same id again: decisions are append-only and immutable
    assert len(store.decisions()) == 1


def test_store_exposes_no_way_to_change_or_remove_a_decision(store):
    for name in dir(store):
        lowered = name.lower()
        assert not any(w in lowered for w in ("update", "delete", "remove", "amend", "edit", "drop")), name


@pytest.mark.parametrize("kickoff", [T0, T0 - timedelta(minutes=1)])
def test_kickoff_guard_rejects_decisions_at_or_after_kickoff(store, kickoff):
    store.add_quotes([_quote(captured=T0 - timedelta(hours=1))])
    qid = store.quotes()["quote_id"].iloc[0]
    with pytest.raises(KickoffGuardError):
        store.append_decision(_decision([qid], kickoff=kickoff))
    assert store.decisions().empty


def test_decision_cannot_use_a_quote_captured_after_it_was_made(store):
    store.add_quotes([_quote(captured=T0 + timedelta(minutes=1))])
    qid = store.quotes()["quote_id"].iloc[0]
    with pytest.raises(DecisionError):
        store.append_decision(_decision([qid]))


def test_decision_must_reference_known_quotes(store):
    with pytest.raises(DecisionError):
        store.append_decision(_decision(["nope"]))


def test_decision_inputs_must_belong_to_the_decided_fixture(store):
    store.add_quotes([_quote()])
    qid = store.quotes()["quote_id"].iloc[0]
    with pytest.raises(DecisionError):
        store.append_decision(_decision([qid], fixture="EPL|2026-10-24|other|team"))


def test_decision_requires_timezone_aware_timestamps(store):
    store.add_quotes([_quote()])
    qid = store.quotes()["quote_id"].iloc[0]
    with pytest.raises(ValueError):
        store.append_decision(_decision([qid], decided_at=T0.replace(tzinfo=None)))


def test_settlement_is_written_once(store):
    store.add_quotes([_quote()])
    qid = store.quotes()["quote_id"].iloc[0]
    d = _decision([qid])
    store.append_decision(d)
    s = Settlement(
        decision_id=d.decision_id,
        closing_price=2.1,
        fair_close_odds=2.2,
        closing_source="x",
        result="home",
        clv_log=-0.1,
        pnl=1.0,
        settled_at=KO + timedelta(days=1),
    )
    store.append_settlement(s)
    with pytest.raises(DecisionError):
        store.append_settlement(s)
    assert len(store.settlements()) == 1
    assert store.unsettled_decisions().empty


def test_budget_state_persists_across_connections(tmp_path):
    path = tmp_path / "p.duckdb"
    with PaperStore(path) as s:
        assert s.budget("odds_api") is None
        s.record_call("odds_api", "soccer_epl", status="ok", called_at=T0, remaining=420, used=80, n_quotes=3)
    with PaperStore(path) as s:
        assert s.budget("odds_api") == 420
        assert s.last_ok_call("odds_api", "soccer_epl") == T0
        assert s.last_ok_call("odds_api", "soccer_spain_la_liga") is None


def test_one_strongly_matching_club_links_events_in_the_same_league_and_kickoff_window(store):
    # A club plays once a day: same league, kickoff within 3h and one club matching is the same fixture,
    # even when the other club's spelling is unrelated ("Stade Brestois 29" vs "Brest").
    store.add_quotes([_quote(home="Brest", away="Nice", league="LIGUE1")])
    store.add_quotes(
        [
            _quote(
                source="polymarket",
                event="pm9",
                home="Stade Brestois 29",
                away="OGC Nice",
                league="LIGUE1",
                odds=None,
                ask=0.4,
            )
        ]
    )
    assert len(store.fixtures()) == 1
