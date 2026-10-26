"""Ledger integrity: strict cross-source fixture linking, code provenance, one decision per strategy and
fixture, the append-only JSONL hash chain, provisional settlements, decision flags, read-only access."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from pitch_edge.paper.fixtures import canonical_team, match_candidate, team_similarity
from pitch_edge.paper.provenance import code_hash, snapshot_code
from pitch_edge.paper.store import Decision, DecisionError, PaperStore, Quote, Settlement, verify_decision_chain

pytestmark = pytest.mark.unit

T0 = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
KO = datetime(2026, 10, 24, 14, 0, tzinfo=UTC)


def _q(
    source="odds_api:pinnacle",
    event="e1",
    home="Ipswich Town",
    away="Nottingham Forest",
    league="EPL",
    kickoff=KO,
    captured=T0,
    sel="home",
    odds=2.0,
):
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
    )


def _d(store, strategy="s", event="e1", **kw):
    q = store.quotes().query("source_event_id == @event").iloc[0]
    base = {
        "strategy_id": strategy,
        "hypothesis_id": "control",
        "decided_at": T0,
        "fixture_key": q["fixture_key"],
        "kickoff_at": KO,
        "venue": "odds_api:pinnacle",
        "selection": "home",
        "price_taken": 2.0,
        "paper_stake_units": 1.0,
        "model_prob": 0.5,
        "inputs_quote_ids": [q["quote_id"]],
        "code_version": "sha",
        "code_hash": "h1",
    }
    base.update(kw)
    return Decision(**base)


@pytest.fixture
def store(tmp_path):
    s = PaperStore(tmp_path / "paper.duckdb")
    yield s
    s.close()


# ---------------------------------------------------------------- I1 fixture linking
def test_paris_fc_and_psg_on_the_same_day_are_different_fixtures(store):
    store.add_quotes([_q(home="Paris Saint Germain", away="Lens", league="LIGUE1")])
    store.add_quotes(
        [
            _q(
                source="polymarket",
                event="pm1",
                home="Paris FC",
                away="Nice",
                league="LIGUE1",
                kickoff=KO + timedelta(hours=2),
                odds=None,
            )
        ]
    )
    assert len(store.fixtures()) == 2


def test_a_token_shared_by_several_clubs_does_not_count_as_containment():
    assert team_similarity("paris", "paris sg") == 1.0  # without context the tokens contain each other
    assert team_similarity("paris", "paris sg", ambiguous_tokens={"paris"}) < 0.85


def test_both_clubs_must_be_similar_not_just_one(store):
    store.add_quotes([_q(home="Alpha United", away="Zeta", league="X")])
    store.add_quotes([_q(source="kalshi", event="k1", home="Alpha United", away="Omega Rovers", league="X")])
    assert len(store.fixtures()) == 2


def test_observed_spellings_link_through_the_paper_alias_table(store):
    store.add_quotes([_q(home="Brest", away="Nice", league="LIGUE1")])
    store.add_quotes(
        [_q(source="polymarket", event="pm2", home="Stade Brestois 29", away="OGC Nice", league="LIGUE1", odds=None)]
    )
    store.add_quotes([_q(event="e2", home="NEC", away="Groningen", league="EREDIVISIE")])
    store.add_quotes([_q(source="kalshi", event="k2", home="Nijmegen", away="Groningen", league="EREDIVISIE")])
    assert len(store.fixtures()) == 2
    assert canonical_team("Stade Rennais 1901") == canonical_team("Rennes")
    assert canonical_team("Espanyol Barcelona") == canonical_team("Espanyol")


def test_more_than_one_passing_candidate_is_refused():
    cands = [("a", "L", KO, "real madrid", "getafe"), ("b", "L", KO + timedelta(hours=1), "real madrid", "getafe")]
    assert match_candidate("L", KO, "real madrid", "getafe", cands) is None
    assert match_candidate("L", KO, "real madrid", "getafe", cands[:1]) == "a"


# ---------------------------------------------------------------- I3 provenance
def test_code_hash_is_deterministic_and_code_is_copied_once(tmp_path):
    src = tmp_path / "pkg"
    (src / "sub").mkdir(parents=True)
    (src / "a.py").write_text("x = 1\n")
    (src / "sub" / "b.py").write_text("y = 2\n")
    h = code_hash(src)
    assert h == code_hash(src) and len(h) == 64
    dest = tmp_path / "code"
    assert snapshot_code(src, dest) == h
    assert (dest / h / "a.py").read_text() == "x = 1\n" and (dest / h / "sub" / "b.py").exists()
    (dest / h / "a.py").write_text("tampered")
    snapshot_code(src, dest)  # already archived: left untouched, never overwritten
    assert (dest / h / "a.py").read_text() == "tampered"
    (src / "a.py").write_text("x = 3\n")
    assert code_hash(src) != h


def test_decisions_store_the_code_hash(store):
    store.add_quotes([_q()])
    store.append_decision(_d(store))
    assert store.decisions().iloc[0]["code_hash"] == "h1"


# ---------------------------------------------------------------- I6 uniqueness + JSONL chain
def test_one_decision_per_strategy_and_fixture(store):
    store.add_quotes([_q()])
    store.append_decision(_d(store))
    with pytest.raises(DecisionError):
        store.append_decision(_d(store))  # new id, same strategy + fixture
    store.append_decision(_d(store, strategy="other"))
    assert len(store.decisions()) == 2


def test_decisions_are_exported_to_a_verifiable_jsonl_hash_chain(tmp_path):
    with PaperStore(tmp_path / "p.duckdb") as s:
        s.add_quotes([_q(), _q(event="e2", home="Arsenal", away="Everton")])
        s.append_decision(_d(s))
        s.append_decision(_d(s, event="e2"))
        path = s.chain_path
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["prev_hash"] == json.loads(lines[0])["hash"]
    assert verify_decision_chain(path) == (True, 2)
    tampered = lines[0].replace('"price_taken": 2.0', '"price_taken": 9.0')
    path.write_text("\n".join([tampered, lines[1]]) + "\n")
    ok, _ = verify_decision_chain(path)
    assert not ok


def test_decisions_missing_from_the_chain_are_exported_on_open(tmp_path):
    with PaperStore(tmp_path / "p.duckdb") as s:
        s.add_quotes([_q()])
        s.append_decision(_d(s))
        path = s.chain_path
    path.unlink()
    with PaperStore(tmp_path / "p.duckdb"):
        pass
    assert verify_decision_chain(path) == (True, 1)


# ---------------------------------------------------------------- C1 provisional settlements, flags
def _settlement(d, source="own_snapshot:x@t"):
    return Settlement(
        decision_id=d.decision_id,
        closing_price=2.0,
        fair_close_odds=2.1,
        closing_source=source,
        result=None,
        clv_log=-0.05,
        pnl=None,
        settled_at=KO + timedelta(days=5),
    )


def test_provisional_settlements_live_apart_and_do_not_close_the_decision(store):
    store.add_quotes([_q()])
    d = _d(store)
    store.append_decision(d)
    store.append_provisional_settlement(_settlement(d))
    with pytest.raises(DecisionError):
        store.append_provisional_settlement(_settlement(d))
    assert len(store.provisional_settlements()) == 1 and store.settlements().empty
    assert list(store.unsettled_decisions()["decision_id"]) == [d.decision_id]
    store.append_settlement(_settlement(d, source="lake:fd_matches.betfair_ex_close"))
    assert store.unsettled_decisions().empty


def test_decision_flags_are_append_only_annotations(store):
    store.add_quotes([_q()])
    d = _d(store)
    store.append_decision(d)
    store.flag_decision(d.decision_id, "void", "kickoff moved", flagged_at=T0)
    flags = store.decision_flags()
    assert list(flags["flag"]) == ["void"] and flags.iloc[0]["reason"] == "kickoff moved"
    assert store.voided_decision_ids() == {d.decision_id}
    assert store.unsettled_decisions().empty  # a void decision is never settled
    with pytest.raises(DecisionError):
        store.flag_decision("nope", "void", "x", flagged_at=T0)


# ---------------------------------------------------------------- minor 8, I7
def test_latest_quotes_has_a_lookback_lower_bound(store):
    store.add_quotes([_q(captured=T0 - timedelta(days=10))])
    assert store.latest_quotes(T0).empty
    assert len(store.latest_quotes(T0, lookback=timedelta(days=30))) == 1


def test_read_only_store_reads_and_refuses_missing_files(tmp_path):
    with PaperStore(tmp_path / "p.duckdb") as s:
        s.add_quotes([_q()])
    with PaperStore(tmp_path / "p.duckdb", read_only=True) as r:
        assert len(r.quotes()) == 1
    assert PaperStore.open_read_only(tmp_path / "missing.duckdb") is None
