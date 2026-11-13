"""`soft-vs-sharp-live-v2`: the R6 live fair-value model (golden test against a synthetic coefficients file),
the coefficient-hash guard, the 20-76 h first-response decision time, and the two arms on one response.

Every model, guard and decision test runs on `SYNTHETIC_COEFFICIENTS`, written to `tmp_path` and loaded with its
own sha256, so they run on a clean checkout. Only `test_coefficients_file_hash_is_the_registered_one` reads the
frozen file in `data/research/r6/` (git-ignored) and is skipped where it is absent."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from pitch_edge.paper.r6_model import COEFFICIENTS_SHA256, CoefficientHashError, R6LiveModel
from pitch_edge.paper.store import PaperStore, Quote
from pitch_edge.paper.strategies import Snapshot, SoftVsSharpLiveV2, available_strategies, run_decide
from pitch_edge.research.devig import devig

pytestmark = pytest.mark.unit

COEF = Path(__file__).resolve().parents[2] / "data" / "research" / "r6" / "v2_live_coefficients.json"
needs_coef = pytest.mark.skipif(not COEF.exists(), reason="frozen R6 coefficients not present in data/")
SEL = ("home", "draw", "away")
KO = datetime(2026, 10, 25, 15, 0, tzinfo=UTC)

PIN = (2.10, 3.40, 3.70)
BOOKS = {
    "williamhill": (2.05, 3.30, 3.80),
    "betvictor": (2.20, 3.30, 3.40),
    "onexbet": (2.15, 3.45, 3.55),
    # betfair_sb_uk missing -> dev 0, miss 1
    "bet365": (2.00, 3.40, 3.90),
    "unibet_uk": (2.10, 3.30, 3.60),
}


def _outcome(intercept, dev_vc, dev_bf):
    return {
        "intercept": intercept,
        "g_avg": 0.05,
        "dev_WH": 0.04,
        "dev_VC": dev_vc,
        "dev_1XB": -0.05,
        "dev_BF": dev_bf,
        "miss_WH": 0.012,
        "miss_VC": -0.008,
        "miss_1XB": 0.005,
        "miss_BF": -0.004,
    }


# Same layout as the frozen file; values chosen so the informative books (VC, BF) move the home probability and
# the hand-built responses below keep the decisions the tests describe.
SYNTHETIC_COEFFICIENTS = {
    "books": {"WH": "williamhill", "VC": "betvictor", "1XB": "onexbet", "BF": "betfair_sb_uk"},
    "dev_clip": 0.5,
    "outcomes": {
        "home": _outcome(0.004, 0.45, 0.40),
        "draw": _outcome(-0.006, 0.10, 0.08),
        "away": _outcome(0.002, 0.02, 0.02),
    },
}
# `fair_value(PIN, BOOKS)` under SYNTHETIC_COEFFICIENTS, pinned to 7 decimals.
PINNED_P_FAIR = [0.4577726, 0.2830118, 0.2592157]


@pytest.fixture
def coef(tmp_path) -> tuple[Path, str]:
    """The synthetic coefficients file in `tmp_path` and its sha256."""
    path = tmp_path / "v2_live_coefficients.json"
    path.write_text(json.dumps(SYNTHETIC_COEFFICIENTS, indent=2))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _logit(p):
    return math.log(p / (1 - p))


def _expected_p_fair(pin, books, coef):
    """Independent re-statement of the registered formula (notes of the v2 registration)."""
    p_pin = devig([list(pin)], "power").probs[0]
    soft = [o for b, o in books.items() if b not in ("pinnacle", "betfair_ex_uk", "smarkets", "matchbook")]
    mean_implied = np.mean([[1 / x for x in o] for o in soft], axis=0)
    p_cons = devig([list(1 / mean_implied)], "power").probs[0]
    short = {"williamhill": "WH", "betvictor": "VC", "onexbet": "1XB", "betfair_sb_uk": "BF"}
    q = []
    for k, o in enumerate(SEL):
        c = coef["outcomes"][o]
        z = _logit(p_pin[k]) + c["intercept"] + c["g_avg"] * (_logit(p_cons[k]) - _logit(p_pin[k]))
        for book, b in short.items():
            if book in books:
                d = _logit(devig([list(books[book])], "power").probs[0][k]) - _logit(p_pin[k])
                z += c[f"dev_{b}"] * max(-0.5, min(0.5, d))
            else:
                z += c[f"miss_{b}"]
        q.append(1 / (1 + math.exp(-z)))
    return [x / sum(q) for x in q]


# ---------------------------------------------------------------- the frozen file
@needs_coef
def test_coefficients_file_hash_is_the_registered_one():
    assert hashlib.sha256(COEF.read_bytes()).hexdigest() == COEFFICIENTS_SHA256


@needs_coef
def test_golden_p_fair_with_the_registered_coefficients():
    """The live strategy's own coefficients on the hand-built response, pinned so a change to the frozen file's
    model or to `fair_value` is caught where the file exists."""
    out = R6LiveModel.load(COEF).fair_value(PIN, BOOKS)
    assert out.p_fair == pytest.approx(_expected_p_fair(PIN, BOOKS, json.loads(COEF.read_text())), abs=1e-12)
    assert out.p_fair == pytest.approx([0.4624651, 0.2808461, 0.2566888], abs=1e-6)


# ---------------------------------------------------------------- the hash guard
def test_a_modified_coefficients_file_is_refused(coef, tmp_path):
    path, sha = coef
    bad = tmp_path / "bad.json"
    data = json.loads(path.read_text())
    data["outcomes"]["home"]["dev_VC"] += 0.01
    bad.write_text(json.dumps(data, indent=2))
    with pytest.raises(CoefficientHashError):
        R6LiveModel.load(bad, expected_sha256=sha)
    model = R6LiveModel.load(path, expected_sha256=sha)
    assert model.sha256 == sha
    assert model.books == {"WH": "williamhill", "VC": "betvictor", "1XB": "onexbet", "BF": "betfair_sb_uk"}


def test_the_guard_defaults_to_the_registered_hash(coef, store):
    """Without an explicit hash, the model and both arms accept only the registered file."""
    path, _ = coef
    with pytest.raises(CoefficientHashError, match=COEFFICIENTS_SHA256):
        R6LiveModel.load(path)
    store.add_quotes(_response(KO - timedelta(hours=40), BOOKS))
    snap = Snapshot.from_store(store, as_of=KO - timedelta(hours=40) + timedelta(minutes=5), code_version="sha")
    for arm in SoftVsSharpLiveV2.ARMS:
        with pytest.raises(CoefficientHashError):
            SoftVsSharpLiveV2(arm, coefficients_path=path).decide(snap)


# ---------------------------------------------------------------- the model
def test_golden_p_fair_on_a_hand_built_response(coef):
    path, sha = coef
    model = R6LiveModel.load(path, expected_sha256=sha)
    out = model.fair_value(PIN, BOOKS)
    expected = _expected_p_fair(PIN, BOOKS, SYNTHETIC_COEFFICIENTS)
    assert out.p_fair == pytest.approx(expected, abs=1e-12)
    assert sum(out.p_fair) == pytest.approx(1.0)
    # pinned numbers, so a silent change in either implementation is caught
    assert out.p_fair == pytest.approx(PINNED_P_FAIR, abs=1e-6)
    assert out.miss == {"WH": 0, "VC": 0, "1XB": 0, "BF": 1}
    assert out.dev["BF"] == [0.0, 0.0, 0.0]
    assert out.dev["VC"][0] < 0 < out.dev["VC"][2]  # betvictor longer on home (2.20 vs 2.10), shorter on away


def test_deviation_beyond_the_clip_is_clipped_and_the_fit_encoding_is_kept_as_a_diagnostic(coef):
    path, sha = coef
    model = R6LiveModel.load(path, expected_sha256=sha)
    books = {**BOOKS, "betvictor": (1.40, 4.50, 7.50)}  # far from Pinnacle: |dev_home| > 0.5
    out = model.fair_value(PIN, books)
    assert out.dev["VC"][0] == 0.5 and out.clipped["VC"] is True
    assert out.p_fair == pytest.approx(_expected_p_fair(PIN, books, SYNTHETIC_COEFFICIENTS), abs=1e-12)
    assert out.p_fair_fit_encoding != pytest.approx(out.p_fair)  # fit treated |dev| > 0.5 as missing


# ---------------------------------------------------------------- the strategy
def _response(captured, books, pin=PIN, event="e1", kickoff=KO):
    quotes = []
    for book, odds in {"pinnacle": pin, **books}.items():
        stamp = (captured - timedelta(minutes=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for s, o in zip(SEL, odds, strict=True):
            quotes.append(
                Quote(
                    source=f"odds_api:{book}",
                    source_event_id=event,
                    home="Ipswich Town",
                    away="Nottingham Forest",
                    league="EPL",
                    kickoff_at=kickoff,
                    selection=s,
                    captured_at=captured,
                    decimal_odds=o,
                    raw_json=json.dumps({"last_update": stamp}),
                )
            )
    return quotes


@pytest.fixture
def store(tmp_path):
    s = PaperStore(tmp_path / "paper.duckdb")
    yield s
    s.close()


def _arms(coef):
    path, sha = coef
    return tuple(
        SoftVsSharpLiveV2(arm, coefficients_path=path, coefficients_sha256=sha) for arm in SoftVsSharpLiveV2.ARMS
    )


def test_both_arms_decide_on_the_first_response_20_to_76h_before_kickoff(store, coef):
    p_pin = devig([list(PIN)], "power").probs[0]
    juicy = {**BOOKS, "williamhill": (round(1.05 / p_pin[0], 2), 3.30, 3.80)}  # > 1.02 / p_fair under both arms
    store.add_quotes(_response(KO - timedelta(hours=80), juicy))  # too early: outside the window
    first = KO - timedelta(hours=70)
    store.add_quotes(_response(first, BOOKS))  # first in-window response: nothing qualifies
    later = KO - timedelta(hours=30)
    store.add_quotes(_response(later, juicy))  # later in-window response must not be used
    pin_arm, model_arm = _arms(coef)
    snap = Snapshot.from_store(store, as_of=later + timedelta(minutes=5), code_version="sha")
    assert pin_arm.decide(snap) == [] and model_arm.decide(snap) == []


def test_arms_record_fair_value_book_deviations_and_whether_the_other_arm_bet(store, coef):
    p_pin = devig([list(PIN)], "power").probs[0]
    price = round(1.05 / p_pin[0], 2)
    books = {**BOOKS, "williamhill": (price, 3.30, 3.80)}
    captured = KO - timedelta(hours=50)
    store.add_quotes(_response(captured, books))
    pin_arm, model_arm = _arms(coef)
    snap = Snapshot.from_store(store, as_of=captured + timedelta(minutes=10), code_version="sha")
    [dp] = pin_arm.decide(snap)
    model_fair = R6LiveModel.load(coef[0], expected_sha256=coef[1]).fair_value(PIN, books).p_fair
    md = model_arm.decide(snap)
    assert dp.strategy_id == "soft-vs-sharp-live-v2-pinnacle"
    assert dp.hypothesis_id == "bookmaker-line-structure-live-soft-vs-sharp-v2"
    assert dp.selection == "home" and dp.venue == "odds_api:williamhill" and dp.price_taken == price
    assert dp.params["arm"] == "pinnacle" and dp.params["p_fair"] == pytest.approx(list(p_pin))
    assert dp.params["other_arm_bets"] is bool(md)
    assert dp.params["response_captured_at"] == captured.isoformat()
    assert dp.params["hours_before_kickoff"] == pytest.approx(50.0)
    assert set(dp.params["model"]["dev"]) == {"WH", "VC", "1XB", "BF"} and dp.params["model"]["miss"]["BF"] == 1
    assert dp.params["coefficients_sha256"] == coef[1]
    if md:
        assert md[0].params["arm"] == "r6_live_model" and md[0].params["other_arm_bets"] is True
        assert md[0].params["p_fair"] == pytest.approx(model_fair)


def test_rule_caps_at_ten_percent_over_pinnacle_and_ignores_exchanges(store, coef):
    captured = KO - timedelta(hours=40)
    books = {**BOOKS, "williamhill": (PIN[0] * 1.11, 3.3, 3.8), "betfair_ex_uk": (2.6, 3.6, 4.0)}
    store.add_quotes(_response(captured, books))
    pin_arm, _ = _arms(coef)
    snap = Snapshot.from_store(store, as_of=captured + timedelta(minutes=10), code_version="sha")
    assert pin_arm.decide(snap) == []


def test_a_stale_first_response_is_a_missed_opportunity(store, coef):
    p_pin = devig([list(PIN)], "power").probs[0]
    captured = KO - timedelta(hours=50)
    store.add_quotes(_response(captured, {**BOOKS, "williamhill": (round(1.05 / p_pin[0], 2), 3.3, 3.8)}))
    pin_arm, _ = _arms(coef)
    snap = Snapshot.from_store(store, as_of=captured + timedelta(hours=2), code_version="sha")
    assert pin_arm.decide(snap) == []


def test_a_bad_coefficients_hash_stops_both_arms_without_stopping_other_strategies(store, coef, tmp_path):
    path, sha = coef
    bad = tmp_path / "bad.json"
    bad.write_text(path.read_text().replace("0.45", "0.46"))
    assert hashlib.sha256(bad.read_bytes()).hexdigest() != sha
    store.add_quotes(_response(KO - timedelta(hours=40), BOOKS))
    arms = [SoftVsSharpLiveV2(arm, coefficients_path=bad, coefficients_sha256=sha) for arm in SoftVsSharpLiveV2.ARMS]
    added = run_decide(store, arms, now=KO - timedelta(hours=40) + timedelta(minutes=5), code_version="sha")
    assert added == {"soft-vs-sharp-live-v2-pinnacle": 0, "soft-vs-sharp-live-v2-r6_live_model": 0}
    assert store.decide_log()["error"].str.contains("CoefficientHashError").all()


def test_both_arms_are_registered_builtins_and_v1_is_unchanged():
    reg = available_strategies()
    assert {"soft-vs-sharp-live", "soft-vs-sharp-live-v2-pinnacle", "soft-vs-sharp-live-v2-r6_live_model"} <= set(reg)


def test_informative_book_deviations_move_the_model_arm_to_a_different_bet(store, coef):
    # betvictor and betfair_sb_uk quote the home side longer than Pinnacle. Under de-vigged Pinnacle the best
    # home price (betvictor 2.30, an informative book) is +6.4%; the model follows the informative books,
    # lowers p_home (0.4627 -> 0.4415) and raises p_away, so its largest edge is the away side at bet365 (+4.0%).
    books = {
        "betvictor": (2.30, 3.20, 3.30),
        "betfair_sb_uk": (2.28, 3.25, 3.30),
        "onexbet": (2.15, 3.45, 3.55),
        "bet365": (2.00, 3.40, 3.90),
        "unibet_uk": (2.10, 3.30, 3.60),
        "williamhill": (2.10, 3.30, 3.80),
    }
    captured = KO - timedelta(hours=30)
    store.add_quotes(_response(captured, books))
    pin_arm, model_arm = _arms(coef)
    snap = Snapshot.from_store(store, as_of=captured + timedelta(minutes=10), code_version="sha")
    [dp] = pin_arm.decide(snap)
    [dm] = model_arm.decide(snap)
    assert (dp.selection, dp.params["book"], dp.params["informative_book"]) == ("home", "betvictor", True)
    assert (dm.selection, dm.params["book"], dm.params["informative_book"]) == ("away", "bet365", False)
    assert dp.params["other_arm_selection"] == "away" and dm.params["other_arm_selection"] == "home"
    assert dm.params["edge"] == pytest.approx(3.90 * dm.params["p_fair"][2] - 1)
    assert dm.params["model"]["miss"] == {"WH": 0, "VC": 0, "1XB": 0, "BF": 0}
    assert pin_arm.last_stats == {"n_fixtures_evaluated": 1, "n_fixtures_qualifying": 1}


# ---------------------------------------------------------------- report
V2 = "bookmaker-line-structure-live-soft-vs-sharp-v2"


class _Ledger:
    def registration(self, hypothesis_id):
        return {"id": V2, "recorded_at": datetime(2026, 10, 21, 7, 13, 57)} if hypothesis_id == V2 else None


def _v2_decision(store, i, arm, decided, selection="home", book="williamhill", clv=None, settle=True):
    from pitch_edge.paper.store import Decision, Settlement

    ko = decided + timedelta(hours=40)
    event = f"v{i}"
    q = Quote(
        source="odds_api:pinnacle",
        source_event_id=event,
        home=f"h{'abcdefghij'[i] * 5}",
        away=f"a{'abcdefghij'[i] * 5}",
        league="EPL",
        kickoff_at=ko,
        selection=selection,
        captured_at=decided,
        decimal_odds=2.0,
    )
    store.add_quotes([q])
    fk = store.quotes().query("source_event_id == @event")["fixture_key"].iloc[0]
    d = Decision(
        strategy_id=f"soft-vs-sharp-live-v2-{arm}",
        hypothesis_id=V2,
        decided_at=decided,
        fixture_key=fk,
        kickoff_at=ko,
        venue=f"odds_api:{book}",
        selection=selection,
        price_taken=2.0,
        paper_stake_units=1.0,
        model_prob=0.5,
        inputs_quote_ids=[q.quote_id],
        code_version="sha",
        params={"arm": arm, "book": book, "informative_book": book in ("betvictor", "betfair_sb_uk")},
    )
    store.append_decision(d)
    if settle:
        store.append_settlement(
            Settlement(
                decision_id=d.decision_id,
                closing_price=2.0,
                fair_close_odds=2.0,
                closing_source="lake:fd_matches.betfair_ex_close",
                result="home",
                clv_log=clv,
                pnl=1.0,
                settled_at=ko + timedelta(days=1),
            )
        )
    return fk


def test_v2_report_per_arm_and_paired_difference(tmp_path):
    from pitch_edge.paper.report import v2_summary
    from pitch_edge.research.stats import block_bootstrap

    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / f"{V2}.yaml").write_text(
        "window:\n  exploratory: [2026-10-21, 2026-10-21]\n  evaluation: [2026-10-22, 2027-06-30]\nmin_n: 150\n"
    )
    with PaperStore(tmp_path / "p.duckdb") as s:
        t = [datetime(2026, 10, 22 + k, 9, tzinfo=UTC) for k in range(5)]
        _v2_decision(s, 0, "pinnacle", t[0], clv=0.03, book="betvictor")  # fixture A: both bet
        s_fk = s.decisions().iloc[0]["fixture_key"]
        # model arm on the same fixture A
        from pitch_edge.paper.store import Decision, Settlement

        q = s.quotes().iloc[0]
        dm = Decision(
            strategy_id="soft-vs-sharp-live-v2-r6_live_model",
            hypothesis_id=V2,
            decided_at=t[0],
            fixture_key=s_fk,
            kickoff_at=t[0] + timedelta(hours=40),
            venue="odds_api:bet365",
            selection="away",
            price_taken=2.0,
            paper_stake_units=1.0,
            model_prob=0.5,
            inputs_quote_ids=[q["quote_id"]],
            code_version="sha",
            params={"arm": "r6_live_model", "book": "bet365", "informative_book": False},
        )
        s.append_decision(dm)
        s.append_settlement(
            Settlement(
                decision_id=dm.decision_id,
                closing_price=2.0,
                fair_close_odds=2.0,
                closing_source="lake:fd_matches.betfair_ex_close",
                result="away",
                clv_log=0.01,
                pnl=1.0,
                settled_at=t[1],
            )
        )
        _v2_decision(s, 1, "pinnacle", t[1], clv=-0.02)  # fixture B: only Pinnacle arm -> model scores 0
        _v2_decision(s, 2, "r6_live_model", t[2], clv=0.04)  # fixture C: only model arm
        _v2_decision(s, 3, "pinnacle", t[3], settle=False)  # fixture D: pending, excluded from the pair
        _v2_decision(s, 4, "pinnacle", datetime(2026, 10, 21, 8, tzinfo=UTC), clv=0.5)  # exploratory window
        rep = v2_summary(s, registry_dir=reg, ledger=_Ledger(), n_boot=500, seed=3)
    arms = rep["arms"].set_index("arm")
    assert arms.loc["pinnacle", "n_decided"] == 3 and arms.loc["pinnacle", "n_final"] == 2
    assert arms.loc["pinnacle", "mean_clv"] == pytest.approx(np.mean([0.03, -0.02]))
    assert arms.loc["pinnacle", "min_n"] == 150
    assert arms.loc["pinnacle", "share_informative"] == pytest.approx(1 / 3)
    assert arms.loc["r6_live_model", "mean_clv"] == pytest.approx(np.mean([0.01, 0.04]))
    paired = rep["paired"]
    diffs = [0.01 - 0.03, 0.0 - (-0.02), 0.04 - 0.0]
    assert paired["n_fixtures"] == 3 and paired["n_pending"] == 1
    assert paired["mean_diff"] == pytest.approx(np.mean(diffs))
    days = [str((x + timedelta(hours=40)).date()) for x in t[:3]]
    boot = block_bootstrap(diffs, days, n_boot=500, seed=3)
    assert (paired["ci_lo"], paired["ci_hi"]) == pytest.approx((boot.lo, boot.hi))
    assert "2026-10-21T07:13:57" in rep["cut"]
