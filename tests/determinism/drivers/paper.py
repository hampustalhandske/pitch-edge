"""Driver: the paper ledger (collect, decide, settle, report) on a fixed schedule of synthetic quotes.

Sixteen fixtures (fourteen Premier League, two Champions League, kickoffs nine hours apart) with Pinnacle and two
soft-book prices. The schedule: three hourly cycles that each collect the quotes captured by then (inserted
directly, no HTTP, with a `collect_log` row) and run `run_decide` for `control-fav-pinnacle` and
`soft-vs-sharp-live`; one collect of the quotes captured two hours before each kickoff (one of them moves a kickoff
by two days, a postponement); then `settle` at two fixed times against a fake close source (football-data Betfair
closes and results for every Premier League fixture but the postponed one and one the lake never closes, a
Pinnacle close for one Champions League fixture and an unusable one for the other).
Every clock is fixed: quotes carry their capture times, decisions their cycle time, settlements and flags the
settle time. The decision rows get a fixed `code_version` and `code_hash`.

Resume: a step of the schedule is recorded in `progress.json` once it has finished; a resumed run skips recorded
steps and re-runs the one that was interrupted (decide skips fixtures a strategy already decided, settle skips
settled decisions).

Decision ids: production draws each `decision_id` from uuid4, and `PaperStore.decisions()` orders rows of one
`decided_at` by that id, which `report.summarise` then sums in (`mean_clv`, the bootstrap block sums), so the
report's last bits follow the random ids. The driver therefore gives every decision the id
`sha256("<strategy_id>|<fixture_key>")[:32]` (unique, since a strategy decides a fixture once, and the same in a
resumed run), which keeps every other source of nondeterminism visible to the comparison; the `salted-ids` defect
hashes a salt in front, which reorders the ids and, for these fixtures, changes the last bit of a `mean_clv`.

Output: every table of the paper ledger, the `decisions.jsonl` chain records and its verdict, and
`report.summarise` (n_boot 500, seed 0, fixed `now`). Named volatile fields: `decision_id` is replaced everywhere
by `<strategy_id>|<fixture_key>`, and the chain's `hash` / `prev_hash`, which hash the decision ids, by
`<volatile>`; the chain is verified by `verify_decision_chain` and its verdict compared.

Defects: `unseeded` runs the report's bootstrap with an unseeded generator; `checkpoint-first` records each step
in `progress.json` before running it, so a resumed run skips the step the kill interrupted; `salted-ids` gives the
decisions the ids `sha256("order-7|<strategy_id>|<fixture_key>")[:32]`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from canon import VOLATILE, driver_args, dump_duckdb, finish, frame_records, kill_after, write_canonical  # noqa: E402

from pitch_edge.paper.report import summarise  # noqa: E402
from pitch_edge.paper.settle import FdClose, PinnacleClose, settle  # noqa: E402
from pitch_edge.paper.store import PaperStore, Quote, verify_decision_chain  # noqa: E402
from pitch_edge.paper.strategies import ControlFavPinnacle, SoftVsSharpLive, run_decide  # noqa: E402

T0 = datetime(2026, 10, 22, 10, 0, tzinfo=UTC)
CYCLES = [T0 + timedelta(hours=h) for h in range(3)]
SELECTIONS = ("home", "draw", "away")
N_FIXTURES = 16
LEAGUES = ["EPL"] * (N_FIXTURES - 2) + ["UCL"] * 2


def _pinnacle(i: int) -> tuple[float, float, float]:
    """Pinnacle 1X2 odds of fixture `i`: a home probability stepping through 0.30..0.51, 2.5% margin."""
    home = 0.30 + 0.03 * (i % 8)
    probs = (home, 0.27, 1.0 - home - 0.27)
    return tuple(round(1.0 / (p * 1.025), 2) for p in probs)  # type: ignore[return-value]


PINNACLE = [_pinnacle(i) for i in range(N_FIXTURES)]
POSTPONED = 5
NO_CLOSE = 4  # a Premier League fixture the lake never closes
FD_RESULTS = {i: SELECTIONS[i % 3] for i in range(N_FIXTURES - 2) if i not in (NO_CLOSE, POSTPONED)}
PINNACLE_CLOSE_OK = N_FIXTURES - 2  # the other Champions League fixture's close is unusable
HYPOTHESIS = SoftVsSharpLive.hypothesis_id
REGISTERED = datetime(2026, 10, 20, 20, 54)


def kickoff(i: int) -> datetime:
    return T0 + timedelta(days=2, hours=9 * i)


def club(i: int, side: str) -> str:
    return f"{side}{'abcdefghijklmnop'[i] * 5}"


SETTLES = [kickoff(N_FIXTURES - 1) + timedelta(hours=4), kickoff(N_FIXTURES - 1) + timedelta(days=5)]
REPORT_NOW = SETTLES[-1] + timedelta(hours=1)


def quotes(i: int, captured: datetime, *, late: bool = False) -> list[Quote]:
    """Pinnacle plus two soft books for fixture `i` captured at `captured`. Even fixtures get a bet365 home price
    that clears the soft-vs-sharp margin; odd ones do not."""
    pin = PINNACLE[i]
    if late:
        pin = tuple(round(o * f, 2) for o, f in zip(pin, (0.97, 1.01, 1.04), strict=True))
    ko = kickoff(i) + (timedelta(days=2) if late and i == POSTPONED else timedelta())
    soft_home = round(pin[0] * (1.06 if i % 2 == 0 else 0.95), 2)
    books = {
        "odds_api:pinnacle": pin,
        "odds_api:bet365": (soft_home, round(pin[1] * 0.96, 2), round(pin[2] * 0.96, 2)),
        "odds_api:williamhill": tuple(round(o * 0.97, 2) for o in pin),
    }
    stamp = (captured - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [
        Quote(
            source=src,
            source_event_id=f"ev{i}",
            home=club(i, "h"),
            away=club(i, "a"),
            league=LEAGUES[i],
            kickoff_at=ko,
            selection=sel,
            captured_at=captured,
            decimal_odds=odds,
            raw_json=json.dumps({"last_update": stamp}),
        )
        for src, prices in books.items()
        for sel, odds in zip(SELECTIONS, prices, strict=True)
    ]


class FixedCloses:
    """Close source answering from fixed values per fixture (matched on the home club)."""

    def _index(self, home: str) -> int:
        return next(i for i in range(len(LEAGUES)) if club(i, "h") == home)

    def fd_match(self, league: str, kickoff_at: datetime, home: str, away: str) -> FdClose | None:
        i = self._index(home)
        if i not in FD_RESULTS:
            return None
        close = tuple(round(o * 1.02, 2) for o in PINNACLE[i])
        return FdClose(betfair_close=close, result=FD_RESULTS[i])  # type: ignore[arg-type]

    def pinnacle_close(self, league: str, kickoff_at: datetime, home: str, away: str) -> PinnacleClose:
        i = self._index(home)
        if i == PINNACLE_CLOSE_OK:
            return PinnacleClose(odds=tuple(round(o * 0.99, 2) for o in PINNACLE[i]))  # type: ignore[arg-type]
        return PinnacleClose(odds=None, reason="draw outcome inactive at the latest observed_at")

    def pm_last_bars(self, asset_ids: list[str], before: datetime) -> dict[str, Any]:
        return {}


class FixedLedger:
    def registration(self, hypothesis_id: str) -> dict[str, Any] | None:
        if hypothesis_id != HYPOTHESIS:
            return None
        return {"id": HYPOTHESIS, "recorded_at": REGISTERED, "registered_at": REGISTERED}


def pin_decision_ids(salt: str) -> None:
    """Make both strategies return their decisions with the content-derived id `sha256(salt + "<strategy>|<fixture>")`
    (see the module docstring)."""

    def decision_id(strategy_id: str, fixture: str) -> str:
        return hashlib.sha256(f"{salt}{strategy_id}|{fixture}".encode()).hexdigest()[:32]

    for cls in (ControlFavPinnacle, SoftVsSharpLive):
        original = cls.decide

        def decide(self: Any, snapshot: Any, original: Any = original) -> list[Any]:
            return [
                dataclasses.replace(d, decision_id=decision_id(d.strategy_id, d.fixture_key))
                for d in original(self, snapshot)
            ]

        cls.decide = decide  # type: ignore[method-assign]


def collect(store: PaperStore, batch: list[Quote], at: datetime) -> None:
    n = store.add_quotes(batch)
    store.record_call("odds_api", "collect", status="ok", called_at=at, n_quotes=n)


def schedule() -> list[tuple[str, Any]]:
    steps: list[tuple[str, Any]] = []
    for c, at in enumerate(CYCLES):
        batch = [q for i in range(len(LEAGUES)) if i % 3 == c for q in quotes(i, at - timedelta(minutes=10))]
        steps.append((f"collect@{at.isoformat()}", lambda s, b=batch, t=at: collect(s, b, t)))
        steps.append(
            (
                f"decide@{at.isoformat()}",
                lambda s, t=at: run_decide(
                    s, [ControlFavPinnacle(), SoftVsSharpLive()], t, code_version="fixture", code_hash="fixture-hash"
                ),
            )
        )
    late = [q for i in range(len(LEAGUES)) for q in quotes(i, kickoff(i) - timedelta(hours=2), late=True)]
    steps.append(("collect@late", lambda s: collect(s, late, kickoff(N_FIXTURES - 1) - timedelta(hours=1))))
    for at in SETTLES:
        steps.append((f"settle@{at.isoformat()}", lambda s, t=at: settle(s, FixedCloses(), t)))
    return steps


def main() -> None:
    args = driver_args()
    work: Path = args.work
    registry = work / "registry"
    registry.mkdir(exist_ok=True)
    (registry / f"{HYPOTHESIS}.yaml").write_text(
        "window:\n  exploratory: [2026-10-20, 2026-10-20]\n  evaluation: [2026-10-21, 2027-06-30]\n"
    )
    db = work / "data" / "paper" / "paper.duckdb"
    progress_path = work / "progress.json"
    progress: list[str] = json.loads(progress_path.read_text()) if progress_path.exists() else []

    def mark(name: str) -> None:
        progress.append(name)
        tmp = progress_path.with_name("progress.json.partial")
        tmp.write_text(json.dumps(progress))
        tmp.replace(progress_path)

    pin_decision_ids("order-7|" if args.defect == "salted-ids" else "")
    if args.kill_after:
        target = {"settle": "_append_settlement_row", "decision": "append_decision"}[args.kill_at]
        kill_after(PaperStore, target, args.kill_after)

    with PaperStore(db) as store:
        for name, step in schedule():
            if name in progress:
                continue
            if args.defect == "checkpoint-first":
                mark(name)
                step(store)
            else:
                step(store)
                mark(name)
        report = summarise(
            store,
            registry_dir=registry,
            ledger=FixedLedger(),
            n_boot=500,
            seed=None if args.defect == "unseeded" else 0,  # type: ignore[arg-type]
            now=REPORT_NOW,
        )
        decisions = store.decisions()
    labels = {str(r.decision_id): f"{r.strategy_id}|{r.fixture_key}" for r in decisions.itertuples()}
    by_id = {"decision_id": labels.get}
    tables = dump_duckdb(
        db, relabel=dict.fromkeys(("decisions", "settlements", "provisional_settlements", "decision_flags"), by_id)
    )
    chain_ok, chain_len = verify_decision_chain(db.parent / "decisions.jsonl")
    chain = []
    for line in (db.parent / "decisions.jsonl").read_text().splitlines():
        entry = json.loads(line)
        record = dict(entry["decision"], decision_id=labels[entry["decision"]["decision_id"]])
        chain.append({"decision": record, "hash": VOLATILE, "prev_hash": VOLATILE if entry["prev_hash"] else ""})
    write_canonical(
        args.out,
        {
            "tables": tables,
            "decision_chain": {"verified": chain_ok, "lines": chain_len, "records": chain},
            "report": frame_records(report),
        },
    )


if __name__ == "__main__":
    main()
    finish()
