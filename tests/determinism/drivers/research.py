"""Driver: the research harness on fixed synthetic inputs, written as canonical output.

Registers two fixture hypotheses of one family, runs `run_trial` over a fixed list of (hypothesis, grid point,
metric) units with fixed per-decision data, grades both hypotheses and exports the family's ledger chain.

Resume: a unit whose (hypothesis, config hash, metric) already has a trial row in the ledger is skipped, so a run
killed after its n-th `Ledger.record_trial` and started again finishes the remaining units only.

Output: the export lines, the trials table and both grades. Named volatile fields, replaced by `<volatile>`:
`registrations.recorded_at` and `trials.run_at` (wall clock when the row was written), every chain `prev_hash` /
`row_hash` and the summary's `upto` (they hash those stamps), the L0 criteria `registered` (value) and
`registered before first trial` (value and threshold) whose texts print the same stamps. `trial_id` (uuid4) is
replaced by the label `trial:<hypothesis>|<config>|<metric>`, and the run's working directory by `<work>`.
The driver checks the exported chain itself (each `row_hash` is sha256 of the previous hash, the table and the
exact row text) and records the verdict, so the hashes are verified even though their values are not compared.

Defects: `unseeded` builds the per-decision values from an unseeded generator; `skip-on-resume` makes a resumed
run treat the first unfinished unit as done.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "unit")]

import duckdb  # noqa: E402
from canon import VOLATILE, canonical, driver_args, finish, frame_records, kill_after, write_canonical  # noqa: E402
from research_fixtures import per_decision, write_registration  # noqa: E402

from pitch_edge.research.gates import grade  # noqa: E402
from pitch_edge.research.ledger import Ledger  # noqa: E402
from pitch_edge.research.registry import config_hash, load  # noqa: E402
from pitch_edge.research.trial import run_trial  # noqa: E402

FAMILY = "determinism-family"
GRID_ID = "determinism-grid-v1"
SINGLE_ID = "determinism-single-v1"
EXTRA = {"clean": True, "within_ceiling": True}
# (hypothesis, config, metric, mean of the per-decision values, seed of their draw)
UNITS: list[tuple[str, dict[str, Any], str, float, int]] = [
    (GRID_ID, {"threshold": 0.5}, "clv_log", 0.06, 1),
    (GRID_ID, {"threshold": 0.6}, "clv_log", 0.03, 2),
    (GRID_ID, {"threshold": 0.7}, "clv_log", 0.0, 3),
    (GRID_ID, {"threshold": 0.8}, "clv_log", 0.02, 4),
    (GRID_ID, {"threshold": 0.5}, "roi", 0.04, 5),
    (SINGLE_ID, {}, "edge_bits", 0.08, 6),
]
# A return-kind grid of four points, so the deflated Sharpe and PBO gates compute numbers instead of `n/a`.
GRID_REGISTRATION = {
    "family": FAMILY,
    "primary_metric": "clv_log",
    "secondary_metrics": ["roi"],
    "grid": {"threshold": [0.5, 0.6, 0.7, 0.8]},
    "extra_fields": {"metric_kind": "return"},
}
GENESIS = "0" * 64


def _scrub(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, dict):
        return {k: _scrub(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v, replacements) for v in value]
    return value


def main() -> None:
    args = driver_args()
    work: Path = args.work
    registry = work / "registry"
    registry.mkdir(exist_ok=True)
    ledger = Ledger(work / "data" / "research" / "ledger.duckdb")
    regs = {
        GRID_ID: write_registration(registry, GRID_ID, **GRID_REGISTRATION),
        SINGLE_ID: write_registration(registry, SINGLE_ID, family=FAMILY, grid={}),
    }
    loaded = {hid: load(path) for hid, path in regs.items()}
    for reg in loaded.values():
        ledger.register(reg)  # False (a no-op) when a resumed run finds it registered

    if args.kill_after:
        assert args.kill_at == "trial", args.kill_at
        kill_after(Ledger, "record_trial", args.kill_after)

    existing = ledger.trials()
    done = {(r.hypothesis_id, r.config_hash, r.metric_name) for r in existing.itertuples()}
    resumed = bool(done)
    skipped_one = False
    for hid, config, metric, mu, seed in UNITS:
        key = (hid, config_hash(config), metric)
        if key in done:
            continue
        if args.defect == "skip-on-resume" and resumed and not skipped_one:
            skipped_one = True
            continue
        data = per_decision(100, 4, mu, seed=None if args.defect == "unseeded" else seed)
        run_trial(loaded[hid], config, data, metric, ledger, "fixture-snapshot", dict(EXTRA), n_boot=2000, seed=0)

    trials = ledger.trials()
    labels = {str(r.trial_id): f"trial:{r.hypothesis_id}|{r.config_json}|{r.metric_name}" for r in trials.itertuples()}
    con = duckdb.connect(str(ledger.path), read_only=True)
    try:
        upto = con.execute("SELECT row_hash FROM ledger_chain ORDER BY seq DESC LIMIT 1").fetchone()[0]
    finally:
        con.close()
    lines = ledger.export_chain(FAMILY, upto, ledger.metric_kinds(FAMILY))

    chain_ok, prev = True, GENESIS
    exported: list[dict[str, Any]] = []
    for line in lines:
        if "row_hash" in line:
            text = f"{line['prev_hash']}|{line['table']}|{line['row_canonical']}"
            chain_ok &= line["prev_hash"] == prev and hashlib.sha256(text.encode()).hexdigest() == line["row_hash"]
            prev = line["row_hash"]
        out = dict(line)
        if "row_canonical" in out:
            row = json.loads(out.pop("row_canonical"))
            if out["table"] == "registrations":
                row["recorded_at"] = VOLATILE
            if out["table"] == "trials":
                row["run_at"] = VOLATILE
                row["extra_json"] = json.loads(row["extra_json"])
            out["row"] = row
        for name in ("prev_hash", "row_hash"):
            if name in out:
                out[name] = VOLATILE
        if "summary" in out:
            out["summary"] = {**out["summary"], "upto": VOLATILE}
        exported.append(out)

    grades = {}
    for hid in loaded:
        g = grade(hid, ledger)
        criteria = []
        for c in g.criteria:
            row = {"rung": c.rung, "name": c.name, "status": c.status, "value": canonical(c.value)}
            row |= {"threshold": c.threshold, "detail": c.detail}
            if c.rung == "L0" and c.name == "registered":
                row["value"] = VOLATILE
            if c.rung == "L0" and c.name == "registered before first trial":
                row["value"] = row["threshold"] = VOLATILE
            criteria.append(row)
        grades[hid] = {
            "rung": g.rung,
            "trial_id": g.trial_id,
            "config_hash": g.config_hash,
            "record_trial_id": g.record_trial_id,
            "criteria": criteria,
        }

    result = {
        "chain_verified": chain_ok,
        "export": exported,
        "trials": frame_records(
            trials, volatile=["run_at"], relabel={"trial_id": labels.get, "extra_json": json.loads}
        ),
        "grades": grades,
    }
    write_canonical(args.out, _scrub(canonical(result), {**labels, str(work.resolve()): "<work>", str(work): "<work>"}))


if __name__ == "__main__":
    main()
    finish()
