"""The append-only research ledger: immutable registrations, validated and hash-chained trial/review
rows, read paths that never create or lock the file, and safe concurrent appends from several
processes."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import duckdb
import pytest
from research_fixtures import native_record, write_registration

from pitch_edge.research.errors import LedgerLockTimeout, ProtocolError
from pitch_edge.research.ledger import Ledger, default_ledger_path
from pitch_edge.research.registry import REPO_ROOT, config_hash, load

pytestmark = pytest.mark.unit

HERE = Path(__file__).resolve().parent
HID = "sharp-close-test-v1"


@pytest.fixture
def reg(tmp_path):
    return load(write_registration(tmp_path, grid={"k": [1, 2]}, secondary_metrics=["roi"]))


@pytest.fixture
def ledger(tmp_path, reg):
    led = Ledger(tmp_path / "ledger.duckdb")
    led.register(reg)
    return led


# --------------------------------------------------------------------------- files and read paths


def test_ledger_lives_in_its_own_file(tmp_path, reg):
    path = tmp_path / "research" / "ledger.duckdb"
    Ledger(path).register(reg)
    tables = {r[0] for r in duckdb.connect(str(path), read_only=True).execute("SHOW TABLES").fetchall()}
    assert {"registrations", "trials", "reviews", "ledger_chain"} <= tables


def test_reads_never_create_the_file(tmp_path):
    path = tmp_path / "nowhere" / "ledger.duckdb"
    led = Ledger(path)
    assert led.registration(HID) is None
    assert led.trials().empty and led.reviews().empty and led.registrations().empty
    assert led.verify_chain().ok
    assert not path.exists() and not path.parent.exists()


def test_default_path_is_under_data_research(monkeypatch, tmp_path):
    monkeypatch.setenv("PITCH_EDGE_DATA_DIR", str(tmp_path))
    assert default_ledger_path() == tmp_path / "research" / "ledger.duckdb"


# --------------------------------------------------------------------------- registrations


def test_register_is_idempotent_for_an_unchanged_file(ledger, reg):
    assert ledger.register(load(reg.source_path)) is False
    row = ledger.registration(reg.id)
    assert row is not None
    assert row["sha256"] == reg.source_sha256 and row["family"] == "sharp-close"
    assert isinstance(row["recorded_at"], datetime)
    assert len(ledger.registrations()) == 1


def test_registration_paths_are_repo_relative_inside_the_repo(tmp_path):
    led = Ledger(tmp_path / "l.duckdb")
    real = sorted((REPO_ROOT / "research" / "registry").glob("*.yaml"))
    if not real:
        pytest.skip("no registry files in this checkout")
    led.register(load(real[0]))
    assert led.registration(real[0].stem)["path"] == f"research/registry/{real[0].name}"


def test_register_refuses_an_edited_registration(ledger, reg):
    reg.source_path.write_text(reg.source_path.read_text().replace("min_n: 300", "min_n: 100"))
    with pytest.raises(ProtocolError, match="bump the version"):
        ledger.register(load(reg.source_path))


def test_register_needs_a_file_backed_registration(tmp_path, reg):
    with pytest.raises(ProtocolError, match="file"):
        Ledger(tmp_path / "l.duckdb").register(type(reg).model_validate(reg.model_dump()))


# --------------------------------------------------------------------------- trial validation (I7)


def test_valid_trials_append_and_are_never_collapsed(ledger, reg):
    first, second = native_record(reg), native_record(reg, metric_value=0.06)
    ledger.record_trial(first)
    ledger.record_trial(second)
    trials = ledger.trials(hypothesis_id=reg.id)
    assert list(trials["trial_id"]) == [first.trial_id, second.trial_id]
    with pytest.raises(duckdb.ConstraintException):
        ledger.record_trial(first)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"config": {"k": 3}}, "grid"),
        ({"metric_name": "sharpe_after_hindsight"}, "metric"),
        ({"p_value": 1.5, "extra": {"p_gt0": 1.5}}, "p_value"),
        ({"metric_value": 0.2}, "ci_lo <= metric"),
        ({"p_value": 0.5}, "p_gt0"),
        ({"direction": "negative"}, "direction"),
        ({"family": "elsewhere"}, "family"),
        ({"config_hash": "0" * 64}, "config_hash"),
        ({"extra": {"n_blocks": None}}, "n_blocks"),
        ({"extra": {"registration_sha256": "f" * 64}}, "registration_sha256"),
    ],
)
def test_record_trial_refuses_inconsistent_rows(ledger, reg, override, match):
    over = dict(override)
    cfg = over.pop("config", None)
    if cfg is not None:
        over.update(config_json=json.dumps(cfg), config_hash=config_hash(cfg))
    with pytest.raises(ProtocolError, match=match):
        ledger.record_trial(native_record(reg, **over))
    assert ledger.trials().empty


def test_record_trial_refuses_after_the_registration_file_changed(ledger, reg):
    reg.source_path.write_text(reg.source_path.read_text() + "\n# edited\n")
    with pytest.raises(ProtocolError, match="edited"):
        ledger.record_trial(native_record(reg))


def test_unregistered_hypothesis_is_refused(tmp_path, reg):
    with pytest.raises(ProtocolError, match="not registered"):
        Ledger(tmp_path / "other.duckdb").record_trial(native_record(reg))


def test_imported_rows_may_be_off_grid_but_must_be_marked(ledger, reg):
    extra = {"imported": True, "window": "exploratory", "is_primary_metric": True}
    cfg = {"source": "max_open", "theta": 0.02}
    rec = native_record(
        reg,
        config_json=json.dumps(cfg),
        config_hash=config_hash(cfg),
        metric_name="clv_log_mean",
        extra_json=json.dumps(extra),
    )
    ledger.record_trial(rec)
    assert len(ledger.trials()) == 1


# --------------------------------------------------------------------------- reviews


def test_reviews_append_with_a_closed_set_of_kinds(ledger, reg):
    ledger.record_review(reg.id, "audit", "SIGNED", {"file": "research/audits/x.md"})
    ledger.record_review(reg.id, "edgelab", "promising")
    reviews = ledger.reviews(reg.id)
    assert list(reviews["kind"]) == ["audit", "edgelab"]
    assert json.loads(reviews["detail_json"].iloc[0]) == {"file": "research/audits/x.md"}
    with pytest.raises(ProtocolError, match="kind"):
        ledger.record_review(reg.id, "vibes", "great")
    with pytest.raises(ProtocolError, match="not registered"):
        ledger.record_review("nope-v1", "audit", "SIGNED")


def test_an_amendment_is_accepted_only_before_the_first_trial(ledger, reg):
    ledger.record_review(reg.id, "amendment", "RECORDED", {"text": "drop incomplete seasons"})
    assert list(ledger.reviews(reg.id)["kind"]) == ["amendment"]
    ledger.record_trial(native_record(reg))
    with pytest.raises(ProtocolError, match="already has trials"):
        ledger.record_review(reg.id, "amendment", "RECORDED", {"text": "too late"})
    assert ledger.verify_chain().ok


def test_ledger_has_no_mutation_api():
    public = [name for name in dir(Ledger) if not name.startswith("_")]
    forbidden = re.compile(r"update|delete|remove|drop|clear|truncate|edit|replace|overwrite|reset|execute|sql|conn")
    assert not [name for name in public if forbidden.search(name)], public


# --------------------------------------------------------------------------- hash chain (I7)


def test_chain_is_intact_after_normal_use(ledger, reg):
    ledger.record_trial(native_record(reg))
    ledger.record_review(reg.id, "audit", "SIGNED")
    report = ledger.verify_chain()
    assert report.ok and report.n_chained == 3 and not report.broken


@pytest.mark.parametrize(
    "tamper",
    [
        "UPDATE trials SET metric_value = 0.5",
        "DELETE FROM trials",
        "UPDATE registrations SET sha256 = 'x'",
        "UPDATE reviews SET verdict = 'SIGNED'",
    ],
)
def test_chain_detects_tampering(ledger, reg, tamper):
    ledger.record_trial(native_record(reg))
    ledger.record_review(reg.id, "audit", "REJECTED")
    con = duckdb.connect(str(ledger.path))
    con.execute(tamper)
    con.close()
    report = ledger.verify_chain()
    assert not report.ok and report.broken


def test_chain_flags_rows_written_around_the_api(ledger, reg):
    ledger.record_trial(native_record(reg))
    rec = native_record(reg)
    con = duckdb.connect(str(ledger.path))
    con.execute(
        "INSERT INTO trials (trial_id, hypothesis_id, family, config_json, config_hash, git_sha, data_snapshot, run_at, "
        "n, metric_name, metric_value, ci_lo, ci_hi, p_value, direction, extra_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, now()::TIMESTAMP + INTERVAL 1 HOUR, ?, ?, ?, ?, ?, ?, ?, ?)",
        [rec.trial_id, rec.hypothesis_id, rec.family, rec.config_json, rec.config_hash, rec.git_sha, rec.data_snapshot,
         rec.n, rec.metric_name, rec.metric_value, rec.ci_lo, rec.ci_hi, rec.p_value, rec.direction, rec.extra_json],
    )  # fmt: skip
    con.close()
    report = ledger.verify_chain()
    assert not report.ok
    assert report.unchained_after_start == {"trials": [rec.trial_id]}


# --------------------------------------------------------------------------- migration


_OLD_SCHEMA = """
CREATE TABLE registrations (id VARCHAR PRIMARY KEY, family VARCHAR NOT NULL, sha256 VARCHAR NOT NULL,
    path VARCHAR NOT NULL, registered_at TIMESTAMP NOT NULL, recorded_at TIMESTAMP NOT NULL);
CREATE TABLE trials (trial_id VARCHAR PRIMARY KEY, hypothesis_id VARCHAR NOT NULL, family VARCHAR NOT NULL,
    config_json VARCHAR NOT NULL, config_hash VARCHAR NOT NULL, git_sha VARCHAR NOT NULL,
    data_snapshot VARCHAR NOT NULL, run_at TIMESTAMP NOT NULL, n BIGINT NOT NULL, metric_name VARCHAR NOT NULL,
    metric_value DOUBLE, ci_lo DOUBLE, ci_hi DOUBLE, p_value DOUBLE, direction VARCHAR NOT NULL,
    extra_json VARCHAR NOT NULL);
CREATE TABLE reviews (hypothesis_id VARCHAR NOT NULL, kind VARCHAR NOT NULL, verdict VARCHAR NOT NULL,
    detail_json VARCHAR NOT NULL, recorded_at TIMESTAMP NOT NULL);
"""


def _dump(path: Path) -> dict[str, list[tuple]]:
    con = duckdb.connect(str(path), read_only=True)
    try:
        return {
            t: con.execute(f"SELECT * FROM {t} ORDER BY ALL").fetchall() for t in ("registrations", "trials", "reviews")
        }
    finally:
        con.close()


def test_migration_of_a_first_version_ledger_is_additive(tmp_path, reg):
    path = tmp_path / "old.duckdb"
    con = duckdb.connect(str(path))
    con.execute(_OLD_SCHEMA)
    con.execute(
        "INSERT INTO registrations VALUES (?, ?, ?, ?, now()::TIMESTAMP - INTERVAL 1 DAY, now()::TIMESTAMP - INTERVAL 1 DAY)",
        [reg.id, reg.family, reg.source_sha256, str(reg.source_path)],
    )
    con.execute("INSERT INTO reviews VALUES (?, 'audit', 'PENDING', '{}', now()::TIMESTAMP - INTERVAL 1 DAY)", [reg.id])
    con.close()
    before = _dump(path)

    led = Ledger(path)
    led.record_trial(native_record(reg))
    after = _dump(path)
    assert after["registrations"] == before["registrations"]
    assert after["reviews"] == before["reviews"]
    assert len(after["trials"]) == 1
    report = led.verify_chain()
    assert report.ok
    assert report.legacy_unchained == {"registrations": 1, "reviews": 1}


def test_migration_on_a_copy_of_the_live_ledger(tmp_path):
    live = REPO_ROOT / "data" / "research" / "ledger.duckdb"
    if not live.exists():
        pytest.skip("no live ledger in this checkout")
    copy = tmp_path / "live-copy.duckdb"
    shutil.copy(live, copy)
    wal = live.with_suffix(".duckdb.wal")
    if wal.exists():
        shutil.copy(wal, copy.with_suffix(".duckdb.wal"))
    before = _dump(copy)
    led = Ledger(copy)
    report = led.verify_chain()  # read-only: no migration yet
    assert report.ok
    reg = load(write_registration(tmp_path, "migration-probe-v1", family="migration-probe"))
    led.register(reg)
    led.record_trial(native_record(reg))
    after = _dump(copy)
    for table in ("registrations", "trials", "reviews"):
        assert set(before[table]) <= set(after[table]), table
    assert len(after["trials"]) == len(before["trials"]) + 1
    assert led.verify_chain().ok


# --------------------------------------------------------------------------- concurrency (I6)


WORKER = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {here!r})
    from research_fixtures import native_record
    from pitch_edge.research.ledger import Ledger
    from pitch_edge.research.registry import load

    led = Ledger({ledger!r}, lock_timeout_s=60)
    reg = load({reg!r})
    for i in range({n}):
        led.record_trial(native_record(reg, data_snapshot="w{tag}-" + str(i)))
        led.trials(hypothesis_id=reg.id)  # interleave reads with writes
    print("done")
    """
)


def test_two_processes_appending_concurrently_lose_nothing(tmp_path, ledger, reg):
    n = 60
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                WORKER.format(here=str(HERE), ledger=str(ledger.path), reg=str(reg.source_path), n=n, tag=t),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for t in "AB"
    ]
    outs = [p.communicate(timeout=240) for p in procs]
    assert all(p.returncode == 0 for p in procs), [o[1][-2000:] for o in outs]
    assert len(ledger.trials()) == 2 * n
    assert ledger.verify_chain().ok


HOLDER = textwrap.dedent(
    """
    import sys, time, duckdb
    con = duckdb.connect({ledger!r})
    print("locked", flush=True)
    time.sleep({hold})
    con.close()
    """
)


def test_writes_give_up_after_the_lock_budget(tmp_path, ledger, reg):
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER.format(ledger=str(ledger.path), hold=4)], stdout=subprocess.PIPE, text=True
    )
    assert holder.stdout is not None and holder.stdout.readline().strip() == "locked"
    impatient = Ledger(ledger.path, lock_timeout_s=0.5)
    start = time.monotonic()
    with pytest.raises(LedgerLockTimeout):
        impatient.record_review(reg.id, "audit", "SIGNED")
    assert time.monotonic() - start < 3
    patient = Ledger(ledger.path, lock_timeout_s=30)
    patient.record_review(reg.id, "audit", "SIGNED")  # waits for the holder to let go
    holder.wait(timeout=30)
    assert len(patient.reviews(reg.id)) == 1
