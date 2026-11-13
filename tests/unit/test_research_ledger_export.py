"""The ledger export lets another project recount a family's tests from the file alone: the chain recomputes
from genesis using only the exported strings, one changed byte breaks every hash after it, unchained family
rows are shown, and the summary counts the family's trials."""

from __future__ import annotations

import hashlib
import json

import duckdb
import pytest
from research_fixtures import native_record, write_registration
from typer.testing import CliRunner

from pitch_edge.cli import app
from pitch_edge.research.errors import ProtocolError
from pitch_edge.research.ledger import GENESIS, Ledger
from pitch_edge.research.registry import load

pytestmark = pytest.mark.unit


def _recompute(lines: list[dict]) -> list[str]:
    """Each chain line's hash recomputed from the previous line's hash and its own exported string."""
    prev, out = GENESIS, []
    for line in lines:
        if "row_hash" not in line:
            continue
        digest = hashlib.sha256(f"{prev}|{line['table']}|{line['row_canonical']}".encode()).hexdigest()
        out.append(digest)
        prev = digest
    return out


@pytest.fixture
def two_families(tmp_path):
    a = load(write_registration(tmp_path, "fam-a-test-v1", family="fam-a", grid={"k": [1, 2]}))
    b = load(write_registration(tmp_path, "fam-b-test-v1", family="fam-b", grid={"k": [1, 2]}))
    led = Ledger(tmp_path / "ledger.duckdb")
    led.register(a)
    led.record_trial(native_record(a, {"k": 1}))
    led.register(b)
    led.record_trial(native_record(b, {"k": 1}))
    led.record_trial(native_record(a, {"k": 2}))
    led.record_trial(native_record(a, {"k": 2}, metric_value=0.06))
    return led, a, b


def _last_hash(led: Ledger) -> str:
    con = duckdb.connect(str(led.path), read_only=True)
    try:
        return con.execute("SELECT row_hash FROM ledger_chain ORDER BY seq DESC LIMIT 1").fetchone()[0]
    finally:
        con.close()


def test_chain_recomputes_from_the_exported_strings_alone(two_families):
    led, a, _ = two_families
    upto = _last_hash(led)
    lines = led.export_chain("fam-a", upto)
    chain = [line for line in lines if "row_hash" in line]
    assert len(chain) == 6  # both families' rows: the chain is global
    assert _recompute(lines) == [line["row_hash"] for line in chain]
    assert chain[0]["prev_hash"] == GENESIS and chain[-1]["row_hash"] == upto
    for line in chain:
        assert json.loads(line["row_canonical"])  # parseable for the fields


def test_one_changed_byte_breaks_every_later_hash(two_families):
    led, _, _ = two_families
    lines = led.export_chain("fam-a", _last_hash(led))
    chain = [line for line in lines if "row_hash" in line]
    tampered = [dict(line) for line in lines]
    i = 2
    target = [line for line in tampered if "row_hash" in line][i]
    s = target["row_canonical"]
    target["row_canonical"] = s[:1] + "x" + s[2:]  # one byte changed
    recomputed = _recompute(tampered)
    assert recomputed[:i] == [line["row_hash"] for line in chain[:i]]
    assert all(r != line["row_hash"] for r, line in zip(recomputed[i:], chain[i:], strict=True))


def test_summary_counts_only_the_family_and_stops_at_upto(two_families):
    led, a, _ = two_families
    lines = led.export_chain("fam-a", _last_hash(led), {a.id: "return"})
    s = lines[-1]["summary"]
    assert s["family"] == "fam-a" and s["n_tests"] == 3 and s["n_distinct_hypothesis_config"] == 2
    assert s["n_return_kind"] == 2 and all(k.startswith(f"{a.id}|") for k in s["test_keys"])

    con = duckdb.connect(str(led.path), read_only=True)
    third = con.execute("SELECT row_hash FROM ledger_chain ORDER BY seq LIMIT 1 OFFSET 2").fetchone()[0]
    con.close()
    early = led.export_chain("fam-a", third)
    assert len([line for line in early if "row_hash" in line]) == 3
    assert early[-1]["summary"]["n_tests"] == 1 and early[-1]["summary"]["n_return_kind"] is None


def test_unchained_family_rows_are_listed_after_the_chain(two_families):
    led, a, _ = two_families
    con = duckdb.connect(str(led.path))
    con.execute("INSERT INTO reviews VALUES (?, 'audit', 'PENDING', '{}', now()::TIMESTAMP - INTERVAL 30 DAY)", [a.id])
    con.close()
    lines = led.export_chain("fam-a", _last_hash(led))
    legacy = [line for line in lines if line.get("chained") is False]
    assert [line["table"] for line in legacy] == ["reviews"]
    assert "row_hash" not in legacy[0] and json.loads(legacy[0]["row_canonical"])["hypothesis_id"] == a.id
    assert lines.index(legacy[0]) > max(i for i, line in enumerate(lines) if "row_hash" in line)


def test_unknown_upto_is_refused(two_families):
    led, _, _ = two_families
    with pytest.raises(ProtocolError, match="not a row_hash"):
        led.export_chain("fam-a", "f" * 64)


def test_cli_writes_jsonl_that_ends_with_the_summary(two_families, tmp_path):
    led, _, _ = two_families
    out = tmp_path / "export" / "fam-a.jsonl"
    res = CliRunner().invoke(
        app,
        [
            "research",
            "export-ledger",
            "--family",
            "fam-a",
            "--upto",
            _last_hash(led),
            "--out",
            str(out),
            "--ledger",
            str(led.path),
        ],
    )
    assert res.exit_code == 0, res.output
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert lines[-1]["summary"]["n_tests"] == 3
    assert _recompute(lines) == [line["row_hash"] for line in lines if "row_hash" in line]


def _yaml_lines(lines: list[dict]) -> list[dict]:
    return [line for line in lines if line.get("table") == "registration_yaml"]


def test_registration_yaml_lines_recompute_to_the_chained_registration_hash(two_families):
    led, a, _ = two_families
    lines = led.export_chain("fam-a", _last_hash(led))
    yamls = _yaml_lines(lines)
    assert [y["hypothesis_id"] for y in yamls] == [a.id]  # only the family's registration
    y = yamls[0]
    assert "row_hash" not in y and "mismatch" not in y
    assert hashlib.sha256(y["yaml"].encode("utf-8")).hexdigest() == y["sha256"]
    reg_row = next(
        json.loads(line["row_canonical"])
        for line in lines
        if line.get("table") == "registrations" and "row_hash" in line
    )
    assert reg_row["id"] == a.id and reg_row["sha256"] == y["sha256"]
    last_chain = max(i for i, line in enumerate(lines) if "row_hash" in line)
    assert lines.index(y) > last_chain and lines.index(y) < lines.index(lines[-1])


def test_unchained_registrations_export_their_yaml_marked_unchained(two_families, tmp_path):
    led, _, _ = two_families
    old = write_registration(tmp_path, "fam-a-legacy-v1", family="fam-a", grid={"k": [1]})
    digest = hashlib.sha256(old.read_bytes()).hexdigest()
    con = duckdb.connect(str(led.path))
    con.execute(
        "INSERT INTO registrations VALUES ('fam-a-legacy-v1', 'fam-a', ?, ?, now()::TIMESTAMP - INTERVAL 30 DAY, "
        "now()::TIMESTAMP - INTERVAL 30 DAY)",
        [digest, str(old)],
    )
    con.close()
    y = next(
        y for y in _yaml_lines(led.export_chain("fam-a", _last_hash(led))) if y["hypothesis_id"] == "fam-a-legacy-v1"
    )
    assert y["chained"] is False and hashlib.sha256(y["yaml"].encode("utf-8")).hexdigest() == digest


def test_edited_registration_file_exports_with_mismatch(two_families):
    led, a, _ = two_families
    a.source_path.write_text(a.source_path.read_text() + "\n# edited after registering\n")
    lines = led.export_chain("fam-a", _last_hash(led))
    y = _yaml_lines(lines)[0]
    assert y["yaml"] is None and y["mismatch"] is True
    assert len(y["sha256"]) == 64 and lines[-1]["summary"]["n_tests"] == 3


def test_missing_registration_file_exports_with_mismatch(two_families):
    led, a, _ = two_families
    a.source_path.unlink()
    y = _yaml_lines(led.export_chain("fam-a", _last_hash(led)))[0]
    assert y["yaml"] is None and y["mismatch"] is True


def test_return_test_keys_hold_only_return_kind_keys(two_families):
    led, a, _ = two_families
    upto = _last_hash(led)
    s = led.export_chain("fam-a", upto, {a.id: "return"})[-1]["summary"]
    assert s["return_test_keys"] == s["test_keys"] and s["n_return_kind"] == len(s["return_test_keys"]) == 2
    s = led.export_chain("fam-a", upto, {a.id: "predictive"})[-1]["summary"]
    assert s["return_test_keys"] == [] and s["n_return_kind"] == 0
    s = led.export_chain("fam-a", upto)[-1]["summary"]
    assert s["return_test_keys"] is None and s["n_return_kind"] is None
