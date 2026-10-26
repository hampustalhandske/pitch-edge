"""Importing study-local `trials.csv` files (R1 and R2 formats) into the ledger as `imported` trials."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from research_fixtures import write_registration

from pitch_edge.research.gates import grade
from pitch_edge.research.importer import METRIC_ALIASES, import_trials_csv
from pitch_edge.research.ledger import Ledger
from pitch_edge.research.registry import config_hash, load

pytestmark = pytest.mark.unit

HID = "sharp-close-test-v1"

R1_HEADER = "id,config_hash,config,registration_sha256,git_sha,data_snapshot,run_at,window,n,n_days,metric,value,ci_lo,ci_hi,p_one_sided,direction"
R2_HEADER = "id,config_hash,git_sha,run_at,n,metric,ci_lo,ci_hi,metric_name,window,label,direction,p_value,sr,skew,kurt,registration_sha256,data_snapshot,config_json"


def _csv(path, header: str, rows: list[dict]) -> None:
    cols = header.split(",")
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


@pytest.fixture
def setup(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    reg = load(
        write_registration(
            tmp_path, grid={"theta": [0.02, 0.04], "side": [True, False]}, primary_metric="clv_log_vs_close"
        )
    )
    ledger.register(reg)
    return ledger, reg


def _r1_row(reg, **over):
    row = {
        "id": reg.id,
        "config_hash": "abc",
        "config": json.dumps({"theta": 0.02, "side": True, "max_stale_hours": 6}),
        "registration_sha256": reg.source_sha256,
        "git_sha": "6d29de4",
        "data_snapshot": "snap",
        "run_at": "2026-10-20T22:42:14+00:00",
        "window": "evaluation",
        "n": 469,
        "n_days": 126,
        "metric": "clv_log_vs_close",
        "value": 0.003,
        "ci_lo": -0.003,
        "ci_hi": 0.009,
        "p_one_sided": 0.17,
        "direction": "positive",
    }
    row.update(over)
    return row


def test_r1_rows_map_onto_registered_grid_points(setup, tmp_path):
    ledger, reg = setup
    csv = tmp_path / "trials.csv"
    _csv(csv, R1_HEADER, [_r1_row(reg), _r1_row(reg, window="exploratory", value=0.001, ci_lo=-0.01, ci_hi=0.01)])
    report = import_trials_csv(csv, reg.id, ledger)
    assert report.imported == 2 and report.duplicates == 0 and not report.refused
    t = ledger.trials(hypothesis_id=reg.id)
    assert set(t["config_hash"]) == {config_hash({"theta": 0.02, "side": True})}
    extras = [json.loads(e) for e in t["extra_json"]]
    assert sorted(e["window"] for e in extras) == ["evaluation", "exploratory"]
    extra = next(e for e in extras if e["window"] == "evaluation")
    assert extra["imported"] is True and extra["is_primary_metric"] is True
    assert extra["original_config"]["max_stale_hours"] == 6 and extra["n_blocks"] == 126
    assert extra["registration_sha256"] == reg.source_sha256
    assert pd.Timestamp(t.iloc[0]["run_at"]) == pd.Timestamp("2026-10-20T22:42:14")  # original time, UTC


def test_import_is_idempotent(setup, tmp_path):
    ledger, reg = setup
    csv = tmp_path / "trials.csv"
    _csv(csv, R1_HEADER, [_r1_row(reg)])
    import_trials_csv(csv, reg.id, ledger)
    again = import_trials_csv(csv, reg.id, ledger)
    assert again.imported == 0 and again.duplicates == 1
    assert len(ledger.trials()) == 1


def test_r2_rows_off_grid_with_a_built_in_metric_alias(tmp_path):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    reg = load(
        write_registration(
            tmp_path,
            grid={"price_source": ["max_open"]},
            primary_metric="clv_log = log(odds_taken * p_close_devig[o]), mean per bet",
        )
    )
    ledger.register(reg)
    csv = tmp_path / "trials.csv"
    base = {
        "id": reg.id,
        "config_hash": "x",
        "git_sha": "6d29de4-dirty",
        "run_at": "2026-10-20T22:07:26Z",
        "n": 40402,
        "ci_lo": 0.012,
        "ci_hi": 0.014,
        "window": "evaluation",
        "label": "grid",
        "direction": "positive",
        "p_value": 0.0,
        "sr": 0.16,
        "skew": -0.02,
        "kurt": 7.2,
        "registration_sha256": reg.source_sha256,
        "data_snapshot": "fd.parquet@2026-10-20",
        "config_json": json.dumps({"source": "max_open", "theta": 0.0}),
    }
    _csv(
        csv,
        R2_HEADER,
        [
            base | {"metric": 0.013, "metric_name": "clv_log_mean"},
            base | {"metric": 0.3, "metric_name": "ols_slope_g", "ci_lo": 0.2, "ci_hi": 0.4, "direction": "two-sided"},
        ],
    )
    report = import_trials_csv(csv, reg.id, ledger)
    assert report.imported == 2 and report.off_grid == 2
    assert report.primary_metrics == ["clv_log_mean"]
    t = ledger.trials(hypothesis_id=reg.id)
    extras = {r["metric_name"]: json.loads(r["extra_json"]) for _, r in t.iterrows()}
    assert extras["clv_log_mean"]["is_primary_metric"] is True and extras["clv_log_mean"]["off_grid"] is True
    assert extras["ols_slope_g"]["is_primary_metric"] is False
    assert extras["clv_log_mean"]["imported_sr"] == 0.16 and "sr" not in extras["clv_log_mean"]
    assert extras["clv_log_mean"]["metric_alias_of"] == reg.primary_metric


def _alias_case(tmp_path, primary: str):
    ledger = Ledger(tmp_path / "ledger.duckdb")
    reg = load(write_registration(tmp_path, primary_metric=primary))
    ledger.register(reg)
    csv = tmp_path / "trials.csv"
    _csv(csv, R1_HEADER, [_r1_row(reg, metric="clv_log_mean", config="{}")])
    return ledger, reg, csv


def test_aliases_are_never_fuzzy(tmp_path):
    # Same leading token as a known alias target, but this exact registered string is not in the table.
    ledger, reg, csv = _alias_case(tmp_path, "clv_log = log(odds * p), mean per bet")
    report = import_trials_csv(csv, reg.id, ledger)
    assert report.primary_metrics == []
    assert json.loads(ledger.trials().iloc[0]["extra_json"])["is_primary_metric"] is False


def test_explicit_alias_argument(tmp_path):
    ledger, reg, csv = _alias_case(tmp_path, "clv_log = log(odds * p), mean per bet")
    report = import_trials_csv(csv, reg.id, ledger, aliases={"clv_log_mean": reg.primary_metric})
    assert report.primary_metrics == ["clv_log_mean"]
    extra = json.loads(ledger.trials().iloc[0]["extra_json"])
    assert extra["is_primary_metric"] is True and extra["metric_alias_of"] == reg.primary_metric


def test_the_built_in_alias_table_is_exact_strings():
    for csv_name, targets in METRIC_ALIASES.items():
        assert targets and all(isinstance(t, str) and t for t in targets), csv_name


def test_explicit_primary_metrics_override_the_heuristic(setup, tmp_path):
    ledger, reg = setup
    csv = tmp_path / "trials.csv"
    _csv(csv, R1_HEADER, [_r1_row(reg, metric="draw_bias")])
    report = import_trials_csv(csv, reg.id, ledger, primary_metrics=["draw_bias"])
    assert report.primary_metrics == ["draw_bias"]
    assert json.loads(ledger.trials().iloc[0]["extra_json"])["is_primary_metric"] is True


def test_rows_of_other_ids_and_bad_rows_are_reported(setup, tmp_path):
    ledger, reg = setup
    csv = tmp_path / "trials.csv"
    _csv(csv, R1_HEADER, [_r1_row(reg, id="someone-else-v1"), _r1_row(reg, direction="sideways"), _r1_row(reg)])
    report = import_trials_csv(csv, reg.id, ledger)
    assert report.imported == 1 and report.other_ids == 1
    assert len(report.refused) == 1 and "direction" in report.refused[0]


def test_import_requires_registration(tmp_path):
    reg = load(write_registration(tmp_path))
    csv = tmp_path / "trials.csv"
    _csv(csv, R1_HEADER, [_r1_row(reg)])
    with pytest.raises(ValueError, match="not registered"):
        import_trials_csv(csv, reg.id, Ledger(tmp_path / "l.duckdb"))


def test_imported_trials_count_in_family_bh(setup, tmp_path):
    ledger, reg = setup
    csv = tmp_path / "trials.csv"
    rows = [
        _r1_row(reg, config=json.dumps({"theta": t, "side": s}), p_one_sided=0.5)
        for t in (0.02, 0.04)
        for s in (True, False)
    ]
    _csv(csv, R1_HEADER, rows)
    import_trials_csv(csv, reg.id, ledger)
    g = grade(reg.id, ledger)
    assert g.rung in ("none", "L0")
