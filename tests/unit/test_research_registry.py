"""Hypothesis registration files: parsing, hashing and grid expansion."""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from pitch_edge.research.errors import ProtocolError
from pitch_edge.research.registry import (
    REPO_ROOT,
    Registration,
    config_hash,
    load,
    relative_registry_path,
    resolve_registry_path,
    sha256,
)

pytestmark = pytest.mark.unit

EXAMPLE = """\
id: pm-vs-sharp-fav-bias-v1
family: polymarket-efficiency
registered_at: 2026-10-20T21:00:00Z
author: quant-researcher
question: >
  Do Polymarket 1X2 prices 24h before kickoff over-price favourites relative to the de-vigged sharp close?
hypothesis:
  h1: Polymarket implied probability for the favourite exceeds the de-vigged sharp close by > 0
  h0: no difference
  direction: positive
  disconfirming: 95% CI of the mean gap includes 0, or sign flips in 2 of 3 sub-periods
decision_time: kickoff - 24h
universe: Polymarket win_home/draw/win_away markets mapped to fd_matches (pm_soccer_match_map)
window:
  exploratory: [2023-09-01, 2025-06-30]
  evaluation:  [2025-07-01, 2026-10-19]
benchmark: betfair_ex close de-vigged (power); pinnacle close where present (sensitivity)
primary_metric: mean_prob_gap
secondary_metrics: [roi_after_fees, brier_skill]
costs: polymarket taker fee 0.05*p*(1-p) per share + half bar high-low spread
grid:
  favourite_threshold: [0.5, 0.6]
  hours_before: [24, 6, 1]
min_n: 300
literature_ceiling: gap of a few percentage points at most for liquid matches
"""


def write_registration(directory: Path, text: str = EXAMPLE, name: str = "pm-vs-sharp-fav-bias-v1") -> Path:
    path = directory / f"{name}.yaml"
    path.write_text(text)
    return path


def test_load_parses_the_protocol_example(tmp_path):
    reg = load(write_registration(tmp_path))
    assert isinstance(reg, Registration)
    assert reg.id == "pm-vs-sharp-fav-bias-v1"
    assert reg.family == "polymarket-efficiency"
    assert reg.registered_at == datetime(2026, 10, 20, 21, 0, tzinfo=UTC)
    assert reg.hypothesis.direction == "positive"
    assert reg.window.exploratory == (date(2023, 9, 1), date(2025, 6, 30))
    assert reg.window.evaluation == (date(2025, 7, 1), date(2026, 10, 19))
    assert reg.secondary_metrics == ["roi_after_fees", "brier_skill"]
    assert reg.min_n == 300
    assert reg.source_path == (tmp_path / "pm-vs-sharp-fav-bias-v1.yaml").resolve()
    assert reg.source_sha256 == sha256(reg.source_path)


def test_sha256_is_of_the_file_bytes(tmp_path):
    path = write_registration(tmp_path)
    assert sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(EXAMPLE + "# a comment changes the hash\n")
    assert sha256(path) != hashlib.sha256(EXAMPLE.encode()).hexdigest()


def test_expand_grid_is_the_cartesian_product_in_file_order(tmp_path):
    configs = load(write_registration(tmp_path)).expand_grid()
    assert len(configs) == 6
    assert configs[0] == {"favourite_threshold": 0.5, "hours_before": 24}
    assert configs[-1] == {"favourite_threshold": 0.6, "hours_before": 1}
    assert len({config_hash(c) for c in configs}) == 6


def test_empty_grid_is_a_single_trial(tmp_path):
    text = EXAMPLE.split("grid:")[0] + "min_n: 300\nliterature_ceiling: small\n"
    configs = load(write_registration(tmp_path, text)).expand_grid()
    assert configs == [{}]


def test_config_hash_is_stable_and_order_independent():
    a = config_hash({"x": 1, "y": [1, 2], "z": "s"})
    assert a == config_hash({"z": "s", "y": [1, 2], "x": 1})
    assert a != config_hash({"x": 2, "y": [1, 2], "z": "s"})
    assert len(a) == 64 and int(a, 16) >= 0
    # Stable across processes: pinned value of sha256('{"x":1}').
    assert config_hash({"x": 1}) == hashlib.sha256(b'{"x":1}').hexdigest()


def test_bad_direction_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        load(write_registration(tmp_path, EXAMPLE.replace("direction: positive", "direction: up")))


def test_unknown_fields_are_kept_under_extensions(tmp_path):
    text = EXAMPLE.replace("min_n: 300", "min_n: 300\nstudy_dir: research/studies/x\nfilters: {max_stale_hours: 6}")
    reg = load(write_registration(tmp_path, text))
    assert reg.extensions == {"study_dir": "research/studies/x", "filters": {"max_stale_hours": 6}}


def test_core_fields_are_still_required(tmp_path):
    with pytest.raises(ValidationError):
        load(write_registration(tmp_path, EXAMPLE.replace("primary_metric: mean_prob_gap\n", "")))


def test_min_n_accepts_a_leading_integer_in_text(tmp_path):
    reg = load(
        write_registration(tmp_path, EXAMPLE.replace("min_n: 300", "min_n: 300 matches per division-level test"))
    )
    assert reg.min_n == 300
    with pytest.raises(ValidationError):
        load(write_registration(tmp_path, EXAMPLE.replace("min_n: 300", "min_n: plenty")))


def test_hypothesis_h1_variants_and_window_extras(tmp_path):
    text = EXAMPLE.replace("  h1: Polymarket", "  h1_predict: Polymarket").replace(
        "  evaluation:  [2025-07-01, 2026-10-19]",
        "  evaluation:  [2025-07-01, 2026-10-19]\n  post_segment: [2026-01-01, 2026-10-19]",
    )
    reg = load(write_registration(tmp_path, text))
    assert reg.hypothesis.h1 is None
    assert "h1_predict" in reg.hypothesis.extensions
    assert reg.window.extensions["post_segment"] == [date(2026, 1, 1), date(2026, 10, 19)]
    with pytest.raises(ValidationError):  # no h1 of any kind
        load(
            write_registration(
                tmp_path,
                EXAMPLE.replace(
                    "  h1: Polymarket implied probability for the favourite exceeds the de-vigged sharp close by > 0\n",
                    "",
                ),
            )
        )


def test_per_grid_point_direction(tmp_path):
    text = EXAMPLE.replace("direction: positive", "direction: per segment (see grid)").replace(
        "  favourite_threshold: [0.5, 0.6]\n  hours_before: [24, 6, 1]",
        "  segment:\n    - {name: longshot, direction: positive}\n    - {name: favourite, direction: negative}\n    - {name: draw, direction: two-sided}",
    )
    reg = load(write_registration(tmp_path, text))
    assert reg.hypothesis.direction == "per-grid-point"
    dirs = [reg.direction_for(c) for c in reg.expand_grid()]
    assert dirs == ["positive", "negative", "two-sided"]
    with pytest.raises(ProtocolError, match="direction"):
        reg.direction_for({"segment": {"name": "x"}})
    assert load(write_registration(tmp_path)).direction_for({"favourite_threshold": 0.5}) == "positive"


def test_metric_kind_inferred_for_old_registrations(tmp_path):
    reg = load(write_registration(tmp_path))  # registered 2026-10-20: before metric_kind became required
    assert reg.metric_kind == "probability" and reg.metric_kind_inferred  # mean_prob_gap
    clv = load(
        write_registration(
            tmp_path, EXAMPLE.replace("primary_metric: mean_prob_gap", "primary_metric: clv_log = log(odds)")
        )
    )
    assert clv.metric_kind == "return"
    other = load(
        write_registration(
            tmp_path, EXAMPLE.replace("primary_metric: mean_prob_gap", "primary_metric: ols_slope_move_on_gap")
        )
    )
    assert other.metric_kind == "return"  # conservative default: DSR applies
    explicit = load(write_registration(tmp_path, EXAMPLE.replace("min_n: 300", "min_n: 300\nmetric_kind: score")))
    assert explicit.metric_kind == "score" and not explicit.metric_kind_inferred


def test_metric_kind_required_for_new_registrations(tmp_path):
    new = EXAMPLE.replace("2026-10-20T21:00:00Z", "2026-10-22T09:00:00Z")
    with pytest.raises(ValidationError, match="metric_kind"):
        load(write_registration(tmp_path, new))
    assert load(write_registration(tmp_path, new + "metric_kind: return\n")).metric_kind == "return"


def test_single_league_flag_defaults_false(tmp_path):
    assert load(write_registration(tmp_path)).single_league is False
    assert load(write_registration(tmp_path, EXAMPLE + "single_league: true\n")).single_league is True


def test_repo_relative_paths(tmp_path):
    inside = REPO_ROOT / "research" / "registry" / "x-v1.yaml"
    assert relative_registry_path(inside) == "research/registry/x-v1.yaml"
    assert resolve_registry_path("research/registry/x-v1.yaml") == inside
    outside = tmp_path / "y-v1.yaml"
    assert relative_registry_path(outside) == str(outside.resolve())
    assert resolve_registry_path(str(outside.resolve())) == outside.resolve()


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "research" / "registry").glob("*.yaml")), ids=lambda p: p.stem)
def test_every_real_registration_loads(path):
    reg = load(path)
    assert reg.id == path.stem
    assert reg.metric_kind in ("return", "probability", "score")
    for cfg in reg.expand_grid():
        assert reg.direction_for(cfg) in ("positive", "negative", "two-sided")


def test_window_must_be_ordered(tmp_path):
    bad = EXAMPLE.replace("evaluation:  [2025-07-01, 2026-10-19]", "evaluation:  [2026-10-19, 2025-07-01]")
    with pytest.raises(ValidationError):
        load(write_registration(tmp_path, bad))


def test_registered_at_must_be_timezone_aware(tmp_path):
    with pytest.raises(ValidationError):
        load(write_registration(tmp_path, EXAMPLE.replace("2026-10-20T21:00:00Z", "2026-10-20 21:00:00")))


def test_id_must_carry_a_version_and_match_the_file_name(tmp_path):
    with pytest.raises(ValidationError):
        load(
            write_registration(
                tmp_path, EXAMPLE.replace("id: pm-vs-sharp-fav-bias-v1", "id: pm-vs-sharp"), "pm-vs-sharp"
            )
        )
    with pytest.raises(ValueError, match="file name"):
        load(write_registration(tmp_path, EXAMPLE, name="other-name-v1"))


def test_grid_values_must_be_lists(tmp_path):
    with pytest.raises(ValidationError):
        load(
            write_registration(tmp_path, EXAMPLE.replace("favourite_threshold: [0.5, 0.6]", "favourite_threshold: 0.5"))
        )


def test_post_hoc_flag_and_wording(tmp_path):
    assert load(write_registration(tmp_path)).is_post_hoc is False
    assert load(write_registration(tmp_path, EXAMPLE + "post_hoc: true\n")).is_post_hoc is True
    worded = EXAMPLE.replace("question: >\n  Do", "question: >\n  POST-HOC (registered after seeing a result). Do")
    assert load(write_registration(tmp_path, worded)).is_post_hoc is True
    assert load(write_registration(tmp_path, EXAMPLE + "posthoc: yes\n")).is_post_hoc is True
