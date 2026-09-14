"""`pitch-edge setup`: never blocks, writes nothing when declined, never echoes a key, docs never drift."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pitch_edge.cli import app
from pitch_edge.keys import OPTIONAL_KEYS, key_status, status_summary, write_env_value

pytestmark = pytest.mark.unit

runner = CliRunner()
N = len(OPTIONAL_KEYS)
N_API = sum(1 for k in OPTIONAL_KEYS if k.kind == "api")


@pytest.fixture
def clean_env(monkeypatch):
    for k in OPTIONAL_KEYS:
        monkeypatch.delenv(k.env, raising=False)


def test_setup_declining_everything_writes_nothing(tmp_path, clean_env):
    env = tmp_path / ".env"
    res = runner.invoke(app, ["setup", "--env-file", str(env)], input="n\n" * N)
    assert res.exit_code == 0, res.output
    assert not env.exists()
    assert f"0 of {N_API} optional API keys unlocked" in res.output and "keyless" in res.output
    assert "place a bet" in res.output  # the guardrail is in the command's own output


def test_setup_writes_key_without_echo_and_status_picks_it_up(tmp_path, clean_env):
    env = tmp_path / ".env"
    res = runner.invoke(app, ["setup", "--env-file", str(env)], input="y\nsekrit-123\n" + "n\n" * (N - 1))
    assert res.exit_code == 0, res.output
    assert "sekrit-123" not in res.output
    text = env.read_text()
    assert "API_FOOTBALL_KEY=sekrit-123" in text and text.startswith("#")
    assert oct(env.stat().st_mode & 0o777) == "0o600"
    rows = {r["env"]: r for r in key_status(dotenv_path=env)}
    assert rows["API_FOOTBALL_KEY"]["set"] and rows["API_FOOTBALL_KEY"]["source"] == ".env"
    assert not rows["EVERYSPORT_API_KEY"]["set"]
    s = status_summary(dotenv_path=env)
    assert s["n_api_set"] == 1 and s["next_unlock"]["env"] == "EVERYSPORT_API_KEY" and not s["anthropic_active"]
    # re-running replaces in place, never duplicates
    write_env_value(env, "API_FOOTBALL_KEY", "new")
    assert env.read_text().count("API_FOOTBALL_KEY=") == 1


def test_non_interactive_never_prompts_and_downstream_runs_keyless(tmp_path, clean_env):
    res = runner.invoke(app, ["setup", "--non-interactive", "--env-file", str(tmp_path / ".env")])
    assert res.exit_code == 0 and "missing" in res.output
    # downstream commands keep working with zero keys (the degrade-gracefully invariant)
    health = runner.invoke(app, ["health"])
    assert health.exit_code == 0, health.output


def test_api_keys_doc_lists_every_registry_entry():
    doc = Path(__file__).resolve().parents[2] / "API_KEYS.md"
    text = doc.read_text()
    for k in OPTIONAL_KEYS:
        assert k.env in text, f"{k.env} missing from API_KEYS.md"
        if k.kind == "api":
            assert k.signup in text, f"sign-up link for {k.name} missing from API_KEYS.md"
    assert "pitch-edge setup" in text


def test_key_status_prefers_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-value-9f3a")
    rows = {r["env"]: r for r in key_status(dotenv_path=tmp_path / "nope")}
    assert rows["ANTHROPIC_API_KEY"]["source"] == "environment"
    assert "sk-secret-value-9f3a" not in str(rows)  # values never leak into status rows
    assert "ANTHROPIC_API_KEY" in os.environ
