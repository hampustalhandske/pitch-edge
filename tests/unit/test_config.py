from __future__ import annotations

import pytest

from pitch_edge.config import Settings, load_dotenv

pytestmark = pytest.mark.unit


def test_load_dotenv_sets_only_missing_vars(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# keys\nAPI_FOOTBALL_KEY="abc"\nEXISTING=from_file\nEMPTY=\nbad line\n')
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.setenv("EXISTING", "from_shell")
    monkeypatch.delenv("EMPTY", raising=False)
    assert load_dotenv(env) == 1
    import os

    assert os.environ["API_FOOTBALL_KEY"] == "abc"
    assert os.environ["EXISTING"] == "from_shell"  # never overridden
    assert "EMPTY" not in os.environ
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)


def test_load_dotenv_missing_file_is_noop(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == 0


def test_settings_paths_follow_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PITCH_EDGE_DATA_DIR", str(tmp_path / "d"))
    s = Settings()
    assert s.db_path == tmp_path / "d" / "pitch_edge.duckdb"
    assert s.vector_dir == tmp_path / "d" / "chroma" and s.anthropic_model == "claude-fable-5-1"
    s.ensure_dirs()
    assert s.artifacts_dir.exists()
