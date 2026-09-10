"""Central runtime configuration.

Everything is overridable through environment variables so the same code
runs locally (DuckDB/Parquet/Chroma on disk) and on GCP (Phase 3) without
code changes. Nothing here ever holds a bookmaker account credential — the
system has no bet-placement capability to configure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

USER_AGENT = "pitch-edge-research/0.2 (+https://github.com/hampusstalhandske/pitch-edge; research, non-commercial)"


def load_dotenv(path: str | Path = ".env") -> int:
    """Minimal .env loader (KEY=VALUE lines, # comments); never overrides variables already set."""
    p = Path(path)
    if not p.exists():
        return 0
    loaded = 0
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ and value:
            os.environ[key] = value
            loaded += 1
    return loaded


load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("PITCH_EDGE_DATA_DIR", "data")))
    anthropic_api_key: str | None = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY"))
    anthropic_model: str = field(default_factory=lambda: _env("PITCH_EDGE_LLM_MODEL", "claude-fable-5-1"))
    embedding_model: str = field(
        default_factory=lambda: _env("PITCH_EDGE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    http_min_interval_s: float = field(default_factory=lambda: float(_env("PITCH_EDGE_HTTP_MIN_INTERVAL", "1.0")))
    gcp_project: str | None = field(default_factory=lambda: os.environ.get("PITCH_EDGE_GCP_PROJECT"))
    gcs_bucket: str | None = field(default_factory=lambda: os.environ.get("PITCH_EDGE_GCS_BUCKET"))
    bigquery_dataset: str = field(default_factory=lambda: _env("PITCH_EDGE_BQ_DATASET", "pitch_edge"))
    fd_archive_dir: str | None = field(default_factory=lambda: os.environ.get("PITCH_EDGE_FD_ARCHIVE_DIR"))
    local_llm_model: str = field(default_factory=lambda: _env("PITCH_EDGE_LOCAL_LLM_MODEL", "llama3.1:8b"))
    local_llm_base_url: str = field(
        default_factory=lambda: _env("PITCH_EDGE_LOCAL_LLM_BASE_URL", "http://localhost:11434")
    )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "pitch_edge.duckdb"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def backtest_dir(self) -> Path:
        """Local, machine-readable backtest output (CSVs, model cards) — one subdir per label.
        Not `reports_dir`: that directory is public and holds only the generated case-study
        write-ups, so nothing here is committed."""
        return self.data_dir / "backtest"

    @property
    def reports_dir(self) -> Path:
        return Path(_env("PITCH_EDGE_REPORTS_DIR", "reports"))

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.vector_dir, self.artifacts_dir, self.backtest_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
