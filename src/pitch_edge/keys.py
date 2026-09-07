"""Single source of truth for the optional credentials — what each unlocks, where to get it.

`pitch-edge setup` and the dashboard's key panel both read this registry, and a test checks that
`API_KEYS.md` names every entry, so the docs and the code cannot drift apart.

None of these is a bookmaker or exchange credential. Nothing accepted here can place an order or
move money: the system has no bet-placement capability to configure.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OptionalKey:
    env: str
    name: str
    unlocks: str
    signup: str
    next_command: str
    kind: str = "api"  # "api" | "gcp"


OPTIONAL_KEYS: tuple[OptionalKey, ...] = (
    OptionalKey(
        "API_FOOTBALL_KEY",
        "API-Football",
        "injuries, confirmed lineups and substitutions — the first data the closing price does not already contain; turns the dossier's 'who is playing' section from provisional to confirmed",
        "https://dashboard.api-football.com/register",
        "uv run pitch-edge ingest",
    ),
    OptionalKey(
        "ODDS_API_KEY",
        "The Odds API",
        "live pre-match prices from real bookmakers: replaces the labelled synthetic Elo quotes in the signal pipeline and feeds the steam detector and the cross-venue lead-lag test",
        "https://the-odds-api.com",
        "uv run pitch-edge schedule",
    ),
    OptionalKey(
        "EVERYSPORT_API_KEY",
        "Everysport",
        "Swedish results and fixtures below Allsvenskan (Superettan, Ettan, Division 2/3) — the leagues where local information is thinnest",
        "https://www.everysport.com",
        "uv run pitch-edge ingest",
    ),
    OptionalKey(
        "ANTHROPIC_API_KEY",
        "Anthropic",
        "Claude Fable 5.1 as the RAG explainer and dossier narrator (cited, verified; it never estimates a probability); without it a deterministic template is used",
        "https://console.anthropic.com",
        'uv run pitch-edge rag "why does the model like the away side?"',
    ),
    OptionalKey(
        "PITCH_EDGE_GCP_PROJECT",
        "Google Cloud project",
        "the GCS/BigQuery mirror and Cloud Run deployment in deploy/ (never required locally)",
        "https://console.cloud.google.com",
        "uv sync --extra cloud && uv run pitch-edge export",
        kind="gcp",
    ),
    OptionalKey(
        "PITCH_EDGE_GCS_BUCKET",
        "GCS bucket",
        "where the Parquet lake is mirrored when the GCP project is set",
        "https://console.cloud.google.com/storage",
        "uv run pitch-edge export",
        kind="gcp",
    ),
)

_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def dotenv_keys(path: str | Path = ".env") -> set[str]:
    """Names of variables with a non-empty value in a .env file (values are never returned)."""
    p = Path(path)
    if not p.exists():
        return set()
    found = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if m and m.group(2).strip().strip('"').strip("'"):
            found.add(m.group(1))
    return found


def key_status(env: Mapping[str, str] | None = None, dotenv_path: str | Path = ".env") -> list[dict]:
    """One row per optional key: set? and where from ('environment' | '.env' | None). No values."""
    env_map: Mapping[str, str] = os.environ if env is None else env
    in_file = dotenv_keys(dotenv_path)
    rows = []
    for k in OPTIONAL_KEYS:
        if env_map.get(k.env):
            source = "environment"
        elif k.env in in_file:
            source = ".env"
        else:
            source = None
        rows.append(
            {
                "env": k.env,
                "name": k.name,
                "kind": k.kind,
                "set": source is not None,
                "source": source,
                "unlocks": k.unlocks,
                "signup": k.signup,
                "next_command": k.next_command,
            }
        )
    return rows


def status_summary(env: Mapping[str, str] | None = None, dotenv_path: str | Path = ".env") -> dict:
    rows = key_status(env, dotenv_path)
    api = [r for r in rows if r["kind"] == "api"]
    missing = [r for r in api if not r["set"]]
    return {
        "n_api_set": sum(r["set"] for r in api),
        "n_api_total": len(api),
        "anthropic_active": any(r["env"] == "ANTHROPIC_API_KEY" and r["set"] for r in rows),
        "gcp_configured": all(r["set"] for r in rows if r["kind"] == "gcp"),
        "active": [r["name"] for r in rows if r["set"]],
        "missing": [r["env"] for r in missing],
        "next_unlock": missing[0] if missing else None,
    }


def write_env_value(path: str | Path, key: str, value: str) -> None:
    """Create or update `KEY=value` in a .env file, owner-read/write only. Never logs the value."""
    p = Path(path)
    lines = p.read_text().splitlines() if p.exists() else []
    out, replaced = [], False
    for line in lines:
        m = _LINE_RE.match(line)
        if m and m.group(1) == key:
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        if not lines:
            out.append("# PITCH-EDGE optional keys — git-ignored. None of these can place a bet or move money.")
        out.append(f"{key}={value}")
    p.write_text("\n".join(out) + "\n")
    try:
        p.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:  # pragma: no cover - platform dependent
        pass
