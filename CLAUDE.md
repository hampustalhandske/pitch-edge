# CLAUDE.md — project context for PITCH-EDGE

Read `README.md` first for scope; this file is the working contract for agents editing this repo.

## What this is

A Python-only football (soccer) intelligence & market-edge research platform: free-data ingestion → DuckDB
warehouse → feature store → Dixon-Coles / GBDT / Lightning GRU & Transformer / PyG GNN → walk-forward
backtests scored on closing-line value → LangChain hybrid RAG (Claude Fable 5.1 explainer) → LangGraph
signal pipeline with a mandatory human approval gate → single-page dark Streamlit dashboard.

## Non-negotiables (never relax, even in "test mode")

- **No automated bet placement, no real-money movement.** The approval gate is structural
  (`interrupt_before=["human_approval"]`, `log_alerts` raises without a human id). Do not add any code path
  that calls a bookmaker/exchange with an order.
- **ToS-clean data only.** Every fetch goes through `pitch_edge.data.http.CachedHttpClient` (honest UA,
  throttle, disk cache, robots check, per-host circuit breaker). Understat, FotMob `/api`, Sofascore,
  Flashscore and X are excluded on purpose — don't add scrapers for them.
- **No unvalidated number reaches the UI.** The dashboard reads warehouse tables/artifacts written by the
  pipeline; it never trains, fetches odds, or computes new model numbers.
- **The LLM explains, never estimates.** `rag/generate.py` quotes numbers from retrieved docs with `[doc_id]`
  citations; `verify_citations` flags anything ungrounded.
- **Walk-forward only.** No shuffled CV across time; bets priced at real vig-inclusive odds; CLV vs closing.
- **Losing results are published.** Don't tune a report until it looks good.

## Commands

```bash
uv sync --extra dev                   # env (torch, lightning, langchain, chromadb, langgraph, streamlit …)
uv run pytest                         # 233 tests, network mocked (responses); markers: unit, integration, network
uv run ruff check . && uv run ruff format . && uv run mypy src
uv run pitch-edge serve               # dashboard
uv run pitch-edge refresh [--fast]    # ingest → features → backtests → RAG → artifacts
uv run pitch-edge {setup|intel|ingest|features|backtest|ablation|lead-lag|referee-study|artifacts|rag|rag-eval|signals|health|export|schedule}
caffeinate -i uv run python scripts/stage_*.py   # restartable long runs (see scripts/README.md)
```

Long runs must be launched with `nohup caffeinate -i … &` (a laptop sleep once turned a 15-minute backtest
into 7 hours). DuckDB is single-writer: never run two pipeline stages at once, and the dashboard opens only
short-lived read-only connections (`_read_cached` / `_query_cached`, cache keyed by DB path).

## Conventions

- Python 3.11, `uv`, `src/` layout, ruff (line length 120, E501 ignored), mypy on `src`. Tests live in
  `tests/unit` and `tests/integration`; every connector gets a mocked-HTTP test; every new model gets a
  synthetic-league test (`tests/conftest.py::make_synthetic_league`).
- New data source → subclass `MatchDataSource` (or a plain class for non-match data), route HTTP through
  `CachedHttpClient`, add a `TABLE_KEYS` entry in `data/storage.py`, an `ingest_*` in `data/ingest.py`
  wrapped by `_run` (health logging), and a mocked test.
- New match model → subclass `MatchModel` (`fit`, `predict_proba`, `card`, optional `observe`), register in
  `models/__init__.py::available_models`, keep leakage checks in the card.
- Charts: use `dashboard/theme.py` tokens (SERIES / OUTCOME / STATUS / INK) and the `pitch_edge` Plotly
  template; one y-axis per chart; colour follows the entity; status colours never reused for series.
- Docs to keep in sync when behaviour changes: `README.md`, `CASE_STUDY.md`, `MODEL_CARDS.md`,
  `API_KEYS.md`, `scripts/README.md`.
- Don't commit or push unless explicitly asked; `data/`, `*.log`, `.env`, large report parquet are ignored.

## State (2026-09-06)

- Warehouse: 239k matches / 41 divisions / 1.97 M odds rows / 7.1 M Transfermarkt rows / 1.34 M Wikipedia pageview rows (340 clubs, keyed by team) / 10.5 k RAG docs.
- football-data.co.uk (Pinnacle early+closing, referee, `SWE.csv`) and Club Elo were down (503/502) for the
  whole build; spine is the open 2000-2025 compilation, so CLV is zero by construction until re-ingest.
  `pitch-edge ingest` upgrades rows in place when they return (order-independent de-dup in `store_matches`).
- Results: all models < market (best −0.024 bits); developing markets further behind; Elo the only feature
  group that matters; deep models trail trees. See `CASE_STUDY.md`.
- Optional keys wired but unset: `API_FOOTBALL_KEY`, `EVERYSPORT_API_KEY`, `ANTHROPIC_API_KEY`
  (`ODDS_API_KEY` connector still to write). See `API_KEYS.md`.
- Stubs by design: stadium audio (no licensed feed), OpenSky charter attribution, GCP deployment (code +
  manifests only), club financial-distress feature (no structured free feed).
- Phase 5 (2026-09-06): `intel/` dossier (CLI `intel`, dashboard tab; every claim cited + `verify_citations`,
  versioned `dossiers` table, section 9 = generated gaps) — the dashboard renders stored dossiers only.
  New feature groups `wiki_attention` (Wikimedia pageviews) and `rotation_load` + referee names (Transfermarkt
  context) are pre-registered in `CASE_STUDY.md` Result 7; `odds/leadlag.py` and `backtest/event_study.py`
  are real methods whose live data was insufficient at build time (hourly scheduler accumulates it).
  `keys.py` is the single registry behind `pitch-edge setup`; a test asserts `API_KEYS.md` matches it.
