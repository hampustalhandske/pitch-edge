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
- **The ToS-clean-only restriction is lifted.** The project owner has explicitly reversed the earlier
  "ToS-clean data only" line ("it is impossible to get data if it is not ToS clean; skip that constraint
  now") — sources like Understat, FotMob `/api`, Sofascore, Flashscore and X (and third-party scrapers such
  as `WhoscoredClient`) are back in scope. Every fetch still goes through
  `pitch_edge.data.http.CachedHttpClient` (honest UA, throttle, disk cache, robots check, per-host circuit
  breaker) — that plumbing is a resilience/politeness mechanism, not a ToS gate, and stays as-is.
- **No unvalidated number reaches the UI.** The dashboard reads warehouse tables/artifacts written by the
  pipeline; it never trains, fetches odds, or computes new model numbers.
- **The LLM explains, never estimates.** `rag/generate.py` quotes numbers from retrieved docs with `[doc_id]`
  citations; `verify_citations` flags anything ungrounded.
- **Walk-forward only.** No shuffled CV across time; bets priced at real vig-inclusive odds; CLV vs closing.
- **Losing results are published.** Don't tune a report until it looks good.

## Commands

```bash
uv sync --extra dev                   # env (torch, lightning, langchain, chromadb, langgraph, streamlit …)
uv run pytest                         # 260 tests, network mocked (responses); markers: unit, integration, network
uv run ruff check . && uv run ruff format . && uv run mypy src
uv run pitch-edge serve               # dashboard
uv run pitch-edge refresh [--fast]    # ingest → features → backtests → RAG → artifacts
uv run pitch-edge {setup|intel|ingest|features|backtest|ablation|lead-lag|referee-study|artifacts|rag|rag-eval|signals|agentic-signals|replay-eval|health|export|schedule}
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
- Optional keys wired but unset: `API_FOOTBALL_KEY`, `ODDS_API_KEY`, `EVERYSPORT_API_KEY`, `ANTHROPIC_API_KEY`.
  See `API_KEYS.md`.
- Stubs by design: stadium audio (no licensed feed), OpenSky charter attribution, GCP deployment (code +
  manifests only), club financial-distress feature (no structured free feed).
- Phase 5 (2026-09-06): `intel/` dossier (CLI `intel`, dashboard tab; every claim cited + `verify_citations`,
  versioned `dossiers` table, section 9 = generated gaps) — the dashboard renders stored dossiers only.
  New feature groups `wiki_attention` (Wikimedia pageviews) and `rotation_load` + referee names (Transfermarkt
  context) are pre-registered in `CASE_STUDY.md` Result 7; `odds/leadlag.py` and `backtest/event_study.py`
  are real methods whose live data was insufficient at build time (hourly scheduler accumulates it).
  `keys.py` is the single registry behind `pitch-edge setup`; a test asserts `API_KEYS.md` matches it.
- Phase 6 (2026-09-09): data-quality fixes — `football_data_co_uk.py` reads the local `data/archive/`
  mirror first (real PS/PSC pairs, skips the network entirely when a season is archived);
  `data/alt/odds_api.py` (real live 1X2 quotes, `live_odds` table) replaces synthetic-only odds in the
  live signals path when a real quote exists; `features/live_lineups.py` overlays confirmed API-Football
  starting-XI value/missing-pct onto upcoming fixtures (`lineup_source: confirmed|provisional`).
  New `agentic-signals` CLI: a second, additive LangGraph orchestrator (`agents/orchestrator.py`) with a
  local-Ollama data-quality screening agent (`agents/selection.py`), a per-league model router reading real
  `by_league.csv` evidence (`agents/router.py`), and a RAG-grounded edge reviewer
  (`agents/reviewer.py`, reuses `verify_citations`) — same `interrupt_before=["human_approval"]` gate,
  `agents/graph.py` and the deterministic `signals` command untouched. `replay-eval` scores the reviewer's
  trust/distrust verdicts against real closing odds/results revealed only after the fact (T0 = latest real
  closing date - 30d, everything before T0 trained, `[T0, T0+30d)` replayed blind).
- Phase 7 (2026-09-10): real football-data.co.uk data (2015/16-2023/24, all 22 main + 16 extra-league
  divisions) backfilled from `data/archive/` into the warehouse — `matches`/`odds` now carry genuine
  Pinnacle early+closing pairs for those seasons instead of the xgabora zero-CLV fallback; `reports/main/`
  regenerated from this real data (dixon_coles/gbdt/gbdt_mkt) so the Phase-6 model router has real
  evidence to read. `rag/generate.py`'s `GroundedGenerator` is now local-first: the local Ollama model
  (same one `agents/llm.py` gives the selection/reviewer agents) is tried before Claude, which is now
  only a fallback when a key is set and local generation fails; `rag/query_parser.py` extracts team
  names from a free-text question (local LLM, degrades to raw-text search) so `pitch-edge rag` retrieves
  team-focused context instead of matching on the raw question alone. `src/pitch_edge/vendor/whoscored/`
  vendors `Ali-Hasan-Khan/Scrape-Whoscored-Event-Data` (MIT) with a real fix for the previously-broken
  fixture-discovery flow (a cookie-consent overlay intercepting the season-select click, a stale-element
  bug on top of it, a missing navigation to the "Fixtures" tab, and a BeautifulSoup/Selenium type bug in
  score parsing — see `vendor/NOTICE.md`) — confirmed working live against real WhoScored pages with
  headless Firefox. Not yet wired into `pipeline.py::upcoming_fixture_frame` or ingestion: the project
  owner deprioritized live fixture-discovery integration in favor of prioritizing the already-archived
  2015-2024 seasons for the agentic simulation; picking this back up is future work, not a currently
  broken path.
- Router change (2026-09-10): `agents/router.py::select_model_for_league` no longer requires
  `edge_bits > 0` — it always returns whichever available model has the *best* edge for a league, even
  when every model's edge is negative there (the honest, currently-common case per the real 2015-2024
  backtest evidence: 0/55 league×model combinations beat the market). A league now returns `None` only
  when there is zero backtest evidence for any available model at all, not merely bad evidence.
  `RiskManager.size()` (unchanged) still ranks and filters proposals strictly by each fixture's own
  `edge`/`min_edge`, so a bad-model league surfaces fewer/no proposals on its own merits rather than the
  router pre-emptively hiding it.
- Reports/data split (2026-09-10): `reports/` is now public-only — one `<label>/CASE_STUDY.md` per test
  (backtest/ablation/replay-eval), nothing else. All machine-readable output (CSVs, model cards,
  per-fixture scores) moved to `data/backtest/<label>/` and `data/artifacts/` (both under the gitignored
  `data/`). `backtest/report.py::write_report`/`render_report` now take separate `data_dir` and
  `report_dir` args; `run_backtests()` gained a `report_dir` param alongside `data_dir` so internal-only
  passes (the replay evaluator's train-only fold) never leak a file into the public tree. The old
  `REPORT.md`/`summary.json` naming and the giant per-league/per-strategy tables are gone — the generated
  report is one page: headline edge_bits vs market and the best-CLV staking result per model. The root
  `CASE_STUDY.md` was reset to an index (the previous hand-written 2026-09-06 narrative had drifted from
  the codebase — GRU/GNN/in-play models and the lead-lag/referee-lag modules it described no longer
  exist) and is populated per-test by the new `.claude/skills/case-study/SKILL.md` Claude Code skill,
  which runs a test, reads only its local data, and writes an honest write-up — never a fabricated
  number. `reports/*` files that were git-tracked before this change were removed from tracking (moved
  under `data/`, not deleted) pending a fresh, verified case study for each label.
