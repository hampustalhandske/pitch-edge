# CLAUDE.md — project context for PITCH-EDGE

Read `README.md` first for scope; this file is the working contract for agents editing this repo.

## What this is

A Python-only football (soccer) betting-market research platform. It asks whether any rule built from
information available before kickoff beats the sharp closing price, and grades every answer on an evidence
ladder. It never places a bet. Three parts:

1. **Research layer** — pre-registered hypotheses, run as counted trials against the de-vigged sharp close,
   graded L0 (registered) → L5 (forward paper-traded) by `src/pitch_edge/research/` (registry, append-only
   hash-chained ledger in `data/research/ledger.duckdb`, block bootstrap, BH, PSR/DSR, PBO, ladder gates).
   Study code and reports live in `research/studies/<id>/`, audits in `research/audits/`. Data comes from the
   owner's lakehouse (`src/pitch_edge/lake/`), strategies can be exported to edgelab for an independent grade
   (`src/pitch_edge/edgelab/`), and the forward test is the paper-trading loop (`src/pitch_edge/paper/`, own
   DB `data/paper/paper.duckdb`).
2. **Deterministic evidence layer** — free-data ingestion → DuckDB warehouse (`data/pitch_edge.duckdb`) →
   feature store → Dixon-Coles / GBDT / stochastic-strength / sequence / sentiment-only walk-forward backtests
   scored against the closing line, stratified into circumstance slices (league, referee, rest-days gap,
   squad-value gap, travel fatigue, weather) with a minimum-sample floor and Benjamini-Hochberg correction
   (`backtest/slices.py`), checkpointed across time so the evidence is never computed with hindsight
   (`backtest/as_of.py`). Nothing here runs an LLM.
3. **`ask` agent** — a LangGraph `StateGraph` (`agents/qa_graph.py`) answering exactly three questions —
   "top N bets", "team A vs team B", "what's on day X" — as of an explicit historical point in time (it reads
   no live odds). Two LLM-touching nodes: `parse_intent` (structures the question) and `judge` (a bounded
   tool-calling agent that narrates already-computed predictions, market edges, backtest evidence and the
   research ladder; it never computes a probability or an edge). It must state the evidence rung behind
   what it narrates and say "no validated edge" while no strategy holds the display floor. The single-page
   NiceGUI dashboard (`dashboard/web.py`) is a front end for the same graph plus a read-only Evidence section.

## Research operating model

Work is run by an orchestrator session that dispatches agents; the plan, team and backlog are in
`research/PLAN.md`, the evidence ladder in `research/STANDARDS.md`, outcomes in `research/LOG.md` (one line
per item, newest last), and what is running / the next step for each unfinished item in `research/HANDOFF.md`.
Before any study or strategy work load the `research-protocol` skill; before touching lake data load
`lakehouse`; before any edgelab job load `edgelab-review`. Project agents live in `.claude/agents/`
(`quant-researcher`, `skeptic-auditor`, `lakehouse-engineer`, `edgelab-liaison`, `data-scout`); the global
`implementer`, `test-writer` and `code-reviewer` build and review code. Nothing is called an edge before a
`skeptic-auditor` signs it, and only `return`-kind strategies can ever be shown as an edge (predictive and
descriptive findings top out at L4).

MCP servers `lakehouse` and `edgelab` (`.mcp.json`, git-ignored; `.mcp.json.example` reads the private checkouts from `$LAKEHOUSE_REPO` and `$EDGELAB_DIR`, exported in the shell; enabled in `.claude/settings.json`, git-ignored, template `.claude/settings.json.example`) give the orchestrator
and agents direct tools: use `mcp__lakehouse__*` for interactive lake lookups and verification, and
`mcp__edgelab__*` for edgelab checks, dry-runs, status and summaries; study data still comes through the Python
client so results are reproducible, and counted edgelab runs need the owner's approval. Project agents only get
the MCP tools listed in their `tools:` frontmatter.

Data comes primarily from the owner's private lakehouse (a separate repo, not part of this one; read only through
its API at `LAKEHOUSE_API_URL`, token in `.env`), mirrored into `lake_*` warehouse tables by `pitch-edge lake-sync`.
Strategies are independently graded by the owner's private edgelab engine (checkout at `$EDGELAB_DIR`); its holdouts are never
unlocked without the owner.

## Non-negotiables (never relax, even in "test mode")

- **No automated bet placement, no real-money movement.** No code path anywhere in the project calls a
  bookmaker/exchange API with an order, and there is no staking or order-placement machinery. The paper loop
  only reads public market data and logs paper decisions in paper units; the `ask` agent only answers a
  question. Do not add any code path that calls a bookmaker/exchange with an order.
- **Data sources are not limited to ToS-clean ones** (owner decision): Understat, FotMob `/api`, Sofascore,
  Flashscore, X and third-party scrapers such as `WhoscoredClient` are in scope. Every fetch still goes through
  `pitch_edge.data.http.CachedHttpClient` (honest UA, throttle, disk cache, robots check, per-host circuit
  breaker) — a resilience/politeness mechanism that stays as-is.
- **No unvalidated number reaches the UI.** The dashboard reads warehouse tables, ledgers and artifacts written
  by the pipeline; it never trains, fetches odds, or computes new model numbers. Below the display floor a
  number is labelled with its rung, never as an edge.
- **The LLM explains, never estimates.** `rag/generate.py` quotes numbers from retrieved docs with `[doc_id]`
  citations; `verify_citations` flags anything ungrounded.
- **Walk-forward and point-in-time only.** No shuffled CV across time; every feature carries
  `known_at <= decision time`; bets priced at real vig-inclusive odds; CLV vs the sharp close.
- **Losing results are published.** Don't tune a report until it looks good; null results stay in the LOG.

## Commands

The shell on this Mac runs under Rosetta while the venv is arm64: prefix every Python/uv command with
`arch -arm64` (e.g. `arch -arm64 uv run pytest`), or compiled wheels fail to import.

```bash
uv sync --extra dev                   # env (torch, lightning, langchain, chromadb, langgraph, nicegui …)
uv run pytest                         # network mocked (responses); markers: unit, integration, network
uv run ruff check . && uv run ruff format . && uv run mypy src
uv run pitch-edge serve               # dashboard — one page, a question box, an Evidence section
uv run pitch-edge ask "top 4 bets"    # the only Q&A entrypoint; also "<team> vs <team>", "what's on <day>"
uv run pitch-edge refresh [--fast]    # ingest → features → backtests → RAG
uv run pitch-edge {setup|ingest|features|backtest|ablation|rag-eval|health|export|schedule}
uv run pitch-edge {ingest-pmxt|map-pmxt|backtest-timing}   # Polymarket order-book archive + tick-timing backtest
uv run pitch-edge lake-sync [--only …] [--full-refresh]   # mirror lakehouse tables into lake_* tables
uv run pitch-edge research {register|status|grade|ledger|import-trials|review}   # ledger + ladder
uv run pitch-edge paper {collect|decide|settle|report|status}   # forward paper-trading ledger
```

Every fetch, backfill and long job must be **resumable** (checkpoint per window/chunk; re-running the same
command skips completed work; never empty-then-refill without a checkpoint) and must **survive the lid being
closed**: always-on loops run under launchd (`scripts/launchd/`, see `scripts/README.md`), and
`research/HANDOFF.md` records what is running and the next step for every unfinished item — read it when
resuming, update it before pausing. Long one-off runs are launched with `nohup caffeinate -i … &`. DuckDB is
single-writer: never run two pipeline stages at once (check `lsof data/pitch_edge.duckdb`), and the dashboard
opens only short-lived read-only connections (`_read_cached` / `_query_cached`, cache keyed by DB path).

## Conventions

- Python 3.11, `uv`, `src/` layout, ruff (line length 120, E501 ignored), mypy on `src`. Tests live in
  `tests/unit` and `tests/integration`; every connector gets a mocked-HTTP test; every new model gets a
  synthetic-league test (`tests/conftest.py::make_synthetic_league`).
- New data source → subclass `MatchDataSource` (or a plain class for non-match data), route HTTP through
  `CachedHttpClient`, add a `TABLE_KEYS` entry in `data/storage.py`, an `ingest_*` in `data/ingest.py`
  wrapped by `_run` (health logging), and a mocked test. New lake table → a `MirrorSpec` in `lake/mirror.py`
  with its known-at column.
- New match model → subclass `MatchModel` (`fit`, `predict_proba`, `card`, optional `observe`), register in
  `models/__init__.py::available_models`, keep leakage checks in the card.
- New study → registration YAML in `research/registry/` before any outcome is scored, code in
  `research/studies/<id>/`, trials through `pitch_edge.research.trial.run_trial`, a `REPORT.md`, a LOG line.
- Charts: use `dashboard/theme.py` tokens (SERIES / OUTCOME / STATUS / INK) and the `pitch_edge` Plotly
  template; one y-axis per chart; colour follows the entity; status colours never reused for series.
- Docs to keep in sync when behaviour changes: `README.md`, `reports/CASE_STUDY.md` (index over the study
  reports), `API_KEYS.md` (a test checks it against `keys.py`), `scripts/README.md`, `research/HANDOFF.md`.
- Don't commit or push unless explicitly asked; `data/`, `*.log`, `.env`, large report parquet are ignored.

## State (2026-11-05)

- **Research headline:** display floor L4, **no validated edge**. The only L4 findings are predictive, not
  strategies: R6's cross-book closing-line model (`closing-line-prediction-cross-book-gbm-v1`, +6.61e-5 bits
  vs the R3 consensus-gap baseline, n=45,467; also `-linear-v1`) and book informativeness
  (`closing-line-prediction-book-informativeness-v1`, VC/Betfair informative, BW/IW/1xBet stale), both signed
  by the auditor. Best strategy rung is L2 (R2-Q3 max-odds stale-soft-open, a quoted-price upper bound; its
  bet365 version was rejected by the audit). Prediction-market (R1, R5), crowd-attention (R4), the outcome
  models (R7, L1) and the other bookmaker-structure studies are null or artefacts. Per-study index:
  `reports/CASE_STUDY.md`; live ladder: `pitch-edge research status`.
- **Family `soft-open-vs-betfair` (N 4):** H1 favourite-open L1 null, H2 late-season stakes uninformative (5 test
  bets), H3 soft-books closing-line GBM null after the family correction (mostly line shopping by its placebo),
  H4 exchange lag (`-exchange-lag-v1`, Pinnacle moves vs Betfair Exchange on OddsPapi paths) registered and
  amended on the ledger, not yet scored.
- **Lake mirror:** 10 `lake_*` mirrors (football-data matches + open/close odds, Polymarket markets and hourly
  bars, Kalshi markets and candles, Understat team matches (xG), Club Elo, Transfermarkt managers/tenures/injuries),
  checkpointed per window in `lake_sync_runs` / `lake_sync_state`. The warehouse spine's football-data `matches`/`odds`
  rows (read by `ask`, the dashboard and the backtests) are built from `lake_fd_matches` by `lake/spine.py`, run at the
  end of `lake-sync` and by `ingest`/`refresh`; no football-data CSV is fetched. A launchd template
  (`scripts/launchd/local.pitch-edge.lake-sync-fd.plist.example`, not installed until the owner loads it) syncs
  `fd_matches` twice a day and skips when the warehouse is locked; the other mirrors still run only by hand.
  `pitch-edge health` shows the mirror's last sync and its lag behind the lake per backtest league. Studies and the
  paper settle read the lake API directly.
- **Paper loop:** `scripts/paper_loop.sh` under launchd agent `local.pitch-edge.paper-loop` — hourly Polymarket,
  Kalshi and the-odds-api collection, every strategy (`control-fav-pinnacle`, `soft-vs-sharp-live`,
  `soft-vs-sharp-live-v2-{pinnacle,r6_live_model}`) and a settle on every cycle against the lake's closes.
  **Paused by the owner since 2026-10-28** (agent unloaded, plist kept; reload only on the owner's word). L5 needs
  ≥ 150 settled evaluation-window decisions per strategy; each strategy has under 10, and the loop covered about
  13% of evaluation-window hours because the laptop sleeps (PE-2, PE-16).
- **`ask` models:** `models.default_models()` (Dixon-Coles, GBDT without market odds, stochastic strength),
  fitted fresh per question; nothing persists fitted weights. Their outcome-models hypotheses are graded L1
  (R7, a registered negative), so every `ask` candidate is a model-vs-market disagreement below the floor.
- **Benchmarks:** Pinnacle close in football-data ends 2025-09; Betfair Exchange close (2024-08+) is the benchmark
  after that (`research/STANDARDS.md`). football-data has no sharp close from 2026-04-17 to 05-04 (PE-26). The
  lake's OddsPapi tables hold Pinnacle, Betfair Exchange and Singbet 1X2 price paths from 2026-01, including UCL,
  UEL and UECL (LK-45).
- **Work tracking:** `docs/ISSUES.md` (PE-ids). Commits name their issue.
- **Optional keys:** see `API_KEYS.md`; `LAKEHOUSE_API_TOKEN` is needed for `lake-sync` and the studies,
  `ODDS_API_KEY` for the paper loop's bookmaker quotes. `PITCH_EDGE_LLM_PROVIDER=groq` routes LLM calls to Groq
  instead of local Ollama.
- **Stubs by design:** stadium audio (no licensed feed), OpenSky charter attribution. The vendored WhoScored
  scraper (`vendor/whoscored/`, see `vendor/NOTICE.md`) is not wired into ingestion.
