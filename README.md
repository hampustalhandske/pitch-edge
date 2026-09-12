# PITCH-EDGE

[![CI](https://github.com/hampusstalhandske/pitch-edge/actions/workflows/ci.yml/badge.svg)](https://github.com/hampusstalhandske/pitch-edge/actions/workflows/ci.yml)

A Python football (soccer) intelligence & market-edge research platform, narrowed to two layers: a
deterministic backtest/evidence pipeline, and a LangGraph `ask` agent that answers exactly three
standardized questions — "top N bets", "team A vs team B", "what's on day X" — grounded in that evidence.
It **surfaces — never places —** anything: there is no staking or order-placement code anywhere in the
project.

> **Hard guardrails.** No automated bet placement under any flag or config. No real-money movement. Every
> number the `ask` agent states traces back to a walk-forward backtest or a real, vig-inclusive market price.

## Quickstart

```bash
uv sync --extra dev            # Python 3.11; torch, lightning, langchain, chromadb, langgraph, nicegui …
uv run pitch-edge refresh      # ingest → features → backtests → RAG index
uv run pitch-edge serve        # dark-mode dashboard on :8501 — one page, a question box
uv run pytest                  # unit + integration tests, network mocked
```

No API keys are required for any of the above. Optional keys (API-Football, The Odds API, Everysport,
Anthropic) are documented in `API_KEYS.md`; `uv run pitch-edge setup` walks through them interactively and
writes a git-ignored `.env`. `uv sync --extra cloud` adds the optional GCS/BigQuery mirror.

```bash
uv run pitch-edge ask "top 4 bets"                 # the only Q&A entrypoint (needs `ollama serve` locally)
uv run pitch-edge ask "Liverpool vs Arsenal"       # --as-of YYYY-MM-DD to answer as of a specific date;
uv run pitch-edge ask "what's on 2024-03-16"       # default: the most recent real closing-odds date
uv run pitch-edge ablation | backtest | rag-eval | health | export
```

`PITCH_EDGE_LLM_PROVIDER=groq` (with `GROQ_CLOUD_API_KEY` set, see `API_KEYS.md`) swaps every `ask`-agent LLM call from local Ollama to [Groq](https://console.groq.com)'s free tier — a real recurring daily quota (1,000 requests/day, 200k tokens/day on the default `openai/gpt-oss-20b`), confirmed several times faster and equally structured-output-reliable than an 8B local model, with no `ollama serve` needed at all.

For results, known limitations and the honest headline numbers, see `CASE_STUDY.md`. Model details are in
`MODEL_CARDS.md`.

## Architecture

```mermaid
flowchart LR
  subgraph Sources["Data Sources"]
    FD[football-data.co.uk<br/>results + multi-book odds]
    CF[Club-Football-Match-Data<br/>2000-2025 spine · Elo]
    TM[Transfermarkt open extract<br/>players · rotation · valuations]
    OF[openfootball<br/>fixtures · Sweden]
    SB[StatsBomb Open Data<br/>events]
    ESPN[ESPN soccer data]
    ALT[weather · travel · referee<br/>news EN/SV · Wikipedia attention]
    PM[Polymarket · Kalshi]
    ODDS[The Odds API<br/>live 1X2 quotes]
  end
  Sources --> HTTP[CachedHttpClient<br/>throttle · disk cache · robots · circuit breaker]
  HTTP --> WH[(DuckDB warehouse<br/>+ Parquet lake)]
  WH --> FS[Feature store<br/>Elo · form · rotation · referee · weather · travel]
  FS --> M1[Dixon-Coles per league]
  FS --> M2[GBDT ± market, ± confirmed lineups]
  FS --> M3[Lightning Transformer]
  M1 & M2 & M3 --> BT[Walk-forward backtester<br/>no-vig edge · Kelly · CLV · calibration · ablation]
  BT --> SL[Circumstance-sliced evidence<br/>league/referee/rest/travel/weather · min-n + FDR · checkpointed]
  WH & SL --> RAG[LangChain hybrid RAG<br/>Chroma ∪ BM25 → RRF → rerank]
  RAG & SL --> ASK[ask LangGraph agent<br/>parse_intent → gather (deterministic) → judge (tool-calling)]
  ASK --> UI[NiceGUI dashboard · one page · a question box]
```

## Models

One `MatchModel` interface (`fit`, `predict_proba`, `card`), identical walk-forward folds for all of them:

| Model | What it is |
|---|---|
| `dixon_coles` | bivariate Poisson + τ correction, exponential decay, one bounded MLE fit per league |
| `gbdt` | histogram GBDT (LightGBM if installed, else scikit-learn HGB) on the feature store |
| `gbdt_mkt` | same + *early* (never closing) no-vig market probabilities |
| `gbdt_squadval` / `gbdt_mkt_squadval` | same recipe + confirmed-lineup squad value / missing-star-value features |
| `transformer_sequence` | PyTorch **Lightning** `TransformerEncoder` over each team's recent match sequence |

Isotonic calibration is fit per outcome on realised past folds only. Model cards live in
`data/backtest/main/model_card_*.json` (local; `reports/main/CASE_STUDY.md` is the public write-up).

## Backtesting

Walk-forward only — train strictly before each fold, periodic retrain, never shuffled across time. Bets are
priced at real quoted odds (Pinnacle early when present, otherwise market-average, and the report says
which). CLV is measured against the closing line. ¼-Kelly, ½-Kelly and flat-stake strategies with hard caps,
Monte Carlo drawdown, per-bet Sharpe. Every run writes a one-page `reports/<label>/CASE_STUDY.md`
automatically; the full per-division/per-strategy breakdown, a feature-group ablation, and
circumstance-sliced evidence (`slice_evidence.csv` — see `backtest/slices.py`) land in
`data/backtest/<label>/` (local, gitignored). Losing strategies are published as-is.

## RAG

Dense retrieval (`Chroma` + `sentence-transformers`) fused with sparse (`rank_bm25`) via reciprocal-rank
fusion, then cross-encoder reranked. Generation is **local-first**: an Ollama model is tried before Claude,
which is used only as a fallback when `ANTHROPIC_API_KEY` is set and local generation fails; a deterministic
template is the final fallback offline. Every answer is citation-verified (`verify_citations`) — a number
absent from the retrieved sources fails the check rather than shipping. `pitch-edge rag-eval` scores
hit@k/MRR against synthetic QA generated from the corpus.

## Agentic layer

One LangGraph `StateGraph` (`agents/qa_graph.py`), answering exactly three standardized questions —
"top N bets", "team A vs team B", "what's on day X" — as of an explicit point in time (`--as-of`, or the
most recent real closing-odds date by default; there is no live-odds feed today, so there is no genuine
"now" to answer as of instead):

- **`parse_intent`** — the only LLM call that classifies the question, into one of the three intents above
  or `unrecognized` ("I can't understand that" — there is no generic freeform fallback).
- **`gather`** — deterministic: resolves candidate fixtures (a real, non-synthetic early Pinnacle quote is
  required), fits every model fresh on data strictly before `as_of` (nothing in this project persists
  fitted weights, so this is a real refit every time, not a cache lookup), computes each fixture's no-vig
  market edge, and picks which model to trust per fixture from real backtest evidence matching that
  fixture's own circumstances (`backtest/slices.py::select_trusted_model`) — never an LLM guess.
- **`judge`** — a bounded, tool-calling react agent (`agents/evidence_tools.py`: `get_model_predictions`,
  `get_backtest_evidence`, `search_context`, each argument-validated before touching any data) that
  narrates the batch `gather` already computed, with one bounded reflection retry if its own citations
  don't check out (`rag/generate.py::verify_citations`). It never computes a probability or an edge itself.

## Dashboard

`uv run pitch-edge serve` launches a [NiceGUI](https://nicegui.io) app (Python-native, FastAPI + Vue/Quasar)
— chosen because it's a direct front end for the same `ask` graph the CLI uses, with no API contract or
separate frontend to maintain. One page: a question box and an answer, nothing else. Dark theme,
short-lived read-only warehouse connections so the dashboard never blocks the pipeline.

## Layout

```
src/pitch_edge/
  config.py, keys.py     settings (env/.env), optional-key registry behind `pitch-edge setup`
  data/http.py            cached, throttled, robots-aware HTTP client with circuit breaker
  data/contracts.py       pandera schemas (fail loudly on drift)
  data/sources/           football-data.co.uk, Club-Football-Match-Data, Transfermarkt, openfootball,
                          StatsBomb, ESPN, TheSportsDB, Club Elo, Everysport
  data/alt/                odds (live + prediction markets), weather, travel, referee, news, Wikipedia attention
  data/storage.py, ingest.py   DuckDB warehouse + orchestrated ingestion with per-source health logging
  features/                pre-match feature store, rotation/lineup context, squad value (leakage-safe)
                          — see `features/README.md` for what's in the warehouse and exactly which
                          columns feed each model
  models/                  dixon_coles, gbdt, sequence (Lightning Transformer), calibration
  backtest/                walk-forward engine, Kelly + Monte Carlo, metrics, circumstance-sliced
                          evidence (slices.py), point-in-time cutoffs (as_of.py), report writer
  odds/                    OddsProvider interface, no-vig maths (utils.py, edge.py), steam/divergence detector
  rag/                     documents (incl. evidence docs), hybrid retrieval, grounded generator
                          (local LLM / Claude), ask-intent query parser
  agents/                  the `ask` LangGraph agent (qa_graph.py, qa_data.py, qa_context.py,
                          evidence_tools.py, llm.py)
  vendor/                  third-party connectors vendored with a documented fix (see `vendor/NOTICE.md`)
  ablation.py, artifacts.py, pipeline.py, scheduler.py   feature ablation, dashboard artifacts, single
                          pipeline entrypoint, APScheduler jobs
  cloud/sync.py            optional GCS + BigQuery mirror
  dashboard/                web.py (NiceGUI, one page), theme.py (dark palette)
  cli.py                   ingest | features | backtest | ablation | ask | rag-eval | artifacts |
                          health | export | refresh | serve | schedule | setup
scripts/                  restartable long-run stage scripts (see `scripts/README.md`)
deploy/                    Dockerfile, Cloud Run / Scheduler commands
tests/                     pytest unit + integration tests, network mocked
```

## Docs

`CASE_STUDY.md` is an index into `reports/<label>/CASE_STUDY.md` — one honest, one-page write-up per
test (backtest/ablation/replay-eval), generated by the `case-study` Claude Code skill
(`.claude/skills/case-study/SKILL.md`); `reports/` holds only those write-ups, never raw data. The
CSVs/model cards behind them are local, under `data/backtest/` and `data/artifacts/` (gitignored).
See also `src/pitch_edge/features/README.md` (what's in the warehouse and exactly which columns
feed each model, generated from the current code rather than hand-maintained), `MODEL_CARDS.md`,
`API_KEYS.md`, `scripts/README.md`, `src/pitch_edge/vendor/NOTICE.md`.
