# API keys — what each one is for, where to get it, what it unlocks

The warehouse pipeline (`ingest`, `features`, `backtest`, `refresh`, `serve`) needs no key: every source it reads
is free and keyless. Three things do need credentials:

- the **research layer** (`lake-sync`, the studies, `paper settle`) reads the owner's lakehouse:
  `LAKEHOUSE_API_URL` and `LAKEHOUSE_API_TOKEN`;
- the **paper loop's bookmaker quotes** come from the-odds-api: `ODDS_API_KEY`;
- the **`ask` agent** needs an LLM: a local `ollama serve` (keyless, the default) or Groq
  (`GROQ_CLOUD_API_KEY` with `PITCH_EDGE_LLM_PROVIDER=groq`).

Set them as environment variables or in a `.env` file in the repo root (`.env` is git-ignored). The optional keys
in the registry below have a guided flow:

```bash
uv run pitch-edge setup                     # shows what each key unlocks, asks "do you have one?", writes .env (input hidden)
uv run pitch-edge setup --non-interactive   # status only — never prompts
```

`setup` never blocks: decline every prompt and nothing is written. Values are never echoed or logged, the `.env`
file is created owner-read/write only, and the registry the command reads (`pitch_edge/keys.py`) is checked by a
test that asserts this page names every entry, so the two cannot drift. Keys are handled in the terminal only;
the dashboard never collects a secret.

```bash
# .env  (never commit this file)
LAKEHOUSE_API_URL=http://127.0.0.1:8600
LAKEHOUSE_API_TOKEN=
ODDS_API_KEY=
API_FOOTBALL_KEY=
EVERYSPORT_API_KEY=
ANTHROPIC_API_KEY=
GROQ_CLOUD_API_KEY=
PITCH_EDGE_LLM_PROVIDER=ollama   # or groq
```

No key here can place a bet or move money; there is no bookmaker/exchange account anywhere in the system, and
every market-data call is a public, read-only GET.

---

## Optional keys in the `setup` registry

### 1. API-Football — injuries, confirmed lineups, substitutions

| | |
|---|---|
| Env var | `API_FOOTBALL_KEY` |
| Sign-up | https://dashboard.api-football.com/register (free plan, no card) |
| Free tier | 100 requests/day; the free plan does not cover the current season, which limits its use for upcoming fixtures |
| Connector | `pitch_edge.data.alt.api_football.APIFootballSource` — tested, off until the key exists |
| Unlocks | `injuries(league, season)`, `lineups(fixture)`, `substitutions(fixture)` — data the closing price does not already contain; `features/live_lineups.py` uses a published starting XI to mark an upcoming fixture's lineup as `confirmed` instead of `provisional` (last XI used, from Transfermarkt) |
| League ids wired | E0 39 · D1 78 · SP1 140 · I1 135 · F1 61 · N1 88 · P1 94 · SWE1 113 · SWE2 114 |
| Budget note | responses are disk-cached, so the daily quota is spent only on new fixtures |

### 2. Everysport — Swedish football below Allsvenskan

| | |
|---|---|
| Env var | `EVERYSPORT_API_KEY` |
| Sign-up | https://www.everysport.com → developer/API section (free registration) |
| Connector | `pitch_edge.data.sources.everysport.EverysportSource` — tested, off until the key exists |
| Unlocks | results and fixtures for Superettan, Ettan, Division 2/3 — deeper Swedish coverage than openfootball/TheSportsDB |

### 3. Anthropic — Claude as the last-resort RAG generator

| | |
|---|---|
| Env var | `ANTHROPIC_API_KEY` |
| Sign-up | https://console.anthropic.com |
| Connector | `pitch_edge.rag.generate.GroundedGenerator` — uses `claude-fable-5-1` (override with `PITCH_EDGE_LLM_MODEL`) via the official SDK |
| Unlocks | a fallback for `rag/generate.py` when the configured provider (Ollama or Groq) is unreachable; an answer is dropped if a single figure fails the citation check. The LLM never produces a probability; every number is quoted from already-computed evidence |
| Cost note | one short answer ≈ a few thousand input tokens; the system prompt is cached |

### 4. Groq Cloud — a hosted alternative to local Ollama

| | |
|---|---|
| Env var | `GROQ_CLOUD_API_KEY` |
| Sign-up | https://console.groq.com |
| Connector | `pitch_edge.agents.llm.get_llm` — with `PITCH_EDGE_LLM_PROVIDER=groq`, every `ask`-agent LLM call (`parse_intent`, `judge`, `structure_context`'s squad-value hint) and `rag/generate.py`'s first generation step go through Groq instead of Ollama; `PITCH_EDGE_GROQ_MODEL` overrides the default `openai/gpt-oss-20b` |
| Unlocks | faster structured-output and tool-calling responses than a local 8B model, with no `ollama serve` needed |
| Cost note | the free tier is a recurring daily quota: 1,000 requests/day and 200,000 tokens/day on `openai/gpt-oss-20b`, with an 8,000-tokens-per-minute ceiling that a `top_bets` question with several `judge` tool calls can hit; a 429 degrades to the same "provider unreachable" fallback as Ollama being down |

## Credentials outside the registry

| Env var | Used by | Notes |
|---|---|---|
| `LAKEHOUSE_API_URL`, `LAKEHOUSE_API_TOKEN` | `pitch_edge.lake.client.Lake` — `pitch-edge lake-sync`, the research studies, `pitch-edge paper settle` | the owner's lakehouse API (`http://127.0.0.1:8600`; use `127.0.0.1`, not `localhost`). `lake-sync` exits with an error when the token is unset |
| `ODDS_API_KEY` | `pitch-edge paper collect --sources odds_api` (`paper/collectors.py`) | the-odds-api free tier, 500 credits/month: live Pinnacle, Betfair Exchange, Smarkets and soft-book quotes, no history. The collector spends 2 credits per league per call, at most once per league per 23.5 h, stops below 60 credits remaining and keeps the projected spend ≤ 450/month. The `ask` agent does not use it |

## Keyless sources worth knowing

| Source | How it is read |
|---|---|
| football-data.co.uk (results, multi-book open/close odds) | for research, through the lake's `fd_matches` (`lake_fd_matches` mirror); the warehouse connector reads the local `data/archive/` mirror first and only then the network |
| Club Elo, Understat, Transfermarkt, Polymarket, Kalshi, Bluesky, Reddit | through the lake for research; the warehouse also ingests Club Elo and a Transfermarkt open extract directly |
| TheSportsDB | public free key `3`, which caps season results and rate-limits player search |
| Wikimedia pageviews API | `pitch-edge ingest` pulls daily club-article views; honest UA, throttled, year-chunks cached |
| Referee appointments | public federation pages, robots-checked (`data/alt/referee_announcements.py`); the Premier League page is JS-rendered, so parsed rows may be zero (logged, not hidden) |

## Not used

- **X / Twitter** — read access is paid ($100+/month).
- **Betfair Exchange, Pinnacle, Sportmonks, Opta/StatsBomb paid APIs** — need funded or B2B accounts; the
  `OddsProvider` / `MatchDataSource` interfaces are ready if that changes. Betfair and Pinnacle prices still
  arrive through football-data (via the lake) and the-odds-api.
- **FotMob, Sofascore, Flashscore** — in scope, no connector yet.
- **Stadium audio** — no rights-cleared feed; the feature extractor exists, the data stays a stub.
