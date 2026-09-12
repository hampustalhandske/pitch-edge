# API keys — what's optional, where to get it, what it unlocks

**Nothing in PITCH-EDGE requires a key.** Every source in the current warehouse is free and keyless.
The keys below are optional add-ons, in priority order. Set them as environment variables (or in a
`.env` file in the repo root — `.env` is git-ignored) and re-run `uv run pitch-edge ingest`.

The easiest way is the guided flow:

```bash
uv run pitch-edge setup            # shows what each key unlocks, asks "do you have one?", writes .env (input hidden)
uv run pitch-edge setup --non-interactive   # status only — never prompts
```

`setup` never blocks: decline every prompt and nothing is written. Values are never echoed or logged, the
`.env` file is created owner-read/write only, and the registry the command reads (`pitch_edge/keys.py`) is
the same one the dashboard's "optional keys" panel uses — a test asserts this page names every entry, so
the two cannot drift. Keys are handled in the terminal only; the dashboard never collects a secret.

```bash
# .env  (never commit this file)
API_FOOTBALL_KEY=
ODDS_API_KEY=
EVERYSPORT_API_KEY=
ANTHROPIC_API_KEY=
```

No key here can place a bet or move money; there is no bookmaker/exchange account anywhere in the system.

---

## 1. API-Football — injuries, confirmed lineups, substitutions  (highest value)

| | |
|---|---|
| Env var | `API_FOOTBALL_KEY` |
| Sign-up | https://dashboard.api-football.com/register (free plan, no card) |
| Free tier | 100 requests/day, current + last season, all endpoints |
| Connector | `pitch_edge.data.alt.api_football.APIFootballSource` — done, tested, off until the key exists |
| Unlocks | `injuries(league, season)`, `lineups(fixture)`, `substitutions(fixture)` → the first data the closing price does **not** already contain; feeds the "who is actually playing" features and the information-asymmetry test in `CASE_STUDY.md`; turns section 3 of the Match Intel dossier from **provisional** (last XI used, from Transfermarkt) to **confirmed** |
| League ids wired | E0 39 · D1 78 · SP1 140 · I1 135 · F1 61 · N1 88 · P1 94 · SWE1 113 · SWE2 114 |
| Budget note | responses are disk-cached, so the 100/day is spent only on new fixtures; ~2 leagues of injuries + lineups per day fits |

## 2. The Odds API — live pre-match odds from real bookmakers

| | |
|---|---|
| Env var | `ODDS_API_KEY` |
| Sign-up | https://the-odds-api.com (free plan: 500 requests/month, no card) |
| Connector | **not written yet** — one class behind `pitch_edge.odds.base.OddsProvider`; ~1 hour of work once a key exists |
| Unlocks | replaces the *labelled synthetic* Elo-derived quotes in the signal pipeline with real soft-book prices (Bet365, Unibet, Pinnacle where licensed…), enables the cross-book **steam detector** (`pitch_edge.odds.steam`) on live snapshots, gives the **lead-lag test** (`pitch_edge.odds.leadlag`) a bookmaker leg next to Kalshi/Polymarket, and lets the scheduler store real pre-match → closing sequences for future CLV |
| Budget note | 500/month ≈ one snapshot of 5 leagues every ~7 hours; the scheduler cadence is configurable |

## 3. Everysport — Swedish football below Allsvenskan

| | |
|---|---|
| Env var | `EVERYSPORT_API_KEY` |
| Sign-up | https://www.everysport.com → developer/API section (free registration) |
| Connector | `pitch_edge.data.sources.everysport.EverysportSource` — done, tested, off until the key exists |
| Unlocks | results and fixtures for Superettan, Ettan, Division 2/3 — deeper Swedish coverage than openfootball/TheSportsDB; the leagues where local information is thinnest and the thesis is most plausible |

## 4. Anthropic — Claude as the RAG explainer

| | |
|---|---|
| Env var | `ANTHROPIC_API_KEY` |
| Sign-up | https://console.anthropic.com |
| Connector | `pitch_edge.rag.generate.GroundedGenerator` — uses `claude-fable-5-1` (override with `PITCH_EDGE_LLM_MODEL`) via the official SDK when the key is present; otherwise a deterministic template |
| Unlocks | a last-resort fallback for `rag/generate.py` if the configured provider (Ollama or Groq) is unreachable (dropped if a single figure fails the citation check). The LLM never produces a probability; every number is quoted from already-computed evidence |
| Cost note | one short answer ≈ a few thousand input tokens; the system prompt is cached |

## 5. Groq Cloud — a free, faster alternative to local Ollama

| | |
|---|---|
| Env var | `GROQ_CLOUD_API_KEY` |
| Sign-up | https://console.groq.com |
| Connector | `pitch_edge.agents.llm.get_llm` — set `PITCH_EDGE_LLM_PROVIDER=groq` to route every `ask`-agent LLM call (`parse_intent`, `judge`, `structure_context`'s squad-value hint) and `rag/generate.py`'s local-first step through Groq instead of Ollama; `PITCH_EDGE_GROQ_MODEL` overrides the default `openai/gpt-oss-20b` |
| Unlocks | noticeably faster structured-output/tool-calling responses than a local 8B Ollama model, without needing `ollama serve` running at all |
| Cost note | free tier is a real recurring daily quota (not a trial credit): 1,000 requests/day and 200,000 tokens/day on `openai/gpt-oss-20b`, with an 8,000-tokens-per-minute ceiling that a bursty run (e.g. a `top_bets` question with `judge` making several tool calls) can hit — a 429 there degrades to this project's existing "provider unreachable" fallback, same as Ollama being down |

---

## Not keys, but worth knowing

| Source | State | What to do |
|---|---|---|
| football-data.co.uk | returning HTTP 503 since 2026-09-05 evening (their outage, keyless) | nothing — `pitch-edge ingest` upgrades the spine in place when it's back: Pinnacle early/closing prices → **real CLV**, referee names, `SWE.csv` with Swedish odds |
| Club Elo | HTTP 502 (their outage, keyless) | nothing — Elo already comes from the spine |
| TheSportsDB | works with the public free key `3`, but caps season results and rate-limits player search | a paid Patreon key (~$3/month) lifts the caps; not needed |
| Google Cloud (Phase 3) | code + manifests in `deploy/`, never run | `gcloud auth application-default login` and `PITCH_EDGE_GCP_PROJECT` / `PITCH_EDGE_GCS_BUCKET` (both also asked for by `pitch-edge setup`) when you want the GCS/BigQuery mirror |
| Wikimedia pageviews API | keyless, generous ToS, used since Phase 5 | nothing — `pitch-edge ingest` pulls daily club-article views; honest UA, throttled, year-chunks cached |
| Referee appointments | public federation pages, robots-checked | nothing — `data/alt/referee_announcements.py` polls the configured pages; the Premier League page is JS-rendered, so parsed rows may be zero (logged, not hidden) |

## Deliberately not obtainable / not used

- **X / Twitter** — read access is paid ($100+/month); RSS + Bluesky (public, currently 403) are the text channels instead.
- **Betfair Exchange, Pinnacle, Sportmonks, Opta/StatsBomb paid** — need funded or B2B accounts; the `OddsProvider` / `MatchDataSource` interfaces are ready if that ever changes.
- **FotMob, Sofascore, Flashscore, Understat** — no public API and their robots.txt / ToS forbid scraping. Not used, by design.
- **Stadium audio** — no rights-cleared feed; the feature extractor exists, the data stays a stub.
