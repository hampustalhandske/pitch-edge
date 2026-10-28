# Evidence standards — the ladder every strategy climbs

A strategy (a rule that turns information available at decision time into a paper bet) is graded by the
highest rung it clears. Rungs are cumulative: a rung is only checked once every rung below it holds.
No rung can be cleared by editing the report; only by evidence produced by code in this repo.

Nothing here ever places a real bet. The top rung is *paper* trading.

| Rung | Name | What must be true |
|---|---|---|
| L0 | Registered | A hypothesis file exists in `research/registry/<id>.yaml` **before** any outcome-bearing data is scored. It states the direction, the primary metric, the disconfirming result, the data window, the decision timestamp, the family it belongs to for multiple testing, and the parameter grid (every grid point is a trial). The file's sha256 and registration time are recorded in the trials ledger; editing it after a run creates a new hypothesis id. |
| L1 | Clean | Every input is point-in-time: each feature/price used for a decision carries `known_at <= decision_ts` (asserted in code, not by inspection). Rerunning with the same config hash and seed reproduces the headline numbers exactly. The run records the git sha, config hash, data snapshot and row counts. |
| L2 | Signal | Walk-forward out-of-sample only. Against the *sharp* benchmark for that market (see below) the strategy shows a positive primary metric with a 95% **block-bootstrap** CI (resampled by match-day, ≥2,000 draws) that excludes zero, on at least 300 decisions. |
| L3 | Survives search | (a) Benjamini-Hochberg q < 0.10 across every hypothesis in its family that has ever been run (the ledger is append-only, so dropped ideas still count). (b) Deflated Sharpe ratio ≥ 0.95 using the ledger's trial count. (c) For parameterised strategies, probability of backtest overfitting (CSCV) < 0.25. (d) Effect size is within the literature ceiling for its kind (a bigger effect is a leakage suspect, not a win, until the auditor clears it). (e) Same sign in at least 2 of 3 chronological sub-periods, and not carried by a single league: with any one league removed, the 95% CI still excludes zero (signs alone are not enough). (f) For predictive (non-return) metrics, DSR and PBO are `n/a`, never "pass". |
| L4 | Independent | (a) The `skeptic-auditor` agent has re-derived the headline number from raw tables with its own code and signed off in `research/audits/<id>.md`. (b) The owner's private edgelab grading engine grades it at least `promising` on the `fast` profile, or, when edgelab refuses on data-fit grounds, the refusal is recorded and the rung is marked `L4-partial`. edgelab is a second opinion, not an oracle: disagreements are investigated and written up. For predictive or descriptive findings (`metric_kind` probability / score) edgelab has nothing to grade: (b) is `n/a`, L4 is the auditor's signature, and L4 is their top rung (L5 applies to strategies only). Only `return`-kind strategies can ever be shown as an edge. |
| L5 | Forward | Paper-traded on fixtures that had not kicked off when the decision was logged: ≥ 150 settled paper bets (or ≥ 8 weeks) with positive CLV against the sharp close, CI reported. Logged decisions are immutable. |

## Sharp benchmarks (what "the market" means)

| Market | Decision price | Benchmark / close |
|---|---|---|
| Bookmaker 1X2, matches ≤ 2025-09 | Pinnacle opening (`pinnacle_*_open`) | Pinnacle closing, de-vigged (power method; Shin as sensitivity). Complete through 2025-09, ~30% in late 2025, none from 2026-02. |
| Bookmaker 1X2, matches ≥ 2025-10 | Pinnacle opening where present, else the Betfair Exchange opening (book sum in [1.00, 1.05], below); `max_*_open` only for families registered before 2026-10-28 | Betfair Exchange closing (`betfair_ex_*_close`, coverage from 2024-08), de-vigged. Over the overlap 2024-08 → 2025-09 report both, so the switch of benchmark is itself measured. |
| Forward paper trades | live quote at `decided_at` (the-odds-api Pinnacle/Betfair/Smarkets; Polymarket/Kalshi books) | Betfair Exchange close from the lake; Polymarket last pre-kickoff bar |
| Polymarket soccer | hourly bar close at decision hour, plus the taker fee 0.05·p·(1−p) and half the bar's high-low as a spread proxy | Polymarket last pre-kickoff bar, and the sharp bookmaker close for the same match |
| Kalshi soccer | hourly candle at decision hour, with Kalshi fees | as Polymarket |

**Reference-book validity (every betting family registered after 2026-10-28).** A Betfair Exchange book used
as a fair reference or a benchmark is valid only when its back-price book sum over the three 1X2 outcomes
(Σ 1/odds) lies in **[1.00, 1.05]**. A book outside that range is too thin to price fairly, so the match is
dropped before any decision and counted in the report. The band comes from market structure, not from any
strategy's result. On the 2024-08 to 2025-06 football-data main-league files, Betfair Exchange **closing** books,
which are liquid, have a median sum of 1.007 and a 99.5th percentile of 1.041, with 0.16% above 1.05. Opening books
have a median of 1.02, but 23% exceed 1.05 and 11% exceed 1.10, many at 1.01-1.10 on every outcome (an empty book).
The registration declares the screen in its universe. A pre-2026-10-28 registration keeps its own rule, and the
report states the share of its decisions that would fail the band. The first such case, H1
`soft-open-vs-betfair-fav-open-v1`, was found after that hypothesis was scored.

**Rules for every family registered after 2026-10-28.**
1. **One named venue.** Every bet is priced at one named venue's quote at decision time: a single bookmaker's
   column (e.g. `bet365_*_open`), the Betfair Exchange, Polymarket or Kalshi (edgelab's E2 P5). A best price across
   books (football-data's `max_*`) is never the price taken, not even as a labelled upper bound. It may appear
   only as a placebo or reference line beside the named-venue result. Line shopping at the max price inflated H1, H2
   and H3 of `soft-open-vs-betfair`.
2. **Count the test bets before registering.** Before a registration is recorded, the rule's bets on the
   evaluation window are counted outcome-blind: from decision-time inputs only, with no close and no result of the
   decision match, the same way H2's train count was taken. A family whose expected test bets are below its
   `min_n` does not register as a test on that window. It waits for forward paper data. H2 expected 300-400 test
   bets and got 5.

Primary metrics, chosen at registration: **CLV** (log of taken odds over de-vigged closing odds, averaged
per bet), **log-loss edge in bits** vs the de-vigged close (`backtest/slices.py` convention), or **Brier
skill vs the market price** for prediction-market strategies. ROI is always reported and never primary
(it has far more variance than CLV for the same information).

## The ratchet

For each market family there is a **champion**: the best strategy that holds the highest rung. A new
strategy in that family is only promoted if it beats the champion on a *paired* comparison over the same
decisions (bootstrap CI of the difference excludes zero), not merely by clearing the rung on its own.
When a champion reaches a new rung, the minimum rung for anything shown in the `ask` agent or dashboard
rises with it. Losing and null results are written up and kept, never deleted.

## Literature ceilings (sanity bounds, not targets)

- Beating the de-vigged Pinnacle close on 1X2 log-loss: published public-data models land between
  −0.03 and +0.005 bits. Anything above +0.02 bits is a leakage suspect.
- CLV vs Pinnacle close from public-information models: a few tenths of a percent at most; above +2%
  per bet over hundreds of bets is a leakage suspect.
- Favourite–longshot bias in sharp closing lines: small (≈1–3% of implied probability at the extremes),
  larger in soft books and lower divisions.
- Prediction-market vs sharp-book gaps: larger and genuinely plausible (thin liquidity, retail flow), but
  must be net of fees and the spread you would actually cross.
