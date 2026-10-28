# Proposed hypotheses for owner approval (2026-10-28)

Drafts only: nothing here is registered or run. Approval of an item means pitch-edge writes its registration in
`research/registry/`, records it on the ledger, and only then scores the test window (`research/STANDARDS.md`).

**Data scope.** Only data the lake holds now: football-data opens and closes (1X2 for Pinnacle, bet365, market
average and max, Betfair Exchange; Betfair columns from 2024-08), the Betfair Exchange close as the sharp reference,
Polymarket and Kalshi bars, and OddsPapi price paths from 2026-01 once the lake loads them (LK-45). The decision is
taken at football-data's opening snapshot (Fri/Tue afternoon, 1-3 days before kickoff) unless stated otherwise.

**Test window.** 2025-07-01 to 2026-10-19 (8,792 matches with Betfair open, Betfair close and max open; 316
match-days), minus 2026-04-17 to 05-04 where football-data has no Betfair close (PE-26). R2-Q2, R2-Q3, R3 and R6
already scored post segments in this window for other rules, so a result here is weaker evidence than a fresh
window; the forward paper loop is the clean test for anything that passes.

**Power.** E2 draft planning figures: per-bet CLV sd about 0.10, about 1,000 bets for 80% power at +1% CLV. That
implies power at +2% CLV of about Φ(0.2·√n − 2.32) for n bets. Match-day clustering lowers the effective n, so these
are upper bounds.

| # | Kind | Hypothesis (one line) | Data | Train / test | Expected test bets | Power at +2% |
|---|---|---|---|---|---|---|
| H1 | basic rule | Backing a strong favourite (Betfair-open fair p ≥ 0.70) at the best soft open, when that price beats the Betfair-open fair price by θ ∈ {0, 1%}, has positive CLV against the Betfair close (soft books under-shorten favourites). | fd 1X2 opens (max, Betfair), Betfair close | train 2024-08 to 2025-06 (choose θ, p cut-off) / test 2025-07 to 2026-10 | 306 at θ = 0, 200 at θ = 1% (counted) | 0.88 / 0.69 |
| H2 | basic rule | In the last 5 rounds, the side that still needs points (title, Europe, relegation) is under-priced at the best soft open against a side with nothing at stake, measured by CLV against the Betfair close. | fd results (point-in-time tables) and 1X2 opens, Betfair close | rule fixed in advance; train 2012-07 to 2024-06 against the Pinnacle close to set θ / test 2025-07 to 2026-10 | about 300-400 (estimate: 13% of matches in the last 5 rounds × about 30% asymmetric; not counted) | about 0.87-0.95 |
| H3 | ML | A gradient-boosted model of the de-vigged sharp close from soft-book opens only (average, max and bet365 levels and spreads, no Pinnacle or Betfair inputs) finds best soft opens priced above the predicted fair close by θ, with positive CLV against the Betfair close. | fd 1X2 opens; target Pinnacle close (train), Betfair close (test) | train 2012-07 to 2024-06 (walk-forward folds) / test 2025-07 to 2026-10 | about 1,500-2,500 at θ = 2% (estimate from R2-Q3's 25% bet rate; not counted) | about 1.00 |
| H4 | rule on price paths | When Pinnacle shortens an outcome by ≥ 3% within an hour before kickoff and the Betfair Exchange back price has not moved, backing on Betfair at that price (within the size offered) has positive CLV against the Pinnacle close (the exchange lags the sharp book). | OddsPapi Pinnacle and Betfair Exchange paths with offered size (LK-45; 6,740 finished matches from 2026-01-13 in 26 sampled tournaments incl. UCL/UEL; bet365 is not in OddsPapi) | train 2026-01-13 to 2026-05-31 / test 2026-06-01 to 2026-10-19 | not yet counted: the backfill finishes about 2026-10-30 | n/a until counted; 80% needs about 250 bets |

**Considered and not proposed.**
- Kalshi longshots against the Betfair close: only about 900 Kalshi 1X2 events since 2026-01 link to a
  football-data match, so the longshot side gives roughly 150 bets (power about 0.5). R1 and R5 found the
  prediction-market rules null.
- Totals or Asian handicap against Betfair: football-data has no Betfair over/under or handicap prices.

**What each would show.** H1 and H2 are cheap and fully specified now. H3 is the ML candidate: it replaces R6's
Pinnacle-open inputs, which end on 2026-01-15, with books that are still published, so it can also run live in the
paper loop. H4 waits for the LK-45 backfill and is the only one that tests intraday timing; it needs the outcome codes
(101/102/103 = home/draw/away) verified against the football-data Betfair close first. A pass on any of them reaches at
most L2 to L3 on this reused window. The claim then needs the auditor, edgelab's grade and at least 150 settled
paper bets.
