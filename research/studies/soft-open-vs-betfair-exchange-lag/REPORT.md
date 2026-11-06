# H4 — Does the Betfair Exchange lag Pinnacle in the last hour?

Registration: `research/registry/soft-open-vs-betfair-exchange-lag-v1.yaml` (family `soft-open-vs-betfair`, family
N 4), with amendments 1-4 on the ledger. Code: `exchange_lag.py`, `pull.py`, `run.py`, shared
`../soft-open-vs-betfair-common/sobf.py`. Secondary declared before scoring: `SECONDARY.md` (commit `2aa931b`).
Counted trial: one, `9caa2889…`, git `2aa931b` (clean). Data: `lake dd9d55801115.96244a759c69 pulled 2026-11-06
(oddspapi 1x2 pinnacle+betfair-ex, 41 chunks sha f55501d987c3, fixtures sha 388eae1ad754)`.

**No lag worth taking.** After Pinnacle shortens an outcome by at least 5% in the hour before kickoff while
Betfair's back price moves by at most 1%, backing it on Betfair earns mean CLV **+0.25% net of 2% commission, 95% CI
[−0.46%, +0.97%], n 361**, one-sided p 0.24. The CI includes 0, which is the registered disconfirming result.
Rung: **L1** (clean, counted null). Gross of commission the mean is +1.69% [+0.98, +2.41]. Betfair does move after
Pinnacle, but by about as much as the commission takes. At 5% commission the mean is −1.94% [−2.67, −1.22].

## Question and rule
Does the exchange lag the sharp book? Decision time t is the first OddsPapi path timestamp with kickoff − 1 h ≤ t <
kickoff at which Pinnacle's price for an outcome is at least s shorter than 60 minutes earlier, while Betfair's back
price changed by at most b. Bet 1 unit at Betfair's back price at t, only if the stake limit in force at t is at least
25 units. Size is checked per outcome before the tie-break. At most one bet per match: the earliest t, then the
largest shortening. The benchmark is the power-de-vigged Pinnacle close (the latest of three rows within 15 minutes
of kickoff, all active, amendment 3). The metric is clv_log = log((1 + (odds − 1)(1 − c)) × p_close) with c = 0.02.
c = 0.02 is the lowest standard rate, so the headline is the favourable case. ROI is not computable: OddsPapi
carries no results.

## Data
- 8,174,211 path rows (Pinnacle and Betfair Exchange 1X2), 6,434 fixtures, all read by the lake (6,424 `ok`, 10
  `no_history`).
- 528 invalid Betfair prices (518 train, 10 eval; price ≤ 1.0, amendment 2) are ignored. Pinnacle has 0.
- Evaluation window 2026-06-01..10-19: 2,387 of 2,399 finished fixtures have both books.
- **June is nearly empty:** 6 finished fixtures (all LaLiga 2). Path rows fall to about 4k a week, and 06-22..29
  has none (off-season and the World Cup). The bets in fact span 2026-07-06..10-18, on 69 match-days in 26
  tournaments.
- Feed outages (days on which a book has no rows) are listed in `count.json`. In the evaluation window there are 2
  Pinnacle days with no kickoffs, and 1 Betfair day covering 54 kickoffs.

## Selection (train 2026-01-13..05-31 only; net of c = 0.02)
Every candidate had at least 100 train bets. The highest net mean was selected, with no tie and no fallback.

| s | b | train bets | mean CLV | 95% CI | gross |
|---|---|---|---|---|---|
| 0.02 | 0 | 1,285 | −1.10% | [−1.42, −0.80] | +0.32% |
| 0.02 | 0.01 | 1,542 | −0.82% | [−1.08, −0.55] | +0.52% |
| 0.03 | 0 | 823 | −0.66% | [−1.08, −0.22] | +0.80% |
| 0.03 | 0.01 | 1,003 | −0.32% | [−0.67, +0.03] | +1.06% |
| 0.05 | 0 | 329 | +0.50% | [−0.62, +1.48] | +2.02% |
| **0.05** | **0.01** | **387** | **+0.67%** | [−0.30, +1.54] | +2.14% |

The selected candidate's train gross (+2.14%) sat at the registration's +2% data-error line, so stale-quote and
timing checks were run on train before eval was read (`SECONDARY.md`). Dropping old own-outcome Betfair quotes did not
move the result. The best-aligned bets showed the largest effect. Betfair later shortened the bet outcome in 80% of
bets. 19 bets with no Betfair row for 15 minutes or more carried about two-thirds of the train mean. The result was
reported, not filtered: choosing a filter after seeing train buckets would be a forking path.

## Headline (evaluation, s = 0.05, b = 0.01, scored once)
Funnel: 750 outcome decisions; 308 fail the size rule; 410 bets, one per match. The close screen drops 49 (9
inactive, 40 stale), leaving 361.

| measure | n | mean CLV | 95% CI |
|---|---|---|---|
| **net of c = 0.02 (counted)** | **361** | **+0.25%** | **[−0.46, +0.97]** |
| gross of commission | 361 | +1.69% | [+0.98, +2.41] |
| net of c = 0.05 | 361 | −1.94% | [−2.67, −1.22] |
| declared secondary: Betfair recorder fresh (< 15 min) | 346 | +0.18% | [−0.59, +0.94] |
| declared secondary: no Betfair row ≥ 15 min | 15 | +1.97% | [−0.58, +4.61] |

- **Ceiling:** net +0.25% and gross +1.69% are both under +2%, so the result is not a data-error suspect and no
  audit is triggered (it claims no rung above L1).
- **Calendar thirds** (06-01..07-17, ..09-01, ..10-19): +2.70% (n 35), −0.08% (n 176), +0.07% (n 150). One of three
  is negative. The first third is thin because June is empty.
- **Leave one tournament out** (26 raw tournament ids, amendment 4): the mean ranges from +0.04% (without National
  League) to +0.41% (without the Conference League). No CI excludes 0. The pooled STANDARDS table has one group
  (OTHER, n 361) because no tournament reaches 100 bets.
- The median offered size at the taken price is 106 units (p10 33). The stake limit is the one in force at t, so a
  limit that changed without a price change may be stale (paths are change-only). The median Betfair own-quote age
  at t is 1.9 minutes; 8.9% are older than 30 minutes.
- In 8 bets the Pinnacle row that opened the decision is also the close (net −1.53%). The other 353 close on a later
  row (net +0.30%).

## Ladder
| rung | criterion | result |
|---|---|---|
| L0 | registered before scoring, chain intact | pass: registered 2026-10-28, amendments 1-4, ledger 685 rows verified |
| L1 | clean, reproducible, provenance | pass: point-in-time asserted, decisions frozen before the close, rerun reproduces, git `2aa931b` |
| L2 | CI excludes 0 in the registered direction | **fail**: [−0.46%, +0.97%]. n 361 ≥ 300, 69 blocks, 2,000 draws |
| L3+ | family BH, DSR, PBO, stability | not evaluated (L2 not cleared) |

Family `soft-open-vs-betfair` (N 4) now has all four hypotheses counted: H1 null, H2 uninformative, H3 null at the
family level and H4 null. `pitch-edge research status` lists H3 at L2, because its single-test CI excludes 0. Its
p 0.0065 misses the family level 1 − 0.975^(1/4) = 0.0063, so the family reading is null (H3 REPORT). H4's p 0.24 is
null at any level.

## Threats to validity
- **Stale quotes:** a Betfair price can look unmoved because the recorder was not seeing Betfair. The declared split
  shows the 15 bets with no Betfair row for 15 minutes or more average higher. Without them the mean is +0.18%.
- **Bettable size:** the stake limit is from the change-only path, and the quote may have been taken before a bet
  at t would land. Both would lower the realised CLV further.
- **Commission:** the headline uses the lowest standard rate. At 5% the mean is clearly negative.
- **Coverage:** OddsPapi sampled 26 tournaments, and June 2026 is nearly empty. The result is for July to October
  club football.

## Next
None in this family. A lag on the exchange this small is gone once commission is paid. A forward test would need a
venue with lower commission than 2% and bettable size checked live, which the paper loop cannot do.
