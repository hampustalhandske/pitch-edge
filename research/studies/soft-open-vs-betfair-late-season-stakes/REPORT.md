# H2 — End-of-season stakes: is the side that still needs points under-priced at the open?

Registration: `research/registry/soft-open-vs-betfair-late-season-stakes-v1.yaml` (family `soft-open-vs-betfair`,
family N 4), plus a ledger amendment recorded 2026-10-28 11:51:45Z before any selection (see Method). Code:
`late_stakes.py`, `run.py`, shared `../soft-open-vs-betfair-common/sobf.py`. Counted trial: one, on the
evaluation window. Data: lake build `e3800576ec61`, football-data main-league files, pulled 2026-10-28.

**Result: uninformative.** The evaluation window gave 5 bets. Mean CLV −7.5%, 95% CI [−12.7%, +6.8%], one-sided
p 0.97. That says nothing either way. Rung: L1 (clean, counted, underpowered).

The 300-400 test bets the proposal expected were a planning error, explained below. The counted trial stays in the
family N.

## Question
In the last five rounds, is the side that still needs points (title, Europe or relegation) under-priced at the best
soft open against a side with nothing at stake? It is measured by CLV against the de-vigged sharp close.

## Method
- **Stakes.** Each team's best and worst possible final rank comes from a point-in-time table, tested against
  fixed zones: rank 1, top 4, top 6 and the bottom 3. Relegation takes precedence where zones overlap, and a rival
  who can only draw level counts toward the worst rank. A side has something at stake when a zone is still open
  to it. A bet is placed only when exactly one side has something at stake: back that side at `max_<side>_open`
  when log(price × p_ref_open) > θ. One bet per match, flat stakes.
- **Reference and benchmark.** Train uses the power-de-vigged Pinnacle open and close. Evaluation uses the Betfair
  Exchange open and close.
- **Amendment, recorded on the ledger before any selection or score.** Only the outcome-blind train count had run.
  - A division-season counts only when every team has exactly 2(T − 1) matches with goals. This excludes every
    Scottish split season, 7 Greek play-off seasons and 2019/20.
  - Standings use only results dated strictly before the decision day: the latest Tuesday or Friday on or before
    the match date, which is football-data's opening snapshot. A first draft cut at the match date, which would
    have let a Sunday match see Saturday's results. That was fixed before any outcome was read.
  - "Last 5 rounds" means 1 to 5 league matches left for the team. The season comes from the lake's `season` column.
- **Note on the ledger row.** The trial's `pit_assertion` text still says "before the match date". The code that
  ran uses the decision-day cutoff (`late_stakes.stakes_context` calls `decision_day`), and the amendment records
  that rule. The text is corrected in `run.py` for any later run.

## Outcome-blind train count (2012-07 to 2024-06; opens and earlier results only)
| θ | bets | without the completeness filter |
|---|---|---|
| 0 | 1,153 | 1,228 |
| **0.02** | **378** | 416 |
| 0.05 | 75 (below the 300 floor) | 91 |

These come from 9,787 last-5-round matches, 2,533 of them with stakes on one side only. Selection then took the
highest mean CLV with at least 300 train bets: θ 0.02, +3.69% [+2.65, +4.77] against the Pinnacle close. θ 0 gave
+1.53% [+0.89, +2.19].

## Evaluation (θ 0.02)
| | last-5 matches | in the PE-26 gap | asymmetric | bets before the gap | bets scored |
|---|---|---|---|---|---|
| with the completeness filter (counted) | 735 | 391 | 209 | 9 | **5** |
| without it (sensitivity) | 956 | 418 | 241 | 11 | 7 |

Counted result: mean CLV −7.5%, CI [−12.7, +6.8], p 0.97, 3 match-days. The sensitivity's 7 bets are just as
uninformative, so no CLV is given for them.

## Why so few bets (planning error)
- **One run-in.** The evaluation window holds a single season's run-in (2025/26, mid-April to May). The proposal's
  estimate of 13% of all window matches would only fit several seasons.
- **The PE-26 gap.** football-data has no Betfair close for 2026-04-17 to 05-04, which removes 391 of the 735
  last-5-round matches.
- **A different reference.** At θ 0.02 against the Betfair-open fair price, 4% of asymmetric matches qualify (9 of
  209). On train, against the Pinnacle open, it was 16% (378 of 2,385). The θ chosen against Pinnacle does not carry
  over to Betfair.

## Ladder
| rung | status |
|---|---|
| L0 registered | yes, before any outcome was read (amendment chained before selection) |
| L1 clean | yes: opens whitelisted (asserted); standings cut at the decision day (tested); the rerun reproduces |
| L2 signal | **no**: 5 bets |

## Threats to validity
- **Underpowered by design.** See above. The registration's test window could not supply the bets it expected.
- **Max-open price.** It is an upper bound no single venue offers (E2 P5). The train CLV of +3.7% is above the +3%
  literature ceiling and is largely line shopping.
- **Uniform zones.** The same fixed zones apply across tiers. Real zones vary by league and season (play-offs, a
  split, European places by cup winners).

## Next test
A rule on season run-ins needs several seasons of a sharp close that has no gap, or forward paper bets over the
2026/27 run-in (April to May 2027) with the loop running.
