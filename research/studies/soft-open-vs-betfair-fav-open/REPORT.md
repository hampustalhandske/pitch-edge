# H1 — Strong favourites at the best soft open against the Betfair close

Registration: `research/registry/soft-open-vs-betfair-fav-open-v1.yaml` (family `soft-open-vs-betfair`, family
N 4). Code: `fav_open.py`, `run.py`, shared `../soft-open-vs-betfair-common/sobf.py`. Counted trial: one, on the
evaluation window. Data: lake build `9ecd619e6916`, football-data main-league files, pulled 2026-10-28.

**Result: null at the family's level, and underpowered.** Mean CLV +0.77% on 125 bets, 95% CI [−0.03%, +1.60%],
one-sided p 0.029. The family-wise level at N 4 is 1 − 0.975^(1/4) = 0.0063, so the primary test fails, and it
would also fail at N 1 (0.025). n is below the registered `min_n` of 300. Rung: L1 (clean, counted null).

## Question
Do soft books under-shorten strong favourites at the open? The rule backs the favourite at the best soft open price
when the Betfair Exchange open makes it at least `p_min` likely and the price beats the Betfair-open fair price by
more than θ. It is scored by CLV against the de-vigged Betfair Exchange close.

## Method
- Decision at football-data's opening snapshot. The fair price is the power-de-vigged Betfair Exchange open, and
  the favourite is its most likely outcome. The price taken is `max_<outcome>_open`, the best price over the books
  football-data tracks. That is an upper bound no single venue offers (E2 P5), so this result is pitch-edge-only.
- edge = log(price × p_fav). One bet per match when p_fav ≥ p_min and edge > θ; flat stakes.
- A match is eligible when all six opens and all three Betfair closes are valid. A favourite quote with
  max / Betfair open > 1.5 is a data error and the match is dropped.
- CLV = log(price × p_close), with p_close the power-de-vigged Betfair Exchange close. The CI is a match-day block
  bootstrap with 2,000 draws.
- Windows: exploratory 2024-08-01 to 2025-06-30, evaluation 2025-07-01 to 2026-10-19, minus 2026-04-17 to 05-04,
  where football-data has no Betfair close (PE-26; 597 rows).

## Selection (exploratory window only)
The registered rule takes the highest mean CLV with at least 100 bets; ties go to the lower θ, then the lower p_min.

| θ | p_min | bets | mean CLV | 95% CI |
|---|---|---|---|---|
| 0 | 0.65 | 423 | +0.54% | [+0.08, +1.02] |
| 0 | 0.70 | 286 | +0.46% | [−0.12, +1.00] |
| 0 | 0.75 | 161 | +0.77% | [+0.04, +1.48] |
| **0.01** | **0.65** | **202** | **+0.86%** | [+0.06, +1.68] |
| 0.01 | 0.70 | 132 | +0.68% | [−0.37, +1.64] |
| 0.01 | 0.75 | 74 | +0.95% | ineligible (< 100 bets) |

## Headline (evaluation, θ 0.01, p_min 0.65)
| n | match-days | mean CLV | 95% CI | one-sided p | flat ROI |
|---|---|---|---|---|---|
| 125 | 83 | +0.77% | [−0.03, +1.60] | 0.029 | −0.35% |

- 8,431 matches were eligible and the rule bet on 1.5% of them.
- Stability: every calendar third is positive (+0.95%, +0.68%, +0.54%). No division has 100 bets, so all are
  pooled as OTHER and leave-one-division-out says nothing.
- The result is under the +3% literature ceiling.

## Dropped matches (examined after scoring)
- **260 data-error drops.** Every one is a broken Betfair Exchange open book, not a bad soft price and not an
  unmapped team. The favourite's open is 1.01-1.10 and the book sum has a median of 1.99, e.g. 1.04 / 1.03 / 1.04.
  All 260 have a valid close.
  - They concentrate in thin leagues: SC2 9.6% of its matches, G1 8.7%, SC3 8.2%, EC 5.7%, B1 5.6%.
  - The drop uses open data only, so it does not depend on the move to the close. 130 of them would have passed
    the rule's p and edge tests, because a broken reference inflates the edge.
- **124 missing quotes.** 115 have an invalid open and 9 have a valid open but no Betfair close. None of those 9
  would have been a bet, so missing closes do not bias the CLV.
- **The ratio screen misses broken books that pass it.** 35 of the 125 bets have a Betfair open book sum above
  1.10. Their mean CLV is −0.84%, against +1.39% on the 90 bets with a book sum of at most 1.10.
  - This split was found after scoring. It is descriptive, not a test, and it changes nothing above.
  - `research/STANDARDS.md` now requires every later betting family to screen the reference book (Betfair book sum
    in [1.00, 1.05], set from the train-window close distribution, not from this split). 35 of H1's 125 bets have a
    sum above 1.10, and 43 lie outside [1.00, 1.05], so the band would drop 34% of them.
  - An H1 v2 is not registered: the evaluation window is now seen, so a v2 would need forward data.

## Ladder
| rung | status |
|---|---|
| L0 registered | yes, before any outcome was read |
| L1 clean | yes: decisions read the whitelisted open columns only (asserted); the rerun reproduces; git sha, snapshot and counts recorded |
| L2 signal | **no**: the CI includes 0, n 125 is below 300, and it fails the family level |

## Threats to validity
- **The price is an upper bound.** The max over books is not one venue's quote, and books limit winners.
- **The evaluation window was reused.** R2-Q2, R2-Q3, R3 and R6 already scored rules on it, so this is weaker
  evidence than a fresh window. The paused paper loop is the clean test.
- **The reference is noisy.** The Betfair open is a thin book in many divisions (see above).
- **There is no placebo for H1.**

## Next test
None registered. A favourite rule with a reference-book screen would be a new trial, scored on forward paper bets
once the loop runs again.
