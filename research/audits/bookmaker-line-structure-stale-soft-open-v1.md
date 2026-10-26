# Audit — bookmaker-line-structure-stale-soft-open-v1 (R2-Q3)

Auditor: skeptic-auditor, 2026-10-20. Audit code: `research/audits/r2-q3/` (no import from the study or `src/`).
Raw data: `fd_matches.odds_all` pulled from the lake 2026-10-20T20:43Z for the 22 main (`mmz4281/`) divisions,
2012-07 onward → `data/research/r2/audit_q3_raw.parquet` (109,582 rows). Run: `arch -arm64 uv run python <script>.py`
inside `research/audits/r2-q3/`, in the order pull → rederive → composition → setter → leadlag → simultaneity →
sensitivity → stability → sameday → book_quality.

## Verdicts

| Claim | Verdict |
|---|---|
| (a) Max-odds upper bound as a descriptive market-structure fact: "the football-data market-max open quote beats the power-de-vigged Pinnacle-open fair price by > 2% on ~25% of matches, and those quotes beat the de-vigged Pinnacle close by +2.08% [1.91, 2.26] on average (n = 11,172, 2019-07 → 2025-06)" | **SIGNED**, as a description of *quoted* prices only. Not an edge or a strategy: executability is untested and cannot be tested from this data (see the placebo). |
| (b) Achievable single-book strategy. The registered one is bet365_open. | **REJECTED**: θ = 0.02 gives +0.46% [−0.06, +0.97], fails L2, and decays by season from +1.52% to −1.31%. Unregistered single books found during the audit: **NEEDS-WORK**. They are hypotheses for a v2 registration, not evidence (William Hill is a data suspect; 1xBet is the only clean candidate, on one season). |

## 1. Registration and provenance

| Check | Evidence | Result |
|---|---|---|
| Registration predates the run | `registered_at` 18:00:00Z; file mtime 18:00:10Z; snapshot pulled 17:56:52Z; ledger `run_at` 18:07:26Z | pass |
| Registration unedited | sha256 of the yaml now = `a08f456b…2fc32` = `registration_sha256` in all 20 ledger rows | pass |
| Reported config = registered config | 6-point grid (max/bet365 × θ 0/0.02/0.05), cap 1.5, power de-vig at open and close, windows as registered. Sensitivities (no cap, cap 1.25, 6 post-segment trials) are registered and in the ledger | pass |
| Pre-registration look at the data | The registration quotes eligible counts and says the 1.5 cap was set "after seeing that some max-odds rows imply an overround as low as 0.55". Those are price-only observations, with no outcomes or closes | acceptable |
| Code provenance | `run.py` and `r2-common/*.py` were modified at 18:11:17Z, *after* the counted run (18:07Z), and `results.json` was rewritten at 18:11:54Z (presumably a dry-ledger rerun, `R2_DRY_LEDGER_DIR`). git sha is `-dirty` and no code hash is recorded, so the code on disk is not provably what produced the ledger rows. Its rerun output equals the ledger (n = 11,172, 0.020845), and my independent code reproduces it. | minor L1 weakness, no effect on the number |

## 2. Independent re-derivation (from raw `odds_all`: BbMxH/D/A before 2019/20, MaxH/D/A after; PSH/D/A; PSCH/D/A)

All 12 grid points reproduce to the stated precision (match-day block bootstrap, 2,000 draws, my own seed):

| window | source | θ | n | CLV (audit) | report |
|---|---|---|---|---|---|
| eval | max | 0.02 | **11,172** | **+2.08% [+1.91, +2.26]** | +2.08% [1.91, 2.26] |
| eval | max | 0 / 0.05 | 29,512 / 2,104 | +0.97% / +3.18% | +0.97% / +3.18% |
| eval | bet365 | 0.02 | 1,994 | +0.46% [−0.06, +0.97] | +0.46% [−0.06, 0.95] |
| expl | max | 0 / 0.02 / 0.05 | 40,402 / 18,836 / 4,386 | +1.33 / +2.65 / +5.23% | same |
| post (BFE ref + bench) | max | 0.02 | 856 | +0.88% [+0.15, +1.57] | +0.88% [0.13, 1.60] |

Stability also reproduces: thirds +2.15 / +2.28 / +1.63%; 22/22 divisions positive (1.0% to 4.0%); leave-one-division-out
2.04 to 2.13%; no cap +2.12%; cap 1.25 +2.09%. Eligibility drops 217 of 45,684 evaluation matches (0.5%), spread across
divisions, with a result mix close to the kept rows. Missingness cannot drive the effect.

## 3. Hunt

**What "max" is.** In the evaluation era (`Max*`), Max is below Pinnacle's open quote on 0.1% of outcomes and below
bet365's on ≈ 0%. Max equals Pinnacle's open quote on 16% of outcomes and exceeds every individually named book on
61-72% of outcomes (median 0.9-1.3% above the best named book). So Max is a superset aggregator that includes
Pinnacle, and 73% of the bets are set by an unnamed book. **Max excludes Betfair Exchange**: in 2024-25 Max < BFE on
39% of outcomes, and Max's median overround (1.022) is higher than BFE's gross (1.020). A small exchange quoted gross
would pull it below that, so the max is not an exchange back price. A commission haircut is shown for completeness
anyway: 2% → +0.77%, 5% → −1.23%. The early Betbrain era is less clean: Max < PS on 8.4% of outcomes in 2012-13 and
2.5% in 2014-15. This affects the exploratory window only.

**Look-ahead / snapshot simultaneity.** Three tests, none supporting "max captured after Pinnacle's open":
1. If the aggregator (which contains Pinnacle) were captured later, P(Max == PS_open) would fall, and P(Max < PS_open)
   would rise, on outcomes where Pinnacle later shortened. Observed: P(Max == PS_open) is 14-21% in every bin of the
   subsequent Pinnacle move. It is *highest* (21.5%) where Pinnacle shortened by > 10%, and P(Max < PS_open) is flat at
   ≈ 0.1%. The aggregator carries the same Pinnacle quote as the open column.
2. Direction match at equal open edge: Pinnacle moves toward the taken price on 53-65% of max bets but 63-70% of bet365
   bets. A late-captured max would be the more predictive of the two, and it is the less predictive.
3. CLV by size of Pinnacle's open→close move (backed side): +3.64% in the smallest-move quintile, falling to −2.55% in
   the largest. The CLV is preserved open edge where Pinnacle did not move. It is not a captured move.
   Late capture would push CLV toward the soft margin (negative), not inflate it.
   Same-day games (Tue/Fri) with kickoff ≤ 15:00 UK, where the snapshot could post-date kickoff: n = 69, CLV +1.75%,
   so not inflated. Pinnacle open == close on 0.13% of bets.
No `*_close` or result column enters the decision. Decision and benchmark come from the same row, so there is no
date/timezone join to slip.

**Benchmark sensitivity (max, θ = 0.02).** De-vig of the close: power +2.08%, Shin +2.20% [2.02, 2.37], multiplicative
+2.72% [2.54, 2.89]. Rule re-run with Shin at open and close: +2.05%; multiplicative: +2.71%. The result is robust to
de-vig choice, and power is the most conservative of the three.

**Multiple testing.** The family ledger holds 322 trials (Q1 254, Q2 24, Q3 20, Q4 24). The headline has bootstrap
z ≈ 23 (SE ≈ 0.09 pp), so it survives BH trivially. DSR: per-bet SR = 0.21 on 11,172 bets against an expected null maximum of
≈ 0.03 for 322 trials, so DSR ≈ 1. The report's "cross-trial variance → DSR ≈ 0" is an artefact of pooling
heterogeneous metrics (OLS slopes, draw bias, CLVs) into one SR variance. The audit's own single-book scan adds
≈ 48 unregistered looks (8 books × 3 θ × 2 windows); see below.

**Dependence.** One bet per match; the CI resamples match-days. Adequate.

**Literature ceiling.** +2.08% sits at the STANDARDS +2% leakage-suspect line. The registration anticipated this for
max odds (Kaunitz et al. 2017). After the checks above I find no leakage. The size is explained by price dispersion
across ~50 books, not by information.

## 4. Placebos

| Placebo | Result | Reading |
|---|---|---|
| Max premium over Pinnacle open (log Max/PS per outcome) shuffled across matches within match-day, 20 permutations | CLV **+4.93%**, edge at open 5.35%, Pinnacle move toward price −0.42% | The CLV does **not** vanish. It rises, because CLV against a near-martingale benchmark rewards *any* quote above fair. Only the information component vanishes (real move −1.81% [−1.99, −1.64] vs −0.42%). |
| Premium taken from the mirrored outcome (H↔A) | CLV +5.16%, move −0.67% | same |
| bet365 open as the "fair" reference instead of Pinnacle open | **−1.06% [−1.23, −0.90]**, n = 18,108 | The edge vanishes. The effect needs the sharp reference. |

Consequence: the headline CLV cannot prove anything about information or achievability. Given that a quote above fair
existed, positive CLV is close to mechanical. The real data is *worse* than the placebo because Pinnacle moves about
half-way toward the max (Pinnacle's Friday/Tuesday quote is noisy and the market consensus leads it). That supports
the descriptive claim (a), and it shows the claim's value rests entirely on whether those quotes were executable,
which this dataset cannot show.

The report's "stale soft lines" interpretation is only partly right. It holds for the part set by Interwetten/Bwin
(Pinnacle barely moves toward them). For bet365-set maxima Pinnacle moves ~80% of the way toward the soft price,
i.e. bet365 deviations are informative, not stale.

## 5. Achievable single books (audit-side, unregistered; evaluation window, θ = 0.02, power close)

| book | n | CLV | Shin | by season | data check |
|---|---|---|---|---|---|
| bet365 (registered) | 1,994 | +0.46% [−0.06, +0.97] | +0.82% | +1.52 → +0.63 → −0.20 → +1.16 → −0.23 → −1.31 | decaying; **fails L2** |
| William Hill | 1,024 | +2.59% [+1.70, +3.47] | +2.74% | 2.5 / 0.7 / 5.1 / 3.2 / 1.2 / 1.4 | **50% of bets are WH quotes *above* the same-snapshot aggregator Max**. Excluding them: −0.45% [−1.70, +0.80]. Data suspect. |
| Bwin | 416 | +2.48% [+1.46, +3.49] | +2.46% | 2.8 / 4.8 / 2.1 / 2.4 / 0.9 / 1.0 | 19% above Max; excluding them: +1.26% [+0.16, +2.35]; decaying |
| Interwetten | 534 | +1.82% [+0.93, +2.67] | +1.58% | no 2024-25 data | excluding above-Max quotes: +1.30% [+0.36, +2.20] |
| 1xBet | 483 | +2.07% [+1.23, +2.87] | +2.21% | 2024-25 only | clean (0.4% above Max); cap 1.10: +2.81% |
| VC Bet | 802 | +0.25% [−0.74, +1.28] | +0.75% | — | null |
| Betfair Sportsbook | 186 | −2.12% [−5.32, +0.70] | — | 2024-25 only | null/negative |
| best of named books | 4,472 | +1.79% [+1.44, +2.14] | +2.01% | 1.8 / 1.0 / 2.6 / 2.1 / 1.4 / 1.7 | inherits the WH issue |

None of these is registered. Each comes from a ≈ 48-look scan, and several sit above the registration's own +2%
single-book data-suspect line. Stake limits and account restrictions at these books are not modelled. They cannot
carry a rung.

## 6. What deserves a forward paper test

A **v2 registration** (new id, family `bookmaker-line-structure`) of: "at decision time, back the best quote among
books the paper ledger can actually observe live (the-odds-api soft books, e.g. 1xBet, Bwin, Interwetten, bet365,
William Hill) when it beats the power-de-vigged sharp price at the same timestamp by > 2%". Scored by CLV vs the
de-vigged Betfair Exchange close (Pinnacle close where available). Add a registered quote-sanity rule: the quote must be
≤ the contemporaneous market max and ≤ 1.10 × the sharp price. The forward ledger timestamps each quote, which settles
the one question history cannot: was the price really there. Report frequency too: max-triggered bets fall from
~4,340 per third to 2,491 in the last third, and the aggregator max's overround rose from 1.008 to 1.022 (2019 → 2025),
so the opportunity is shrinking. The max_open version must never be shown as an edge in `ask` or the dashboard.

## Rejection reasons (exact)

- (b) bet365_open, the only registered achievable source: θ = 0.02 CI [−0.06%, +0.97%] includes 0 (registered
  disconfirming result), and it is negative in 3 of the last 4 seasons.
