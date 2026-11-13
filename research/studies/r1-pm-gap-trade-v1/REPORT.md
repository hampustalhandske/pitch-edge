# r1-pm-gap-trade-v1 — Buy Polymarket outcomes priced away from the sharp open (paper)

**Question.** At kickoff−T (T ∈ {24, 6, 2}h), buy a Polymarket (PM) outcome when its all-in cost is below the
de-vigged sharp open by more than θ (θ ∈ {0.02, 0.04, 0.06}) — Yes side — or buy the complement when PM is too
high — No side. Does it beat the de-vigged Betfair Exchange close after fees and spread? Registration:
`research/registry/r1-pm-gap-trade-v1.yaml`. Ledger: `trials.csv` (18 evaluation trials).

## Data, point in time
- Universe and kickoff rule as `r1-pm-accuracy-v1`, main football-data leagues only (only they carry open odds).
- Sharp open = Pinnacle open, else Betfair open (booksum ≤ 1.05), used only when `open_known_at ≤ T`, where
  `open_known_at` = the latest Tue/Fri 18:00 Europe/London before kickoff (football-data notes: odds "collected
  Friday afternoons … Tuesday afternoons for midweek games"). Check on the exploratory window (no outcomes): PM is
  closest to the open 3–6h *before* that bound (summed abs distance 0.0225 at −3h, 0.0248 at 0h, 0.0298 at +12h),
  so the bound is conservative.
- Cost per share = last trade (≤ 6h old) + max(half bar high−low, 0.005) + fee 0.05·p·(1−p).
- Code asserts bar `known_at ≤ T` and `open_known_at ≤ T`.
- **Data gap (PE-26).** football-data published no Betfair Exchange close for 2026-04-20 to 2026-05-03 and for about
  half the matches of 2026-04-17, 04-19 and 05-04. In the evaluation window, 132 of the 167 matches dated 2026-04-17
  to 05-04 have no Betfair close, so their trades have no CLV against it and are not in the headline (counted on
  `data/research/r1/fd.parquet`, before the study's other filters). The numbers are the recorded run; they exclude
  those matches.

## Headline (evaluation 2025-11-01 → 2026-10-19; CLV = log(fair close / cost) vs de-vigged Betfair close)
| T | θ | side | n | CLV vs BF close [95% CI] | CLV vs PM last | ROI |
|---|---|---|---|---|---|---|
| 2h | 0.02 | yes | 75 | −3.69% [−5.01, −2.41] | −4.3% | −27% |
| 2h | 0.02 | no | 46 | −3.46% [−5.09, −1.92] | −2.6% | −41% |
| 6h | 0.02 | yes | 48 | −3.17% [−4.99, −1.21] | −4.2% | −35% |
| 6h | 0.02 | no | 40 | −2.53% [−4.20, −0.85] | −1.9% | −35% |
| all θ ≥ 0.04, all T = 24h | | | 0–9 each | uninformative | | |

- No trial reaches the registered n ≥ 300 (max 75): PM sits within ≈ 1–2 pp of the sharp open net of costs.
- T−24h is almost empty (≤ 2 decisions per trial): the open is known 24h ahead for only about half the fixtures
  and only 47% of matches have fresh PM trades on all outcomes that early.
- PBO (CSCV, 16 blocks, 18 configurations) = 0.63; DSR ≈ 0 for every trial with n > 2 (SR0 = 0.46).
- Exploratory: every θ = 0.02 configuration negative too (−2.2% to −3.2%).

## Why it loses
In `r1-pm-convergence-v1` the sharp line moves toward PM between open and close (slope of Betfair close − open on
the PM−open gap +0.84 at T−6h, +0.91 at T−2h). The football-data "open" is a Friday/Tuesday snapshot, stale by
T−6h (median summed open–close distance 0.040). A PM price "below the open" mostly reflects news PM has already
priced. Spread and fee are 3–6% of the cost per share.

## Ladder
| Rung | Criterion | Result |
|---|---|---|
| L0 | registered before scoring | pass |
| L1 | PIT asserted; rerun reproduces ledger exactly | pass (code uncommitted at sha 6d29de4) |
| L2 | positive CLV, CI excludes 0, n ≥ 300 | fail — every informative trial negative; max n = 75 |
| L3 | BH / DSR / PBO | fail (no BH rejections; DSR ≈ 0; PBO 0.63) |

Rung **L1, null to negative**.

## Ledger note
Exploratory rows include one design run where the YAML 1.1 parser read `side: [yes, no]` as booleans, so both
sides were scored as "no" (configs `"side": true/false`). Code now maps booleans back to side names; the
registration was not edited; evaluation rows come only from the fixed code.

## Next
A fresher fair value at T is needed — live Betfair/Pinnacle via the-odds-api (P1 forward ledger), not
football-data's Friday snapshot.
