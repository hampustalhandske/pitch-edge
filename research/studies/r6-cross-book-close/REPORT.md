# R6 — Predicting the sharp close from the full cross-book picture at the open

Family `closing-line-prediction`. Registrations (ledger 2026-10-20 22:15:15–22Z, before any evaluation score):
`research/registry/closing-line-prediction-cross-book-linear-v1.yaml` (H1), `…-cross-book-gbm-v1.yaml` (H2),
`…-book-informativeness-v1.yaml` (H3). Code: `r6lib.py`, `run.py`, `h3.py` (exploratory: `explore.py`, `trials.csv`;
post-hoc: `diagnostics.py`, `v2_coefficients.py`). Outputs: `data/research/r6/`. Ledger rows: 14 + 16 + 26.

## Question
How much better can the de-vigged Pinnacle close be predicted from every book's open quote than from R3's
consensus-gap OLS (avg/max gaps), and which books' opening deviations are information rather than staleness?

## Data, decision time, windows
`data/research/r2/audit_q3_raw.parquet` only (raw football-data per-book columns, 22 main divisions, lake pull
2026-10-20T20:43Z). 97,644 Pinnacle-close matches 2012-07 → 2025-06 plus 2,929 post-segment matches (2025-07 →
2026-10-19, Betfair Exchange close). Decision = the football-data Fri/Tue pre-closing snapshot; features read only
allow-listed open columns, and `assert_open_only` runs on every fold (recomputing with every close/result/non-open
column blanked must give identical features). Exploratory 2012-07 → 2019-06; evaluation 2019/20 → 2024/25:
n = 45,467 matches, 1,610 match-days.

## Method
- Target: logit move of the power-de-vigged Pinnacle open → close; predicted close p̂ ∝ σ(logit p_open + move).
- Baseline: R3's per-outcome OLS of the move on [1, g_avg, g_max], refit in the same walk-forward.
- Core books (≥ 90% open coverage every season 2012/13–2022/23, chosen by coverage only): B365, BW, IW, WH, VC.
- Features: each core book's de-vigged logit deviation from Pinnacle's open (clipped ±0.5; missing book imputed by the
  mean of present ones, flagged), book overrounds, Pinnacle overround, max booksum, cross-book dispersion, g_avg,
  g_max; H2 adds logit p_open.
- H1 grid: OLS; ridge (λ chosen on the last training season). H2 grid: tiny HistGradientBoosting; ridge + tiny HGB on
  the ridge residual. Walk-forward by season (fit on all prior seasons).
- Primary metric: Δ = KL(close‖base) − KL(close‖model) in bits; match-day block bootstrap (2,000); league = division
  for leave-one-out CIs. `metric_kind: probability` (DSR/PBO n/a).
- H3, per (match, outcome): slope of the Pinnacle move on the book's opening deviation, marginal and partial
  (controlling for g_avg); each book's self share (slope of its own open→close correction on its deviation). On the
  same rows Pinnacle share + self share + persistence = 1. BH across 8 books × 2 slope types; labels fixed in advance.

## Headline (evaluation, power; first run per config is the result of record)
| | Δ bits vs R3 baseline [95% CI] | ΔR² move | vs Pinnacle open | post-2025 Betfair segment (n = 2,929) |
|---|---|---|---|---|
| H1 ols | +4.52e-5 [+3.81e-5, +5.17e-5] | +0.82 pp | +8.06e-5 | +8.6e-5 [+4.8e-5, +1.31e-4] |
| H1 ridge | +4.47e-5 [+3.80e-5, +5.06e-5] | +0.80 pp | +8.01e-5 | +1.10e-4 [+7.7e-5, +1.44e-4] |
| H2 tiny | +6.02e-5 [+5.33e-5, +6.69e-5] | +1.08 pp | +9.56e-5 | +5.0e-5 [+5e-6, +8.7e-5] |
| **H2 res_tiny** | **+6.61e-5 [+5.75e-5, +7.41e-5]** | +1.17 pp | +1.01e-4 | +1.25e-4 [+9.8e-5, +1.55e-4] |

H2 vs H1 ridge, paired: res_tiny +2.14e-5 [+1.69e-5, +2.61e-5]; tiny +1.55e-5 [+1.00e-5, +2.10e-5]. Shin
sensitivity: ridge +4.56e-5, res_tiny +7.11e-5. Result log-loss vs baseline: ridge +1.04e-4 [+0.8e-5, +2.0e-4],
res_tiny +1.49e-4 [+0.2e-4, +2.8e-4].

**Scale.** Mean KL(close‖open) is 0.00528 bits: the baseline removes 0.67% of it, ridge 1.5%, res_tiny 1.9% (OOS R² of
the move 0.84% / 1.64% / 2.0%). The full cross-book picture predicts the close ≈ 3× better than R3's baseline but still
explains only ≈ 2% of the open→close divergence. Nothing here beats the close.

## Stability
| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | thirds | divisions > 0 | leave-one-division-out |
|---|---|---|---|---|---|---|---|---|---|
| ridge (e-5) | 4.0 | 5.8 | 3.0 | 4.8 | 5.6 | 3.5 | 5.0 / 3.9 / 4.5 | 21/22 (EC −1.1e-5) | est 4.06–4.87e-5, every CI lo ≥ 3.4e-5 |
| res_tiny (e-5) | 5.9 | 7.5 | 6.6 | 8.8 | 6.7 | 4.1 | 6.8 / 7.7 / 5.4 | 21/22 (SC3 −2.6e-5) | est 6.02–6.93e-5, every CI lo ≥ 5.2e-5 |

Smallest gain in the most liquid leagues (E0 ridge 1.5e-5, D1 2.5e-5), largest in second tiers and smaller top
flights (I2 12.2e-5, G1 13.2e-5, SP2 9.8e-5; res_tiny 15.4 / 20.1 / 15.7e-5).

## H3 — book informativeness (evaluation; 1XB/BF/BFE 2024/25 only)
Registered contrast VC − IW: +0.115 [+0.079, +0.149] marginal, +0.230 [+0.164, +0.292] partial (n = 100,961 rows).

| book | label (pre-registered) | marginal slope | partial (beyond consensus) [BH q] | self share | persistence |
|---|---|---|---|---|---|
| VC (BetVictor) | informative | 0.249 [0.224, 0.274] | +0.226 [0.194, 0.256] (.001) | 0.34 | 0.41 |
| BF (Betfair Sportsbook)\* | informative | 0.235 [0.176, 0.295] | +0.208 [0.133, 0.283] (.001) | 0.28 | 0.49 |
| BFE (Betfair Exchange open)\* | informative | 0.195 [0.134, 0.250] | +0.170 [0.108, 0.224] (.001) | 0.71 | 0.09 |
| B365 | informative | 0.134 [0.116, 0.152] | +0.058 [0.034, 0.082] (.001) | 0.46 | 0.40 |
| WH (William Hill) | informative, borderline | 0.106 [0.084, 0.128] | +0.025 [−0.001, 0.052] (.038) | 0.29 | 0.60 |
| IW (Interwetten) | stale | 0.136 [0.106, 0.169] | −0.009 [−0.050, 0.032] (.74) | 0.36 | 0.51 |
| 1XB (1xBet)\* | stale | 0.132 [0.063, 0.197] | −0.038 [−0.148, 0.073] (.81) | 0.50 | 0.37 |
| BW (bwin) | stale (anti-informative) | −0.001 [−0.022, 0.020] | −0.260 [−0.292, −0.227] | 0.68 | 0.32 |

\* 2024/25 only (≈ 22k rows, 264 match-days). VC is informative every season (partial 0.13–0.29); B365's partial ≈ 0
except 2021 and 2024; IW decays (+0.09, +0.19 in 2019–20, then negative); WH's label is not robust (2012–2025 joint
coefficient −0.04 to −0.07, largest persistence); BW is the clearest stale book (closes 68% of its own deviation;
Pinnacle moves away from it given the consensus). Drop-one-book ablations agree: removing VC costs ridge 1.7e-5 /
res_tiny 1.6e-5; BW 0.9e-5 / 1.1e-5 (used to discount the consensus); WH, B365, IW ≤ 0.4e-5 each.

## Ladder (`pitch-edge research grade`; predictive metrics, DSR/PBO n/a)
| rung | H1 (ols) | H2 (res_tiny) | H3 (partial) |
|---|---|---|---|
| L0 registered before trials, unedited, chain intact | pass | pass | pass |
| L1 clean, archived code, reproduced | pass | pass | pass |
| L2 CI > 0, n ≥ 300, 2,000 draws, ≥ 30 blocks | pass | pass | pass |
| L3a BH across family (m = 24) | q 0.0013 | q 0.0013 | q 0.0013 |
| L3d within ceiling | pass | pass | pass |
| L3e 3/3 thirds; every LOO-division CI excludes 0 | pass, 22/22 | pass, 22/22 | pass, 22/22 |
| L4 auditor / edgelab | pending | pending | pending |

All three reach **L3 on the harness** — predictive/descriptive findings, not strategies; nothing is called an edge.

## Post-hoc diagnostics (not registered, not counted)
1. Evaluation > exploratory (ridge +4.5e-5 vs +1.6e-5) is the baseline weakening, not the model: the model's gain over
   the open is stable (ridge 7.2e-5 → 8.0e-5; res_tiny 7.7e-5 → 1.01e-4), while the baseline's fell 5.6e-5 → 3.5e-5
   (R² 1.7% → 0.84%) when football-data's Avg/Max changed source in 2019/20. Single-book columns did not change source.
2. Placebo (book block permuted within match-day): ridge still +8.4e-6 [+1.5e-6, +1.5e-5], res_tiny +8.9e-6 — ≈ 8e-6
   of Δ is re-weighting g_avg/g_max against the weak baseline; the book-information part is ≈ 3.6e-5 (ridge) and
   5.7e-5 (res_tiny).
3. Horizon: same-day snapshot rows (n = 7,833) res_tiny −3.4e-6 [−1.7e-5, +1.0e-5]; other rows +8.1e-5 [+7.1e-5,
   +9.0e-5] — the information exists 1–3 days out, not on match day.
4. Rows where a core book quotes above the same-row Max (n = 1,347): +7.6e-5 [−0.3e-5, +1.5e-4] — no sign the gain
   comes from mis-timed quotes.
5. Post segment positive for all four configs but confounded by a broken baseline (Max excludes Pinnacle on 27% of
   2025/26 rows; IW/VC/WH absent) — not interpretable as confirmation.

## Threats to validity
- Simultaneity is not observable (no per-book timestamps): if VC's or Betfair's "open" were captured after Pinnacle's,
  "informative" would mean look-ahead. VC supplies most of H1/H2. Diagnostics 3–4 argue against it but don't prove it.
- The baseline is weak after 2019: read the "vs Pinnacle open" column for the cross-book gain.
- H2's two settings were the best two of five on exploratory data (disclosed in the registration).
- 1XB, BF, BFE rest on one season; BFE drops 1,141 rows (5%) as quote errors.
- A dry run on a scratch ledger printed exploratory numbers after the YAMLs were written and before registration;
  nothing was changed. Typed `registered_at` 22:12Z; ledger 22:15Z (authoritative). Dirty tree; code archived by hash.

## Implications for the forward test `bookmaker-line-structure-live-soft-vs-sharp-v1`
- **onexbet: stale** (11 of the first 20 decisions) — prices above Pinnacle fair there keep their CLV if taken before
  1xBet moves; consistent with the audit's 1xBet +2.07%, but one season.
- **betvictor, betfair_sb_uk: informative** — Pinnacle closes ≈ 21–23% of their deviation beyond the consensus;
  expect ≈ a quarter of the apparent edge to vanish; these are the bets most likely to be wrong-footed.
- **williamhill:** deviations persist to the close — CLV vs close is mechanical; a data suspect in the R2-Q3 audit.
- **Betfair Exchange open:** noisy sharp fallback (self share 0.71); prefer Pinnacle as the fair reference.
- Most forward books (unibet, betsson, winamax, tipico, …) have no history here.
- Information is a 1–3-day phenomenon; match-day soft deviations carried none (untested live).
- A forward v2 only as a paired test: `DRAFT-bookmaker-line-structure-live-soft-vs-sharp-v2.yaml` (unregistered) —
  fair value = Pinnacle + frozen coefficients on live-observable books (betvictor, betfair_sb_uk, onexbet, williamhill)
  and the odds-api consensus, 20–76 h before kickoff; arms pinnacle vs r6_live_model, paired CLV
  (`v2_coefficients.py` → `data/research/r6/v2_live_coefficients.json`).

## Audit (2026-10-21): H1, H2, H3 SIGNED with limits
`research/audits/closing-line-prediction-r6.md`. Independent re-derivation matches (H1 OLS +4.516e-5, per-match
diff 1e-15; H3 slopes to 4 d.p.); no look-ahead. Capture-timing threat rejected with a powered bracket test: VC's
open sits above the aggregator Max (captured with Pinnacle's open) on only 0.02–0.09% of rows, flat across later-move
bins — a capture 5% of the way to its close would give ≈ 7.3% on lengthening outcomes. The horizon pattern runs the
opposite way to a capture lag. Limits that travel with every number:
- **Quote era-refit gains**: against R3's baseline refit per data era (Avg/Max source change 2019/20), H1 ridge is
  **+3.48e-5 [2.89, 4.09]e-5** and H2 res_tiny **+5.62e-5 [4.83, 6.43]e-5**; against that baseline every placebo is
  ≤ 0 or n.s. VC alone ≈ half the gain (+1.60e-5; without VC +1.78e-5).
- **2024/25 ≈ 0** vs the era baseline (ridge −3e-7, res_tiny +5.6e-6) — the season football-data has no VC/IW quotes.
- Book labels: BW is a suppressor — **stale, not "anti-informative"** (never a signal to bet against bwin); IW's label
  depends on the control (+0.071 [0.032, 0.107] against the other-named-books consensus); Betfair Exchange open timing
  cannot be verified; onexbet "stale" is weak (one season, partial CI [−0.16, +0.07]); williamhill not safe.
For the forward test: betvictor (safe; data to 2023/24) and bet365 (weakly informative) are usable labels;
betfair_sb_uk provisional; onexbet, Betfair Exchange, williamhill not safe to interpret.

## Next
Pre-registered era-aware baseline for future closing-line studies; refit the R3 baseline with per-era coefficients; register the v2 forward test after
review.
