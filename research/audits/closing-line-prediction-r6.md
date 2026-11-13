# Audit: R6 cross-book close prediction (H1, H2, H3)

Hypotheses: `closing-line-prediction-cross-book-linear-v1` (H1), `closing-line-prediction-cross-book-gbm-v1` (H2),
`closing-line-prediction-book-informativeness-v1` (H3). Study: `research/studies/r6-cross-book-close/REPORT.md`.

| id | verdict |
|---|---|
| H1 cross-book-linear-v1 | **SIGNED**, with the scope note on the baseline below |
| H2 cross-book-gbm-v1 | **SIGNED**, with the same scope note |
| H3 book-informativeness-v1 | **SIGNED** for the registered primary (VC − IW contrast). The per-book labels are only partly safe: two are wrong as worded or depend on the control used (see §H3) |

Auditor: skeptic-auditor, 2026-10-21. Audit code is in `research/audits/r6/`: `alib.py`, `rederive.py`, `timing.py`,
`power.py`, `placebo.py`, `placebo2.py`, `era_h2.py`, `extra.py`, `h3check.py`, `loo.py`, `misc.py`. None of it
imports the study's code or `pitch_edge`. It has its own power de-vig (Newton on the exponent), its own universe,
features, OLS, ridge and λ rule, scoring and block bootstrap. Its only input is
`data/research/r2/audit_q3_raw.parquet`. JSON outputs are in `research/audits/r6/out/` and per-match frames in
`data/research/r6_audit/`. Ledger and `src/` were read only.

I found no bug. There is no look-ahead in the features. The capture-timing threat that the study itself named fails
every test I could build, and the one that matters has demonstrated power. The headline numbers reproduce exactly
for H1 and H3, and to 3 s.f. for H2. Placebos vanish against a fairly fitted baseline. Two qualifications go with the
signature and must travel with the numbers:

1. **About a fifth of Δ comes from the registered baseline being weak, not from book information.** Δ is measured
   against R3's OLS. That OLS is pooled across the 2019/20 change of the Avg/Max source. Refit it per era and H1 falls
   from +4.5e-5 to +3.5e-5, and H2 res_tiny from +6.6e-5 to +5.6e-5. The registered claim ("beats the R3 baseline")
   is true as stated. The cross-book information claim should quote the era-refit numbers.
2. **The gain is zero in 2024/25 against the era-refit baseline** (H1 ridge −3e-7, res_tiny +5.6e-6, hgb_tiny −2.0e-5).
   That is the season football-data stops carrying VC and IW. The full-window claim survives, but the football-data
   model has no demonstrated value once VC is gone.

## 0. Registration and configuration

| check | result | evidence |
|---|---|---|
| Registered before first trial | PASS | Ledger registration times are 22:15:15Z (H1), 22:15:18Z (H2) and 22:15:22Z (H3). First trials are 22:16:03Z, 22:20:40Z and 22:27:13Z. File mtimes are 00:11:06 CEST, i.e. 22:11:06Z. |
| File unchanged | PASS | Current sha256 values `d8919e9c…`, `8c5e91d6…`, `f63bd962…` equal the ledger's. |
| Reported config = registered | PASS | Grids {ols, ridge}, {tiny, res_tiny} and {marginal, partial} are all run and all reported. The HGB hyper-parameters in `r6lib.HGB_CONFIGS["tiny"]` match the YAML. The ridge λ grid and its last-training-season rule match. |
| Forking paths | disclosed | H2's two settings were the best 2 of 5 on exploratory data (`trials.csv`), which the registration states. The evaluation window was scored once per config. A dry run used exploratory seasons only (`R6_DRY_LEDGER`, eval = 2017-18). |
| Multiple testing | PASS | Family `closing-line-prediction` has 10 hypotheses and 24 configs (`research status`). Primary p = 0.0005 (bootstrap floor), so BH q = 0.0013. DSR/PBO are n/a (probability metrics), not a pass. |

## 1. Re-derived headline

| | study (ledger) | re-derived | match |
|---|---|---|---|
| H1 ols Δ bits, n | +4.516e-5 [3.81, 5.17]e-5, 45,467 | +4.516e-5 [3.86, 5.20]e-5, 45,467 | exact. Per-match max abs diff in KL is 1.2e-15 and the row sets are identical (0 only-mine, 0 only-study). |
| H1 ridge | +4.467e-5 [3.80, 5.06]e-5 | +4.467e-5 [3.84, 5.12]e-5 | exact (same λ path; per-match diff 1.2e-15) |
| H2 res_tiny | +6.608e-5 [5.75, 7.41]e-5 | +6.615e-5 [5.82, 7.45]e-5 | 3 s.f. HGB split thresholds react to 1e-15 feature differences, so the per-match max diff is 4e-4 bits. |
| H3 contrast VC − IW | marginal 0.1152, partial 0.2303 | 0.1152, 0.2303 | exact |
| H3 per-book slopes (8 books × 2) | REPORT table | identical to 4 d.p. (VC 0.2492/0.2259, BW −0.0009/−0.2597, …) | exact |

The CIs differ only through the bootstrap RNG. The CI is robust to coarser clustering: by ISO week, H1 ols
[3.74, 5.27]e-5 and res_tiny [5.69, 7.51]e-5; by division-season (≈130 clusters), H1 ols [3.26, 5.79]e-5 and
res_tiny [5.18, 8.15]e-5.

## 2. Look-ahead and leakage in the features

| check | result | evidence |
|---|---|---|
| Close / result columns reachable | PASS | `OPEN_COLUMNS` contains only `{PS, B365, BW, IW, WH, VC, 1XB, BF, BFE, BbAv, BbMx, Avg, Max}{H,D,A}`. No `…C…`, `ft_result`, AH or O/U column is in it. `_OpenOnly` raises on any other read. `assert_open_only` blanks every non-open column and requires identical features, on every test fold. My features, built independently from the registration text, match the study's to 1e-15. |
| Imputation across rows or seasons | PASS | Missing books are imputed with the same row's mean of the present books (else clipped g_avg). Overround centring uses training-fold means only. Standardisation uses training-fold statistics only. |
| λ / model selection on test | PASS | λ is chosen on the last *training* season. HGB has no early stopping. Train seasons are strictly before the test season (asserted). |
| Duplicates / mis-joins | PASS | 0 duplicate fixture keys in the window. Decision and target come from the same row, so there is no join to slip. |
| Kickoff before the snapshot | PASS (immaterial) | 438 eval matches kick off on the snapshot day at or before 16:00 UK: Δ −3.1e-6. The 63 with kickoff before 13:00 give Δ +4.4e-5 (n too small to matter). Neither group carries the result. |
| Universe selection | PASS | Rows need a PS close and a result. PSC coverage is 100% in every eval season. Model and baseline are scored on identical rows. |

## 3. Snapshot timing (VC / Betfair captured later than Pinnacle?)

Football-data has no per-book timestamps. Two observational designs share the same moment structure. In one, VC is
captured at the same time as Pinnacle and holds information Pinnacle prices later. In the other, VC is captured later,
so its deviation contains part of Pinnacle's own later move. They give the same slope, self share and persistence.
"Does VC predict the direction better than its own correction implies?" therefore cannot tell them apart:
Pinnacle share + self share + persistence = 1 under both. So I used quantities that behave differently under the two
designs.

**T1, aggregator bracket (decisive for relative capture time).** The R2-Q3 audit showed the aggregator Max is captured
with Pinnacle's open. Re-checked here on 2019-24: P(Max == PS_open) is 14-22% in every bin of the subsequent
Pinnacle move, and P(Max < PS_open) is ≤ 0.14% throughout. If VC's open column were captured after the aggregator,
two things would follow wherever the price later lengthened: VC_open > Max_open would appear, and VC's frequent ties
with Max (VC == Max on 3-7% of outcomes) would break upward. Observed:

| subsequent Pinnacle odds move | n | observed P(VC > Max) | if VC captured 5% of the way to its close | 20% | 50% | at its close |
|---|---|---|---|---|---|---|
| > +10% | 13,621 | **0.09%** | 7.3% | 19.9% | 59% | 84% |
| +5 to +10% | 14,984 | 0.06% | (>+5%: 4.9%) | (12.1%) | 19% | 51% |
| −0.5 to +0.5% | 6,765 | 0.04% | | | 0.7% | 2.9% |

P(VC > Max) is flat at 0.02-0.09% across all nine move bins (2012-18: about 0.6%, also flat). Even a capture 5% of
the way toward VC's close would show up roughly 80 times over. BF (2024/25) is also bracketed: observed 0.05-0.4%, flat.
Under a 5% lag it would be 13% on >+10% moves. 1XB and B365 are bracketed too. So VC, BF, 1XB and B365 are captured
in the same snapshot as the aggregator, and therefore as Pinnacle's open. **BFE cannot be tested this way.** The
exchange back price is above the aggregator Max on 34-48% of rows (it is not bracketed), and that share rises from
0.34 to 0.48 as the price later lengthens.

**T2, horizon.** If capture were late by a fixed clock lag after the Fri/Tue snapshot, the captured move would be
largest where Pinnacle moves fastest, i.e. same-day games a few hours before kickoff. If instead the capture were late
by an amount growing with the gap, the bracket test above would already fail. Observed is the reverse of the fixed-lag
prediction:

| snapshot → kickoff (2019-24, UK kickoff times) | VC partial slope | cov(move, dev_VC) | ridge Δ vs era baseline |
|---|---|---|---|
| same day (< 8 h) | 0.126 | Fri 5.1e-4 / Tue 5.3e-4 | +9.7e-6 [−0.7e-6, +2.0e-5] |
| next day | 0.226 | Sat 8.1e-4 | +3.4e-5 [2.5, 4.2]e-5 |
| 2 days | 0.239 | Sun 13.1e-4 | +4.9e-5 [3.8, 6.2]e-5 |
| 3 days (Mon) | 0.396 | Mon 12.6e-4 | +6.3e-5 [3.6, 9.0]e-5 |

The exploratory era replicates this: VC partial on Friday games is 0.005, against 0.15 on Saturday and Sunday. The
information is a 1-3-day phenomenon and is weakest exactly where a timing artefact would be strongest.

**T3, Pinnacle-is-best subsets.** VC partial is 0.266 [0.196, 0.336] where PS_open == Max, 0.253 where Pinnacle is the
best named price, and 0.216 where it is not. VC's informativeness does not depend on Pinnacle's quote being the
stale-looking top price.

**T4, Betfair pair (2024/25).** Fitted jointly, BF and BFE stay separately informative: BF 0.19 [0.12, 0.27], BFE 0.16
[0.10, 0.21]. With B365 and 1XB added, BF is 0.21 and BFE 0.15. Their deviations residualised on g_avg correlate only
0.09. A common Betfair capture time would load on a shared component, and there is little of one.

Conclusion: VC's (and BF's) informativeness is not a relative capture-time artefact at any scale these data can
resolve. What cannot be excluded is source-side refresh staleness *inside* one snapshot, e.g. a Pinnacle quote the
source had not refreshed. That too would be strongest on same-day games, and it is weakest there.

## 4. Placebos and the baseline

| run (auditor pipeline) | vs registered R3 baseline | vs era-refit baseline |
|---|---|---|
| H1 ols, real | +4.52e-5 [3.86, 5.20]e-5 | **+3.53e-5 [2.89, 4.18]e-5** |
| H1 ridge, real | +4.47e-5 [3.84, 5.12]e-5 | **+3.48e-5 [2.89, 4.09]e-5** |
| H2 res_tiny (study frame) | +6.61e-5 | **+5.62e-5 [4.83, 6.43]e-5** |
| H2 tiny (study frame) | +6.02e-5 | +5.03e-5 [4.34, 5.72]e-5 |
| P1 test block permuted within match-day, ridge (30 draws) | mean +4.3e-6, sd 1.8e-6, range [−0.5, +8.2]e-6 | mean **−5.6e-6**, range [−10.4, −1.7]e-6 |
| P1 ols (50 draws) | mean −4.3e-6, range [−9.0, +2.7]e-6 | −1.75e-5 (one draw) |
| P2 permuted in train *and* test | ols +2.1e-6 [−4.3, +8.1]e-6; ridge +7.6e-6 [+2.1, +12.6]e-6 | ols −7.8e-6; ridge −2.3e-6 [−7.1, +2.1]e-6 |
| P3 certainly stale (same home team's block ≥ 300 d earlier) | ols +2.8e-6 n.s.; ridge +1.2e-5 [5.7e-6, 1.8e-5] | ols −7.1e-6; ridge +2.1e-6 [−4.0, +8.0]e-6 |
| No individual books at all (ridge on baseline + overrounds) | +5.0e-6 [1.7, 8.5]e-6 | −4.8e-6 |

The era-refit baseline is per-outcome OLS on [1, g_avg, g_max], fitted only on seasons with the same Avg/Max source
(≤ 2018/19 Betbrain, ≥ 2019/20 football-data). 2019/20 has no same-era history and keeps the pooled fit.

The study's placebo (+8.4e-6 for ridge) sits at the top of my 30-draw null against the R3 baseline. Its explanation
is confirmed: ridge's shrinkage of g_avg/g_max improves on a pooled OLS whose coefficients are stale after 2019/20.
Every placebo is ≤ 0 or n.s. against the era-refit baseline. Leave-one-division-out against the era baseline: the
lowest CI lower bound is 2.5e-5 (ridge, without I2) and 4.2e-5 (res_tiny). Thirds are 3.4/5.3/1.7e-5 (ridge) and
5.1/9.2/2.5e-5 (res_tiny). All are positive, but the last third is clearly weaker.

By season against the era baseline (×1e-5), ridge: 4.0, 2.1, 9.1, 2.7, 3.1, **−0.03**. res_tiny: 5.9, 3.8, 12.7, 6.6,
4.2, **0.6**. The 2021 spike reflects an era baseline fitted on only two seasons. The 2024/25 zero is the season
without VC/IW.

Book ablations (ridge vs era baseline): all books +3.48e-5; without VC +1.78e-5; VC alone +1.60e-5; no books −4.8e-6.
VC is the largest single contributor at about half the gain. "Most of the gain" overstates it slightly.

Effect size is 6.6e-5 bits against a registered ceiling of 1e-3 and a STANDARDS ceiling of +0.02 bits vs the close.
That is far inside both, and nothing here beats the close.

## H3: book informativeness

The registered primary (VC − IW > 0) reproduces exactly and is robust. VC's partial is 0.21-0.25 under every control
I tried (g_avg; a leave-one-out consensus of the other named books; both; all five named books jointly, ± g_avg). IW
is −0.01 to +0.07 under the same controls. The contrast is positive in all three thirds (0.06 / 0.34 / 0.26).

**BW "anti-informative" −0.26 is a suppressor coefficient, not evidence that Pinnacle moves away from bwin.** It is
also not only the mechanical "BW is inside g_avg" artefact:

- BW's marginal slope is 0.00 [−0.02, +0.02]. Pinnacle neither moves toward nor away from BW.
- BW's partial stays strongly negative without g_avg in the regression: −0.20 [−0.23, −0.17] against the leave-one-out
  named consensus, and −0.20 in the joint named-books regression. With logit-level and outcome controls added it is
  −0.26.
- BW's deviation correlates 0.55 with the other named books' mean deviation. Its effective weight in g_avg is high:
  the regression weight of dev_BW in g_avg is 0.16, and the implied weight −slope/coef(g_avg) is 0.28-0.46 by season,
  consistent with bwin-feed clones in the aggregator.

So BW carries none of the information but shares the uninformative common part of the soft-book deviations.
Conditional on the others, it therefore gets a negative weight that subtracts that part. The registered label "stale"
is correct by the registered rule (self share 0.68 vs Pinnacle share ≈ 0). The words "anti-informative" and "Pinnacle
moves away from it given the consensus" should not be read as a fade signal, and they should be dropped. Its ablation
value (discounting the consensus) is real.

Other labels depend on the control:

- IW: partial vs g_avg is −0.009 (labelled stale), but vs the leave-one-out named consensus it is +0.071
  [0.032, 0.107].
- WH: +0.025 to +0.040 depending on the control, borderline.
- B365: +0.06 to +0.08 under every control, robust.
- 1XB, BF, BFE: one season each (2024/25, 264 match-days).

## Is the book ranking safe for interpreting the forward test (`bookmaker-line-structure-live-soft-vs-sharp-v1`)?

Only in part.

| book | use in forward interpretation |
|---|---|
| betvictor (VC) | **Safe.** Informative in every season, under every control and at every horizon > same-day, and capture-time synchronous by the bracket test. The football-data evidence stops in 2023/24. |
| betfair_sb_uk (BF) | **Provisional.** Bracket-synchronous and informative, but one season only. |
| bet365 | Safe as "weakly informative" (partial ≈ 0.06). |
| onexbet (1XB) | **Weak.** One season, partial CI [−0.16, +0.07]. "Stale" rests on self share 0.50 vs Pinnacle 0.13. Do not lean on "1xBet CLV is real staleness" beyond that. |
| Betfair Exchange open (BFE) | **Not safe.** Capture timing cannot be verified (not bracketed by Max), self share 0.71. |
| williamhill (WH) | **Not safe.** The label depends on the control; high persistence makes CLV mechanical, as the report says. |
| bwin (BW) | Not a forward book. Do not use its negative partial as a signal. |

Two further limits on transfer:

- Every coefficient describes football-data's Fri/Tue snapshot 1-3 days out. Same-day deviations carried nothing
  (VC partial 0.13, model Δ ≈ 0). Live odds-api quotes taken at other horizons are a different object.
- With VC absent (2024/25), the cross-book model's gain over a fairly fitted baseline is zero. The v2 forward draft's
  value therefore rests on the live betvictor feed.

## Required before anything downstream quotes these results

1. Quote the cross-book information gain against the era-refit baseline: H1 +3.5e-5 [2.9, 4.1]e-5 and H2 res_tiny
   +5.6e-5 [4.8, 6.4]e-5. State that it is ≈ 0 in 2024/25.
2. Remove "anti-informative" for BW. Report IW's label as control-dependent.
3. Treat BFE's informativeness as timing-unverified.
