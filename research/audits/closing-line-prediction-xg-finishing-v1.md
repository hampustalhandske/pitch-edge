# Audit — closing-line-prediction-xg-finishing-v1

**Verdict: NEEDS-WORK**

Auditor: skeptic-auditor, 2026-10-20. Audit code: `research/audits/r3-xg/rederive.py`, `research/audits/r3-xg/extra.py`
(none of the study's code is imported). Outputs: `research/audits/r3-xg/out/results.json`, `out/extra.json`,
`out/rederived_power.parquet`.

I found no bug: no look-ahead, no mis-joins, the number reproduces exactly, and both placebos kill the effect.
The verdict is not SIGNED because the registration's own disconfirmer, "a single division carries it", is met
under any reading that uses uncertainty rather than sign alone. The Bundesliga supplies 57% of the summed effect
from 17% of the rows. Without it, the effect is +7.4e-6 bits [−5.3e-6, +2.1e-5], p = 0.13. The claim as written
("xG finishing luck improves prediction of the close in the top 5 leagues") is therefore not supported. What the
evidence does support is a Bundesliga effect plus, elsewhere, something that cannot be told apart from zero.

## Headline, re-derived

Independent pipeline: raw lake pulls `fd_odds.parquet` / `fd_all.parquet` / `us_team_matches.parquet` /
`club_source_map.parquet`. I wrote my own power and Shin de-vig, snapshot date, Understat ↔ football-data mapping,
feature, per-outcome OLS walk-forward and KL scoring.

| | Study (ledger) | Re-derived | Match |
|---|---|---|---|
| power Δ bits, n | +1.501e-5 [4.0e-6, 2.7e-5], 10,647 | +1.501e-5 [3.0e-6, 2.7e-5], 10,647 | exact to 4 s.f. (CI differs only by bootstrap RNG) |
| shin Δ bits, n | +1.594e-5 [5.8e-6, 2.7e-5], 10,647 | +1.594e-5 [5.0e-6, 2.7e-5], 10,647 | exact |
| power Δ R² move | +0.269 pp | +0.269 pp | exact |
| feature values | — | max abs diff 3.3e-16 on 19,646 shared rows | exact |

Study rows with the feature that are not in mine: 894. These are exactly the post-2025-07 Betfair segment, which
my universe stops before. Mine-only rows: 0.

## Checks

| # | Check | Result | Evidence |
|---|---|---|---|
| 0 | Registration predates run, config unchanged | PASS | Ledger `registrations.recorded_at` 2026-10-20 21:03:17Z; first trial 21:05:45Z. Ledger sha256 `c715267c…7f4b` = current file. Grid `devig: [power, shin]`, both run, both reported. The hand-entered `registered_at: 21:05:00Z` is 45 s before the run; the ledger's `recorded_at` is the authoritative time. The panel (incl. features) was built at 21:00:34Z, before registration, but no eval-window score exists from before 21:05:45Z (`data/research/r3/dry/` holds only pi/manager exploratory files). Exploration ran a single spec (20 matches, shrink 10, taken from the dossier's F032/33), so no forking paths are visible. |
| 1a | Look-ahead: window strictly before snapshot | PASS | The feature uses Understat UTC days < snapshot_date. Understat `date` carries a kickoff time in UTC, not midnight. Eval rows whose latest used day is ≥ snapshot: 0. |
| 1b | Match itself in its own window | PASS | 10,608 / 10,647 eval rows matched to their own Understat record (same mapped teams, ±7 d). Own record dated before snapshot: **0**. 9 rows have an fd date ≠ Understat date. In all 9 the Understat date is *later* (e.g. Lyon–Marseille fd 2022-02-01, Understat 2022-02-02), which is conservative. |
| 1c | Fixtures between snapshot and kickoff | PASS | 25 Understat fixtures fall between a snapshot and its kickoff. None are used, by construction. |
| 1d | Snapshot-time uncertainty (if "open" was earlier than assumed) | PASS | Cutoff moved to snapshot − 1 d: +1.49e-5 [2.8e-6, 2.6e-5]. Snapshot − 3 d: +1.37e-5 [2.1e-6, 2.5e-5]. Dropping the most recent match (lag 1): +1.19e-5 [2.3e-6, 2.2e-5]. Last match alone: +7.3e-6 [−2.3e-6, 1.7e-5]. The signal does not come from a match the open price had not seen. |
| 1e | Team-name joins | PASS | I built an independent mapping by fixture co-occurrence (same league, date ±1, same score). It agrees with `club_source_map` on all 168 mapped Understat teams. The 4 left unmapped by `club_source_map` (Parma 204, Coventry, Le Mans, Elversberg) are dropped. They affect both models equally, and Parma's absence is not a mis-join. |
| 2 | Survivorship / selection | PASS (minor) | Rows need both teams to have Understat history within 150 d, so promoted teams' debuts drop out. Both models are fitted and scored on the same rows. 38 eval rows (E1, I2, SP2, F2, D2: relegated-team pairs) fall outside the registered universe of 5 divisions. Registered top-5 only: power +1.463e-5 [3.1e-6, 2.5e-5], n=10,609; Shin +1.559e-5 [4.6e-6, 2.5e-5]. This is immaterial. |
| 3 | Benchmark | PASS | De-vigged Pinnacle close is the target, with power and Shin re-implemented independently. Costs: none apply (prediction of the close, no bet). |
| 4 | Multiple testing | PASS | The family has 7 hypotheses and 64 trial rows in the ledger (all metrics, both grid points). Primary p = 0.0015 (Shin) / 0.0035 (power) survives BH at q = 0.10 over the family's primary metrics. DSR and PBO are not meaningful here: the metric is predictive, not a return, and the grid has 2 points. The harness's L3 "pass" on (b) and (c) carries no information. |
| 5 | Dependence | PASS | Clustering coarser than match-day barely moves the CI. Power: by ISO week of the snapshot [3.2e-6, 2.7e-5]; by league-season (≈30 clusters) [1.7e-6, 2.9e-5]. |
| 6 | Placebo: shuffle teams within league-season | PASS | Each team's luck series is replaced by another team's from the same division-season, at the same snapshot, 100 permutations. Null mean −6.6e-7, max +1.9e-6. Actual +1.50e-5 exceeds all 100. |
| 7 | Placebo: shuffle within match-day | PASS | 100 permutations: null mean −3.2e-7, max +2.9e-6. Actual exceeds all 100. |
| 8 | Placebo: certainly stale (lagged 38 matches, ≈ one season) | PASS | −4.6e-6 [−8.9e-6, −3e-7]. The effect vanishes and turns slightly negative, which is the expected cost of extra regressors. |
| 9 | Richer baseline (adds logit p_open and its square per outcome, to both models; per-outcome intercepts already absorb a home indicator) | WEAKENS | +1.42e-5. Match-day CI [7.7e-7, 2.7e-5]; ISO-week CI [−4e-8, 2.8e-5]; league-season CI [−7.8e-7, 2.9e-5]. The point estimate survives, but the CI reaches zero under coarser clustering. |
| 10 | Is it "luck", or just recent form? | INFORMATIVE | Shrunk goal difference instead of luck: −1.2e-6 (null). Shrunk xG difference: +3.9e-6 [−2.5e-6, 1.0e-5] (n.s.). Luck added on top of a GD baseline: +1.58e-5. The signal is specifically goals *relative to* xG. Sign is as theorised: luck_home coefficient on the home-outcome move is −0.030 and luck_away is +0.041. Lucky teams drift out from open to close. |
| 11 | Stability: chronological thirds | PASS (letter) / weak | 2019-07→2021-08 +2.8e-5 [5.5e-6, 5.0e-5]; 2021-08→2023-05 **−1.2e-5** [−3.1e-5, 7.6e-6]; 2023-05→2025-05 +2.9e-5 [1.3e-5, 4.5e-5]. Negative in 1 of 3 thirds, so the registered "negative in 2 of 3" disconfirmer is not met. |
| 12 | **Stability: single division carries it** | **FAIL (CI reading)** / pass (sign-only reading) | By division: D1 +5.0e-5 [2.2e-5, 7.7e-5]; SP1 +1.5e-5 [−6.8e-6, 3.7e-5]; F1 +1.2e-5 [−1.4e-5, 3.8e-5]; E0 +6.8e-6 [−1.6e-5, 3.0e-5]; I1 −3.0e-6 [−2.7e-5, 2.1e-5]. Leave-one-out: without D1 the effect is **+7.4e-6 [−5.3e-6, +2.1e-5], p = 0.13**; removing any other division leaves it significant. D1 = 17% of rows and 57% of the summed Δ. It is not a one-season artefact: D1 is positive in 4/6 seasons, with +1.1e-4 in 2020 and +8.4e-5 in 2022. Outside D1, seasons 2021 and 2022 are negative. The report's "leave-one-league-out 6/6 positive" checks signs only. |
| 13 | Out-of-sample confirmation (post segment) | NULL | Post-2025-07 Betfair close: −1.9e-6 / −2.4e-6 [−3.3e-5, +2.6e-5], n = 894. This is underpowered rather than disconfirming, but it gives no confirmation. |
| 14 | Literature ceiling | PASS | Δ is 1.5e-5 bits against a 0.002-bit ceiling. Δ R² is 0.27 pp against a 2 pp ceiling. |

## Economic magnitude

Mean KL(Pinnacle close ‖ Pinnacle open) over the eval rows is 0.00429 bits per match. The baseline removes
0.0000323 of that. The xG feature removes a further 0.000015 bits, which is **0.35% of the open→close divergence**
(0.37% of the baseline's residual KL). Per match this equals 1.0e-5 nats of log-loss on the close distribution.
For comparison, the published range for public models against the close spans 0.035 bits. Statistically the effect
is detectable over 10k matches. Economically it is negligible on its own. It cannot pay a bookmaker margin, and the
companion strategy's ablation (LOG.md, soft-book-vs-model-close) already found that "the xG model adds ~nothing"
to that strategy.

## Why NEEDS-WORK and not SIGNED or REJECTED

- **Not REJECTED:** I found no leakage, mapping error or computational error. The effect reproduces exactly and
  survives team and match-day permutations, cutoff shifts and a lag-1 staling.
- **Not SIGNED:** the registration lists "a single division carries it" as a disconfirming result. Without the
  Bundesliga, the remaining four leagues (8,786 rows) give a CI that includes zero, and no other league is
  individually distinguishable from zero. The registration does not define "carries". The study applied a
  sign-only reading, which is the most lenient one. The evidence ladder's L3(e) asks for the effect to be "not
  carried by a single league", and I do not consider it met. The richer-baseline CI also touches zero under
  week clustering, and the only fresh out-of-sample segment is null.

## Required before re-submission

1. Restate the claim as "Bundesliga-driven; top-5 pooled +1.5e-5 bits, ex-Bundesliga +7e-6 [−5e-6, +2e-5]" in
   `research/studies/r3-features/REPORT.md` and `research/LOG.md`. Replace "6/6 LOLO positive" with the
   leave-one-out CIs.
2. Any v2 registration must define "carried by a single division" operationally, e.g. "leave-one-division-out
   CI excludes 0 for every division". It must also state in advance whether the hypothesis is Bundesliga-specific.
3. Confirmation should come from data not yet scored: the 2025/26 season onward against the Betfair close, which
   at n ≈ 900 is badly underpowered for a 1.5e-5 effect, or a forward test. Re-mining 2014–2025 does not count.
4. Re-grade with the fixed harness. DSR and PBO should be marked not applicable for this predictive metric with a
   2-point grid, not passed.
