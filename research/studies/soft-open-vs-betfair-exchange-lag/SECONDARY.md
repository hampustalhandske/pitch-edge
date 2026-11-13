# H4 declared secondary and train-window data checks

Written and committed after `--select-only` and before `--score`, so before any evaluation-window outcome or close
is read. This is not an amendment: the registration (`soft-open-vs-betfair-exchange-lag-v1`, amendments 1-4) and its
selection stand unchanged, and the headline is scored and reported exactly as registered.

Pull of record: `lake dd9d55801115.96244a759c69 pulled 2026-11-06 (oddspapi 1x2 pinnacle+betfair-ex, 41 chunks sha
f55501d987c3, fixtures sha 388eae1ad754)`. Selection (train window 2026-01-13..05-31 only): s = 0.05, b = 0.01. It had
the highest net mean, with no tie and no fallback, and every candidate was eligible.

## Declared secondary (uncounted, reported beside the headline, never in its place)

**Betfair recorder freshness at the decision time.** For each scored evaluation bet, the recorder age is the
decision time t minus the `observed_at` of the most recent valid Betfair Exchange row of any outcome of that
fixture at or before t (`live_ts`). This is the same row the registered liveness rule uses, and that rule allows up
to 60 minutes. The evaluation bets are split into recorder age under 15 minutes and 15 minutes or more. For each
part the report gives n, the mean net clv_log with a match-day block-bootstrap 95% CI, and the gross mean. This
split is not a counted test, adds no trial to the family, and does not change the headline or its rung.

Why: on the train window, 19 of 387 scored bets (5%) had recorder age of 15 minutes or more. They averaged net
+0.098 and carried about two-thirds of the train mean. A Betfair price that looks unmoved because the recorder was
not seeing Betfair is a stale-quote risk. Applying the filter only now, after seeing train results by bucket, would
choose the analysis from the outcomes, so it is reported, not applied.

## Train-window checks (2026-01-13..05-31, selected s = 0.05, b = 0.01; evaluation untouched)

All: n 387, net +0.0067 [-0.0030, +0.0154], gross +0.0214. The gross mean is at the registration's +2% data-error
line, which led to these checks:

- **Own-quote age.** The Betfair own-outcome quote age at t (time since that price last changed; paths are
  change-only) has a median of 3.5 min and p90 of 54 min. Dropping bets with a quote older than X minutes gives
  net +0.0042 (X = 1, n 142), +0.0074 (5, n 212), +0.0074 (15, n 291) and +0.0064 (30, n 331). Stale own quotes do
  not drive the result.
- **Timestamp skew.** Pinnacle row minus Betfair row used at t has a median of +3.2 min. Bets with |skew| <= 1 min
  give net +0.0154 [+0.0027, +0.0298] (n 100): the best-aligned bets show the largest effect, which argues against a
  misalignment artefact.
- **Follow-through.** Betfair's next row for the bet outcome after t was shorter than the taken price for 80% of
  the 355 bets that have one (median -1.6%). Bets Betfair followed: net +0.0125 [+0.0038, +0.0198] (n 284). Bets it
  did not follow: -0.0290 (n 71). That is the pattern of a real lag.
- **Recorder freshness.** Under 15 min: n 368, net +0.0020 [-0.0063, +0.0095], gross +0.0167. 15 min or more: n 19,
  net +0.0982. With the same filter on every candidate, the selection is unchanged (s = 0.05, b = 0.01 at +0.0020;
  next s = 0.05, b = 0 at -0.0007).

Verdict on train: no artefact in timestamp alignment or own-quote staleness. The train effect is small with a CI
that includes 0, and a stale-recorder tail inflates it.

## Also stated in the report

- The +2% ceiling check on evaluation, gross and net. Above +2% the result is a data-error suspect and goes to audit.
- If the evaluation n after the close screen is under 300, the result is underpowered, as registered.
- Evaluation coverage is thin in June 2026: path rows fall to about 4k a week, and 2026-06-22..29 has none, most
  likely the World Cup break. The outage days are listed in `count.json`.

Checks code: `data/research/h4_train_diagnostics.py` (git-ignored, train chunks only).
