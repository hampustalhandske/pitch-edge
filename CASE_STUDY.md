# Case study — can public data beat the closing line? (First full run, 2026-09-06)

**Question.** Using only free, ToS-clean data, do our models add information beyond the bookmaker's
closing price — and does any of the "alternative" data (referee tendencies, kickoff weather,
travel/fatigue, Elo, rolling form) earn its place?

**Setup.** 41,939 matches from 11 European divisions (E0, E1, D1, SP1, I1, F1, N1, P1, B1, SC0, T1),
July 2015 → September 2026, from the open Club-Football-Match-Data spine. Walk-forward: train on
everything before each 45-day fold, predict the fold, roll (90 folds, 40,894 out-of-sample predictions
per model). Isotonic calibration fitted only on realised past folds. Bets priced at the market-average
price with a 3 % edge threshold; ¼-Kelly, ½-Kelly and flat 1 % staking, hard per-bet caps.

**A caveat that matters.** football-data.co.uk (which carries Pinnacle *early* and *closing* prices) was
returning HTTP 503 for the whole run, so the only price available was the market-average **closing**
price. That makes two things true by construction: CLV is exactly zero, and any bet is placed *against
the closing line* — the sharpest number in the market. The results below should be read as "how far are
we from the closing line", not as a P&L claim.

## Result 1 — every model is less informative than the closing price

| model | log-loss | market log-loss | Δ (bits) |
|---|---|---|---|
| Dixon-Coles (per league, 2-yr decay) | 1.0084 | 0.9716 | −0.053 |
| GBDT (feature store, no market input) | 0.9975 | 0.9716 | −0.037 |
| GBDT + early market probs | 0.9885 | 0.9716 | −0.024 |

Multiclass Brier follows the same order (0.596 / 0.594 / 0.587 vs 0.578). Betting anything with a "3 %
edge" against a price that is better than the model produces the expected outcome: −5 % to −8 % ROI on
24-30k bets, bankroll exhausted under every staking rule. That is the honest baseline, and it is what a
recruiter at a trading desk *should* expect from a public-data model marked against the close.

The interesting part is the gap ordering: adding the market's own (early) probability as a feature
closes ~40 % of the gap, which says the remaining information is mostly *in the price*, not in our
features.

## Result 2 — feature-group ablation (GBDT, ¼-Kelly, 90-day retrain)

| group removed | log-loss without | Δ log-loss | verdict |
|---|---|---|---|
| Elo (ours + CFMD at kickoff) | 1.0167 | **+0.0188** | the only group that clearly matters |
| travel / fatigue | 0.9980 | +0.0001 | noise |
| rolling form (5/10-match) | 0.9979 | −0.0000 | redundant given Elo |
| weather at kickoff | 0.9968 | **−0.0011** | slightly *hurts* out of sample |
| referee home-bias | 0.9979 | 0.0000 | no effect — the spine has no referee data |

So: of the three "real" alternative-data features, weather is mildly harmful (over-fitting a weak
signal), travel is neutral, and referee bias could not be tested at all because the fallback spine lacks
referee names. That last point is exactly why the football-data.co.uk spine matters — it has the referee
column for the English divisions.

## Result 3 — where is the model *closest* to the market?

`reports/main/by_league.csv` (regenerated every run) breaks Result 1 down by division for the
market-aware GBDT:

| division | n | model − market (bits) |
|---|---|---|
| D1 Bundesliga | 3,285 | −0.011 |
| T1 Süper Lig | 3,585 | −0.012 |
| E1 Championship | 5,961 | −0.017 |
| N1 Eredivisie | 3,235 | −0.020 |
| SP1 La Liga | 4,121 | −0.024 |
| P1 Primeira Liga | 3,328 | −0.024 |
| B1 Pro League | 2,985 | −0.030 |
| E0 Premier League | 4,100 | −0.030 |
| F1 Ligue 1 | 3,765 | −0.032 |
| SC0 Premiership | 2,384 | −0.035 |
| I1 Serie A | 4,109 | −0.039 |

The gap is smallest in the Bundesliga, the Turkish Süper Lig and the Championship and largest in Serie A,
Scotland and the Premier League. That is **not** a clean "under-covered markets are softer" pattern —
the Bundesliga is as well-covered as any league — so the information-asymmetry thesis is *not supported*
by this run. The honest reading is that these differences (0.01–0.04 bits on 2–6k matches) are mostly
about how predictable each league is, not about market softness, and the thesis needs the developing-
market divisions (ARG, BRA, MEX, JPN, Scandinavia — loaded, not yet in the backtest set) and an early
price to be tested properly.

## Result 4 — the developing-market slice (the thesis test proper)

Same protocol on 51,481 matches from the 16 under-covered divisions (ARG, BRA, MEX, JAP, USA, NOR, SWE,
DEN, POL, ROM, RUS, CHN, IRL, FIN, AUT, SUI), 2013 → 2024, 60-day retrain, 50,309 out-of-sample predictions:

| model | log-loss | market log-loss | Δ (bits) |
|---|---|---|---|
| Dixon-Coles | 1.0518 | 1.0044 | −0.068 |
| GBDT | 1.0401 | 1.0044 | −0.052 |
| GBDT + early market | 1.0213 | 1.0044 | −0.024 |

The market itself is less certain here (log-loss 1.004 vs 0.972 in Europe — these leagues are harder to
predict), but our models are **further** behind it, not closer. Per division the gap is smallest in Mexico
and MLS (−0.03 bits) and largest in Norway, Romania and Austria (−0.07 to −0.08). So the naive version of the
thesis — "public models do better against soft-market prices" — is rejected on this data. What the thesis
actually needs is information the price doesn't have yet (lineups, injuries, local news), which is what the
Transfermarkt, Swedish-feed and API-Football layers are for; the aggregate spine alone doesn't provide it.

## Result 5 — the deep-learning entries (PyTorch Lightning)

GRU and Transformer sequence encoders on the European slice from 2018 (29,240 predictions, 90-day retrain,
time-ordered validation split, early stopping, temperature scaling):

| model | log-loss | market log-loss | Δ (bits) |
|---|---|---|---|
| gru_sequence | 1.0418 | 0.9743 | −0.097 |
| transformer_sequence | 1.0364 | 0.9743 | −0.090 |
| (GBDT on the same era) | ≈ 0.998 | 0.9743 | ≈ −0.034 |

Both deep models are calibrated and sane, both lose to gradient boosting on aggregate features, and attention
edges recurrence by a hair. Ten matches of team-centric history is simply thin evidence next to Elo and
long-window form. Reported as such; they stay in the comparison because "we tried the deep models and they
lost to trees" is more useful than a deleted row.

*Method note.* The first Lightning run produced log-loss 1.15 — worse than uniform. Cause: temperature
scaling fitted on 64-row validation slices in the early walk-forward folds picked a sharpening temperature
(T≈0.5), making already-uncertain predictions over-confident. Fix: temperature ≥ 1 only, fitted only when the
validation slice has ≥ 300 rows; early stopping only with ≥ 200 validation rows; lower learning rate. The
bug and the fix are in `MODEL_CARDS.md` because that is exactly the kind of thing a reviewer should see.

## Result 6 — does the RAG layer actually retrieve the right document?

Hybrid retrieval (Chroma dense ∪ BM25, reciprocal-rank fusion, cross-encoder rerank) over 10,541 documents,
scored on 80 synthetic questions generated from the corpus itself ("Who won X vs Y on <date>?", "What does
the gbdt model say about A against B?", news headlines): **hit@1 0.99, hit@5 1.00, MRR 0.99**
(`reports/rag_eval.csv`, `pitch-edge rag-eval`). The questions are templated, so this is an upper bound
on real usage; it does establish that the answer layer is grounded in the document it should be, and the
citation verifier flags any figure in an answer that does not appear in a retrieved source.

## Result 7 — pre-registered: does attention-velocity / rotation load / referee actually help? (Phase 5)

**Pre-registration (written before the run, 2026-09-06 21:00 UTC).** Three new pre-match signals were
added to the feature store, each with a hypothesis and a fixed test, so the result cannot be tuned after
the fact. Test: the same walk-forward GBDT ablation as Result 2 (11 European divisions, 2015-07 →, 90-day
retrain, ¼-Kelly), Δ log-loss when the group is removed; a group "earns its place" if removing it worsens
log-loss by ≥ 0.001 (the size of the weather effect in Result 2, the smallest we could distinguish there).

| group | features | hypothesis H1 | what we expect if H0 |
|---|---|---|---|
| `wiki_attention` | `pv_home_anom`, `pv_away_anom`, `pv_home_z`, `pv_away_z`, `pv_diff` — log-ratio and robust z of a club's English-Wikipedia article views on D-1 vs a 28-day trailing baseline (D-2 and earlier); match-day views never used | attention spikes carry news (injury, managerial, transfer) that Elo/form do not; **removing the group worsens log-loss by ≥ 0.001** | club-page attention is dominated by the fixture itself and by results already in Elo → Δ ≈ 0 |
| `rotation_load` | `rot_*_minutes_7d` (squad minutes per starter-equivalent in all competitions, trailing 7 days), `rot_*_days_since_any`, `rot_*_midweek_cup` (non-league fixture 2–4 days before), `rot_load_diff` — from the open Transfermarkt extract | midweek cup/European load the spine cannot see predicts weekend under-performance; **Δ ≥ 0.001** | rest days from league fixtures already capture most of it (Result 2 found travel/fatigue ≈ 0) → Δ ≈ 0 |
| `referee` (re-test) | `ref_cards_per_game`, `ref_home_bias` — now active because Transfermarkt games supply the referee name for ~70 k matches | referee tendency shifts 1X2 probabilities enough to register; **Δ ≥ 0.001** | the effect is on cards/fouls, not on who wins → Δ ≈ 0 (the literature suggests this) |
| referee-announcement lag (event study) | pre → post announcement no-vig move, strong-tendency (\|z\| ≥ 1) vs neutral (\|z\| ≤ 0.5) referees, Welch one-sided t, α = 0.05 | strong-tendency announcements move the price more | no difference — or, more likely this early, **insufficient paired announcements** (the poller started 2026-09-06 and the Premier League page is JavaScript-rendered) |
| cross-venue lead-lag | lagged cross-correlation + Granger F-test between Polymarket and Kalshi on the same fixture/outcome, hourly grid, ≥ 24 aligned steps | one venue leads by a stable 1–2 h | **insufficient**: the warehouse held 1–2 snapshots per market when this was written; the hourly scheduler now accumulates history |

Results are appended below exactly as produced by `scripts/stage_e.py`, whichever way they go.

### Result 7 — outcome (run 2026-09-06 22:02–22:10 UTC, `scripts/stage_e.py`, `reports/ablation_phase5.csv`)

Feature store rebuilt with the context joins: referee names now present for **43.7 %** of the 41,939 matches
(Transfermarkt), rotation load for **54.9 %** (clubs that map to a Transfermarkt id), attention anomalies for
**99.1 %** (340 club articles, 1.34 M daily rows). Same 90-day-retrain walk-forward GBDT as Result 2.

| group removed | log-loss full (all groups) | log-loss without | Δ log-loss | verdict against the pre-registered criterion |
|---|---|---|---|---|
| `referee` (now active) | 0.9999 | 0.9994 | −0.0005 | noise (\|Δ\| < 0.001) — H0; referee tendency does not move 1X2 |
| `travel_fatigue` | 0.9999 | 0.9990 | −0.0009 | noise, leaning harmful — H0 (unchanged from Result 2) |
| `wiki_attention` | 0.9999 | 0.9979 | **−0.0019** | **rejected**: removing it *improves* out-of-sample log-loss |
| `rotation_load` | 0.9999 | 0.9979 | **−0.0020** | **rejected**: removing it *improves* out-of-sample log-loss |

Reading. Both headline hypotheses are rejected, and not marginally: the "full" model with every new group
(log-loss 0.9999) is *worse* than the Result 1 GBDT without them (0.9975), and dropping either new group
recovers most of that. The market benchmark on the same rows is 0.9713. So:

* **Attention velocity** at club-article level carries no information the tree can use beyond Elo and form —
  most pre-match attention *is* the fixture and the last result, and the tail of genuine spikes (a sacking, a
  transfer, an injury breaking in local press) is too rare to learn from 42 k rows without over-fitting the
  rest. A player-level version (the injured striker's page, not the club's) is the obvious next test and is
  not built here.
* **Rotation load** from cup/European minutes is either already in the price and in rest-days, or too noisy
  when 45 % of matches have no mapping. It does *not* rescue the fatigue thesis.
* **Referee** is finally testable and is a no-op for match outcome, which is what the literature predicts (the
  effect is on cards and fouls). The tendency table is still useful descriptively — the dossier quotes it.

Decision (pre-registered rule applied, not a post-hoc choice): the two rejected groups stay in the feature
store and in every future ablation, but the default `gbdt` model excludes them (`DEFAULT_EXCLUDED_PREFIXES` in
`models/gbdt.py`, recorded in every model card). The headline number does **not** improve as a result —
it stays at −0.037 bits (gbdt) / −0.024 bits (gbdt + early market) — it simply does not get worse.

**Referee-announcement event study** (`reports/referee_lag.csv`): 275 referees with ≥ 20 matches of tendency
data, **0 paired announcements** → `insufficient`. The Premier League appointments page is JavaScript-rendered
(the parser finds nothing in the HTML), no other federation source is configured yet, and the poller only
started today. The test exists and is unit-tested on synthetic pairs; it will run for real once a parseable
source and a few weeks of hourly snapshots exist. Nothing is filled in.

**Cross-venue lead-lag** (`reports/lead_lag.csv`): **0 shared fixture-outcomes with ≥ 24 aligned hourly steps**
→ `insufficient`. At the time of the run the warehouse held one Kalshi pull and two Polymarket pulls, and the
Polymarket rows were US-politics markets returned under the "soccer" tag (now filtered out). The hourly
scheduler job snapshots both venues; the method (lagged cross-correlation + Granger F-test in both directions)
is verified on synthetic series where one venue leads by one step.

Net: Phase 5 added three real signals and two real tests and **moved the headline result by zero** — which is
exactly what the pre-registration was for.

## What changes the answer

1. **Early vs closing prices** (football-data.co.uk Pinnacle `PS*`/`PSC*`): the only way to measure CLV
   and to bet at a price the model could actually have taken.
2. **Player-level information**: the open Transfermarkt extract (1.9 M appearances, 631 k substitutions,
   valuations) and StatsBomb events give squad quality, absence and rotation signals the aggregate
   spine cannot see.
3. **Under-covered markets with local text**: Swedish Allsvenskan/Superettan/Ettan results plus SVT,
   Sportbladet, Expressen, DN, GP feeds — the sentiment-velocity scanner now runs in Swedish too.
4. **Cross-venue prices**: Polymarket and Kalshi snapshots (457 football series on Kalshi) for divergence
   detection against bookmaker no-vig probabilities.

All four are wired in; the next full `pitch-edge refresh` re-runs this study with them. Losing results
will keep being published here — that is the point of the document.
