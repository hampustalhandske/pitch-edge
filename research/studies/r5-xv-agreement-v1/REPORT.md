# r5-xv-agreement-v1 — How far apart are Kalshi and Polymarket?

**Question.** For the same 1X2 outcome and hour, how far apart are Kalshi and Polymarket, and how often does the gap
exceed the round-trip cost? Registration: `research/registry/r5-xv-agreement-v1.yaml` (family `cross-venue`).

## Data and n
Verified Kalshi↔Polymarket universe (`research/studies/r5-common/mapping.py`: 49.5% of closed Kalshi GAME events map
and pass kickoff verification — 70–82% in top leagues, ≈ 0 for friendlies/USL; 100% outcome-level resolution
agreement on 20,974 outcomes, diagnostic only). Exploratory 2025-08-01 → 2025-12-31 (1,680 events, not counted);
evaluation 2026-01-01 → 2026-10-19: 5,494 events, 290 match-days. Hours kickoff−48h → kickoff: the Kalshi candle
ending at H is known at H (Kalshi writes quotes even in no-trade hours); the Polymarket bar for hour_start = H−1h is
required (a trade happened). 60.5% of Kalshi outcome-hours had no Polymarket trade and were excluded.

## Method
gap = Kalshi mid − Polymarket last trade; cost = Kalshi fee 0.07p(1−p) + Polymarket fee 0.05p(1−p) + Kalshi
half-spread + Polymarket half-spread proxy max(0.5¢, half the bar's high−low). Primary value 1{|gap| > cost} − 0.05.
(Kalshi fee: the series API returns `fee_type` quadratic / quadratic_with_maker_fees, multiplier 1; the 0.07
coefficient comes from a search summary of Kalshi's fee-schedule PDF, which was not fetchable — flagged.)

## Headline
| Hours to kickoff | Share > cost | Primary value | n | Mean \|gap\| | 99th pct \|gap\| | Mean signed gap |
|---|---|---|---|---|---|---|
| 1–6h | 0.22% | −4.78 pp [−4.83, −4.72] | 71,449 | 0.75¢ [0.74, 0.76] | 2.5¢ | +0.02¢ |
| 7–24h | 0.36% | — | 121,728 | 0.78¢ | 3.0¢ | −0.05¢ |
| 25–48h | 0.57% | — | 68,188 | 0.93¢ | 5.0¢ | −0.17¢ [−0.21, −0.13] |

Median |gap| 0.5¢ in every bucket; mean cost 3.5–4.0¢. With the true (stale) Polymarket book (≈ 1,235 events):
3.4% / 1.0% / 0.4%. Thirds flat. 0.0% of hours exceed cost in EPL, LaLiga, Serie A, Ligue 1, MLS, Championship;
exceedances sit in thin series (women's UCL 3–7%, Leagues Cup, NWSL, friendlies).

## Ladder
L0, L1 pass; L2: n passes but the CI is entirely negative. **Rung L1 (null)** — the venues agree to well within costs.

## Threats
Mid vs last trade is non-synchronous within the hour; the Polymarket spread is a proxy; the mapping drops thin
leagues (not missing at random).
