# Walk-forward backtest — squad_value

Walk-forward, no shuffling. Bets priced at early Pinnacle odds (`PS`), CLV measured vs Pinnacle closing (`PSC`), edge threshold 3%, retrain every 90 days.

## Calibration (all test predictions)

| model | multiclass_brier | multiclass_log_loss | brier_home | log_loss_home | brier_draw | log_loss_draw | brier_away | log_loss_away | market_multiclass_brier | market_multiclass_log_loss | n_predictions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gbdt | 0.5938 | 0.9988 | 0.2183 | 0.63 | 0.1865 | 0.5601 | 0.189 | 0.5595 | 0.5776 | 0.9713 | 40357 |
| gbdt_mkt | 0.5877 | 0.99 | 0.2152 | 0.6216 | 0.1862 | 0.5602 | 0.1863 | 0.5528 | 0.5776 | 0.9713 | 40357 |
| gbdt_squadval | 0.5927 | 0.9973 | 0.2178 | 0.6261 | 0.1864 | 0.5621 | 0.1885 | 0.559 | 0.5776 | 0.9713 | 40357 |
| gbdt_mkt_squadval | 0.587 | 0.9895 | 0.2148 | 0.6201 | 0.1861 | 0.5614 | 0.186 | 0.5521 | 0.5776 | 0.9713 | 40357 |

## Model vs market by division (where is the information gap?)

| model | league_code | n | log_loss | market_log_loss | edge_bits |
| --- | --- | --- | --- | --- | --- |
| gbdt | E1 | 5889 | 1.0557 | 1.0343 | -0.031 |
| gbdt | P1 | 3292 | 0.9425 | 0.9209 | -0.0312 |
| gbdt | B1 | 2939 | 1.0124 | 0.99 | -0.0324 |
| gbdt | F1 | 3704 | 1.0076 | 0.9835 | -0.0347 |
| gbdt | E0 | 4050 | 0.9814 | 0.9558 | -0.037 |
| gbdt | T1 | 3541 | 1.0063 | 0.9803 | -0.0375 |
| gbdt | D1 | 3240 | 1.0051 | 0.9774 | -0.04 |
| gbdt | N1 | 3190 | 0.9671 | 0.937 | -0.0435 |
| gbdt | I1 | 4049 | 0.9808 | 0.949 | -0.0458 |
| gbdt | SP1 | 4071 | 0.9961 | 0.9635 | -0.0471 |
| gbdt | SC0 | 2357 | 0.995 | 0.9456 | -0.0712 |
| gbdt_mkt | T1 | 3541 | 0.9909 | 0.9803 | -0.0152 |
| gbdt_mkt | D1 | 3240 | 0.9886 | 0.9774 | -0.0162 |
| gbdt_mkt | F1 | 3704 | 0.9964 | 0.9835 | -0.0185 |
| gbdt_mkt | B1 | 2939 | 1.0031 | 0.99 | -0.0189 |
| gbdt_mkt | E1 | 5889 | 1.05 | 1.0343 | -0.0228 |
| gbdt_mkt | P1 | 3292 | 0.9378 | 0.9209 | -0.0244 |
| gbdt_mkt | SC0 | 2357 | 0.9639 | 0.9456 | -0.0264 |
| gbdt_mkt | SP1 | 4071 | 0.9838 | 0.9635 | -0.0293 |
| gbdt_mkt | E0 | 4050 | 0.9761 | 0.9558 | -0.0294 |
| gbdt_mkt | N1 | 3190 | 0.9618 | 0.937 | -0.0358 |
| gbdt_mkt | I1 | 4049 | 0.9875 | 0.949 | -0.0555 |
| gbdt_mkt_squadval | D1 | 3240 | 0.9855 | 0.9774 | -0.0117 |
| gbdt_mkt_squadval | T1 | 3541 | 0.9898 | 0.9803 | -0.0137 |
| gbdt_mkt_squadval | F1 | 3704 | 0.9936 | 0.9835 | -0.0145 |
| gbdt_mkt_squadval | B1 | 2939 | 1.0 | 0.99 | -0.0145 |
| gbdt_mkt_squadval | E1 | 5889 | 1.0494 | 1.0343 | -0.0219 |
| gbdt_mkt_squadval | SC0 | 2357 | 0.9622 | 0.9456 | -0.0238 |
| gbdt_mkt_squadval | E0 | 4050 | 0.9765 | 0.9558 | -0.0299 |
| gbdt_mkt_squadval | N1 | 3190 | 0.9611 | 0.937 | -0.0348 |
| gbdt_mkt_squadval | P1 | 3292 | 0.9468 | 0.9209 | -0.0373 |
| gbdt_mkt_squadval | SP1 | 4071 | 0.9912 | 0.9635 | -0.0401 |
| gbdt_mkt_squadval | I1 | 4049 | 0.9783 | 0.949 | -0.0422 |
| gbdt_squadval | D1 | 3240 | 0.9954 | 0.9774 | -0.0261 |
| gbdt_squadval | E1 | 5889 | 1.0551 | 1.0343 | -0.03 |
| gbdt_squadval | P1 | 3292 | 0.9425 | 0.9209 | -0.0312 |
| gbdt_squadval | F1 | 3704 | 1.0056 | 0.9835 | -0.0319 |
| gbdt_squadval | B1 | 2939 | 1.0124 | 0.99 | -0.0324 |
| gbdt_squadval | T1 | 3541 | 1.0033 | 0.9803 | -0.0332 |
| gbdt_squadval | E0 | 4050 | 0.98 | 0.9558 | -0.0349 |
| gbdt_squadval | N1 | 3190 | 0.9659 | 0.937 | -0.0416 |
| gbdt_squadval | I1 | 4049 | 0.98 | 0.949 | -0.0447 |
| gbdt_squadval | SP1 | 4071 | 0.9991 | 0.9635 | -0.0514 |
| gbdt_squadval | SC0 | 2357 | 0.9921 | 0.9456 | -0.067 |

## Staking results (including the losers)

| model | strategy | n_bets | roi | mean_clv_pct | sharpe | hit_rate | total_profit | max_drawdown | final_bankroll | clv_positive_share | clv_t_stat | bet_price_source | closing_price_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gbdt | kelly_quarter | 29506 | -0.1162 | 0.0 | -0.0493 | 0.286 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt | kelly_half | 29506 | -0.1853 | 0.0 | -0.0493 | 0.286 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt | flat_1pct | 29747 | -0.0729 | 0.0 | -0.0497 | 0.2883 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt_mkt | kelly_quarter | 24023 | -0.0924 | 0.0 | -0.0454 | 0.29 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt_mkt | kelly_half | 24023 | -0.103 | 0.0 | -0.0454 | 0.29 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt_mkt | flat_1pct | 24290 | -0.0767 | 0.0 | -0.0453 | 0.294 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt_squadval | kelly_quarter | 29156 | -0.1115 | 0.0 | -0.051 | 0.2885 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt_squadval | kelly_half | 29156 | -0.1753 | 0.0 | -0.051 | 0.2885 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt_squadval | flat_1pct | 29418 | -0.0787 | 0.0 | -0.0509 | 0.2917 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt_mkt_squadval | kelly_quarter | 23998 | -0.0944 | 0.0 | -0.0394 | 0.3007 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt_mkt_squadval | kelly_half | 23998 | -0.105 | 0.0 | -0.0394 | 0.3007 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt_mkt_squadval | flat_1pct | 24307 | -0.0884 | 0.0 | -0.0395 | 0.3051 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |

## Reading this honestly

- `mean_clv_pct` and `clv_t_stat` are the evidence of edge; `roi` over a few hundred bets is mostly noise.
- A model whose `multiclass_log_loss` is worse than `market_multiclass_log_loss` is *less* informative than the closing line on its own.
- Fractional Kelly vs flat: Kelly compounds edge *and* error — compare drawdowns, not just final bankroll.