# Walk-forward backtest — main

Walk-forward, no shuffling. Bets priced at early Pinnacle odds (`PS`), CLV measured vs Pinnacle closing (`PSC`), edge threshold 3%, retrain every 90 days.

## Calibration (all test predictions)

| model | multiclass_brier | multiclass_log_loss | brier_home | log_loss_home | brier_draw | log_loss_draw | brier_away | log_loss_away | n_predictions | market_multiclass_brier | market_multiclass_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dixon_coles | 0.596 | 1.0084 | 0.2194 | 0.6357 | 0.1864 | 0.5621 | 0.1903 | 0.5701 | 40894 | 0.5778 | 0.9716 |
| gbdt | 0.5936 | 0.9975 | 0.2181 | 0.6265 | 0.1863 | 0.5591 | 0.1892 | 0.5624 | 40894 | 0.5778 | 0.9716 |
| gbdt_mkt | 0.5868 | 0.9885 | 0.2146 | 0.6179 | 0.1859 | 0.5579 | 0.1862 | 0.5563 | 40894 | 0.5778 | 0.9716 |
| gru_sequence | 0.5966 | 1.0418 | 0.218 | 0.6706 | 0.1867 | 0.57 | 0.1919 | 0.5712 | 29240 | 0.5795 | 0.9743 |
| transformer_sequence | 0.5976 | 1.0364 | 0.2186 | 0.6495 | 0.1867 | 0.5827 | 0.1923 | 0.5834 | 29240 | 0.5795 | 0.9743 |

## Model vs market by division (where is the information gap?)

| model | league_code | n | log_loss | market_log_loss | edge_bits |
| --- | --- | --- | --- | --- | --- |
| dixon_coles | B1 | 2985 | 1.0109 | 0.9894 | -0.031 |
| dixon_coles | SC0 | 2384 | 0.9682 | 0.9459 | -0.0321 |
| dixon_coles | N1 | 3235 | 0.9604 | 0.9367 | -0.0342 |
| dixon_coles | E1 | 5961 | 1.0584 | 1.034 | -0.0353 |
| dixon_coles | P1 | 3328 | 0.9491 | 0.9196 | -0.0426 |
| dixon_coles | I1 | 4109 | 0.9819 | 0.9493 | -0.0471 |
| dixon_coles | E0 | 4100 | 0.9906 | 0.9575 | -0.0477 |
| dixon_coles | F1 | 3765 | 1.0189 | 0.9851 | -0.0488 |
| dixon_coles | D1 | 3285 | 1.023 | 0.9787 | -0.0638 |
| dixon_coles | SP1 | 4121 | 1.0159 | 0.9629 | -0.0765 |
| dixon_coles | T1 | 3585 | 1.068 | 0.9807 | -0.1259 |
| gbdt | E1 | 5961 | 1.0512 | 1.034 | -0.0248 |
| gbdt | N1 | 3235 | 0.9583 | 0.9367 | -0.0312 |
| gbdt | P1 | 3328 | 0.9425 | 0.9196 | -0.0331 |
| gbdt | B1 | 2985 | 1.0138 | 0.9894 | -0.0351 |
| gbdt | T1 | 3585 | 1.0054 | 0.9807 | -0.0356 |
| gbdt | F1 | 3765 | 1.0104 | 0.9851 | -0.0366 |
| gbdt | D1 | 3285 | 1.0058 | 0.9787 | -0.0391 |
| gbdt | I1 | 4109 | 0.9778 | 0.9493 | -0.0411 |
| gbdt | SP1 | 4121 | 0.9923 | 0.9629 | -0.0424 |
| gbdt | SC0 | 2384 | 0.9798 | 0.9459 | -0.049 |
| gbdt | E0 | 4100 | 0.9946 | 0.9575 | -0.0535 |
| gbdt_mkt | D1 | 3285 | 0.986 | 0.9787 | -0.0105 |
| gbdt_mkt | T1 | 3585 | 0.9888 | 0.9807 | -0.0117 |
| gbdt_mkt | E1 | 5961 | 1.0454 | 1.034 | -0.0165 |
| gbdt_mkt | N1 | 3235 | 0.9506 | 0.9367 | -0.02 |
| gbdt_mkt | SP1 | 4121 | 0.9792 | 0.9629 | -0.0235 |
| gbdt_mkt | P1 | 3328 | 0.9363 | 0.9196 | -0.0241 |
| gbdt_mkt | B1 | 2985 | 1.0103 | 0.9894 | -0.0302 |
| gbdt_mkt | E0 | 4100 | 0.9785 | 0.9575 | -0.0302 |
| gbdt_mkt | F1 | 3765 | 1.0072 | 0.9851 | -0.0319 |
| gbdt_mkt | SC0 | 2384 | 0.9705 | 0.9459 | -0.0354 |
| gbdt_mkt | I1 | 4109 | 0.9765 | 0.9493 | -0.0393 |
| gru_sequence | B1 | 2232 | 1.0158 | 0.9878 | -0.0404 |
| gru_sequence | E0 | 2929 | 0.9926 | 0.9631 | -0.0426 |
| gru_sequence | E1 | 4233 | 1.0724 | 1.0369 | -0.0512 |
| gru_sequence | T1 | 2633 | 1.0332 | 0.9846 | -0.0701 |
| gru_sequence | D1 | 2348 | 1.0296 | 0.971 | -0.0846 |
| gru_sequence | P1 | 2390 | 0.9868 | 0.9219 | -0.0936 |
| gru_sequence | SP1 | 2940 | 1.0446 | 0.9702 | -0.1074 |
| gru_sequence | F1 | 2594 | 1.0754 | 0.9909 | -0.1218 |
| gru_sequence | I1 | 2929 | 1.056 | 0.9631 | -0.134 |
| gru_sequence | N1 | 2290 | 1.0433 | 0.9354 | -0.1556 |
| gru_sequence | SC0 | 1691 | 1.1149 | 0.9355 | -0.2587 |
| transformer_sequence | B1 | 2232 | 1.0029 | 0.9878 | -0.0218 |
| transformer_sequence | SP1 | 2940 | 1.0056 | 0.9702 | -0.0511 |
| transformer_sequence | E0 | 2929 | 1.0074 | 0.9631 | -0.0639 |
| transformer_sequence | I1 | 2929 | 1.0129 | 0.9631 | -0.0718 |
| transformer_sequence | F1 | 2594 | 1.0465 | 0.9909 | -0.0801 |
| transformer_sequence | E1 | 4233 | 1.0924 | 1.0369 | -0.0801 |
| transformer_sequence | T1 | 2633 | 1.0468 | 0.9846 | -0.0898 |
| transformer_sequence | P1 | 2390 | 0.9984 | 0.9219 | -0.1104 |
| transformer_sequence | N1 | 2290 | 1.0316 | 0.9354 | -0.1388 |
| transformer_sequence | D1 | 2348 | 1.0716 | 0.971 | -0.1452 |
| transformer_sequence | SC0 | 1691 | 1.071 | 0.9355 | -0.1955 |

## Staking results (including the losers)

| model | strategy | n_bets | roi | mean_clv_pct | sharpe | hit_rate | total_profit | max_drawdown | final_bankroll | clv_positive_share | clv_t_stat | bet_price_source | closing_price_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dixon_coles | kelly_quarter | 29325 | -0.054 | 0.0 | -0.0658 | 0.2883 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| dixon_coles | kelly_half | 29325 | -0.0527 | 0.0 | -0.0658 | 0.2883 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| dixon_coles | flat_1pct | 29573 | -0.0782 | 0.0 | -0.0655 | 0.2915 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt | kelly_quarter | 29740 | -0.0782 | 0.0 | -0.0525 | 0.2771 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt | kelly_half | 29740 | -0.0673 | 0.0 | -0.0525 | 0.2771 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt | flat_1pct | 29953 | -0.0835 | 0.0 | -0.0524 | 0.2796 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt_mkt | kelly_quarter | 23908 | -0.0672 | 0.0 | -0.0419 | 0.2839 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt_mkt | kelly_half | 23908 | -0.0612 | 0.0 | -0.0419 | 0.2839 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt_mkt | flat_1pct | 24208 | -0.0824 | 0.0 | -0.0422 | 0.2881 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gru_sequence | kelly_quarter | 20968 | -0.208 | 0.0 | -0.0604 | 0.3222 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gru_sequence | kelly_half | 20968 | -0.2428 | 0.0 | -0.0604 | 0.3222 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gru_sequence | flat_1pct | 21182 | -0.1573 | 0.0 | -0.0602 | 0.3257 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| transformer_sequence | kelly_quarter | 20751 | -0.1341 | 0.0 | -0.0541 | 0.3184 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| transformer_sequence | kelly_half | 20751 | -0.1503 | 0.0 | -0.0541 | 0.3184 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| transformer_sequence | flat_1pct | 20997 | -0.1112 | 0.0 | -0.0537 | 0.3228 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |

## Reading this honestly

- `mean_clv_pct` and `clv_t_stat` are the evidence of edge; `roi` over a few hundred bets is mostly noise.
- A model whose `multiclass_log_loss` is worse than `market_multiclass_log_loss` is *less* informative than the closing line on its own.
- Fractional Kelly vs flat: Kelly compounds edge *and* error — compare drawdowns, not just final bankroll.