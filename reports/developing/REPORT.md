# Walk-forward backtest — developing



## Calibration (all test predictions)

| model | multiclass_brier | multiclass_log_loss | brier_home | log_loss_home | brier_draw | log_loss_draw | brier_away | log_loss_away | n_predictions | market_multiclass_brier | market_multiclass_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dixon_coles | 0.628 | 1.0518 | 0.2361 | 0.6714 | 0.1941 | 0.5795 | 0.1977 | 0.5877 | 50309 | 0.6008 | 1.0044 |
| gbdt | 0.6242 | 1.0401 | 0.2343 | 0.6614 | 0.1934 | 0.576 | 0.1965 | 0.58 | 50309 | 0.6008 | 1.0044 |
| gbdt_mkt | 0.6102 | 1.0213 | 0.2269 | 0.6453 | 0.1928 | 0.5745 | 0.1905 | 0.5669 | 50309 | 0.6008 | 1.0044 |

## Model vs market by division (where is the information gap?)

| model | league_code | n | log_loss | market_log_loss | edge_bits |
| --- | --- | --- | --- | --- | --- |
| dixon_coles | MEX | 3771 | 1.0491 | 1.0293 | -0.0286 |
| dixon_coles | USA | 4795 | 1.0323 | 1.0103 | -0.0318 |
| dixon_coles | ARG | 5018 | 1.064 | 1.036 | -0.0404 |
| dixon_coles | POL | 3352 | 1.0663 | 1.0382 | -0.0405 |
| dixon_coles | JAP | 3713 | 1.0617 | 1.0308 | -0.0446 |
| dixon_coles | SUI | 2118 | 1.0502 | 1.0084 | -0.0603 |
| dixon_coles | BRA | 4408 | 1.0464 | 1.0006 | -0.0661 |
| dixon_coles | DEN | 2453 | 1.0551 | 1.0081 | -0.0678 |
| dixon_coles | AUT | 2140 | 1.0443 | 0.9937 | -0.073 |
| dixon_coles | ROM | 3381 | 1.0505 | 0.9996 | -0.0735 |
| dixon_coles | NOR | 2787 | 1.0475 | 0.9932 | -0.0784 |
| dixon_coles | SWE | 2800 | 1.0347 | 0.9801 | -0.0787 |
| dixon_coles | IRL | 2123 | 1.0141 | 0.9542 | -0.0864 |
| dixon_coles | RUS | 2779 | 1.0645 | 0.9843 | -0.1157 |
| dixon_coles | FIN | 2123 | 1.0998 | 0.9995 | -0.1448 |
| dixon_coles | CHN | 2538 | 1.0519 | 0.9332 | -0.1713 |
| gbdt | MEX | 3771 | 1.0538 | 1.0293 | -0.0352 |
| gbdt | POL | 3352 | 1.0628 | 1.0382 | -0.0354 |
| gbdt | USA | 4795 | 1.0358 | 1.0103 | -0.0369 |
| gbdt | JAP | 3713 | 1.0565 | 1.0308 | -0.0371 |
| gbdt | SUI | 2118 | 1.0372 | 1.0084 | -0.0415 |
| gbdt | ARG | 5018 | 1.0658 | 1.036 | -0.0429 |
| gbdt | BRA | 4408 | 1.0304 | 1.0006 | -0.043 |
| gbdt | DEN | 2453 | 1.0395 | 1.0081 | -0.0452 |
| gbdt | NOR | 2787 | 1.0253 | 0.9932 | -0.0463 |
| gbdt | AUT | 2140 | 1.0333 | 0.9937 | -0.0572 |
| gbdt | FIN | 2123 | 1.0442 | 0.9995 | -0.0645 |
| gbdt | ROM | 3381 | 1.0449 | 0.9996 | -0.0655 |
| gbdt | SWE | 2800 | 1.0256 | 0.9801 | -0.0656 |
| gbdt | RUS | 2779 | 1.0314 | 0.9843 | -0.0679 |
| gbdt | IRL | 2123 | 1.0179 | 0.9542 | -0.0918 |
| gbdt | CHN | 2538 | 1.0 | 0.9332 | -0.0964 |
| gbdt_mkt | USA | 4795 | 1.018 | 1.0103 | -0.0112 |
| gbdt_mkt | POL | 3352 | 1.0476 | 1.0382 | -0.0135 |
| gbdt_mkt | BRA | 4408 | 1.0103 | 1.0006 | -0.014 |
| gbdt_mkt | JAP | 3713 | 1.041 | 1.0308 | -0.0148 |
| gbdt_mkt | SUI | 2118 | 1.02 | 1.0084 | -0.0166 |
| gbdt_mkt | MEX | 3771 | 1.0417 | 1.0293 | -0.0179 |
| gbdt_mkt | ARG | 5018 | 1.0497 | 1.036 | -0.0197 |
| gbdt_mkt | DEN | 2453 | 1.0224 | 1.0081 | -0.0206 |
| gbdt_mkt | NOR | 2787 | 1.0093 | 0.9932 | -0.0233 |
| gbdt_mkt | CHN | 2538 | 0.9521 | 0.9332 | -0.0272 |
| gbdt_mkt | ROM | 3381 | 1.0193 | 0.9996 | -0.0285 |
| gbdt_mkt | FIN | 2123 | 1.021 | 0.9995 | -0.0311 |
| gbdt_mkt | SWE | 2800 | 1.0034 | 0.9801 | -0.0335 |
| gbdt_mkt | RUS | 2779 | 1.0094 | 0.9843 | -0.0362 |
| gbdt_mkt | AUT | 2140 | 1.0209 | 0.9937 | -0.0392 |
| gbdt_mkt | IRL | 2123 | 1.0141 | 0.9542 | -0.0864 |

## Staking results (including the losers)

| model | strategy | n_bets | roi | mean_clv_pct | sharpe | hit_rate | total_profit | max_drawdown | final_bankroll | clv_positive_share | clv_t_stat | bet_price_source | closing_price_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dixon_coles | kelly_quarter | 41868 | -0.1227 | 0.0 | -0.0576 | 0.264 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| dixon_coles | kelly_half | 41868 | -0.1524 | 0.0 | -0.0576 | 0.264 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| dixon_coles | flat_1pct | 42174 | -0.0858 | 0.0 | -0.0578 | 0.2656 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt | kelly_quarter | 39906 | -0.0696 | 0.0 | -0.0558 | 0.2702 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt | kelly_half | 39906 | -0.0615 | 0.0 | -0.0558 | 0.2702 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt | flat_1pct | 40222 | -0.0936 | 0.0 | -0.0557 | 0.2722 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt_mkt | kelly_quarter | 29582 | -0.1215 | 0.0 | -0.0528 | 0.2844 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt_mkt | kelly_half | 29582 | -0.1404 | 0.0 | -0.0528 | 0.2844 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |
| gbdt_mkt | flat_1pct | 30122 | -0.0965 | 0.0 | -0.052 | 0.2894 | -1000.0 | 1.0 | 0.0 | 0.0 | 0.0 | Mkt | bet_price_no_closing_available |

## Reading this honestly

- `mean_clv_pct` and `clv_t_stat` are the evidence of edge; `roi` over a few hundred bets is mostly noise.
- A model whose `multiclass_log_loss` is worse than `market_multiclass_log_loss` is *less* informative than the closing line on its own.
- Fractional Kelly vs flat: Kelly compounds edge *and* error — compare drawdowns, not just final bankroll.