# Walk-forward backtest — main

Walk-forward, no shuffling. Bets priced at early Pinnacle odds (`PS`), CLV measured vs Pinnacle closing (`PSC`), edge threshold 3%, retrain every 90 days.

## Model vs market (all divisions combined)

| model | n | log_loss | market_log_loss | edge_bits |
| --- | --- | --- | --- | --- |
| gru_sequence | 29240 | 1.0418 | 0.9743 | -0.0974 |
| transformer_sequence | 29240 | 1.0364 | 0.9743 | -0.0897 |
| gbdt_mkt | 41764 | 0.997 | 0.9715 | -0.0369 |
| dixon_coles | 1192 | 1.0808 | 0.9623 | -0.171 |
| gbdt | 1192 | 1.155 | 0.9623 | -0.278 |

## Best staking result per model (ranked by CLV, not ROI)

| model | strategy | n_bets | mean_clv_pct | roi | sharpe |
| --- | --- | --- | --- | --- | --- |
| dixon_coles | flat_1pct | 734 | -0.0088 | -0.1314 | -0.0624 |
| gbdt | kelly_quarter | 777 | -0.0025 | -0.0875 | -0.0585 |
| gbdt_mkt | flat_1pct | 20445 | 0.0018 | -0.0594 | -0.0213 |
| gru_sequence | kelly_quarter | 20968 | 0.0 | -0.208 | -0.0604 |
| transformer_sequence | kelly_quarter | 20751 | 0.0 | -0.1341 | -0.0541 |

_`mean_clv_pct` is the evidence of edge; `roi` over a few hundred bets is mostly noise. Negative `edge_bits` means the model is less informative than the closing price._