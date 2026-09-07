# Run scripts

Thin, logged wrappers around the library used for the recorded runs (`pitch-edge refresh` does the same
end-to-end; these split it into restartable stages so a 7-hour laptop sleep or a 503 from one source
doesn't cost the whole run):

| script | what it does | typical time |
|---|---|---|
| `stage_a.py` | feature store from the warehouse → Dixon-Coles / GBDT / GBDT+market walk-forward backtests (label `main`) | ~15 min |
| `stage_a2.py` | artifacts (team strengths, feature importance, GNN embeddings, in-play paths, data universe) → RAG index → ablation → first signal batch to the approval gate | ~12 min |
| `stage_b.py` | Sweden (openfootball), Kalshi, Swedish + English news, Polymarket, TheSportsDB | ~4 min |
| `stage_b2.py` | Transfermarkt open dataset (210 MB DuckDB → 7.1 M rows) | ~1 min after download |
| `stage_c.py` | developing-market backtest (label `developing`) + GRU on the main slice, same run id | ~30-60 min |
| `stage_d.py` (+ `stage_d2.py`, the re-run with the corrected temperature-scaling recipe, same run id) | rebuild the hybrid RAG index (Chroma ∪ BM25 + rerank), retrieval eval, Lightning GRU + Transformer backtests (2018→ slice) | ~30-45 min |
| `stage_e.py` | Phase 5: Wikipedia pageviews → feature store **with** Transfermarkt rotation/referee context + attention anomalies → pre-registered ablation (`reports/ablation_phase5.csv`) → referee-lag event study → cross-venue lead-lag → two dossiers. `python scripts/stage_e.py 3 4` runs only those steps | ~10 min pageviews + ~10 min ablation |

Run with `caffeinate -i uv run python scripts/<name>.py` so macOS doesn't sleep mid-run.
