# Run scripts

The one-time restartable staged-build scripts (`stage_a.py` ... `stage_e.py`) used for the initial
research build have been removed now that the project runs as a standing daily loop instead of a
manual multi-hour build. Use these instead:

- `uv run pitch-edge refresh [--fast]` — ingest → features → backtest → RAG index, one shot
- `uv run pitch-edge schedule` — the standing local loop (hourly ingest, nightly refresh + retrain)

Run either with `caffeinate -i` on macOS so a laptop sleep doesn't cost the run.
