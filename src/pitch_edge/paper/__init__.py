"""Forward paper-trading ledger (evidence rung L5 in `research/STANDARDS.md`).

Collects read-only market quotes (the-odds-api bookmakers, Polymarket and Kalshi public books) into
its own DuckDB file, lets strategies log immutable paper decisions on fixtures that have not kicked
off, and settles them later against a sharp closing price (closing-line value) and the result.

Nothing in this package places an order or calls an authenticated trading endpoint: every HTTP
call is a public, read-only market-data GET. Stakes are paper units.
"""
