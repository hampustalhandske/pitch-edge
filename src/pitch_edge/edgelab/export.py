"""Write a binary-market dataset in the folder layout edgelab's `parquet` provider reads.

edgelab grades a prediction-market strategy by re-simulating it from market data, so pitch-edge
hands it three frames per market instead of a bet log:

* `book_snapshots.parquet`: the quoted Yes book (one level) per market and timestamp,
* `resolutions.parquet`: the 0/1 outcome and when it was known,
* `features.parquet`: per market a `kickoff_ts` row (epoch seconds) and a `market_type` row (numeric
  code) that the engine reads itself, plus one `pe_model_p` row per forecast (the model's P(Yes),
  stamped with the `known_at` at which pitch-edge would have known it).

An empty `bars.parquet` is written because the provider requires the file. Every invariant edgelab's
schemas and audit enforce is checked here first and a violation raises `EdgelabExportError` naming the
rule; nothing is repaired.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

MARKET_TYPE_CODES: Mapping[str, int] = {
    "win_home": 1,
    "win_away": 2,
    "draw": 3,
    "total": 4,
    "spread_home": 5,
    "spread_away": 6,
    "btts": 7,
}
"""edgelab's numeric `market_type` codes (`edgelab.instruments.prediction_market.MARKET_TYPE_CODES`)."""

MODEL_FEATURE = "pe_model_p"
"""Feature name carrying the model's P(Yes) for the market named in the row's `symbol`."""

_BARS_COLUMNS = "ts open high low close volume as_traded_close as_traded_volume".split()


class EdgelabExportError(ValueError):
    """The inputs break an invariant edgelab would refuse or a pitch-edge safety rule."""


@dataclasses.dataclass(frozen=True)
class ExportReport:
    """What `export_prediction_market_dataset` wrote.

    Attributes:
        path: The dataset folder.
        n_markets: Markets exported.
        n_snapshots: Rows in `book_snapshots`.
        n_resolutions: Rows in `resolutions`.
        n_feature_rows: Rows in `features`.
        first_ts: Earliest snapshot time (UTC).
        last_ts: Latest snapshot time (UTC).
    """

    path: Path
    n_markets: int
    n_snapshots: int
    n_resolutions: int
    n_feature_rows: int
    first_ts: pd.Timestamp
    last_ts: pd.Timestamp


def _utc(frame: pd.DataFrame, column: str, where: str) -> pd.Series:
    """`frame[column]` as tz-aware UTC timestamps; a tz-naive or unparseable column raises."""
    if column not in frame.columns:
        raise EdgelabExportError(f"{where}: missing column {column!r}.")
    series = frame[column]
    try:
        converted = pd.to_datetime(series, utc=False)
    except (ValueError, TypeError) as exc:
        raise EdgelabExportError(f"{where}.{column}: not timestamps ({exc}).") from exc
    if getattr(converted.dt, "tz", None) is None:
        raise EdgelabExportError(f"{where}.{column}: timestamps must be timezone-aware (UTC).")
    if converted.isna().any():
        raise EdgelabExportError(f"{where}.{column}: contains missing timestamps.")
    return converted.dt.tz_convert("UTC").astype("datetime64[ns, UTC]")


def _need(frame: pd.DataFrame, columns: list[str], where: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise EdgelabExportError(f"{where}: missing column(s) {missing}.")


def _finite(series: pd.Series, where: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise EdgelabExportError(f"{where}: contains missing or non-finite values.")
    return values


def _snapshot_frame(snapshots: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalise price snapshots to level-0 `book_snapshots` rows and check them."""
    where = "snapshots"
    _need(snapshots, ["market", "ts"], where)
    if {"bid", "ask"} <= set(snapshots.columns):
        bid = _finite(snapshots["bid"], "snapshots.bid")
        ask = _finite(snapshots["ask"], "snapshots.ask")
    elif {"price", "spread"} <= set(snapshots.columns):
        price = _finite(snapshots["price"], "snapshots.price")
        spread = _finite(snapshots["spread"], "snapshots.spread")
        if (spread < 0).any():
            raise EdgelabExportError("snapshots.spread: negative spread.")
        bid, ask = price - spread / 2.0, price + spread / 2.0
    else:
        raise EdgelabExportError("snapshots: need columns (bid, ask) or (price, spread).")
    if "size" in snapshots.columns:
        bid_size = ask_size = _finite(snapshots["size"], "snapshots.size")
    elif {"bid_size", "ask_size"} <= set(snapshots.columns):
        bid_size = _finite(snapshots["bid_size"], "snapshots.bid_size")
        ask_size = _finite(snapshots["ask_size"], "snapshots.ask_size")
    else:
        raise EdgelabExportError("snapshots: need `size` or (`bid_size`, `ask_size`).")
    if (bid_size < 0).any() or (ask_size < 0).any():
        raise EdgelabExportError("snapshots: negative size.")
    if not ((bid > 0).all() and (ask < 1).all() and (bid < 1).all() and (ask > 0).all()):
        raise EdgelabExportError("snapshots: bid and ask must lie strictly inside (0, 1).")
    if (bid > ask).any():
        raise EdgelabExportError("snapshots: crossed quote (bid > ask).")
    out = pd.DataFrame(
        {
            "ts": _utc(snapshots, "ts", where).to_numpy(),
            "market": snapshots["market"].astype(str).to_numpy(),
            "level": np.zeros(len(snapshots), dtype="int64"),
            "bid_price": bid,
            "bid_size": bid_size.astype("float64"),
            "ask_price": ask,
            "ask_size": ask_size.astype("float64"),
            "source": source,
        }
    )
    out["ts"] = pd.to_datetime(out["ts"], utc=True).astype("datetime64[ns, UTC]")
    if out.empty:
        raise EdgelabExportError("snapshots: no rows.")
    if out.duplicated(["market", "ts", "level"]).any():
        raise EdgelabExportError("snapshots: duplicate (market, ts).")
    return out.sort_values(["market", "ts", "level"], kind="mergesort").reset_index(drop=True)


def _resolution_frame(resolutions: pd.DataFrame) -> pd.DataFrame:
    where = "resolutions"
    _need(resolutions, ["market", "resolved_at", "outcome"], where)
    outcome = _finite(resolutions["outcome"], "resolutions.outcome")
    if not np.isin(outcome, (0.0, 1.0)).all():
        raise EdgelabExportError("resolutions.outcome: must be 0 or 1.")
    source = (
        resolutions["resolution_source"].astype(str) if "resolution_source" in resolutions.columns else "pitch-edge"
    )
    out = pd.DataFrame(
        {
            "market": resolutions["market"].astype(str).to_numpy(),
            "resolved_at": _utc(resolutions, "resolved_at", where).to_numpy(),
            "outcome": outcome.astype("int64"),
            "resolution_source": source,
        }
    )
    out["resolved_at"] = pd.to_datetime(out["resolved_at"], utc=True).astype("datetime64[ns, UTC]")
    if out["market"].duplicated().any():
        raise EdgelabExportError("resolutions: duplicate market.")
    return out.sort_values(["market", "resolved_at"], kind="mergesort").reset_index(drop=True)


def _kickoff_seconds(series: pd.Series) -> np.ndarray:
    """Epoch seconds from tz-aware timestamps or numeric epoch seconds."""
    if pd.api.types.is_numeric_dtype(series):
        return _finite(series, "features.kickoff_ts")
    stamps = pd.to_datetime(series)
    if getattr(stamps.dt, "tz", None) is None:
        raise EdgelabExportError("features.kickoff_ts: timestamps must be timezone-aware.")
    return (stamps.dt.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def _market_type_code(series: pd.Series) -> np.ndarray:
    codes = []
    for value in series:
        if isinstance(value, str):
            if value not in MARKET_TYPE_CODES:
                raise EdgelabExportError(
                    f"features.market_type: unknown {value!r}; use one of {sorted(MARKET_TYPE_CODES)}."
                )
            codes.append(MARKET_TYPE_CODES[value])
        else:
            if int(value) not in MARKET_TYPE_CODES.values():
                raise EdgelabExportError(f"features.market_type: unknown code {value!r}.")
            codes.append(int(value))
    return np.asarray(codes, dtype="float64")


def export_prediction_market_dataset(
    snapshots: pd.DataFrame,
    resolutions: pd.DataFrame,
    features: pd.DataFrame,
    out_dir: str | Path,
    *,
    source: str = "real",
    allow_in_play: bool = False,
    include_market_type: bool = True,
) -> ExportReport:
    """Validate binary-market data and write an edgelab `parquet` dataset folder.

    Args:
        snapshots: One row per market and timestamp of the Yes book. Columns `market`, `ts`
            (tz-aware), and either `bid`, `ask` or `price`, `spread` (full spread), and either
            `size` or `bid_size`, `ask_size`. Prices are probabilities in (0, 1).
        resolutions: One row per market: `market`, `resolved_at` (tz-aware), `outcome` (1 Yes, 0 No),
            optional `resolution_source`.
        features: One row per model forecast: `market`, `ts` (when the forecast describes),
            `known_at` (when pitch-edge would have had it; never before `ts`), `model_p` (P(Yes) in
            (0, 1)), plus `kickoff_ts` (tz-aware or epoch seconds) and `market_type` (name or code,
            see `MARKET_TYPE_CODES`) which must be constant per market.
        out_dir: Folder to create; files in it are overwritten.
        source: `real` for venue quotes or `synthetic` for modelled ones (bookmaker odds converted
            to prices are `synthetic`); recorded per snapshot row.
        allow_in_play: Permit snapshots after kickoff. Off by default so a fill can never use a
            price a venue would not honour once play has started.
        include_market_type: Write the per-market `market_type` feature. edgelab charges every
            market that carries one the Polymarket sports taker-fee curve; pass `False` for markets
            that are not Polymarket contracts (bookmaker odds), which are then charged the flat
            commission only. `features.market_type` is optional in that case and ignored if present.

    Returns:
        An `ExportReport`.

    Raises:
        EdgelabExportError: Any of: a tz-naive timestamp; a price outside (0, 1) or a crossed quote;
            duplicate (market, ts) rows; a market without a resolution, forecast or kickoff (or the
            reverse); `outcome` not 0/1; `known_at < ts`; `model_p` outside (0, 1); `resolved_at`
            before the market's last snapshot, before kickoff, or before a forecast's `known_at`;
            a snapshot at or after kickoff unless `allow_in_play`; kickoff or market type that
            varies within a market.
    """
    if source not in ("real", "synthetic"):
        raise EdgelabExportError("source must be 'real' or 'synthetic'.")
    book = _snapshot_frame(snapshots, source)
    res = _resolution_frame(resolutions)

    where = "features"
    _need(
        features,
        ["market", "ts", "known_at", "model_p", "kickoff_ts", *(["market_type"] if include_market_type else [])],
        where,
    )
    feat = pd.DataFrame(
        {
            "symbol": features["market"].astype(str).to_numpy(),
            "ts": _utc(features, "ts", where).to_numpy(),
            "known_at": _utc(features, "known_at", where).to_numpy(),
            "model_p": _finite(features["model_p"], "features.model_p"),
            "kickoff": _kickoff_seconds(features["kickoff_ts"]),
            "mtype": _market_type_code(features["market_type"]) if include_market_type else 0.0,
        }
    )
    for col in ("ts", "known_at"):
        feat[col] = pd.to_datetime(feat[col], utc=True).astype("datetime64[ns, UTC]")
    if feat.empty:
        raise EdgelabExportError("features: no rows.")
    if ((feat["model_p"] <= 0) | (feat["model_p"] >= 1)).any():
        raise EdgelabExportError("features.model_p: must lie strictly inside (0, 1).")
    if (feat["known_at"] < feat["ts"]).any():
        raise EdgelabExportError("features: known_at earlier than ts (look-ahead).")
    if feat.duplicated(["symbol", "ts"]).any():
        raise EdgelabExportError("features: duplicate (market, ts) forecast.")
    per_market = feat.groupby("symbol")
    if (per_market["kickoff"].nunique() > 1).any() or (per_market["mtype"].nunique() > 1).any():
        raise EdgelabExportError("features: kickoff_ts / market_type vary within a market.")
    meta = per_market[["kickoff", "mtype"]].first()

    markets = set(book["market"])
    for name, other in (("resolutions", set(res["market"])), ("features", set(meta.index))):
        if markets != other:
            diff = sorted(markets ^ other)[:5]
            raise EdgelabExportError(f"{name} and snapshots cover different markets, e.g. {diff}.")

    last_snap = book.groupby("market")["ts"].max()
    first_snap = book.groupby("market")["ts"].min()
    r = res.set_index("market")
    if (r["resolved_at"] < last_snap.reindex(r.index)).any():
        raise EdgelabExportError("resolutions: resolved_at before the market's last snapshot.")
    kick = pd.to_datetime(meta["kickoff"], unit="s", utc=True).astype("datetime64[ns, UTC]")
    if (r["resolved_at"] < kick.reindex(r.index)).any():
        raise EdgelabExportError("resolutions: resolved_at before kickoff.")
    if not allow_in_play and (last_snap.reindex(kick.index) >= kick).any():
        raise EdgelabExportError("snapshots: a snapshot at or after kickoff (allow_in_play=False).")
    late = feat["known_at"] > r["resolved_at"].reindex(feat["symbol"]).to_numpy()
    if late.any():
        raise EdgelabExportError("features: a forecast known_at after its market resolved.")

    meta_rows = []
    for market, row in meta.iterrows():
        stamp = first_snap[market]
        pairs = [("kickoff_ts", row["kickoff"])] + ([("market_type", row["mtype"])] if include_market_type else [])
        for name, value in pairs:
            meta_rows.append({"ts": stamp, "symbol": market, "known_at": stamp, "name": name, "value": float(value)})
    model_rows = pd.DataFrame(
        {
            "ts": feat["ts"],
            "symbol": feat["symbol"],
            "known_at": feat["known_at"],
            "name": MODEL_FEATURE,
            "value": feat["model_p"],
        }
    )
    meta_frame = pd.DataFrame(meta_rows)
    meta_frame["ts"] = pd.to_datetime(meta_frame["ts"], utc=True).astype("datetime64[ns, UTC]")
    meta_frame["known_at"] = meta_frame["ts"]
    feats = pd.concat([meta_frame, model_rows], ignore_index=True)
    if feats.duplicated(["symbol", "ts", "name"]).any():
        raise EdgelabExportError("features: duplicate (symbol, ts, name).")
    feats = feats.sort_values(["symbol", "ts", "name"], kind="mergesort").reset_index(drop=True)

    bars = pd.DataFrame({c: pd.Series(dtype="float64") for c in _BARS_COLUMNS})
    bars["ts"] = pd.Series(dtype="datetime64[ns, UTC]")
    bars["symbol"] = pd.Series(dtype="str")
    bars = bars[["ts", "symbol", *[c for c in _BARS_COLUMNS if c != "ts"]]]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(out / "bars.parquet", index=False)
    book.to_parquet(out / "book_snapshots.parquet", index=False)
    res.to_parquet(out / "resolutions.parquet", index=False)
    feats.to_parquet(out / "features.parquet", index=False)
    return ExportReport(
        path=out,
        n_markets=len(markets),
        n_snapshots=len(book),
        n_resolutions=len(res),
        n_feature_rows=len(feats),
        first_ts=book["ts"].min(),
        last_ts=book["ts"].max(),
    )
