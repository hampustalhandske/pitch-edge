"""Stage-2 pull for R4 into data/research/r4/: Bluesky hourly aggregates, Polymarket bars, Reddit pre-kickoff counts.

Writes:
  bsky_hourly.parquet     per (query_id, hr) for every Bluesky search query: `n` posts, `n_reply`, `n_pos`, `n_neg`
                          (posts matching the fixed lexicon in `lexicon.py`), where `hr` is the UTC hour of the
                          post's `indexed_at` (the Bluesky AppView's index time = when the post became public; for
                          `time_flag = 'ok'` posts it equals the client `created_at` to a median 1.4 s, for
                          back- or future-dated posts `created_at` is not trustworthy). A post in bucket `hr` is
                          known at `hr + 1h`. Like/repost/reply counts are as-retrieved and are never pulled.
                          Weekly chunks 2026-01-05 -> 2026-10-20 (the search backfill's coverage).
  pm_bars.parquet         pm_soccer_bars_1h rows for every condition of a PM 1X2 group (`pm_groups.parquet`) with
                          kickoff in [2026-01-05, 2026-10-20], hour_start in [ko - 49h, ko) (a bar is known at
                          hour_start + 1h)
  reddit_preko.parquet    per linked Reddit match thread: comments created before ko - 24h, ko - 6h, ko - 1h, ko
                          (counts only; no bodies, no scores)
  snapshot_<stage>.json   pull start/finish time and row counts per stage

Usage: nohup caffeinate -i arch -arm64 uv run python research/studies/r4-common/pull_data.py [reddit|bars|bsky ...] \
         > data/research/r4/pull_<stage>.log 2>&1 &
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r1-common"))
sys.path.insert(0, str(HERE))

from lake import REPO, sql  # noqa: E402
from lexicon import NEGATIVE, POSITIVE, sql_pattern  # noqa: E402

OUT = REPO / "data" / "research" / "r4"
BSKY_FROM = pd.Timestamp("2026-01-05", tz="UTC")
BSKY_TO = pd.Timestamp("2026-10-20", tz="UTC")
CHUNK = 250

BSKY_SQL = """
select m.source_id as query_id, date_trunc('hour', p.indexed_at) as hr, count(*)::BIGINT as n,
       sum(case when p.is_reply then 1 else 0 end)::BIGINT as n_reply,
       sum(case when regexp_matches(lower(p.text), ?) then 1 else 0 end)::BIGINT as n_pos,
       sum(case when regexp_matches(lower(p.text), ?) then 1 else 0 end)::BIGINT as n_neg
from bluesky_posts p join bluesky_post_matches m on m.author_hash = p.author_hash and m.rkey = p.rkey
where m.source_type = 'query' and p.indexed_at >= ?::TIMESTAMPTZ and p.indexed_at < ?::TIMESTAMPTZ
group by 1, 2
"""


def pull_bsky() -> pd.DataFrame:
    parts = []
    t = BSKY_FROM
    while t < BSKY_TO:
        e = min(t + pd.Timedelta(days=7), BSKY_TO)
        df = sql(BSKY_SQL, [sql_pattern(POSITIVE), sql_pattern(NEGATIVE), t.isoformat(), e.isoformat()])
        parts.append(df)
        print(f"bsky {t.date()} -> {e.date()}: {len(df)} rows, {int(df.n.sum())} posts", flush=True)
        t = e
    return pd.concat(parts, ignore_index=True)


def pull_bars(groups: pd.DataFrame) -> pd.DataFrame:
    g = groups[(groups.ko >= BSKY_FROM) & (groups.ko <= BSKY_TO)]
    pairs = pd.concat([g[["ko"]].assign(condition_id=g[f"cond_{s}"]) for s in ("home", "draw", "away")])
    pairs = pairs.drop_duplicates("condition_id").sort_values("ko").reset_index(drop=True)
    parts = []
    n_chunks = (len(pairs) - 1) // CHUNK + 1
    for i in range(0, len(pairs), CHUNK):
        ch = pairs.iloc[i : i + CHUNK]
        lo = (ch.ko.min() - pd.Timedelta(hours=49)).isoformat()
        hi = ch.ko.max().isoformat()
        marks = ",".join("?" * len(ch))
        got = sql(
            f"""select asset_id, hour_start, condition_id, outcome_label, open, high, low, close, volume_usdc,
                       n_trades
                from pm_soccer_bars_1h
                where condition_id in ({marks}) and hour_start >= ?::TIMESTAMPTZ and hour_start < ?::TIMESTAMPTZ""",
            [*ch.condition_id.tolist(), lo, hi],
        )
        got = got.merge(ch, on="condition_id", how="inner")
        keep = (got.hour_start >= got.ko - pd.Timedelta(hours=49)) & (got.hour_start < got.ko)
        parts.append(got[keep].drop(columns="ko"))
        print(f"bars chunk {i // CHUNK + 1}/{n_chunks}: {int(keep.sum())} rows", flush=True)
    return pd.concat(parts, ignore_index=True)


def pull_reddit_preko(groups: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    lk = links[links.kind.isin(["match", "pre-match"])].merge(groups[["group_id", "ko"]], on="group_id")
    parts = []
    for i in range(0, len(lk), CHUNK):
        ch = lk.iloc[i : i + CHUNK]
        marks = ",".join("?" * len(ch))
        got = sql(
            f"""select post_id, created_utc from reddit_comments where post_id in ({marks})
                and created_utc < ?::TIMESTAMPTZ""",
            [*ch.post_id.tolist(), ch.ko.max().isoformat()],
        )
        got = got.merge(ch[["post_id", "group_id", "ko"]], on="post_id")
        rec = got.assign(
            lt_24=got.created_utc < got.ko - pd.Timedelta(hours=24),
            lt_6=got.created_utc < got.ko - pd.Timedelta(hours=6),
            lt_1=got.created_utc < got.ko - pd.Timedelta(hours=1),
            lt_0=got.created_utc < got.ko,
        )
        agg = rec.groupby(["post_id", "group_id"])[["lt_24", "lt_6", "lt_1", "lt_0"]].sum().reset_index()
        parts.append(ch[["post_id", "group_id", "kind"]].merge(agg, on=["post_id", "group_id"], how="left"))
        print(f"reddit chunk {i // CHUNK + 1}: {len(got)} pre-ko comments", flush=True)
    out = pd.concat(parts, ignore_index=True)
    return out.fillna(dict.fromkeys(("lt_24", "lt_6", "lt_1", "lt_0"), 0))


def main(stages: list[str]) -> None:
    """Run the named stages (`reddit`, `bars`, `bsky`; default all); each writes `snapshot_<stage>.json`."""
    groups = pd.read_parquet(OUT / "pm_groups.parquet")
    links = pd.read_parquet(OUT / "reddit_links.parquet")
    for stage in stages or ["reddit", "bars", "bsky"]:
        snap: dict = {"started_at": datetime.now(UTC).isoformat()}
        if stage == "reddit":
            df = pull_reddit_preko(groups, links)
            df.to_parquet(OUT / "reddit_preko.parquet")
        elif stage == "bars":
            df = pull_bars(groups)
            df.to_parquet(OUT / "pm_bars.parquet")
        elif stage == "bsky":
            df = pull_bsky()
            df.to_parquet(OUT / "bsky_hourly.parquet")
            snap["bsky_posts"] = int(df.n.sum())
        else:
            raise SystemExit(f"unknown stage {stage}")
        snap["rows"] = len(df)
        snap["finished_at"] = datetime.now(UTC).isoformat()
        (OUT / f"snapshot_{stage}.json").write_text(json.dumps(snap, indent=2))
        print(stage, snap, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
