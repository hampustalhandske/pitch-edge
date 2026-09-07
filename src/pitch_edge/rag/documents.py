"""Turn warehouse rows into retrievable natural-language documents.

Document types: match reports (from results + stats + odds), model
explanations (feature values + probabilities per prediction), news items,
StatsBomb event summaries (xG, shots, passing volume per team). Every
document carries structured metadata so answers can cite match_id / source.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def match_documents(matches: pd.DataFrame, limit: int | None = None) -> list[Document]:
    docs: list[Document] = []
    df = matches.sort_values("date", ascending=False)
    if limit:
        df = df.head(limit)
    for _, r in df.iterrows():
        bits = [
            f"{r['home_team']} {int(r['home_goals'])}-{int(r['away_goals'])} {r['away_team']} "
            f"({r.get('league', '')}, {pd.Timestamp(r['date']).date()}, season {r.get('season', '')})."
        ]
        if pd.notna(r.get("home_shots")) and pd.notna(r.get("away_shots")):
            bits.append(f"Shots {int(r['home_shots'])}-{int(r['away_shots'])}")
            if pd.notna(r.get("home_shots_on_target")):
                bits.append(f"on target {int(r['home_shots_on_target'])}-{int(r['away_shots_on_target'])}")
        if pd.notna(r.get("home_corners")):
            bits.append(f"corners {int(r['home_corners'])}-{int(r['away_corners'])}")
        if pd.notna(r.get("home_yellows")):
            bits.append(f"yellow cards {int(r['home_yellows'])}-{int(r['away_yellows'])}")
        if isinstance(r.get("referee"), str) and r["referee"]:
            bits.append(f"referee {r['referee']}")
        odds = [c for c in ("PSH", "PSD", "PSA") if c in r.index and pd.notna(r[c])]
        if len(odds) == 3:
            bits.append(f"Pinnacle pre-match odds {r['PSH']:.2f}/{r['PSD']:.2f}/{r['PSA']:.2f}")
        if pd.notna(r.get("home_elo")) and pd.notna(r.get("away_elo")):
            bits.append(f"Elo {r['home_elo']:.0f} vs {r['away_elo']:.0f}")
        docs.append(
            Document(
                f"match:{r['match_id']}",
                "; ".join(bits) + ".",
                {
                    "type": "match",
                    "match_id": str(r["match_id"]),
                    "date": str(pd.Timestamp(r["date"]).date()),
                    "home_team": r["home_team"],
                    "away_team": r["away_team"],
                    "league": str(r.get("league", "")),
                },
            )
        )
    return docs


def prediction_documents(
    predictions: pd.DataFrame, model_name: str, features: pd.DataFrame | None = None
) -> list[Document]:
    docs: list[Document] = []
    feats = features.set_index("match_id") if features is not None and not features.empty else None
    for _, r in predictions.iterrows():
        text = (
            f"Model {model_name} on {r['home_team']} vs {r['away_team']} ({pd.Timestamp(r['date']).date()}): "
            f"P(home)={r['p_home']:.1%}, P(draw)={r['p_draw']:.1%}, P(away)={r['p_away']:.1%}."
        )
        if pd.notna(r.get("mkt_home")):
            text += f" No-vig market: {r['mkt_home']:.1%}/{r['mkt_draw']:.1%}/{r['mkt_away']:.1%}."
            best = max(("home", "draw", "away"), key=lambda o: r[f"edge_{o}"])
            text += f" Largest edge: {best} ({r[f'edge_{best}']:+.1%})."
        if feats is not None and r["match_id"] in feats.index:
            f = feats.loc[r["match_id"]]
            why = []
            if pd.notna(f.get("elo_diff")):
                why.append(f"Elo gap {f['elo_diff']:+.0f}")
            if pd.notna(f.get("form_diff_r5")):
                why.append(f"5-match points-per-game gap {f['form_diff_r5']:+.2f}")
            if pd.notna(f.get("away_rest_days")):
                why.append(f"away rest {f['away_rest_days']:.0f}d")
            if pd.notna(f.get("away_travel_km")):
                why.append(f"away travel {f['away_travel_km']:.0f}km")
            if pd.notna(f.get("ref_home_bias")):
                why.append(f"referee home-bias {f['ref_home_bias']:+.2f}")
            if pd.notna(f.get("wx_precipitation")):
                why.append(f"precipitation {f['wx_precipitation']:.1f}mm")
            if why:
                text += " Drivers: " + ", ".join(why) + "."
        docs.append(
            Document(
                f"pred:{model_name}:{r['match_id']}",
                text,
                {
                    "type": "prediction",
                    "model": model_name,
                    "match_id": str(r["match_id"]),
                    "home_team": r["home_team"],
                    "away_team": r["away_team"],
                    "date": str(pd.Timestamp(r["date"]).date()),
                },
            )
        )
    return docs


def news_documents(news: pd.DataFrame) -> list[Document]:
    docs = []
    for _, r in news.iterrows():
        text = f"{r.get('title', '')}. {r.get('summary', '')}".strip()
        meta = {
            "type": "news",
            "feed": str(r.get("feed", "")),
            "published_at": str(r.get("published_at", "")),
            "team": str(r.get("team") or ""),
            "sentiment": float(r.get("sentiment", 0.0) or 0.0),
            "link": str(r.get("link", "")),
        }
        docs.append(Document(f"news:{r['item_id']}", text, meta))
    return docs


def statsbomb_documents(events: pd.DataFrame, matches: pd.DataFrame) -> list[Document]:
    docs = []
    meta_by_id = matches.set_index("statsbomb_match_id") if "statsbomb_match_id" in matches else pd.DataFrame()
    for mid, ev in events.groupby("statsbomb_match_id"):
        parts = []
        for team, te in ev.groupby("team"):
            shots = te[te["type"] == "Shot"]
            passes = te[te["type"] == "Pass"]
            comp = passes["pass_outcome"].isna().mean() if len(passes) else 0
            top = te[te["type"] == "Pass"]["player"].value_counts().head(3).index.tolist()
            parts.append(
                f"{team}: xG {shots['shot_xg'].fillna(0).sum():.2f} from {len(shots)} shots, "
                f"{len(passes)} passes at {comp:.0%} completion, most passes by {', '.join(map(str, top))}"
            )
        header = ""
        if not meta_by_id.empty and mid in meta_by_id.index:
            m = meta_by_id.loc[mid]
            header = f"{m['home_team']} {int(m['home_goals'])}-{int(m['away_goals'])} {m['away_team']} ({m['league']}, {pd.Timestamp(m['date']).date()}). "
        docs.append(
            Document(
                f"sb:{mid}", header + " | ".join(parts) + ".", {"type": "statsbomb", "statsbomb_match_id": int(mid)}
            )
        )
    return docs
