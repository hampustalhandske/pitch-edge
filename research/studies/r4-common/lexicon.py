"""The R4 sentiment lexicon, fixed before any Bluesky text was aggregated or any price move was looked at.

Two English word lists aimed at pre-match crowd talk about one club: NEGATIVE is bad team news and pessimism
(injuries, suspensions, crisis, expecting to lose), POSITIVE is good team news and optimism (players back,
confidence, expecting to win). A post counts as positive (negative) when its lower-cased text matches the
POSITIVE (NEGATIVE) pattern anywhere, as whole words; a post can be both. No weighting, no negation handling,
no LLM. `sql_pattern(words)` renders a list as one RE2 regular expression for DuckDB's `regexp_matches`.
"""

from __future__ import annotations

NEGATIVE = [
    "injury",
    "injuries",
    "injured",
    "ruled out",
    "out injured",
    "doubt",
    "doubtful",
    "suspended",
    "suspension",
    "banned",
    "sidelined",
    "absent",
    "absence",
    "missing",
    "setback",
    "hamstring",
    "crisis",
    "sacked",
    "sacking",
    "awful",
    "terrible",
    "worried",
    "worrying",
    "nervous",
    "dreading",
    "dire",
    "shambles",
    "disaster",
    "hopeless",
    "embarrassing",
    "rubbish",
    "lose",
    "losing",
    "defeat",
    "beaten",
    "relegation",
]
POSITIVE = [
    "fit again",
    "fitness boost",
    "boost",
    "returns",
    "back in training",
    "available again",
    "confident",
    "confidence",
    "unbeaten",
    "brilliant",
    "excellent",
    "fantastic",
    "superb",
    "amazing",
    "excited",
    "buzzing",
    "victory",
    "dominant",
    "in form",
    "great form",
    "winning streak",
    "come on",
    "three points",
    "get the win",
]


def sql_pattern(words: list[str]) -> str:
    """RE2 pattern matching any of `words` as whole words (spaces inside a phrase match one or more spaces)."""
    alts = "|".join(w.replace(" ", r"\s+") for w in words)
    return rf"\b({alts})\b"
