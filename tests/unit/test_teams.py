from __future__ import annotations

import pytest

from pitch_edge.data.teams import TeamNameResolver, normalise

pytestmark = pytest.mark.unit

CANON = ["Man United", "Man City", "Tottenham", "Wolves", "Nott'm Forest", "Bayern Munich", "Paris SG", "Ath Madrid"]


def test_normalise_strips_suffixes_and_accents():
    assert normalise("Manchester United FC") == "manchester united"
    assert normalise("Bayern München") == "bayern munchen"
    assert normalise("Nott'm Forest") == "nottm forest"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("ManUnited", "Man United"),
        ("Manchester United FC", "Man United"),
        ("Tottenham Hotspur FC", "Tottenham"),
        ("Wolverhampton Wanderers", "Wolves"),
        ("Nottingham Forest", "Nott'm Forest"),
        ("FC Bayern München", "Bayern Munich"),
        ("Paris Saint-Germain", "Paris SG"),
        ("Atlético de Madrid", "Ath Madrid"),
        ("Man City", "Man City"),
    ],
)
def test_resolver_aliases(raw, expected):
    assert TeamNameResolver(CANON).resolve(raw) == expected


def test_resolver_returns_none_for_unknown():
    assert TeamNameResolver(CANON).resolve("Zebra Rovers") is None


def test_resolver_fuzzy_match():
    r = TeamNameResolver(["Real Sociedad", "Real Madrid"])
    assert r.resolve("Real Sociedad de Futbol") == "Real Sociedad"
