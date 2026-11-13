"""Frozen R6 live fair-value model for `soft-vs-sharp-live-v2` (arm `r6_live_model`).

Formula (registration `bookmaker-line-structure-live-soft-vs-sharp-v2`, `notes`), per outcome o:

    logit q_o = logit p_pin_o + a_o + b_o * g_cons_o + sum_b (c_b,o * dev_b,o + m_b,o * miss_b)
    p_fair    = q / sum(q)

- p_pin: power-de-vigged live Pinnacle 1X2 from the response (`pitch_edge.research.devig`, as in the fit).
- dev_b,o = logit p_b_o - logit p_pin_o with p_b the power-de-vigged 1X2 of book b in the same response,
  clipped to [-0.5, 0.5]; a book without a complete 1X2 gets dev = 0 and miss_b = 1. Books b: WH williamhill,
  VC betvictor, 1XB onexbet, BF betfair_sb_uk.
- g_cons_o = logit of the power-de-vigged mean implied probability across the response's soft books, minus
  logit p_pin_o.
- a, b, c, m: frozen in `data/research/r6/v2_live_coefficients.json`, loaded only when its sha256 equals
  `COEFFICIENTS_SHA256` (`CoefficientHashError` otherwise).

Encoding caveat: the fitting script (`research/studies/r6-cross-book-close/v2_coefficients.py`) treated a
deviation outside [-0.5, 0.5] as *missing* (dev = 0, miss = 1 for that outcome) rather than clipping it, and
fitted g on football-data's Avg (mean decimal odds over its own book list). The registered rule (clip; mean
implied probability over the-odds-api soft books) is what decides; `p_fair_fit_encoding` records the
out-of-range-as-missing variant alongside so the difference can be measured.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pitch_edge.research.devig import devig

COEFFICIENTS_SHA256 = "57f7d4245ab002698d961b2ec0919284dcd92d6a75d2a9bd059aafeba170907c"
OUTCOMES = ("home", "draw", "away")
# Same definition as `soft-vs-sharp-live` (v1): every book except Pinnacle and the exchanges is soft.
NON_SOFT_BOOKS = frozenset({"pinnacle", "betfair_ex_uk", "betfair_ex_eu", "betfair_ex_au", "smarkets", "matchbook"})


class CoefficientHashError(RuntimeError):
    """The coefficients file is not the one frozen at registration."""


def default_coefficients_path() -> Path:
    from pitch_edge.config import get_settings

    return get_settings().data_dir / "research" / "r6" / "v2_live_coefficients.json"


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _complete(odds: Any) -> bool:
    return odds is not None and len(odds) == 3 and all(o is not None and np.isfinite(o) and o > 1.0 for o in odds)


def power_fair(odds: tuple[float, float, float] | list[float]) -> list[float]:
    """Power-de-vigged probabilities of one 1X2 (the research de-vig implementation used in the fit)."""
    return [float(x) for x in devig([list(odds)], "power").probs[0]]


@dataclass(frozen=True)
class FairValue:
    """Model output for one response: `p_fair` (registered encoding), `p_fair_fit_encoding` (diagnostic),
    per-book `dev` (registered, clipped) and `miss`, which books were `clipped`, `g_cons`, `p_pin`."""

    p_fair: list[float]
    p_fair_fit_encoding: list[float]
    p_pin: list[float]
    g_cons: list[float]
    dev: dict[str, list[float]]
    miss: dict[str, int]
    clipped: dict[str, bool]

    def as_params(self) -> dict[str, Any]:
        return {
            "p_fair_model": self.p_fair,
            "p_fair_fit_encoding": self.p_fair_fit_encoding,
            "g_cons": self.g_cons,
            "dev": self.dev,
            "miss": self.miss,
            "clipped": self.clipped,
        }


class R6LiveModel:
    """The frozen coefficients plus `fair_value(pinnacle_odds, books)`."""

    def __init__(self, coefficients: dict[str, Any], sha256: str):
        self.coefficients = coefficients
        self.sha256 = sha256
        self.books: dict[str, str] = dict(coefficients["books"])  # short code -> the-odds-api key
        self.clip = float(coefficients.get("dev_clip", 0.5))

    @classmethod
    def load(cls, path: str | Path | None = None, expected_sha256: str = COEFFICIENTS_SHA256) -> R6LiveModel:
        """Load the coefficients file; raises CoefficientHashError unless its sha256 is `expected_sha256`."""
        p = Path(path) if path else default_coefficients_path()
        raw = p.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_sha256:
            raise CoefficientHashError(f"{p} has sha256 {digest}, not the registered {expected_sha256}")
        return cls(json.loads(raw), digest)

    def fair_value(
        self, pinnacle: tuple[float, float, float], books: dict[str, tuple[float, float, float]]
    ) -> FairValue:
        """Fair 1X2 probabilities from Pinnacle's quote and the other books' quotes in the same response.

        `books` maps the-odds-api bookmaker key -> (home, draw, away) decimal odds; non-soft books in it are
        ignored for the consensus. Raises ValueError when Pinnacle's 1X2 is incomplete or no soft book is.
        """
        if not _complete(pinnacle):
            raise ValueError("a complete Pinnacle 1X2 is required")
        p_pin = power_fair(pinnacle)
        soft = [o for b, o in books.items() if b not in NON_SOFT_BOOKS and _complete(o)]
        if not soft:
            raise ValueError("no complete soft-book 1X2 in the response")
        mean_implied = np.mean([[1.0 / x for x in o] for o in soft], axis=0)
        p_cons = power_fair([1.0 / x for x in mean_implied])
        g_cons = [_logit(p_cons[k]) - _logit(p_pin[k]) for k in range(3)]

        dev: dict[str, list[float]] = {}
        raw_dev: dict[str, list[float] | None] = {}
        miss: dict[str, int] = {}
        clipped: dict[str, bool] = {}
        for short, key in self.books.items():
            odds = books.get(key)
            if _complete(odds):
                p_b = power_fair(odds)  # type: ignore[arg-type]
                d = [_logit(p_b[k]) - _logit(p_pin[k]) for k in range(3)]
                raw_dev[short] = d
                dev[short] = [max(-self.clip, min(self.clip, x)) for x in d]
                miss[short] = 0
                clipped[short] = any(abs(x) > self.clip for x in d)
            else:
                raw_dev[short] = None
                dev[short] = [0.0, 0.0, 0.0]
                miss[short] = 1
                clipped[short] = False

        q, q_fit = [], []
        for k, o in enumerate(OUTCOMES):
            c = self.coefficients["outcomes"][o]
            base = _logit(p_pin[k]) + c["intercept"] + c["g_avg"] * g_cons[k]
            z, z_fit = base, base
            for short in self.books:
                z += c[f"dev_{short}"] * dev[short][k] + c[f"miss_{short}"] * miss[short]
                rd = raw_dev[short]
                if rd is not None and abs(rd[k]) <= self.clip:
                    z_fit += c[f"dev_{short}"] * rd[k]
                else:
                    z_fit += c[f"miss_{short}"]
            q.append(_sigmoid(z))
            q_fit.append(_sigmoid(z_fit))
        return FairValue(
            p_fair=[x / sum(q) for x in q],
            p_fair_fit_encoding=[x / sum(q_fit) for x in q_fit],
            p_pin=p_pin,
            g_cons=g_cons,
            dev=dev,
            miss=miss,
            clipped=clipped,
        )
